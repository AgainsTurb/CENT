"""Root-cause diagnostics for the CENT v5 DiT failure.

Four measurements, no training:
  1. is the VAE latent actually N(0,1)?   (a diffusion prior assumes it is)
  2. what did the denoiser learn to predict: eps or x0?
  3. round-trip: real z0 + noise at t=50 -> DDIM -> how far from z0?
  4. sample from pure noise -> is it near the real latent manifold?
"""
import sys
sys.path.insert(0, "A:/Papers/CENT")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ZD, P, TDIFF = 96, 384, 100

from cent.data_strokes import StrokeLineDataset


def _refs(self, w, e):
    return torch.zeros(int(self.k_refs), int(self.img_h), 64), torch.ones(int(self.k_refs))


StrokeLineDataset._refs = _refs

base = StrokeLineDataset("A:/Papers/CENT/data/canonical_lines/v1", "train", k_refs=1,
                         img_h=64, canvas_w=1024, paired_only=False, limit=0)
NL = 1200
Xl, Tl = [], []
for rid in base.usable[:NL]:
    pts, pen = [], []
    for s in base.strokes_of(rid):
        a = np.asarray(s, np.float32)
        if len(a) < 2:
            continue
        pts.append(a)
        pen.append(np.r_[np.zeros(len(a) - 1, np.float32), 1.0])
    p = np.concatenate(pts, 0)
    lo, hi = p.min(0), p.max(0)
    p = (p - lo) / np.maximum(hi - lo, 1e-6)
    idx = np.linspace(0, len(p) - 1, P)
    Xl.append(np.stack([np.interp(idx, np.arange(len(p)), p[:, 0]),
                        np.interp(idx, np.arange(len(p)), p[:, 1])], 1))
    text = base.records[rid].get("text", "") or ""
    t = np.zeros(8, np.int64)
    n = min(len(text), 8)
    for k in range(n):
        t[k] = base.char2idx.get(text[k], base.pad_id)
    Tl.append(t)
X = torch.tensor(np.stack(Xl)).float().to(DEV)
T = torch.tensor(np.stack(Tl)).long().to(DEV)
print(f"[data] {X.shape[0]} lines from the local corpus", flush=True)


class Enc(nn.Module):
    def __init__(s, h=192):
        super().__init__()
        s.n = nn.Sequential(nn.Conv1d(2, h, 5, 2, 2), nn.GELU(), nn.Conv1d(h, h, 5, 2, 2), nn.GELU())
        s.mu = nn.Conv1d(h, ZD, 1)
        s.lv = nn.Conv1d(h, ZD, 1)

    def forward(s, x):
        f = s.n(x)
        return s.mu(f), s.lv(f)


def temb(t, dim=256):
    h = dim // 2
    fr = torch.exp(-math.log(10000.) * torch.arange(h, device=t.device) / h)
    a = t.reshape(-1, 1) * fr.reshape(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


class Den(nn.Module):
    def __init__(s, d=320, heads=8, depth=4):
        super().__init__()
        s.inp = nn.Conv1d(ZD, d, 1)
        s.tm = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.ctx = nn.MultiheadAttention(d, heads, batch_first=True)
        s.bl = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, .1, batch_first=True,
                                                         norm_first=True) for _ in range(depth)])
        s.norm = nn.LayerNorm(d)
        s.out = nn.Conv1d(d, ZD, 1)

    def forward(s, zt, t, txt):
        h = s.inp(zt).transpose(1, 2) + s.tm(temb(t)).unsqueeze(1)
        h = h + s.ctx(h, txt, txt, need_weights=False)[0]
        for b in s.bl:
            h = b(h)
        return s.out(s.norm(h).transpose(1, 2))


def ca(t, T_=TDIFF, s0=.008):
    f = math.cos((t / T_ + s0) / (1 + s0) * math.pi / 2) ** 2
    return f / math.cos(s0 / (1 + s0) * math.pi / 2) ** 2


enc, den, emb = Enc().to(DEV), Den().to(DEV), nn.Embedding(2680, 320).to(DEV)
ck = torch.load("runs/v5/vae.pt", map_location=DEV)
enc.load_state_dict(ck["enc"])
ck2 = torch.load("runs/v5/dit.pt", map_location=DEV)
den.load_state_dict(ck2["den"])
emb.load_state_dict(ck2["emb"])

print("== 1. LATENT DISTRIBUTION (diffusion prior assumes N(0,1)) ==", flush=True)
with torch.no_grad():
    mu, lv = enc(X.transpose(1, 2))
    z = mu
    print(f"z0: mean {z.mean():+.3f}  std {z.std():.3f}  per-dim std {z.std(dim=(0,2)).mean():.3f}  "
          f"||z|| {z.norm(dim=(1,2)).mean():.1f}   [N(0,I): std 1.000, ||z|| {math.sqrt(ZD * 96):.1f}]",
          flush=True)
    print(f"KL(q||N(0,I)) = {(-0.5 * (1 + lv - mu ** 2 - lv.exp()).mean()).item():.3f} per sample", flush=True)

print("== 2. WHAT DID THE DENOISER LEARN? ==", flush=True)
eps = torch.randn_like(z)
for t in (50, 300, 700):
    a = ca(t)
    zt = math.sqrt(a) * z + math.sqrt(1 - a) * eps
    with torch.no_grad():
        ctx = emb(T).mean(1, keepdim=True)
        pr = den(zt, torch.full((X.shape[0],), t / TDIFF, device=DEV), ctx)
    e1 = ((pr - eps) ** 2).mean().item()
    e2 = ((pr - z) ** 2).mean().item()
    print(f"t={t:3d}: MSE(pred,eps)={e1:.4f}  MSE(pred,x0)={e2:.4f}  -> {'EPS' if e1 < e2 else 'X0'}",
          flush=True)

print("== 3. ROUND-TRIP from real z0 at t=50 (want < 0.10) ==", flush=True)
torch.manual_seed(0)
n = 8
z0 = z[:n]
ctx = emb(T[:n]).mean(1, keepdim=True)
zz = math.sqrt(ca(50)) * z0 + math.sqrt(1 - ca(50)) * torch.randn_like(z0)
ts = list(np.linspace(50, 0, 15).astype(int))
with torch.no_grad():
    for i, t in enumerate(ts):
        e = den(zz, torch.full((n,), t / TDIFF, device=DEV), ctx)
        z0h = (zz - math.sqrt(1 - ca(t)) * e) / math.sqrt(ca(t))
        zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
              if i + 1 < len(ts) else z0h)
    print(f"relative error {((zz - z0).norm(dim=(1, 2)) / z0.norm(dim=(1, 2))).mean().item():.3f}",
          flush=True)

print("== 4. SAMPLE FROM PURE NOISE: on-manifold? ==", flush=True)
with torch.no_grad():
    zz = torch.randn(n, ZD, 96, device=DEV)
    ts = list(np.linspace(TDIFF - 1, 0, 25).astype(int))
    for i, t in enumerate(ts):
        e = den(zz, torch.full((n,), t / TDIFF, device=DEV), ctx)
        z0h = (zz - math.sqrt(1 - ca(t)) * e) / math.sqrt(ca(t))
        zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
              if i + 1 < len(ts) else z0h)
    dd = torch.cdist(zz.reshape(n, -1), z.reshape(z.shape[0], -1)).min(1).values
    dr = torch.cdist(z[:200].reshape(200, -1), z.reshape(z.shape[0], -1)).min(1).values
    print(f"||z_sampled|| {zz.norm(dim=(1, 2)).mean():.1f} vs real {z.norm(dim=(1, 2)).mean():.1f}",
          flush=True)
    print(f"dist to nearest real: sampled {dd.mean():.2f} vs real-real {dr.mean():.2f}", flush=True)
