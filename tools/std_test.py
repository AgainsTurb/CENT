"""Does latent standardisation fix the DiT? Local experiment, no GPU rental.

  1. encode N lines with the existing VAE  (frozen, from runs/v5/vae.pt)
  2. standardise:  z' = (z - mu_z) / sd_z      (mu_z, sd_z stored)
  3. retrain the SAME small denoiser on z' with eps-prediction
  4. measure: round-trip error at t=50, sampled ||z|| vs real, distance to manifold
  5. render GT / RECON / SAMPLE so the verdict is visual, not only numeric
"""
import sys
sys.path.insert(0, "A:/Papers/CENT")
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ZD, P, TDIFF = 96, 384, 100
NL, STEPS, BS = 2000, 12000, 64

from cent.data_strokes import StrokeLineDataset


def _refs(self, w, e):
    return torch.zeros(int(self.k_refs), int(self.img_h), 64), torch.ones(int(self.k_refs))


StrokeLineDataset._refs = _refs
base = StrokeLineDataset("A:/Papers/CENT/data/canonical_lines/v1", "train", k_refs=1,
                         img_h=64, canvas_w=1024, paired_only=False, limit=0)
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
print(f"[data] {X.shape[0]} lines", flush=True)


class Enc(nn.Module):
    def __init__(s, h=192):
        super().__init__()
        s.n = nn.Sequential(nn.Conv1d(2, h, 5, 2, 2), nn.GELU(), nn.Conv1d(h, h, 5, 2, 2), nn.GELU())
        s.mu = nn.Conv1d(h, ZD, 1)
        s.lv = nn.Conv1d(h, ZD, 1)

    def forward(s, x):
        f = s.n(x)
        return s.mu(f), s.lv(f)


class Dec(nn.Module):
    def __init__(s, k=8, h=192):
        super().__init__()
        s.k = k
        s.up = nn.Sequential(nn.ConvTranspose1d(ZD, h, 4, 2, 1), nn.GELU(),
                             nn.ConvTranspose1d(h, h, 4, 2, 1), nn.GELU(), nn.Conv1d(h, 6 * k, 1))

    def forward(s, z):
        return F.interpolate(s.up(z), size=P, mode="linear", align_corners=False)


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


def render(xy, w=1024, h=96):
    img = np.ones((h, w), np.float32)
    pp = (np.clip(xy, 0, 1) * np.array([w - 1, h - 1])).astype(int)
    for a, b in zip(pp[:-1], pp[1:]):
        n = int(np.hypot(*(b - a))) + 1
        for u in np.linspace(0, 1, n):
            x, y = (a + (b - a) * u).astype(int)
            img[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = 0
    return img


enc, dec = Enc().to(DEV), Dec().to(DEV)
ck = torch.load("runs/v5/vae.pt", map_location=DEV)
enc.load_state_dict(ck["enc"])
dec.load_state_dict(ck["dec"])
with torch.no_grad():
    zs = []
    for j in range(0, X.shape[0], 512):
        mu, _ = enc(X[j:j + 512].transpose(1, 2))
        zs.append(mu)
    Z = torch.cat(zs)
MU, SD = Z.mean(), Z.std()
Zn = (Z - MU) / SD
print(f"[latent] raw std {Z.std():.3f}  ->  standardised std {Zn.std():.3f}", flush=True)

emb = nn.Embedding(2680, 320).to(DEV)
den = Den().to(DEV)
opt = torch.optim.AdamW(list(den.parameters()) + list(emb.parameters()), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
ema = {k: v.detach().clone() for k, v in den.state_dict().items()}
N = Zn.shape[0]
for s in range(STEPS):
    i = torch.randint(0, N, (BS,), device=DEV)
    z0 = Zn[i]
    tt = torch.randint(0, TDIFF, (BS,), device=DEV)
    a = torch.tensor([ca(float(v)) for v in tt], device=DEV).view(-1, 1, 1)
    eps = torch.randn_like(z0)
    zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
    ctx = emb(T[i]).mean(1, keepdim=True)
    loss = F.mse_loss(den(zt, tt / TDIFF, ctx), eps)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
    opt.step()
    sched.step()
    with torch.no_grad():
        for k, v in den.state_dict().items():
            ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
    if s % 2000 == 0:
        print(f"[dit {s}] eps-mse {loss.item():.4f}", flush=True)
den.load_state_dict(ema)

n = 8
z0 = Zn[:n]
ctx = emb(T[:n]).mean(1, keepdim=True)
with torch.no_grad():
    a50 = ca(50)
    zz = math.sqrt(a50) * z0 + math.sqrt(1 - a50) * torch.randn_like(z0)
    ts = list(np.linspace(50, 0, 20).astype(int))
    for i, t in enumerate(ts):
        e = den(zz, torch.full((n,), t / TDIFF, device=DEV), ctx)
        z0h = (zz - math.sqrt(1 - ca(t)) * e) / math.sqrt(ca(t))
        zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
              if i + 1 < len(ts) else z0h)
    rt = ((zz - z0).norm(dim=(1, 2)) / z0.norm(dim=(1, 2))).mean().item()
    print(f"[round-trip t=50] relative error {rt:.3f}  (want < 0.10)", flush=True)

    zz = torch.randn(n, ZD, 96, device=DEV)
    ts = list(np.linspace(TDIFF - 1, 0, 30).astype(int))
    for i, t in enumerate(ts):
        e = den(zz, torch.full((n,), t / TDIFF, device=DEV), ctx)
        z0h = (zz - math.sqrt(1 - ca(t)) * e) / math.sqrt(ca(t))
        zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
              if i + 1 < len(ts) else z0h)
    dd = torch.cdist(zz.reshape(n, -1), Zn.reshape(N, -1)).min(1).values
    dr = torch.cdist(Zn[:200].reshape(200, -1), Zn.reshape(N, -1)).min(1).values
    print(f"[manifold] ||z_sampled|| {zz.norm(dim=(1, 2)).mean():.1f} vs real {Zn.norm(dim=(1, 2)).mean():.1f}"
          f"   dist {dd.mean():.2f} vs real-real {dr.mean():.2f}", flush=True)

    def to_xy(z):
        q = dec(z).view(-1, 8, 6, P).permute(0, 3, 1, 2)
        pi = torch.softmax(q[..., 0], -1)
        return torch.stack([(q[..., 1] * pi).sum(-1), (q[..., 2] * pi).sum(-1)], -1).cpu().numpy()

    rec = to_xy(MU + SD * z0)[0]
    smp = to_xy(MU + SD * zz)[0]

rows = [("GT", render(X[0].cpu().numpy())), ("RECON", render(rec)), ("SAMPLE (standardised)", render(smp))]
out = Image.new("L", (1024, len(rows) * 116 + 20), 255)
d = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(rows):
    y = 20 + i * 116
    d.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("results/std_test.png")
print(json.dumps({"sheet": "results/std_test.png", "round_trip": round(rt, 3),
                  "latent_std_before": round(float(Z.std()), 3), "lines": N}))
