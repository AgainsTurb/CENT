"""Capacity sweep for the latent denoiser, plus a mean-predictor reference.

Question 1: does a bigger denoiser fix the 0.32 round-trip error?
Question 2 (control): is the denoiser even beating "predict the dataset mean"? If its loss is
close to the constant-mean MSE, it has collapsed to the average latent and no amount of
sampling will produce a line.

Usage:  python tools/cap_test.py <width> <depth> <steps>
Data + latents are cached after the first run, so repeats are fast.
"""
import sys, os, json, math, time
sys.path.insert(0, "A:/Papers/CENT")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ZD, P, TDIFF, NL = 96, 384, 100, 2000
WIDTH = int(sys.argv[1]) if len(sys.argv) > 1 else 512
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
BS = 64
CACHE = "runs/v5/local_lat.npz"

if os.path.exists(CACHE):
    d = np.load(CACHE)
    X, T, Z = torch.tensor(d["X"]).float().to(DEV), torch.tensor(d["T"]).long().to(DEV), torch.tensor(d["Z"]).float().to(DEV)
    print(f"[cache] {X.shape[0]} lines", flush=True)
else:
    from cent.data_strokes import StrokeLineDataset

    def _refs(self, w, e):
        return torch.zeros(int(self.k_refs), int(self.img_h), 64), torch.ones(int(self.k_refs))

    StrokeLineDataset._refs = _refs
    base = StrokeLineDataset("A:/Papers/CENT/data/canonical_lines/v1", "train", k_refs=1,
                             img_h=64, canvas_w=1024, paired_only=False, limit=0)
    Xl, Tl = [], []
    for rid in base.usable[:NL]:
        pts = []
        for s in base.strokes_of(rid):
            a = np.asarray(s, np.float32)
            if len(a) >= 2:
                pts.append(a)
        p = np.concatenate(pts, 0)
        lo, hi = p.min(0), p.max(0)
        p = (p - lo) / np.maximum(hi - lo, 1e-6)
        idx = np.linspace(0, len(p) - 1, P)
        Xl.append(np.stack([np.interp(idx, np.arange(len(p)), p[:, 0]),
                            np.interp(idx, np.arange(len(p)), p[:, 1])], 1))
        text = base.records[rid].get("text", "") or ""
        t = np.zeros(8, np.int64)
        for k in range(min(len(text), 8)):
            t[k] = base.char2idx.get(text[k], base.pad_id)
        Tl.append(t)
    X = torch.tensor(np.stack(Xl)).float().to(DEV)
    T = torch.tensor(np.stack(Tl)).long().to(DEV)

    class Enc(nn.Module):
        def __init__(s, h=192):
            super().__init__()
            s.n = nn.Sequential(nn.Conv1d(2, h, 5, 2, 2), nn.GELU(), nn.Conv1d(h, h, 5, 2, 2), nn.GELU())
            s.mu = nn.Conv1d(h, ZD, 1)
            s.lv = nn.Conv1d(h, ZD, 1)

        def forward(s, x):
            f = s.n(x)
            return s.mu(f), s.lv(f)

    enc = Enc().to(DEV)
    ck = torch.load("runs/v5/vae.pt", map_location=DEV)
    enc.load_state_dict(ck["enc"])
    with torch.no_grad():
        Z = torch.cat([enc(X[j:j + 512].transpose(1, 2))[0] for j in range(0, X.shape[0], 512)])
    np.savez(CACHE, X=X.cpu().numpy().astype(np.float16), T=T.cpu().numpy().astype(np.int16),
             Z=Z.cpu().numpy().astype(np.float16))
    print(f"[cache] built {X.shape[0]} lines", flush=True)

MU, SD = Z.mean(), Z.std()
Zn = (Z - MU) / SD
N = Zn.shape[0]


def temb(t, dim=256):
    h = dim // 2
    fr = torch.exp(-math.log(10000.) * torch.arange(h, device=t.device) / h)
    a = t.reshape(-1, 1) * fr.reshape(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


class Den(nn.Module):
    def __init__(s, d, depth, heads=8):
        super().__init__()
        s.inp = nn.Conv1d(ZD, d, 1)
        s.tm = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.tp = nn.Linear(320, d)
        s.ctx = nn.MultiheadAttention(d, heads, batch_first=True)
        s.bl = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, .1, batch_first=True,
                                                         norm_first=True) for _ in range(depth)])
        s.norm = nn.LayerNorm(d)
        s.out = nn.Conv1d(d, ZD, 1)

    def forward(s, zt, t, txt):
        h = s.inp(zt).transpose(1, 2) + s.tm(temb(t)).unsqueeze(1)
        h = h + s.ctx(h, s.tp(txt), s.tp(txt), need_weights=False)[0]
        for b in s.bl:
            h = b(h)
        return s.out(s.norm(h).transpose(1, 2))


def ca(t, T_=TDIFF, s0=.008):
    f = math.cos((t / T_ + s0) / (1 + s0) * math.pi / 2) ** 2
    return f / math.cos(s0 / (1 + s0) * math.pi / 2) ** 2


# control: how good is "always predict the dataset mean"?
mean_mse = ((Zn - Zn.mean(0, keepdim=True)) ** 2).mean().item()
print(f"[control] constant-mean MSE {mean_mse:.4f}  (a denoiser must beat this clearly)", flush=True)

emb = nn.Embedding(2680, 320).to(DEV)
den = Den(WIDTH, DEPTH).to(DEV)
nparam = sum(p.numel() for p in den.parameters())
print(f"[model] width {WIDTH} depth {DEPTH} params {nparam/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(list(den.parameters()) + list(emb.parameters()), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
ema = {k: v.detach().clone() for k, v in den.state_dict().items()}
t0 = time.time()
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
        print(f"[{s}] eps-mse {loss.item():.4f}", flush=True)
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
    print(f"[round-trip] relative error {rt:.3f}  (want < 0.15)", flush=True)

print(json.dumps({"width": WIDTH, "depth": DEPTH, "params_M": round(nparam / 1e6, 1),
                  "steps": STEPS, "round_trip": round(rt, 3),
                  "mean_mse": round(mean_mse, 4), "sec": round(time.time() - t0, 1)}))
