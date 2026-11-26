"""Is the latent's DIMENSIONALITY the blocker? Pool it and re-run the same test.

Baseline (measured, identical protocol): 96x96 latent, 512x8 denoiser, 8k steps -> round-trip 0.321
This script repeats it at 48x48 and 24x24 with everything else held fixed.

No VAE retrain is involved: the latents are average-pooled, which only changes how many numbers
the denoiser must model. If round-trip falls sharply with pooling, dimensionality is the blocker.
"""
import sys, json, math, time
sys.path.insert(0, "A:/Papers/CENT")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, TDIFF, BS, STEPS = 384, 100, 64, 8000
d = np.load("runs/v5/local_lat.npz")
Z0 = torch.tensor(d["Z"]).float().to(DEV)
T = torch.tensor(d["T"]).long().to(DEV)
print(f"[cache] {Z0.shape[0]} latents, raw {tuple(Z0.shape[1:])}", flush=True)


def temb(t, dim=256):
    h = dim // 2
    fr = torch.exp(-math.log(10000.) * torch.arange(h, device=t.device) / h)
    a = t.reshape(-1, 1) * fr.reshape(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


def ca(t, T_=TDIFF, s0=.008):
    f = math.cos((t / T_ + s0) / (1 + s0) * math.pi / 2) ** 2
    return f / math.cos(s0 / (1 + s0) * math.pi / 2) ** 2


def make_den(zd, w, depth, heads=8):
    class Den(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Conv1d(zd, w, 1)
            s.tm = nn.Sequential(nn.Linear(256, w), nn.SiLU(), nn.Linear(w, w))
            s.tp = nn.Linear(320, w)
            s.ctx = nn.MultiheadAttention(w, heads, batch_first=True)
            s.bl = nn.ModuleList([nn.TransformerEncoderLayer(w, heads, 4 * w, .1, batch_first=True,
                                                             norm_first=True) for _ in range(depth)])
            s.norm = nn.LayerNorm(w)
            s.out = nn.Conv1d(w, zd, 1)

        def forward(s, zt, t, txt):
            h = s.inp(zt).transpose(1, 2) + s.tm(temb(t)).unsqueeze(1)
            h = h + s.ctx(h, s.tp(txt), s.tp(txt), need_weights=False)[0]
            for b in s.bl:
                h = b(h)
            return s.out(s.norm(h).transpose(1, 2))
    return Den().to(DEV)


def run(pool, w=512, depth=8):
    torch.manual_seed(0)
    Z = Z0 if pool == 1 else F.avg_pool2d(Z0, pool)
    zd, L = Z.shape[1], Z.shape[2]
    Zn = (Z - Z.mean()) / Z.std()
    mean_mse = ((Zn - Zn.mean(0, keepdim=True)) ** 2).mean().item()
    emb = nn.Embedding(2680, 320).to(DEV)
    den = make_den(zd, w, depth)
    nparam = sum(p.numel() for p in den.parameters())
    opt = torch.optim.AdamW(list(den.parameters()) + list(emb.parameters()), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    ema = {k: v.detach().clone() for k, v in den.state_dict().items()}
    N = Zn.shape[0]
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
        sch.step()
        with torch.no_grad():
            for k, v in den.state_dict().items():
                ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        if s % 4000 == 0:
            print(f"  [pool{pool} {s}] eps-mse {loss.item():.4f}", flush=True)
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
        zz2 = torch.randn(n, zd, L, device=DEV)
        ts = list(np.linspace(TDIFF - 1, 0, 30).astype(int))
        for i, t in enumerate(ts):
            e = den(zz2, torch.full((n,), t / TDIFF, device=DEV), ctx)
            z0h = (zz2 - math.sqrt(1 - ca(t)) * e) / math.sqrt(ca(t))
            zz2 = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
                   if i + 1 < len(ts) else z0h)
        dd = torch.cdist(zz2.reshape(n, -1), Zn.reshape(N, -1)).min(1).values.mean().item()
    out = {"pool": pool, "latent": f"{zd}x{L}", "dims": zd * L, "params_M": round(nparam / 1e6, 1),
           "round_trip": round(rt, 3), "mean_mse": round(mean_mse, 4),
           "sample_dist": round(dd, 2), "sec": round(time.time() - t0, 1)}
    print(json.dumps(out), flush=True)
    return out


res = []
for pool in (2, 4):
    print(f"== pooled by {pool}x ==", flush=True)
    res.append(run(pool))
print(json.dumps({"baseline_96x96_round_trip": 0.321, "results": res}))
