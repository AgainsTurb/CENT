"""Is the sampler's STARTING POINT the bug? Two measurements plus visuals.

At t=99 the cosine schedule gives a_99 = 2.5e-4, so the first x0 estimate is amplified ~60x.
Standard DDIM practice starts where a_t is ~0.01-0.05. This script compares:
  A) sample from pure noise starting at t=99   (ours)
  B) sample from pure noise starting at t=90   (a ~0.010, amplification ~10x)
  C) round-trip from real z0 at t=50, decoded  -- is a 0.32 latent error visually tolerable?

All three are decoded and rendered, because a distance number alone has already misled us twice.
"""
import sys, json, math, time
sys.path.insert(0, "A:/Papers/CENT")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ZD, P, TDIFF, BS, STEPS = 96, 384, 100, 64, 6000
d = np.load("runs/v5/local_lat.npz")
X = torch.tensor(d["X"]).float().to(DEV)
T = torch.tensor(d["T"]).long().to(DEV)
Z = torch.tensor(d["Z"]).float().to(DEV)
MU, SD = Z.mean(), Z.std()
Zn = (Z - MU) / SD
N = Zn.shape[0]
print(f"[cache] {N} latents", flush=True)


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
    def __init__(s, w=320, depth=4, heads=8):
        super().__init__()
        s.inp = nn.Conv1d(ZD, w, 1)
        s.tm = nn.Sequential(nn.Linear(256, w), nn.SiLU(), nn.Linear(w, w))
        s.tp = nn.Linear(320, w)
        s.ctx = nn.MultiheadAttention(w, heads, batch_first=True)
        s.bl = nn.ModuleList([nn.TransformerEncoderLayer(w, heads, 4 * w, .1, batch_first=True,
                                                         norm_first=True) for _ in range(depth)])
        s.norm = nn.LayerNorm(w)
        s.out = nn.Conv1d(w, ZD, 1)

    def forward(s, zt, t, txt):
        h = s.inp(zt).transpose(1, 2) + s.tm(temb(t)).unsqueeze(1)
        h = h + s.ctx(h, s.tp(txt), s.tp(txt), need_weights=False)[0]
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


dec = Dec().to(DEV)
ck = torch.load("runs/v5/vae.pt", map_location=DEV)
dec.load_state_dict(ck["dec"])
emb = nn.Embedding(2680, 320).to(DEV)
den = Den().to(DEV)
opt = torch.optim.AdamW(list(den.parameters()) + list(emb.parameters()), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
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
    sch.step()
    with torch.no_grad():
        for k, v in den.state_dict().items():
            ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
    if s % 3000 == 0:
        print(f"[den {s}] eps-mse {loss.item():.4f}", flush=True)
den.load_state_dict(ema)
print(f"[den] trained in {time.time()-t0:.0f}s", flush=True)


def ddim(zz, t_start, nsteps):
    ts = list(np.linspace(t_start, 0, nsteps).astype(int))
    with torch.no_grad():
        for i, t in enumerate(ts):
            e = den(zz, torch.full((zz.shape[0],), t / TDIFF, device=DEV), ctx)
            z0h = (zz - math.sqrt(1 - ca(t)) * e) / max(math.sqrt(ca(t)), 1e-8)
            zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
                  if i + 1 < len(ts) else z0h)
    return zz


def to_xy(z):
    q = dec(z).view(-1, 8, 6, P).permute(0, 3, 1, 2)
    pi = torch.softmax(q[..., 0], -1)
    return torch.stack([(q[..., 1] * pi).sum(-1), (q[..., 2] * pi).sum(-1)], -1).detach().cpu().numpy()


n = 8
ctx = emb(T[:n]).mean(1, keepdim=True)
torch.manual_seed(0)
res = {}
imgs = [("GT", render(X[0].cpu().numpy()))]
imgs.append(("RECON", render(to_xy(Zn[:1])[0])))

for tag, t_start in (("A_t99", 99), ("B_t90", 90)):
    zz = torch.randn(n, ZD, 96, device=DEV)
    zz = ddim(zz, t_start, 30)
    with torch.no_grad():
        dd = torch.cdist(zz.reshape(n, -1), Zn.reshape(N, -1)).min(1).values.mean().item()
    res[tag] = {"start": t_start, "sample_dist": round(dd, 2),
                "norm": round(zz.norm(dim=(1, 2)).mean().item(), 1)}
    imgs.append((f"SAMPLE {tag} dist={dd:.1f}", render(to_xy(zz[:1])[0])))
    print(f"[{tag}] start t={t_start}  dist {dd:.2f}  ||z|| {res[tag]['norm']}", flush=True)

z0 = Zn[:n]
zz = math.sqrt(ca(50)) * z0 + math.sqrt(1 - ca(50)) * torch.randn_like(z0)
zz = ddim(zz, 50, 40)
rt = ((zz - z0).norm(dim=(1, 2)) / z0.norm(dim=(1, 2))).mean().item()
res["C_roundtrip"] = round(rt, 3)
imgs.append((f"ROUND-TRIP decoded rel={rt:.2f}", render(to_xy(zz[:1])[0])))
print(f"[C] round-trip rel {rt:.3f} (decoded for visual check)", flush=True)

out = Image.new("L", (1024, len(imgs) * 116 + 20), 255)
dr = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(imgs):
    y = 20 + i * 116
    dr.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("results/sched_test.png")
print(json.dumps({"sheet": "results/sched_test.png", **res}))
