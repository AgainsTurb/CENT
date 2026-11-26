"""CENT v6c: continue the v6b DiT with the three fixes indicated by its own results.

  1. LR 1e-3 -> 3e-4          (v6b eps-MSE drifted 0.226 -> 0.314: an optimisation symptom)
  2. CFG drops CONTENT only   (v6b dropped style+content together; cfg=2.0 collapsed the line)
  3. warm start from dit6b.pt (it is the first model with character-scale structure)

40k steps. Gates unchanged: round-trip, then samples at cfg 1.0 / 1.5 / 2.5.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEV = "cuda"
P, TDIFF = 384, 100
ZD, LAT = 192, 96
STEPS = int(os.environ.get("STEPS", 40000))
BS, P_DROP, LR = 64, 0.15, 3e-4

d = np.load("/workspace/cache/lines.npz")
X = torch.tensor(d["X"]).float().to(DEV)
T = torch.tensor(d["T"]).long().to(DEV)
W = torch.tensor(d["W"]).long().to(DEV)
N = len(X)
ref = torch.zeros(N, dtype=torch.long, device=DEV)
wn = W.cpu().numpy()
for w in np.unique(wn):
    idx = np.where(wn == w)[0]
    ref[idx] = torch.tensor(np.roll(idx, 1) if len(idx) > 1 else idx, device=DEV)


class Enc(nn.Module):
    def __init__(s, h=256):
        super().__init__()
        s.n = nn.Sequential(nn.Conv1d(2, h, 5, 2, 2), nn.GELU(), nn.Conv1d(h, h, 5, 2, 2), nn.GELU())
        s.mu = nn.Conv1d(h, ZD, 1)
        s.lv = nn.Conv1d(h, ZD, 1)

    def forward(s, x):
        f = s.n(x)
        return s.mu(f), s.lv(f)


class Dec(nn.Module):
    def __init__(s, k=8, h=256):
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


def ca(t, T_=TDIFF, s0=.008):
    f = math.cos((t / T_ + s0) / (1 + s0) * math.pi / 2) ** 2
    return f / math.cos(s0 / (1 + s0) * math.pi / 2) ** 2


class Content(nn.Module):
    def __init__(s, d=512, vocab=2680):
        super().__init__()
        s.emb = nn.Embedding(vocab + 1, 256)
        s.net = nn.Sequential(nn.Conv1d(256, d, 5, padding=2), nn.GELU(),
                              nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2))

    def forward(s, ids):
        e = s.emb(ids)
        e = F.interpolate(e.transpose(1, 2), size=LAT, mode="nearest")
        return s.net(e)


class Den(nn.Module):
    def __init__(s, w=512, depth=8, heads=8, dtext=512):
        super().__init__()
        s.inp = nn.Conv1d(2 * ZD, w, 1)
        s.tm = nn.Sequential(nn.Linear(256, w), nn.SiLU(), nn.Linear(w, w))
        s.tp = nn.Linear(dtext, w)
        s.ctx = nn.MultiheadAttention(w, heads, batch_first=True)
        s.bl = nn.ModuleList([nn.TransformerEncoderLayer(w, heads, 4 * w, .1, batch_first=True,
                                                         norm_first=True) for _ in range(depth)])
        s.norm = nn.LayerNorm(w)
        s.out = nn.Conv1d(w, ZD, 1)
        s.null_ctx = nn.Parameter(torch.zeros(1, dtext, LAT))

    def forward(s, zt, t, zref, ctext, drop=None):
        h = s.inp(torch.cat([zt, zref], 1)).transpose(1, 2) + s.tm(temb(t)).unsqueeze(1)
        c = ctext
        if drop is not None:
            c = c.clone()
            c[drop] = s.null_ctx[0]
        c = s.tp(c.transpose(1, 2))
        h = h + s.ctx(h, c, c, need_weights=False)[0]
        for b in s.bl:
            h = b(h)
        return s.out(s.norm(h).transpose(1, 2))


ckv = torch.load("/workspace/out/vae6.pt", map_location=DEV)
enc, dec = Enc().to(DEV), Dec().to(DEV)
enc.load_state_dict(ckv["enc"])
dec.load_state_dict(ckv["dec"])
enc.eval()
MU, SD = ckv["MU"].to(DEV), ckv["SD"].to(DEV)
with torch.no_grad():
    Z = torch.cat([enc(X[j:j + 1024].transpose(1, 2))[0] for j in range(0, N, 1024)])
Zn = (Z - MU) / SD

content, den = Content().to(DEV), Den().to(DEV)
ckd = torch.load("/workspace/out/dit6b.pt", map_location=DEV)
den.load_state_dict(ckd["den"])
content.load_state_dict(ckd["content"])
print(f"[warm start] dit6b loaded; {sum(p.numel() for p in list(den.parameters())+list(content.parameters()))/1e6:.1f}M",
      flush=True)

prm = list(den.parameters()) + list(content.parameters())
opt = torch.optim.AdamW(prm, lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
ema = {k: v.detach().clone() for k, v in den.state_dict().items()}
ema_c = {k: v.detach().clone() for k, v in content.state_dict().items()}
t0 = time.time()
for s in range(STEPS):
    i = torch.randint(0, N, (BS,), device=DEV)
    z0 = Zn[i]
    tt = torch.randint(0, TDIFF, (BS,), device=DEV)
    a = torch.tensor([ca(float(v)) for v in tt], device=DEV).view(-1, 1, 1)
    eps = torch.randn_like(z0)
    zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
    drop = torch.rand(BS, device=DEV) < P_DROP           # content-only dropout
    loss = F.mse_loss(den(zt, tt / TDIFF, Zn[ref[i]], content(T[i]), drop), eps)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(prm, 1.0)
    opt.step()
    sch.step()
    with torch.no_grad():
        for k, v in den.state_dict().items():
            ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        for k, v in content.state_dict().items():
            ema_c[k].mul_(0.999).add_(v.detach(), alpha=0.001)
    if s % 4000 == 0:
        print(f"[ditc {s}] eps-mse {loss.item():.4f}", flush=True)
den.load_state_dict(ema)
content.load_state_dict(ema_c)
print(f"[ditc] trained in {time.time()-t0:.0f}s", flush=True)
torch.save({"den": den.state_dict(), "content": content.state_dict()}, "/workspace/out/dit6c.pt")


def render(xy, w=1024, h=96):
    img = np.ones((h, w), np.float32)
    pp = (np.clip(xy, 0, 1) * np.array([w - 1, h - 1])).astype(int)
    for a, b in zip(pp[:-1], pp[1:]):
        n = int(np.hypot(*(b - a))) + 1
        for u in np.linspace(0, 1, n):
            x, y = (a + (b - a) * u).astype(int)
            img[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = 0
    return img


def to_xy(z):
    q = dec(z).view(-1, 8, 6, P).permute(0, 3, 1, 2)
    pi = torch.softmax(q[..., 0], -1)
    return torch.stack([(q[..., 1] * pi).sum(-1), (q[..., 2] * pi).sum(-1)], -1).detach().cpu().numpy()


n = 4
ct = content(T[:n])
zref = Zn[ref[:n]]
nullc = content(T[:n])


def guided(zz, t_start, nsteps, scale, zr, c):
    ts = list(np.linspace(t_start, 0, nsteps).astype(int))
    with torch.no_grad():
        for i, t in enumerate(ts):
            tt = torch.full((zz.shape[0],), t / TDIFF, device=DEV)
            e_c = den(zz, tt, zr, c)
            if scale != 1.0:
                e_u = den(zz, tt, zr, c, torch.ones(zz.shape[0], dtype=torch.bool, device=DEV))
                e = e_u + scale * (e_c - e_u)
            else:
                e = e_c
            z0h = (zz - math.sqrt(1 - ca(t)) * e) / max(math.sqrt(ca(t)), 1e-8)
            zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
                  if i + 1 < len(ts) else z0h)
    return zz


res = {}
rows = [("GT", render(X[0].cpu().numpy())), ("RECON", render(to_xy(Z[:1])[0]))]
z0 = Zn[:n]
zz = math.sqrt(ca(50)) * z0 + math.sqrt(1 - ca(50)) * torch.randn_like(z0)
zz = guided(zz, 50, 50, 1.0, zref, ct)
rt = ((zz - z0).norm(dim=(1, 2)) / z0.norm(dim=(1, 2))).mean().item()
res["round_trip_t50"] = round(rt, 3)
print(f"[gate] round-trip {rt:.3f}  (v6b 0.189)", flush=True)

for scale in (1.0, 1.5, 2.5):
    torch.manual_seed(0)
    zz = torch.randn(n, ZD, LAT, device=DEV)
    zz = guided(zz, 90, 60, scale, zref, ct)
    with torch.no_grad():
        dd = torch.cdist(zz.reshape(n, -1), Zn.reshape(N, -1)).min(1).values.mean().item()
    res[f"dist_cfg{scale}"] = round(dd, 1)
    rows.append((f"SAMPLE cfg={scale} dist={dd:.1f}", render(to_xy(MU + SD * zz)[0])))
    print(f"[gate] cfg {scale}: dist {dd:.1f}", flush=True)

out = Image.new("L", (1024, len(rows) * 116 + 20), 255)
dr = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(rows):
    y = 20 + i * 116
    dr.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("/workspace/out/cent_v6c.png")
print(json.dumps({"sheet": "out/cent_v6c.png", **res}), flush=True)
