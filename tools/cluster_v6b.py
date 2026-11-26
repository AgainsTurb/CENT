"""CENT v6b: rebuild the CONDITIONING and retrain only the DiT.

Diagnosis that motivates this (from cent_v6.png): round-trip decoded to legible but WRONG
content, i.e. the sampler learned the marginal glyph distribution and ignored the text. The
text condition was one mean-pooled vector of <=8 characters and a single cross-attention.

Changes here, all in the conditioning:
  * content: per-character embeddings expanded to the LATENT LENGTH, then a conv encoder
    (DiffInk does exactly this; a global vector cannot say which character belongs where)
  * style  : a reference line from the SAME writer, encoded by the frozen VAE encoder
  * CFG    : condition dropped with p=0.15; guided sampling eps_u + s*(eps_c - eps_u)
The VAE is loaded from out/vae6.pt and frozen.
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
DIT_STEPS = int(os.environ.get("DIT_STEPS", 30000))
BS = 64
P_DROP = 0.15

d = np.load("/workspace/cache/lines.npz")
X = torch.tensor(d["X"]).float().to(DEV)
T = torch.tensor(d["T"]).long().to(DEV)
TL = torch.tensor(d["TL"]).long().to(DEV).clamp_min(1)
W = torch.tensor(d["W"]).long().to(DEV)
N = len(X)

# reference index: another line by the same writer (for the style prompt)
order = np.argsort(W.cpu().numpy(), kind="stable")
ref = torch.zeros(N, dtype=torch.long, device=DEV)
for w in np.unique(W.cpu().numpy()):
    idx = np.where(W.cpu().numpy() == w)[0]
    if len(idx) < 2:
        ref[idx] = idx[0]
        continue
    nxt = np.roll(idx, 1)
    ref[idx] = torch.tensor(nxt, device=DEV)
print(f"[data] {N} lines, refs built", flush=True)


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


ck = torch.load("/workspace/out/vae6.pt", map_location=DEV)
enc, dec = Enc().to(DEV), Dec().to(DEV)
enc.load_state_dict(ck["enc"])
dec.load_state_dict(ck["dec"])
enc.eval()
MU, SD = ck["MU"].to(DEV), ck["SD"].to(DEV)
print(f"[vae6] loaded, MU {MU.item():.3f} SD {SD.item():.3f}", flush=True)

with torch.no_grad():
    zs = []
    for j in range(0, N, 1024):
        zs.append(enc(X[j:j + 1024].transpose(1, 2))[0])
    Z = torch.cat(zs)
Zn = (Z - MU) / SD
print(f"[latent] {tuple(Zn.shape)} std {Zn.std():.3f}", flush=True)


def temb(t, dim=256):
    h = dim // 2
    fr = torch.exp(-math.log(10000.) * torch.arange(h, device=t.device) / h)
    a = t.reshape(-1, 1) * fr.reshape(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


def ca(t, T_=TDIFF, s0=.008):
    f = math.cos((t / T_ + s0) / (1 + s0) * math.pi / 2) ** 2
    return f / math.cos(s0 / (1 + s0) * math.pi / 2) ** 2


class Content(nn.Module):
    """Per-character embeddings expanded to the latent length, then a conv encoder."""

    def __init__(s, d=512, vocab=2680):
        super().__init__()
        s.emb = nn.Embedding(vocab + 1, 256)
        s.net = nn.Sequential(nn.Conv1d(256, d, 5, padding=2), nn.GELU(),
                              nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
                              nn.Conv1d(d, d, 5, padding=2))

    def forward(s, ids):
        e = s.emb(ids)                                      # [B,8,256]
        e = F.interpolate(e.transpose(1, 2), size=LAT, mode="nearest")   # each char owns a span
        return s.net(e)                                     # [B,d,LAT]


class Den(nn.Module):
    def __init__(s, w=512, depth=8, heads=8, dtext=512):
        super().__init__()
        s.inp = nn.Conv1d(2 * ZD, w, 1)                      # noisy latent + style prompt
        s.tm = nn.Sequential(nn.Linear(256, w), nn.SiLU(), nn.Linear(w, w))
        s.tp = nn.Linear(dtext, w)
        s.ctx = nn.MultiheadAttention(w, heads, batch_first=True)
        s.bl = nn.ModuleList([nn.TransformerEncoderLayer(w, heads, 4 * w, .1, batch_first=True,
                                                         norm_first=True) for _ in range(depth)])
        s.norm = nn.LayerNorm(w)
        s.out = nn.Conv1d(w, ZD, 1)
        s.null_ctx = nn.Parameter(torch.zeros(1, dtext, LAT))

    def forward(s, zt, t, zs_ref, ctext, drop=None):
        cond = torch.cat([zt, zs_ref], 1)
        if drop is not None:
            cond = cond.clone()
            cond[drop] = 0.0
        h = s.inp(cond).transpose(1, 2) + s.tm(temb(t)).unsqueeze(1)
        c = ctext
        if drop is not None:
            c = c.clone()
            c[drop] = s.null_ctx[0]
        c = s.tp(c.transpose(1, 2))
        h = h + s.ctx(h, c, c, need_weights=False)[0]
        for b in s.bl:
            h = b(h)
        return s.out(s.norm(h).transpose(1, 2))


content = Content().to(DEV)
den = Den().to(DEV)
prm = list(content.parameters()) + list(den.parameters())
opt = torch.optim.AdamW(prm, lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DIT_STEPS, eta_min=1e-5)
ema = {k: v.detach().clone() for k, v in den.state_dict().items()}
ema_c = {k: v.detach().clone() for k, v in content.state_dict().items()}
print(f"[dit] {sum(p.numel() for p in prm)/1e6:.1f}M params, CFG p={P_DROP}", flush=True)
t0 = time.time()
for s in range(DIT_STEPS):
    i = torch.randint(0, N, (BS,), device=DEV)
    z0 = Zn[i]
    zref = Zn[ref[i]]
    tt = torch.randint(0, TDIFF, (BS,), device=DEV)
    a = torch.tensor([ca(float(v)) for v in tt], device=DEV).view(-1, 1, 1)
    eps = torch.randn_like(z0)
    zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
    ct = content(T[i])
    drop = torch.rand(BS, device=DEV) < P_DROP
    loss = F.mse_loss(den(zt, tt / TDIFF, zref, ct, drop), eps)
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
    if s % 5000 == 0:
        print(f"[dit {s}] eps-mse {loss.item():.4f}", flush=True)
den.load_state_dict(ema)
content.load_state_dict(ema_c)
print(f"[dit] trained in {time.time()-t0:.0f}s", flush=True)
torch.save({"den": den.state_dict(), "content": content.state_dict()}, "/workspace/out/dit6b.pt")


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


print("== gates ==", flush=True)
n = 4
z0 = Zn[:n]
zref = Zn[ref[:n]]
ct = content(T[:n])


def guided(zz, t_start, nsteps, scale):
    ts = list(np.linspace(t_start, 0, nsteps).astype(int))
    with torch.no_grad():
        for i, t in enumerate(ts):
            tt = torch.full((zz.shape[0],), t / TDIFF, device=DEV)
            e_c = den(zz, tt, zref, ct)
            if scale != 1.0:
                e_u = den(zz, tt, torch.zeros_like(zref), ct,
                          torch.ones(zz.shape[0], dtype=torch.bool, device=DEV))
                e = e_u + scale * (e_c - e_u)
            else:
                e = e_c
            z0h = (zz - math.sqrt(1 - ca(t)) * e) / max(math.sqrt(ca(t)), 1e-8)
            zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
                  if i + 1 < len(ts) else z0h)
    return zz


res = {}
rows = [("GT", render(X[0].cpu().numpy())), ("RECON", render(to_xy(Z[:1])[0]))]
zz = math.sqrt(ca(50)) * z0 + math.sqrt(1 - ca(50)) * torch.randn_like(z0)
zz = guided(zz, 50, 50, 1.0)
rt = ((zz - z0).norm(dim=(1, 2)) / z0.norm(dim=(1, 2))).mean().item()
res["round_trip_t50"] = round(rt, 3)
print(f"[gate] round-trip {rt:.3f} (v6 was 0.199)", flush=True)

for scale in (1.0, 2.0):
    torch.manual_seed(0)
    zz = torch.randn(n, ZD, LAT, device=DEV)
    zz = guided(zz, 90, 60, scale)
    with torch.no_grad():
        dd = torch.cdist(zz.reshape(n, -1), Zn.reshape(N, -1)).min(1).values.mean().item()
    res[f"sample_dist_s{scale}"] = round(dd, 1)
    rows.append((f"SAMPLE cfg={scale} dist={dd:.1f}", render(to_xy(MU + SD * zz)[0])))
    print(f"[gate] cfg {scale}: dist {dd:.1f}", flush=True)

out = Image.new("L", (1024, len(rows) * 116 + 20), 255)
dr = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(rows):
    y = 20 + i * 116
    dr.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("/workspace/out/cent_v6b.png")
print(json.dumps({"sheet": "out/cent_v6b.png", **res}), flush=True)
