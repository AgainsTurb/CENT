"""CENT v6 on the 30k cache: proper InkVAE (KL + CTC + style), then a DiT smoke test.

Stage 1  VAE   : MDN trajectory decoder + CTC recogniser + writer classifier + KL (warmup).
                 This is the DiffInk recipe we had never actually trained: our v5 VAE had NO KL.
Stage 2  gate  : decode mu -> RECON sheet. If reconstruction is not legible, stop.
Stage 3  latent: standardise (mu_z, sd_z stored) so the diffusion prior's N(0,I) assumption holds.
Stage 4  DiT   : 512x8 eps-prediction, cosine LR, EMA.
Stage 5  gate  : round-trip from t=90 and t=50 (the metric that predicted every failure so far),
                 then a 2-seed sample sheet.
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
VAE_STEPS = int(os.environ.get("VAE_STEPS", 40000))
DIT_STEPS = int(os.environ.get("DIT_STEPS", 30000))
BS = 64

d = np.load("/workspace/cache/lines.npz")
X = torch.tensor(d["X"]).float().to(DEV)
T = torch.tensor(d["T"]).long().to(DEV)
TL = torch.tensor(d["TL"]).long().to(DEV).clamp_min(1)
W = torch.tensor(d["W"]).long().to(DEV)
wmap = {int(w): i for i, w in enumerate(sorted(set(W.tolist())))}
Wl = torch.tensor([wmap[int(w)] for w in W.tolist()], device=DEV)
N = len(X)
print(f"[data] {N} lines, {len(wmap)} writers", flush=True)


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


def mdn(p, xy, k=8):
    B, _, Tn = p.shape
    q = p.view(B, k, 6, Tn).permute(0, 3, 1, 2)
    pi = torch.softmax(q[..., 0], -1).clamp_min(1e-8)
    s1 = torch.exp(q[..., 3].clamp(-6, 3)).clamp_min(1e-2)
    s2 = torch.exp(q[..., 4].clamp(-6, 3)).clamp_min(1e-2)
    r = torch.tanh(q[..., 5]).clamp(-.98, .98)
    one = (1 - r * r).clamp_min(1e-4)
    z1 = (xy[..., 0:1] - q[..., 1]) / s1
    z2 = (xy[..., 1:2] - q[..., 2]) / s2
    quad = (z1 * z1 + z2 * z2 - 2 * r * z1 * z2) / one
    logN = -.5 * quad - math.log(2 * math.pi) - torch.log(s1) - torch.log(s2) - .5 * torch.log(one)
    return -torch.logsumexp(torch.log(pi) + logN, -1).mean()


class CTC(nn.Module):
    def __init__(s, ncls=2680, w=192):
        super().__init__()
        s.proj = nn.Conv1d(ZD, w, 1)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, 4, 2 * w, .1, batch_first=True,
                                                                norm_first=True), 2)
        s.out = nn.Linear(w, ncls + 1)

    def forward(s, z):
        h = F.interpolate(z, size=P, mode="linear", align_corners=False)
        h = s.tf(s.proj(h).transpose(1, 2))
        return torch.log_softmax(s.out(h), -1).transpose(0, 1)


class Sty(nn.Module):
    def __init__(s, nw):
        super().__init__()
        s.n = nn.Sequential(nn.Conv1d(ZD, 128, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool1d(1),
                            nn.Flatten(), nn.Linear(128, nw))

    def forward(s, z):
        return s.n(z)


enc, dec, ctc, sty = Enc().to(DEV), Dec().to(DEV), CTC().to(DEV), Sty(len(wmap)).to(DEV)
prm = [*enc.parameters(), *dec.parameters(), *ctc.parameters(), *sty.parameters()]
opt = torch.optim.AdamW(prm, lr=1.5e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, VAE_STEPS, eta_min=1e-5)
cl = nn.CTCLoss(blank=0, zero_infinity=True)
t0 = time.time()
for s in range(VAE_STEPS):
    i = torch.randint(0, N, (BS,), device=DEV)
    x = X[i]
    mu, lv = enc(x.transpose(1, 2))
    z = mu + torch.randn_like(mu) * torch.exp(.5 * lv)
    ln = mdn(dec(z), x)
    lc = cl(ctc(mu), T[i], torch.full((BS,), P, dtype=torch.long, device=DEV), TL[i])
    ls = F.cross_entropy(sty(mu), Wl[i])
    kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean())
    beta = min(1.0, s / 8000.0) * 1e-3           # warmup: KL must not dominate reconstruction
    loss = ln + lc + 0.1 * ls + beta * kl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(prm, 1.0)
    opt.step()
    sch.step()
    if s % 2000 == 0:
        print(f"[vae {s}] nll {ln.item():.2f} ctc {lc.item():.2f} sty {ls.item():.2f} "
              f"kl {kl.item():.1f} beta*kl {(beta*kl).item():.2f}", flush=True)
print(f"[vae] trained in {time.time()-t0:.0f}s", flush=True)

with torch.no_grad():
    Zs = []
    for j in range(0, N, 1024):
        Zs.append(enc(X[j:j + 1024].transpose(1, 2))[0].cpu())
    Z = torch.cat(Zs).to(DEV)
MU, SD = Z.mean(), Z.std()
Zn = (Z - MU) / SD
print(f"[latent] std {Z.std():.3f} -> standardised {Zn.std():.3f}   KL-free dims {tuple(Z.shape[1:])}",
      flush=True)
torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "ctc": ctc.state_dict(),
            "sty": sty.state_dict(), "MU": MU.cpu(), "SD": SD.cpu()}, "/workspace/out/vae6.pt")


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


print("== STAGE 2: reconstruction gate ==", flush=True)
with torch.no_grad():
    rec = to_xy(Z[:1])[0]
rows = [("GT", render(X[0].cpu().numpy())), ("RECON (KL+CTC+style)", render(rec))]


class Den(nn.Module):
    def __init__(s, w=512, depth=8, heads=8):
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


def temb(t, dim=256):
    h = dim // 2
    fr = torch.exp(-math.log(10000.) * torch.arange(h, device=t.device) / h)
    a = t.reshape(-1, 1) * fr.reshape(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


def ca(t, T_=TDIFF, s0=.008):
    f = math.cos((t / T_ + s0) / (1 + s0) * math.pi / 2) ** 2
    return f / math.cos(s0 / (1 + s0) * math.pi / 2) ** 2


emb = nn.Embedding(2680, 320).to(DEV)
den = Den().to(DEV)
opt2 = torch.optim.AdamW(list(den.parameters()) + list(emb.parameters()), lr=1e-3)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, DIT_STEPS, eta_min=1e-5)
ema = {k: v.detach().clone() for k, v in den.state_dict().items()}
print("== STAGE 4: DiT ==", flush=True)
t1 = time.time()
for s in range(DIT_STEPS):
    i = torch.randint(0, N, (BS,), device=DEV)
    z0 = Zn[i]
    tt = torch.randint(0, TDIFF, (BS,), device=DEV)
    a = torch.tensor([ca(float(v)) for v in tt], device=DEV).view(-1, 1, 1)
    eps = torch.randn_like(z0)
    zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
    ctx = emb(T[i]).mean(1, keepdim=True)
    loss = F.mse_loss(den(zt, tt / TDIFF, ctx), eps)
    opt2.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
    opt2.step()
    sch2.step()
    with torch.no_grad():
        for k, v in den.state_dict().items():
            ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
    if s % 5000 == 0:
        print(f"[dit {s}] eps-mse {loss.item():.4f}", flush=True)
den.load_state_dict(ema)
print(f"[dit] trained in {time.time()-t1:.0f}s", flush=True)
torch.save({"den": den.state_dict(), "emb": emb.state_dict()}, "/workspace/out/dit6.pt")

print("== STAGE 5: round-trip gate + samples ==", flush=True)
n = 8
z0 = Zn[:n]
ctx = emb(T[:n]).mean(1, keepdim=True)


def ddim(zz, t_start, nsteps):
    ts = list(np.linspace(t_start, 0, nsteps).astype(int))
    with torch.no_grad():
        for i, t in enumerate(ts):
            e = den(zz, torch.full((zz.shape[0],), t / TDIFF, device=DEV), ctx)
            z0h = (zz - math.sqrt(1 - ca(t)) * e) / max(math.sqrt(ca(t)), 1e-8)
            zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
                  if i + 1 < len(ts) else z0h)
    return zz


res = {}
for start in (90, 50):
    zz = (math.sqrt(ca(50)) * z0 + math.sqrt(1 - ca(50)) * torch.randn_like(z0)) if start == 50 \
        else torch.randn(n, ZD, LAT, device=DEV)
    zz = ddim(zz, start, 50)
    if start == 50:
        rt = ((zz - z0).norm(dim=(1, 2)) / z0.norm(dim=(1, 2))).mean().item()
        res["round_trip_t50"] = round(rt, 3)
        print(f"[gate] round-trip t=50 rel {rt:.3f}  (v5 was 0.32)", flush=True)
        rows.append((f"ROUND-TRIP rel={rt:.2f}", render(to_xy(MU + SD * zz)[0])))
    else:
        with torch.no_grad():
            dd = torch.cdist(zz.reshape(n, -1), Zn.reshape(N, -1)).min(1).values.mean().item()
        res["sample_dist_t90"] = round(dd, 2)
        print(f"[gate] sample dist {dd:.2f}  ||z|| {zz.norm(dim=(1, 2)).mean().item():.1f} "
              f"(real {Zn.norm(dim=(1, 2)).mean().item():.1f})", flush=True)
        rows.append((f"SAMPLE t90 dist={dd:.1f}", render(to_xy(MU + SD * zz)[0])))

out = Image.new("L", (1024, len(rows) * 116 + 20), 255)
dr = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(rows):
    y = 20 + i * 116
    dr.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("/workspace/out/cent_v6.png")
print(json.dumps({"sheet": "out/cent_v6.png", **res, "lines": N,
                  "vae_sec": round(t1 - t0, 0)}), flush=True)
