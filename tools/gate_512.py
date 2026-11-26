"""Local gate before any GPU spend.

 1. warm-start `back` by MSE (the step that was missing and caused the divergence)
 2. fine-tune the 18432 -> 512 bottleneck with MDN nll, render RECON
 3. train a small denoiser on the 512-d latents
 4. ADHERENCE: different transcripts, same seed vs same transcript, different seed
    pass = ratio > 0.30, RECON legible
"""
import sys
sys.path.insert(0, "A:/Papers/CENT")
import json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, ZD, LAT, KB, ZB = 384, 96, 96, 8, 512
NL, BS = 2000, 32
TDIFF = 100

from cent.data_strokes import StrokeLineDataset


def _refs(self, w, e):
    return torch.zeros(int(self.k_refs), int(self.img_h), 64), torch.ones(int(self.k_refs))


StrokeLineDataset._refs = _refs
base = StrokeLineDataset("A:/Papers/CENT/data/canonical_lines/v1", "train", k_refs=1,
                         img_h=64, canvas_w=1024, paired_only=False, limit=0)
X, T, W = [], [], []
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
    X.append(np.stack([np.interp(idx, np.arange(len(p)), p[:, 0]),
                       np.interp(idx, np.arange(len(p)), p[:, 1])], 1))
    text = base.records[rid].get("text", "") or ""
    a = np.zeros(32, np.int64)
    for k in range(min(len(text), 32)):
        a[k] = base.char2idx.get(text[k], base.pad_id)
    T.append(a)
    W.append(int(base.records[rid]["writer"]))
X = torch.tensor(np.stack(X)).float().to(DEV)
T = torch.tensor(np.stack(T)).long().to(DEV)
W = torch.tensor(W).long().to(DEV)
N = len(X)
print(f"[data] {N} lines", flush=True)


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
    def __init__(s, k=KB, h=192):
        super().__init__()
        s.k = k
        s.up = nn.Sequential(nn.ConvTranspose1d(ZD, h, 4, 2, 1), nn.GELU(),
                             nn.ConvTranspose1d(h, h, 4, 2, 1), nn.GELU(), nn.Conv1d(h, 6 * k, 1))

    def forward(s, z):
        return F.interpolate(s.up(z), size=P, mode="linear", align_corners=False)


def mdn(p, xy, k=KB):
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


enc, dec = Enc().to(DEV), Dec().to(DEV)
ck = torch.load("runs/v5/vae.pt", map_location=DEV)
enc.load_state_dict(ck["enc"])
dec.load_state_dict(ck["dec"])
enc.eval()
for p in list(enc.parameters()) + list(dec.parameters()):
    p.requires_grad_(False)
with torch.no_grad():
    Zf = torch.cat([enc(X[j:j + 512].transpose(1, 2))[0] for j in range(0, N, 512)])
Zflat = Zf.reshape(N, -1)
nll0 = mdn(dec(Zf), X).item()
print(f"[base] nll {nll0:.3f}", flush=True)

proj = nn.Sequential(nn.Linear(ZD * LAT, ZB), nn.GELU(), nn.Linear(ZB, ZB)).to(DEV)
back = nn.Linear(ZB, ZD * LAT).to(DEV)

# ---- step 1: warm start `back` so back(proj(z)) ~= z (this is what was missing before)
opt = torch.optim.AdamW(list(proj.parameters()) + list(back.parameters()), lr=2e-3)
for s in range(400):
    i = torch.randint(0, N, (128,), device=DEV)
    loss = F.mse_loss(back(proj(Zflat[i])), Zflat[i])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
with torch.no_grad():
    ae = F.mse_loss(back(proj(Zflat)), Zflat).item()
print(f"[warm start] autoencode MSE on z {ae:.5f} (z var {Zflat.var():.5f})", flush=True)

# ---- step 2: nll fine-tune
for s in range(1500):
    i = torch.randint(0, N, (BS,), device=DEV)
    z = back(proj(Zflat[i])).view(-1, ZD, LAT)
    loss = mdn(dec(z), X[i])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if s % 500 == 0:
        print(f"[ft {s}] nll {loss.item():.3f}", flush=True)

with torch.no_grad():
    Zb = proj(Zflat)
    nllb = mdn(dec(back(Zb).view(-1, ZD, LAT)), X).item()
    q = dec(back(Zb)[:1].view(-1, ZD, LAT)).view(-1, KB, 6, P).permute(0, 3, 1, 2)
    pi = torch.softmax(q[..., 0], -1)
    rec = torch.stack([(q[..., 1] * pi).sum(-1)[0], (q[..., 2] * pi).sum(-1)[0]], -1).detach().cpu().numpy()[0]
print(f"[bottleneck] nll {nllb:.3f}  (full {nll0:.3f})", flush=True)

MU, SD = Zb.mean(), Zb.std()
Zn = (Zb - MU) / SD
res = {"nll_full": round(nll0, 3), "nll_bottleneck": round(nllb, 3), "autoencode_mse": round(ae, 5)}

ref = torch.zeros(N, dtype=torch.long, device=DEV)
wn = W.cpu().numpy()
for w in np.unique(wn):
    idx = np.where(wn == w)[0]
    ref[idx] = torch.tensor(np.roll(idx, 1) if len(idx) > 1 else idx, device=DEV)

# ---- step 3: denoiser on 512-d vectors
emb = nn.Embedding(2680 + 1, 256).to(DEV)


def cvec(ids):
    e = emb(ids)
    return e.mean(1)


def temb(t, d=256):
    h = d // 2
    fr = torch.exp(-math.log(10000.) * torch.arange(h, device=t.device) / h)
    a = t.reshape(-1, 1) * fr.reshape(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


class Den(nn.Module):
    def __init__(s, w=1024):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(ZB * 2 + 256 + 256, w), nn.SiLU(),
                              nn.Linear(w, w), nn.SiLU(), nn.Linear(w, ZB))

    def forward(s, zt, t, zref, c):
        x = torch.cat([zt, zref, c, temb(t)], -1)
        return s.net(x)


den = Den().to(DEV)
opt2 = torch.optim.AdamW(list(den.parameters()) + list(emb.parameters()), lr=1e-3)
P_DROP = 0.15
t0 = time.time()
for s in range(8000):
    i = torch.randint(0, N, (64,), device=DEV)
    z0 = Zn[i]
    tt = torch.randint(0, TDIFF, (64,), device=DEV)
    a = torch.cos((tt / TDIFF + .008) / 1.008 * math.pi / 2) ** 2
    a = (a / math.cos(.008 / 1.008 * math.pi / 2) ** 2).view(-1, 1)
    eps = torch.randn_like(z0)
    zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
    c = cvec(T[i])
    drop = torch.rand(64, device=DEV) < P_DROP
    c = c.clone()
    c[drop] = 0.0
    loss = F.mse_loss(den(zt, tt / TDIFF, Zn[ref[i]], c), eps)
    opt2.zero_grad(set_to_none=True)
    loss.backward()
    opt2.step()
    if s % 2000 == 0:
        print(f"[den {s}] eps-mse {loss.item():.4f}", flush=True)
print(f"[den] trained in {time.time()-t0:.0f}s", flush=True)


def ca(t):
    f = math.cos((t / TDIFF + .008) / 1.008 * math.pi / 2) ** 2
    return f / math.cos(.008 / 1.008 * math.pi / 2) ** 2


def sample(ids, zr, seed, nsteps=40, t_start=90):
    torch.manual_seed(seed)
    c = cvec(ids)
    zz = torch.randn(ids.shape[0], ZB, device=DEV)
    ts = list(np.linspace(t_start, 0, nsteps).astype(int))
    with torch.no_grad():
        for i, t in enumerate(ts):
            e = den(zz, torch.full((ids.shape[0],), t / TDIFF, device=DEV), zr, c)
            z0h = (zz - math.sqrt(1 - ca(t)) * e) / max(math.sqrt(ca(t)), 1e-8)
            zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
                  if i + 1 < len(ts) else z0h)
    return zz


# ---- step 4: adherence
ida = T[0:1].repeat(2, 1)
idb = T[100:101].repeat(2, 1)
zr = Zn[ref[0:1]].repeat(2, 1)
s1 = sample(ida, zr, 0)
s2 = sample(ida, zr, 1)
s3 = sample(idb, zr, 0)
div = (s1 - s2).norm(dim=1).mean().item()
sens = (s1 - s3).norm(dim=1).mean().item()
ratio = sens / max(div, 1e-8)
res.update({"sens": round(sens, 3), "div": round(div, 3), "ratio": round(ratio, 3)})
print(f"[adherence] sens {sens:.2f}  div {div:.2f}  ratio {ratio:.3f}  (pass > 0.30)", flush=True)


def render(xy, w=1024, h=96):
    img = np.ones((h, w), np.float32)
    pp = (np.clip(xy, 0, 1) * np.array([w - 1, h - 1])).astype(int)
    for a, b in zip(pp[:-1], pp[1:]):
        n = int(np.hypot(*(b - a))) + 1
        for u in np.linspace(0, 1, n):
            x, y = (a + (b - a) * u).astype(int)
            img[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = 0
    return img


def to_xy(zb):
    z = back(zb).view(-1, ZD, LAT)
    q = dec(z).view(-1, KB, 6, P).permute(0, 3, 1, 2)
    pi = torch.softmax(q[..., 0], -1)
    return torch.stack([(q[..., 1] * pi).sum(-1), (q[..., 2] * pi).sum(-1)], -1).detach().cpu().numpy()


rows = [("GT", render(X[0].cpu().numpy())),
        ("RECON via 512-d", render(to_xy(Zb[:1])[0])),
        ("SAMPLE text=line0", render(to_xy(MU + SD * s1)[0])),
        ("SAMPLE text=line100", render(to_xy(MU + SD * s3)[0]))]
out = Image.new("L", (1024, len(rows) * 116 + 20), 255)
dr = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(rows):
    y = 20 + i * 116
    dr.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("results/gate_512.png")
print(json.dumps({"sheet": "results/gate_512.png", **res}), flush=True)
