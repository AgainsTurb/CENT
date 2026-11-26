"""Is the latent compressible to a diffusion-friendly size, and does text predict it better?

1. Fine-tune a 18432 -> 512 bottleneck (flatten, linear, linear back, unflatten) between the
   frozen v5 encoder and decoder. Reconstruction must survive: measured as MDN nll and a
   rendered RECON row.
2. In the 512-dim space, measure again:
       mean-baseline rel.err  (how much of the norm is line-specific)
       text -> z  rel.err     (how much of that the transcript explains)
   A learnable latent needs the text number to fall well below the mean number.
"""
import sys
sys.path.insert(0, "A:/Papers/CENT")
import json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, ZD, LAT, KB = 384, 96, 96, 8
NL, STEPS, BS = 2000, 1200, 32
ZB = 512

from cent.data_strokes import StrokeLineDataset


def _refs(self, w, e):
    return torch.zeros(int(self.k_refs), int(self.img_h), 64), torch.ones(int(self.k_refs))


StrokeLineDataset._refs = _refs
base = StrokeLineDataset("A:/Papers/CENT/data/canonical_lines/v1", "train", k_refs=1,
                         img_h=64, canvas_w=1024, paired_only=False, limit=0)
X, T32 = [], []
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
    T32.append(a)
X = torch.tensor(np.stack(X)).float().to(DEV)
T = torch.tensor(np.stack(T32)).long().to(DEV)
N = len(X)


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
for p in enc.parameters():
    p.requires_grad_(False)
for p in dec.parameters():
    p.requires_grad_(False)

proj = nn.Sequential(nn.Linear(ZD * LAT, ZB), nn.GELU(), nn.Linear(ZB, ZB)).to(DEV)
back = nn.Sequential(nn.Linear(ZB, ZD * LAT)).to(DEV)


def enc_mu(x):
    with torch.no_grad():
        mu, _ = enc(x.transpose(1, 2))
    return mu


with torch.no_grad():
    Zfull = torch.cat([enc_mu(X[j:j + 512]) for j in range(0, N, 512)])
res = {}
print(f"[base] nll without bottleneck {mdn(dec(Zfull), X).item():.3f}", flush=True)

opt = torch.optim.AdamW(list(proj.parameters()) + list(back.parameters()), lr=2e-3)
t0 = __import__("time").time()
for s in range(STEPS):
    i = torch.randint(0, N, (BS,), device=DEV)
    z = back(proj(Zfull[i].reshape(BS, -1))).view(-1, ZD, LAT)
    loss = mdn(dec(z), X[i])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if s % 400 == 0:
        print(f"[ft {s}] nll {loss.item():.3f}", flush=True)
print(f"[ft] {STEPS} steps in {__import__('time').time()-t0:.0f}s", flush=True)

with torch.no_grad():
    Zb = proj(Zfull.reshape(N, -1))
    recon_z = back(Zb).view(-1, ZD, LAT)
    nll_b = mdn(dec(recon_z), X).item()
    q = dec(recon_z[:1]).view(-1, KB, 6, P).permute(0, 3, 1, 2)
    pi = torch.softmax(q[..., 0], -1)
    rec = torch.stack([(q[..., 1] * pi).sum(-1)[0], (q[..., 2] * pi).sum(-1)[0]], -1).detach().cpu().numpy()[0]
print(f"[bottleneck] nll {nll_b:.3f}  (was {mdn(dec(Zfull), X).item():.3f})", flush=True)

Zb = (Zb - Zb.mean()) / Zb.std()
res["nll_full"] = round(mdn(dec(Zfull), X).item(), 3)
res["nll_bottleneck"] = round(nll_b, 3)


def rel(pred, tgt):
    return (torch.norm(pred - tgt, dim=1) / torch.norm(tgt, dim=1)).mean().item()


mean_pred = Zb.mean(0, keepdim=True).expand_as(Zb)
res["mean_baseline"] = round(rel(mean_pred, Zb), 3)
print(f"[512-d] mean-baseline rel.err {res['mean_baseline']:.3f}", flush=True)

emb = nn.Embedding(2680, 128).to(DEV)
head = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, ZB)).to(DEV)
o2 = torch.optim.AdamW(list(emb.parameters()) + list(head.parameters()), lr=2e-3)
for s in range(1200):
    i = torch.randint(0, N, (64,), device=DEV)
    loss = F.mse_loss(head(emb(T[i]).mean(1)), Zb[i])
    o2.zero_grad(set_to_none=True)
    loss.backward()
    o2.step()
with torch.no_grad():
    r = rel(head(emb(T).mean(1)), Zb)
res["text_to_z"] = round(r, 3)
print(f"[512-d] text -> z rel.err {r:.3f}   (want well below {res['mean_baseline']:.3f})", flush=True)


def render(xy, w=1024, h=96):
    img = np.ones((h, w), np.float32)
    pp = (np.clip(xy, 0, 1) * np.array([w - 1, h - 1])).astype(int)
    for a, b in zip(pp[:-1], pp[1:]):
        n = int(np.hypot(*(b - a))) + 1
        for u in np.linspace(0, 1, n):
            x, y = (a + (b - a) * u).astype(int)
            img[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = 0
    return img


rows = [("GT", render(X[0].cpu().numpy())), ("RECON via 512-d bottleneck", render(rec))]
out = Image.new("L", (1024, 2 * 116 + 20), 255)
dr = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(rows):
    y = 20 + i * 116
    dr.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("results/bottleneck_test.png")
torch.save({"proj": proj.state_dict(), "back": back.state_dict()}, "runs/v5/bottleneck.pt")
print(json.dumps({"sheet": "results/bottleneck_test.png", **res}), flush=True)
