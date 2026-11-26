"""CENT v6d: (1) measure text adherence, (2) repair conditioning by CHANNEL CONCAT.

Step 1 is a measurement we have never made and that every future run must pass:
    text sensitivity  = || sample(T_a, seed s) - sample(T_b, seed s) ||
    seed diversity    = || sample(T_a, seed s1) - sample(T_a, seed s2) ||
    ratio = sensitivity / diversity      (0 => the transcript is ignored)

Step 2 is DiffInk's actual fusion: content is concatenated CHANNEL-WISE with the noisy latent
and the style prompt, [x_t, x_ref, Z], so the network cannot route around it. v6b/v6c injected
content only through cross-attention, and guidance had no effect at all (95.3 / 95.4 / 95.8),
which is what "ignored" looks like.
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
STEPS = int(os.environ.get("STEPS", 30000))
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
        return s.net(F.interpolate(e.transpose(1, 2), size=LAT, mode="nearest"))


class DenCat(nn.Module):
    """Content fused by CHANNEL CONCAT: [x_t, x_ref, Z] -> first conv (DiffInk's design)."""

    def __init__(s, w=512, depth=8, heads=8, dtext=512):
        super().__init__()
        s.inp = nn.Conv1d(2 * ZD + dtext, w, 1)
        s.tm = nn.Sequential(nn.Linear(256, w), nn.SiLU(), nn.Linear(w, w))
        s.bl = nn.ModuleList([nn.TransformerEncoderLayer(w, heads, 4 * w, .1, batch_first=True,
                                                         norm_first=True) for _ in range(depth)])
        s.norm = nn.LayerNorm(w)
        s.out = nn.Conv1d(w, ZD, 1)

    def forward(s, zt, t, zref, ctext):
        h = s.inp(torch.cat([zt, zref, ctext], 1)).transpose(1, 2)
        h = h + s.tm(temb(t)).unsqueeze(1)
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

content, den = Content().to(DEV), DenCat().to(DEV)
ckd = torch.load("/workspace/out/dit6c.pt", map_location=DEV)
content.load_state_dict(ckd["content"])
print("[warm start] content encoder loaded from dit6c (denoiser rebuilt for concat)", flush=True)


def sample(model, ids, zr, seed, nsteps=60, t_start=90):
    torch.manual_seed(seed)
    B = ids.shape[0]
    ct = content(ids)
    zz = torch.randn(B, ZD, LAT, device=DEV)
    ts = list(np.linspace(t_start, 0, nsteps).astype(int))
    with torch.no_grad():
        for i, t in enumerate(ts):
            e = model(zz, torch.full((B,), t / TDIFF, device=DEV), zr, ct)
            z0h = (zz - math.sqrt(1 - ca(t)) * e) / max(math.sqrt(ca(t)), 1e-8)
            zz = (math.sqrt(ca(ts[i + 1])) * z0h + math.sqrt(1 - ca(ts[i + 1])) * e
                  if i + 1 < len(ts) else z0h)
    return zz


def adherence(model, tag):
    """ratio = text sensitivity / seed diversity.  0 => transcript ignored."""
    ids_a = T[0:1].repeat(2, 1)
    ids_b = T[100:101].repeat(2, 1)
    zr = Zn[ref[0:1]].repeat(2, 1, 1)
    s1 = sample(model, ids_a, zr, 0)
    s2 = sample(model, ids_a, zr, 1)          # same text, different seed
    s3 = sample(model, ids_b, zr, 0)          # different text, same seed
    div = (s1 - s2).norm(dim=(1, 2)).mean().item()
    sens = (s1 - s3).norm(dim=(1, 2)).mean().item()
    print(f"[adherence {tag}] text-sensitivity {sens:.2f}  seed-diversity {div:.2f}  "
          f"ratio {sens/max(div,1e-6):.3f}", flush=True)
    return {"tag": tag, "sens": round(sens, 2), "div": round(div, 2),
            "ratio": round(sens / max(div, 1e-6), 3)}, s1, s3


res = {}
before, s1_b, s3_b = adherence(den, "before (concat, untrained)")

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
    ct = content(T[i])
    drop = torch.rand(BS, device=DEV) < P_DROP
    ct = ct.clone()
    ct[drop] = 0.0                                   # content dropped, style kept (DiffInk)
    loss = F.mse_loss(den(zt, tt / TDIFF, Zn[ref[i]], ct), eps)
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
        print(f"[ditd {s}] eps-mse {loss.item():.4f}", flush=True)
den.load_state_dict(ema)
content.load_state_dict(ema_c)
print(f"[ditd] trained in {time.time()-t0:.0f}s", flush=True)
torch.save({"den": den.state_dict(), "content": content.state_dict()}, "/workspace/out/dit6d.pt")

after, s1_a, s3_a = adherence(den, "after")
res["before"] = before
res["after"] = after


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


rows = [("GT (line 0)", render(X[0].cpu().numpy())),
        ("RECON", render(to_xy(Z[:1])[0])),
        ("SAMPLE text=line0", render(to_xy(MU + SD * s1_a)[0])),
        ("SAMPLE text=line100", render(to_xy(MU + SD * s3_a)[0]))]
out = Image.new("L", (1024, len(rows) * 116 + 20), 255)
dr = ImageDraw.Draw(out)
for i, (lab, a) in enumerate(rows):
    y = 20 + i * 116
    dr.text((6, y - 14), lab, fill=0)
    out.paste(Image.fromarray((a * 255).astype(np.uint8)), (0, y))
out.save("/workspace/out/cent_v6d.png")
print(json.dumps({"sheet": "out/cent_v6d.png", **res}), flush=True)
