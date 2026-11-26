"""CENT v4 — end-to-end rewrite modelled on DiffInk.

Two stages, exactly as DiffInk does it:

    InkVAE   : (x, y, pen) trajectory  ->  latent z            [1D conv encoder/decoder,
                                                                 MDN + pen head, KL + style]
    InkDiT   : noise + text + style prompt -> z                 [small diffusion transformer]

Why this replaces CENT v3's autoregressive Bezier MDN, in one line: there is no chained
decoding, so nothing compounds along the line, and the latent is regularised by a style
classifier, so style is a trained property rather than an added vector.

Smoke test (CPU or GPU, minutes):

    python tools/cent_v4.py --lines 200 --vae-steps 300 --dit-steps 300
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3

MAXP = 384          # points per line (resampled)
LAT = 96            # latent length
ZDIM = 64           # latent channels
TDIFF = 100         # diffusion steps


# --------------------------------------------------------------------------- data
def resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Linear resample a polyline to n points (keeps geometry, fixes length)."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], axis=1)


def build_line(rid, base, n_pts=MAXP):
    """strokes -> one (x, y, pen) trajectory, pen = 1 at a stroke break."""
    st = base.strokes_of(rid)
    out, pen = [], []
    for k, s in enumerate(st):
        a = np.asarray(s, np.float32)
        a = resample(a, max(6, int(len(a))))
        out.append(a)
        p = np.zeros(len(a), np.float32)
        if k < len(st) - 1:
            p[-1] = 1.0                      # pen-up marker at the end of the stroke
        pen.append(p)
    pts = np.concatenate(out, 0)
    pn = np.concatenate(pen, 0)
    lo = pts.min(0); hi = pts.max(0)
    pts = (pts - lo) / np.maximum(hi - lo, 1e-6)          # normalise to the unit box
    t = np.linspace(0, len(pts) - 1, n_pts)
    x = np.interp(t, np.arange(len(pts)), pts[:, 0])
    y = np.interp(t, np.arange(len(pts)), pts[:, 1])
    p = np.interp(t, np.arange(len(pts)), pn) > 0.5
    pen3 = np.stack([1.0 - p, p, np.zeros_like(p)], 1).astype(np.float32)   # down, up, eoc
    xy = np.stack([x, y], 1).astype(np.float32)
    return xy, pen3


# --------------------------------------------------------------------------- VAE
class Enc(nn.Module):
    def __init__(self, cin=5, h=128, z=ZDIM):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, h, 5, 2, 2), nn.GELU(),
                                 nn.Conv1d(h, h, 5, 2, 2), nn.GELU(),
                                 nn.Conv1d(h, h, 5, 2, 2), nn.GELU())
        self.mu = nn.Conv1d(h, z, 1); self.lv = nn.Conv1d(h, z, 1)

    def forward(self, x):
        f = self.net(x)
        return self.mu(f), self.lv(f)


class Dec(nn.Module):
    def __init__(self, z=ZDIM, h=128, k=8, npts=MAXP):
        super().__init__()
        self.npts, self.k = npts, k
        self.up = nn.Sequential(nn.ConvTranspose1d(z, h, 4, 2, 1), nn.GELU(),
                                nn.ConvTranspose1d(h, h, 4, 2, 1), nn.GELU(),
                                nn.ConvTranspose1d(h, h, 4, 2, 1), nn.GELU(),
                                nn.Conv1d(h, 6 * k + 3, 1))

    def forward(self, z):
        o = self.up(z)
        o = F.interpolate(o, size=self.npts, mode="linear", align_corners=False)
        return o[:, :6 * self.k], o[:, 6 * self.k:]          # MDN params, pen logits


def mdn_nll(params, xy, k=8):
    """Negative log-likelihood of a 2D mixture density (Graves form).

        log N = -0.5*quad - log(2*pi) - log(s1) - log(s2) - 0.5*log(1-rho^2)
        NLL   = -logsumexp(log pi + log N)
    (An earlier version halved the quadratic twice, which produced a non-physical 0.77
    and then NaN once the variances collapsed.)
    """
    B, _, T = params.shape
    p = params.view(B, k, 6, T).permute(0, 3, 1, 2)          # [B,T,k,6]
    pi = torch.softmax(p[..., 0], -1).clamp_min(1e-8)
    mu1, mu2 = p[..., 1], p[..., 2]
    s1 = torch.exp(p[..., 3].clamp(-6, 3)).clamp_min(1e-2)
    s2 = torch.exp(p[..., 4].clamp(-6, 3)).clamp_min(1e-2)
    rho = torch.tanh(p[..., 5]).clamp(-0.98, 0.98)
    one = (1 - rho ** 2).clamp_min(1e-4)
    z1 = (xy[..., 0:1] - mu1) / s1
    z2 = (xy[..., 1:2] - mu2) / s2
    quad = (z1 ** 2 + z2 ** 2 - 2 * rho * z1 * z2) / one
    logN = -0.5 * quad - math.log(2 * math.pi) - torch.log(s1) - torch.log(s2) \
           - 0.5 * torch.log(one)
    return -torch.logsumexp(torch.log(pi) + logN, dim=-1).mean()


class InkVAE(nn.Module):
    def __init__(self, n_writers, n_classes=2680):
        super().__init__()
        self.enc, self.dec = Enc(), Dec()
        self.style = nn.Sequential(nn.Conv1d(ZDIM, 128, 3, 2, 1), nn.GELU(),
                                   nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Linear(128, n_writers))
        # CTC ON THE LATENT. DiffInk's key regulariser: a recogniser must be able to read the
        # text out of z, which forces the latent to carry glyph identity instead of drifting.
        self.ctc_proj = nn.Conv1d(ZDIM, 128, 1)
        self.ctc_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True, norm_first=True), 2)
        self.ctc_out = nn.Linear(128, n_classes + 1)

    def ctc_logits(self, z):
        h = F.interpolate(z, size=MAXP, mode="linear", align_corners=False)
        h = self.ctc_proj(h).transpose(1, 2)
        h = self.ctc_out(self.ctc_tf(h))
        return torch.log_softmax(h, -1).transpose(0, 1)      # [T,B,V] for CTCLoss

    def encode(self, x): return self.enc(x)

    def forward(self, x, xy):
        mu, lv = self.enc(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        mp, pp = self.dec(z)
        nll = mdn_nll(mp, xy)
        lpen = F.binary_cross_entropy_with_logits(pp[:, 1], x[:, 1])
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).mean()
        return z, nll, lpen, kl


# --------------------------------------------------------------------------- diffusion
def _temb(t, dim=256):
    """Sinusoidal timestep embedding (the DiT MLP expects a fixed-width vector)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([a.sin(), a.cos()], dim=-1)


class Denoiser(nn.Module):
    """Small DiT-style denoiser: adaLN on the timestep, cross-attention to the text tokens."""
    def __init__(self, z=ZDIM, d=128, heads=4, depth=4, tdim=256):
        super().__init__()
        self.inp = nn.Conv1d(z, d, 1)
        self.tmlp = nn.Sequential(nn.Linear(tdim, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1,
                                                                batch_first=True,
                                                                norm_first=True)
                                     for _ in range(depth)])
        self.ctx = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Conv1d(d, z, 1)

    def forward(self, zt, t, txt):
        h = self.inp(zt).transpose(1, 2)
        h = h + self.tmlp(_temb(t)).unsqueeze(1)
        h = self.ctx(h, txt, txt, need_weights=False)[0] + h
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h).transpose(1, 2))


def cosine_alpha(t, T=TDIFF, s=0.008):
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    return f / torch.cos((torch.tensor(0.0) / T + s) / (1 + s) * math.pi / 2) ** 2


@torch.no_grad()
def ddim(den, txt, shape, steps=20, dev="cpu"):
    z = torch.randn(shape, device=dev)
    ts = torch.linspace(TDIFF - 1, 0, steps).long()
    for i, t in enumerate(ts):
        tt = torch.full((shape[0],), int(t), device=dev)
        ab = cosine_alpha(torch.tensor(float(t), device=dev))
        eps = den(z, tt / TDIFF, txt)
        z0 = (z - (1 - ab).sqrt() * eps) / ab.sqrt()
        if i + 1 < len(ts):
            an = cosine_alpha(torch.tensor(float(ts[i + 1]), device=dev))
            z = an.sqrt() * z0 + (1 - an).sqrt() * eps
        else:
            z = z0
    return z


# --------------------------------------------------------------------------- render
def render(xy, w=1024, h=96, thick=1):
    img = np.ones((h, w), np.float32)
    p = (np.clip(xy, 0, 1) * np.array([w - 1, h - 1])).astype(int)
    for a, b in zip(p[:-1], p[1:]):
        n = int(np.hypot(*(b - a))) + 1
        for u in np.linspace(0, 1, n):
            x, y = (a + (b - a) * u).astype(int)
            img[max(y - thick, 0):y + thick + 1, max(x - thick, 0):x + thick + 1] = 0.0
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=600)
    ap.add_argument("--vae-steps", type=int, default=300)
    ap.add_argument("--dit-steps", type=int, default=300)
    ap.add_argument("--out", default="results/cent_v4.png")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    cfg = V3Config.load("cent/configs/v3_overfit1.json")
    ds = StrokeDatasetV3(cfg.canonical_root, "train", k_refs=cfg.k_refs, img_h=cfg.img_height,
                         canvas_w=cfg.canvas_width, max_chars=cfg.max_chars,
                         max_strokes_char=cfg.max_strokes_char,
                         max_strokes_line=cfg.max_strokes_line, join_tau=cfg.join_tau,
                         n_classes=cfg.n_classes, limit=a.lines)
    xs, ys, ws, ts, tlens = [], [], [], [], []
    for i in range(len(ds)):
        rid = ds.base.usable[i]
        xy, pen = build_line(rid, ds.base)
        xs.append(np.concatenate([xy, pen], 1))
        ys.append(xy)
        ws.append(int(ds.base.records[rid]["writer"]))
        nc = int(ds[i]["n_chars"])
        tvec = ds[i]["text"][:nc].numpy().astype(np.int64)
        ids = np.zeros(8, np.int64); ids[:min(len(tvec), 8)] = tvec[:8]
        ts.append(ids); tlens.append(min(len(tvec), 8))
    X = torch.tensor(np.stack(xs)).transpose(1, 2).to(dev)      # [B,5,P]
    XY = torch.tensor(np.stack(ys)).to(dev)                     # [B,P,2]
    W = torch.tensor(ws).to(dev); T = torch.tensor(np.stack(ts)).to(dev)
    TL = torch.tensor(tlens).to(dev)
    wmap = {int(w): j for j, w in enumerate(sorted(set(ws)))}
    Wl = torch.tensor([wmap[int(w)] for w in ws]).to(dev)
    print(f"[data] {X.shape[0]} lines | points {X.shape[2]} | writers {len(wmap)}", flush=True)

    vae = InkVAE(len(wmap), cfg.n_classes).to(dev)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(vae.parameters(), lr=2e-3)
    t0 = time.time()
    for s in range(a.vae_steps):
        z, nll, lpen, kl = vae(X, XY)
        ls = F.cross_entropy(vae.style(z), Wl)
        lp = vae.ctc_logits(z)
        ilen = torch.full((X.shape[0],), MAXP, dtype=torch.long, device=dev)
        lctc = ctc(lp, T, ilen, TL)
        loss = nll + lpen + 1e-4 * kl + 0.1 * ls + 1.0 * lctc
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step()
        if s % 100 == 0:
            print(f"[vae {s}] nll {nll.item():.3f} pen {lpen.item():.3f} "
                  f"kl {kl.item():.3f} style {ls.item():.3f} ctc {lctc.item():.2f}", flush=True)

    with torch.no_grad():
        mu, _ = vae.enc(X)
        Z = mu.detach()
    txt_emb = torch.nn.Embedding(cfg.n_classes + 1, 128).to(dev)
    den = Denoiser().to(dev)
    opt = torch.optim.AdamW(list(den.parameters()) + list(txt_emb.parameters()), lr=2e-3)
    for s in range(a.dit_steps):
        # rebuilt every step: reusing a graph across backward passes frees its tensors
        ctx = txt_emb(T).mean(1, keepdim=True)
        t = torch.randint(0, TDIFF, (Z.shape[0],), device=dev)
        ab = cosine_alpha(t.float()).view(-1, 1, 1)
        eps = torch.randn_like(Z)
        zt = ab.sqrt() * Z + (1 - ab).sqrt() * eps
        pred = den(zt, t.float() / TDIFF, ctx)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 100 == 0:
            print(f"[dit {s}] eps-mse {loss.item():.4f}", flush=True)

    with torch.no_grad():
        ctx = txt_emb(T[0:1]).mean(1, keepdim=True)
        zs = ddim(den, ctx, (1, ZDIM, LAT), steps=20, dev=dev)
        _, pp = vae.dec(zs)
        k = 8
        mp = vae.dec(zs)[0].view(1, k, 6, MAXP).permute(0, 3, 1, 2)
        pi = torch.softmax(mp[..., 0], -1)
        mu1 = (mp[..., 1] * pi).sum(-1)[0].cpu().numpy()
        mu2 = (mp[..., 2] * pi).sum(-1)[0].cpu().numpy()
        gen = np.stack([mu1, mu2], 1)
    with torch.no_grad():
        mu0, _ = vae.enc(X[0:1])
        rp = vae.dec(mu0)[0].view(1, 8, 6, MAXP).permute(0, 3, 1, 2)
        rpi = torch.softmax(rp[..., 0], -1)
        rec = torch.stack([(rp[..., 1] * rpi).sum(-1)[0], (rp[..., 2] * rpi).sum(-1)[0]], 1)
        rec = rec.cpu().numpy()
    gt = XY[0].cpu().numpy()
    rows = [("GT line0", render(gt)), ("RECON (must match GT)", render(rec)),
            ("CENT v4 sample", render(gen))]
    H, Wd = 96, 1024
    out = Image.new("L", (Wd, len(rows) * (H + 20) + 20), 255); d = ImageDraw.Draw(out)
    for i, (lab, arr) in enumerate(rows):
        y = 20 + i * (H + 20); d.text((6, y - 14), lab, fill=0)
        out.paste(Image.fromarray((arr * 255).astype(np.uint8)), (0, y))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    print(json.dumps({"out": a.out, "vae_steps": a.vae_steps, "dit_steps": a.dit_steps,
                      "sec": round(time.time() - t0, 1), "device": dev}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
