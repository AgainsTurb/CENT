"""Tests 0-2 for the BoundaryDiT: does the generative path work at all?

0. VAE round-trip        : encode a real line, decode it (no DiT). Reference.
1. DiT at t~0            : feed the real latent at low noise, predict x0, decode.
2. Overfit one line      : train on that single line, then sample from noise.
   Also: sample with a REAL style and the ALIGNED layout, which is what the
   current generate() does NOT do (it uses a random style and an even split).

Every stage decodes through the frozen VAE and renders the ink, so the failure
is visible rather than inferred from a loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.data.tokens import CharVocab  # noqa: E402
from cent.model import CENTModel  # noqa: E402
from cent.pipeline import cache as cache_mod  # noqa: E402
from cent.pipeline.common import gmm_to_points, points_to_strokes  # noqa: E402
from cent.viz import figure_panels, render_strokes  # noqa: E402

D, DEPTH, HEADS, ZD, STRIDE = 256, 6, 8, 64, 8
STEPS = 1500


def layout_from_char_ids(cid: torch.Tensor, T_lat: int, stride: int):
    """frame_char: the character occupying each latent frame (from alignment)."""
    B, T = cid.shape
    fc = torch.full((B, T_lat), -1, dtype=torch.long, device=cid.device)
    fm = torch.zeros(B, T_lat, dtype=torch.bool, device=cid.device)
    for f in range(T_lat):
        lo, hi = f * stride, min(T, (f + 1) * stride)
        if lo >= hi:
            break
        chunk = cid[:, lo:hi]
        vals, couts = torch.mode(chunk, dim=1)
        fc[:, f] = vals
        fm[:, f] = (vid := (chunk >= 0).any(1))
    return fc, fm


@torch.no_grad()
def ddim_sample(model, z_shape, fchar, fmask, ctx, style, control=0.5, steps=20):
    z = torch.randn(z_shape, device=next(model.parameters()).device)
    T = model.n_steps
    ts = torch.linspace(T - 1, 0, steps).long().clamp(0, T - 1)
    for i, t in enumerate(ts):
        tb = torch.full((z.shape[0],), int(t), dtype=torch.long, device=z.device)
        out = model.dit(z, tb, frame_char=fchar, frame_mask=fmask,
                        char_ctx=ctx, style=style, bnd_commit=None)
        x0 = out["x0"]
        a = model.alphas_cumprod[t]
        a_next = model.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=z.device)
        eps = (z - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-6)
        z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
    return z


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load("runs_box/vae_ctc60k.pt", map_location="cpu", weights_only=False)
    chars = list(ck.get("vocab") or [])
    vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})

    store = cache_mod.LineStore(Path("datasets"), "train")
    al = np.load("datasets/processed/train/align.npz", allow_pickle=True)

    model = CENTModel(n_chars=max(64, len(vocab)), n_writers=int(ck.get("n_writers") or 8),
                      d=D, z_dim=ZD, vae_d=128, stride=STRIDE, dit_depth=DEPTH,
                      dit_heads=HEADS, n_mix=20, n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)
    print(f"device={dev}  DiT params={sum(p.numel() for p in model.dit.parameters())/1e6:.2f} M")

    # ---- pick one real line
    idx = None
    for i in range(len(store)):
        lo, hi = int(store.offsets[i]), int(store.offsets[i + 1])
        nch = int(al["char_ids"][lo:hi].max()) + 1 if lo < hi else 0
        if 300 < hi - lo < 900 and 8 < nch < 26 and (al["char_ids"][lo:hi] >= 0).mean() > 0.95:
            idx = i
            break
    lo, hi = int(store.offsets[idx]), int(store.offsets[idx + 1])
    t = cache_mod.line_to_tensor(store.get(idx), vocab)
    dxy = torch.from_numpy(t["dxy"])[None].to(dev)
    pen = torch.from_numpy(t["pen"])[None].to(dev)
    cid = torch.from_numpy(al["char_ids"][lo:hi].astype(np.int64))[None].to(dev)
    mask = torch.ones_like(pen, dtype=torch.bool)
    cs = torch.from_numpy(t["char_seq"])[None].to(dev)
    cm = torch.ones_like(cs, dtype=torch.bool)

    print(f"line {idx}: T={dxy.shape[1]} chars={cs.shape[1]}  text={t['text'][:30]}")

    with torch.no_grad():
        x = torch.cat([dxy, torch.nn.functional.one_hot(pen, 4).float()], -1)
        mu, _, lm = model.vae.encode(x, mask)
        z0 = mu
        ctx = model.text(cs, cm)
        style = model.style_proj((z0 * lm.unsqueeze(-1)).sum(1) / lm.sum(1).clamp_min(1).unsqueeze(-1))
        T_lat = z0.shape[1]
        fchar, fmask = layout_from_char_ids(cid, T_lat, STRIDE)

        def decode_to_points(z):
            gmm, penl = model.vae.decode(z)
            pts = torch.cumsum(gmm_to_points(gmm), dim=1)[0].cpu().numpy()
            pen_pred = penl[0, : len(pts)].argmax(-1).cpu().numpy()
            return pts, pen_pred

        # ---- test 1: VAE round-trip (reference)
        pts_vae, pen_vae = decode_to_points(z0)

        # ---- test 2: DiT at low noise (x0 at t=0 is what it should learn)
        t0 = torch.zeros(1, dtype=torch.long, device=dev)
        out0 = model.dit(z0, t0, frame_char=fchar, frame_mask=fmask, char_ctx=ctx, style=style)
        pts_dit0, pen_dit0 = decode_to_points(out0["x0"])

        # ---- test 3: sample from noise BEFORE training (expect garbage)
        z_pre = ddim_sample(model, z0.shape, fchar, fmask, ctx, style)
        pts_pre, pen_pre = decode_to_points(z_pre)

    def epe(a, b):
        n = min(len(a), len(b))
        return float(np.abs(a[:n] - b[:n]).mean())

    truth = np.cumsum(dxy[0].cpu().numpy(), axis=0)
    print(f"\nEPE vs truth  VAE round-trip: {epe(pts_vae, truth):.3f}")
    print(f"EPE vs truth  DiT@t=0 (untrained): {epe(pts_dit0, truth):.3f}")
    print(f"EPE vs truth  DDIM sample (untrained): {epe(pts_pre, truth):.3f}")

    # ---- overfit the single line
    opt = torch.optim.AdamW(model.dit.parameters(), lr=2e-4)
    model.train()
    g = torch.Generator(device=dev).manual_seed(0)
    for step in range(1, STEPS + 1):
        tb = torch.randint(0, model.n_steps, (1,), device=dev, generator=g)
        a = model.alphas_cumprod[tb].view(-1, 1, 1)
        z_t = a.sqrt() * z0 + (1 - a).sqrt() * torch.randn_like(z0)
        out = model.dit(z_t, tb, frame_char=fchar, frame_mask=fmask, char_ctx=ctx, style=style)
        loss = (((out["x0"] - z0) ** 2) * fmask.unsqueeze(-1)).sum() / (fmask.sum() * z0.shape[-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dit.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            print(f"  overfit step {step:4d}  loss={loss.item():.5f}")

    with torch.no_grad():
        out1 = model.dit(z0, t0, frame_char=fchar, frame_mask=fmask, char_ctx=ctx, style=style)
        pts_dit1, pen_dit1 = decode_to_points(out1["x0"])
        z_post = ddim_sample(model, z0.shape, fchar, fmask, ctx, style)
        pts_post, pen_post = decode_to_points(z_post)

    print(f"\nEPE vs truth  DiT@t=0 (after overfit): {epe(pts_dit1, truth):.3f}")
    print(f"EPE vs truth  DDIM sample (after overfit): {epe(pts_post, truth):.3f}")

    panels = [
        ("TARGET (real line)", render_strokes(points_to_strokes(truth, t["pen"]), height=120)),
        (f"VAE round-trip  EPE={epe(pts_vae, truth):.2f}",
         render_strokes(points_to_strokes(pts_vae, pen_vae), height=120, colour=(30, 90, 160))),
        (f"DiT@t=0 before overfit  EPE={epe(pts_dit0, truth):.2f}",
         render_strokes(points_to_strokes(pts_dit0, pen_dit0), height=120, colour=(180, 60, 60))),
        (f"DiT@t=0 after overfit  EPE={epe(pts_dit1, truth):.2f}",
         render_strokes(points_to_strokes(pts_dit1, pen_dit1), height=120, colour=(40, 140, 70))),
        (f"DDIM sample after overfit  EPE={epe(pts_post, truth):.2f}",
         render_strokes(points_to_strokes(pts_post, pen_post), height=120, colour=(120, 60, 160))),
    ]
    figure_panels(panels, "runs_box/dit_tests_0_2.png",
                  title="BoundaryDiT tests 0-2: can it paint a latent it has seen?")
    print("figure -> runs_box/dit_tests_0_2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
