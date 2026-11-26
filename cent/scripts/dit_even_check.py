"""Local check with the FIXED inference contract.

Two defects were making the box samples scribbles:
  1. style = torch.randn()  -> out of distribution; must come from a real line
  2. even-split layout at inference vs aligned layout in training

This trains a small DiT with the EVEN layout (so train == inference) and samples
unseen text using a real style vector, then renders target vs prediction.
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
N_TRAIN = 8


def even_layout(n_chars: int, T_lat: int, device):
    """Character id per latent frame, evenly split. Same rule training AND sampling."""
    fc = torch.full((1, T_lat), -1, dtype=torch.long, device=device)
    fm = torch.zeros(1, T_lat, dtype=torch.bool, device=device)
    per = T_lat / max(1, n_chars)
    for f in range(T_lat):
        c = int(f / per)
        if c < n_chars:
            fc[0, f] = c
            fm[0, f] = True
    return fc, fm


def ddim(model, shape, fc, fm, ctx, style, steps=20):
    z = torch.randn(shape, device=next(model.parameters()).device)
    T = model.n_steps
    ts = torch.linspace(T - 1, 0, steps).long().clamp(0, T - 1)
    for i, t in enumerate(ts):
        tb = torch.full((shape[0],), int(t), dtype=torch.long, device=z.device)
        out = model.dit(z, tb, frame_char=fc, frame_mask=fm, char_ctx=ctx, style=style)
        a = model.alphas_cumprod[t]
        an = model.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=z.device)
        eps = (z - a.sqrt() * out["x0"]) / (1 - a).sqrt().clamp_min(1e-6)
        z = an.sqrt() * out["x0"] + (1 - an).sqrt() * eps
    return z


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load("runs_box/vae_ctc60k.pt", map_location="cpu", weights_only=False)
    chars = list(ck.get("vocab") or [])
    vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})
    store = cache_mod.LineStore(Path("datasets"), "train")

    model = CENTModel(n_chars=max(64, len(vocab)), n_writers=int(ck.get("n_writers") or 8),
                      d=D, z_dim=ZD, vae_d=128, stride=STRIDE, dit_depth=DEPTH,
                      dit_heads=HEADS, n_mix=20, n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)

    # collect usable lines
    items = []
    for i in range(len(store)):
        t = cache_mod.line_to_tensor(store.get(i), vocab)
        if t is None or len(t["char_seq"]) < 5 or len(t["dxy"]) > 700:
            continue
        items.append((i, t))
        if len(items) >= N_TRAIN + 1:
            break
    print(f"lines collected: {len(items)}  DiT params={sum(p.numel() for p in model.dit.parameters())/1e6:.2f} M")

    def prep(t):
        dxy = torch.from_numpy(t["dxy"])[None].to(dev)
        pen = torch.from_numpy(t["pen"])[None].to(dev)
        mask = torch.ones_like(pen, dtype=torch.bool)
        cs = torch.from_numpy(t["char_seq"])[None].to(dev)
        cm = torch.ones_like(cs, dtype=torch.bool)
        with torch.no_grad():
            x = torch.cat([dxy, torch.nn.functional.one_hot(pen, 4).float()], -1)
            mu, _, lm = model.vae.encode(x, mask)
            ctx = model.text(cs, cm)
            style = model.style_proj((mu * lm.unsqueeze(-1)).sum(1) / lm.sum(1).clamp_min(1).unsqueeze(-1))
        T_lat = mu.shape[1]
        fc, fm = even_layout(int(cs.shape[1]), T_lat, dev)
        return mu, fc, fm, ctx, style, dxy

    train = [prep(t) for _, t in items[:N_TRAIN]]
    hold_i, hold_t = items[N_TRAIN]
    hold = prep(hold_t)
    print(f"train lines {N_TRAIN}  |  held-out line {hold_i}: {hold_t['text'][:24]}")

    opt = torch.optim.AdamW(model.dit.parameters(), lr=2e-4)
    model.train()
    g = torch.Generator(device=dev).manual_seed(0)
    for step in range(1, STEPS + 1):
        mu, fc, fm, ctx, style, _ = train[step % len(train)]
        tb = torch.randint(0, model.n_steps, (1,), device=dev, generator=g)
        a = model.alphas_cumprod[tb].view(-1, 1, 1)
        z_t = a.sqrt() * mu + (1 - a).sqrt() * torch.randn_like(mu)
        out = model.dit(z_t, tb, frame_char=fc, frame_mask=fm, char_ctx=ctx, style=style)
        loss = (((out["x0"] - mu) ** 2) * fm.unsqueeze(-1)).sum() / (fm.sum() * mu.shape[-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dit.parameters(), 1.0)
        opt.step()
        if step % 300 == 0:
            print(f"  step {step:4d} loss={loss.item():.5f}")

    # ---- sample the HELD-OUT text, with a real style from a training line
    mu_h, fc_h, fm_h, ctx_h, style_h, dxy_h = hold
    with torch.no_grad():
        z = ddim(model, mu_h.shape, fc_h, fm_h, ctx_h, style_h)
        gmm, penl = model.vae.decode(z)
        pts = torch.cumsum(gmm_to_points(gmm), dim=1)[0].cpu().numpy()
        pen = penl[0, : len(pts)].argmax(-1).cpu().numpy()
        # reference: VAE round-trip of the same line
        gmm2, penl2 = model.vae.decode(mu_h)
        pts_r = torch.cumsum(gmm_to_points(gmm2), dim=1)[0].cpu().numpy()
        pen_r = penl2[0, : len(pts_r)].argmax(-1).cpu().numpy()

    truth = np.cumsum(dxy_h[0].cpu().numpy(), axis=0)
    n = min(len(pts), len(truth))
    epe = float(np.abs(pts[:n] - truth[:n]).mean())
    print(f"\nheld-out sample EPE vs truth: {epe:.3f}   (VAE round-trip is the floor)")

    figure_panels([
        (f"TARGET (held-out, unseen in training)", render_strokes(points_to_strokes(truth, hold_t["pen"]), height=120)),
        ("VAE round-trip (ceiling)", render_strokes(points_to_strokes(pts_r, pen_r), height=120, colour=(30, 90, 160))),
        (f"DiT sample, even layout + real style   EPE={epe:.2f}",
         render_strokes(points_to_strokes(pts, pen), height=120, colour=(40, 140, 70))),
    ], "runs_box/dit_even_fix.png",
        title=f"Fixed inference contract: even layout both sides, real style ({STEPS} steps, {N_TRAIN} lines)")
    print("figure -> runs_box/dit_even_fix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
