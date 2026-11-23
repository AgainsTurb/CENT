"""Sample lines from a trained CENT v2 (stroke-level) model and render them for inspection.

Everything measured so far used a one-step x0 estimate. This runs a real DDIM loop over the stroke
sequence, conditioned on the transcript and the writer's reference lines, and writes a contact sheet
so the output can be looked at.

Sequence length: the model generates a fixed-length stroke sequence, so N is taken from the
ground-truth line. That is a declared limitation -- the model does not yet predict how many strokes
a line needs -- and it must be stated wherever these images are shown.

Usage
    python -m benchmark.sample_strokes --config configs_stroke_r5.json \
        --ckpt runs/stroke_r5/ckpt_last.pt --split val --n 4 --out runs/stroke_r5/sheet.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cent.config import CENTConfig
from cent.data_strokes import StrokeLineDataset, collate_strokes
from cent.models.diffusion import Diffusion
from cent.models.line import LineStyleEncoder
from cent.models.line_stroke import StrokeDenoiser
from cent.models.stroke_graph import render_strokes_splat


@torch.no_grad()
def ddim_sample(den, diff, shape, mask, st, text, tmask, seed=0, steps=50, pos=None, scalars=None):
    """DDIM (eta=0) over the stroke payload sequence."""
    dev = mask.device
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(shape, device=dev, generator=g)
    ts = torch.linspace(diff.T - 1, 0, steps, device=dev).long()
    ab = diff.alpha_bar.to(dev)
    for i, t in enumerate(ts):
        tt = torch.full((shape[0],), int(t), device=dev, dtype=torch.long)
        eps = den(x, tt, mask, st, text, tmask, pos=pos, scalars=scalars)
        a_t = ab[tt].view(-1, 1, 1)
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-5)).clamp(-2, 2)
        if i + 1 < len(ts):
            t_next = ts[i + 1]
            a_n = ab[t_next].view(-1, 1, 1)
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps
        else:
            x = x0
    return x


def endpoint_gaps(x0, mask):
    p0, p3 = x0[..., 0:2], x0[..., 6:8]
    return (p3[:, :-1] - p0[:, 1:]).norm(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="sheet.png")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = CENTConfig.load(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    den = StrokeDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads, n_classes=cfg.n_classes,
                         k_refs=cfg.k_refs, max_strokes=cfg.max_tokens).to(dev).eval()
    sty = LineStyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types,
                           n_heads=cfg.n_heads).to(dev).eval()
    den.load_state_dict(ck["denoiser"]); sty.load_state_dict(ck["style"])
    diff = Diffusion(T=cfg.T).to(dev)

    ds = StrokeLineDataset(cfg.canonical_root, a.split, k_refs=cfg.k_refs, img_h=cfg.img_size,
                           canvas_w=cfg.canvas_width, max_strokes=cfg.max_tokens,
                           tau=cfg.join_tau, paired_only=True, limit=max(a.n, 2))
    batch = collate_strokes([ds[i] for i in range(min(a.n, len(ds)))])

    mask = batch["mask"].to(dev)
    text, tmask = batch["text"].to(dev), batch["text_mask"].to(dev)
    z = sty(batch["refs"].to(dev), batch["ref_mask"].to(dev))
    st = torch.cat([z[0].unsqueeze(1), z[1], z[2].unsqueeze(1)], 1)

    gt = batch["tokens"].to(dev)
    x0 = ddim_sample(den, diff, gt.shape, mask, st, text, tmask, seed=a.seed, steps=a.steps,
                     pos=batch["pos"].to(dev), scalars=batch["scalars"].to(dev))

    # render (rasteriser: 1 = ink; the corpus/display convention is 0 = ink)
    gen_img = 1.0 - render_strokes_splat(x0, mask, cfg.img_size, cfg.canvas_width)
    gt_render = 1.0 - render_strokes_splat(gt, mask, cfg.img_size, cfg.canvas_width)
    target = batch["target_img"][:, 0]

    gaps = endpoint_gaps(x0, mask)
    j = batch["join"][:, :-1].to(dev)
    valid = (mask[:, :-1] * mask[:, 1:]) > 0.5
    pred_rate = float((gaps[valid] < a.tau).float().mean()) if bool(valid.any()) else float("nan")
    gt_rate = float(j[valid].mean()) if bool(valid.any()) else float("nan")

    rows, meta = [], []
    for i in range(min(a.n, x0.shape[0])):
        n = int(mask[i].sum())
        rows.append((f"", [target[i].cpu().numpy(), gen_img[i, 0].cpu().numpy()]))
        rows.append((f"", [gt_render[i, 0].cpu().numpy()]))
        meta.append({"text": batch["texts"][i], "strokes": n,
                     "gt_join_rate": float(j[i, : max(n - 1, 0)].mean()) if n > 1 else None,
                     "pred_join_rate": float((gaps[i, : max(n - 1, 0)] < a.tau).float().mean()) if n > 1 else None})

    from benchmark.visualise import contact_sheet
    out = contact_sheet(rows, a.out, glyphs_path="")
    print(json.dumps({"sheet": out, "ddim_steps": a.steps, "tau": a.tau,
                      "pred_join_rate": round(pred_rate, 4), "gt_join_rate": round(gt_rate, 4),
                      "samples": meta}, indent=2, ensure_ascii=False))
    print("row order per sample: [real scan, GENERATED image] then [GT strokes rendered]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
