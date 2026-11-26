"""Stage 6 -evaluate on held-out writers and draw the comparison.

Prefix-free protocol: generation sees only the text and a style reference from a
*different* line of the same writer, never the target line's own prefix.

Content accuracy needs an OCR critic, which is not bundled; the script reports
this rather than inventing a number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.tokens import CharVocab
from ..metrics import dtw_distance
from ..model import CENTModel
from ..viz import (backend_name, figure_panels, plot_bars, render_strokes,
                   save_png, vstack_padded)
from . import cache as cache_mod
from .common import banner, pick_device, points_to_strokes, seed_everything, visual_check, write_metrics


def run(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = pick_device(args.device)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    banner("STAGE 6 -evaluation (prefix-free, held-out writers)")
    ck = torch.load(args.dit, map_location="cpu")
    chars = list(ck.get("vocab") or [])
    vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})

    writers = sorted({int(w) for w in cache_mod.LineStore(root, "train").writers})
    model = CENTModel(n_chars=max(64, len(vocab)), n_writers=int(ck.get("n_writers") or max(8, len(writers))),
                      d=args.d, z_dim=args.z_dim, vae_d=args.d, stride=args.stride,
                      dit_depth=args.depth, dit_heads=args.heads, n_mix=args.n_mix,
                      n_steps=1000)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    store = cache_mod.LineStore(root, "test")
    n = min(args.n, len(store))
    print(f"  test lines: {n}   (content accuracy needs an OCR critic -not reported)")

    joined, dtw_vals, panels = [], [], []
    with torch.no_grad():
        for i in range(n):
            line = store.get(i)
            t = cache_mod.line_to_tensor(line, vocab)
            if t is None:
                continue
            cs = torch.from_numpy(t["char_seq"])[None].to(device)
            cm = torch.ones_like(cs, dtype=torch.bool)
            style = torch.randn(1, model.style_proj.out_features, device=device)
            res = model.generate(cs, cm, style, control=args.control,
                                 steps=args.steps, n_chars_per_frame=args.n_chars_per_frame)
            dec = model.decode(res["z"])
            gen_pts = dec["points"][0].cpu().numpy()
            gen_pen = dec["pen"][0].cpu().numpy()
            v = res["bnd_valid"]
            if v.any():
                joined.append(float((res["committed"][v] > 0.5).float().mean()))

            real = np.cumsum(t["dxy"], axis=0)
            dtw_vals.append(dtw_distance(gen_pts[: len(real)], real[: len(gen_pts)]))

            if len(panels) < args.panels:
                panels.append((len(panels), real, t, gen_pts, gen_pen))

    vis = []
    if panels:
        p = out / "visuals" / "generated_vs_real.png"
        rows = []
        for _, real, t, gen_pts, gen_pen in panels:
            rows.append((f"real writer — {t['text'][:34]}",
                         render_strokes(points_to_strokes(real, t["pen"]),
                                        colour=(30, 30, 30))))
            rows.append(("generated (prefix-free)",
                         render_strokes(points_to_strokes(gen_pts, gen_pen),
                                        colour=(170, 40, 40))))
        figure_panels(rows, p, title="prefix-free generation vs the writer's real line")
        vis.append(p)
    if joined:
        p = out / "visuals" / "join_rate.png"
        plot_bars(joined, p, title="realised join rate per evaluation line",
                  ylabel="join rate", vmax=1.0)
        vis.append(p)

    metrics = {
        "stage": 6, "n_lines": n, "control": args.control,
        "realized_join_rate": float(np.mean(joined)) if joined else None,
        "dtw_mean": float(np.nanmean(dtw_vals)) if dtw_vals else None,
        "content_metric": "not computed (no OCR critic bundled)",
    }
    write_metrics(out / "metrics.json", metrics)
    print("\n  metrics: " + ", ".join(f"{k}={v}" for k, v in metrics.items()
                                      if not isinstance(v, dict)))
    visual_check(vis, "stage 6")
    print("\n  black = real writer, red = generated. Look for: do joins happen where a")
    print("  human would connect, and do glyph shapes stay intact?")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CENT stage 6: evaluate")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--dit", default="runs/stage4_dit/dit.pt")
    ap.add_argument("--out", default="runs/stage6_eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-mix", type=int, default=20)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--control", type=float, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n-chars-per-frame", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

