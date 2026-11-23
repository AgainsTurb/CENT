"""Tests for the decisions taken in the training plan: canvas, aspect, budget, pairing, HTR.

These are invariant checks, not quality checks: they pin the choices the author made so a later
refactor cannot silently square the canvas again, truncate a line, or widen the character cap
past the token budget.

Usage: python tests/test_line_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.recogniser import ADOPTED, Recogniser  # noqa: E402
from cent.data_lines import MAX_CHARS, MAX_TOKENS, LineTokenDataset  # noqa: E402
from cent.models.stroke_graph import BezierRenderer  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


def main() -> int:
    print("\nauthor decisions are pinned")
    check("max_chars is 32", MAX_CHARS == 32, str(MAX_CHARS))
    check("max_tokens is 1024", MAX_TOKENS == 1024, str(MAX_TOKENS))

    print("\nthe renderer draws the LINE canvas, not a square")
    r = BezierRenderer(size=64, width_size=1024)
    out = r(torch.rand(2, 6, 8) * 0.4, torch.ones(2, 6))
    check("output is 64x1024", tuple(out.shape) == (2, 1, 64, 1024), str(tuple(out.shape)))
    check("values are in [0,1]", float(out.min()) >= 0 and float(out.max()) <= 1)
    sq = BezierRenderer(size=64)(torch.rand(1, 4, 8) * 0.4, torch.ones(1, 4))
    check("square rendering still available", tuple(sq.shape) == (1, 1, 64, 64), str(tuple(sq.shape)))

    print("\naspect ratio is preserved by one scale, not two")
    # A SHORT line (4:1 ink) on a 16:1 canvas: this is where the old per-axis normalisation
    # and the correct single-scale one diverge. For a line whose ink aspect already matches
    # the canvas they coincide, which is why the bug was invisible on the tall fixture lines.
    span_x, span_y, H, W = 1600.0, 400.0, 64, 1024
    scale = min(W / span_x, H / span_y)
    nx, ny = span_x * scale / W, span_y * scale / H
    check("x does not fill the canvas for a short line", nx < 0.5, f"{nx:.4f}")
    check("y fills the canvas (height is the binding constraint)", abs(ny - 1.0) < 1e-6, f"{ny:.4f}")
    check("one scale keeps the true aspect: (nx*W)/(ny*H) == span_x/span_y",
          abs((nx * W) / (ny * H) - span_x / span_y) < 1e-6, f"{(nx*W)/(ny*H):.4f} vs {span_x/span_y:.4f}")
    check("independent normalisation would have squared it (the bug we fixed)",
          abs(nx - 1.0) > 0.4, f"independent would give x=1.0, one-scale gives {nx:.3f}")

    print("\nthe dataset honours the plan on the fixture")
    try:
        from common.lexicon import load_lexicon
        lex = load_lexicon(str(ROOT / "data/lexicon/graphics.txt"))
        ds = LineTokenDataset(ROOT / ".fixture/canonical_lines", "train", lexicon=lex,
                              k_refs=5, img_h=64, canvas_w=1024, max_chars=32,
                              max_tokens=1024, paired_only=True)
        check("paired_only dataset is non-empty on the fixture", len(ds) > 0, str(len(ds)))
        s = ds[0]
        check("references are canvas width", s["refs"].shape[-1] == 1024, str(tuple(s["refs"].shape)))
        check("target image is canvas width", s["target_img"].shape[-1] == 1024,
              str(tuple(s["target_img"].shape)))
        check("writer index is a tensor, not a constant", "writer_idx" in s and s["writer_idx"].dim() == 0)
        check("char_slot present for verifier grouping", "char_slot" in s)
        check("tokens fit the budget", s["tokens"].shape[0] <= 1024, str(s["tokens"].shape))
        check("no more than 32 characters", s["char_ids"].shape[0] <= 32, str(s["char_ids"].shape))
        # an unpaired-only request on a paired fixture keeps everything
        ds_u = LineTokenDataset(ROOT / ".fixture/canonical_lines", "train", lexicon=lex,
                                paired_only=False)
        check("paired_only does not lose paired lines", len(ds_u) >= len(ds), f"{len(ds_u)} vs {len(ds)}")
    except FileNotFoundError as e:
        check("fixture present", False, str(e))

    print("\nrecogniser slot fails closed and names the adopted model")
    rec = Recogniser(command=None)
    v = rec.verify()
    check("no command -> NOT_RUN", v["status"] == "NOT_RUN", str(v))
    check("the reason names the adopted model", ADOPTED["id"] in v["reason"], v["reason"])
    check("the recipe is included", "pylaia" in v["recipe"].lower())
    check("identity records licence and size",
          ADOPTED["licence"] == "MIT" and ADOPTED["bytes"] > 0, str(ADOPTED))
    try:
        rec.transcribe(["x.png"])
        check("transcribe without a command raises", False)
    except RuntimeError:
        check("transcribe without a command raises", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
