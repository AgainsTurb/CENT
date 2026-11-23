"""Rebuild the synthetic canonical-line fixture.

Creates .dgrl pages (images) and .wptt pages (trajectories) for 8 writers x 3 lines, with
lexicon-correct stroke counts plus one connector stroke per boundary, then runs the canonical
builder on them. Every line ends up PAIRED (image + trajectory), which is what the first
training campaign uses.

Usage: python tests/make_line_fixture.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from test_casia_lines import write_dgrl, write_wptt  # noqa: E402

from CENT.common.canonical_lines import build_lines  # noqa: E402
from CENT.common.lexicon import load_lexicon  # noqa: E402

TEXTS = ["人大人大", "山水人大", "大山大水"]


def main() -> int:
    lex = load_lexicon(str(ROOT / "data/lexicon/graphics.txt"))
    tmp = Path(tempfile.mkdtemp(prefix="cent_fixture_"))
    dg, wp = tmp / "dgrl", tmp / "wptt"
    dg.mkdir()
    wp.mkdir()

    bmp = np.full((28, 170), 255, np.uint8)
    bmp[8:22, 6:70] = 0
    for w in range(1, 9):
        write_dgrl(dg / f"{w:03d}-P04.dgrl", [(t, 3 + i * 4, 5, bmp) for i, t in enumerate(TEXTS)])
        pool, lines, cur = [], [], 0
        for t in TEXTS:
            idxs = []
            for ch in t:
                for k in range(lex[ch]):
                    pool.append(np.array([[cur * 2.0 + k * 1.5, k * 1.0],
                                          [cur * 2.0 + k * 1.5 + 2, k * 1.0 + 1.5]], np.float32))
                    idxs.append(len(pool) - 1)
                cur += 1
            for _ in range(len(t) - 1):          # one connector stroke per boundary
                pool.append(np.array([[cur * 2.0, 20.0], [cur * 2.0 + 1.0, 21.0]], np.float32))
                idxs.append(len(pool) - 1)
            lines.append((t, idxs))
        write_wptt(wp / f"{w:03d}-P04.wptt", pool, lines)

    meta = build_lines(dg, wp, str(ROOT / ".fixture/canonical_lines"), lexicon=lex)
    print("counts", meta["counts"], "paired", meta["paired_lines"],
          "with_strokes", meta["lines_with_strokes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
