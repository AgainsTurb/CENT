"""Calibrate the stroke-level connection threshold on real data.

The join variable in CENT v2 is geometric:

    gap(i) = || end(stroke i) - start(stroke i+1) || / median_stroke_diagonal

`tau` decides how close two consecutive strokes must be to count as connected. It cannot be
guessed: the right value comes from the corpus's own gap distribution. This prints that
distribution and the join rate it implies at several thresholds, so the choice is recorded rather
than assumed -- and so a reader can see that the label is not a free parameter tuned on the result.

Usage
    python -m benchmark.calibrate_join --canonical data/canonical_lines/v1 --split train --n 800
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cent.data_strokes import StrokeLineDataset, join_gaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--paired-only", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ds = StrokeLineDataset(a.canonical, a.split, paired_only=a.paired_only, limit=0)
    n = min(a.n, len(ds))
    gaps = []
    strokes_per_line = []
    for k in range(n):
        i = ds.usable[k]
        st = ds.strokes_of(i)
        strokes_per_line.append(len(st))
        g = join_gaps(st)
        if len(g):
            gaps.append(g)
    g = np.concatenate(gaps) if gaps else np.zeros(1)

    pct = {f"p{q}": round(float(np.percentile(g, q)), 4) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    rates = {f"tau_{t:g}": round(float((g < t).mean()), 4) for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75, 1.0)}
    rep = {
        "lines_measured": n,
        "boundaries": int(len(g)),
        "gap_percentiles": pct,
        "join_rate_by_tau": rates,
        "strokes_per_line": {
            "mean": round(float(np.mean(strokes_per_line)), 2),
            "median": float(np.median(strokes_per_line)),
            "max": int(np.max(strokes_per_line)),
        },
        "note": ("gap is normalised by the median stroke bbox diagonal; a small gap means the pen "
                 "visually continues across the boundary"),
    }
    print(json.dumps(rep, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
