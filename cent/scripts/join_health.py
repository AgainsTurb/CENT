"""Health check for the join label: is the 86% real, and are the joins real?

Three questions, each with its own evidence:
  1. Is the 86% uniform across writers (suspicious: parser artefact) or does it
     vary (expected: individual writing style)?
  2. Does the pen actually continue across a labelled join when you look at it?
  3. Do joins and lifts separate on gap, per writer?
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.pipeline.common import points_to_strokes  # noqa: E402
from cent.viz import render_strokes  # noqa: E402

TAU = 0.12


def main() -> int:
    d = Path("datasets/processed/train")
    lines = np.load(d / "lines.npz", allow_pickle=True)
    al = np.load(d / "align.npz", allow_pickle=True)
    pts = lines["points"].astype(np.float64)
    sid = lines["stroke_ids"]
    off = lines["offsets"]
    wr = lines["writers"]
    bp, bl = al["boundary_points"], al["boundary_line"]

    per = defaultdict(lambda: {"n": 0, "join": 0, "lift": 0, "same_far": 0})
    recs = []
    for L in range(len(off) - 1):
        lo, hi = off[L], off[L + 1]
        h = float(np.ptp(pts[lo:hi, 1])) or 1.0
        w = int(wr[L])
        for i in np.where(bl == L)[0]:
            p = int(bp[i])
            if p < 1 or lo + p >= hi:
                continue
            g = float(np.linalg.norm(pts[lo + p] - pts[lo + p - 1])) / h
            same = sid[lo + p] == sid[lo + p - 1]
            s = per[w]
            s["n"] += 1
            if same and g <= TAU:
                s["join"] += 1
            elif same:
                s["same_far"] += 1
            else:
                s["lift"] += 1
            recs.append((w, L, p, g, same, h))

    print(f"writers={len(per)}  boundaries={len(recs)}  tau={TAU}")
    print(f"{'writer':>6} {'n':>6} {'join':>7} {'lift':>7} {'same>tau':>10}")
    rates = []
    for w in sorted(per):
        s = per[w]
        r = s["join"] / max(1, s["n"])
        rates.append(r)
        print(f"{w:>6} {s['n']:>6} {s['join']:>7} {s['lift']:>7} {s['same_far']:>10}   rate={r:.3f}")
    rates = np.array(rates)
    print(f"\nper-writer join rate: min={rates.min():.3f} max={rates.max():.3f} "
          f"mean={rates.mean():.3f} sd={rates.std():.3f}")

    # ---- figure: per-writer rates + example joins
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.3, 1, 1, 1, 1], hspace=0.5)
    ax = fig.add_subplot(gs[0])
    ws = sorted(per)
    ax.bar([str(w) for w in ws], [per[w]["join"] / max(1, per[w]["n"]) for w in ws],
           color="#3f9d4f")
    ax.axhline(rates.mean(), color="k", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("join rate")
    ax.set_title(f"Join rate per writer (tau={TAU}) — expect a SPREAD, not four identical bars",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # pick real joins: same stroke, small gap, distinct writers
    cand = [r for r in recs if r[4] and r[3] <= 0.06]
    seen, picks = set(), []
    for r in cand:
        if r[0] in seen:
            continue
        seen.add(r[0])
        picks.append(r)
        if len(picks) == 4:
            break

    for row, (w, L, p, g, same, h) in enumerate(picks):
        ax2 = fig.add_subplot(gs[row + 1])
        lo, hi = off[L], off[L + 1]
        a, b = max(lo, lo + p - 150), min(hi, lo + p + 150)
        strokes = points_to_strokes(pts[a:b], sid[a:b])
        img = render_strokes(strokes, height=140, boundary_x=[float(pts[lo + p, 0])],
                             colour=(20, 20, 20))
        ax2.imshow(img)
        ax2.set_title(f"writer {w} · gap={g:.4f} · same stroke across the red line",
                      fontsize=9, loc="left")
        ax2.axis("off")

    out = Path("runs_join/join_health.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.1)
    print("figure ->", out)

    verdict = ("healthy: rates vary across writers"
               if rates.std() > 0.05 else
               "SUSPICIOUS: all writers nearly identical — check stroke parsing")
    print("verdict:", verdict, f"(sd={rates.std():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
