"""One figure that shows (a) the gap statistic and (b) what a join actually looks like.

Top panel: distribution of point-to-point gaps, in three groups
  - WITHIN stroke   : normal sampling step inside a stroke (the reference scale)
  - AT boundary, same stroke   : candidates for a join
  - AT boundary, stroke changes: candidates for a pen lift
Bottom panels: two real boundary crops, one of each kind, with the boundary marked.

If joins are a real, separable phenomenon, the "same stroke" group should look
like the within-stroke group (a spike near 0) and the "stroke changes" group
should sit far to the right, with a visible empty gap between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.pipeline.common import points_to_strokes  # noqa: E402
from cent.viz import render_strokes  # noqa: E402


def main() -> int:
    d = Path("datasets/processed/train")
    lines = np.load(d / "lines.npz", allow_pickle=True)
    al = np.load(d / "align.npz", allow_pickle=True)
    pts = lines["points"].astype(np.float64)
    sid = lines["stroke_ids"]
    off = lines["offsets"]
    bp, bl = al["boundary_points"], al["boundary_line"]

    within, same_gap, change_gap = [], [], []
    for L in range(len(off) - 1):
        lo, hi = off[L], off[L + 1]
        seg, ss = pts[lo:hi], sid[lo:hi]
        h = float(np.ptp(seg[:, 1])) or 1.0
        step = np.linalg.norm(np.diff(seg, axis=0), axis=1) / h
        cont = (ss[1:] == ss[:-1])
        within.extend(step[cont].tolist())

        m = np.where(bl == L)[0]
        for i in m:
            p = int(bp[i])
            if p < 1 or lo + p >= hi:
                continue
            g = float(np.linalg.norm(pts[lo + p] - pts[lo + p - 1])) / h
            (same_gap if sid[lo + p] == sid[lo + p - 1] else change_gap).append(g)

    within, same_gap, change_gap = map(np.asarray, (within, same_gap, change_gap))
    print(f"within-stroke steps : n={len(within)}  median={np.median(within):.4f}  p99={np.percentile(within,99):.4f}")
    print(f"boundary same stroke: n={len(same_gap)} median={np.median(same_gap) if len(same_gap) else float('nan'):.4f}")
    print(f"boundary stroke chg : n={len(change_gap)} median={np.median(change_gap):.4f}")

    # ---------------- figure
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.45)

    ax = fig.add_subplot(gs[0])
    bins = np.linspace(0, 0.16, 60)
    ax.hist(within, bins=bins, label=f"within a stroke (n={len(within)})",
            color="#4a7fb5", alpha=0.85)
    ax.hist(same_gap, bins=bins, label=f"boundary, stroke continues (n={len(same_gap)})",
            color="#3f9d4f", alpha=0.75)
    ax.hist(change_gap, bins=bins, label=f"boundary, stroke changes = LIFT (n={len(change_gap)})",
            color="#c0392b", alpha=0.75)
    for t in (0.02, 0.03, 0.05):
        ax.axvline(t, color="k", linestyle="--", linewidth=0.8)
        ax.text(t, ax.get_ylim()[1] * 0.92, f"tau={t}", fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_xlabel("gap between consecutive points, divided by line height")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Where do the gaps live?  A separable label needs TWO clusters", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # two example crops
    join_i = int(np.argmax([gap for gap in same_gap])) if len(same_gap) else 0
    # pick a boundary in the same-stroke group with a near-zero gap, and one that changes stroke
    idx_same = [i for i in range(len(bp)) if int(bl[i]) < len(off) - 1
                and bp[i] >= 1 and sid[off[bl[i]] + bp[i]] == sid[off[bl[i]] + bp[i] - 1]]
    idx_chg = [i for i in range(len(bp)) if int(bl[i]) < len(off) - 1
               and bp[i] >= 1 and sid[off[bl[i]] + bp[i]] != sid[off[bl[i]] + bp[i] - 1]]

    for row, (idx, cap) in enumerate([
            (idx_same[:1], "JOIN candidate: same stroke across the boundary (pen never lifted)"),
            (idx_chg[:1], "LIFT candidate: the stroke id changes at the boundary (pen lifted)")]):
        ax2 = fig.add_subplot(gs[row + 1])
        if not idx:
            ax2.axis("off")
            continue
        i = idx[0]
        L, p = int(bl[i]), int(bp[i])
        lo, hi = off[L], off[L + 1]
        a, b = max(lo, lo + p - 130), min(hi, lo + p + 130)
        strokes = points_to_strokes(pts[a:b], sid[a:b])
        img = render_strokes(strokes, height=150, boundary_x=[float(pts[lo + p, 0])],
                             colour=(20, 20, 20))
        ax2.imshow(img)
        ax2.set_title(cap, fontsize=9, loc="left")
        ax2.axis("off")

    out = Path("runs_join/join_visual_test.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.1)
    print("figure ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
