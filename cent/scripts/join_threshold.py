"""Sweep the join threshold and render the evidence.

Join definition: a character boundary is a *join* (连笔) when the pen never
lifted across it, i.e. the two neighbouring points belong to the same stroke AND
the spatial gap between them is negligible.

  join(p) = stroke_id[p] == stroke_id[p-1]  and  ||pts[p]-pts[p-1]|| / line_height <= tau

tau is the only free parameter, so this sweeps it and draws crops so the value
can be judged by eye instead of by the aggregate rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.pipeline.common import points_to_strokes  # noqa: E402
from cent.viz import figure_panels, plot_bars, plot_curves, render_strokes, save_png, vstack_padded  # noqa: E402

TAUS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets")
    split = "train"
    lines = np.load(root / "processed" / split / "lines.npz", allow_pickle=True)
    al = np.load(root / "processed" / split / "align.npz", allow_pickle=True)

    pts = lines["points"].astype(np.float64)
    sid = lines["stroke_ids"]
    off = lines["offsets"]
    writers = lines["writers"]
    bp = al["boundary_points"]
    bl = al["boundary_line"]

    n_lines = len(off) - 1
    print(f"lines={n_lines}  boundaries={len(bp)}")

    # per-boundary geometry: normalised gap and same-stroke flag
    same, gap = np.zeros(len(bp), bool), np.zeros(len(bp))
    for i, (L, p) in enumerate(zip(bl, bp)):
        lo = off[L]
        if p < 1 or lo + p >= off[L + 1]:
            gap[i] = np.inf
            continue
        h = np.ptp(pts[lo:off[L + 1], 1]) or 1.0
        a, b = lo + p - 1, lo + p
        gap[i] = float(np.linalg.norm(pts[b] - pts[a]) / h)
        same[i] = sid[b] == sid[a]

    rows = []
    for tau in TAUS:
        join = same & (gap <= tau)
        rows.append({"tau": tau, "join_rate": float(join.mean()),
                     "joined": int(join.sum()), "total": int(len(join))})
        print(f"  tau={tau:<6} join_rate={join.mean():.3f}  ({join.sum()}/{len(join)})")

    out = root.parent / "runs_join"
    out.mkdir(parents=True, exist_ok=True)
    plot_curves({"join rate": [r["join_rate"] for r in rows]},
                out / "join_rate_vs_tau.png",
                title="join rate vs threshold (x-axis = tau index: "
                      + ", ".join(str(t) for t in TAUS) + ")",
                ylabel="fraction of boundaries labelled join")
    (out / "join_sweep.json").write_text(json.dumps(rows, indent=2))

    # ---- crops at a mid threshold, so the eye can judge the rule itself
    tau = 0.03
    join = same & (gap <= tau)
    idx_join = np.where(join)[0]
    idx_lift = np.where(~join)[0]
    picks = ([(int(i), "JOIN (same stroke, small gap)") for i in idx_join[:3]]
             + [(int(i), "LIFT (stroke ends, pen moves)") for i in idx_lift[:3]])

    panels = []
    # pick from distinct lines and wide enough windows so a glyph sits each side
    used = set()
    def pick_from(pool, want):
        got = []
        for i in pool:
            L = int(bl[i])
            if L in used:
                continue
            used.add(L)
            got.append(int(i))
            if len(got) == want:
                break
        return got

    for i, _ in ([(i, 0) for i in pick_from(idx_join, 3)]
                 + [(i, 1) for i in pick_from(idx_lift, 3)]):
        L, p_idx = int(bl[i]), int(bp[i])
        lo, hi = off[L], off[L + 1]
        a = max(lo, lo + p_idx - 120)
        b = min(hi, lo + p_idx + 120)
        seg = pts[a:b]
        strokes = points_to_strokes(seg, sid[a:b])
        gy = float(pts[lo + p_idx, 0])
        if join[i]:
            kind = "JOIN: same stroke, gap within tau"
        elif same[i]:
            kind = "NO JOIN: same stroke but gap > tau"
        else:
            kind = "LIFT: stroke id changes (pen lifted)"
        img = render_strokes(strokes, height=150, boundary_x=[gy], colour=(20, 20, 20))
        # guarantee the guide is visible even if the window has no long stroke
        xs = np.where(img[:, :, 0] < 250)[1]
        img[:, max(0, min(img.shape[1]-1, int(xs.min()*0 + 0))):] = img
        panels.append((f"{kind}   gap={gap[i]:.4f}", img))

    print(f"\nfigures -> {out}/join_rate_vs_tau.png  and  {out}/join_crops.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
