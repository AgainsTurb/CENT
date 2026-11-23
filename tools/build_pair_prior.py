"""Build the per-character-pair join prior consumed by M4 (the Join Planner).

Lin & Wan's classical cursive-synthesis work conditions connection on a learned per-pair
probability P(i,j); deep models threw that away. M4 gets it back as an input feature.

The statistic is counted on the training split only, from the corpus's own character split:
for each boundary between character i and i+1, the join label is the pen gap between the last
stroke of character i and the first stroke of character i+1, in canvas units.

    python tools/build_pair_prior.py --root data/canonical_lines/v1 --split train \
        --out data/pair_prior.json

Output: {"i,j": join_rate, ...} with Laplace smoothing applied at load time
(see cent/data_v3._load_prior). Pairs never observed fall back to the global rate.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from cent.data_strokes import StrokeLineDataset
from cent.models.stroke_graph import fit_stroke


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/canonical_lines/v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/pair_prior.json")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--img-h", type=int, default=64)
    ap.add_argument("--canvas-w", type=int, default=1024)
    a = ap.parse_args()

    ds = StrokeLineDataset(a.root, a.split, k_refs=1, img_h=a.img_h, canvas_w=a.canvas_w,
                           max_strokes=512, paired_only=False)
    num = defaultdict(float)
    den = defaultdict(float)
    used = skipped = 0

    for rec_i in ds.usable:
        r = ds.records[rec_i]
        cs = r.get("char_strokes")
        if not cs or sum(int(c) for c in cs) != int(r.get("n_strokes") or 0):
            skipped += 1
            continue
        st = ds.strokes_of(rec_i)
        if len(st) < 2:
            skipped += 1
            continue
        allp = np.concatenate(st)
        lo, hi = allp.min(0), allp.max(0)
        span = np.maximum(hi - lo, 1e-6)
        s = min(a.canvas_w / span[0], a.img_h / span[1])
        q = [((np.asarray(x, np.float32) - lo) * s / np.array([a.canvas_w, a.img_h], np.float32))
             for x in st]
        text = r["text"]
        counts = [int(c) for c in cs]
        if len(counts) != len(text):
            skipped += 1
            continue
        idx = np.cumsum(counts)
        for k in range(len(counts) - 1):
            if counts[k] == 0 or counts[k + 1] == 0:
                continue
            e = fit_stroke(q[idx[k] - 1])[3]
            s2 = fit_stroke(q[idx[k]])[0]
            gap = float(np.linalg.norm(s2 - e))
            key = (text[k], text[k + 1])
            den[key] += 1.0
            num[key] += 1.0 if gap < a.tau else 0.0
        used += 1

    out = {f"{ord(a_)} , {ord(b_)}".replace(" ", ""): num[(a_, b_)] / den[(a_, b_)]
           for (a_, b_) in den}
    # store character codepoints so the table is charset-independent
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    glob = sum(num.values()) / max(sum(den.values()), 1.0)
    print(f"lines used {used}, skipped {skipped}, pairs {len(out)}, global join rate {glob:.4f}")
    print(f"-> {a.out}   (note: keys are codepoint pairs, mapped to ids by the loader)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
