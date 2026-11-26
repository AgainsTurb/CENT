"""Measure the join (连笔) rate of a trajectory corpus.

This is the measurement the Chinese handwriting-generation literature has not
published: for every adjacent character pair, is the pen still down when the
next character begins?

Two labels are computed per boundary:
  * ``same_stroke``  — the two characters share a physical stroke (exact, but
    only detectable when per-point character ids survive the release)
  * ``gap``          — Euclidean distance between the last point of char i and
    the first point of char i+1, in line-height-normalised units. A join is
    ``gap <= tau``.

Outputs overall and per-writer rates, plus the gap distribution for lift vs
join boundaries, so the threshold can be chosen from data rather than assumed.

Usage
-----
    python scripts/measure_boundary_stats.py --fixture --tau 0.06
    python scripts/measure_boundary_stats.py --word-dir datasets/SCUT-COUCH2009/WORD8888
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.data.format_scut import (  # noqa: E402
    assign_strokes_to_chars,
    load_scutcouch_textline,
    load_scutcouch_word_dir,
)
from cent.data.synthetic import build_fixture_corpus  # noqa: E402
from cent.data.tokens import normalize_points  # noqa: E402


def per_boundary_records(sample, tau: float, height: float | None = None):
    char_of = sample.stroke_char
    fallback = char_of is None
    if char_of is None:
        char_of = assign_strokes_to_chars(sample, None)
    if char_of is None:
        return []
    pts = sample.points()
    if len(pts) == 0:
        return []

    sid = sample.stroke_ids()
    cid = np.asarray([char_of[s] for s in sid])
    norm = normalize_points(pts, height=height)

    recs = []
    for t in range(1, len(cid)):
        if cid[t] == cid[t - 1]:
            continue
        same = bool(sid[t] == sid[t - 1])
        gap = float(np.linalg.norm(norm[t] - norm[t - 1]))
        recs.append({
            "writer": sample.writer,
            "gap": gap,
            "same_stroke": same,
            "join": bool(same or gap <= tau),
            "fallback": fallback,
        })
    return recs


def report(recs, tau: float, name: str) -> None:
    if not recs:
        print(f"\n[{name}] no boundaries found")
        return
    joins = np.array([r["join"] for r in recs], dtype=float)
    gaps = np.array([r["gap"] for r in recs])
    same = np.array([r["same_stroke"] for r in recs], dtype=float)

    print(f"\n[{name}]  boundaries={len(recs)}  tau={tau}")
    fb = float(np.mean([r["fallback"] for r in recs]))
    if fb > 0.5:
        print(f"  !! WARNING: {fb*100:.0f}% of boundaries used the GEOMETRIC fallback "
              f"(no per-character stroke map in this release).")
        print(f"     Character assignment is inferred, so the join rate below is an "
              f"upper bound, not a measurement. Use a release that ships .SDX/.SGM "
              f"(SCUT-COUCH words) for a valid number.")
    print(f"  join rate            : {joins.mean():.3f}   "
          f"({int(joins.sum())} joins / {len(joins)})")
    print(f"  same-stroke fraction : {same.mean():.4f}")
    print(f"  gap (all)            : mean {gaps.mean():.3f}  "
          f"p10 {np.percentile(gaps, 10):.3f}  p50 {np.percentile(gaps, 50):.3f}  "
          f"p90 {np.percentile(gaps, 90):.3f}")
    if (joins > 0.5).any():
        print(f"  gap | join           : mean {gaps[joins > 0.5].mean():.3f}")
    if (joins < 0.5).any():
        print(f"  gap | lift           : mean {gaps[joins < 0.5].mean():.3f}")

    by_w: dict[str, list] = defaultdict(list)
    for r in recs:
        by_w[r["writer"]].append(r["join"])
    rates = sorted(((w, float(np.mean(v))) for w, v in by_w.items()),
                   key=lambda kv: kv[1], reverse=True)
    print(f"  per-writer join rate : {len(rates)} writers, "
          f"max {rates[0][1]:.2f} ({rates[0][0]}), "
          f"min {rates[-1][1]:.2f} ({rates[-1][0]}), "
          f"mean {np.mean([r for _, r in rates]):.3f}")
    print("  (a wide spread across writers is the signal a style-conditioned "
          "boundary prior should capture)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word-dir", default=None)
    ap.add_argument("--unipen-dir", default=None)
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--tau", type=float, default=0.06)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.fixture:
        man = build_fixture_corpus(ROOT / "datasets" / "_smoke", n_lines=10, n_words=14,
                                   seed=1, cursive_rate=0.4)
        args.unipen_dir, args.word_dir = man["unipen_dir"], man["word_dir"]

    if args.word_dir:
        samples = list(load_scutcouch_word_dir(args.word_dir))
        if args.limit:
            samples = samples[: args.limit]
        recs = [r for s in samples for r in per_boundary_records(s, args.tau)]
        report(recs, args.tau, "Word/char corpus")

    if args.unipen_dir:
        samples = list(load_scutcouch_textline(args.unipen_dir))
        if args.limit:
            samples = samples[: args.limit]
        recs = [r for s in samples for r in per_boundary_records(s, args.tau)]
        report(recs, args.tau, "Textline corpus")

    if not (args.word_dir or args.unipen_dir):
        print("Pass --word-dir / --unipen-dir / --fixture")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
