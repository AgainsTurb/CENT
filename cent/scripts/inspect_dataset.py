"""Inspect a SCUT-COUCH (or fixture) release: sizes, writers, vocabulary, shapes.

Usage
-----
    python scripts/inspect_dataset.py --word-dir datasets/SCUT-COUCH2009/WORD8888
    python scripts/inspect_dataset.py --unipen-dir datasets/SCUT-COUCH2009/Textline_NU
    python scripts/inspect_dataset.py --fixture            # synthetic smoke corpus
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.data.format_scut import (  # noqa: E402
    load_scutcouch_textline,
    load_scutcouch_word_dir,
)
from cent.data.synthetic import build_fixture_corpus  # noqa: E402


def describe(samples, name: str) -> None:
    if not samples:
        print(f"\n[{name}] no samples found")
        return
    n_chars = np.array([len(s.text) for s in samples])
    n_strk = np.array([s.n_strokes for s in samples])
    n_pts = np.array([s.n_points for s in samples])
    writers = Counter(s.writer for s in samples)
    chars = Counter(ch for s in samples for ch in s.text)

    print(f"\n[{name}]  {len(samples)} samples")
    print(f"  writers          : {len(writers)}  "
          f"(top: {', '.join(f'{w}:{c}' for w, c in writers.most_common(3))})")
    print(f"  chars/sample     : mean {n_chars.mean():.1f}  min {n_chars.min()}  max {n_chars.max()}")
    print(f"  strokes/sample   : mean {n_strk.mean():.1f}  min {n_strk.min()}  max {n_strk.max()}")
    print(f"  points/sample    : mean {n_pts.mean():.0f}  min {n_pts.min()}  max {n_pts.max()}")
    print(f"  unique characters: {len(chars)}")
    print(f"  sample transcripts:")
    for s in samples[:3]:
        print(f"    [{s.writer}] {s.text!r}  strokes={s.n_strokes} points={s.n_points}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word-dir", default=None)
    ap.add_argument("--unipen-dir", default=None)
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.fixture:
        man = build_fixture_corpus(ROOT / "datasets" / "_smoke", n_lines=10, n_words=14,
                                   seed=1, cursive_rate=0.4)
        args.unipen_dir = man["unipen_dir"]
        args.word_dir = man["word_dir"]

    if args.unipen_dir:
        s = list(load_scutcouch_textline(args.unipen_dir))
        describe(s[: args.limit] if args.limit else s, "Textline (Unipen)")
    if args.word_dir:
        s = list(load_scutcouch_word_dir(args.word_dir))
        describe(s[: args.limit] if args.limit else s, "Word/char (binary .IDX/.DAT)")

    if not (args.unipen_dir or args.word_dir):
        print("Nothing to inspect: pass --word-dir / --unipen-dir / --fixture")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
