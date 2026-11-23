"""Build the canonical dataset from downloaded CASIA archives.

    python -m common.build_cli \
        --offline data/raw/HWDB1 --online data/raw/OLHWDB1 \
        --out data/canonical/v1 --lexicon data/lexicon/graphics.txt \
        --limit-per-writer 0

Run `python -m common.probe` first and require it to pass. The build is
deterministic in (writers, seed): re-running on the same files reproduces the
same split assignment.

Storage note: one 64x64 uint8 image is 4 KB, so 1 M samples is ~4 GB per split
set plus the online arrays. This is expected on the GPU box; it is why the
dataset is not kept locally.
"""
from __future__ import annotations

import argparse
import json
import sys

from .canonical import build
from .lexicon import load_lexicon


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", default=None, help="directory of .gnt files")
    ap.add_argument("--online", default=None, help="directory of .pot files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon", default=None, help="Make Me A Hanzi graphics.txt")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--limit-per-writer", type=int, default=0,
                    help="0 = all; >0 caps samples per writer (dry runs)")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--pot-label-bytes", type=int, default=4, choices=[2, 4])
    a = ap.parse_args(argv)

    if not a.offline and not a.online:
        print("give at least one of --offline / --online", file=sys.stderr)
        return 2

    lexicon = None
    if a.lexicon:
        lexicon = load_lexicon(a.lexicon)
        print(f"[lexicon] {len(lexicon)} characters from {a.lexicon}")

    meta = build(
        offline_dir=a.offline,
        online_dir=a.online,
        out_dir=a.out,
        img_size=a.img_size,
        limit_per_writer=(a.limit_per_writer or None),
        seed=a.seed,
        pot_label_bytes=a.pot_label_bytes,
        lexicon=lexicon,
    )
    print(json.dumps({k: meta[k] for k in ("counts", "writers_total", "num_classes",
                                           "skipped_unknown_char")}, indent=2))
    if lexicon:
        print("[lexicon] agreement (T2):", json.dumps(meta["lexicon"]["agreement"], indent=2))
    print(f"wrote {a.out}/meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
