"""Probe real CASIA files before anything else runs.

    python -m common.probe gnt data/raw/Online/../Gnt1.0Train/001.gnt
    python -m common.probe pot data/raw/Pot1.0Train/001.pot --limit 20
    python -m common.probe dir data/raw/Pot1.0Train --kind pot

A probe that reports `ok: false` means the documented layout does not match the
file, and the build must not run: a parser that mis-frames a record produces
plausible-looking glyphs that are silently wrong. This is the first command in
the runbook for that reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .casia import probe_gnt, probe_pot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["gnt", "pot", "dir"])
    ap.add_argument("path")
    ap.add_argument("--kind", choices=["gnt", "pot"], default="pot", help="for mode=dir")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args(argv)

    files: list[Path] = []
    if a.mode == "dir":
        files = sorted(Path(a.path).glob(f"*.{a.kind}"))[:3]
        if not files:
            print(f"no *.{a.kind} files under {a.path}", file=sys.stderr)
            return 2
    else:
        files = [Path(a.path)]

    bad = 0
    for f in files:
        if a.mode == "gnt" or (a.mode == "dir" and a.kind == "gnt"):
            r = probe_gnt(f, limit=a.limit)
            print(json.dumps(r.__dict__, ensure_ascii=False, indent=2))
            bad += 0 if r.ok else 1
        else:
            r, lb = probe_pot(f, limit=a.limit)
            d = r.__dict__ | {"label_bytes": lb}
            print(json.dumps(d, ensure_ascii=False, indent=2))
            bad += 0 if r.ok else 1

    if bad:
        print(f"\n{bad}/{len(files)} file(s) FAILED the layout check. Do not build yet.",
              file=sys.stderr)
        return 1
    print(f"\n{len(files)}/{len(files)} file(s) consistent with the documented layout.")
    print("Pass --limit higher on one file to be more confident before a full build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
