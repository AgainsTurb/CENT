"""Probe real CASIA **line** files before building anything.

    python -m common.probe_lines dgrl /data/casia/HWDB2/Gnt2.0/013-P04.dgrl
    python -m common.probe_lines wptt /data/casia/OLHWDB2/Pot2.0/013-P04.wptt
    python -m common.probe_lines dir  /data/casia/HWDB2/Gnt2.0 --kind dgrl --limit 5

A probe that reports ok=false means the documented layout does not match the file. Do
not build in that case: a mis-framed record yields plausible-looking lines that are
silently wrong, and every downstream number inherits the error.

Exit code 0 = every probed file consistent; 1 = at least one failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .casia_lines import probe_dgrl, probe_wptt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["dgrl", "wptt", "dir"])
    ap.add_argument("path")
    ap.add_argument("--kind", choices=["dgrl", "wptt"], default="dgrl", help="for mode=dir")
    ap.add_argument("--limit", type=int, default=5)
    a = ap.parse_args(argv)

    if a.mode == "dir":
        files = sorted(Path(a.path).glob(f"*.{a.kind}"))[: a.limit]
        kind = a.kind
        if not files:
            print(f"no *.{a.kind} files under {a.path}", file=sys.stderr)
            return 2
    else:
        files, kind = [Path(a.path)], a.mode

    bad = 0
    for f in files:
        res = probe_dgrl(f) if kind == "dgrl" else probe_wptt(f)
        print(json.dumps(res.__dict__, ensure_ascii=False, indent=2))
        bad += 0 if res.ok else 1

    if bad:
        print(f"\n{bad}/{len(files)} file(s) FAILED. Do not build yet.", file=sys.stderr)
        return 1
    print(f"\n{len(files)}/{len(files)} file(s) consistent with the documented layout.")
    print("If connectors matter, also check that .wptt lines report more strokes than the")
    print("sum of lexicon counts for their characters -- otherwise there is nothing to learn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
