"""End-to-end test of the canonical line builder + LineDataset, on synthetic pages.

Confirms: line records carry text and image, per-character stroke splits are marked as
inferred, connectors are counted where the lexicon predicts fewer strokes than the line
has, writer-disjoint splits hold, and PNG round-trips pixel-exactly.

Usage: python tests/test_canonical_lines.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from CENT.common.canonical_lines import LineDataset, build_lines  # noqa: E402
from test_casia_lines import write_dgrl, write_wptt  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


# canonical stroke counts for the characters used below
LEX = {"人": 2, "大": 3, "山": 3, "水": 4}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="canonline_"))
    dg, wp = tmp / "dgrl", tmp / "wptt"
    dg.mkdir(), wp.mkdir()

    bmp = np.full((24, 90), 255, np.uint8)
    bmp[6:18, 4:40] = 0

    for w in (1, 2, 3, 4, 5, 6):
        write_dgrl(dg / f"{w:03d}-P04.dgrl", [("人大", 3, 5, bmp), ("山水", 30, 5, bmp)])

    # page 1: line 1 = 人大 with 5 strokes (2+3 canonical = 5, no connector)
    # page 1: line 2 = 山水 with 9 strokes (3+4 = 7 canonical, 2 surplus -> connectors)
    for w in (1, 2, 3, 4, 5, 6):
        strokes = [np.array([[i * 4.0, i * 2.0], [i * 4.0 + 3, i * 2.0 + 2]], np.float32) for i in range(14)]
        write_wptt(wp / f"{w:03d}-P04.wptt", strokes,
                   [("人大", list(range(5))), ("山水", list(range(5, 14)))])

    print("\nbuild (offline + online + lexicon)")
    meta = build_lines(dg, wp, tmp / "canon", lexicon=LEX)
    check("records emitted", sum(meta["counts"].values()) == 12, str(meta["counts"]))
    check("all lines paired with trajectories", meta["paired_lines"] == 12, str(meta["paired_lines"]))
    check("charset from transcripts", set("人大山水").issubset(set(meta["charset"])),
          str(meta["charset"]))
    check("splits are writer-disjoint",
          set(meta["splits"]["train"]).isdisjoint(meta["splits"]["val"])
          and set(meta["splits"]["val"]).isdisjoint(meta["splits"]["test"]))

    ds = LineDataset(tmp / "canon", "train")
    check("dataset non-empty", len(ds) > 0, str(len(ds)))
    item = ds[0]
    check("record has text", item["text"] in ("人大", "山水"), item["text"])
    check("image decoded", item["image"] is not None and item["image"].ndim == 2)
    check("image height is 64", item["image"].shape[0] == 64, str(item["image"].shape))
    check("png round-trip is pixel exact vs ink polarity",
          int(item["image"].min()) == 0 and int(item["image"].max()) == 255,
          f"min={item['image'].min()} max={item['image'].max()}")

    # connector accounting: 山水 line has 9 strokes vs 7 canonical -> 2 connectors
    by_text = {}
    for i in range(len(ds)):
        r = ds.records[i]
        by_text.setdefault(r["text"], r)
    ren = by_text["人大"]
    shan = by_text["山水"]
    check("line 人大 has no inferred connectors", ren["n_connectors"] == 0, str(ren["n_connectors"]))
    check("line 山水 has 2 inferred connectors", shan["n_connectors"] == 2, str(shan["n_connectors"]))
    check("char_strokes recorded for 山水", len(shan["char_strokes"]) == 2, str(shan["char_strokes"]))
    check("meta states the split is inferred", "INFERRED" in meta["note"], meta["note"][:60])

    print("\nbuild (offline only: no trajectories)")
    meta2 = build_lines(dg, None, tmp / "canon_off", lexicon=LEX)
    check("offline-only build works", sum(meta2["counts"].values()) == 12, str(meta2["counts"]))
    check("offline-only marks nothing paired", meta2["paired_lines"] == 0, str(meta2["paired_lines"]))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    print("\nNOTE: synthetic pages only. Probe real .dgrl/.wptt files before building.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
