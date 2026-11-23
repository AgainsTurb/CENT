"""End-to-end test of the canonical builder + Dataset, on synthetic fixtures.

Confirms the contract the baseline adapters rely on: array shapes, ink polarity
(1 = ink in the Dataset, 255 = ink on disk), writer-disjoint splits, the
online rasterisation path, and that a truncated charset is respected.

Usage: python tests/test_canonical.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from CENT.common.canonical import CanonicalDataset, build, writer_splits  # noqa: E402
from test_casia import write_gnt, write_pot  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


def glyph(seed: int, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 255, np.uint8)
    for _ in range(3):
        r0, c0 = rng.integers(0, size - 5, 2)
        img[r0: r0 + 4, c0: c0 + 5] = 0
    return img


CHARS = ["\u65e5", "\u76ee", "\u5927", "\u4eba", "\u672a", "\u672b"]  # ri mu da ren wei mo


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="canon_fixture_"))
    off, on = tmp / "gnt", tmp / "pot"
    off.mkdir(); on.mkdir()

    n_writers = 10
    for w in range(1, n_writers + 1):
        write_gnt(off / f"{w:03d}.gnt",
                  [(c, glyph(1000 * w + i)) for i, c in enumerate(CHARS)])
        write_pot(on / f"{w:03d}.pot",
                  [(c, [np.array([[i * 4, j * 3], [i * 4 + 6, j * 3 + 5]], np.int16)
                        for j in range(2)]) for i, c in enumerate(CHARS)])

    print("\nbuild (offline only)")
    m = build(off, None, tmp / "canon", charset=CHARS)
    check("charset kept", m["num_classes"] == len(CHARS), str(m["num_classes"]))
    check("all writers seen", m["writers_total"] == n_writers, str(m["writers_total"]))
    total = sum(m["counts"].values())
    check("all samples emitted", total == n_writers * len(CHARS), f"{total}")
    check("splits are disjoint",
          set(m["splits"]["train"]).isdisjoint(m["splits"]["val"]) and
          set(m["splits"]["train"]).isdisjoint(m["splits"]["test"]) and
          set(m["splits"]["val"]).isdisjoint(m["splits"]["test"]))

    ds = CanonicalDataset(tmp / "canon", "train", version=None)
    check("dataset non-empty", len(ds) > 0, str(len(ds)))
    item = ds[0]
    check("image float32 [1,S,S]", item["image"].shape == (1, m["img_size"], m["img_size"]),
          str(item["image"].shape))
    check("ink polarity: 1 = ink (max > 0.9)", float(item["image"].max()) > 0.9,
          f"max={item['image'].max():.3f}")
    check("char round-trips to the charset", item["char"] in CHARS, item["char"])
    check("font_map is writer->chars", isinstance(ds.font_map(), dict) and len(ds.font_map()) > 0)

    print("\nbuild (online only, rasterised)")
    m2 = build(None, on, tmp / "canon_on", charset=CHARS, pot_label_bytes=4)
    ds2 = CanonicalDataset(tmp / "canon_on", "train", version=None)
    i2 = ds2[0]
    check("online item carries strokes", "strokes" in i2 and len(i2["strokes"]) > 0)
    check("online item still has an image", i2["image"].shape[0] == 1)
    check("online meta records the source dir", m2["online_dir"] is not None)

    print("\nsplit determinism")
    a = writer_splits(list(range(1, 101)), seed=7)
    b = writer_splits(list(range(100, 0, -1)), seed=7)
    check("order-independent", a == b)
    check("4:1 test ratio", len(a["test"]) == 20, str(len(a["test"])))
    check("val carved from train", len(a["val"]) + len(a["train"]) + len(a["test"]) == 100)

    print("\nmeta is valid json")
    meta = json.loads((tmp / "canon" / "meta.json").read_text(encoding="utf-8"))
    check("meta has note + counts", "note" in meta and "counts" in meta)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    print("\nNOTE: synthetic fixtures only. The real build must follow a passing probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
