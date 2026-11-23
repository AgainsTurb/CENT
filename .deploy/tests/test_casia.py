"""Tests for the CASIA readers.

IMPORTANT SCOPE NOTE: these tests exercise the documented record layout using
synthetic .gnt/.pot files this script writes itself. They prove the reader's
internal consistency (header accounting, y-flip, pen-lift splitting, stroke-count
agreement, error on truncation). They are NOT evidence that the real corpus
parses -- that can only be established by `probe_gnt`/`probe_pot` on the actual
files on the GPU server. Run this any time; run the probe before trusting a run.

Usage: python tests/test_casia.py
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CENT.common.casia import (  # noqa: E402
    RecordFormatError,
    iter_gnt,
    iter_pot,
    probe_gnt,
    probe_pot,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


def gb_bytes(char: str) -> bytes:
    return char.encode("gb2312")


def write_gnt(path: Path, samples: list[tuple[str, np.ndarray]]) -> None:
    """Minimal writer following the documented layout (row 0 = bottom)."""
    with open(path, "wb") as fh:
        for char, img in samples:
            h, w = img.shape
            bottom_up = np.flipud(img).astype(np.uint8)
            total = 4 + 2 + 2 + 2 + w * h
            fh.write(struct.pack(">I", total))
            fh.write(gb_bytes(char).rjust(2, b"\x00")[:2])
            fh.write(struct.pack(">HH", w, h))
            fh.write(bottom_up.tobytes())


def write_pot(path: Path, samples: list[tuple[str, list[np.ndarray]]]) -> None:
    with open(path, "wb") as fh:
        for char, strokes in samples:
            label = int.from_bytes(gb_bytes(char), "big")
            pts = []
            for si, s in enumerate(strokes):
                for x, y in s:
                    pts.append((int(x), int(-y)))   # store y-up
                if si != len(strokes) - 1:
                    pts.append((-1, 0))            # pen lift between strokes
            body = struct.pack(">I", label) + struct.pack(">H", len(strokes))
            body += b"".join(struct.pack(">hh", x, y) for x, y in pts)
            fh.write(struct.pack(">I", 4 + len(body)))
            fh.write(body)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="casia_fixture_"))
    print(f"fixtures in {tmp}")

    # ---- offline ----
    g = np.full((7, 5), 255, np.uint8)   # 255 = background
    g[0, :] = 0          # ink on the TOP row of the intended image
    g[:, 0] = 0          # ink on the LEFT column
    gnt = tmp / "001.gnt"
    write_gnt(gnt, [("\u65e5", g), ("\u5927", np.full((9, 9), 255, np.uint8))])  # ri, da

    print("\n.gnt")
    recs = list(iter_gnt(gnt))
    check("two records parsed", len(recs) == 2, f"got {len(recs)}")
    check("character decoded as ri", recs[0].char == "\u65e5", f"got {recs[0].char!r}")
    check("shape preserved", recs[0].bitmap.shape == (7, 5), str(recs[0].bitmap.shape))
    check("y-flip applied (ink at top)", recs[0].bitmap[0, 0] == 0 and recs[0].bitmap[-1, -1] == 255,
          f"corner values {recs[0].bitmap[0,0]}/{recs[0].bitmap[-1,-1]}")
    check("blank glyph stays blank", int(recs[1].bitmap.min()) == 255)
    pr = probe_gnt(gnt)
    check("probe reports consistent", pr.ok and pr.samples == 2, pr.detail)

    # truncation must raise, not silently yield a short glyph
    bad = tmp / "bad.gnt"
    data = gnt.read_bytes()
    bad.write_bytes(data[:-3])
    raised = False
    try:
        list(iter_gnt(bad))
    except RecordFormatError:
        raised = True
    check("truncated record raises RecordFormatError", raised)

    # ---- online ----
    pot = tmp / "001.pot"
    strokes = [
        np.array([[10, 10], [20, 20], [30, 30]], np.int16),
        np.array([[40, 5], [50, 6]], np.int16),
        np.array([[60, 60]], np.int16),
    ]
    write_pot(pot, [("\u4eba", strokes)])  # ren
    print("\n.pot")
    on = list(iter_pot(pot, label_bytes=4))
    check("one online record", len(on) == 1)
    check("stroke count matches header", on[0].n_strokes == 3, str(on[0].n_strokes))
    check("stroke split correctly", [len(s) for s in on[0].strokes] == [3, 2, 1],
          str([len(s) for s in on[0].strokes]))
    check("character decoded as ren", on[0].char == "\u4eba", f"got {on[0].char!r}")
    # write_pot stores y negated (y-up); the reader negates back, so a round trip
    # must return the original coordinate. Asserting the round trip is the point.
    check("stroke y round-trips through the y-flip", on[0].strokes[0][0, 1] == 10,
          str(on[0].strokes[0][0]))

    res, lb = probe_pot(pot)
    check("probe_pot discovers 4-byte label", res.ok and lb == 4, f"{res.detail} (lb={lb})")

    # a 2-byte-label file must NOT parse as 4-byte: the stroke count will disagree
    pot2 = tmp / "002.pot"
    with open(pot2, "wb") as fh:
        label16 = int.from_bytes(gb_bytes("\u4eba"), "big")
        body = struct.pack(">H", label16) + struct.pack(">H", 2)
        body += struct.pack(">hh", 1, 2) + struct.pack(">hh", -1, 0) + struct.pack(">hh", 3, 4)
        fh.write(struct.pack(">I", 4 + len(body)))
        fh.write(body)
    res2, lb2 = probe_pot(pot2)
    check("probe_pot falls back to 2-byte label", res2.ok and lb2 == 2, f"{res2.detail} (lb={lb2})")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    print("\nNOTE: passing these tests does not validate the real corpus. Run")
    print("      `python -m CENT.common.probe <gnt|pot> <file>` on the server first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
