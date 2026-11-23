"""Tests for the line-level readers (.dgrl, .wptt) and the .pot size-field fix.

SCOPE: synthetic files written here to the supplier's published tables. This proves
the readers' internal consistency (header accounting, transcript decoding, bitmap
sizing, stroke-to-character dealing, connector detection, size-field discovery). It
does NOT prove the real corpora parse -- only `probe_dgrl`/`probe_wptt` on the actual
files can do that. Run the probes before any build.

Usage: python tests/test_casia_lines.py
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CENT.common.casia_lines import (  # noqa: E402
    LineFormatError, infer_char_strokes, iter_dgrl, iter_pot_compat, iter_wptt,
    probe_dgrl, probe_wptt,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


def gb(s: str) -> bytes:
    return s.encode("gb18030")


def write_dgrl(path: Path, lines: list[tuple[str, int, int, np.ndarray]],
               extra_header: int = 0) -> None:
    illu = b"#synthetic"
    code_len = 2
    header_size = 4 + 8 + len(illu) + 1 + 20 + 2 + extra_header
    with open(path, "wb") as fh:
        fh.write(struct.pack(">I", header_size))
        fh.write(b"DGRL" + b"\x00" * 4)
        fh.write(illu + b"\x00")
        fh.write(b"GB".ljust(20, b"\x00"))
        fh.write(struct.pack(">H", code_len))
        if extra_header:
            fh.write(b"\x00" * extra_header)
        h, w = lines[0][3].shape
        fh.write(struct.pack(">H", 8))                 # bits per pixel
        fh.write(struct.pack(">III", h, w, len(lines)))
        for text, top, left, bmp in lines:
            codes = b"".join(gb(c) for c in text)
            fh.write(struct.pack(">I", len(text)))
            fh.write(codes)
            fh.write(struct.pack(">IIII", top, left, bmp.shape[0], bmp.shape[1]))
            fh.write(bmp.astype(np.uint8).tobytes())


def write_wptt(path: Path, strokes: list[np.ndarray], lines: list[tuple[str, list[int]]],
               page_index: int = 7) -> None:
    illu = b"#synthetic"
    code_len = 2
    header_size = 4 + 8 + len(illu) + 1 + 20 + 2
    stroke_bytes = b""
    for s in strokes:
        stroke_bytes += struct.pack(">H", len(s))
        for x, y in s:
            stroke_bytes += struct.pack(">HH", int(round(x * 10)), int(round(y * 10)))
    line_bytes = struct.pack(">H", len(lines))
    for text, idxs in lines:
        line_bytes += struct.pack(">H", len(idxs))
        line_bytes += b"".join(struct.pack(">H", i) for i in idxs)
        line_bytes += struct.pack(">H", len(text))
        line_bytes += b"".join(gb(c) for c in text)
    sample_len = 4 + len(stroke_bytes) + len(line_bytes)
    with open(path, "wb") as fh:
        fh.write(struct.pack(">I", header_size))
        fh.write(b"WPTT" + b"\x00" * 4)
        fh.write(illu + b"\x00")
        fh.write(b"GB".ljust(20, b"\x00"))
        fh.write(struct.pack(">H", code_len))
        fh.write(struct.pack(">II", sample_len, page_index))
        fh.write(struct.pack(">I", len(strokes)))
        fh.write(stroke_bytes)
        fh.write(line_bytes)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="casialines_"))

    print("\n.dgrl")
    bmp = np.full((30, 120), 255, np.uint8)
    bmp[10:20, 5:40] = 0
    d = tmp / "013-P04.dgrl"
    write_dgrl(d, [("人大人大", 5, 10, bmp), ("山水", 40, 12, bmp)])
    lines = list(iter_dgrl(d))
    check("two lines parsed", len(lines) == 2, str(len(lines)))
    check("transcript decoded", lines[0].text == "人大人大", lines[0].text)
    check("bitmap shape matches", lines[0].bitmap.shape == (30, 120), str(lines[0].bitmap.shape))
    check("ink polarity preserved (0=ink)", lines[0].bitmap[15, 20] == 0 and lines[0].bitmap[0, 0] == 255)
    check("page stem recorded", lines[0].page == "013-P04", lines[0].page)
    pr = probe_dgrl(d)
    check("probe ok", pr.ok and pr.lines == 2, pr.detail)

    # truncation must raise
    bad = tmp / "bad.dgrl"
    bad.write_bytes(d.read_bytes()[:-5])
    raised = False
    try:
        list(iter_dgrl(bad))
    except LineFormatError:
        raised = True
    check("truncated page raises", raised)

    print("\nheader tolerance (the supplier tables declare a larger header)")
    d2 = tmp / "013-P05.dgrl"
    write_dgrl(d2, [("人大", 5, 10, bmp)], extra_header=1)
    try:
        l2 = list(iter_dgrl(d2))
        check("a header 1 byte larger than the parsed fields is accepted", len(l2) == 1, str(len(l2)))
        check("transcript still decodes after the skip", l2[0].text == "人大", l2[0].text)
    except LineFormatError as e:
        check("a header 1 byte larger than the parsed fields is accepted", False, str(e))

    print("\n.wptt")
    strokes = [
        np.array([[10, 10], [20, 20], [30, 10]], np.float32),
        np.array([[35, 12], [45, 12]], np.float32),
        np.array([[46, 13], [55, 15]], np.float32),   # connector
        np.array([[60, 20], [70, 25]], np.float32),
    ]
    w = tmp / "013-P04.wptt"
    # line 1: chars 人,大 with 1 stroke each; stroke 2 is surplus -> connector
    write_wptt(w, strokes, [("人大", [0, 1, 2]), ("山", [3])])
    page = iter_wptt(w)
    check("page index parsed", page.page_index == 7, str(page.page_index))
    check("strokes parsed", len(page.strokes) == 4, str(len(page.strokes)))
    check("coords rescaled from x10", abs(page.strokes[0].points[0, 0] - 10.0) < 1e-6,
          str(page.strokes[0].points[0]))
    check("two lines parsed", len(page.lines) == 2, str(len(page.lines)))
    check("line transcript decoded", page.lines[0].text == "人大", page.lines[0].text)
    check("stroke indices kept", page.lines[0].stroke_indices == [0, 1, 2],
          str(page.lines[0].stroke_indices))
    # The corpus does NOT store a per-character stroke split; the split is INFERRED.
    groups, conns = infer_char_strokes(page.lines[0], lexicon_counts=[1, 1])
    check("lexicon-guided split deals strokes per character", groups == [[0], [1]], str(groups))
    check("surplus stroke inferred as a connector", conns == [2], str(conns))
    g2, c2 = infer_char_strokes(page.lines[1], lexicon_counts=[1])
    check("single-character line has no connector", c2 == [] and g2 == [[3]], f"{g2} {c2}")
    _g3, c3 = infer_char_strokes(page.lines[0])
    check("without a lexicon, absence of evidence is treated as a lift", c3 == [], str(c3))
    pw = probe_wptt(w)
    check("probe ok", pw.ok and pw.lines == 2, pw.detail)
    check("probe reports the line stroke count",
          bool(pw.example) and pw.example["strokes_in_first_line"] == 3, str(pw.example))

    print("\n.pot size-field discovery")
    for size_bytes in (2, 4):
        p = tmp / f"pot{size_bytes}.pot"
        fmt = ">H" if size_bytes == 2 else ">I"
        body = struct.pack(">I", int.from_bytes(gb("人"), "big")) + struct.pack(">H", 2)
        body += struct.pack(">hh", 1, 2) + struct.pack(">hh", -1, 0) + struct.pack(">hh", 3, 4)
        with open(p, "wb") as fh:
            fh.write(struct.pack(fmt, size_bytes + len(body)))
            fh.write(body)
        recs, used = iter_pot_compat(p)
        check(f".pot with {size_bytes}-byte size parses", len(recs) == 1 and used == size_bytes,
              f"records={len(recs)} used={used}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    print("\nNOTE: synthetic fixtures only. Run `python -m common.probe_lines` on the real")
    print("      .dgrl/.wptt files before building anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
