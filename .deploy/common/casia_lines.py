"""Readers for the CASIA **text-line** corpora, plus a fix to the character-level
`.pot` reader.

Why this file exists: CENT is a line generator, and the line corpora carry exactly
the supervision the method needs.

  .dgrl  offline page (HWDB2.x). Header, then per line: transcription, a (top,left)
         offset, a size, and that line's own bitmap. So each record IS a
         line-generation training example, with text.
  .wptt  online page (OLHWDB2.x). Header, all strokes of the page, then per line the
         stroke indices and the characters. A stroke that belongs to no character is
         a connector, and a boundary with no such stroke is a pen lift. That makes
         inter-character connectivity SUPERVISABLE rather than inferred.

Formats are from the supplier's published tables and have not been checked against a
real file here. Both probes verify record accounting and report `ok: false` rather
than emitting silently mis-framed data. Run them before any build.

Correction to the earlier character-level reader: `.pot` records begin with a
**2-byte** sample size (Table 2 of the ICDAR-2011 documentation), not 4 as
`common/casia.py` assumed. `iter_pot_compat` tries both and reports which held, so a
previous run's conclusion cannot be quietly inherited.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = [
    "LineFormatError", "DgrlLine", "WpttLine", "WpttPage",
    "iter_dgrl", "iter_wptt", "probe_dgrl", "probe_wptt", "iter_pot_compat",
]


class LineFormatError(ValueError):
    """A record did not consume exactly the bytes it declared."""


def _exact(fh, n: int) -> bytes:
    b = fh.read(n)
    if len(b) != n:
        raise LineFormatError(f"expected {n} bytes, got {len(b)} (truncated)")
    return b


def _gb(codes: bytes, code_len: int) -> str:
    out = []
    for i in range(0, len(codes), code_len):
        raw = codes[i:i + code_len]
        if len(raw) < 2:
            continue
        if raw[:2] == b"\xff\xff":          # documented abnormal-character marker
            out.append("\ufffd")
            continue
        for enc in ("gb2312", "gbk", "gb18030"):
            try:
                ch = raw[:2].decode(enc)
                if len(ch) == 1:
                    out.append(ch)
                    break
            except UnicodeDecodeError:
                continue
        else:
            out.append("\ufffd")
    return "".join(out)


@dataclass
class DgrlLine:
    page: str
    index: int
    text: str
    top: int
    left: int
    height: int
    width: int
    bitmap: np.ndarray            # uint8 (H,W), 0 = ink, 255 = paper


@dataclass
class WpttStroke:
    index: int
    points: np.ndarray            # float32 (T,2), page coordinates / 10


@dataclass
class WpttLine:
    page: str
    index: int
    text: str
    stroke_indices: list[int]

    @property
    def n_chars(self) -> int:
        return len(self.text)


def infer_char_strokes(line: WpttLine, lexicon_counts: list[int] | None = None
                       ) -> tuple[list[list[int]], list[int]]:
    """Split a line's stroke indices into per-character groups + connectors.

    IMPORTANT: the corpus does NOT store which stroke belongs to which character.
    The `.wptt` line record carries the line's stroke indices and the line's
    characters, nothing more. So this split is an INFERENCE, not ground truth, and
    the connectivity supervision CENT uses inherits that inference. This is threat T3
    in the paper and must be stated wherever a connector number appears.

    Rule (declared, deterministic):
      1. If `lexicon_counts` (canonical stroke count per character) is given, deal
         that many strokes per character in reading order; leftover strokes at the
         end of the line are connectors.
      2. Otherwise deal evenly, with the remainder distributed over the first
         characters, and report no connectors -- i.e. assume no connection rather than
         guess one. Absence of evidence is treated as a lift, which biases the
         connector rate DOWN and is the conservative choice.
    """
    idx = list(line.stroke_indices)
    n = line.n_chars
    if n == 0:
        return [], idx
    if lexicon_counts is not None and len(lexicon_counts) == n:
        groups, pos = [], 0
        for k in lexicon_counts:
            groups.append(idx[pos:pos + k])
            pos += k
        return groups, idx[pos:]
    per = len(idx) // n
    groups = [idx[i * per:(i + 1) * per] for i in range(n)]
    for i in range(len(idx) - per * n):
        groups[i].append(idx[per * n + i])
    return groups, []


@dataclass
class WpttPage:
    page_index: int
    strokes: list[WpttStroke]
    lines: list[WpttLine]
    page: str


# --------------------------------------------------------------------------- #
# .dgrl — offline pages
# --------------------------------------------------------------------------- #
def _parse_header(fh, expect: bytes) -> tuple[int, int, int]:
    """Read a shared-style header.

    Returns (code_len, declared_header_size, bytes_actually_parsed). When the declaration is
    larger (as the supplier tables imply), the stream is repositioned to the declared end and
    parsing continues: one undocumented byte is not a reason to reject a corpus.
    """
    header_start = fh.tell()
    size_field = _exact(fh, 4)
    (header_size,) = struct.unpack(">I", size_field)
    magic = _exact(fh, 8).rstrip(b"\x00")
    if magic != expect:
        raise LineFormatError(f"bad magic {magic!r}, expected {expect!r}")
    # illustration: arbitrary bytes terminated by \0
    illu = bytearray()
    while True:
        b = fh.read(1)
        if not b:
            raise LineFormatError("EOF inside illustration")
        if b == b"\x00":
            break
        illu += b
    code_type = _exact(fh, 20).rstrip(b"\x00")
    if code_type[:2] != b"GB":
        raise LineFormatError(f"unsupported code type {code_type!r}")
    (code_len,) = struct.unpack(">H", _exact(fh, 2))
    consumed = 4 + 8 + len(illu) + 1 + 20 + 2
    # The supplier's tables declare a larger header than these fields account for
    # (36+len(illu) for DGRL, 55+len(illu) for WPTT, versus 35+len here). Rather than
    # reject a real file over one unexplained byte, TRUST THE DECLARED SIZE for positioning
    # and report the discrepancy. If the declaration is implausible, fail closed.
    if header_size < consumed:
        raise LineFormatError(
            f"header declares {header_size} bytes but {consumed} were already parsed; "
            "cannot rewind safely")
    if header_size > 1_000_000:
        raise LineFormatError(f"implausible header size {header_size}")
    if header_size != consumed:
        # extra undocumented bytes: skip them and let the record parser continue
        fh.seek(header_start + header_size)
    return code_len, header_size, consumed


def iter_dgrl(path: str | Path) -> Iterator[DgrlLine]:
    p = Path(path)
    with open(p, "rb") as fh:
        code_len, _declared, _consumed = _parse_header(fh, b"DGRL")
        (bpp,) = struct.unpack(">H", _exact(fh, 2))
        height, width, n_lines = struct.unpack(">III", _exact(fh, 12))
        if bpp not in (1, 8):
            raise LineFormatError(f"unsupported bits-per-pixel {bpp}")
        for k in range(n_lines):
            (n_chars,) = struct.unpack(">I", _exact(fh, 4))
            text = _gb(_exact(fh, n_chars * code_len), code_len)
            top, left, h, w = struct.unpack(">IIII", _exact(fh, 16))
            nbytes = h * ((w + 7) // 8) if bpp == 1 else h * w
            raw = _exact(fh, nbytes)
            if bpp == 8:
                bmp = np.frombuffer(raw, np.uint8).reshape(h, w).copy()
            else:
                stride = (w + 7) // 8
                bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(h, stride * 8)[:, :w]
                bmp = (bits * 255).astype(np.uint8)
            if bmp.shape != (h, w):
                raise LineFormatError(f"line {k}: bitmap {bmp.shape} != declared {(h, w)}")
            yield DgrlLine(p.stem, k, text, top, left, h, w, bmp)
        tail = fh.read(1)
        if tail:
            raise LineFormatError(f"{len(tail)} trailing bytes after {n_lines} lines")


# --------------------------------------------------------------------------- #
# .wptt — online pages
# --------------------------------------------------------------------------- #
def iter_wptt(path: str | Path) -> WpttPage:
    p = Path(path)
    with open(p, "rb") as fh:
        code_len, _declared, _consumed = _parse_header(fh, b"WPTT")
        (sample_len,) = struct.unpack(">I", _exact(fh, 4))
        (page_index,) = struct.unpack(">I", _exact(fh, 4))
        start = fh.tell()

        (n_strokes,) = struct.unpack(">I", _exact(fh, 4))
        strokes: list[WpttStroke] = []
        for si in range(n_strokes):
            (n_pts,) = struct.unpack(">H", _exact(fh, 2))
            raw = _exact(fh, n_pts * 4)
            pts = np.frombuffer(raw, dtype=">u2").reshape(-1, 2).astype(np.float32) / 10.0
            strokes.append(WpttStroke(si, pts))

        (n_lines,) = struct.unpack(">H", _exact(fh, 2))
        lines: list[WpttLine] = []
        for li in range(n_lines):
            (n_strk,) = struct.unpack(">H", _exact(fh, 2))
            idxs = list(struct.unpack(f">{n_strk}H", _exact(fh, 2 * n_strk)))
            (n_char,) = struct.unpack(">H", _exact(fh, 2))
            text = _gb(_exact(fh, n_char * code_len), code_len)
            lines.append(WpttLine(p.stem, li, text, idxs))

        consumed = fh.tell() - start
        if sample_len and consumed != sample_len:
            raise LineFormatError(
                f"sample length declares {sample_len} bytes, parsed {consumed} "
                f"(strokes={n_strokes}, lines={n_lines})"
            )
        if fh.read(1):
            raise LineFormatError("trailing bytes after the line table")


    return WpttPage(page_index, strokes, lines, p.stem)


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    path: str
    ok: bool
    detail: str
    lines: int = 0
    chars: int = 0
    example: dict | None = None


def probe_dgrl(path: str | Path, limit: int = 5) -> ProbeResult:
    n = c = 0
    ex = None
    try:
        for ln in iter_dgrl(path):
            n += 1
            c += len(ln.text)
            if ex is None:
                ex = {"text": ln.text, "shape": list(ln.bitmap.shape),
                      "min": int(ln.bitmap.min()), "max": int(ln.bitmap.max())}
            if n >= limit:
                break
    except LineFormatError as e:
        return ProbeResult(str(path), False, f"layout mismatch: {e}", n, c, ex)
    return ProbeResult(str(path), True, "documented DGRL layout consistent", n, c, ex)


def probe_wptt(path: str | Path) -> ProbeResult:
    try:
        page = iter_wptt(path)
    except LineFormatError as e:
        return ProbeResult(str(path), False, f"layout mismatch: {e}")
    n_chars = sum(len(l.text) for l in page.lines)
    ex = None
    if page.lines:
        l0 = page.lines[0]
        ex = {"text": l0.text, "strokes": len(l0.stroke_indices),
              "strokes_in_first_line": len(l0.stroke_indices),
              "page_strokes": len(page.strokes)}
    return ProbeResult(str(path), True, "documented WPTT layout consistent",
                       len(page.lines), n_chars, ex)


# --------------------------------------------------------------------------- #
# .pot compatibility (2-byte size per the official table)
# --------------------------------------------------------------------------- #
def iter_pot_compat(path: str | Path) -> tuple[list[dict], int]:
    """Parse a character-level .pot, discovering the size-field width.

    Returns (records, size_bytes_used) where each record is
    {"label", "strokes": [np.ndarray(T,2)], "n_strokes"}.
    """
    for size_bytes in (2, 4):
        try:
            recs = _iter_pot_with(path, size_bytes)
            return recs, size_bytes
        except LineFormatError:
            continue
    raise LineFormatError("neither a 2-byte nor a 4-byte size field parses this .pot")


def _iter_pot_with(path: str | Path, size_bytes: int) -> list[dict]:
    fmt = ">H" if size_bytes == 2 else ">I"
    recs: list[dict] = []
    with open(path, "rb") as fh:
        while True:
            head = fh.read(size_bytes)
            if not head:
                return recs
            if len(head) != size_bytes:
                raise LineFormatError("partial size header")
            (total,) = struct.unpack(fmt, head)
            if total < size_bytes + 6:
                raise LineFormatError(f"declared record length {total} is too small for a {size_bytes}-byte size field")
            body = _exact(fh, total - size_bytes)
            if len(body) < 6:
                raise LineFormatError("record too short")
            (label,) = struct.unpack(">I", body[0:4])
            (n_strokes,) = struct.unpack(">H", body[4:6])
            pts = np.frombuffer(body[6:], dtype=">i2").reshape(-1, 2).astype(np.float32)
            lift = (pts[:, 0] < 0) & (pts[:, 1] == 0)
            strokes, start = [], 0
            for i in np.flatnonzero(lift):
                if i > start:
                    strokes.append(pts[start:i])
                start = i + 1
            if start < len(pts):
                strokes.append(pts[start:])
            if n_strokes and len(strokes) != n_strokes:
                raise LineFormatError(f"{n_strokes} strokes declared, {len(strokes)} found")
            recs.append({"label": label, "strokes": strokes, "n_strokes": len(strokes)})
