"""CASIA-OLHWDB / CASIA-HWDB binary readers.

The corpora are not present on this machine (they are access-gated and large),
so nothing here is validated against real bytes. Two consequences shaped the
design:

1. **The record layout is probed, not assumed.** `probe_gnt` / `probe_pot` read
   the first records, try the documented layout, and report whether the byte
   counts are self-consistent (each record's declared length must be consumed
   exactly). If a layout does not check out, the probe says so instead of
   silently producing garbage.
2. **Every parsed record carries its own consistency check.** A record that does
   not consume its declared length raises `RecordFormatError` rather than
   yielding a truncated glyph.

Documented layouts (Liu et al., ICDAR 2011):
  .gnt  offline isolated character, one record per sample:
        [4B record bytes][2B GB code][2B width][2B height][width*height B bitmap]
        bitmap row 0 is the BOTTOM row, background = 255.
  .pot  online isolated character, one record per sample:
        [4B record bytes][4B GB code][2B n_strokes][(int16 x, int16 y) ...]
        in which a point (-1, 0) marks a pen lift / stroke boundary.

The online label is documented as 4 bytes (it carries the GB code plus a
variant flag); some redistributions use 2. `record_layout` is therefore a
parameter, and `probe_pot` discovers which one a file actually uses.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = [
    "RecordFormatError",
    "OfflineChar",
    "OnlineChar",
    "probe_gnt",
    "probe_pot",
    "iter_gnt",
    "iter_pot",
    "gbk_from_label",
]


class RecordFormatError(ValueError):
    """A record did not consume exactly its declared length."""


@dataclass
class OfflineChar:
    label: int          # raw label as stored
    char: str           # decoded character, '' if undecodable
    width: int
    height: int
    bitmap: np.ndarray  # uint8, shape (height, width), row 0 = TOP after flip


@dataclass
class OnlineChar:
    label: int
    char: str
    n_strokes: int
    strokes: list[np.ndarray]   # each (T, 2) float32, glyph coords, y flipped to image order


def gbk_from_label(label: int, width: int = 2) -> str:
    """Decode a GB2312/GBK label to a character.

    The corpora store the GB code, not Unicode. A 4-byte label holds the 2-byte
    GB code in its low half (the high half carries a variant/session flag).
    """
    try:
        if width == 4:
            code = label & 0xFFFF
        else:
            code = label & 0xFFFF
        raw = struct.pack(">H", code)
        # GB2312 first, then GBK (level-2 chars live in GBK extensions).
        for enc in ("gb2312", "gbk", "gb18030"):
            try:
                ch = raw.decode(enc)
                if len(ch) == 1:
                    return ch
            except UnicodeDecodeError:
                continue
    except (struct.error, OverflowError):
        pass
    return ""


def _read_exact(fh, n: int) -> bytes:
    b = fh.read(n)
    if len(b) != n:
        raise RecordFormatError(f"expected {n} bytes, got {len(b)} (truncated file?)")
    return b


def iter_gnt(path: str | Path) -> Iterator[OfflineChar]:
    """Yield offline isolated characters from a .gnt file."""
    with open(path, "rb") as fh:
        while True:
            head = fh.read(4)
            if not head:
                return
            if len(head) != 4:
                raise RecordFormatError("partial 4-byte length header at EOF")
            (total,) = struct.unpack(">I", head)
            body = _read_exact(fh, total - 4)
            if len(body) < 6:
                raise RecordFormatError(f"record too short: {len(body) + 4} bytes")
            (label,) = struct.unpack(">H", body[0:2])
            width, height = struct.unpack(">HH", body[2:6])
            pixels = body[6:]
            if width * height != len(pixels):
                raise RecordFormatError(
                    f"declared {width}x{height}={width * height} px but record holds "
                    f"{len(pixels)} px (first bytes: {body[:8].hex()})"
                )
            bitmap = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
            # The corpus stores row 0 at the bottom; flip so row 0 is the top.
            bitmap = np.flipud(bitmap).copy()
            yield OfflineChar(label, gbk_from_label(label, 2), width, height, bitmap)


def iter_pot(path: str | Path, label_bytes: int = 4) -> Iterator[OnlineChar]:
    """Yield online isolated characters from a .pot file."""
    if label_bytes not in (2, 4):
        raise ValueError("label_bytes must be 2 or 4")
    with open(path, "rb") as fh:
        while True:
            head = fh.read(4)
            if not head:
                return
            if len(head) != 4:
                raise RecordFormatError("partial 4-byte length header at EOF")
            (total,) = struct.unpack(">I", head)
            body = _read_exact(fh, total - 4)
            if len(body) < label_bytes + 2:
                raise RecordFormatError(f"record too short: {len(body) + 4} bytes")
            if label_bytes == 4:
                (label,) = struct.unpack(">I", body[0:4])
            else:
                (label,) = struct.unpack(">H", body[0:2])
            (n_strokes,) = struct.unpack(">H", body[label_bytes:label_bytes + 2])
            pts_bytes = body[label_bytes + 2:]
            if len(pts_bytes) % 4:
                raise RecordFormatError(
                    f"point payload {len(pts_bytes)} B is not a multiple of 4"
                )
            pts = np.frombuffer(pts_bytes, dtype=">i2").reshape(-1, 2).astype(np.float32)
            strokes = _split_on_pen_lift(pts)
            if n_strokes and len(strokes) != n_strokes:
                # A mismatch means the layout guess is wrong (e.g. 2-byte label).
                raise RecordFormatError(
                    f"declared {n_strokes} strokes but found {len(strokes)} "
                    f"stroke segments (label_bytes={label_bytes})"
                )
            strokes = [_to_image_coords(s) for s in strokes]
            yield OnlineChar(label, gbk_from_label(label, label_bytes), len(strokes), strokes)


def _split_on_pen_lift(pts: np.ndarray) -> list[np.ndarray]:
    """Split a point sequence at (-1, 0) pen-lift markers."""
    if len(pts) == 0:
        return []
    lift = (pts[:, 0] < 0) & (pts[:, 1] == 0)
    out, start = [], 0
    for i in np.flatnonzero(lift):
        if i > start:
            out.append(pts[start:i])
        start = i + 1
    if start < len(pts):
        out.append(pts[start:])
    return out


def _to_image_coords(stroke: np.ndarray) -> np.ndarray:
    """Pen coordinates are y-up; convert to image order (y-down) per stroke."""
    s = stroke.astype(np.float32, copy=True)
    s[:, 1] = -s[:, 1]
    return s


@dataclass
class ProbeResult:
    path: str
    ok: bool
    detail: str
    samples: int
    chars_decoded: int
    example: dict | None


def probe_gnt(path: str | Path, limit: int = 200) -> ProbeResult:
    """Check the documented .gnt layout against the first `limit` records."""
    n = decoded = 0
    example = None
    try:
        for rec in iter_gnt(path):
            n += 1
            if rec.char:
                decoded += 1
            if example is None:
                example = {"char": rec.char, "label": rec.label,
                           "shape": list(rec.bitmap.shape),
                           "min": int(rec.bitmap.min()), "max": int(rec.bitmap.max())}
            if n >= limit:
                break
    except RecordFormatError as e:
        return ProbeResult(str(path), False, f"layout mismatch: {e}", n, decoded, example)
    return ProbeResult(str(path), True, "documented layout consistent", n, decoded, example)


def probe_pot(path: str | Path, limit: int = 50) -> tuple[ProbeResult, int]:
    """Discover the online label width (2 or 4 bytes) that makes records consistent."""
    best: tuple[ProbeResult, int] | None = None
    for lb in (4, 2):
        n = decoded = 0
        example = None
        try:
            for rec in iter_pot(path, label_bytes=lb):
                n += 1
                if rec.char:
                    decoded += 1
                if example is None:
                    example = {"char": rec.char, "label": rec.label,
                               "strokes": rec.n_strokes,
                               "points_in_first_stroke": int(len(rec.strokes[0])) if rec.strokes else 0}
                if n >= limit:
                    break
        except RecordFormatError as e:
            res = ProbeResult(str(path), False, f"label_bytes={lb}: {e}", n, decoded, example)
            if best is None:
                best = (res, lb)
            continue
        res = ProbeResult(str(path), True, f"label_bytes={lb} consistent", n, decoded, example)
        return res, lb
    assert best is not None
    return best
