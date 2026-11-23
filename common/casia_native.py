"""Corrected CASIA line parsers, written against the supplier's own format tables.

The earlier `.dgrl` / `.wptt` readers were built from a second-hand description and were wrong in
ways that only appear on real files. The actual layouts, from NLPR's published tables:

`.dgrl` (offline page, Table 4 of the offline database page)
    Size of header   int32 LE   = 36 + strlen(illustr)
    Format code      8B         "DGRL"
    Illustration     (hs - 36)  bytes
    Code type        20B        "GB"
    Code length      2B         2 for GB
    Bits per pixel   2B         1 or 8
    Height, Width, LineNum  3 x int32 LE
    per line: CharNum int32, CharNum x CodeLength label bytes, Top int32, Left int32,
              Height int32, Width int32, bitmap of H*((W+7)//8) bytes (1bpp) or H*W (8bpp)

    Note the label block comes FIRST and the geometry after it, and that the 8-bit bitmaps are one
    byte per pixel -- both were wrong before, which is why a real file failed at the first record.

`.wptt` (online page, Table 4 of the online database page)
    Size of header   4B (54 + strlen(illustr) + 1)
    Format code      8B         "WPTT"
    Illustration, Code type 20B, Code length 2B, Sample length 4B, Page index 4B
    Stroke number    int32
    per stroke:      PointNum u16, PointNum x (x u16, y u16)   [coords are x10]
    Line number      u16
    per line:        StrokeNum u16, StrokeNum x u16 index, CharNum u16,
                     CharNum x CodeLength label bytes

    The identity  SampleLen = 4+4+4 + sum(2 + 4*pts) + 2 + sum(2 + 2*ns + 2 + nc*codeLen)
    is used as a self-check: a parse that does not consume exactly the declared length is refused
    rather than silently misaligned.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .casia_lines import DgrlLine, LineFormatError, WpttLine, WpttPage, WpttStroke


def _decode_gb(label: bytes, code_len: int) -> str:
    """Decode the label block; 0xff marks a garbage/out-of-class character."""
    out = []
    for i in range(0, len(label), code_len):
        chunk = label[i:i + code_len]
        if not chunk or chunk[0] == 0xFF:
            continue
        try:
            out.append(chunk.decode("gbk", errors="ignore"))
        except Exception:
            out.append("")
    return "".join(out)


def _unpack_1bpp(buf: bytes, off: int, h: int, w: int) -> np.ndarray:
    stride = (w + 7) // 8
    raw = np.frombuffer(buf, np.uint8, h * stride, off)
    bits = np.unpackbits(raw).reshape(h, stride * 8)[:, :w]
    return np.where(bits > 0, 0, 255).astype(np.uint8)      # 1 = ink = 0, matching 8bpp convention


def iter_dgrl(path: str | Path):
    """Yield DgrlLine records from one page file."""
    p = Path(path)
    buf = p.read_bytes()
    if len(buf) < 40:
        raise LineFormatError(f"{p.name}: too short")
    (hs,) = struct.unpack_from("<I", buf, 0)
    fmt = buf[4:12].rstrip(b"\x00")
    if not fmt.startswith(b"DGR"):
        raise LineFormatError(f"{p.name}: bad magic {fmt!r}, expected DGRL")
    illu = hs - 36
    if illu < 0 or 12 + illu + 24 != hs + 0:
        # 4 size + 8 fmt + illu + 20 code type + 2 code len + 2 bpp == hs
        if not (4 + 8 + illu + 20 + 2 + 2 == hs):
            raise LineFormatError(f"{p.name}: header {hs} inconsistent with illustration {illu}")
    off = 4 + 8 + illu
    code_type = buf[off:off + 20].rstrip(b"\x00"); off += 20
    if not code_type.startswith(b"GB"):
        raise LineFormatError(f"{p.name}: unsupported code type {code_type!r}")
    (code_len,) = struct.unpack_from("<H", buf, off); off += 2
    (bpp,) = struct.unpack_from("<H", buf, off); off += 2
    if off != hs:
        raise LineFormatError(f"{p.name}: parsed {off} bytes of a declared {hs}-byte header")
    if code_len not in (1, 2, 4):
        raise LineFormatError(f"{p.name}: implausible code length {code_len}")

    h, w, n_lines = struct.unpack_from("<III", buf, off); off += 12
    for k in range(n_lines):
        (n_chars,) = struct.unpack_from("<I", buf, off); off += 4
        label = buf[off:off + n_chars * code_len]; off += n_chars * code_len
        text = _decode_gb(label, code_len)
        top, left, lh, lw = struct.unpack_from("<IIII", buf, off); off += 16
        nbytes = lh * ((lw + 7) // 8) if bpp == 1 else lh * lw
        if off + nbytes > len(buf):
            raise LineFormatError(f"{p.name}: line {k} bitmap runs past EOF")
        if bpp == 1:
            bmp = _unpack_1bpp(buf, off, lh, lw)
        else:
            bmp = np.frombuffer(buf, np.uint8, nbytes, off).reshape(lh, lw).copy()
        off += nbytes
        yield DgrlLine(p.stem, k, text, top, left, lh, lw, bmp)


def iter_wptt(path: str | Path) -> WpttPage:
    """Parse one online page file. Self-validates against the declared sample length."""
    p = Path(path)
    buf = p.read_bytes()
    (hs,) = struct.unpack_from("<I", buf, 0)
    fmt = buf[4:12].rstrip(b"\x00")
    if not fmt.startswith(b"WPTT"):
        raise LineFormatError(f"{p.name}: bad magic {fmt!r}, expected WPTT")
    if not (0 < hs < len(buf)):
        raise LineFormatError(f"{p.name}: implausible header {hs}")

    last_err: Exception | None = None
    for code_len in (2, 1, 4):        # GB is 2; the others are only tried if the check fails
        try:
            page = _parse_wptt_body(buf, hs, code_len, p.stem)
            return page
        except Exception as e:        # noqa: BLE001 - a failed candidate is expected
            last_err = e
    raise LineFormatError(f"{p.name}: no code length produced a consistent parse ({last_err})")


def _parse_wptt_body(buf: bytes, hs: int, code_len: int, page: str) -> WpttPage:
    off = hs
    sample_len, page_index, n_strokes = struct.unpack_from("<III", buf, off); off += 12
    if n_strokes > 100_000:
        raise LineFormatError(f"implausible stroke count {n_strokes}")
    strokes: list[WpttStroke] = []
    for s in range(n_strokes):
        (n_pts,) = struct.unpack_from("<H", buf, off); off += 2
        if n_pts > 20_000 or off + n_pts * 4 > len(buf):
            raise LineFormatError(f"stroke {s}: implausible point count {n_pts}")
        pts = np.frombuffer(buf, "<u2", n_pts * 2, off).reshape(n_pts, 2).astype(np.float32) / 10.0
        off += n_pts * 4
        strokes.append(WpttStroke(s, pts))

    (n_lines,) = struct.unpack_from("<H", buf, off); off += 2
    lines: list[WpttLine] = []
    for k in range(n_lines):
        (n_s,) = struct.unpack_from("<H", buf, off); off += 2
        idx = np.frombuffer(buf, "<u2", n_s, off); off += n_s * 2
        (n_c,) = struct.unpack_from("<H", buf, off); off += 2
        label = buf[off:off + n_c * code_len]; off += n_c * code_len
        lines.append(WpttLine(page, k, _decode_gb(label, code_len), [int(i) for i in idx]))

    consumed = off - hs
    if consumed != sample_len:
        raise LineFormatError(f"consumed {consumed} bytes but sample length declares {sample_len}")
    return WpttPage(page_index, strokes, lines, page)
