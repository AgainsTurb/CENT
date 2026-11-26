"""Figures for every training stage.

matplotlib is used when available (installed on the GPU box) because axes,
labels, legends and anti-aliased strokes make these images actually readable.
A dependency-free numpy renderer is kept as a fallback so the pipeline still
produces something on a bare interpreter.

Two kinds of functions:
  * ``figure_*`` / ``plot_*``  -> draw with matplotlib and write a PNG directly
  * ``render_strokes`` and the stacking helpers -> return uint8 RGB arrays,
    used to compose side-by-side comparison panels (backend independent)
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

try:  # pragma: no cover - exercised by availability, not by tests
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # noqa: BLE001
    plt = None
    HAVE_MPL = False

_CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
              "Source Han Sans SC", "PingFang SC", "Arial Unicode MS",
              "WenQuanYi Zen Hei", "SimSun")
CJK_FONT: str | None = None


def _setup_fonts() -> str | None:
    """Pick a CJK-capable font so transcripts in captions are not tofu."""
    global CJK_FONT
    if not HAVE_MPL:
        return None
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in _CJK_FONTS:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                CJK_FONT = name
                return name
    except Exception:  # noqa: BLE001
        pass
    return None


_setup_fonts()

# ---------------------------------------------------------------- PNG (no PIL)


def save_png(arr: np.ndarray, path: str | Path) -> Path:
    """Write an HxWx3 uint8 RGB array as PNG (zlib only, no Pillow)."""
    import struct

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += chunk(b"IDAT", zlib.compress(raw, 6))
    out += chunk(b"IEND", b"")
    path.write_bytes(out)
    return path


def _fig_array(fig) -> np.ndarray:
    """Rasterise a matplotlib figure to an RGB array (no PIL needed)."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def _finish(fig, path: str | Path, *, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- numpy canvas


def canvas(h: int, w: int, bg: int = 255) -> np.ndarray:
    return np.full((h, w, 3), bg, dtype=np.uint8)


def _line(arr, p0, p1, colour, thickness: int = 1) -> None:
    h, w = arr.shape[:2]
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        for oy in range(thickness):
            for ox in range(thickness):
                xx, yy = x0 + ox, y0 + oy
                if 0 <= xx < w and 0 <= yy < h:
                    arr[yy, xx] = colour
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def render_strokes(strokes, *, height: int = 180, pad: int = 10, thickness: int = 2,
                   colour=(20, 20, 20), bg: int = 255,
                   boundary_x: list[float] | None = None,
                   boundary_colour=(220, 60, 60)) -> np.ndarray:
    """Rasterise strokes to an RGB array (used for comparison panels)."""
    strokes = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) > 0]
    if not strokes:
        return canvas(height, 4 * height, bg)
    pts = np.concatenate(strokes, axis=0)
    if not np.isfinite(pts).all():      # diverged model: draw a blank panel
        return canvas(height, 4 * height, bg)
    lo = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - lo, 1e-6)
    scale = (height - 2 * pad) / span[1]
    width = int(span[0] * scale + 2 * pad) + 1
    img = canvas(height, max(width, 8), bg)

    def to_px(p):
        return ((p[0] - lo[0]) * scale + pad, (p[1] - lo[1]) * scale + pad)

    for s in strokes:
        q = np.stack([to_px(p) for p in s])
        for i in range(len(q) - 1):
            _line(img, q[i], q[i + 1], colour, thickness)
    for bx in (boundary_x or []):
        x = (bx - lo[0]) * scale + pad
        if 0 <= x < img.shape[1]:
            img[:, int(x)] = np.array(boundary_colour, dtype=np.uint8)
    return img


def hstack_padded(images, gap: int = 8, bg: int = 255) -> np.ndarray:
    if not images:
        return canvas(8, 8, bg)
    h = max(i.shape[0] for i in images)
    w = sum(i.shape[1] for i in images) + gap * (len(images) - 1)
    out = canvas(h, w, bg)
    x = 0
    for im in images:
        out[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return out


def vstack_padded(images, gap: int = 10, bg: int = 255) -> np.ndarray:
    if not images:
        return canvas(8, 8, bg)
    w = max(i.shape[1] for i in images)
    h = sum(i.shape[0] for i in images) + gap * (len(images) - 1)
    out = canvas(h, w, bg)
    y = 0
    for im in images:
        out[y : y + im.shape[0], : im.shape[1]] = im
        y += im.shape[0] + gap
    return out


def grid_of(images, cols: int = 2, gap: int = 10, bg: int = 255) -> np.ndarray:
    rows = [hstack_padded(images[i : i + cols], gap, bg)
            for i in range(0, len(images), cols)]
    return vstack_padded(rows, gap, bg)


# ---------------------------------------------------------------- matplotlib figures


def figure_strokes(strokes, path: str | Path, *, title: str | None = None,
                   boundary_x: list[float] | None = None,
                   colour: str = "#1a1a1a", figsize=(13, 1.7), dpi: int = 150,
                   linewidth: float = 1.1) -> Path:
    """Publication-quality stroke drawing, with optional boundary guides."""
    strokes = [np.asarray(s, dtype=float) for s in strokes if len(s) > 1]
    if not HAVE_MPL or not strokes:
        return save_png(render_strokes(strokes, boundary_x=boundary_x), path)
    fig, ax = plt.subplots(figsize=figsize)
    for s in strokes:
        ax.plot(s[:, 0], s[:, 1], color=colour, linewidth=linewidth,
                solid_capstyle="round", solid_joinstyle="round")
    if boundary_x:
        for bx in boundary_x:
            ax.axvline(bx, color="#d33", linewidth=0.7, alpha=0.55, linestyle="--")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, loc="left", pad=2)
    return _finish(fig, path, dpi=dpi)


def plot_curves(series: dict, path: str | Path, *, title: str, ylabel: str = "loss",
                xlabel: str = "step", logy: bool = False, dpi: int = 140,
                smooth: int = 0) -> Path:
    """Multi-series line chart with real axes and a legend."""
    if not HAVE_MPL:
        vals = list(series.values())
        return save_png(_fallback_line(vals), path)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for name, ys in series.items():
        ys = np.asarray([y for y in ys if np.isfinite(y)], dtype=float)
        if ys.size == 0:
            continue
        if smooth and ys.size > smooth:
            k = np.ones(smooth) / smooth
            ys = np.convolve(ys, k, mode="valid")
        ax.plot(ys, label=name, linewidth=1.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    return _finish(fig, path, dpi=dpi)


def plot_bars(values, path: str | Path, *, title: str, labels=None,
              ylabel: str = "", vmax: float | None = None,
              hline: float | None = None, hline_label: str = "",
              dpi: int = 140) -> Path:
    if not HAVE_MPL:
        return save_png(_fallback_bars(values), path)
    vals = [float(v) if np.isfinite(v) else 0.0 for v in values]
    fig, ax = plt.subplots(figsize=(max(4.0, 0.28 * len(vals) + 2.2), 2.9))
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color="#2f6fbb", width=0.75)
    if vmax is not None:
        ax.set_ylim(0, vmax)
    if hline is not None:
        ax.axhline(hline, color="#d33", linestyle="--", linewidth=1.0,
                   label=hline_label or f"{hline:.2f}")
        ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if labels is not None and len(labels) <= 30:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(l) for l in labels], rotation=90, fontsize=6)
    else:
        ax.set_xlabel("index")
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, path, dpi=dpi)


def plot_hist(values, path: str | Path, *, title: str, bins: int = 25,
              xlabel: str = "", ylabel: str = "count", dpi: int = 140) -> Path:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not HAVE_MPL or vals.size == 0:
        return save_png(_fallback_bars(vals[:40] if vals.size else [0.0]), path)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.hist(vals, bins=bins, color="#3a7d44", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, path, dpi=dpi)


def plot_scatter(xs, ys, path: str | Path, *, title: str, xlabel: str = "",
                 ylabel: str = "", hline: float | None = None, dpi: int = 140) -> Path:
    if not HAVE_MPL:
        return save_png(_fallback_bars(ys), path)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.scatter(xs, ys, s=18, color="#7a3fa0", alpha=0.8, edgecolors="none")
    if hline is not None:
        ax.axhline(hline, color="#d33", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _finish(fig, path, dpi=dpi)


def figure_panels(rows: list[tuple[str, np.ndarray]], path: str | Path, *,
                  title: str = "", dpi: int = 150) -> Path:
    """Stack several RGB arrays vertically, each with a small caption."""
    if not HAVE_MPL:
        return save_png(vstack_padded([r[1] for r in rows]), path)
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(13, 0.42 * n + 0.35 * n))
    axes = np.atleast_1d(axes)
    for ax, (cap, img) in zip(axes, rows):
        ax.imshow(img)
        ax.set_title(cap, fontsize=8, loc="left", pad=1)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10, y=1.0)
    return _finish(fig, path, dpi=dpi)


# ---------------------------------------------------------------- fallbacks


def _fallback_bars(vals) -> np.ndarray:
    vals = [float(v) for v in vals]
    n = max(1, len(vals))
    h, w = 140, max(64, n * 14 + 16)
    img = canvas(h, w)
    vmax = max(1e-6, max(vals) if vals else 1.0)
    for i, v in enumerate(vals):
        hgt = int(v / vmax * (h - 20))
        x0 = 6 + i * ((w - 12) // n)
        img[h - 6 - hgt : h - 6, x0 : x0 + max(2, (w - 12) // n - 2)] = (47, 111, 187)
    return img


def _fallback_line(series) -> np.ndarray:
    h, w = 140, 320
    img = canvas(h, w)
    all_v = [v for s in series for v in s if np.isfinite(v)]
    if not all_v:
        return img
    lo, hi = min(all_v), max(all_v)
    rng = hi - lo if hi > lo else 1.0
    for k, s in enumerate(series):
        col = ((200, 60, 60), (40, 120, 200), (40, 160, 90))[k % 3]
        pts = [(4 + i * (w - 8) / max(1, len(s) - 1), h - 6 - (v - lo) / rng * (h - 16))
               for i, v in enumerate(s) if np.isfinite(v)]
        for a, b in zip(pts, pts[1:]):
            _line(img, a, b, col, 1)
    return img


def backend_name() -> str:
    if not HAVE_MPL:
        return "numpy-fallback"
    return f"matplotlib (CJK font: {CJK_FONT or 'none — captions will drop CJK'})"
