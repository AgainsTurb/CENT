"""Contact sheets for looking at generated lines with your eyes.

A number can be right while the line looks wrong: joins that satisfy a rate but kink, characters
that are the right class but the wrong hand. So every checkpoint gets a sheet.

For line images (1024 px wide) the natural reading order is VERTICAL: ground truth above the
generated image above the generated trajectory, with a label line between samples. A horizontal
strip of three 1024-wide tiles cannot fit, and truncating them was hiding exactly what the sheet
exists to show.

Labels are drawn from the Unifont glyph table, so the tool needs numpy and the project's own PNG
writer, nothing else.

Usage
    python -m benchmark.visualise_run --config runs/casia_r1/config.json \
        --ckpt runs/casia_r1/ckpt_last.pt --split val --n 8 --out runs/casia_r1/sheet_val.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.canonical_lines import _save_png
from common.content_render import load_glyphs

INK, PAPER = 0, 255


def _text(text: str, glyphs: dict[int, np.ndarray], h: int = 12) -> np.ndarray:
    """Tiny label renderer: one glyph per cell, cell width 0.6 x height."""
    if not text:
        return np.full((h, 1), PAPER, np.uint8)
    cell = max(4, int(round(h * 0.6)))
    img = np.full((h, cell * len(text)), PAPER, np.uint8)
    for i, ch in enumerate(text):
        g = glyphs.get(ord(ch))
        if g is None:
            continue
        gh, gw = g.shape
        rows = (np.arange(h) * gh / h).astype(int).clip(0, gh - 1)
        cols = (np.arange(cell) * gw / cell).astype(int).clip(0, gw - 1)
        img[:, i * cell:(i + 1) * cell][g[rows][:, cols]] = INK
    return img


def _to_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    a = np.clip(a, 0, 1) * 255 if float(np.nanmax(a)) <= 1.0 else np.clip(a, 0, 255)
    return a.astype(np.uint8)


def _pad_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.full((h, w), PAPER, np.uint8)
    hh, ww = min(h, img.shape[0]), min(w, img.shape[1])
    out[:hh, :ww] = img[:hh, :ww]
    return out


def contact_sheet(rows: list[tuple[str, list[np.ndarray]]], out_path: str,
                  glyphs_path: str = "baselines/diffbrush/files/unifont.pickle",
                  gap: int = 5, label_h: int = 13, layout: str = "vertical") -> str:
    """Write a PNG sheet.

    rows: (label, [images]). With layout="vertical" (default) the images of a row are stacked
    top-to-bottom, which is how line images should be read; "horizontal" stacks them left to
    right, and the sheet widens to fit instead of truncating.
    """
    glyphs = load_glyphs(glyphs_path) if glyphs_path else {}
    blocks: list[np.ndarray] = []
    for label, imgs in rows:
        imgs = [_to_u8(i) for i in imgs if i is not None]
        if not imgs:
            continue
        if layout == "horizontal":
            h = max(i.shape[0] for i in imgs)
            w = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1)
        else:
            h = sum(i.shape[0] for i in imgs) + gap * (len(imgs) - 1)
            w = max(i.shape[1] for i in imgs)
        body = np.full((h, w), PAPER, np.uint8)
        if layout == "horizontal":
            x = 0
            for im in imgs:
                body[:im.shape[0], x:x + im.shape[1]] = im
                x += im.shape[1] + gap
        else:
            y = 0
            for im in imgs:
                body[y:y + im.shape[0], :im.shape[1]] = im
                y += im.shape[0] + gap
        lab = _pad_to(_text(label, glyphs, label_h), w, label_h)
        blocks.append(np.vstack([lab, body]))
        blocks.append(np.full((gap, w), PAPER, np.uint8))

    if not blocks:
        raise ValueError("contact_sheet got no rows")
    W = max(b.shape[1] for b in blocks)
    sheet = np.vstack([_pad_to(b, W, b.shape[0]) for b in blocks])
    _save_png(Path(out_path), sheet)
    return out_path


def trajectory_image(strokes: list[np.ndarray], h: int = 64, w: int = 1024,
                     margin: int = 4) -> np.ndarray:
    """Rasterise a stroke list faithfully: each sample from its own curve, one pixel per point."""
    img = np.full((h, w), PAPER, np.uint8)
    pts = [s for s in strokes if isinstance(s, np.ndarray) and s.ndim == 2 and len(s)]
    if not pts:
        return img
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1e-6)
    scale = min((w - 2 * margin) / span[0], (h - 2 * margin) / span[1])
    for s in pts:
        q = (np.asarray(s, np.float32) - lo) * scale
        q[:, 0] += margin
        q[:, 1] += margin
        xi = np.clip(q[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(q[:, 1].round().astype(int), 0, h - 1)
        img[yi, xi] = INK
    return img
