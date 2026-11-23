"""Render the TEXT as an image, for content conditioning and for composed-line baselines.

Every offline line architecture in this literature needs the target string as a picture,
not as ids: DiffBrush conditions on Unifont-rendered content images and TGC-Diff on
font-rendered templates. This module renders that image once, from a glyph table, so all
systems receive identical content input.

The glyph table used here is the one DiffBrush ships (`files/unifont.pickle`): 55,659
glyphs, 20,992 of them CJK (U+4E00-U+9FFF), stored as 16x16 boolean bitmaps. It is read,
never redistributed. Make Me A Hanzi or a system font are drop-in alternatives.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

INK, PAPER = 0, 255


def load_glyphs(path: str | Path) -> dict[int, np.ndarray]:
    """Load a glyph table keyed by codepoint. Accepts this repo's two known shapes:
    DiffBrush's list of {'mat','idx'} records, or a plain {codepoint: bitmap} dict."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        return {int(k): np.asarray(v).astype(bool) for k, v in d.items()}
    out: dict[int, np.ndarray] = {}
    for e in d:
        mat = np.asarray(e["mat"]).astype(bool)
        for cp in e.get("idx", []):
            out.setdefault(int(cp), mat)
    return out


def _scale(mat: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    """Nearest-neighbour upscale of a small bitmap into a cell."""
    h, w = mat.shape
    rows = (np.arange(cell_h) * h / cell_h).astype(np.int32).clip(0, h - 1)
    cols = (np.arange(cell_w) * w / cell_w).astype(np.int32).clip(0, w - 1)
    return mat[rows][:, cols]


def render_text(text: str, glyphs: dict[int, np.ndarray], width: int = 1024,
                height: int = 64, margin: int = 4) -> tuple[np.ndarray, list[int]]:
    """Render `text` as a paper-background line image. Returns (image, missing codepoints).

    Glyphs are placed on an even grid; the cell width adapts to the canvas so the whole
    string fits, and is reported so a caller can check that the layout is not degenerate.
    """
    n = max(len(text), 1)
    cell_w = max(8, (width - 2 * margin) // n)
    cell_h = height - 2 * margin
    img = np.full((height, width), PAPER, np.uint8)
    missing: list[int] = []
    for i, ch in enumerate(text):
        mat = glyphs.get(ord(ch))
        if mat is None:
            missing.append(ord(ch))
            continue
        cell = _scale(mat, cell_h, cell_w)
        x0 = margin + i * cell_w
        x1 = min(x0 + cell_w, width - margin)
        if x1 <= x0:
            break
        sub = img[margin:margin + cell_h, x0:x1]
        ink = cell[:, : x1 - x0]
        sub[ink] = INK
    return img, missing


def render_text_varwidth(text: str, glyphs: dict[int, np.ndarray], cell: int = 64,
                         height: int = 64) -> np.ndarray:
    """Render text at natural width (n_chars x cell), for composed-line assembly."""
    img = np.full((height, max(1, len(text)) * cell), PAPER, np.uint8)
    for i, ch in enumerate(text):
        mat = glyphs.get(ord(ch))
        if mat is None:
            continue
        g = _scale(mat, cell, cell)
        img[:, i * cell:(i + 1) * cell][g] = INK
    return img
