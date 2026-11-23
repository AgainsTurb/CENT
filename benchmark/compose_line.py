"""Compose character (or word) images into a text line.

Two classes of baseline need this:

  * native line generators (DiffBrush, TGC-Diff, DiffInk, OLHWG) need no assembly;
  * character/word generators (GANwriting, HWT, VATr, DiffusionPen, MetaScript, SDT,
    One-DM, ...) cannot emit a line at all. The paper reports them under a **composed**
    label: generate each character independently, then place the glyphs with a layout.

Two layouts are supported, and both are reported, because they answer different questions:

  uniform  characters on an even grid, no jitter. The floor: it shows what a character
           generator does with no layout model. This is the "mechanically stitched"
           baseline the line-level papers criticise, implemented deliberately.
  style    advance width, baseline height and per-character jitter estimated from the k
           reference LINES, so the composed line inherits the target writer's spacing.
           This is the fair composed setting: it gives the character model the same
           layout information CENT generates.

Reporting composed results without saying which layout was used would be misleading, so
`compose()` returns the layout mode with the image.
"""
from __future__ import annotations

import numpy as np

INK, PAPER = 0, 255
NOMINAL_CELL = 64   # nominal glyph advance for the uniform layout, in pixels


def estimate_layout(refs: list[np.ndarray], n_chars: int, height: int = 64) -> dict:
    """Estimate advance, baseline, and jitter from reference lines (ink = dark).

    Advance = ink width / number of characters (the references are transcribed lines, so
    the count is known). Baseline = mean ink-centroid row. Jitter = spread of per-reference
    advances and centroids, which is what makes a composed line stop looking typeset.
    """
    adv, base = [], []
    for r in refs:
        ink = r < 128
        if not ink.any():
            continue
        cols = np.flatnonzero(ink.any(0))
        rows = np.flatnonzero(ink.any(1))
        adv.append((cols[-1] - cols[0] + 1) / max(n_chars, 1))
        base.append(float(rows.mean()))
    if not adv:
        return {"advance": None, "baseline": height / 2, "jitter_adv": 0.0,
                "jitter_y": 0.0, "source": "no usable reference line"}
    return {
        "advance": float(np.mean(adv)),
        "baseline": float(np.mean(base)),
        "jitter_adv": float(np.std(adv)),
        "jitter_y": float(np.std(base)),
        "source": f"{len(adv)} reference lines",
    }


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    rows = (np.arange(h) * img.shape[0] / h).astype(np.int32).clip(0, img.shape[0] - 1)
    cols = (np.arange(w) * img.shape[1] / w).astype(np.int32).clip(0, img.shape[1] - 1)
    return img[rows][:, cols]


def compose(char_images: list[np.ndarray], layout: dict | None = None,
            width: int = 1024, height: int = 64, rng: np.random.Generator | None = None,
            mode: str = "style") -> tuple[np.ndarray, dict]:
    """Place per-character images into one line.

    char_images are single-channel, 0 = ink. Anything that is not "uniform" uses the
    style statistics if they are present, and falls back to uniform with a recorded
    reason rather than inventing numbers.
    """
    if not char_images:
        raise ValueError("compose() needs at least one character image")
    n = len(char_images)
    rng = rng or np.random.default_rng(0)
    reason = None

    if mode == "uniform" or not layout or layout.get("advance") is None:
        if mode != "uniform":
            reason = (layout or {}).get("source", "no layout supplied")
        # Nominal glyph cell: a fixed advance, shrunk only if the string would not fit.
        # Stretching a glyph to fill the canvas would change the glyph itself, which is
        # not what "uniform layout" should mean.
        cell = min(NOMINAL_CELL, max(8, width // n))
        adv = [float(cell)] * n
        base_y = np.full(n, height / 2)
        jit_a = jit_y = 0.0
    else:
        adv_mean = float(layout["advance"])
        adv = rng.normal(adv_mean, max(layout.get("jitter_adv", 0.0), 1e-6), n).clip(4, None)
        base_y = rng.normal(layout["baseline"], max(layout.get("jitter_y", 0.0), 1e-6), n)
        jit_a, jit_y = layout.get("jitter_adv", 0.0), layout.get("jitter_y", 0.0)

    total = float(np.sum(adv))
    img = np.full((height, width), PAPER, np.uint8)
    x = max(0.0, (width - total) / 2.0)
    for i, ch in enumerate(char_images):
        w = max(4, int(round(adv[i])))
        g = _resize_to(ch, height - 8, w)
        x0 = int(round(x))
        if x0 + w > width:
            w = max(1, width - x0)
            g = g[:, :w]
        if w <= 0:
            break
        y0 = int(np.clip(base_y[i] - (height - 8) / 2, 0, height - (height - 8)))
        region = img[y0:y0 + g.shape[0], x0:x0 + w]
        region[g < 128] = INK
        x += adv[i]

    meta = {"mode": "uniform" if (mode == "uniform" or reason) else "style",
            "requested_mode": mode, "n_chars": n,
            "advance_mean": float(np.mean(adv)), "advance_jitter": jit_a,
            "baseline_jitter": jit_y,
            "fallback_reason": reason}
    return img, meta
