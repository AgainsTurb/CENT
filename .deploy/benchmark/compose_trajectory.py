"""Compose per-character ONLINE trajectories into a line trajectory.

An online character model (SDT, DiffInk-in-composed-mode, any trajectory generator) emits
pen point sequences for single characters. To compare it with CENT on a line, those
sequences have to become a line: placed along a baseline and ordered.

The honest consequence, which this module makes explicit rather than hides:

    every character boundary is a PEN LIFT, so the connector rate of a composed online
    baseline is 0 BY CONSTRUCTION.

That is not a measurement of the model — it is a property of the composition. Reporting it
as if the model had chosen not to connect would be false. So `compose_trajectory` returns a
`declared` block saying so, and the connectivity metrics then read
`connector_rate = 0.0, source = "construction"` rather than "measured".

Layout is the same two-mode contract as the image composer: `uniform` (fixed advance) and
`style` (advance/baseline/jitter estimated from the k reference lines).
"""
from __future__ import annotations

import numpy as np

__all__ = ["compose_trajectory", "trajectory_to_polylines"]


def _rescale(strokes: list[np.ndarray], target_h: float, advance: float) -> list[np.ndarray]:
    """Scale a character's strokes to target height and place at x=0, baseline-centred."""
    pts = np.concatenate(strokes) if strokes else np.zeros((1, 2), np.float32)
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    scale = target_h / span[1]
    out = []
    for s in strokes:
        q = (np.asarray(s, np.float32) - lo) * scale
        out.append(q)
    return out


def compose_trajectory(char_strokes: list[list[np.ndarray]], layout: dict | None = None,
                       height: int = 64, rng: np.random.Generator | None = None,
                       mode: str = "style", nominal_cell: int = 64
                       ) -> tuple[list[np.ndarray], dict]:
    """Place per-character stroke lists into one line trajectory.

    Returns (line_strokes, meta). `line_strokes` is a list of (T,2) arrays in reading order;
    consecutive entries come from different characters and therefore imply a pen lift,
    which is what `meta["declared"]` records.
    """
    if not char_strokes:
        raise ValueError("compose_trajectory() needs at least one character")
    n = len(char_strokes)
    rng = rng or np.random.default_rng(0)

    if mode == "uniform" or not layout or layout.get("advance") is None:
        adv = [float(min(nominal_cell, max(8, 1024 // n)))] * n
        base = np.full(n, height / 2.0)
        fallback = None if mode == "uniform" else (layout or {}).get("source", "no layout supplied")
    else:
        adv = rng.normal(layout["advance"], max(layout.get("jitter_adv", 0.0), 1e-6), n).clip(4, None)
        base = rng.normal(layout["baseline"], max(layout.get("jitter_y", 0.0), 1e-6), n)
        fallback = None

    out: list[np.ndarray] = []
    x = 0.0
    for i, strokes in enumerate(char_strokes):
        if not strokes:
            x += float(adv[i])
            continue
        scaled = _rescale(strokes, target_h=height * 0.75, advance=float(adv[i]))
        # centre each character's ink in its own advance, on its own baseline
        w = max((s[:, 0].max() for s in scaled), default=0.0)
        for s in scaled:
            q = s.copy()
            q[:, 0] += x + (float(adv[i]) - w) / 2.0
            q[:, 1] += base[i] - (height * 0.75) / 2.0 + (height * 0.75) / 2.0
            out.append(q)
        x += float(adv[i])

    meta = {
        "mode": "uniform" if (mode == "uniform" or fallback) else "style",
        "requested_mode": mode,
        "n_chars": n,
        "n_strokes": len(out),
        "advance_mean": float(np.mean(adv)),
        "declared": {
            "connector_rate": 0.0,
            "source": "construction",
            "why": ("a composed online baseline concatenates independently generated "
                    "characters, so every boundary is a pen lift. This number is a property "
                    "of the composition, not a measurement of the model."),
        },
        "fallback_reason": fallback,
    }
    return out, meta


def trajectory_to_polylines(strokes: list[np.ndarray]) -> np.ndarray:
    """(S,2) point array with a NaN row inserted at every pen lift.

    Plotting libraries break a line at NaN, so this is the simplest faithful rendering of
    a trajectory whose pen lifts are real.
    """
    rows = []
    for i, s in enumerate(strokes):
        rows.append(np.asarray(s, np.float32))
        if i != len(strokes) - 1:
            rows.append(np.full((1, 2), np.nan, np.float32))
    return np.concatenate(rows) if rows else np.zeros((0, 2), np.float32)
