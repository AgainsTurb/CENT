"""Lexicon access: character -> canonical stroke count and order.

Source: Make Me A Hanzi `graphics.txt` (Arphic Public License). Each line is a
JSON object with `character` and `strokes` (SVG paths in proper stroke order), so
`len(strokes)` is the canonical stroke count. See refs/overrides.json (`hanzi`)
and analysis/lexicon_stats.py for the licence and the inventory analysis.

Why this module exists: the paper's central claim is that the stroke count is
FIXED by a lexicon rather than predicted. That only holds if training and
inference both consult the lexicon. Without it, the model quietly learns the
observed counts and the "hard constraint" is just a label at inference time.
`topology_target` below is the single place where that is decided, and any
observed count that disagrees with the lexicon is reported, not silently used.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_lexicon(path: str | Path) -> dict[str, int]:
    """Read graphics.txt -> {character: canonical stroke count}."""
    out: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ch, st = rec.get("character"), rec.get("strokes")
            if ch and isinstance(st, list) and st:
                out[ch] = len(st)
    return out


def topology_target(observed: int, lexicon_count: int | None,
                    mode: str = "lexicon") -> tuple[int, bool]:
    """Decide how many strokes the model should be conditioned on.

    Args:
        observed: stroke count found in the online sample.
        lexicon_count: canonical count, or None if the character is not in the lexicon.
        mode: "lexicon"  -> use the lexicon count (the paper's hard constraint);
              "observed" -> use what the writer actually did (ablation A2, the
                            unconstrained control).

    Returns:
        (target_count, mismatch) where mismatch is True when the writer's own
        count differs from the lexicon. That flag is the T2 statistic.
    """
    if mode == "observed" or lexicon_count is None:
        return observed, (lexicon_count is not None and observed != lexicon_count)
    return lexicon_count, observed != lexicon_count


def fit_strokes_to_count(strokes: list, target: int) -> tuple[list, str]:
    """Truncate or pad a stroke list to exactly `target` strokes.

    Truncation drops trailing strokes (the writer used more than the canon).
    Padding repeats the last stroke with zero length, which renders nothing: a
    padding stroke must not draw ink, or the constraint would corrupt the image.

    Returns (strokes, action) with action in {"exact", "truncated", "padded"}.
    """
    if len(strokes) == target:
        return strokes, "exact"
    if len(strokes) > target:
        return strokes[:target], "truncated"
    out = list(strokes)
    import numpy as np
    blank = np.zeros((1, 2), np.float32)
    out.extend([blank] * (target - len(strokes)))
    return out, "padded"
