"""Bridge to the shared (torch-free) layer.

`common/` is deliberately dependency-light so the baseline adapters can use it without
pulling in torch. `cent/` needs a couple of its helpers; they are re-exported here so the
import path lives in one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.canonical_lines import _read_png as read_png  # noqa: E402,F401
from common.lexicon import fit_strokes_to_count, topology_target  # noqa: E402,F401

__all__ = ["read_png", "topology_target", "fit_strokes_to_count"]
