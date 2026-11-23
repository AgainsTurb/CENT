"""Re-export the topology helpers from the shared layer.

cent/ and common/ are separate packages so the baseline adapters can use
common/ without pulling in torch. This module keeps the import local to one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.lexicon import fit_strokes_to_count, topology_target  # noqa: E402,F401

__all__ = ["topology_target", "fit_strokes_to_count"]
