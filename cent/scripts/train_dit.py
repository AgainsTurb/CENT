#!/usr/bin/env python
"""Stage 4/5: train BoundaryDiT and calibrate cursive control

Thin wrapper: the implementation lives in `cent.pipeline.stage_dit`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cent.pipeline.stage_dit import main

if __name__ == "__main__":
    raise SystemExit(main())
