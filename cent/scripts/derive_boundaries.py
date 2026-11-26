#!/usr/bin/env python
"""Stage 3: CTC-align -> point-level boundaries + join labels

Thin wrapper: the implementation lives in `cent.pipeline.stage_boundaries`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cent.pipeline.stage_boundaries import main

if __name__ == "__main__":
    raise SystemExit(main())
