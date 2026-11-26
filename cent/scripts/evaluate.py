#!/usr/bin/env python
"""Stage 6: prefix-free evaluation on held-out writers

Thin wrapper: the implementation lives in `cent.pipeline.stage_eval`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cent.pipeline.stage_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
