#!/usr/bin/env python
"""Run the whole pipeline, stage by stage

Thin wrapper: the implementation lives in `cent.pipeline.run_all`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cent.pipeline.run_all import main

if __name__ == "__main__":
    raise SystemExit(main())
