#!/usr/bin/env python
"""Stage 1/2: train the InkVAE (characters or lines)

Thin wrapper: the implementation lives in `cent.pipeline.stage_vae`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cent.pipeline.stage_vae import main

if __name__ == "__main__":
    raise SystemExit(main())
