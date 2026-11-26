#!/usr/bin/env bash
# One-shot setup for the CENT GPU box (vast.ai base image).
# Usage:  bash remote_setup.sh
set -euo pipefail

echo "== hardware =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python3 -V || true
df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1

echo "== python deps =="
pip install -q --upgrade pip
# torch first (CUDA 12.x wheel); the base image usually has no torch
python3 - <<'PY' || pip install -q torch --index-url https://download.pytorch.org/whl/cu124
import torch, sys
print("torch already present:", torch.__version__, "cuda:", torch.cuda.is_available())
PY
pip install -q numpy pyyaml tqdm matplotlib

echo "== versions =="
python3 - <<'PY'
import torch, numpy, matplotlib
print("torch  :", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("numpy  :", numpy.__version__)
print("mpl    :", matplotlib.__version__)
from cent.viz import backend_name
print("figures:", backend_name())
PY

echo "== architecture smoke (no training) =="
python3 scripts/smoke_test.py

echo "== data (downloads WPTT if absent, ~76 MB) =="
python3 scripts/setup_data.py || echo "data fetch failed; rerun before training"

echo
echo "READY. Next:"
echo "  python3 scripts/run_all.py --stop-after stage2     # sanity + VAE"
echo "  tail -f runs/stage2_vae/metrics.json"
