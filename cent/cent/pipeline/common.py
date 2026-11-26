"""Shared plumbing for the stage scripts."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        print("[warn] cuda requested but unavailable; falling back to cpu")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def gate(name: str, ok: bool, detail: str, *, strict: bool) -> bool:
    """Print a gate verdict; return False if it should stop the run."""
    tag = "PASS" if ok else "FAIL"
    print(f"\n[gate {name}] {tag} — {detail}")
    if not ok and strict:
        print("[gate] stopping (use --force to continue anyway)")
    return ok or not strict


def write_metrics(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("written_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def visual_check(paths: list[Path], stage: str) -> None:
    print(f"\n[visual check — {stage}] inspect these before trusting the numbers:")
    for p in paths:
        print(f"    {p}")
    print("    (metrics alone are not an acceptance criterion)")


def gmm_to_points(gmm: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """(B,T,M,6) -> (B,T,2) per-step deltas.

    mode="mean"  : mixture expectation sum_m pi_m * mu_m  (MMSE, the default).
    mode="argmax": highest-weight component only.

    The argmax was the original behaviour and it collapsed reconstruction to a
    straight line: for pen strokes the mixture usually has a large "stay put"
    component, so argmax picks a near-zero mean at almost every step while the
    expectation still carries the displacement.
    """
    pi = torch.softmax(gmm[..., 0], dim=-1)
    mu = gmm[..., 1:3]
    if mode == "mean":
        return (pi.unsqueeze(-1) * mu).sum(2)
    best = pi.argmax(-1)
    b = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    return torch.gather(mu, 2, b).squeeze(2)


def points_to_strokes(points: np.ndarray, pen: np.ndarray) -> list[np.ndarray]:
    """Split a decoded point stream at pen-up events."""
    strokes, cur = [], []
    for i in range(len(points)):
        if i > 0 and pen[i] != 0 and cur:
            strokes.append(np.asarray(cur, dtype=np.float32))
            cur = []
        cur.append(points[i])
    if cur:
        strokes.append(np.asarray(cur, dtype=np.float32))
    return [s for s in strokes if len(s) > 1]
