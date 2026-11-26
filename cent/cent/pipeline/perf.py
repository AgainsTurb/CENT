"""Performance plumbing shared by the training stages.

Everything here is *numerics-preserving*: bf16 autocast (Ampere has native
bf16, so no GradScaler is needed), TF32 for the fp32 matmuls that remain, and
losses/reductions forced back to fp32 so no accuracy is traded for speed.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch


def setup_perf(device: torch.device, *, amp: bool = True) -> dict:
    """Enable TF32 / cudnn autotune and report what is active."""
    info = {"amp": False, "tf32": False, "cudnn_benchmark": False, "bf16": False}
    if device.type != "cuda":
        return info
    torch.backends.cudnn.benchmark = True
    info["cudnn_benchmark"] = True
    # TF32 for the residual fp32 convolutions / matmuls (Ampere+)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        info["tf32"] = True
    except Exception:  # noqa: BLE001
        pass
    info["bf16"] = bool(torch.cuda.is_bf16_supported())
    info["amp"] = bool(amp and info["bf16"])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    return info


@contextlib.contextmanager
def autocast(device: torch.device, enabled: bool):
    """bf16 autocast on CUDA; a no-op everywhere else."""
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def loader_kwargs(device: torch.device, workers: int = 4) -> dict:
    """DataLoader flags that overlap host->device copies with compute."""
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "prefetch_factor": 4 if workers > 0 else None,
        "drop_last": False,
    }


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def maybe_compile(model, enabled: bool, *, label: str = "model"):
    """torch.compile when asked, with a readable fallback."""
    if not enabled:
        return model
    try:
        return torch.compile(model, dynamic=False)
    except Exception as e:  # noqa: BLE001
        print(f"[perf] torch.compile({label}) unavailable: {e}")
        return model


def describe(info: dict) -> str:
    return (f"amp={'bf16' if info['amp'] else 'off'}  tf32={info['tf32']}  "
            f"cudnn_benchmark={info['cudnn_benchmark']}")
