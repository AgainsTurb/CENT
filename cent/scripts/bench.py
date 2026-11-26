"""Measure real step times and numerics, then project stage durations.

Runs the actual model shapes (d=128, z=64, DiT depth 3) on the local GPU, with
and without bf16 autocast, and reports:
  * ms/step for the VAE stage (batch 64, T=512)
  * ms/step for the DiT stage (batch 16, T_lat=64, ~26 boundaries)
  * the fp32-vs-bf16 loss delta, so "AMP does not change the numerics" is measured
  * projected hours for Stages 2 and 4 on a reference 3090

Local 3080 (cc 8.6) and the rented 3090 (cc 8.6) are the same architecture; the
scaling factor below accounts for the 3090's higher bf16 throughput.

    python scripts/bench.py --target-speedup 1.15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.losses import boundary_loss, masked_mean, vae_losses  # noqa: E402
from cent.models.boundary_dit import BoundaryDiT  # noqa: E402
from cent.models.inkvae import InkVAE  # noqa: E402
from cent.models.text_encoder import TextEncoder  # noqa: E402
from cent.pipeline import perf  # noqa: E402


def _time(fn, warmup=3, iters=10) -> float:
    """Median-ish ms/step with backward."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def bench_vae(device, *, amp: bool, batch: int = 64, T: int = 512):
    model = InkVAE(d=128, z_dim=64, stride=8, n_mix=20,
                   n_chars=2655, n_writers=336).to(device).train()
    dxy = torch.randn(batch, T, 2, device=device) * 0.02
    pen = torch.zeros(batch, T, dtype=torch.long, device=device)
    pen[:, ::7] = 1
    mask = torch.ones(batch, T, dtype=torch.bool, device=device)
    seq = torch.randint(1, 2655, (batch, 26), device=device)
    cmask = torch.ones(batch, 26, dtype=torch.bool, device=device)
    widx = torch.randint(0, 336, (batch,), device=device)

    def step():
        with perf.autocast(device, amp):
            x = torch.cat([dxy, torch.nn.functional.one_hot(pen, 4).float()], -1)
            out = model(x, pen, mask, sample=True)
        losses = vae_losses(out, dxy, pen, seq, cmask, widx, mask)
        model.zero_grad(set_to_none=True)
        losses["total"].backward()

    def loss_only():
        with perf.autocast(device, amp):
            x = torch.cat([dxy, torch.nn.functional.one_hot(pen, 4).float()], -1)
            out = model(x, pen, mask, sample=False)
        return float(vae_losses(out, dxy, pen, seq, cmask, widx, mask)["total"].detach())

    ms = _time(step)
    return model, ms, loss_only()


def bench_dit(device, *, amp: bool, batch: int = 16, T_lat: int = 64,
              d: int = 128, z: int = 64):
    text = TextEncoder(n_chars=2655, d=d).to(device).train()
    dit = BoundaryDiT(z_dim=z, d=d, depth=3, n_heads=4, n_chars=2655).to(device).train()
    z0 = torch.randn(batch, T_lat, z, device=device)
    fchar = torch.arange(T_lat, device=device).repeat(batch, 1) // 2
    fmask = torch.ones(batch, T_lat, dtype=torch.bool, device=device)
    labels = torch.randint(0, 2, (batch, T_lat - 1), device=device).float()
    lmask = torch.zeros(batch, T_lat - 1, dtype=torch.bool, device=device)
    lmask[:, 1:26] = True
    seq = torch.randint(1, 2655, (batch, 26), device=device)
    cmask = torch.ones(batch, 26, dtype=torch.bool, device=device)
    style = torch.randn(batch, d, device=device)
    t = torch.randint(0, 1000, (batch,), device=device)
    noise = torch.randn_like(z0)
    z_t = z0 + noise * 0.1

    def step():
        with perf.autocast(device, amp):
            ctx = text(seq, cmask)
            out = dit(z_t, t, frame_char=fchar, frame_mask=fmask,
                      char_ctx=ctx, style=style)
        l_diff = masked_mean((out["x0"].float() - z0) ** 2, fmask)
        l_bnd = boundary_loss(out["bnd_logits"].float(), labels, lmask)
        (l_diff + l_bnd).backward()

    def loss_only():
        with perf.autocast(device, amp):
            ctx = text(seq, cmask)
            out = dit(z_t, t, frame_char=fchar, frame_mask=fmask,
                      char_ctx=ctx, style=style)
        return float(masked_mean((out["x0"].float() - z0) ** 2, fmask).detach())

    ms = _time(step)
    return ms, loss_only()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-speedup", type=float, default=1.15,
                    help="3090 vs 3080 bf16 throughput ratio")
    ap.add_argument("--train-lines", type=int, default=16466,
                    help="training lines after the 80/20 writer split")
    ap.add_argument("--vae-batch", type=int, default=64)
    ap.add_argument("--dit-batch", type=int, default=16)
    ap.add_argument("--epochs-vae", type=int, default=30)
    ap.add_argument("--steps-dit", type=int, default=60000)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA — cannot benchmark")
        return 2
    device = torch.device("cuda")
    info = perf.setup_perf(device, amp=True)
    print(f"device: {torch.cuda.get_device_name(0)}  |  {perf.describe(info)}\n")

    print("== InkVAE step (stage 2) ==")
    _, ms_fp32, loss32 = bench_vae(device, amp=False)
    _, ms_bf16, loss16 = bench_vae(device, amp=True)
    print(f"  fp32 : {ms_fp32:7.1f} ms/step")
    print(f"  bf16 : {ms_bf16:7.1f} ms/step   speedup x{ms_fp32/ms_bf16:.2f}")
    rel = abs(loss32 - loss16) / max(abs(loss32), 1e-9)
    print(f"  loss fp32={loss32:.4f}  bf16={loss16:.4f}  rel.diff={rel*100:.3f}%")

    print("\n== BoundaryDiT step (stage 4, VAE frozen + latents cached) ==")
    ms_d32, l32 = bench_dit(device, amp=False, batch=a.dit_batch)
    ms_d16, l16 = bench_dit(device, amp=True, batch=a.dit_batch)
    print(f"  fp32 : {ms_d32:7.1f} ms/step")
    print(f"  bf16 : {ms_d16:7.1f} ms/step   speedup x{ms_d32/ms_d16:.2f}")
    rel_d = abs(l32 - l16) / max(abs(l32), 1e-9)
    print(f"  diff loss fp32={l32:.4f}  bf16={l16:.4f}  rel.diff={rel_d*100:.3f}%")

    k = 1.0 / a.target_speedup
    vae_ms = ms_bf16 * k
    dit_ms = ms_d16 * k
    steps_vae = (a.train_lines // a.vae_batch + 1) * a.epochs_vae
    vae_h = steps_vae * vae_ms / 1000 / 3600
    dit_h = a.steps_dit * dit_ms / 1000 / 3600

    print("\n== projected wall-clock on the rented 3090 ==")
    print(f"  (3080 measurements x {k:.2f} for the 3090's higher bf16 rate)")
    print(f"  stage 2  : {steps_vae:6d} steps x {vae_ms:6.1f} ms = {vae_h*60:6.1f} min")
    print(f"  stage 4  : {a.steps_dit:6d} steps x {dit_ms:6.1f} ms = {dit_h*60:6.1f} min")
    print(f"  stage 3  : alignment is forward-only, ~{a.train_lines*0.0015:.0f} min (estimate)")
    total_h = vae_h + dit_h + 0.1
    print(f"  TOTAL    : {total_h:.2f} h  ->  ${total_h*0.1367:.2f} of the $3.11 credit")
    print(f"\n  at 60k DiT steps the whole run costs about "
          f"${total_h*0.1367:.2f}; the budget allows {3.106/0.1367:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
