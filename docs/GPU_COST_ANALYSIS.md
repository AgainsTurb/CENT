# GPU rental: cost per unit of work, not price per hour

Prices read 2026-11-22 from the Vast.ai API (live offers, this account) and public trackers
[RunPod pricing](https://www.runpod.io/pricing),
[grabgpu](https://grabgpu.com/api/gpus.md),
[unitedcompute](https://unitedcompute.ai/gpu-price-tracker),
[gpufinder](https://gpufinder.dev/gpu/rtx-3090-vs-rtx-4090),
[computescreener](https://computescreener.com/gpus).
Market prices move hourly; treat the ranking as stable and the dollars as indicative.

## 0. The premise needs correcting

"RTX 3090 is almost twice the price of RTX 3080" is true **on RunPod Secure Cloud** ($0.50 vs
n/a) but false on the marketplace. Live on Vast.ai today:

| Card | Cheapest reliable offer seen | Notes |
|---|---|---|
| RTX 3080 10 GB | **$0.109/h** | verified, reliability 0.995 |
| RTX 3090 24 GB | **$0.122/h** | reliability 0.996; other hosts $0.146–0.148 |
| RTX 4090 24 GB | **$0.377–0.402/h** | the $0.377 one is unverified; verified is $0.402 |
| RTX 5090 32 GB | **$0.334/h** | reliability 0.996, but only 21 GB disk |
| L40S 45 GB | $0.535/h | |
| A100 80 GB (SXM) | ~$0.52–0.84/h | tracker range |

**A 3090 is ~1.1x a 3080 on the marketplace, not 2x.** The 2x figure comes from comparing a
Secure Cloud instance with a Community one — RunPod itself lists the 3090 at $0.50 secure vs
$0.22 community, a 2.3x spread on the *same card*. If you are paying 2x for a 3090, the fix is
the provider tier, not the GPU. That single change is worth more than any model choice below.

## 1. Price per hour is the wrong metric

What matters is the **cost of a fixed amount of work**. Define one unit = one hour of measured
RTX 3080 throughput (our anchor: 0.264 s/step at batch 16, 64x1024, all losses).

```
cost per unit of work = price_per_hour / speed_relative_to_3080
```

Speed factors below are **estimates** for our workload (a 5.7 M model, many small CNN/matmul
kernels, launch-overhead sensitive), derived from FP32 throughput and Vast's `dlperf`, and
deliberately conservative. The authoritative number is `benchmark/bench_step.py` on the card.

| Card | $/h | Speed vs 3080 (est.) | **$ per 3080-hour** | Rank |
|---|---|---|---|---|
| **RTX 3090** | 0.122 | 1.2–1.5 | **0.081–0.102** | **1** |
| RTX 5090 | 0.334 | 2.2–3.2 | 0.104–0.152 | 2 |
| RTX 3080 | 0.109 | 1.0 | 0.109 | 3 |
| RTX 4090 | 0.390 | 1.8–2.5 | 0.156–0.216 | 4 |
| L40S | 0.535 | 1.6–2.2 | 0.243–0.334 | 5 |
| A100 80 GB | 0.840 | 1.5–2.0 | 0.420–0.560 | 6 |

### The break-even view (robust to my uncertainty)

For card X to beat the 3090 on cost, it must be faster than the 3090 by at least `price_X / 0.122`:

| Card | Must be faster than a 3090 by | Actually is (est.) | Verdict |
|---|---|---|---|
| RTX 4090 | **3.2x** | ~1.6x | loses on cost by 2x |
| RTX 5090 | **2.7x** | ~1.8–2.4x | costs 15–35% more, but **~2x faster** |
| L40S | 4.4x | ~1.4x | never for this workload |
| A100 | 6.9x | ~1.3x | a large-model card; wasted on 5.7 M params |

**The 4090 is the one to avoid**: on Vast it costs 3.2x the 3090 for ~1.6x the speed. It is
dominated by the 5090, which is cheaper per hour *and* faster.

## 2. What the campaign actually costs

Campaign = ~188 units of 3080-work (main 200 epochs + validations + 3 seeds + 11 ablations).

| Card | Wall clock | Cost | Comment |
|---|---|---|---|
| **RTX 3090** | **139 h (5.8 d)** | **$17** | cheapest; no driver/CUDA surprises |
| **RTX 5090** | **70 h (2.9 d)** | **$23** | half the wall clock for +$6 |
| RTX 3080 | 188 h (7.8 d) | $20 | the card I measured on; no reason to rent it |
| RTX 4090 | 87 h (3.6 d) | $34 | pays more and waits longer than the 5090 |
| L40S | 99 h (4.1 d) | $53 | |
| A100 80 GB | 107 h (4.5 d) | $90 | |

With interruptible/bid instances (our run checkpoints every epoch, so a preemption costs at most
one epoch): 3090 bids seen at **$0.093/h**, 5090 at **$0.25/h** → the campaign drops to **$13** and
**$18** respectively.

## 3. Recommendation

**Rent one RTX 3090 at ~$0.12/h** unless wall clock matters. It is the cheapest per unit of work,
it is Ampere (sm_86) so the existing CUDA 12.6 / PyTorch build runs unmodified, and the campaign
finishes for about **$17 in 5.8 days**. Enable spot/bid and it is **$13**.

**If ~6 days of waiting is the real cost, rent one RTX 5090** at ~$0.33–0.42/h: **$20–23 and ~3
days**. You pay roughly 20–35% more money for half the calendar time. That is the right trade if
anyone is waiting on the result.

**Do not rent a 4090 for this.** It is the worst of both on the marketplace.

## 4. Two caveats that decide whether the 5090 is usable

1. **CUDA 12.8 is required for Blackwell.** sm_120 (5090) is not supported by CUDA 12.6/12.7
   PyTorch builds. The instance needs a `torch` built for cu128 (`pip install torch --index-url
   https://download.pytorch.org/whl/cu128`) and the 12.8+ driver. Budget 15 minutes of setup and
   verify with `python -c "import torch;print(torch.cuda.get_device_capability())"` → expect
   `(12, 0)`. Our local build (cu126) would fall back to CPU or fail on a 5090.
2. **Disk.** The cheapest 5090 offer had **21 GB**. The campaign needs ~15 GB (3 GB raw + ~3 GB
   unpacked + 1–2 GB canonical + ~1 GB checkpoints), so it fits — but the baseline weights
   (7.2 GB) do not fit alongside unless raw archives are deleted after the canonical build.
   Prefer a 5090 offer with ≥50 GB, or do the baseline sweep on the same card after cleaning up.

Also note the 5090/3090 offers I found are marked *deverified* by Vast (host not re-verified
recently) but with reliability ≥0.995. For a resumable job that checkpoints per epoch, that is an
acceptable risk; for a one-shot 6-day run without checkpoints it would not be.

## 5. What not to conclude from this

- **Do not scale by TFLOPS.** FP16 marketing numbers (5090: 210 TFLOPS vs 3090: 71) predict a 3x
  speedup that a 5.7 M model will not see, because the workload is launch-bound. That is exactly
  why the 3090 wins on cost despite being the older card.
- **A100/L40S are not "better GPUs" here.** They are built for large models and fp16/bf16
  throughput; at 5.7 M params with fp32 tensors they leave most of the chip idle and cost 4–7x.
  They become right if CENT ever grows to a pixel-space model.
- **Do not rent 2 GPUs to halve the clock.** 2 x 3090 gives ~1.8x (DDP) for 2x the price: $25 and
  104 h, worse than one 5090 on both axes.
- **Prices are hourly-moving.** The ranking held across every tracker I read today, but confirm the
  live price before booking, and re-run `benchmark/bench_step.py` on the rented card rather than
  trusting the speed column above.
