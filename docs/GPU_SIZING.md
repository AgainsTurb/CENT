# GPU box sizing, training budget, and visual verification

> **SUPERSEDED IN PART (2026-11-22, later the same day).** The step times below were measured
> before a profile found that a Python loop in the verifier group assembly cost 40% of every
> step. After vectorising it the step is **5.2x faster** (0.264 -> 0.051 s at batch 16;
> 1.43M lines/h at batch 64), so the epoch and campaign estimates here are pessimistic by that
> factor. Read `docs/FIRST_RUN_BUDGET.md` for the current numbers and
> `docs/GPU_COST_ANALYSIS.md` for rental cost. The VRAM, storage and verification sections of
> this document are still valid.

Measured on this machine (RTX 3080, 10.7 GB) with `benchmark/bench_step.py`, at the real tensor
shapes: 32-character lines, 64x1024 canvas, every loss active. Numbers below are **measurements**
unless marked as extrapolation.

## 1. What the model actually costs

| Component | Size |
|---|---|
| Line denoiser | 4.25 M params |
| Style encoder | 0.72 M |
| Per-character verifier | 0.46 M |
| Writer head | 0.26 M |
| **Total** | **5.69 M params** (~23 MB fp32; ~70 MB with Adam states) |

This is a small model. Compute is not the constraint; the offline image loss was.

### The one design change this sizing forced

The distance rasteriser materialises `[B, T, 16, H, W]` to compare every stroke segment against
every pixel. At character scale (40 strokes, 64x64) that is ~10 MB. At **line** scale
(~470 strokes, 64x1024) it is **~316 GB** — it cannot run. So the line model uses a **splatting
rasteriser** (`render_strokes_splat`): evaluate each Bezier at 9 points, bilinearly scatter into
the canvas, light separable blur. Cost is `O(points)`, memory is negligible, and gradients still
reach the control points. Measured within the step times below.

## 2. Measured step time and VRAM (RTX 3080, 10.7 GB)

| Case | Tokens | Batch | Step | Peak VRAM | Throughput |
|---|---|---|---|---|---|
| mean line (32 chars x 11 strokes) | 415 | 16 | **0.264 s** | **1.42 GB** | 218k lines/h |
| p90 line (32 chars x 17 strokes) | 607 | 16 | 0.267 s | 1.56 GB | 216k lines/h |
| mean line | 415 | 32 | 0.510 s | 2.71 GB | 226k lines/h |
| mean line | 415 | 64 | 1.071 s | 5.33 GB | 215k lines/h |
| mean line, canvas 512 | 415 | 16 | 0.257 s | 0.93 GB | 224k lines/h |

Two things to notice. Throughput is **flat across batch sizes** (~215-226k lines/h), so this is
compute-bound on the small model, not memory-bound. And **canvas 1024 costs almost nothing versus
512** (0.264 vs 0.257 s) because the splat renderer is cheap — so there is no reason to shrink the
canvas.

## 3. Steps and epochs

Training set for the first campaign (paired lines, 756 train writers) ≈ **38,800 lines**.

```
steps_per_epoch = 38,800 / batch
epoch_time      = steps_per_epoch x sec_per_step
```

| Batch | Steps/epoch | Epoch time (3080) | 100 epochs | 200 epochs |
|---|---|---|---|---|
| 16 | 2,425 | **10.7 min** | 17.8 h | 35.6 h |
| 32 | 1,213 | 10.3 min | 17.2 h | 34.4 h |
| 64 | 607 | 10.8 min | 18.0 h | 36.0 h |

**Recommendation: 200 epochs at batch 32 (~242,600 steps), checkpoint every epoch, validate every
5 epochs, stop early if the validation join-rate error has not improved for 20 epochs.** The model
is small enough that under-training is the likelier failure than over-training; 242k steps is
~25 passes over every line per parameter-equivalent, which is modest for a 5.7 M model. If the
validation curve is still improving at 200 epochs, continue to 400.

## 4. What to rent

| Need | Minimum | Comfortable |
|---|---|---|
| **CENT training** (B=16-64) | 8 GB VRAM | **16-24 GB** |
| **Baseline campaign** (SD-UNet at 64x1024, DiffInk DiT 2.8 GB weights) | 24 GB | **40-48 GB** |
| System RAM | 16 GB | **32 GB** |
| Storage | 20 GB | **50-100 GB SSD** |
| CPU | 4 cores | **8-16 cores** (dataloader workers) |
| Disk speed | any | local NVMe; the loader reads ~50k small PNGs per epoch |

**Recommendation: one 40 GB-class GPU (A100 40GB, L40S 48GB, or A6000 48GB) for the whole
campaign.** CENT training uses ~3 GB of it; the headroom is for the baselines, which are the real
VRAM consumers. If cost dominates, a 24 GB RTX 4090 runs CENT comfortably (5.3 GB at B=64) and
most baselines at reduced batch; rent the bigger card only for the baseline sweep.

Storage budget:

| Item | Size |
|---|---|
| Raw archives (CASIA lines + ICDAR-2013) | 2.62 GB |
| Unpacked pages | ~3 GB (estimate) |
| Canonical lines (52,230 PNGs at 64x1024 + trajectories) | 0.3-2 GB |
| Baseline weights (DiffBrush 1.09 + TGC-Diff 2.82 + DiffInk 2.96 + SDT/MetaScript 0.3) | ~7.2 GB |
| HTR model | 65 MB |
| Checkpoints (5.7 M params, keep 3) | <1 GB |
| **Total** | **~15 GB**, budget 50 GB |

## 5. Extrapolation to other GPUs

Do **not** trust a table here — run `benchmark/bench_step.py` on the rented box for 10 steps and
read `sec_per_step`. That takes under a minute and replaces every guess. As a starting point only,
scaling from the measured 3080 (which is compute-bound on small tensors, so large-GPU advantage is
muted):

| GPU | Expected speed vs 3080 | 200 epochs (est.) |
|---|---|---|
| RTX 3080 / 3090 | 1.0x | 35 h |
| RTX 4090 | 1.5-2.0x | 18-24 h |
| A100 40GB | 1.5-2.5x | 14-24 h |
| H100 80GB | 2-3x | 12-18 h |

These are **estimates**, not measurements. For a 5.7 M model the ceiling is launch overhead and
dataloader latency, not FLOPs; a bigger card will not scale linearly, and the way to find out is
the 10-step measurement.

**Cost:** hours x the provider's hourly rate. At an indicative A100 40GB rate the CENT run is
single-digit dollars; the baseline sweep will dominate the bill because those are 1-3 GB models at
line resolution, not because of CENT.

## 6. Before the run: a 30-minute pre-flight

1. `python -m benchmark.bench_step --batch 32 --chars 32 --strokes 11 --canvas-w 1024 --gpu --steps 10`
   → record `sec_per_step` and `peak_vram_gb` in the ledger.
2. `python -m common.probe_lines dir data/raw/dgrl --kind dgrl --limit 10` and the same for `wptt`.
3. `python -c "from common.canonical_lines import build_lines; ..."` → record `counts`,
   `paired_lines`, `lines_with_strokes`, and **the join rate per writer**.
4. `python -m cent.train_lines --config cent/configs/casia_r1.json --dry-run` → all five losses
   print with finite values.
5. 50 real steps → confirm the losses move and that `join` is heading below its prior
   (`-r log r - (1-r) log(1-r)`; at r=0.35 that is **0.6474 nats**).

## 7. How to check results visually

Run after the first epoch, then at each validation:

```bash
python -m benchmark.visualise_run --config runs/casia_r1/config.json \
    --ckpt runs/casia_r1/ckpt_last.pt --split val --n 6 --out runs/casia_r1/sheet_val.png
```

Each sample block is: **label (text + predicted join count)**, **ground truth**, **generated image**,
**generated trajectory**, then the **k reference lines** the style came from. Read them in that
order; the trajectory row is what tells you whether a plausible-looking image came from a coherent
pen path or from pixels that happen to be dark.

What to check, in priority order:

1. **Joins.** At a joined boundary the ink is continuous and the direction is smooth; at a lift
   there is a clean gap. Look for a kink — continuous ink that turns sharply. That is the failure
   our parameterisation is supposed to make impossible, so a kink is a real bug, not a quality
   complaint.
2. **Kerning and baseline.** Uneven gaps or a wandering baseline mean the layout tokens are not
   carrying the writer's spacing.
3. **Per-character legibility.** The `legib` loss should be falling; if characters are wrong but
   images look plausible, the image loss is dominating and `lambda_image` needs revisiting.
4. **Style match against the reference row.** Slant, stroke weight, character size. If the
   generated hand looks like nobody in particular, `z_g` is not reaching the denoiser.
5. **Join control.** Sample the same text with joins forced off, forced to the writer's rate, and
   per-boundary overridden. A controllable variable should change the line; if it does not, the
   claim is empty.

Failure signatures worth naming:

| What you see | Likely cause |
|---|---|
| solid black/dark blob | image loss dominating early, or renderer `sigma` too large |
| ghost-thin strokes | splat `sigma` too small; increase to ~1.5 px |
| straight horizontal streaks | layout tokens collapsed — check their normalisation |
| characters correct but no joins | join BCE too weak; check its prior and raise `lambda` on the join term |
| joins present but spiky | control points not anchored; verify A9 configuration |

## 8. Honest unknowns at sizing time

- **Dataloader throughput is unmeasured on real data.** The step benchmark uses synthetic tensors.
  If the loader cannot supply ~60 lines/s with 8 workers, the GPU waits and the epoch estimate
  doubles. Measure one epoch's wall clock before trusting the table.
- **The "~3 GB unpacked" figure and the canonical size are estimates**, not measurements; the real
  numbers come from the first build.
- **Extrapolations across GPUs are estimates** and are labelled as such.
- **Convergence is unknown.** 200 epochs is a judgement based on model size and dataset size, not
  on an observed training curve. The validation curve decides, and the run is resumable per epoch
  precisely so it can be extended.
