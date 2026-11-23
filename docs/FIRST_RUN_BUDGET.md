# First run under $1: where the time actually went

Measured on this machine's RTX 3080, production shape (32 chars, 64x1024, all five losses).
The short version: the first run was never a GPU problem. **One Python loop was 40% of the step.**

## 1. Profile first: the loop, not the model

Per-step timings, batch 16:

| Component | ms |
|---|---|
| **verifier group assembly (Python loop)** | **143.6** |
| denoiser forward | 11.2 |
| renderer + image loss | 6.4 |
| join-logit gather (also a loop) | 5.0 |
| style encoder | 39.5 |
| verifier CE, writer head | ~2 |

`build_char_groups` looped over `batch x characters` (16x32 = 512 iterations), each doing boolean
masking and indexing on GPU tensors. That is ~10 kernel launches per character: pure launch
overhead for negligible arithmetic. It cost more than the denoiser that does the actual work.

**Fix:** vectorised, using the invariant that tokens are emitted in character order, so
`(sample, character)` keys form contiguous runs — `unique_consecutive` + `repeat_interleave` gives
the within-group rank with no loop (`cent/models/groups.py`).

| | before | after |
|---|---|---|
| group assembly | 103–143 ms | **1.14 ms** |
| verifier CE (loss value) | 8.085438 | **8.085438** (identical, diff 0.0) |

## 2. Measured result

| Config | s/step | lines/h |
|---|---|---|
| batch 16, before | 0.2638 | 218k |
| batch 16, after | **0.0510** | 1.13M |
| batch 32, after | 0.0866 | 1.33M |
| **batch 64, after** | **0.1608** | **1.43M** |
| batch 128 | OOM on a 10.7 GB card (5.33 GB at 64) | — |

**5.2x faster**, no change to the model, the loss, or the maths.

## 3. AMP is measurably useless here (negative result)

You suggested CUDA AMP. I implemented it (`--amp bf16|fp16`, autocast on the three learned
forwards) and measured:

| Precision | s/step (B=16) |
|---|---|
| fp32 | **0.0509** |
| bf16 autocast | 0.0514 |
| fp16 autocast | 0.0515 |

Autocast is *slightly slower*: the tensors are small, so the cast overhead exceeds the tensor-core
gain, and the workload is launch-bound rather than FLOP-bound. This is the expected outcome for a
5.7 M model and it is why the 3090 beats newer cards on cost per unit work (§GPU_COST_ANALYSIS).
AMP is implemented and left available for when the model grows (a pixel-space variant would
benefit); it is off by default because it does not help.

## 4. The new bottleneck is the dataloader

At 1.43M lines/h the GPU consumes ~400 lines/s. `LineTokenDataset.__getitem__` runs one
`fit_stroke` (least-squares Bézier fit) per stroke, measured at **19.6 us**:

| strokes/line | CPU cost | throughput per worker |
|---|---|---|
| 250 | 4.9 ms | 204 lines/s |
| 350 | 6.9 ms | 146 lines/s |
| 470 | 9.2 ms | 109 lines/s |

With 8 workers that is ~800–1600 lines/s, so it *can* keep up — but it is now the same order of
magnitude as the GPU, and any worker count below 4 makes the GPU wait.

**Fix, implemented: pre-tokenise once.** `cent/cache_lines.py` walks the split once, writes flat
fp16 arrays, and `CachedLineDataset` then does memmap reads + `torch.from_numpy` with no
per-sample Python. Verified equivalent to the live dataset within fp16 quantisation (max payload
difference 2.3e-04; images exact to 1/255) and ~5x faster per sample on the fixture.

```bash
python -m cent.cache_lines --canonical data/canonical_lines/v1 --split train \
    --lexicon data/lexicon/graphics.txt --out data/cache/train --paired-only
python -m cent.cache_lines --canonical data/canonical_lines/v1 --split val  --out data/cache/val
```

Size: ~40k lines x 415 tokens x 8 floats in fp16 ≈ 530 MB per split. Trivial against a 50 GB disk.

## 5. What a run now costs

Batch 64, 38,800 training lines = 607 steps/epoch.

| | 3080 (measured) | 3090 (est. 1.15x) |
|---|---|---|
| per epoch | 98 s | **~85 s (1.4 min)** |
| 30 epochs — a first look | 49 min | ~43 min (**$0.09**) |
| 50 epochs | 1.4 h | ~1.2 h (**$0.14**) |
| 100 epochs | 2.7 h | ~2.4 h (**$0.29**) |
| **200 epochs — the planned main run** | 5.4 h | **~4.7 h ($0.57)** |

**The full 200-epoch main run now fits inside the $1 budget**, with room for ~2x dataloader
inefficiency. The earlier estimate of $17 was 30x too high because it was taken before the
profile; the GPU was mostly waiting on Python.

Add validations (DDIM-50, 3,000 lines ≈ 13 min each on the old speed; now proportionally less) and
checkpointing, and a complete first campaign stays **under $1**.

## 6. Recommended first run

Do not spend the $1 on the full campaign yet. Spend 9 cents learning whether the method works:

| Stage | Time on 3090 | Cost | Question it answers |
|---|---|---|---|
| 1. 30 epochs, 1 seed | ~43 min | $0.09 | do the losses move; does `join` fall below its prior 0.6474 nats; do the contact sheets show strokes rather than blobs |
| 2. inspect + decide | — | — | if garbage: stop, fix, re-run at this price |
| 3. 100 epochs, 1 seed | ~2.4 h | $0.29 | is the validation curve still improving; first real join-rate number |
| 4. 200 epochs, 3 seeds | ~14 h | $1.7 | the paper's headline result + seeds |

Budget one 3090-hour ($0.12) for setup, dataset build and pre-flight, and the whole path above is
**under $2**, with the first decisive answer at **9 cents**.

## 7. Other levers, and why most are not worth it

| Lever | Verdict |
|---|---|
| AMP (bf16/fp16) | measured: **no gain** |
| canvas 1024 → 512 | measured 0.264 → 0.257 s before the fix: **~3%**, and it weakens the line fidelity. Not worth it |
| fewer reference lines | the style encoder is now the largest single component (39.5 ms); k=5 → 3 would save ~16 ms. Changes the protocol, so only as an ablation |
| smaller model (dim 192, depth 3) | ~1.3x, but changes what is being tested. Not for the first run |
| `torch.compile` | plausible 1.2–1.5x, but recompiles on our variable sequence lengths and adds a failure mode before we know the method works. Revisit after a working run |
| pre-tokenisation | **implemented**; removes the loader ceiling |
| larger batch | free throughput (1.13M → 1.43M lines/h from B=16 to B=64), at 5.3 GB. Use B=64 |

The ordering matters: one loop cost 40% of the step, while the GPU-side "optimisations" everyone
reaches for first are worth single digits. Profile before optimising.
