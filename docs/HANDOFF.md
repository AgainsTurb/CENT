# CENT — handoff state

_Last updated: 2026-11-22, at instance teardown._

The GPU instance is **destroyed**. Nothing below depends on it; everything needed to resume is on
your Drive and in this repository.

## Where things are

| what | where | size |
|---|---|---|
| canonical dataset (391 MB) | `gd:CENT/datasets/canonical_lines_v1/` and `.tar.gz` | 391 MB |
| pre-tokenised cache | `gd:CENT/datasets/cache/` | 3.3 GB |
| ICDAR-2013 lines | `gd:CENT/datasets/icdar13/` | 0.16 GB |
| **all 14 runs** (checkpoints, logs, configs) + code snapshot | `gd:CENT/artifacts/cent_artifacts.tar.gz` | 297,509,293 B |
| checksums | `gd:CENT/CENT_SHA256.txt` | — |
| code, docs, results | `A:\Papers\CENT` (this repo) | — |

Drive total: **7.167 GB**. The artifacts bundle's size matches the local file exactly; Google Drive
does not expose SHA256, so the size check is the verification that was possible.

To restore on a fresh box:

```bash
# rclone config must exist at ~/.config/rclone/rclone.conf (remote name: gd)
rclone copy gd:CENT/datasets/canonical_lines_v1.tar.gz /workspace/cent/data/
tar xzf canonical_lines_v1.tar.gz -C /workspace/cent/data/
rclone copy gd:CENT/datasets/cache /workspace/cent/data/cache
rclone copy gd:CENT/artifacts/cent_artifacts.tar.gz /workspace/
tar xzf /workspace/cent_artifacts.tar.gz -C /workspace/
python tools/sync_box.py make     # then upload + verify on the box before any run
```

## What is proven

- **Data pipeline.** Native `.dgrl`/`.wptt` parsers (the earlier versions were wrong: endianness,
  magic, illustration-length arithmetic, 8-bit vs 1-bpp). 144 tests pass. 815 writers, 41,710
  stroke-bearing lines, 30,302 usable train lines, 2,680 classes.
- **Representation.** A line is an ordered stroke sequence; each stroke is a cubic Bézier
  (`PAYLOAD=8`). The GT render of a line's payloads is **legible handwriting**, and the training
  target is now self-consistent with the loss's rasteriser (**IoU = 1.0** exactly).
- **Style conditioning.** The encoder used to read reference images whose pixels are 98.5% paper and
  returned the same vector for every writer (0.05% differentiation, swap effect 0.0%). Reading ink
  coverage fixed it: **42% differentiation, swap 2.2%, writer loss 5.09 -> 1.11**.
- **Text conditioning.** Transcript swap moves the denoiser output by 15.1%.
- **Continuity supervision.** Geometric loss on predicted endpoints (scale-normalised), not a
  classifier. Sampled join rate reaches **0.1404 against a ground-truth 0.1333**.
- **Sampler + evaluation.** DDIM over the stroke sequence, contact sheets, join-rate and gap metrics.
- **Reproducibility.** `tools/sync_box.py` tarballs the tree, hashes it *and* asserts feature markers
  (`w_img`, splat target, `local_window`, ...), because a stale bundle once passed a pure hash check.

## What does not work, and what was ruled out

The generator produces **smears, not strokes**, in every configuration tried. Falsified in order:

| hypothesis | test | result |
|---|---|---|
| parsers / canonical build | native parsers, real-file probes, 144 tests | sound |
| inverted image convention | ink/paper mismatch | fixed |
| timestep coverage | `T=1000 -> 200` (81 visits/level) | not the limit |
| style conditioning | paper-pixel pooling collapse | fixed, still no strokes |
| text-stroke alignment | position + budget scalars + local character window | implemented, no visible effect |
| image supervision | scan is a *different writing session*; self-consistent render, IoU 1.0 | correct now, still flat |
| data volume | 8,631 -> 30,302 lines | **falsified** (diff 0.1093 -> 0.1005, image unchanged) |

**Key negative result worth keeping:** `.dgrl` scans and `.wptt` trajectories are two separate
writing sessions (casual pen + scan vs Anoto pen on dot paper), paired by writer and text only.
A scan is therefore **not** a valid pixel target for a trajectory; it is valid only as a *style*
reference. Any future design must respect this.

## The decision that was pending

CENT's denoiser is ~5.5M parameters; the baselines (DiffBrush, TGC-Diff, One-DM) are 100M+ SD UNets.

- **A. Scale the diffusion model** — dim 512 / depth 8 (~25M), more epochs. ~$1.5, uncertain.
- **B. Switch to autoregressive stroke generation** — stroke *k+1* conditioned on stroke *k*'s
  endpoint. Trains by teacher forcing (no noise schedule, no timestep-coverage problem) and makes
  continuity **structural**, which is the paper's actual thesis. Recommended.

## Spend

Approximately **$3.5** of GPU time across ten training runs and the data work. Instance destroyed, so
billing has stopped.

## v3 revision (written, not trained)

The generator was rebuilt as an autoregressive stroke model after ten diffusion runs failed to
produce a legible line. Entry points:

| file | role |
|---|---|
| `docs/ARCHITECTURE_V3.md` | module/responsibility spec, boundary contract, staged gates |
| `cent/models/v3.py` | M1-M8 |
| `cent/data_v3.py` | per-char grouping, relative payloads, boxes, joins, pair priors |
| `cent/train_v3.py` | staged trainer (`--stage s1|s2|s3`, `--limit`, `--dry-run`) |
| `benchmark/sample_v3.py` | AR decoder with the boundary contract |
| `tools/build_pair_prior.py` | per-pair join prior for M4 |

Status: all files parse; **nothing has been executed**, no GPU rented, by instruction. The S1
gate (overfit one line) is the first thing to run before any campaign.

The earlier v1/v2 material below remains accurate for what was built and measured; §"The
decision that was pending" is now resolved in favour of option B (autoregressive).
