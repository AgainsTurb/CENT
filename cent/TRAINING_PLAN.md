# CENT — datasets and training procedure

Two questions answered here, in order: **which data**, then **what to run, in what order**.
Facts are separated from estimates throughout: everything marked **[measured]** was
verified on disk in this project; **[estimate]** is a guess to be replaced by the real number.

---

# Part 1 — The datasets

## 1.1 The decision

| # | Dataset | Role | Size | Status |
|---|---|---|---|---|
| **D1** | **CASIA-OLHWDB 2.0** (WPTT) | **primary — lines + text + writers** | 1,678 train / 420 test pages · 420 writers · 20,573 lines · 540,009 chars · 26.4 chars/line **[measured]** | **on disk, parsed, verified** |
| **D2** | **CASIA-OLHWDB 1.0–1.2** (`.pot`) | **character-level pretraining + vocabulary coverage** | ~3.9 M isolated chars · 1,020 writers · 7,185 classes | not downloaded (same NLPR page) |
| **D3** | **Hand-annotated cursive line set** | **evaluation only** | target 300–500 lines, connection strokes marked | **to be built (contribution)** |
| D4 | SCUT-COUCH official `WORD8888` | upgrade: authoritative per-character segmentation | 8,888 words × writers | **locked** — application |
| D5 | COUCH09 (DOLPHIN copy) | optional unlabelled style/geometry pretraining | 145 writers × 1,000 words, stroke idx, **no text** **[measured]** | on disk (zipped) |
| ✗ | DOLPHIN preprocessed `.pkl` | — | loses stroke + transcript | **do not use** |
| ✗ | DCOH-120K / EvaHan / HIT-MW | — | no access / historical / no trajectories | out of scope for training |

**D1 alone is sufficient to run the whole pipeline.** D2 is strongly recommended
(it is what makes contribution (iii) meaningful, and it gives 7,185-class coverage
instead of OLHWDB2.0's 2,655). D3 is mandatory for the paper to be checkable.
D4 is an upgrade, not a blocker.

## 1.2 Why exactly these

- **D1** is the only corpus that has all three of {trajectory, transcript, writer id} at
  line granularity, unencrypted. It is also the corpus every baseline is measured on,
  so results are comparable by construction.
- **D2** exists for two reasons. First, coverage: a 2,655-class vocabulary cannot write
  arbitrary text; 7,185 can. Second, the paper's third contribution is *about* the
  char→line transfer, so character-level training data is not a convenience, it is part
  of the claim.
- **D3** exists because **no automatic proxy can validate a continuity claim** (§3 of
  `survey/dataset_verification.md`). 300–500 lines is enough for a credible human study
  and a per-writer continuity metric, and it is small enough to annotate in a week.
- **D5** is optional and unlabelled: no transcript, so it cannot train content
  conditioning. Keep it for self-supervised style pretraining only.

## 1.3 Splits

| Split | Writers | Pages | Source |
|---|---|---|---|
| train | 269 | ~1,342 **[estimate]** | 80 % of the released train writers |
| val | 67 | ~336 **[estimate]** | held out **by writer** from the released train split |
| test | 84 | 420 **[measured]** | released test split, untouched |
| eval-cursive | — | 300–500 lines | D3, never used for training |

The released splits were counted on disk: **train 1,678 pages / 336 writers, test 420
pages / 84 writers, with zero writer overlap** `[measured]`. The val split is carved out
of the released *train* writers so that model selection never touches the official test
pages, and writer-disjointness is preserved everywhere.

---

# Part 2 — Training procedure, step by step

The architecture (`InkVAE` → CTC → `BoundaryDiT`) is already written and smoke-tested.
The stages below are ordered so that **each stage produces the supervision the next one
needs**. That ordering is the important part: the boundary labels do not exist in any
corpus, so they are *derived* in Stage 3 from the model trained in Stage 2.

```
D2 chars ─► S1 InkVAE(chars) ─┐
                              ├─► S2 InkVAE(lines) ─► S3 boundary+join labels ─► S4 BoundaryDiT ─► S5 control ─► S6 eval
D1 lines ─────────────────────┘                                                    (freeze VAE)
```

## Stage 0 — data preparation (no gradients)

1. Parse D1 pages (`cent/data/format_wptt.py`), keep lines with ≥2 chars.
2. Normalise **per line**: divide by line height, keep the line's own aspect and
   spacing. Do **not** re-centre or rescale per character — that is precisely OLHWG's
   mistake and it destroys layout style.
3. Build the vocabulary from *training* transcripts only (test text unseen).
4. Pre-compute per sample: point sequence, `(dx, dy)`, 4-way pen state, stroke ids,
   transcript, writer, line id. Cache to `.npz`.
5. Gate: every page parses (`consumed == sample_len`), 26.4 chars/line ±1 **[measured]**.

## Stage 1 — InkVAE on characters (D2)

| | |
|---|---|
| Init | random |
| Data | D2 isolated characters, ~3.9 M |
| Objective | `L_gmm + 2·L_pen + 1e-3·L_kl + 1·L_ctc + 0.5·L_style` |
| Batch | 128–256, 1–2 epochs is plenty |
| Freeze | nothing |
| Purpose | a trajectory manifold and a char-decodable latent **before** lines |

**Gate G1:** CTC greedy decode of held-out characters ≥ 90 % exact-match **[estimate]**.
If not, the latent will not align on lines either — fix here, do not proceed.

## Stage 2 — InkVAE on lines (D1) + CTC calibration

| | |
|---|---|
| Init | from Stage 1 |
| Data | D1 train lines; val for monitoring |
| Objective | same form, `L_ctc` weighted by line length |
| Epochs | until val CTC stops improving |
| Purpose | **character-aligned latents on whole lines** — the prerequisite for boundaries |

**Gate G2 (critical):** CTC greedy decode on val lines, character accuracy ≥ 85 %
**[estimate]**. This single number decides whether the boundary plan works. If it is
near chance, the bottleneck is the latent resolution (raise `stride` 8→4) or the
vocabulary, not the DiT.

## Stage 3 — derive point-level boundaries and join labels (no gradients)

This is the stage that replaces the missing annotation.

1. For each training line, run **Viterbi CTC alignment** → a character id per point.
2. Boundary = the first point whose char id changes.
3. **Join label = `stroke_id[t] == stroke_id[t-1]`** at that boundary. Exact by
   definition: the pen was still down across the character change.
4. Discard boundaries with low alignment margin (keep the top ~90 % by posterior gap).
5. Report the statistics that no one has published: overall join rate, **per-writer join
   rate**, gap distribution, and the fraction of lines containing ≥1 join.

**Gate G3:** the join rate must be plausible and non-zero. Expect single-digit percent
overall with a wide per-writer spread. **If it comes out ≈ 0, stop and re-scope** — the
corpus has no cursive, and the honest move is the hand-annotated set plus cross-script
transfer, not a fabricated claim.

This stage doubles as the paper's first empirical contribution: *the measured cursive
rate of the standard Chinese benchmark*.

## Stage 4 — BoundaryDiT (the main model)

| | |
|---|---|
| Init | random |
| Data | cached VAE latents + transcripts + Stage-3 boundary/join labels + writer |
| VAE | **frozen** (optionally fine-tuned later at 10× lower LR) |
| Objective | `L_diff` (masked latent MSE) `+ λ_bnd · BCE(join, pos_weight ≈ 1/join_rate)` |
| Curriculum | **(a) phrases → (b) full lines.** Phrases are cropped 2–5-character spans taken from aligned lines, so no new data is needed and boundaries are dense |
| Steps | ~200 k at batch 256 **[estimate]** |
| Purpose | the boundary-slot mechanism and the controllable join |

**Gate G4:** boundary F1 on val clearly above the trivial "always lift" baseline, and
`dit_diff` falling. Then look at samples: joins must appear where the model is confident,
and the render must not bend glyphs to fake a connection.

## Stage 5 — control calibration

Sweep `control ∈ {0, 0.25, 0.5, 0.75, 1.0}` on val lines; measure the realised join rate.
**Gate G5:** response is monotone and close to the requested rate (fit a calibration
curve and report it). This is what turns the knob from a claim into a measurement.

## Stage 6 — evaluation (held-out writers, prefix-free)

Report on D1 test writers + D3 cursive set, with a style reference taken from a
**different line of the same writer** — never the target line's own prefix (DiffInk's
protocol; see `survey/CENT_prior_art_survey.md` §3.2).

| Axis | Metric |
|---|---|
| Content | AR / CR with an OCR critic |
| Style | writer-classification accuracy; DTW to the writer's own lines |
| **Continuity** | **F1_Cursive, CRE, gap statistics on D3**; human preference study |
| Layout | inter-character geometry (OLHWG's ∇1–∇8) for comparability |
| Efficiency | chars/s |

Baselines: decoupled OLHWG-style (reproduce), DiffInk (weights released), our AR
baseline (`cent/models/baseline_ar.py`), plus the ablations in Stage 7.

## Stage 7 — ablations (each one answers a reviewer objection)

1. **No boundary slots** → collapses to a plain line diffusion model (≠ DiffInk, but close).
2. **No join supervision** (labels removed) → shows the boundary head is doing the work.
3. **No control** → the knob is not free-riding on the model prior.
4. **No character pretraining (skip Stage 1)** → quantifies contribution (iii).
5. **Stroke-granularity boundaries** (whole strokes to characters) vs **point-level**
   (Stage 3) → shows why point-level alignment matters.
6. **Per-character normalisation** (OLHWG-style) → expected failure, the motivation figure.

---

# Part 3 — Implementation status

All stage scripts are written and the whole chain runs end to end (verified with
`python scripts/run_all.py --smoke`, ~4 s, on the real corpus as well as fixtures).

| File | Purpose | State |
|---|---|---|
| `cent/pipeline/fetch.py` | detect → download → verify → extract datasets | **done** |
| `cent/pipeline/cache.py` | WPTT → `lines.npz`, writer-disjoint train/val/test | **done** |
| `cent/pipeline/align.py` | CTC Viterbi → point-level boundaries → join labels | **done** |
| `cent/pipeline/stage_vae.py` | Stages 1–2, with CTC monitoring + reconstruction PNGs | **done** |
| `cent/pipeline/stage_boundaries.py` | Stage 3, join statistics + alignment PNGs | **done** |
| `cent/pipeline/stage_dit.py` | Stages 4–5, phrase/line curriculum, control curve | **done** |
| `cent/pipeline/stage_eval.py` | Stage 6, prefix-free eval + real-vs-generated panel | **done** |
| `cent/pipeline/run_all.py` | sequential driver, gates, visual checkpoints | **done** |
| `cent/viz.py` | dependency-free PNG rendering (no matplotlib/PIL on the GPU box) | **done** |
| `cent/data/format_pot.py` | OLHWDB 1.x characters for optional Stage 1 | written, **not yet verified on real files** |

Every stage writes `metrics.json` **and** PNGs, and prints the image paths under a
"visual check" heading. `--strict` turns a failed gate into a non-zero exit;
`--force` (implied by `--smoke`) continues anyway.

## Compute — measured, not estimated

python scripts/bench.py runs the real shapes (d=128, z=64, DiT depth 3) on a local
RTX 3080 (same cc 8.6 as the rented 3090; the 3090 is ~15% faster):

| Stage | ms/step | steps | wall-clock (3090) |
|---|---|---|---|
| 2 — InkVAE lines (batch 64, T~512) | 9.7 | 7,740 (30 ep) | **~1.3 min** |
| 3 — alignment (forward only) | — | 16.5k lines | **~25 min** |
| 4 — BoundaryDiT (batch 16, cached latents) | 15.3 | 60k | **~15 min** |
| total | | | **~0.4 h ≈ $0.05** |

Three measured findings that changed the code:

1. **AMP is not the win here.** bf16 gave 1.06x on the VAE (11.1 -> 10.5 ms) and was
   **slower** on the DiT (14.7 -> 19.4 ms): at d=128 the models are latency-bound, not
   compute-bound. AMP is on for Stage 2, off for Stage 4; losses stay fp32 (measured
   bf16-vs-fp32 loss delta 0.7% / 1.5%).
2. **Vectorising build_layout gave 5.5x** (80.7 -> 14.7 ms/step, identical output).
   The Python loop over batch x frames was the real bottleneck, not the arithmetic.
3. **Caching frozen-VAE latents** removes the per-step encode, making Stage 4 a pure
   GPU loop.

Budget (\.11 at \.1367/h ~ 22 h) is ~50x more than a 60k-step run needs, so the
surplus should buy **more steps and more seeds**: a 200k-step DiT with 3 seeds is
still under \.20.
