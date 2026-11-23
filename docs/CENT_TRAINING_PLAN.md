# CENT: how to train, validate, infer and evaluate

Written 2026-11-22, before any real run. It answers, in order: what the datasets are for, what
must be fixed first, what is trained, how to start, how to validate, how to infer, how to
evaluate, and what is still missing.

**Status: the eight blockers listed in Section 0 are fixed and covered by tests** (see Section 0a).
What remains before a real run is data and the recogniser, not code.

---

## 0a. Status: decisions taken and implemented (2026-11-22)

| Decision | Choice | Implemented |
|---|---|---|
| Canvas | **64 x 1024** | renderer takes a non-square canvas (`render_strokes(..., width_size=1024)`); trainer and evaluator both pass it |
| `max_chars` | **32** | `MAX_CHARS=32`, `MAX_TOKENS=1024`; longer lines dropped and counted, never truncated |
| Legibility term | **verifier enabled** | `StrokeVerifier` wired into the trainer, applied per character via `char_slot` grouping; loss logged as `legib` |
| Offline image loss | **lambda_image = 1.0** | unchanged, now computed against a 64x1024 target instead of a square crop |
| Recogniser | **published CASIA HTR** | `benchmark/recogniser.py`: slot + identity + calibration protocol |
| First campaign | **paired lines only** | `paired_only=True`; skipped lines are counted (`n_unpaired`) |

All eight blockers in the table below are fixed:

| # | Fix applied |
|---|---|
| P1 | Parser **trusts the declared header size** and repositions past undocumented bytes instead of rejecting the file; 1-bpp bitmaps use the row-padded stride `H*((W+7)//8)`. Proven by a test whose fixture declares one extra header byte. |
| P2 | Single-scale normalisation into the canvas frame; a test asserts `(nx*W)/(ny*H) == span_x/span_y`. |
| P3 | Non-square renderer, verified `(2, 1, 64, 1024)`. |
| P4 | References and targets are canvas width (1024); the whole line is resized, never cropped. |
| P5 | `max_chars=32`, `max_tokens=1024` — measured demand 437 tokens at 32 level-2 characters. |
| P6 | Verifier implemented and logged (`legib` appears in every training record). |
| P7 | Classifier-free dropout on the character condition (`drop_char_p`). |
| P8 | Writer head trained against real writer ids from the record. |

**Adopted recogniser**, verified through the Hub API:

```
Teklia/pylaia-casia-hwdb2   licence MIT   PyLaia, language zh
revision 96011bc249ad374b1e3877638ffbc934ad6f73aa
dataset Teklia/CASIA (CASIA-HWDB2 line level)
files: model/, weights.ckpt, syms.txt, tokens.txt, lexicon.txt, language_model.arpa.gz   65.2 MB
```

It is **not downloaded here** (the standing rule defers artefacts to the GPU box), and the
decode command is deliberately a template: PyLaia's CLI differs between versions, and guessing
it would put an unverified command in the ledger. `benchmark/recogniser.py` fails closed, names
the model, and refuses to emit AR/CR until a working command is configured — because AR/CR is a
property of the reader, and the reader must be fixed and calibrated first.

## 0. Blocking defects (all fixed — kept for the record)

| # | Defect | Where | Fix |
|---|---|---|---|
| P1 | **Parsers reject real files.** Header size is asserted as `35+len(illus)`; the supplier's tables say `36+len` (DGRL) and `55+len` (WPTT). | `common/casia_lines.py:169` | Reconcile the arithmetic against one real page of each format, then assert. Also fix the 1-bpp bitmap stride to `H*((W+7)//8)`. |
| P2 | **Aspect ratio is destroyed.** `norm()` divides x and y by their *separate* spans, so a 1:16 line becomes 1:1. | `cent/data_lines.py:119-122` | Normalise by a single scale (`max(span_x, span_y)`) and centre, or normalise in the line canvas frame directly. |
| P3 | **The renderer is square.** `BezierRenderer(size=64)` renders 64×64 for a 64×1024 line, so the image loss compares a square render against the first 64 px of the target. | `cent/train_lines.py:60`, `cent/models/stroke_graph.py:176` | Add an aspect: renderer takes `(h, w)` (e.g. 64×1024, or 32×512 for speed) and pixel coordinates live in that frame. |
| P4 | **Style references and targets are hard-coded to 256 px.** A 1024 px line is truncated to its first ~6 characters. | `cent/data_lines.py:89,108,173` | Make the width a config value matching the canvas; downscale the whole line rather than truncating. |
| P5 | **Token budget overflows.** `MAX_CHARS=40`, `MAX_TOKENS=512`; level-2 glyphs average 11.69 strokes, so a 40-character line needs **547** tokens and is silently truncated mid-character. | `cent/data_lines.py:30-31` | Set `max_tokens=1024` (the denoiser already allocates 1024 positions) and `max_chars=32`; drop longer lines and report how many. |
| P6 | **The legibility term is not implemented.** `lambda_legibility` exists in config; no verifier is built or used. The paper's objective includes it. | `cent/train_lines.py` | Implement the per-character verifier (as in `cent/models/diffusion.py::StrokeVerifier`) trained on real stroke groups, or remove the term from the paper. |
| P7 | **No classifier-free-guidance dropout**, though `drop_char_p`/`guidance` are configured and sampling claims CFG. | `cent/train_lines.py` | Drop the text condition with probability `drop_char_p` during training; sample with and without guidance. |
| P8 | **The writer head is trained against a constant.** `cross_entropy(whead(z_g), torch.zeros(...))` teaches one class. | `cent/train_lines.py:119` | Feed the true writer id from the record. |

Measured token demand (strokes + one layout token per character + one connector per
boundary, from `analysis/lexicon_stats.json`; level-1 mean 9.77 strokes, level-2 mean 11.69):

| characters | level-1 tokens | level-2 tokens | fits 512? |
|---|---|---|---|
| 24 | 281 | 328 | yes |
| 30 | 352 | 410 | yes |
| 32 | 376 | 437 | yes |
| 36 | 423 | 492 | yes |
| 40 | 470 | **547** | **no** |

`max_chars=32` with `max_tokens=1024` leaves headroom even for 32 characters of 24 strokes each
(831 tokens, the observed maximum).

A smoke test on real data must pass after P1–P5 before P6–P8 matter; the whole list is
roughly a day of work and none of it changes the method.

---

## 1. What each dataset is for

| Corpus | Role | Why |
|---|---|---|
| **CASIA-HWDB2.0–2.2** (`.dgrl`, 52,230 lines) | **images + transcripts**; the offline appearance target | each page record *is* a line: transcript, offset, bitmap |
| **CASIA-OLHWDB2.0–2.2** (`.wptt`, 52,220 lines) | **trajectories + inferred connectors** | strokes per line + transcripts; the per-character split is inferred (declared rule) |
| **Paired lines** (same writer, page, line, matching text) | **supervision for connectivity in the image domain** | only paired lines let us validate the image-only join proxy |
| **ICDAR-2013** (3,432 lines, 60 writers) | external test set | writer-disjoint from the corpus |
| **Lexicon** (`data/lexicon/graphics.txt`) | the topology oracle | stroke count and order per character |
| **Unifont glyph table** (DiffBrush's `unifont.pickle`) | content images for every offline baseline | shared content input, not a training target |

Training needs lines that **have strokes** (online or paired): a line with an image but no
trajectory gives no connector supervision and no geometry target. Image-only lines are kept
for style statistics and for the offline image objective, not as the primary signal.

## 2. Splits

Standard writer partition: **816 train / 203 test** (offline), **1,000 / 200** (online).
Carve a **60-writer validation set from the training side** — never from test.

```
train  756 writers  (~38.8k lines)
val     60 writers  (~3.0k lines)   gate only
test   203 writers                   reported once, after the gate passes
```

Splitting is by writer, never by line: the same writer must not appear on both sides, or the
style metric is measured on people the model has seen.

## 3. Build the data (on the GPU box)

```bash
# 1. probe first — a mis-framed record yields plausible garbage
python -m common.probe_lines dir /data/casia/HWDB2/Gnt2.0 --kind dgrl --limit 10
python -m common.probe_lines dir /data/casia/OLHWDB2/Pot2.0 --kind wptt

# 2. build the canonical line set
python - <<'PY'
from common.canonical_lines import build_lines
from common.lexicon import load_lexicon
meta = build_lines('/data/casia/HWDB2/Gnt2.0', '/data/casia/OLHWDB2/Pot2.0',
                   'data/canonical_lines/v1',
                   lexicon=load_lexicon('data/lexicon/graphics.txt'))
print(meta['counts'], 'paired', meta['paired_lines'], 'with strokes', meta['lines_with_strokes'])
PY
```

**Gate the build on numbers, not on hope.** Record and check:

- `counts` per split (expect ≈ 38.8k / 3.0k / 10.4k for the 756/60/203 partition);
- `paired_lines` — if this is small, connectivity supervision is mostly trajectory-only;
- `lines_with_strokes` — the effective training set;
- **join rate per writer** (fraction of boundaries with an inferred connector). If it is ~0 for
  most writers, the data has no cursive signal and the central claim cannot be tested — this
  is the single most important pre-flight number;
- **transcript vs stroke-group agreement** (the T2/T3 statistics). Mismatches mean the inferred
  split is wrong for those lines; report the fraction.

## 4. What is trained

One model, trained end to end on a line:

| Component | Trained? | Notes |
|---|---|---|
| Line denoiser (1-D U-Net over tokens) | **yes** | predicts noise and the join logit |
| Line style encoder | **yes** | consumes k reference lines, emits `z_g, z_l, z_κ` |
| Writer head | **yes** (auxiliary) | needs real writer ids (P8) |
| Per-character verifier | yes, if P6 is kept | trained on real stroke groups |
| Bézier renderer | **no** | analytic and differentiable |
| Lexicon topology | **no** | fixed oracle |
| HTR / recogniser | separate model | frozen; never trained with CENT |

Not trained at inference time: nothing. The connector, layout and stroke geometry are all
outputs of the same denoiser.

**No pretrained initialisation.** The denoiser consumes control-point tokens (415 x 8 numbers), not
image latents, and the decoder is analytic — so an SD-class VAE or UNet has nothing to attach to.
Everything trains from scratch in torch + numpy (`docs/PRETRAINING_DECISION.md`). The baselines
that do reuse a pretrained VAE (DiffBrush: SD1.5; TGC-Diff: SD3.5) need it only because they
generate pixels.

## 5. How to start training

```bash
# config: copy and fill in the real numbers
cp cent/configs/cent_r1.json cent/configs/casia_r1.json
```

Values to set in `casia_r1.json` (each is a decision, not a default):

```json
{
  "canonical_root": "data/canonical_lines/v1",
  "lexicon_path": "data/lexicon/graphics.txt",
  "img_size": 64,            // canvas height; width set separately (P3)
  "canvas_width": 1024,      // P3/P4: the line canvas
  "k_refs": 5,               // style lines, matches the protocol
  "n_classes": 2703,         // from meta.json num_classes, NOT 3755
  "n_writers": 1019,
  "dim": 256, "depth": 4, "n_heads": 4,
  "T": 1000,
  "batch_size": 16,
  "lr": 2e-4, "weight_decay": 0.01, "grad_clip": 1.0,
  "epochs": 100,
  "lambda_image": 1.0, "lambda_legibility": 0.1, "lambda_writer": 0.1,
  "drop_char_p": 0.1,
  "num_workers": 8, "device": "cuda", "seed": 0,
  "run_name": "casia_r1", "out_dir": "runs"
}
```

```bash
# dry run: one batch, prints shapes and four losses, writes nothing
python -m cent.train_lines --config cent/configs/casia_r1.json --dry-run

# then the real run (resumable: last checkpoint every epoch)
python -m cent.train_lines --config cent/configs/casia_r1.json
```

**Protocol for the first hour.** Run 50 steps, read `runs/<name>/train_log.jsonl`, and compute
step time before committing to a full schedule:

```
steps_per_epoch = ceil(train_lines / batch_size)          ≈ 38,800 / 16 ≈ 2,425
epochs          = chosen so total_steps ≈ 150k–300k       (≈ 60–120 epochs)
wall_clock      = steps * sec_per_step / 3600
```

If `sec_per_step` makes the budget unacceptable, the levers in order of least harm are:
reduce canvas width 1024 → 512 (still line-shaped), batch size 16 → 32, then dim 256 → 192.
Do **not** reduce `k_refs` below 3 or shorten `max_chars` below 24 first — both change what
the model must learn.

**Losses actually optimised** (after P6–P8): diffusion noise MSE on token payloads, BCE on the
join decision, image L1 through the renderer, writer classification, and (optionally) the
per-character legibility CE. Log each separately; if `join` does not fall below its prior
(`-r log r - (1-r) log(1-r)`; at a join rate of r = 0.35 that prior is **0.6474 nats**), the connectivity objective is
not learning and the run should be stopped, not extended.

**Seeds.** Three seeds for the headline configuration; one seed for ablations that cannot
afford three, with the seed stated in the ledger.

## 6. Validation and the gate

Evaluate on the 60 validation writers after every epoch (cheap subset) and in full before any
test run:

```bash
python -m cent.eval_lines --config runs/casia_r1/config.json \
    --ckpt runs/casia_r1/ckpt_last.pt --split val --max-lines 400 --out runs/casia_r1/val.json
```

The report already contains `join_rate_error` (the primary metric), `continuity_error`,
`connector_rate_pred/ref`, `image_l1`, `ssim`, and explicit `NOT_RUN` rows for speed profile,
AR/CR and FID.

**Gate (declared before running, closed by the author, never by a script):**

| Check | Pass condition |
|---|---|
| join rate error | lower than the training prior's trivial predictor (always-lift and always-join baselines are the floor) |
| continuity | finite and reported; no run is accepted without it |
| image agreement | `ssim` above the all-paper-image baseline |
| stability | no NaN, join BCE below the prior, checkpoint loads |

A gate FAIL returns to the configuration, not to the test set. The test split is touched
once, after a PASS.

## 7. Inference

```bash
python -m cent.eval_lines --config runs/casia_r1/config.json \
    --ckpt runs/casia_r1/ckpt_last.pt --split test --max-lines 2000 --out runs/casia_r1/test.json
```

What inference produces, per line: the token sequence `G`, from which

- `Traj(G)` — an ordered trajectory with pen events taken from the join variables;
- `R(G)` — the rendered line image at the canvas size.

Sampling: `T=1000` DDPM steps as configured (switch to DDIM 50 for the reported numbers if
speed matters, and record which). Guidance scale is a reported parameter; sample at the
default and at one alternative for the ablation.

**Controllable at inference** (this is the paper's claim, so it must be exercised): the join
variables can be overridden — force all-lift (handprint), force the writer's own rate, or
per-boundary override. Each override is an experiment with its own row, not a demo.

## 8. Evaluation: what each number needs

| Metric | Needs | Status |
|---|---|---|
| join-rate error, continuity error, gap similarity | trajectories (ours) | implemented |
| image-only join proxy | line images + inferred GT joins | implemented, needs validation on paired lines |
| AR, CR | a **frozen Chinese line recogniser** | not built; must be trained once on the training split and its own AR/CR reported |
| D_CER, D_WER | a trainable recogniser run on generated data | same recogniser, second use |
| FID, KID | Inception-v3 features | `torchvision` present; extractor must be fixed and recorded |
| HWD | VGG16 handwriting backbone (height 32) | weights not obtained; report `NOT_RUN` until they are |
| speed profile | timestamps | not stored in the canonical layer; `NOT_RUN` |
| AR/CR for image-only baselines | recogniser + validated proxy | blocked on the recogniser |

The recogniser is the long pole: it gates AR/CR, D_CER/D_WER, and every baseline's content
column. Train it **before** the comparison campaign, on training writers only, and record its
accuracy on real test lines — that number bounds every AR/CR we report.

## 9. Ablation matrix (what each run changes)

| ID | One-line config change |
|---|---|
| A1 | drop connector tokens entirely (emergent joins only) |
| A2 | remove the join BCE, predict joins at inference from geometry |
| A3 | set `z_κ` to zero (no connector style) |
| A4 | disable the lexicon constraint (topology predicted, not fixed) |
| A5 | `k_refs ∈ {1, 5, 15}` |
| A6 | allow stroke merging inside characters (relaxed count) |
| A7 | fixed spacing instead of generated layout |
| A8 | `lambda_legibility ∈ {0, 0.1, 1.0}` (the content-loss weight the anchor paper ablates) |
| A9 | free connector endpoints instead of anchored (the mechanism test) |

A9 is the one that decides whether continuity-by-construction is worth anything: same
backbone, same data, only the parameterisation changes.

## 10. Order of operations

1. P1 parse fix → probe one real `.dgrl` and `.wptt`.
2. Build canonical set → check the four pre-flight numbers (counts, paired, strokes, **join rate**).
3. P2–P5 (aspect, canvas, ref width, token budget) → dry run on real data.
4. P6–P8 (verifier, CFG dropout, writer ids) → dry run, then 50 steps and a time estimate.
5. Full run, seed 0 → validate each epoch → gate.
6. Recogniser training (independent) → unblocks AR/CR and the baselines' content columns.
7. Gate PASS → test split once → connectivity + image metrics.
8. Baselines: composed ones through the composer, native ones through their adapters, all
   scored by `benchmark/score_lines.py` with the same recogniser.
9. Ablations, user study, timing.

## 11. Open decisions for the author

1. **Canvas width** 1024 (protocol-faithful, expensive) vs 512 (cheaper, still line-shaped).
2. **`max_chars`** 32 (covers most lines, drops the tail) vs 40 with a bigger budget.
3. **Legibility term**: implement the verifier (paper-faithful) or drop it (simpler, and the
   paper's objective changes).
4. **Offline image loss weight**: it is the most expensive term per step; keep at 1.0 or reduce.
5. **Recogniser**: train our own CRNN on the training split, or adopt a published CASIA HTR
   (faster, but its identity and accuracy must be recorded).
6. Whether the first campaign trains only on paired lines (cleaner connectivity supervision,
   smaller set) or on all stroke-bearing lines (larger, weaker supervision).
