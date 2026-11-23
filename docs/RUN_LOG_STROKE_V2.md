# Stroke-level reformulation: first real run on a rented 3090

Instance: RTX 3090, 24 GB, 12 cores, 32 GB RAM, Switzerland, **$0.256/h**.
Total spend for this session: **≈$0.58** (setup, data, two training runs). Instance left **stopped**.

## 1. The reformulation

The character-decomposed formulation was abandoned because the corpus cannot support it: measured
0.597 strokes per lexicon stroke, and only 0.8% of lines have at least as many strokes as the
lexicon demands (see `docs/FIRST_RUN_FINDING.md`).

The stroke-level model keeps the Bézier payload, the renderer, the style encoder and the metrics,
and changes what a token is:

| | before | now |
|---|---|---|
| token | one stroke of one character, or a connector, or a layout marker | one stroke |
| conditioning | per-token character id | the whole transcript, via cross-attention |
| "join" | an inferred connector token | **geometric**: `gap(i) = ||end(i) - start(i+1)|| / median_stroke_diagonal`, joined iff `gap < tau` |
| legibility | per-character verifier | not available (no per-character segmentation); line-level legibility must come from the frozen HTR at evaluation |

**Tau was calibrated, not guessed.** On 600 real paired lines (58,880 boundaries):

| gap percentile | p10 | p50 | p90 |
|---|---|---|---|
| value | 0.204 | 0.578 | 1.464 |

`tau=0.35` gives a **25.5% join rate** — non-degenerate, so the label carries real signal, and the
same number appears on the training log's `join_rate_gt`.

## 2. Bugs found and fixed on the way

1. **Inverted image convention.** The corpus stores ink=0/paper=1; the rasteriser returns
   ink=1/paper=0. Run 1 compared inverted images, so a blank render scored 0.015 while the loss sat
   at **0.977** and could never fall. Fixed (`1.0 - img - target`), and the same inversion was
   present in the old character-level trainer.
2. **Class imbalance in the image loss.** Ink is **1.54%** of a line's pixels. Added ink weighting
   (×20) and a soft-Dice term.
3. **Join BCE on uninformative timesteps.** At `t=900` the input is almost pure noise, so the head
   can only emit the class prior. Run 2 restricts the BCE to `t < 300`.
4. `nn.Embedding` received `-1` padding (device-side assert); padding now uses the real pad id.
5. Transcripts longer than the text position table crashed the encoder; truncated and guarded.

## 3. What the runs show

Both runs used 8,631 paired training lines, batch 32, 64×1024 canvas, 0.52 s/step.

**Run 1 (9 epochs, stopped):** identified the inverted-image bug.
**Run 2 (12 epochs, 3,240 steps) — loss trajectories:**

| loss | first 1/8 | last 1/8 | verdict |
|---|---|---|---|
| diffusion | 0.200 | **0.099** | learning |
| join BCE | 0.552 | 0.526 | prior is 0.556 → only 0.03 nats gained |
| image | 0.606 | 0.604 | flat |
| writer | 5.35 | **3.77** | learning |

**Validation (400 val lines, 33,139 boundaries), join decision at t=50:**

| predictor | join rate | precision | recall | F1 | rate error |
|---|---|---|---|---|---|
| **CENT v2 (12 epochs)** | 0.028 | 0.695 | 0.086 | **0.153** | **0.201** |
| always-lift | 0.000 | – | – | 0.000 | 0.229 |
| always-join | 1.000 | 0.229 | 1.000 | **0.373** | 0.771 |

Ground-truth join rate: 0.229.

**Verdict: the join predictor is not working yet.** It beats both trivial baselines on join-*rate*
error (0.201 vs 0.229 / 0.771) but its **F1 0.153 is below the always-join baseline's 0.373**. It
has collapsed toward "lift": precision 0.70 at recall 0.09. The pipeline is validated end to end;
the model is not.

## 4. Diagnosis and the next step

Three causes, in order of likely size:

1. **Under-training.** 12 epochs = 3,240 steps = 12 passes over 8,631 lines for a 5.5 M model.
   Everything except the diffusion term is still moving at the end (writer loss fell 5.35 → 3.77
   over the run). This is the dominant explanation and it is cheap to test: 100–200 epochs is
   **~$1.5–3** at the measured 0.52 s/step.
2. **Class imbalance in the join head.** 23% positives with a plain BCE under 12 epochs produces
   exactly the observed collapse. Fix: `pos_weight ≈ 3`, or a focal loss, plus a validation-tuned
   threshold instead of a fixed 0.5.
3. **The image term is weakly conditioned.** Placing ~100 strokes exactly, at 64×1024 with 1.5% ink,
   is a hard L1 target; Dice barely moved it. Consider rendering the loss at 32×512, or reducing
   `lambda_image` and letting the diffusion/vector losses carry the geometry (they already do).

**Do not conclude the method fails from this run.** What is established: the data pipeline, the
formulation and the metrics all run on real Chinese line data at 0.52 s/step, and the diffusion and
writer objectives learn. What is not established: that the join head can predict boundary
connectivity. That needs a longer run (same code, 100+ epochs, ~$2) before any claim.

## 5. Numbers worth keeping

| quantity | value |
|---|---|
| NLPR download, 1 connection | 150 KB/s (throttled) |
| NLPR download, `aria2c -x16` | **38 MB/s** |
| corpus | 2.62 GB → 5,272 pages → 41,710 stroke-bearing lines, 11,778 paired |
| canonical set | 391 MB, 13,768 images, 815 writers, 2,680 classes |
| strokes per line | mean 99.1, median 101, max 211 |
| step time (B=32, 99 strokes) | 0.524 s |
| epoch time | 2.4 min |
| VRAM | 4.3 GB |
| PCIe under load | **Gen2 ×16** (lanes satisfied; generation is Gen2, not Gen4) |
| cost, 12 epochs | ≈$0.26 |

## 6. Artifacts

`results/cent_results_3090.tar.gz` (40 MB, downloaded): both runs' `train_log.jsonl`, checkpoints
(22 MB each), configs and stdout logs. The instance is **stopped** with the corpus, canonical set,
cache and rclone installed; a restart is enough to resume, and the dataset upload to Google Drive is
blocked only on an rclone remote credential.

---

## 7. Second session: six fixes, and the answer to "does it work?"

The instance was restarted, the join **label alignment** was inspected, and a supervised probe was
run to separate "needs more training" from "will not work as designed".

### Bugs fixed in this session

| # | Bug | Evidence it mattered |
|---|---|---|
| 6 | **Join label off by one.** The head at token k predicts the boundary (stroke k-1, stroke k) = `join[k-1]`, but the trainer paired it with `join[k]`. | Every boundary label was shifted by one stroke. On a ~99-stroke line that corrupts essentially all labels — this alone explains a head that never left the prior. |
| 7 | **Three config fields existed only on the box**, and shipping the local tree reverted them (`ink_weight`, `dice_weight`, `join_pos_weight`), and re-broke the text-position guard. | Runs 3a/3b crashed with a device-side assert or trained with the inverted image loss. Fixed by patching locally and verifying all six patterns on the box *before* launching. |
| 8 | **Positive-class weighting** for the join head (`pos_weight=3`, ~23% positives). | Prevents the collapse toward "lift" seen in run 2. |

### Run 3 (corrected alignment, pos_weight, 3+ epochs)

| loss | epoch 0 | epoch 3 | prior |
|---|---|---|---|
| diffusion | 0.264 | **0.103** | – |
| join | 1.044 | 1.066 | **1.05** (with pos_weight 3) |
| image | 0.607 | 0.640 | ~0.74 for a blank render |
| writer | 5.63 | 5.24 | 6.38 (chance) |

Diffusion learns; the join head sits **at its prior**; the image term barely moves.

### The decisive probe

If the join label were learnable from the tokens, a small supervised network given the two
consecutive stroke payloads should recover it. It does not:

| probe | BCE | prior | gain |
|---|---|---|---|
| labels normalised by per-line stroke scale | 0.5791 | 0.6186 | 0.04 |
| labels on **absolute** canvas gap < 0.05 | 0.5168 | 0.5439 | 0.027 |

A 2-layer MLP on the exact payload pairs gets F1 **0.010** at threshold 0.5 and 0.54 only when the
threshold is moved to 0.3. Two contributing causes:

1. **The relative label depends on a line-level statistic** (the median stroke diagonal), which a
   local pair cannot supply; the model would have to infer it globally. My own normalisation made
   the label partly non-local.
2. **Even the absolute version is only weakly determined** by the payload pair as the network sees
   it, which points at the representation rather than at step count.

## 8. Verdict

**The model does not work yet, and it is not merely under-trained.** Three runs (9, 12 and 3+
epochs), eight fixes, and the probe above all point the same way: the auxiliary join classifier is
the wrong device. Time on the GPU was not the binding constraint — the label and its supervision
were.

What is established: the corpus parses, the canonical set builds, the stroke-level formulation runs
at 0.52 s/step, and the diffusion and writer objectives train.

What is not: that a per-boundary classifier trained on noisy hidden states can predict connectivity.

## 9. Recommended change (not yet implemented)

Stop *predicting* joins; make them an **outcome** of where the model puts stroke endpoints, and
score them post-hoc with the metrics that already exist (join-rate error, continuity error).

Concretely:
1. Drop `join_head` and its BCE.
2. Add an **endpoint-continuity loss** on the predicted clean strokes: for boundaries the writer
   actually joined, penalise `||end(x0_k) - start(x0_{k+1})||`; for lifted boundaries, only penalise
   if the gap falls below a floor. This supervises the same phenomenon through geometry instead of
   through a classifier.
3. Report joins by measuring the generated output — the same way the baselines are measured — so
   the claim is about what the model *draws*, not about an internal probe.

That is a smaller change than the character-to-stroke reformulation and it removes the failing
component rather than training it harder.

---

## 10. The fix, applied and measured (run 5) — it works

Following §9, the join *classifier* was removed and connectivity is now supervised as geometry on
the predicted clean strokes:

```
gap(k)      = || end(x0_k) - start(x0_{k+1}) ||  /  mean_stroke_chord(line)
loss_cont   = mean_joined gap^2                       (pull connected strokes together)
            + mean_lifted relu(floor - gap)^2         (keep lifted boundaries apart)
floor = 0.5  (in units of the mean stroke chord)
```

Two details mattered:

1. **Absolute gaps have a degenerate solution.** With the gap in canvas units, the loss is minimised
   by shrinking the whole line: measured in run 4, `gap_J` and `gap_L` fell *together*
   (0.135 / 0.144, ratio 1.07) while the continuity loss looked healthy. Dividing by the line's mean
   stroke chord removes that shortcut — shrinking everything leaves the ratio unchanged.
2. **Averaging gaps over all timesteps is a misleading diagnostic.** At `t=900` the x0 estimate is
   noise, so a gap averaged over all `t` mostly measures the prior. All numbers below are at
   `t=50` with clean input, i.e. the model's own one-step geometry estimate.

### Run 5 (12 epochs, 3,240 steps, 0.531 s/step)

| loss | first 1/6 | last 1/6 |
|---|---|---|
| diffusion | 0.210 | 0.111 |
| continuity | 0.633 | **0.172** |
| writer | 5.31 | **3.80** |
| image | 0.607 | 0.606 (still flat) |

### Validation (300 lines, 33k boundaries, t=50)

| quantity | value |
|---|---|
| mean gap at **joined** boundaries | **0.062** canvas units |
| mean gap at **lifted** boundaries | **0.158** |
| separation | **2.54x** |

Join decision from the predicted geometry, at the **label's own threshold** τ=0.05:

| predictor | join rate | rate error | precision | recall | F1 |
|---|---|---|---|---|---|
| **CENT v2 run 5** | 0.208 | **0.004** | 0.405 | 0.413 | **0.409** |
| always-lift | 0.000 | 0.204 | – | – | 0.000 |
| always-join | 1.000 | 0.796 | 0.204 | 1.000 | 0.339 |
| previous version (join classifier) | 0.028 | 0.201 | 0.695 | 0.086 | 0.153 |

Earlier checkpoints of the same run reached **F1 0.529** at τ=0.08 (epoch 5) and **F1 0.531** at
τ=0.05 (epoch 9) with rate error 0.037, so per-boundary quality varies across epochs while the
aggregate rate stays close.

**Verdict: the mechanism works.** The model separates connected from lifted boundaries by 2.5x and
reproduces the corpus join rate to within **0.004**, where the trivial baselines are off by 0.20 and
0.80. Per-boundary F1 (0.41-0.53) is modest and will need more training and a real sampler.

## 11. Remaining issues, in priority order

1. **No sampler yet.** All evaluation above uses a one-step x0 estimate at t=50, not a generated
   sample. The paper's protocol needs DDIM sampling over the stroke sequence; until that exists,
   these numbers describe the geometry estimate, not generated lines.
2. **The image loss is flat (0.607) in every run.** Ink is 1.5% of pixels and ~100 strokes must land
   within a pixel or two; L1+Dice is not discriminating at 64x1024. Either render the loss at lower
   resolution, or accept that the vector losses carry geometry and treat the image term as
   regularisation only.
3. **Per-boundary F1 needs more epochs** (100+ rather than 12) before it is a headline number.
4. Text conditioning is untested: nothing yet shows the model uses the transcript rather than
   memorising the style. A text-swap ablation is the cheapest check.

## 12. Reproducibility: the sync mechanism

Three runs were lost to local/box divergence (a fix applied on the box was overwritten by a stale
local copy, twice; once a fix was reverted the same way). `tools/sync_box.py` now makes a tarball
plus a sha256 manifest of every source file, and the box verifies before a run:

```
python tools/sync_box.py make      # local: dist/cent_sync.tar.gz + dist/sync_hashes.json
# upload both, extract, then on the box:
python tools/sync_box.py verify    # -> "63 expected, 63 present, 0 differ, 0 missing / TREE MATCHES"
```

Run 5 was launched only after that check passed, and it passed again afterwards.

---

## 13. Correction: the earlier "it works" was reconstruction, not generation

A DDIM sampler was implemented (`benchmark/sample_strokes.py`) and used to generate from run 5. The
images are **scattered dots, not strokes** (`results/sheet_val_r5.png`). On sampled output the
predicted join rate is 0.036 against a ground truth of 0.140.

**What §10 measured, and what it did not.** The "2.54x separation / rate error 0.004" was computed
at t=50 with the *clean input*: `x0 = (x_t - sqrt(1-a) eps)/sqrt(a)` where `x_t` is barely noised.
That is a near-identity reconstruction, not a generation. It shows the model has learned a local
denoising operator; it says nothing about whether it can produce a line from noise. It cannot yet.

## 14. Why: timestep coverage

The overfit test (a single line, then a single line for 3,000 steps) plateaus instead of memorising:

| run | steps | diff loss |
|---|---|---|
| overfit, 400 steps | 400 | 1.233 -> 0.391 (min 0.22) |
| overfit, 3,000 steps | 3,000 | 1.233 -> 0.196 (min 0.066) |

A model that had fit one line would sit near zero. The reason is arithmetic: with `T=1000` and 3,000
steps, **each noise level is visited about three times**. A diffusion model has to fit a different
denoising function at every t, so the effective amount of training is the number of *visits per
timestep*, not the number of steps.

Also fixed here: `loss_diff` was divided by `B*N` including padding, diluting it and making it
incomparable across batches; it is now normalised by the number of valid tokens.

## 15. Run 6 (in progress): coverage first

`T=200` instead of 1000, 60 epochs. At 269 steps/epoch that is 16,140 steps -> **81 visits per noise
level**, a 5x improvement in coverage per unit of compute, for the same step time (0.55 s).

The next thing to report is a sampled sheet from run 6 and, if the strokes are coherent, the
first text-swap ablation (does the transcript control the content, or has the model memorised
style?).

---

## 16. Two more defects found by probing the conditioning paths

Before spending another two GPU-hours, the two conditioning paths were probed directly: hold the
input fixed, swap one condition, measure how much the denoiser output moves.

| swap | change in eps | verdict |
|---|---|---|
| transcript | 0.0617 (15.1% of signal) | **alive** |
| style references | **0.0000 (0.0%)** | **dead** |

### The style encoder was collapsed

`StyleEncoder.forward` pooled CNN features with a global mean over raw reference images. The corpus
stores **paper = 1.0 / ink = 0.0**, and a line is ~1.5% ink, so every writer pooled to nearly the
same near-white vector:

```
||z_g(writer A) - z_g(writer B)|| = 0.005      against      ||z_g|| = 11.18      -> 0.05%
```

All writers were the same style as far as the denoiser could tell. That also explains why the writer
classification loss never fell below ~5.17 (chance = ln 592 = 6.38) in runs 5 and 6.

**Fix:** the encoder now reads **ink coverage** (`1 - image`), normalised per image by its own max.
This makes it scale-invariant across writers, and it matches what the rasteriser produces (ink = 1),
so references and renders now share a convention. On a random-init encoder the same measurement goes
from 0.05% to **1.8%** differentiation before any training.

### Also fixed

`loss_diff` was divided by `B*N` including padding, which diluted it and made it incomparable across
batches; it is now normalised by the number of valid tokens.

## 17. Run 7 (in progress)

`T=200`, 60 epochs, fixed style input. Health signals to watch, in order:

1. **writer loss** — 5.22 at the start; it should fall toward 3.5-4.0 if style is now discriminative.
   A flat 5.2 means the global-mean pooling is still the bottleneck and `z_l` (attention pooling)
   should replace `z_g` as the conditioning vector.
2. **style-swap sensitivity** — must be non-zero after training; 0.0% is a hard fail.
3. **sampled output** — strokes rather than the dot clouds of run 5.
4. **text swap** — already 15.1%, so the transcript path is wired; keep it as a regression check.

## 18. The plan from here

| step | cost | decision it settles |
|---|---|---|
| finish run 7, sample it | ~2 h, $0.55 | does fixed style + T=200 produce strokes at all |
| if strokes: 200-300 epochs | ~7 h, $1.8 | enough visits per timestep for real quality |
| text-swap and style-swap ablations | ~1 h, $0.25 | is conditioning causal, or memorised |
| image-loss study (32x512 render, or lambda_image 1.0 vs 0.1) | ~1.5 h, $0.4 | the image term has been flat (0.607) in every run |

The image loss being flat in five consecutive runs is the one known dead weight: ink is 1.5% of
pixels and ~100 strokes must land within a pixel or two, so L1+Dice at 64x1024 is not discriminating.
Either the loss is rendered at a lower resolution, or `lambda_image` drops and the vector losses carry
the geometry.

---

## 19. Run 7 checkpoint at epoch 19 (34%): style fixed, generation still not strokes

No errors; 5,495 steps, 0.531 s/step, ETA 1.6 h.

| signal | before the fix | at epoch 19 | verdict |
|---|---|---|---|
| `writer` loss (chance = 6.38) | 5.09 | **1.98** | style encoder now discriminates writers |
| `z_g` writer differentiation | 0.05% | **42.1%** | fixed |
| style swap -> change in eps | **0.0%** | **2.2%** | conditioning is now causal |
| diffusion loss (T=200) | 0.279 | 0.131 | still falling |
| continuity loss | 0.629 | 0.167 | still falling |
| image loss | 0.607 | 0.605 | **flat in six runs** |
| sampled join rate vs GT | 0.036 / 0.140 | 0.056 / 0.133 | closer, still low |

The generated rows in `results/sheet_ep19.png` improved from *scattered dots* (run 5) to a
**denser banded smear** with some horizontal structure -- progress, but still not legible strokes.

### Reading of the situation

The mechanics are right: the style path is now demonstrably causal, text is causal, both losses fall,
and the sampler runs end to end. What is missing is **stroke-level structure**: the model outputs
something close to the conditional mean over plausible stroke placements, which renders as a smear.

The likely cause is the absence of any text-to-stroke alignment. A line of 26 characters becomes ~78
strokes, and nothing tells the denoiser which character a given stroke belongs to. With no alignment
signal, the best strategy available to the model is to average over placements -- exactly the blur
observed.

Two candidate remedies, in order of expected value:

1. **Implicit alignment conditioning**: give each stroke token its normalised position in the line
   (`i / (N-1)`) and the line length, so the model can associate "first third of the line" with the
   first characters. Cheap: two extra scalars per token, no architectural change.
2. **Character-count conditioning**: feed the transcript length and let the model infer the expected
   stroke budget per character (CASIA gives ~3 strokes/char after cursive merging).

Neither needs new data; both are small changes. They should be tried before spending 7 GPU-hours on
more epochs of the current conditioning, because more epochs sharpen what the conditioning allows,
and right now the conditioning does not allow alignment.

---

## 20. Run 7 at epoch 37 (65%): metrics good, images unchanged

| signal | ep 19 | ep 37 |
|---|---|---|
| diffusion loss | 0.131 | **0.079** |
| continuity loss | 0.167 | 0.147 |
| writer loss | 1.98 | **1.17** |
| image loss | 0.605 | 0.584 |
| **sampled** join rate vs GT 0.133 | 0.056 | **0.175** |
| generated image | banded smear | **banded smear** |

Every scalar is improving, and the sampled join rate is now in the right range. The generated picture
is essentially unchanged from epoch 19: a grey dithered band, not strokes.

### What that pattern means

Metrics improving while the image stays a smear is the signature of a model that emits the
**conditional mean** of many plausible stroke layouts. The diffusion loss can fall because the mean
is a good predictor of noise; the render still blurs because averaging over placements produces a
band. Compounding it: the payloads are normalised per line by their own ink bounding box, so the same
normalised coordinates mean different absolute geometry for different lines.

The model has sequence-position embeddings, so "where in the sequence" is available. What it does not
have is any tie between a stroke and the character it belongs to. That is the conditioning gap, and
more epochs sharpen what the conditioning allows rather than widening it.

**Decision: stop optimising epochs; add alignment conditioning.** Run 7 is left to finish (0.84 h)
purely as the end-point record, and the next run adds:

1. per-stroke normalised line position `i/(N-1)` and the stroke count `N` as input scalars;
2. the transcript length and the implied stroke budget (`N / C`, about 3 for CASIA cursive);
3. a local text window: cross-attend to the characters near `i/(N-1) * C` rather than to the whole
   transcript, so the model can bind a stroke neighbourhood to a character neighbourhood.

All three are conditioning changes, not new data or new losses.

---

## 21. Run 8: alignment conditioning, implemented

Run 7 was stopped at epoch 39 (its end-point samples were unchanged from epoch 19), and the three
alignment changes from §20 were implemented:

| change | where | what it gives the model |
|---|---|---|
| per-stroke position `i/(N-1)` plus `N/M` and `C/64` as input features | `data_strokes` -> `StrokeDenoiser.feat` | "where in the line am I", and the line's stroke/character budget |
| **local text window** | `StrokeDenoiser.forward` `attn_mask` | each stroke attends to ~3 characters around the character its position implies, instead of the whole transcript |
| mask non-empty guarantee | same | an all-False attention row gives NaN and a device-side assert; two `where`-based fallbacks prevent it |

Side effect: the local window cuts attention cost, and the step time dropped from 0.531 s to
**0.390 s** while doing more work.

Run 8 is training (60 epochs, ETA ~1.7 h). At **epoch 4** the sampled output is still dots
(`results/sheet_r8_early.png`, sampled join rate 0.0035 vs GT 0.133) -- too early to judge, exactly as
run 7 looked at the same point.

### The checkpoints that matter

- **epoch ~25**: if alignment is working, strokes should start appearing as elongated marks rather
  than dots, and the sampled join rate should climb from 0.003 toward the 0.13 ground truth.
- **epoch 60**: the sheet to compare against run 7's banded smear.

If epoch 25 still shows dots, alignment was not the binding constraint and the next suspect is the
payload parameterisation itself (8 scalars per stroke normalised per line), not more epochs.

---

## 22. ROOT CAUSE FOUND: the image and the strokes were never in the same frame

Run 8 (alignment) reached epoch 31 and the picture was still a smear. Before spending more epochs I
asked the question the sheets had been hinting at: *does the ground-truth stroke geometry even line
up with the ground-truth scan?*

| quantity | value |
|---|---|
| GT-render ink fraction | 0.0079 |
| real-scan ink fraction | 0.0080 |
| **IoU(render, scan)** | **0.003 - 0.009** |

The ink *amount* matches almost exactly, so the payloads and the rasteriser are scale-correct. The
*positions* do not overlap at all. Measured per line:

```
line 0:  scan ink occupies      cols 1-268   of the 1024-wide canvas
         trajectory occupies    cols 0-1024
line 1:  scan ink occupies      cols 3-217
         trajectory occupies    cols 0-793
```

**The scan is a padded strip whose ink fills only ~25% of the width, while the trajectory is
normalised to fill the entire width.** Two different frames.

### What this explains

1. **The image loss has been meaningless in all eight runs.** It sat at ~0.60 no matter what,
   because it was comparing ink in one frame against ink in another. No amount of training could
   reduce it.
2. **The model had no spatial anchor.** With the image term carrying no signal and the transcript
   giving only coarse content, nothing tied stroke geometry to a line layout -- which is exactly the
   conditional-mean smear we kept seeing. The dots and bands were a *symptom*, not the disease.
3. The join-rate metric matching (0.1298 sampled vs 0.1333 GT) was therefore misleading: strokes
   collapsed into a band, where gaps are accidentally the right size.

This is a data-construction defect, upstream of any model choice, and it invalidates the image-based
part of every run so far -- including the "image loss is flat" observation, which I had attributed to
task difficulty in §11 and §18. That attribution was wrong.

### The fix (not yet applied)

Map the trajectory into the **image's ink bounding box** instead of its own:

```
rows, cols = where(scan < 0.5)            # the ink actually present in the target
q_x = c0 + (x - x_lo) / span_x * (c1 - c0)
q_y = r0 + (y - y_lo) / span_y * (r1 - r0)
```

or, upstream and cleaner: crop the canonical images to their ink bbox so both are tight and fill the
canvas. The second needs a 20-minute rebuild of the canonical set; the first is ~10 lines in
`StrokeLineDataset.__getitem__`.

Until this is fixed, retraining is pointless -- the supervision has to be correct before the model is
judged. Training was stopped rather than burning GPU-hours on a broken target.

### What still stands

The style-encoder fix (writer loss 5.09 -> 1.17, `z_g` differentiation 0.05% -> 42%, style swap
0.0% -> 2.2%) is independent of this bug and remains valid. The parsers, the canonical builder, the
tokeniser, the sampler, the metrics and the hash-verified sync are all sound. The next run must use
aligned targets.

---

## 23. Fix applied, audited, and run 9 started

### The finding, stated precisely

The `.dgrl` scan and the `.wptt` trajectory are **not the same strokes**. CASIA's offline corpus was
written with an ordinary pen and scanned; the online corpus was written with an Anoto pen on
dot-pattern paper. They are paired by writer and text, not by geometry. Proof by measurement: after
cropping scans to their ink bbox -- which removes any frame offset -- IoU(rendered trajectory, scan)
stayed at 0.008-0.013. There is no geometric correspondence to align, because it does not exist.

Consequences for the method, and for the paper:

- **A scan cannot be a pixel target for a trajectory.** Supervising the render against a different
  writing session teaches the model to reproduce something it never drew. This is why the image loss
  never fell below ~0.60 in nine runs.
- **Scans remain valid as style references.** Writer identity transfers across renditions (that is
  what the style encoder measures: 42% differentiation, writer loss 1.17).
- The image target is now a **render of the line's own strokes, through the loss's own rasteriser**.

### Three bugs fixed

| bug | evidence | fix |
|---|---|---|
| scan used as the pixel target of a *different* writing session | IoU 0.008 even after bbox cropping | target = render of this line's payloads |
| target and loss used **different rasterisers** (numpy polyline ink 0.0124 vs splat 0.0014) | IoU 0.008 with the *same* strokes | target rendered with `render_strokes_splat` on the same payloads |
| image loss averaged over all timesteps, where high-noise `x0` is garbage | overfitting ONE line left it at 0.52 while the vector loss reached 0.09 | weight the image term to `t < 50` |

After the second fix, **IoU(render, target) = 1.0** exactly. After the third, a single-line overfit
reaches **diff 0.090** (before: 0.196).

### Sync hardened

A stale bundle passed the hash check because the box's manifest had come from the same stale upload.
`tools/sync_box.py verify` now also asserts **feature markers** -- `w_img`,
`mean(dim=(1,2,3))`, `render_strokes_splat(_tk`, `local_window`, `self.feat`,
`ink = (1.0 - refs)` -- so a tree missing a fix fails loudly even if its hashes agree with themselves.
That check, run against the box, is now a precondition for launching.

### Run 9 (in progress)

40 epochs, T=200, alignment + fixed style + self-consistent target + low-noise image loss.
Epoch 1, step 322: diff 0.162, img 0.506, writer 5.06, 0.532 s/step, no errors.

The question run 9 answers: with the image term finally meaningful, does the model place strokes where
a line's strokes belong, instead of collapsing to the conditional mean?

---

## 24. Run 9 at epoch 21: supervision is now correct, and the limit is data/capacity

Stopped at epoch 21/40 because the image term had plateaued: **img 0.514 -> 0.513 over 21 epochs**.

| signal | start | ep 21 | reading |
|---|---|---|---|
| diffusion | 0.193 | 0.112 | plateaued |
| continuity | 0.597 | 0.160 | plateaued |
| image | 0.514 | **0.513** | flat despite a target with IoU = 1.0 |
| writer (chance 6.38) | 4.69 | **1.11** | style is learned |
| sampled join rate (GT 0.1333) | - | **0.1404** | correct aggregate |

### The one thing this run did prove

In `results/sheet_r9_ep21.png` the **GT-stroke rows are legible Chinese handwriting**, while the
model's rows are dense horizontal smears. The payload representation, the Bézier fit and the
rasteriser can therefore represent a line faithfully -- the model simply is not producing it. That
separates "representation is broken" (false) from "model cannot generate it" (true).

### Why: data and capacity, in that order of testability

- **Data.** Runs 5-9 trained on 8,631 paired lines, a number imposed by the now-invalid requirement
  that every line have a scan. With the target rendered from the line's own strokes, the scan is no
  longer needed: **30,302 lines are available**, a 3.5x increase, for free.
- **Capacity.** CENT's denoiser is ~5.5M parameters (dim 256, depth 4). The Chinese handwriting
  baselines in the audit (DiffBrush, TGC-Diff, One-DM) are Stable-Diffusion UNets, 100M+ parameters,
  and train on far more than 30k lines. Training a 5.5M transformer from scratch to generate Chinese
  handwriting lines is at the far edge of what that scale can do.

### Run 10 (in progress): single-variable data test

Same model, same schedule (`T=200`), 15 epochs, `paired_only=False` -> **30,302 lines**. One variable
changes, so the outcome is interpretable:

- strokes appear -> data was the binding constraint; scale epochs.
- still smears -> capacity is the constraint; next run uses dim 384 / depth 6.

ETA ~2.4 h, ~$0.60.

---

## 25. Run 10: data falsified as the constraint

Stopped at epoch 7/15. Same model, same schedule, **30,302 lines instead of 8,631** (3.5x):

| run | lines | steps | diffusion | image |
|---|---|---|---|---|
| 9 | 8,631 | 6,076 | 0.1093 | 0.5131 |
| 10 | 30,302 | 8,191 | **0.1005** | **0.5136** |

3.5x data moved the vector loss by 0.009 and the image loss by **nothing**. The generated picture is
unchanged: dotted bands beside legible GT renders (`results/sheet_r10_ep7.png`).

### What has now been tested and ruled out

| hypothesis | how it was tested | verdict |
|---|---|---|
| parser / canonical building | native parsers, 144 tests, probes on real files | sound |
| timestep coverage | `T=1000 -> 200`, 81 visits per level | not the limit |
| style conditioning | encoder read paper; fixed to ink | fixed (writer 5.09 -> 1.11, z_g 0.05% -> 42%) |
| text-stroke alignment | position/scalars + local character window | implemented, no visible effect |
| image supervision | scan -> self-consistent render, IoU 1.0, low-noise masking | correct now, still flat |
| data volume | 8,631 -> 30,302 lines | **falsified** |

### What is left: capacity, or formulation

CENT's denoiser is ~5.5M parameters (dim 256, depth 4). Two candidate next steps:

**A. Scale the diffusion model** -- dim 512, depth 8 (~25M), more epochs. 1-2 runs, ~$1.5, uncertain:
the baselines are 100M+ SD UNets.

**B. Replace the generator with an autoregressive stroke model** -- stroke *k+1* conditioned on
stroke *k*'s endpoint and the character neighbourhood. This is the standard formulation for online
handwriting (sequence modelling, teacher forcing), it trains without a noise schedule or a timestep
coverage problem, and -- decisively for this paper -- **continuity becomes structural**: the next
stroke starts where the previous one ended, so "continuity by construction" is true of the
architecture rather than enforced by a loss term.

Recommendation: **B**. The current design asks a 5.5M diffusion transformer to model the joint
distribution of 100 chained strokes; the AR formulation gives it the chain for free, which is exactly
the paper's thesis. A is a capacity gamble on a formulation that has not yet produced a single legible
line in nine runs.
