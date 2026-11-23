# CENT v3 — offline analysis of the conditioning failure

_No GPU running. All figures below are measurements already made; nothing here is a projection._

## 1. What is measured

| experiment | data | steps | final stroke NLL | free-run error vs GT |
|---|---|---|---|---|
| S1 single line | 1 line | 14,000 | **−23.2** | **0.072** (teacher-forced 0.065) |
| 4-line conditioning | 4 lines, 2 writers | 6,000 | **−1.66** | **0.44 – 2.13** |

The single-line case is solved: free-running error is within 0.007 of teacher-forced, so exposure
bias is negligible at that scale, and the rendered line is the target line.

## 2. The conditioning matrix (the result that matters)

All 16 (text × style) combinations, error to each ground truth, on the 4-line set:

```
                 GT0    GT1    GT2    GT3
text0 style0   2.128  1.921  1.315  2.056   argmin=2
text0 style2   2.098  1.893  1.292  2.026   argmin=2
text1 style0   1.516  1.525  0.990  1.447   argmin=2
text2 style0   0.633  0.616  0.544  0.559   argmin=2
text3 style0   0.592  0.572  0.440  0.537   argmin=2
```

1. **argmin = 2 in every one of the 16 cells.** The diagonal is never selected: the output does
   not follow the transcript.
2. **Style is inert.** Swapping reference lines changes the error by ~0.03; swapping the text
   changes it by ~1.5 — a 50× asymmetry.
3. **The output varies with text but always lands near GT2** → one memorised attractor, emitted
   with small variations. Mode collapse, not a retrieval of the correct line.

## 3. Confounder that must be removed first

The 4-line run ended at NLL **−1.66** after 6,000 steps. The single line needed 14,000 steps to
reach −23. Four lines therefore received roughly **one tenth** of the optimisation the one-line
case required, *and* four times the data. At NLL −1.66 the model has not fitted the training set at
all, so "conditioning is inert" and "the model is untrained" are currently **indistinguishable**.

This is the single most important caveat: no architectural conclusion is licensed yet.

## 4. Hypotheses, ranked, each with its test

| # | hypothesis | test | cost |
|---|---|---|---|
| H1 | under-training only | train the 4-line set to NLL ≤ −15 (~30-40k steps) | ~1.5 h, $0.25 |
| H2 | capacity allows ignoring the text | same, dim 192 / depth 4 — too small to memorise 4 sequences independently | same run |
| H3 | text path wired but unused | decode with the transcript replaced by random ids; output must change | minutes |
| H4 | character carrier still wrong | print one batch's `char_idx` per position against its text; they must correspond | minutes |
| H5 | style path inert for architectural reasons | swap refs on the **single-line** model (where the model is known good) and measure output distance | minutes |
| H6 | mode collapse intrinsic to AR on 4 samples | add samples (8, then 16); the diagonal should appear as data grows | ~2 h |

Note on H4: a lagged character carrier (position *k* carrying the character of stroke *k*−1) was
found and fixed. With a lag the model literally cannot know which character it is starting, which
fits every symptom. That fix has **not yet been tested at convergence** — the 6,000-step run used it
but never fitted the data.

## 5. Recommended sequence with acceptance criteria

| step | run | accept | reject → next |
|---|---|---|---|
| 1 | 4 lines, char-carrier fix, to NLL ≤ −15 | argmin on the diagonal in ≥ 3 of 4 rows | → step 2 |
| 2 | same, dim 192/depth 4 | diagonal holds with less capacity | → step 3 |
| 3 | random-transcript ablation | output changes appreciably | text path unwired → redesign M6 |
| 4 | 8 then 16 lines | diagonal holds as data grows | mode collapse → H6 |
| 5 | only now: 30k lines | join-rate error beats 0.204 / 0.796 baselines on sampled output | — |

The full campaign (≈2.6 h, $0.54 for 15 epochs) remains **blocked** until step 1 passes.

## 6. What is proven and can be built on

- **Data pipeline**: native `.dgrl`/`.wptt` parsers; 78 strokes per line including the 14
  inter-character **connectors** (an earlier revision silently dropped them, 64 of 78); per-stroke
  character labels with connectors inheriting the preceding character.
- **Token ↔ geometry**: `start_k = end_{k−1} + Δ_k`, verified to **5.96e-08** against the original
  polylines. Precision matters: a `cumsum` reconstruction once produced y = −11.85 on a [0,1]
  canvas and every "failed generation" image for several rounds was that artefact.
- **Decoding**: dominant-mixture-component, not the mixture mean (the mean collapses bimodal
  stroke directions to zero vertical extent — the flat-dash artefact); relative path for both
  teacher-forced and free-running; sampling is still unusable (random walk, error 8.8).
- **Count head**: exact (CE 0.0008); replaces the per-stroke `stop` classifier that degenerated to
  one stroke per character.
- **Rendering**: `draw_payload` asserts the number of strokes drawn, and the aspect must match the
  model canvas (1024:64). Earlier figures stretched handwriting 2.4× vertically, which is why the
  ground truth itself looked unreadable.

## 7. Harness checklist (each item cost a run)

1. Never launch while a checkpoint is being written — wait for the process to exit.
2. Never `pgrep -f` a pattern that appears in the command itself; kill by the PIDs from
   `nvidia-smi --query-compute-apps`.
3. Print the selected data and **assert** it before training (a selection loop silently collected
   280 writers, then 1 line per writer).
4. Keep local and box trees identical and verify by hash **and** feature markers before every run.
5. A loss curve alone is not evidence: single-line NLL −40 was once an identity map caused by an
   unshifted teacher-forcing target. Always report the generation-time metric beside the loss.

## 8. Bottom line

CENT v3 **can memorise and free-run a single line** and its data/geometry/rendering stack is
verified to floating-point. It has **not** been shown to read the transcript or the style, and on
four lines it collapses onto one attractor. The next action is one converged 4-line run
(≈$0.25) with the character-carrier fix — not the full campaign.
