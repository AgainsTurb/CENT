# Fix plan after the generation collapse

_Status at time of writing: training indicators green, two-row visual wrong. Four gate attempts,
each with a lower loss and a worse picture._

| attempt | NLL | free_err | ooc/78 | what the picture showed |
|---|---|---|---|---|
| no clamp | −18 | 0.210 | 51 | line, half the strokes off-canvas |
| anchor clamp | −25 | 0.349 | 52 | clusters (anchor/stroke unit mismatch) |
| page clamp | −25 | 0.339 | 39 | compressed smear, x crowded left |
| + standardisation | −30 | 0.264 | 39 | repeated identical loops |

## What is established

1. **Data and tokenisation are correct.** 78 strokes/line including 14 connectors; counts exact
   78/78; recurrence verified to 5.96e-08 against the original polylines.
2. **Layout is learnable.** Anchor L1 falls 4.0 -> 0.047.
3. **The loss does not predict the render.** NLL improves at every step above while the picture
   degrades. Any claim resting on the loss curve is void.
4. **Δ is systematically too small.** Measured `gt_dx_mean 0.0041` vs `gen_dx_mean 0.0007`, while
   `dy` was nearly right. That is what produced the left-compressed smear.
5. **Standardisation changed the objective's scale.** NLL is not comparable across that change;
   an NLL of −30 on unit-variance targets means σ has collapsed, i.e. the mixture is overconfident
   and the likelihood is being gamed. Must report mean log-σ, not only NLL.
6. **Deterministic argmax over a mixture is the decoder defect.** Mixture mean flattened bimodal
   stroke directions (dashes); argmax component collapses to one archetype (loops). Both are the
   same underlying error: decoding a multimodal conditional with a deterministic rule.

## Ordered plan — one change at a time, gate on the picture

**Step 0 — re-establish the baseline.** Config: 1 line, full prefix + BOS, mixture mean, **no
anchors, no count-driven loop, raw targets**, 14k steps. Expected free_err ≈ 0.07 and a legible
line. If it does not reproduce, the regression is elsewhere and that is the finding. Tag the
commit `baseline-legible` on success.

**Step 1 — σ floor and σ reporting.** Clamp log-σ from below (per dimension, e.g. 1e-2 in
standardised units) and log the mean per-dimension σ every 100 steps. Acceptance: σ stays away
from the floor and the picture does not degrade.

**Step 2 — sequence-level decoding instead of step-level argmax.** Have `decode` return the
generated token tensor, then choose among N sampled candidates by **sequence log-likelihood**
(one teacher-forced forward over each candidate). This is the principled replacement for both the
mixture mean and the argmax component, and it needs no anchors or unit conversion. Acceptance:
two-row gate legible; `ooc -> small`.

**Step 3 — anchors as conditioning, in the stroke frame.** Only after Step 2 passes. Convert
anchors to normalised coordinates (`/canvas_w`, `/canvas_h`) before they enter memory, assert that
anchor and stroke ranges overlap, and never use an anchor as a placement rectangle.

**Step 4 — standardisation, with σ monitored.** Re-introduce, and reject the change if σ collapses.

**Step 5 — more lines.** 2 -> 4 -> 8, with the gate at every step. Only then the full corpus.

## Non-negotiable instrumentation

- Every config switch must assert that its output **differs** from the alternative before any
  comparison is believed. (The A/B run printed `err 0.3527 ooc 25` for three "different" variants:
  the patches never applied. Silent no-op patches have cost several runs.)
- Every geometry value gets a printed range and a frame assertion.
- The two-row image is the acceptance test; the loss is telemetry.

## What would justify a pretrained prior

Only after Steps 0–2: if `ooc ≈ 0` and `err < 0.15` but the row is still not legible, the failure
is appearance, and a frozen-VAE perceptual loss on the rendered line is the right instrument
(InkSpire: FID 7.92 frozen-VAE vs 20.48 from-scratch, CR 92.9 vs 59.3). Until then a prior would
render misplaced strokes more attractively.
