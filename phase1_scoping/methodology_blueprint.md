# Methodology Blueprint — CENT

**Stage:** 1 (RESEARCH) · Phase 1 (Scoping)
**Role:** research_architect_agent
**Companion:** `rq_brief.md` (same directory)
**Date:** 2026-11-21

---

## 1. Research paradigm and design

| Dimension | Choice | Justification |
|---|---|---|
| Paradigm | Design science + positivist evaluation | The contribution is an artefact (CENT) and a claim about representational sufficiency; both must be built and measured. |
| Design | Computational modelling with controlled, pre-specified experiments | Ablations are the primary evidence: each design commitment must be isolated. |
| Data | Secondary (existing research corpora) | No new collection; paired online/offline writer data already exists. |
| Inference | Frequentist with bootstrap CIs, plus a declared equivalence/worse-case bound for the legibility claim | We claim a *reduction in stroke errors*, not a point estimate of superiority; an interval is the honest object. |

**Falsifiability statement.** The central claim — that topology-constrained generation reduces
stroke-level errors — is falsified if, at equal realism budget, the constrained model's stroke
error rate is not lower than the best unconstrained baseline's (95% CI excluding zero), or if its
FID cost exceeds a pre-declared acceptability band (set at protocol time, before runs).

## 2. Data strategy

### 2.1 Primary corpora

| Corpus | Content | Role |
|---|---|---|
| `CASIA-OLHWDB1.0–1.2` | Online isolated characters, stroke-segmented pen data, 1,020 writers | Stroke-graph ground truth (topology + geometry + order) |
| `CASIA-HWDB1.0–1.2` | Offline isolated character images (scans), same writer pool | Renderer supervision; offline evaluation target |
| `ICDAR-2013 Chinese handwriting` competition set | 60 writers × 3,755 characters, offline | Public secondary benchmark for comparability with One-DM / InkSpire |

**Load-bearing assumption to verify before any experiment (threat T1).** The online and offline
corpora are assumed to share writers *and* character inventory, enabling paired supervision. This
must be confirmed against the CASIA database documentation (Liu et al., ICDAR 2011) — specifically
whether the same writer wrote the same character in both sessions. **If unpaired**, the plan
falls back to: (a) renderer trained with a weak image-level perceptual/adversarial objective and
cycle-consistency to a stroke parser, and (b) the online-only claim, with the offline claim
demoted to future work. This decision point is recorded, not assumed.

### 2.2 Stroke topology source

Per-character canonical stroke count and order from an open lexicon (`Make Me A Hanzi`,
`graphics.txt`, derived from the Arphic PL KaitiM GB / PL UKai fonts) — a *deterministic* input, not
learned. Licence, verified 2026-11-21: `graphics.txt` (stroke paths and medians, in proper stroke
order, 9,000+ characters) is under the **Arphic Public License**; `dictionary.txt` is under
LGPL-3.0-or-later. The two are separate works with separate terms, and the model's topology table is
a derived work of the former, so it must be redistributed under the Arphic Public License, not
CC-BY and not the MIT licence of the `hanzi-writer` animation code. The lexicon is derived from a
*font*, so its geometry is typographic: CENT takes only **topology** (stroke count and order) from
it and learns geometry from handwriting. Characters whose lexicon stroke count conflicts with the
corpus's majority stroke count are excluded and counted (reported as a coverage statistic).

### 2.3 Splits

- **Writer-disjoint**: 860 train / 60 validation / 100 test writers (subject to the corpus's actual
  partition; the released train/test partition is respected where it exists).
- **Four-cell evaluation grid**, following the field's convention:
  - IV-S (seen character, seen writer) · IV-U (seen char, unseen writer) · OOV-S (unseen char, seen
    writer) · OOV-U (unseen char, unseen writer).
- **Few-shot references**: k ∈ {1, 5} isolated characters, disjoint from the target characters.
- **Seeds**: 3 training seeds per configuration; report mean ± 95% bootstrap CI over test items.

## 3. Model specification (the artefact under test)

CENT is a two-stage decoder over one latent. Only the stroke-graph latent is stochastic.

```
character ID c ──► topology lookup (deterministic: N_c strokes, canonical order)
                      │
k style images ──► style encoder ──► (z_global, {z_local per stroke-type})
                      │
                      ▼
        stroke-graph diffusion  ──►  G = {(b_i, pen_i)}_{i=1..N_c}     (Bézier ctrl pts + pen state)
                      │                        │
                      │                        ├──► Bézier decode ──► online trajectory (x, y, t, pen)
                      │                        └──► differentiable renderer ──► ink map ──► refiner ──► image
                      ▼
              content verifier (stroke-graph → glyph class) : training-time loss only
```

**Components**
1. **Topology oracle** — deterministic; returns stroke count/order per character.
2. **Style encoder** — hybrid metric + classification training (the DiffusionPen lesson), produces a
   global vector and per-stroke-type local vectors. Trained with triplet + CE + a content-invariance
   objective (style features must not predict character identity).
3. **Stroke-graph denoiser** — 1-D U-Net over the concatenated stroke sequence; cross-attends to
   style; conditioned on character embedding. Non-autoregressive (the Diff-Writer lesson: global
   structure beats sequential decoding).
4. **Differentiable Bézier renderer** — rasterises predicted strokes to an ink map with a fixed
   round-cap pen model; differentiable w.r.t. control points, so image losses reach the generator.
5. **Texture refiner** — small conv refiner for ink bleed, pressure variation, paper texture; trained
   adversarially or with a perceptual loss.
6. **Content verifier** — a stroke-graph classifier used *only in training* as a legibility loss
   (does the predicted graph decode to the requested character?). Not used at inference.

### 3.2 Out-of-vocabulary content path (Devil's Advocate C4)

An unseen character has no learned identity embedding, so identity conditioning alone cannot
generate it. For OOV characters the denoiser is conditioned on a **rendered archetype** — a 64×64
standard-font image of the character passed through a small convolutional content encoder — instead
of an ID embedding. Topology still comes from the lexicon (known for every character in the
inventory). The OOV claim is therefore precise and testable: **unseen geometry under known
topology**. This is the prediction the representation should win; if it does not, the claim fails.
In-vocabulary characters use the ID embedding plus the rendered archetype (the archetype path is
always available, which also removes the train/test asymmetry).

## 4. Evaluation protocol

| Construct | Metric | Instrument | Failure it detects |
|---|---|---|---|
| Legibility | Character error rate (CER) of a fixed, frozen HTR recogniser on rendered images | Pretrained Chinese HTR (reported exactly; same checkpoint for all methods) | Wrong character |
| Stroke correctness | Stroke-count accuracy; stroke-order normalised edit distance **and** order-equivalence-class distance (orders that yield the same glyph under a declared merge rule); stroke-level F1 after matching | Hungarian matching between predicted and reference stroke sequences; both order measures reported (Devil's Advocate C3) | The headline claim |
| Realism | FID (character-level, per-writer averaged) | Fixed Inception feature extractor; same preprocessing for all methods | Blurry / artificial |
| Style fidelity | Writer verification accuracy (same/different writer), writer retrieval top-1 | Held-out writer-disjoint protocol, frozen encoder | Style collapse |
| Style-content disentanglement | Character classification accuracy from generated images (should be high) × writer classification (should be high) | Two frozen probes | Entanglement |
| Human judgement | Pairwise forced choice: real vs generated; and "same writer?" | N ≥ 30 native readers, pre-registered protocol, IRB-style consent | Metric blind spots |

**Critical separation.** Legibility and stroke correctness are reported *separately* from realism.
A single headline number would hide exactly the trade-off CENT is about.

**Baselines** (reproduced where code exists; otherwise cited with their reported numbers *and a
note that they were not re-run*): GANwriting, HWT, VATr, One-DM, DiffusionPen, Diff-Writer,
WordStylist, and — if released in time — HiSAC.

## 5. Ablations (each isolates one commitment)

| ID | Ablation | Question |
|---|---|---|
| A1 | Direct pixel diffusion (no stroke latent) | Is the stroke latent necessary, or does topology-constrained *loss* suffice? |
| A2 | Topology constraint off (free stroke count/order) | Quantifies the constraint's contribution to stroke correctness |
| A3 | Global style only (no per-stroke local style) | Is the local factorisation earning its complexity? (SDT predicts yes) |
| A4 | Neural renderer only (no differentiable Bézier) | Value of the analytic renderer |
| A5 | k = 1 vs 5 vs 15 references | Sample-efficiency curve |
| A6 | Verifier loss on/off | Whether the training-time legibility loss adds anything beyond the constraint |
| A7 | Cursive tolerance (allow stroke merging) | The T2 realism/latency trade-off |

## 6. Validity and reliability

- **Construct validity**: stroke-order edit distance is only meaningful if the reference order is
  well-defined; report lexicon-corpus disagreement rate, and report only the order-equivalence-class
  measure as the primary stroke-order statistic. The stroke-order claim is restricted to
  regular-script (楷書) writing; cursive stroke merging is out of scope and is measured separately
  as ablation A7.
- **Internal validity**: identical splits, identical frozen evaluators, identical preprocessing,
  seeds reported, no test-set tuning.
- **External validity**: two corpora (CASIA, ICDAR-2013); explicitly no claim beyond Chinese
  isolated characters.
- **Statistical validity**: bootstrap over test *items* (characters), not pixels; paired tests
  against each baseline; multiplicity acknowledged (Holm correction where a family is tested).
- **Reliability**: deterministic evaluators pinned by hash; the generative runs are not
  byte-reproducible and are reported as such with seeds and config.

## 7. Ethics (declared now, executed at Stage 2.5)

- **Dual use**: handwriting imitation enables forgery. The paper will state the risk, restrict the
  release to a research licence with a provenance/watermarking mechanism, and refuse signature data.
- **Consent**: CASIA/ICDAR corpora were collected under research consent; we rely on their terms and
  add nothing new. Human evaluation participants are consented and compensated.
- **AI disclosure**: the manuscript declares AI-assisted tooling.
- **Attribution**: every baseline's reported number is cited to its source; none is invented.

## 8. Compute and experiment budget (declared, not spent)

| Item | Estimate |
|---|---|
| Style encoder pretraining | ~6 GPU-hours (A100-class) |
| Stroke-graph diffusion (per seed) | ~40 GPU-hours |
| Renderer + refiner (per seed) | ~20 GPU-hours |
| Evaluation (all metrics, all baselines) | ~15 GPU-hours |
| Ablations A1–A7 (single seed each, then 3 seeds for A1/A2) | ~120 GPU-hours |
| **Total** | **~600–800 GPU-hours** |

This exceeds what can be executed inside this session. The empirical-scope decision is therefore a
**user checkpoint**, not an engineering default: the paper is written either (a) as a
method + pre-registered protocol with **no reported results**, or (b) after real runs whose
provenance is recorded in a ledger. Option (a) is the only honest default until compute is
committed; option (b) is the only path to a results-bearing paper.

## 9. Threats carried from the RQ brief

T1 (pairing) — verified or the offline claim is demoted.
T2 (rigidity) — ablation A7 quantifies it; the paper must report it as a finding, not bury it.
T3 (small test set) — bootstrap CIs will be wide; report them.
T4 (metric validity) — human study required, or absence declared.
T5 (no result without provenance) — enforced by the experiment ledger and the Stage 2.5/4.5 gates.
