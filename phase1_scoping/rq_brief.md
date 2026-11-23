# Research Question Brief — CENT

**Stage:** 1 (RESEARCH) · Phase 1 (Scoping)
**Role:** research_question_agent
**Working title:** *CENT: Structure-First Chinese Handwriting Generation via a Shared Stroke-Graph Latent*
**Date:** 2026-11-21

---

## 1. Primary research question

> Can Chinese handwriting be generated so that **glyph identity is guaranteed by construction** —
> the right strokes, in the right number, in the right order — while a writer's style is captured
> from only a **few isolated characters**, and while the **online trajectory** and the **offline
> image** of the same sample remain mutually consistent because both are decoded from one
> intermediate representation?

The question is a *design* question: it asks whether a particular representational commitment
(structure first, render second) is sufficient for the three properties above, and what it costs
in realism.

## 2. Why this question (problem statement)

Chinese handwriting generation has advanced rapidly through pixel-first and trajectory-first
models — GANwriting-style style tokens, visual-archetype content encodings (VATr), latent
diffusion with hybrid metric/classification style extractors (DiffusionPen), one-shot diffusion
(One-DM), online diffusion over point sequences (Diff-Writer), and unified latent diffusion
transformers (InkSpire). Each conditions on *character identity* and *style*, then synthesises
pixels or trajectories directly.

The consequence is a failure mode that the field measures but does not remove: a diffusion or
GAN sampler is free to produce a **plausible glyph that is the wrong character**. A missing
stroke turns 大 into 人; an extra stroke turns 日 into 目; a reordered or merged stroke produces a
character that is legible to a human as *something*, just not as the requested thing. Recognition-
based metrics (CER) detect this only after the fact, and they conflate it with rendering quality.
For Chinese — 3,755 common characters distinguished by small stroke-level differences — this is
the dominant error mode, and it is the one that matters for the two most credible applications:
handwriting synthesis as recogniser training data, and stroke-order-aware instruction.

## 3. Sub-questions

| # | Sub-question | Type |
|---|---|---|
| SQ1 | Is a **canonical stroke graph** (per-character stroke topology from a standard stroke-order lexicon; per-stroke geometry as cubic Bézier control points with pen state) a sufficient intermediate representation to couple online trajectories and offline renderings? | Representational |
| SQ2 | Can a writer's style be factorised into a **global** component (slant, handedness, aspect, ink) and a **local** component (per-stroke curvature, length, junction offsets), and inferred from k ≤ 5 **isolated character** references rather than whole text lines? | Modelling |
| SQ3 | Does constraining generation to the stroke graph **eliminate stroke-level errors** (missing / extra / substituted strokes) that pixel-first diffusion exhibits, and what does it cost in realism (FID) and naturalness? | Empirical |
| SQ4 | Can a differentiable Bézier renderer plus a light texture refiner produce realistic offline images from stroke graphs, supervised by **paired** online/offline writer data? | Engineering |
| SQ5 | What evaluation protocol measures Chinese handwriting generation honestly — legibility (recogniser CER), stroke correctness (stroke count / order edit distance), style fidelity (writer verification / retrieval), realism (FID), and human judgement — and how do these trade off? | Methodological |

## 4. FINER assessment

| Criterion | Assessment | Note |
|---|---|---|
| **Feasible** | Moderate–high | Paired online (`CASIA-OLHWDB1.x`, stroke-segmented `.pot`) and offline (`CASIA-HWDB1.x`) isolated-character data from the same 1,020 writers is available for academic research, and the pairing was verified in Phase 2 (see `phase2_literature/synthesis.md` §2); stroke-order topology is available from an open lexicon (`Make Me A Hanzi` `graphics.txt`, 9,000+ characters, **Arphic Public License**). Compute is the binding constraint, not data access. |
| **Interesting** | High | Reframes the field's central failure (illegible or wrong glyphs) as a *representation* problem rather than a sampling problem. |
| **Novel** | Moderate–high | Structure-first generation exists piecemeal (radical-constrained GAN; hierarchical style aggregation), but no prior work uses a **per-stroke graph as the single latent that both modalities decode from**, with topology fixed by construction. |
| **Ethical** | Sensitive | Handwriting imitation is dual-use (forgery, impersonation). Requires an explicit dual-use section, consent framing, and a synthetic-data marking/watermark proposal. |
| **Relevant** | High | Serves HTR data augmentation (the field's own standard use), stroke-order instruction, and personal handwriting fonts for accessibility. |

## 5. Scope boundaries

**In scope**
- Isolated Chinese characters (GB2312 level-1, 3,755 classes), offline image *and* online trajectory.
- Few-shot style adaptation from k ∈ {1, 5} **isolated characters** (not text lines).
- Seen/unseen writer × seen/unseen character evaluation grid.

**Out of scope (declared, with reasons)**
- Multi-line layout and line-level spacing — treated as complementary prior work; a layout module is orthogonal to the stroke-graph claim and would confound it.
- Non-Chinese scripts; Latin and Japanese results are cited only as context.
- Signature generation and writer identification *attacks* — excluded on dual-use grounds.
- Real-time on-device deployment and model compression.
- Any empirical claim about state-of-the-art superiority **until experiments are run**; see the Stage-1 checkpoint decision on empirical scope.

## 6. Positioning against the nearest prior work

| Prior work | What it fixes | What CENT changes |
|---|---|---|
| GANwriting (Kang et al., 2020) | Style tokens for few-shot word images | Latin words; no structural prior; pixel-first |
| VATr (Pippi et al., CVPR 2023) | Visual archetypes as content queries, long-tail characters | Still pixel-first: topology unconstrained |
| DiffusionPen (Nikolaidou et al., 2024) | 5-shot latent diffusion + hybrid style extractor | Style space improved; glyph topology still free |
| One-DM (Dai et al., ECCV 2024) | One-shot style mimicker; includes a Chinese setting | One reference *line*, not isolated characters; no stroke constraint |
| Diff-Writer (Ren et al.) | Non-autoregressive online Chinese characters via DDPM | Point sequence, not a stroke graph; offline image not produced |
| HiSAC (2025) | Radical/component/stroke style aggregation, spatial + frequency | Style *aggregation* is hierarchical; the generated object is still an image/trajectory. CENT makes the stroke graph the generated object |
| SDT (Dai et al., 2023) | Writer-level vs character-level style disentanglement, contrastive | Disentangles style; does not constrain glyph structure |
| InkSpire (ICLR 2026) | Unified style/content/noise latent, bilingual, multi-line infilling | Unified latent but not stroke-explicit; correctness still emergent |
| RC-GAN (Ye et al., 2024) | Radical-consistency loss for HCCAG | Radical-level constraint as a *loss*; CENT fixes topology as a *constraint* |

**The gap CENT occupies:** every listed system makes glyph correctness an *emergent* property of a
learned sampler. CENT makes it a *structural* property of the representation, and derives online
and offline outputs from the same latent.

## 7. Provisional contributions (to be defended or cut after Phase 3)

1. **A shared stroke-graph latent** that is the single generative target; online trajectory and
   offline image are both deterministic decodes of it.
2. **Topology-constrained generation:** stroke count and order are fixed per character by an open
   stroke-order lexicon; the model generates only geometry. Wrong-glyph errors of the
   missing/extra-stroke kind are eliminated by construction, not by loss shaping.
3. **Isolated-character few-shot style**: a global/local style factorisation inferred from k ≤ 5
   character images, with low-rank modulation of the stroke generator.
4. **A paired-supervision training scheme** using online/offline data from the same writers, and a
   differentiable Bézier renderer that makes the whole pipeline trainable end-to-end.
5. **An honest evaluation protocol** that separates stroke correctness from rendering realism, and
   reports the trade-off rather than a single headline number.

## 8. Declared threats to validity (carried forward)

- **T1 — Stroke-extraction dependence.** The claim rests on stroke-level ground truth. If the
  paired online/offline correspondence is weaker than assumed (different sessions, different
  content), the renderer's supervision becomes noisy and the method degrades to online-only.
  Must be verified against the CASIA documentation before any experiment is designed around it.
- **T2 — Constrained generation can look synthetic.** Fixing topology may cost naturalness: real
  writers merge strokes, lift the pen late, and vary stroke count in cursive. A structure-first
  model may produce "correct but rigid" glyphs. This is a real trade-off, not a bug to hide.
- **T3 — Benchmark saturation.** ICDAR-2013 Chinese handwriting data (60 writers, 3,755 chars) is
  small; writer-disjoint splits leave few test writers. Results may not generalise to the full
  1,020-writer set.
- **T4 — Metric validity.** CER depends on the recogniser; FID on Inception features trained on
  natural images. Neither is a clean measure of handwriting quality. The protocol must include a
  human judgement component or explicitly declare the absence of one.
- **T5 — Evaluation must not outrun the evidence.** No number in the eventual paper may be
  reported unless it was produced by a run whose provenance is recorded. This is a hard constraint
  on the whole pipeline, not a stylistic preference.

## 9. Naming

**CENT** = **C**hin**E**se ha**N**dwri**T**ing. The expansion is load-bearing for the paper's
framing (a Chinese-specific model, not a multilingual one with a Chinese benchmark).
