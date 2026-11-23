# Phase 2 — Investigation: annotated corpus, prior-art proximity, gap analysis

**Stage:** 1 (RESEARCH) · Phase 2 (Investigation) → feeds Phase 3 (Analysis)
**Role:** bibliography_agent + source_verification_agent (consolidated record)
**Date:** 2026-11-21
**Read-scope legend:** `abs` = abstract/highlights only; `full` = full text retrieved;
`schema` = registry metadata only (Crossref/arXiv). Method-level claims are made only where the
read-scope supports them.

---

## 1. Resolution status

Every candidate reference was resolved against Crossref, then arXiv, by
`refs/verify_refs.py` (accept ratio ≥ 0.87 on normalised title similarity). Results in
`refs/verified_refs.json` and `refs/verification_report.md`. **A reference appears in the manuscript
only if its status is `ACCEPTED`.** Unmatched candidates are listed in the report and are excluded —
including one deliberately planted probe entry, which is expected to fail.

## 2. Load-bearing verification: the CASIA pairing (resolves Devil's Advocate C2)

The two-modality design assumed that the online and offline corpora could be paired per writer and
per character. The supplier's own documentation establishes this in the strong form needed:

- The samples were "produced by 1,020 writers using **Anoto pen on papers** for obtaining **both**
  online trajectory data and offline images" — one act of writing yields both modalities.
- "The same partitioning of writers is taken between online and offline datasets and between
  isolated characters and handwritten texts datasets."
- File formats confirm the granularity: a `.POT` file stores, per sample, "the class label (4-byte
  GB code), **number of strokes**, and sequence of (x, y) coordinates of stroke points with
  **(−1,0) denoting pen lift**"; the matching `.GNT` file stores the grayscale image with the
  corresponding GB class label.
- Scale: isolated-character datasets contain ≈3.9 M samples over 7,356 classes (7,185 Chinese
  characters + 171 symbols), split into DB1.0–1.2.

**Consequence, stated precisely:** pairing is guaranteed at **(writer, character)** granularity, and
stroke segmentation is native to the online data. It is *not* guaranteed that the specific physical
instance whose strokes we read is the same instance whose raster we render — different samples of the
same character by the same writer are different acts of writing. CENT therefore treats the image as a
*style-consistent target*, not as a pixel-exact rendering of the trajectory. This distinction is a
limitation to state in the paper, not a detail to bury. Source: `casia`, read: `full` (PDF text).

**Fallback no longer needed.** The "unpaired" branch in the blueprint (cycle-consistency to a stroke
parser) stays documented as a contingency but is not the plan.

## 3. Field map (four families, not one)

| Family | Representative work | Objective | Generative object |
|---|---|---|---|
| **A. Few-shot Latin HTG** | `ganwriting`, `hwt`, `vatr`, `vatr_pp`, `wordstylist`, `diffusionpen`, `one_dm`, `emuru`, `hi_gan` | Render a word/line in a writer's style from k style samples | Pixels (directly or via a VAE latent) |
| **B. Chinese offline HTG** | `one_dm` (Chinese setting), `metascript`, `rcgan`, `inkspire` | Same, for Chinese characters / lines | Pixels |
| **C. Chinese online (trajectory) HTG** | `diffwriter`, `layout_glyph`, `sdt`, `hisac` | Generate pen trajectories in a style | Point sequences (+ pen state) |
| **D. Stroke-structured font generation** | `scfont`, `difffont`, `cffont`, `fontdiffuser`, `dp_font`, `stroke2font`, `lvgh`, `lffont`, `mxfont`, `dgfont` | Produce a *typeface*, not an individual's hand | Glyph outlines / stroke vectors / pixels |

CENT sits at **C ∩ D**: an individual's hand (C's objective) generated as strokes (D's representation),
rendered to an image (B's output) while remaining decodable as a trajectory (C's output).

## 4. What is already established (do not re-claim as novel)

1. **A hybrid metric-learning + classification style extractor gives a better-conditioned style
   space than either alone** — `diffusionpen`, read: `full` (method section retrieved).
2. **Encoding content as a *visual archetype* rather than a one-hot ID improves rare-character
   generation** — `vatr` (CVPR 2023), `abs`+`full` (method section retrieved). This is the basis of
   CENT's OOV content path (Devil's Advocate C4).
3. **Non-autoregressive generation captures global character structure better than sequential
   decoding for Chinese** — `diffwriter`, `abs`; the authors state autoregressive decoding "often
   makes it difficult to capture the global structure".
4. **Stroke-order information, used as a conditioning signal, measurably reduces stroke errors** —
   `dp_font` (IJCAI 2024), `full`. Their ablation shows that conditioning on stroke order (and
   count) produces better results than no stroke condition, one-bit encoding, or count-only
   encoding, and that models without explicit stroke-order encoding "may generate characters with
   stroke errors". **This is the closest prior art to CENT's central claim and must be cited as
   such.**
5. **Decomposing characters into radicals / components / strokes improves style fidelity and
   generalisation** — `hisac` (2025), `abs`; the authors build fine-grained spatial correspondences
   at each level plus a multi-band frequency encoder, and plug into either a Transformer (online) or
   a diffusion decoder (offline).
6. **Writer-level and character-level style are separable, and the separation matters** — `sdt`,
   `abs`+`full` (abstract and method highlights): two contrastive objectives capture writer
   commonalities and per-sample detail, and "the two learned style representations provide
   information at different frequency magnitudes".
7. **Unifying style, content and noise in one latent is feasible and removes handcrafted encoders** —
   `inkspire` (ICLR 2026), `full` (paper PDF text retrieved); trained bilingually on IAM and
   CASIA-HWDB2.0–2.2 + ICDAR2013.
8. **Vector stroke sequences are a viable generative target for glyphs** — `lvgh` (2025), `full`:
   strokes as cubic Bézier parameter groups, predicted by an LLM; `stroke2font` (2026), `abs`:
   "stroke element graph" with Bézier parameterisation, structure decoupled from style; `scfont`
   (AAAI 2019), `abs`: transfers "writing trajectories with separated strokes" from a reference
   font.

## 5. What remains genuinely open (CENT's niche, stated narrowly)

1. **Hard topology constraint for *handwriting*, not conditioning.** DP-Font *conditions* on stroke
   order; a conditioning signal can be ignored by the sampler, and their own text attributes residual
   stroke errors to models lacking it. No prior work fixes stroke count and order *by construction* in
   a handwriting generator, where intra-writer variability (stroke merging, late pen-lift) makes the
   constraint harder than in font synthesis.
2. **One latent, two modalities, from handwriting.** Font work (D) generates outlines; trajectory
   work (C) generates points. Nobody decodes *the same* stroke-graph sample to both an online
   trajectory and an offline image, so nobody can state that the two are consistent. HiSAC couples
   the two tasks through a *shared style model* with separate decoders, which is a weaker coupling
   than a shared *content-and-geometry* latent.
3. **Few-shot from isolated characters, not lines.** The field's convention is k = 5–15 *text lines*,
   i.e. hundreds of characters per style sample. The realistic personalisation setting — "here are
   five characters I wrote" — is a strictly harder style-inference problem and is under-benchmarked.
4. **A metric that separates stroke correctness from rendering realism.** CER conflates "wrong
   character" with "ugly rendering"; FID ignores glyph identity entirely. The field reports both but
   does not dissociate them, so a method that trades realism for correctness and one that trades
   correctness for realism can post the same headline.

## 6. Prior-art proximity table (the honesty device the Devil's Advocate asked for)

| Work | Same representation? | Same setting? | Same output? | Delta CENT claims |
|---|---|---|---|---|
| `dp_font` | Stroke order as a **condition** (36-dim type vector) | Fonts, not handwriting | Image | Condition → **hard constraint**; handwriting; stroke geometry generated, not just conditioned |
| `scfont` | Separated-stroke trajectories from a *reference font* | Fonts | Outline | No reference trajectories available for a hand; geometry must be *generated* |
| `stroke2font` | "Stroke element graph" + Bézier | Fonts | Vector glyph | Handwriting; paired multi-modal decode; few-shot writer style |
| `lvgh` | Ordered Bézier strokes as tokens | Any glyph | Vector SVG | Not a writer's hand; autoregressive LLM; no offline image |
| `hisac` | Radical/component/stroke hierarchy | Chinese handwriting ✓ | Online **or** offline (separate decoders) | One latent, both modalities; topology fixed, not learned |
| `diffwriter` | Point sequence + pen state | Chinese online ✓ | Trajectory | Stroke-graph (stroke-indexed), plus an image |
| `inkspire` | Unified style/content/noise latent | Bilingual HTG ✓ | Image | Stroke-explicit latent; topology constrained |
| `one_dm` | One-shot style | Chinese ✓ (60 writers) | Image | Few-shot from isolated characters; stroke constraint |

**The honest one-sentence delta:** *CENT does not propose a better sampler or a better style encoder.
It proposes that the generated object be the stroke graph itself, and it tests whether that
representational choice buys stroke-level correctness and cross-modal consistency at a stated cost
in realism.*

## 7. Contradictions and uncertainties to carry forward

- **Rigidity vs correctness.** DP-Font shows stroke constraints help; the same literature shows
  real writers merge strokes and deviate from canonical order (`casia` corpus includes cursive
  writers; `hisac` notes difficulty "mimicking styles with extensive cursive connections"). These
  two observations predict opposite things about CENT's naturalness. Ablation A7 exists to resolve
  it empirically rather than by assertion.
- **Style-sample size.** `diffusionpen` uses k=5, `vatr`/`ganwriting` k=15, `one_dm` k=1 but
  recommends references longer than four characters. There is no settled convention for *isolated
  characters* as references — CENT's k ∈ {1,5} grid is a choice, not a standard, and must be
  justified.
- **Benchmark fragmentation.** Chinese HTG papers variously report on ICDAR-2013 (60 writers), CASIA
  HWDB1.x (1,020 writers) or HWDB2.x lines, which makes cross-paper numbers non-comparable. CENT
  must report on a fixed evaluation set and say plainly that cited baseline numbers were not
  re-run.

## 8. Source-quality assessment

All accepted references are peer-reviewed venue papers, arXiv preprints, or registry-indexed
journal articles. Preprints (`metascript`, `diffwriter`, `layout_glyph`, `vatr_pp`, `emuru`,
`scriptvit`, `lvgh`) are labelled as preprints in the bibliography and are not used as sole support
for a quantitative claim. The CASIA data documentation is the supplier's own technical report and is
the authoritative source for §2. No predatory-venue or citation-ring signal was observed in the
accepted set; the planted probe was correctly rejected.
