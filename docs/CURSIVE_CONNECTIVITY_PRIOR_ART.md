# Cursive connectivity: prior art, and what CENT can still claim

Written 2026-11-22 after a targeted search for inter-character connection work. The
conclusion is uncomfortable and therefore stated first.

## 0. Verdict

**"Model the connection between characters" is not a new idea.** It exists in three
independent bodies of work, one of them two decades old. What remains open is narrower:
**continuity as a construction rather than a label**, **connectivity measurable in the
offline image domain**, and **Chinese connectivity as a supervised, validated quantity**.

## 1. Prior art, three clusters

### 1.1 Classical connection models (2005–2015) — the origin of our formulation

| Work | What it already does |
|---|---|
| Lin & Wan, *Style-preserving English handwriting synthesis*, Pattern Recognition 40(7):2097–2109, 2007 — DOI 10.1016/j.patcog.2006.11.024 | Samples **whether to connect each adjacent pair**, decoupling a writer-independent pair prior from a writer-level **cursiveness scalar**; fits the ligature with a polynomial. |
| Wang, Wu, Xu, Shum, *Combining shape and physical models for online cursive handwriting synthesis*, IJDAR 7(4):219–227, 2005 — DOI 10.1007/s10032-004-0131-6 | **Tri-unit model**: head/body/tail per character plus a separate **concatenation stroke** model, fused by minimising a **concatenation energy containing a direction-alignment term** and a gap term. |
| Chen, Lin, Jian, Shen, Chen, *Data-driven handwriting synthesis in a conjoined manner*, CGF 34(7):235–244, 2015 — DOI 10.1111/cgf.12762 | Cursive probability per pair + trajectory optimisation, extended to whole paragraphs with layout. |

These are the same three ingredients our method is built from — join decision, connector
stroke, explicit smoothness objective — and Wang et al. even measure direction alignment at
the join, which is the quantity we claim the modern metric suite omits. They were
writer-specific template systems on Latin script. **The paper must cite this lineage and say
we are advancing its instantiation, not inventing the idea.**

### 1.2 BoundInk (arXiv:2604.02103) — the direct modern competitor

Submitted 2026-04-02, v3 2026-09-20; originally titled *CASHG: Context-Aware Stylized Online
Handwriting Generation*, renamed at v3. Authors Jinsu Shin, Sungeun Hong, JinYeong Bak. Code
at `github.com/jinsu0000/cashg-official` (released pipeline is **English-only**: IAM + BRUSH).

- treats inter-character boundaries as **explicit generation units**;
- **four-way pen state**, `PM` / `PU` / **`CursiveEOC`** / `EOC`, where `CursiveEOC` means
  "end of character with the pen still down" — a supervised cursive join;
- bigram sliding-window decoder (predecessor–current), gated context fusion, three-stage
  curriculum;
- **CSM metric suite**: `F1_Cursive`, `CRE`, `KGS`, `SSS`;
- results: normalised DTW reduced 17.6–47.8%; blind human preference 78.0–82.6%.

**Its Chinese finding is the most useful sentence for us:** under the OLHWG-compatible Chinese
protocol **only `KGS` is defined**, because "the evaluation set lacks cursive-positive
boundaries and whitespace runs". Their Chinese glyph metric also drops slightly
(0.8789 → 0.8718). In Chinese, connectivity is not merely unmodelled — it is unmeasurable
with the current benchmark.

### 1.3 UniCalli (arXiv:2510.13745) — Chinese, with connecting strokes

Xu, Wang, Chen, Wu, Wen, Chao, Chen. Column-level Chinese calligraphy generation and
recognition in one diffusion framework; reports **ligature continuity** as an explicit quality
axis and generates connecting strokes conditioned on adjacent characters. Code:
`github.com/EnVision-Research/UniCalli`.

Difference from us: **calligraphy columns** (vertical, artistic, ~8k historical pieces from a
curated dataset) versus **everyday handwriting lines** (CASIA, horizontal, 1,019 writers).
The capability exists in Chinese; the domain and the supervision do not overlap.

### 1.4 Supporting observations

- DiffInk (ICLR 2026) argues decoupled layout "neglects structural dependencies between
  adjacent characters" and that stitching artefacts are "readily detected by the style
  classifier" — it supports our motivation without modelling connectors explicitly.
- HWT (ICCV 2021) claims to preserve "cursive patterns and connectivity"; ScrabbleGAN notes
  that handwriting is a local process where each character is influenced by its neighbours.
- Urdu/Arabic ligature GANs: cursive is obligatory in those scripts, so ligatures are
  segmented and generated as units — a different (and easier) supervision regime.

## 2. What we can still claim, ranked

1. **Continuity by construction.** A pen-state label constrains the pen's *state*; anchoring
   the connector's control points to its neighbours' endpoints constrains its *geometry*, so
   the positional discontinuity at a join is exactly zero and only direction remains. This is
   a mechanism difference with an observable signature — **direction error at the join** —
   which the existing suite does not measure. Observable, falsifiable, cheap to test (A9).
2. **Connection metrics that survive the Chinese regime.** `F1_Cursive` is undefined with no
   positives; `SSS` is undefined with no spaces. **Join-rate error** stays defined (two
   systems that join nothing and a writer who joins nothing correctly score zero), and we
   report F1 as *undefined* rather than zero where it cannot be computed.
3. **Image-only connectivity, validated.** The entire offline family (DiffBrush, TGC-Diff,
   InkSpire) has never been scored on connectivity because connectivity has been measured from
   trajectories. An ink-only proxy, with its agreement against paired trajectories reported as
   a number, brings that family into the comparison. No prior work does this.
4. **Connectivity as a style-conditioned target.** Join rate measured from the writer's own
   reference lines, and overridable at inference. Classical work had a global cursiveness
   scalar; BoundInk predicts per boundary but exposes no knob.
5. **A hard intra-character constraint.** Lexicon-fixed stroke count and order on a line that
   is otherwise free. Nobody in the connectivity literature does this, and it is also the
   design's main risk (cursive execution merges strokes).

## 3. Consequences for the paper

- Claims removed: "existing line generators do not represent connectivity"; "connectivity
  metrics — the field has none"; "no per-boundary control" as a novelty claim.
- Claims added, in this order: continuity by construction; metrics defined in the Chinese
  regime; validated image-only proxy; style-conditioned join rate.
- BoundInk becomes a cited baseline and the system to beat; UniCalli is cited for Chinese
  connecting strokes; the classical trio is cited as the idea's origin.
- The protocol adopts the CSM constructs, states that our implementations are ours because the
  released formulas are not public, and adds direction error plus the offline proxy.
- Ablation A9 (anchored vs free connector endpoints) is the one that tests the mechanism.

## 4. Not verified

- BoundInk's code licence; whether its released checkpoint is usable; the exact CSM formulas
  (names and intent only).
- UniCalli's code licence and weights.
- Whether any Chinese paper models 连笔 for *handwriting* rather than calligraphy: none found,
  but absence at this search depth is not proof.
