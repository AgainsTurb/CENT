# Can CENT learn the join without a layout? — architecture note

**Question asked.** Without pre-generating or defining a layout, can the model learn to connect the last stroke of character *i* to the first stroke of character *i+1*?

**Short answer.** Yes — but **not** by generating the two characters and then joining them. The join must be decided *before the tail of character i is drawn*, because a tail drawn for a pen-lift is the wrong shape for a connection. Every architecture that generates first and joins afterwards is therefore forced into geometric repair (bending the glyph, fitting a Bézier, deforming the trace), which is exactly the weakness CENT should avoid. The architecture that works makes the boundary a **first-class generation event inside one continuous line-level stream** — and in that formulation **no layout module is needed at all**, because position is just the coordinate the model emits.

---

## 1. Why layout is not the thing you need to solve

A layout module answers "where does character *i* start and how big is it". But in trajectory space that question is already answered by the coordinates the model emits. OLHWG needed a layout module only because it generates each character on its own normalised canvas and then rescales it into a predicted box — the box is an artefact of the decoupling, not a requirement of the task.

So there are two independent decisions, and prior work conflates them:

| Decision | Who currently makes it |
|---|---|
| **Where** the pen goes next (layout, spacing, baseline drift) | Implicit in autoregressive coordinates (Graves-style), or explicit in a box predictor (OLHWG), or inherited from a font renderer (One-DM) |
| **Whether** the pen lifts at a character boundary (the join) | Nobody, for Chinese. Implicit only in DiffInk; impossible in OLHWG |

CENT only has to make the second one explicit. The first one comes free — and can be regularised, not predicted by a separate network.

---

## 2. The formalisation that already exists (in English)

CASHG / BoundInk (SKKU, 2026) formalise exactly this, and it is worth adopting their vocabulary rather than inventing one. Four pen states per time step:

`{ PM (pen move, pen down), PU (pen up), CursiveEOC (end of character, pen still down), EOC (normal end of character) }`

- **CursiveEOC** is the join: an end-of-character event where the pen does **not** lift, so the next character continues from the current position.
- The head predicts, per step, a **K=20 GMM over (Δx, Δy)** plus **4-way pen-state logits**; losses are masked GMM NLL + masked cross-entropy.
- The label is **derivable, not hand-annotated**: a boundary is CursiveEOC when the distance between the last point of character *i* and the first pen-move point of character *i+1* is below a threshold (they use 0.005 in normalised coordinates). For SCUT-COUCH, the per-character stroke indices make this *exact* rather than thresholded — a stroke that spans two characters' index sets **is** a connection.
- Training uses a **bigram-aware sliding window** (predecessor + current character only), because the phenomenon is boundary-local, plus a **Vertical Drift Loss** on the bigram stream to stop y-drift accumulating across boundaries.
- A three-stage curriculum: glyphs → bigrams → sentences.

Two things follow immediately. First, **the capability is known and reproducible** — it is not a research gamble. Second, **it has not been done for Chinese**: CASHG's own Chinese experiments are glyph-level only, and under the OLHWG protocol it could report just one metric because "the evaluation set lacks cursive-positive boundaries". That is the opening.

Also critical, and easy to miss: BoundInk's stated limitation is "pen-state supervision inherits segmentation quality; a bad cut biases CursiveEOC", and its remaining failure mode is **autoregressive drift on long lines**. Both are known, both have partial fixes (curriculum, VDL, coarse-to-fine decoding).

---

## 3. Why "generate both characters, then connect them" is the wrong shape

This is the variant the question literally describes, and it is worth being explicit about why it fails — because it is the intuitive choice and it produces a weak paper.

If character *i* is already drawn, its final stroke ends in the position and direction chosen under the assumption that nothing follows. To connect, you must either
- **deform** the tail (Chen et al. 2015: energy-minimising shape+trajectory optimisation; Samsung ICASSP 2026: trace-deformation solved as a tridiagonal system), or
- **overwrite** it with a fitted curve (WriteFreely's quadratic-Bézier trick; the polynomial ligature of the classic ink-synthesis systems).

All of these are post-hoc repair. They bend a glyph that was generated for a different purpose, they cannot coordinate the tail direction with the landing direction, and they inherit whatever error the two glyphs already had. The literature demonstrates this: Samsung's method needs a *head/tail classifier* plus a convex optimisation per pair, and it is bounded to Latin/Cyrillic.

The stronger formulation is that **the boundary decision conditions the generation of the tail** — the model chooses CursiveEOC *while* it is still writing character *i*, so the final stroke can be shaped to end in the right place and direction, and the first stroke of *i+1* can be shaped to receive it. That is a single joint decision over the pair, and it is the reason a **bigram window** rather than independent characters is the right inductive bias.

---

## 4. Three architectures for CENT, with honest trade-offs

### Option A — Autoregressive line stream + style-conditioned boundary state
Take the CASHG formulation, drop the English assumptions, and add:
- a **style-conditioned boundary prior**: predict `p(CursiveEOC | char_i, char_{i+1}, writer code)` rather than a per-writer-agnostic classifier, so the writer's ligature propensity is *modelled*, not averaged. At inference the prior can be **tempered** (β) to produce the same text at different cursiveness — cheap, controllable continuation.
- a **joint-continuity term** at the boundary: penalise the second difference of the trajectory across the junction (C1 smoothness) and penalise any self-intersection inside the bridge.
- **drift control**: coarse-to-fine / next-scale decoding (WriteFreely's VAR-style scheme) or scheduled sampling.

*Pros:* cheapest; reuses SDT/OLHWG components; the controllable-β knob is a clean, demonstrable contribution.
*Cons:* incremental against CASHG unless the control and the Chinese data story carry it; AR drift is a real risk at 30+ characters.

### Option B — Interleaved boundary latents in a diffusion model (recommended shape)
Extend DiffInk's latent line into an **interleaved sequence**: `[c₁, b₁, c₂, b₂, …, cₙ]`, where `cᵢ` is character *i*'s latent span and `bᵢ` is a **boundary slot** that attends to both neighbours.
- `bᵢ` decodes to one of two things: **∅ + a jump offset** (pen lift) or **a short pen-down bridge** (join).
- A small boundary head classifies `bᵢ` (lift/join) and the bridge geometry is conditioned on the class, so the join is generated, not spliced.
- Because decoding is **non-autoregressive**, there is **no drift accumulation** and no error cascade — the single biggest weakness of Option A and of every AR baselines in the table (WLU "characters stick together"; FontRNN "error accumulates").
- Control is trivial and interpretable: clamp the `bᵢ` class, or interpolate in the boundary-slot latent space. That is a genuine mechanism, not a post-hoc filter.

*Pros:* strongest CVPR fit — explicit boundary variable, no layout module, no drift, native controllability; each claim is separately ablatable (remove boundary slots → collapses to DiffInk; remove control → still works).
*Cons:* needs a VAE/latent that keeps character spans aligned (DiffInk's OCR-in-latent loss exists for exactly this reason); risk that the boundary slot is ignored if the latent is too smooth — must be probed (see §6).

### Option C — Explicit pair-conditioned boundary module
A lightweight predictor `p(connect | cᵢ, c_{i+1}, style)` plus a bridge-geometry head, feeding a line decoder that conditions on the bridge latents. This is Chen et al. (2015)'s structure, learned and neural.
*Pros:* most interpretable; the connection probability is directly inspectable and measurable; easy to argue "we model the writer's ligature propensity explicitly".
*Cons:* reintroduces a two-stage pipeline with the coupling problem that OLHWG suffers from, unless the bridge latents are fed back into the glyph decoder (so it becomes a soft version of Option B).

**My read:** Option B is the paper. Option A is the ablation that shows why B is needed. Option C is the analysis experiment (measure the learned `p(connect)` distribution per writer — that figure alone is a contribution).

---

## 5. The four failure modes to design against, up front

1. **Class imbalance.** In CASIA the CONNECT class is near-empty (two independent 2026 papers report no cursive positives in the Chinese protocol). On SCUT-COUCH words it is not — but it is still a minority. Weighted/focal CE on the boundary head, and report the positive rate so reviewers can see the prior.
2. **Boundary labels inherit segmentation errors.** If the per-character stroke indices are wrong, the join label is wrong. On SCUT-COUCH this is exact by construction; on any corpus where you threshold, state the threshold and report sensitivity to it.
3. **Content fidelity vs. join.** A bridge that looks smooth but distorts the glyph is a failure. Keep an OCR-in-latent or content critic (DiffInk's `L_ocr`, One-DM's OCR checkpoint) so the join cannot be bought with legibility.
4. **No metric.** Nothing in the Chinese literature measures the join (OLHWG measures eight geometric box distances + recognition). You need a Chinese analogue of F1_Cursive / CRE, plus a human study. Define it before training, or the claim is unfalsifiable.

---

## 6. The cheap experiment that decides the architecture

Before building Option B, run a **boundary-only probe** — a few GPU-hours, and it settles whether the signal exists:

1. From SCUT-COUCH words, extract every adjacent character pair with its true label (join / lift, from the stroke indices).
2. Train a small model on `(character i trajectory, character i+1 trajectory, writer id)` → `join?`.
3. Report AUC, and the per-writer join rate.

What the outcome tells you:
- **AUC well above chance, writer-conditioned signal present** → the join is learnable from geometry+style, Option B is safe, and the learned per-writer join rate is already a paper figure.
- **AUC near chance** → joins are not predictable from glyph geometry, so the control knob must come from a latent style code, not from pair features; design accordingly.
- **Join rate ≈ 0 in your chosen corpus** → you are on the wrong corpus (this is what happened with CASIA); move to SCUT-COUCH words.

This probe is also the sanity check for the data: it tells you, numerically, whether "cursive continuity" is a real phenomenon in the corpus you are about to spend months on — the measurement nobody in the Chinese generation literature has published.

---

## 7. One-paragraph answer

Yes — a model can learn the join with no layout module, because layout is just the coordinate stream and the join is just *the absence of a pen lift*. Make the boundary an explicit state (`CursiveEOC`) or an explicit latent slot (`bᵢ`), decide it **before** the tail of the current character is committed, and condition it on the writer. Do **not** generate characters first and join them afterwards — that is geometric repair, and it is already published in three weaker forms (energy optimisation 2015, trace deformation 2026, Bézier joining 2026). The research risk is not architectural; it is that no Chinese *line* benchmark contains positive join examples — which is why the corpus choice in `cursive_chinese_line_data.md` §9 matters more than the model.
