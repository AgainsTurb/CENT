# Devil's Advocate — Checkpoint 1 (Phase 1 Scoping)

**Role:** devils_advocate_agent (deep-research, checkpoint 1)
**Inputs:** `rq_brief.md`, `methodology_blueprint.md`
**Date:** 2026-11-21
**Verdict: REVISE** (blocking items 2 and 4 must be resolved before Phase 2; the rest are carried)

---

## C1 — The novelty is narrower than the framing implies. **Severity: high (framing, not fatal)**

The brief positions CENT against systems that "make glyph correctness emergent". But HiSAC
(2025) already decomposes characters into radicals, components and strokes with fine-grained
spatial correspondence, already couples online trajectory and offline image through a shared style
model, and already reports VOTA-level results on CASIA-OLHWDB. Diff-Writer already does
non-autoregressive diffusion for online Chinese characters. One-DM already reports a Chinese
setting. The residue — "generate the stroke graph itself, with topology fixed" — is a real
difference, but the brief's positioning table lets a reader conclude the delta is larger than it is.

**Required fix.** One sentence, up front, that states exactly what changes and what does not:
*CENT does not claim a better sampler; it claims a different generative object.* The paper must
then *test* that claim as a representation ablation (A1/A2), not as an FID race. If the ablation
does not favour the stroke graph, the paper's contribution is the negative result plus the
protocol — and it must say so.

## C2 — The paired-supervision assumption is load-bearing and unverified. **Severity: critical (blocking)**

Every part of the two-modality story depends on online and offline data from the same writer for
the same character. Threat T1 lists this as "to be verified", which is correct — but the
methodology blueprint *already designs the renderer supervision around it*. If pairing is weaker
than assumed, the renderer is trained on unpaired data and the central "one latent, two
modalities, mutually consistent" claim has no supervision behind it.

**Required fix before Phase 2 proceeds.** Verify against the CASIA documentation (Liu et al.,
ICDAR 2011) what is actually shared: writer pool only, or writer × character × session. Record the
finding verbatim with a page/section anchor. The blueprint must carry both branches (paired /
unpaired) with the unpaired branch's fallback spelled out — it currently names the fallback but
does not adjust the claims.

## C3 — Canonical stroke order is being used as ground truth for a phenomenon that tolerates variation. **Severity: medium (construct validity)**

Real writers deviate from canonical order, merge strokes, and lift late. Scoring "stroke-order
error" against a lexicon's canonical order will penalise legitimate variation and inflate the
apparent advantage of a model that was *trained* to produce canonical order — a circularity: the
model is optimised toward the metric's definition of correct.

**Required fix.** Report (a) canonical-order edit distance and (b) an order-equivalence-class
measure (orders that yield the same glyph under a defined merge rule), and restrict the
stroke-order claim to regular-script writing, declaring cursive as out of scope. Without this, the
headline metric measures conformance to our own assumption.

## C4 — The out-of-vocabulary content path is missing. **Severity: critical (blocking, and a hole in the method)**

The brief promises generalisation to the full 3,755-character inventory and evaluates OOV
characters. But the stroke-graph denoiser is conditioned on *character identity* — and an unseen
character has no learned identity embedding. The brief never says how. This is not a detail: it is
the difference between "generalises to unseen characters" and "cannot generate an unseen character
at all".

**Required fix.** For OOV characters, condition on a **rendered archetype** (a standard font image
of the character) through a small content encoder, following the VATr lesson, rather than on an ID
embedding. Then the OOV claim is about unseen *geometry* under known topology — which is exactly
what the stroke-graph representation should be good at, and is a genuinely testable prediction.
The blueprint has been amended accordingly (see §3.2).

## C5 — Dual use is declared but not addressed. **Severity: medium (ethics)**

Handwriting imitation enables forgery. The brief says the paper "will state the risk" and mentions
"watermarking" — but nothing in the method produces a watermark, and a vague ethical paragraph is
not a mitigation. Either the method carries a provenance mechanism (e.g., a style-conditioned
residual signature detectable by a trained probe, or a licence-gated release), or the paper must
say plainly that no technical mitigation is offered and rely on access control.

**Required fix.** Choose one and implement or declare it. Do not claim a mitigation the artefact
does not have.

## C6 — A results-free paper may not survive review. **Severity: high (publication viability)**

If the empirical-scope decision lands on "method + protocol, no results", the honest artefact is a
protocol paper. Reviewers at a CV venue will ask for numbers; a workshop or a venue with a
registered-report track is the realistic home. The user should decide this *now*, because it
determines the paper's genre, length, and target venue, and it determines whether Stage 1 needs to
budget compute.

---

## Verdict

**REVISE.** Items C2 and C4 are blocking: C4 is fixed in the blueprint as of this checkpoint; C2
requires a documentation check at the start of Phase 2 and a recorded finding. C1, C3, C5, C6 are
carried into the Stage-1 checkpoint as decisions the author must make, because each changes what
the paper claims.
