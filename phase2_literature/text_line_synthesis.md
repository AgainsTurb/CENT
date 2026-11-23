# Corrected field map — Chinese handwritten **text-line** generation with cursive connection

**Written 2026-11-21 to correct a scoping error.** The manuscript and code built CENT as an
*isolated-character* generator and treated text-line systems as "not comparable". That was the
wrong task. The target is:

> generate a whole Chinese handwritten **text line**, in a given writer's style, inferred from a few
> reference **lines**, with **coherent cursive connections between characters** — handwriting that
> looks like a person writing quickly.

Everything below re-grounds the work on that target. Read this instead of
`synthesis.md` where the two disagree; `synthesis.md` remains the record of the character-level
literature and is still correct about it.

---

## 1. What the task actually is, in three separable degrees of freedom

A line generator must decide, per boundary between consecutive characters:

| Degree of freedom | What varies | Why it matters |
|---|---|---|
| **Intra-character topology** | stroke count and order within a character | fixed by a lexicon; a wrong count makes a different character |
| **Inter-character connectivity** | pen lifts vs a *connector stroke* joining character *i* to *i+1* | this is what makes fast writing look fast; it is style-dependent, not character-dependent |
| **Layout** | baseline drift, slant, spacing, character box jitter | carries as much writer identity as the glyphs |

Character-level work fixes (1) and ignores (2) and (3). Text-line work addresses (3) and treats (2)
implicitly, as a by-product of a pixel or trajectory generator. **CENT's corrected claim is about
(2): connectivity is a first-class generated object, not an emergent one.**

## 2. Methods that actually do line-level generation

Ranked by proximity to the target, with code status verified by cloning.

| Method | Venue | Modality | Code | What it does |
|---|---|---|---|---|
| **DiffBrush** | ICCV 2025 | offline, Chinese+English | `dailenson/DiffBrush`, MIT | first diffusion line generator; content-decoupled style learning (row/column masking) + line and word discriminators |
| **TGC-Diff** | ICDAR 2025 (Best Paper) | offline, Chinese | `newbie2niubility/TGC-Diff`, MIT | cascaded: Layout-Diff models spatial style, Imitating-Diff models glyph style, joined by a Template |
| **DiffInk** | ICLR 2026 | **online**, line-level | `awei669/DiffInk`, MIT | latent diffusion transformer over pen trajectories; explicitly argues it "generates more natural character connections rather than mechanically stitching bounding boxes" |
| **OLHWG** | ICLR 2025 | **online**, line-level | `singularityrms/OLHWG` | decouples layout from glyph: LSTM layout planner + 1-D U-Net glyph diffusion |
| **InkSpire** | ICLR 2026 | offline, bilingual | repo `newbie2niubility/InkSpire` is **empty** | unified style/content/noise latent; masked multi-line infilling |
| **One-DM** | ECCV 2024 | word-level | `dailenson/One-DM`, MIT | one-shot word mimicker; used by DiffBrush as a Chinese *word* comparison, not a line one |
| **HWT / VATr / DiffusionPen / WordStylist** | 2021–24 | word-level | released | the word-level line of work; not line generators |
| **SDT** | CVPR 2023 | online, character (+offline extension) | released | style disentangled at writer and character level |
| **MetaScript / RC-GAN / CalliGAN** | 2020–24 | character-level | released / partial | character generators; superseded for our task |

**Cursive-specific prior art** (older, but the only work that names connectivity as the problem):
- *Combining Shape and Physical Models for Online Cursive Handwriting Synthesis* (Lehigh) — deforms
  head/tail pieces of adjacent letters and minimises a concatenation energy so the joined trajectory
  stays smooth **and** each letter stays consistent with its model. This is the classical statement
  of the problem CENT now targets.
- *StrokeBank* (IAAI 2014) — explicitly notes that shape-grammar methods "do not hold for joined-up
  writing, where the strokes are largely cursive and connected". Good quotable framing.
- *The Cursive Transformer* (2025) — tokenised pen offsets + plain GPT; English only; shows the
  tokenisation route is viable, not the Chinese route.

## 3. Data: the line-level corpora, and their file formats

This is where the previous scoping error was most costly. Both modalities have **line-level,
character-segmented** ground truth.

### Offline — CASIA-HWDB2.0–2.2 (`.dgrl`)
52,230 lines, 1,019 writers, 1,349,414 characters. One `.dgrl` file = one page, stored as a series
of line records:

```
header  : 4B header size | 8B "DGRL" | illustration (\0-terminated) | 20B code type ("GB")
          | 2B code length | 2B bits per pixel
image   : 4B height | 4B width | 4B number of lines
per line: 4B char count | char codes (code_length each) | 4B+4B (top,left) | 4B H | 4B W | bitmap
```
So each line arrives with **its transcription and its own bitmap**, which is exactly a
line-generation training example. Downloads: HWDB2.0–2.2 train/test, 128–709 MB each.

### Online — CASIA-OLHWDB2.0–2.2 (`.wptt`)
52,220 lines, 5,092 pages, 1,348,904 characters. One `.wptt` file = one page:

```
header   : 4B header size | 8B "WPTT" | illustration (\0) | 20B code type | 2B code length
           | 4B sample length | 4B page index
strokes  : 4B stroke number, then per stroke: 2B point count, then (x,y) as unsigned short pairs,
           coordinates x10, NO pen-down/pen-up flags (stroke boundaries are the segmentation)
lines    : 2B line number, then per line:
             2B line stroke number | stroke indices (2B each)
             2B line char number   | char codes (code length each)
```
**This is the load-bearing discovery for CENT — with an important limit.** Each line
carries the *line's stroke index list* and the *character list*. It does **not** carry a
per-character stroke assignment: the published table has `line stroke number`,
`line stroke index (concatenated)`, `line char number`, `chars` — and nothing that maps
a stroke to a character. Earlier revisions of this note claimed otherwise, and that
claim was wrong.

So connectivity is **inferrable, not given**: a per-character split must be recovered
by a declared rule (deal the line's strokes out in reading order under the lexicon's
counts; treat the surplus as connectors). That inference is threat T3 in the paper, and
the code implements it conservatively — where evidence is insufficient it records a
*lift*, which biases the connector rate downward. A metric built on it must say so
every time it is reported.

### Test set
ICDAR-2013: 3,432 lines, 60 writers, offline and online, with character-level transcripts.
Metrics convention: **AR/CR** via error-correcting string matching (not simple CER).

### Access note
The NLPR download page serves the line data directly (no application gate for the text-line
archives, unlike what the earlier draft assumed for the whole corpus). A Hugging Face mirror
`Teklia/CASIA-HWDB2-line` exists with 52.2k lines at height 128 and text labels, split 33.4k/8.3k —
convenient for smoke runs; its provenance is CASIA and its redistribution terms should be checked
before use in a released artefact.

## 4. Metrics the field uses, and the one it lacks

| Construct | Convention | Comment |
|---|---|---|
| Content | AR = (N−D−S−I)/N, CR = (N−D−S)/N, character-level via DP string matching | the field's standard; report both |
| Realism | FID | on line images |
| Style | writer classification / retrieval, style-score classifiers trained on real lines | |
| **Connectivity** | **none** | no standard number |

**The gap:** no line-level work reports a number for *whether the connections look right*. You can
score 99% CR while producing mechanically spaced glyphs with no ligature, and no published metric
will notice. That is the metric CENT is positioned to introduce:

- **connector rate** — fraction of character boundaries the model joins with ink (compare with the
  writer's own rate on reference lines);
- **pen-lift agreement** — for online output, agreement between generated pen-up events and the
  writer's, per boundary;
- **continuity error** — at connected boundaries, the trajectory discontinuity (distance + tangent
  mismatch) at the join, which the classical concatenation-energy work minimises;
- **speed realism** — intra-stroke velocity profile statistics, since "writing fast" is the user's
  actual perceptual target and is invisible to AR/CR.

## 5. What CENT should now be

Keep the structure-first idea where it is *strongest* — inside characters — and stop pretending it
applies across the line:

1. **Intra-character**: stroke count and order from the lexicon (hard constraint, unchanged). This
   is still a real guarantee no line-level baseline offers.
2. **Inter-character**: a **connector latent** per boundary — a binary connect/lift decision plus,
   when connected, a cubic Bézier joining the exit of character *i* to the entry of *i+1*, with the
   pen never lifting. This is the new generative object.
3. **Line**: character sequence from the requested string, with the layout (baseline, spacing,
   per-character box) carried by the style latent inferred from **k reference lines**.
4. **Dual decode**: the same line latent decodes to an online trajectory (continuous pen, correct
   pen-up events) and to an offline line image through the differentiable rasteriser.
5. **Style**: from k reference *lines* (not characters) — global (slant, spacing, ink, baseline
   drift) + local (per-stroke-type geometry, connector shape), matching how the line papers
   condition on references.

**Why this is not already done.** DiffBrush and TGC-Diff generate pixels and never represent
connectivity; DiffInk and OLHWG generate trajectories with connections but treat them as emergent,
and neither exposes a per-boundary control. Nobody reports a connectivity metric. The classical
cursive work models the join but predates deep generative models and works on letter sequences, not
a 7,000-class script.

## 6. Prior-art proximity, stated narrowly

| Work | Connectivity | Intra-char topology | Modality | Style source |
|---|---|---|---|---|
| DiffBrush | emergent | unconstrained | offline | k lines |
| TGC-Diff | emergent | unconstrained | offline | k lines |
| DiffInk | emergent | unconstrained | online | k samples |
| OLHWG | emergent | unconstrained | online | k samples |
| classical cursive synthesis | **modelled** (concatenation energy) | letter models | online, small alphabet | writer model |
| CENT (corrected) | **generated, controllable, measured** | **fixed by lexicon** | **both** | **k lines** |

## 7. Corrections this forces on the existing artefact

| Item | Was | Must become |
|---|---|---|
| Task | isolated character | text line with cursive connection |
| Datasets | HWDB1.x/OLHWDB1.x (characters) | HWDB2.x/OLHWDB2.x (lines), ICDAR-2013 lines |
| Parsers | `.gnt` / `.pot` | add `.dgrl` / `.wptt`; the `.pot` header size is **2 B**, not 4 B as previously coded |
| Style input | k isolated characters | k reference **lines** |
| Generative object | stroke graph for one character | character stroke graphs + **connector latents** |
| Metrics | character CER, FID | AR/CR, FID, style score + **connectivity metrics** |
| Baselines | GANwriting, HWT, VATr, DiffusionPen, MetaScript, SDT | DiffBrush, TGC-Diff, DiffInk, OLHWG (+ InkSpire cited-only) |
| Paper | line-level claims were excluded | line-level is the task |

## 8. Carried threats

- **T1 format risk**: `.wptt`/`.dgrl` parsers are written from the published tables and have not seen
  a real file. Probe first; a mis-framed record yields plausible garbage.
- **T2 rigidity vs fluency**: a hard intra-character constraint may fight cursive execution, where
  writers reduce stroke counts. This is now a *central* tension, not a footnote: the corrected
  design must measure how often real writers deviate inside characters while connecting outside.
- **T3 connectivity ground truth is inferred, not given**: `.wptt` gives a line's stroke
  indices and its characters but no per-character stroke assignment, so "connected" must be
  recovered by a declared rule. That definition is ours; state it with every number it
  produces, and prefer the conservative direction (call it a lift when unsure).
- **T4 metric novelty**: a new connectivity metric has no precedent, so it must be simple enough to
  be obviously right and reported alongside AR/CR, never instead of it.
