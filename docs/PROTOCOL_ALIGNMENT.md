# Protocol alignment: what the field does, what we were missing, what we adopt

Written after reading the two most relevant line-generation papers end to end (retrieved
2026-11-22 from arXiv HTML). Every claim below is from the retrieved text; quotation is
paraphrased where marked.

Sources
- DiffBrush, *Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation*, ICCV 2025 — https://arxiv.org/html/2508.03256v1
- OLHWG, *Decoupling Layout from Glyph in Online Chinese Handwriting Generation*, ICLR 2025 — https://arxiv.org/html/2410.02309v3
- DiffInk, ICLR 2026 — https://arxiv.org/abs/2509.23624 (abstract-level)

## 1. DiffBrush's protocol (offline, Chinese lines)

| Item | DiffBrush |
|---|---|
| Data | CASIA-HWDB 2.0–2.2, **52,230 lines, 1,019 writers** |
| Split | **816 train / 203 test** writers ("following the standard split") |
| Resolution | all images **64 × 1024**, and Unifont renders the text into content images |
| Style input | k reference lines; the table has a "**Shot**" column: one-shot vs few-shot methods |
| Realism | FID, IS, GS |
| Style | **HWD** — Euclidean distance in VGG16 features from a net pretrained on a large corpus of handwritten text images |
| Content | AR, CR (Chinese line recognition conventions); plus **D_CER / D_WER** — train an OCR *on each method's generated training set*, then test it on the **real** test set |
| Human | a user study |
| Baselines | TS-GAN (one-shot), CSA-GAN (few), VATr (few), DiffusionPen (few), One-DM (one) — all on **the same training set and the same protocol** |
| Ablations | CTC-loss weight ∈ {1, 0.1, 0.01, 0.001}; discriminator weight ∈ {0.1, 0.05} |

Two things to note. First, three of its five baselines are **word-level** models: the paper
composes them to lines and compares anyway, which legitimises our composed-line mode.
Second, D_CER/D_WER is a *data-utility* metric, not a fidelity metric — it asks whether
synthetic lines are good enough to train a recogniser. We had nothing of that kind.

## 2. OLHWG's protocol (online, Chinese)

| Item | OLHWG |
|---|---|
| Character stage | train CASIA-OLHWDB 1.0–1.2 (≈3.7M chars, 1,020 writers); test **ICDAR-2013** (60 writers, the 3,755 GB2312 level-1 set) |
| Line stage | CASIA-OLHWDB 2.0–2.2, ≈52k lines, "1,200 authors", 1.3M chars |
| Split | **1,000 train / 200 test** writers |
| Content | **AR, CR** using a text-line recogniser from Chen et al.; that recogniser reaches **0.962 AR / 0.958 CR** on real test lines, which bounds every reported AR/CR |
| Layout quality | **eight binary geometric features** comparing generated vs real layout (e.g. mean ratio of bounding-box widths) |
| Style/realism | FID, KID, HWD on ICDAR-2013 lines (its comparison table reports e.g. One-DM FID 34.36 / KID 28.37 / HWD 0.80 / CR 73.19) |

Its own framing is the same as ours: "text lines can naturally be divided into two
components: layout and glyphs". The difference is what we do with the layout: OLHWG
generates boxes and fills them, so a pen never crosses a box boundary; CENT generates the
pen path including what happens between boxes.

## 3. What our experiments were missing

Ordered by how much they would cost us at review.

1. **Wrong split arithmetic.** We wrote 816/204. The standard is **816/203** (816 + 203 =
   1,019 writers). For the online corpus, the comparable split is **1,000/200**. Fixed.
2. **We cropped lines at 1024 px.** Cropping truncates characters while the transcript
   still lists them, silently corrupting every content metric. Now padded to 64×1024, and
   the builder raises rather than crops.
3. **No content images.** Every offline baseline is content-image conditioned; we had no
   renderer. Now `common/content_render.py` renders text from the glyph table DiffBrush
   itself ships (55,659 glyphs, 20,992 CJK), so all baselines get identical content input.
4. **No HWD.** Our "writer verification" is a different construct. HWD is the field's
   style number and is now implemented (`benchmark/metrics.py`), pending the VGG16
   handwriting net.
5. **No KID, no IS/GS.** KID is what the Chinese-line table reports. Implemented.
6. **No recogniser identity.** AR/CR is meaningless without naming the recogniser and its
   own accuracy on real lines (OLHWG states 0.962/0.958). We must fix one and report both.
7. **No D_CER/D_WER.** The "is the synthetic data useful" test. Cheap to add once a
   recogniser exists, and it is the metric most relevant to our data-augmentation claim.
8. **One-shot vs few-shot not separated.** DiffBrush reports a Shot column; we should
   report k=1 explicitly rather than only k=5.
9. **No layout-quality metric.** OLHWG's eight binary geometric features give a
   layout-validity number. Our connectivity metrics do not measure layout, and a reviewer
   who knows OLHWG will ask for it.
10. **No user study.** Both papers run one, and a style claim is a perceptual claim.
11. **No runtime.** DiffInk makes efficiency a selling point; a comparison table without
    inference time invites the question.
12. **Ablations are ours but thin in one place.** DiffBrush ablates the CTC weight, i.e.
    how strongly content accuracy is enforced. Our A4 (topology constraint on/off) covers
    the equivalent for structure, but we should also ablate the content-loss weight, since
    that is the knob their paper shows to be decisive.

## 4. What we adopt

Adopted as-is:
- 64×1024 padded lines; Unifont-rendered content images.
- 816/203 offline, 1,000/200 online; validation writers carved **from the training side**
  and used only for the gate.
- Metrics: **AR, CR** (frozen recogniser, its own accuracy reported), **FID**, **KID**,
  **HWD**, and **D_CER/D_WER**; IS/GS optional if the features become available.
- One-shot reported separately from few-shot.
- A user study, and inference time per line.

Adopted with a difference, and stated in the paper:
- **Baselines.** Native line generators are compared as shipped. Word- and character-level
  models are compared in **composed** mode, with the layout named (`uniform` = the
  stitched floor; `style` = layout estimated from the same k reference lines CENT uses).
  A composed number without its layout is uninterpretable, so the table carries the layout
  as a column.
- **Layout quality.** We report OLHWG's geometric features *plus* our connectivity
  metrics, and we argue connectivity is the part layout features cannot see.

What we add that neither paper reports:
- connector rate, pen-lift agreement, continuity error, speed profile — i.e. a number for
  the one thing that separates a fast handwritten line from a row of glyphs.

## 5. Consequences for the comparison table

The table must be structured, not flat, because the systems are not the same kind of thing:

| Group | Systems | What the number means |
|---|---|---|
| Native line, offline | DiffBrush, TGC-Diff, CENT | full line in one model |
| Native line, online | DiffInk, OLHWG, CENT (trajectory decode) | pen trajectory for the whole line |
| Composed (word/char → line) | One-DM, VATr, DiffusionPen, HWT, GANwriting, TS-GAN, CSA-GAN, MetaScript, SDT (+ extras from the survey) | glyph quality × layout quality, reported with the layout named |
| Cited only | InkSpire (no code), OLHWG (no licence/weights) | published numbers, marked not re-run |

Collapsing these into one ranking would be the easiest way to inflate our result and the
fastest way to lose a reviewer. The table keeps the grouping, and the connectivity columns
apply only to systems that emit geometry.

## 6. Still unverified

- Whether OLHWG's "1,200 authors" and "1,000/200" are consistent with DiffBrush's
  "1,019 writers, 816/203" — the corpora are described differently (offline vs online
  pages, and writer counts differ across the supplier's documentation). We adopt each
  paper's numbers for its own modality and say so.
- The exact VGG16 handwriting net behind HWD, and whether it is released. Until it is,
  HWD is reported as NOT_RUN with that reason rather than approximated with ImageNet
  features (which would make our HWD incomparable to the papers we cite).
- Chen et al.'s recogniser availability. If it is not obtainable, an alternative must be
  fixed, and its own AR/CR reported so the comparison remains interpretable.

## 7. Baseline-availability checklist (added after an error)

An audit of OLHWG concluded "no weights, no licence" and was wrong on both counts: the
weights and the pre-processed line data are published on a Hugging Face repository whose
dataset card declares MIT, reached from a URL the README describes as a *dataset* page. The
README was read; the file listing inside that repository was not.

Before any baseline is called unavailable, all of these must be checked:

1. `git ls-files` on the code repository (licence file, checkpoints, sample data).
2. **The file listing of every URL the README links to** — for Hugging Face,
   `GET /api/datasets/<id>/tree/<rev>` or `/api/models/<id>/tree/<rev>`; read `siblings`, sizes
   and any `lfs` entries.
3. The dataset/model card's metadata — `cardData.license` and the `tags` array — because a
   model or dataset card can declare a licence that the code repository does not.
4. Which entry point loads which filename. OLHWG's `generate_line.py` loads the
   `450000`-step pair that was released, while `evaluate_line.py` still names `290000`; taking
   the stale name as the requirement produced a false "missing checkpoint" verdict.
5. Whether the released artefact is *data* or *weights* — and if it is labelled "dataset",
   whether it also contains `.pth`/`.safetensors` files.
6. Only then: "unavailable", with the evidence attached.

The licence question must be answered per artefact, not per project.
