# Chinese offline handwriting generation — method survey and diagnosis of CENT v3

_Target task (restated): given a text string and a few reference lines of one writer's Chinese
handwriting, generate a handwritten **text line** in that writer's style._

## 1. What the field actually does

### 1.1 Line-level Chinese generation — the two reference points

**OLHWG — "Decoupling Layout from Glyph in Online Chinese Handwriting Generation", ICLR 2025**
([proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0e0df370cd1ce52c09bb5c1aa584625d-Abstract-Conference.html)).
Factorises the problem:

```
p(X, Layout_X | C, X_ref) = p(Layout_X | C, X_ref) · Π_i p(x_i | c_i, X_ref, Box_i)
```

- **Layout generator**: autoregressive over characters, "in-context-like" learning from the text
  and the style references; emits one box per character as `[height, width, vertical centre,
  horizontal offset]`.
- **Font synthesizer**: a **character embedding dictionary** + a **multi-scale calligraphy style
  encoder** + a **1D U-Net diffusion denoiser**, generating each glyph inside its box.

The key assumption is written down explicitly: *when the box is given, the generation of a
character is independent of the other characters.* That is exactly why they cannot express a
connection between characters — the limitation quoted in §3.

**TGC-Diff — "Template-Guided Cascaded Diffusion for Stylized Handwritten Chinese Text-Line
Generation", ICDAR 2025 Best Paper**
([DOI](https://doi.org/10.1007/978-3-032-04614-7_9), [code](https://github.com/newbie2niubility/TGC-Diff)).
Cascaded, with a **Template** construct joining two diffusion models:

- **Layout-Diff** learns the *spatial* style (where characters sit);
- **Imitating-Diff** learns the *handwriting* style, conditioned on a **latent aggregation module**
  that fuses content and calligraphy style;
- refinement in **pixel space with an L1 loss weighted by a high-frequency mask**, specifically to
  sharpen fine detail and improve readability;
- built on a **pretrained VAE + ResNet18** backbone.

### 1.2 Character-level methods (style transfer family)

- **MetaScript** ([arXiv:2312.16251](https://arxiv.org/html/2312.16251)) — GAN, few-shot, trained on
  **~1M character images from 300 writers** (CASIA-HWDB-1.1) with rendered prototype images.
- **One-DM**, **Diff-Writer**, **HiGAN+**, **GANwriting**, **DiffusionPen** — one-shot/few-shot
  character imitation; almost all condition a **UNet/decoder** on a *global style code* injected by
  **AdaIN/FiLM at every block**, and supervise in **image space** (L1/perceptual/adversarial).
- **HWT (Handwriting Transformer)** and Graves-style models do sequence modelling, but over **point
  sequences with pen states**, and HWT uses **two-stage pretraining** (arbitrary trajectory
  prediction, then content-conditioned generation).

### 1.3 Surveys

- *From pixels to strokes: A survey on generative handwriting*, IJDAR 2026
  ([DOI](https://doi.org/10.1007/s10032-026-00609-2)).
- *Advancements in Chinese font generation since deep learning era: A survey*
  ([arXiv:2508.06900](https://arxiv.org/abs/2508.06900)) — organises the field by how many reference
  samples are needed (many-shot vs few-shot).

### 1.4 The common recipe

Every method that produces *recognisable Chinese text* shares four ingredients:

| ingredient | OLHWG | TGC-Diff | MetaScript / char-level |
|---|---|---|---|
| **explicit layout**, supervised | layout generator, per-char boxes | Layout-Diff | implicit (single char) |
| **glyph content prior** | character embedding dictionary | template + content features | prototype renderings |
| **dense style injection** | multi-scale style encoder → every block | latent aggregation → Imitating-Diff | AdaIN/FiLM everywhere |
| **pixel/visual supervision** | diffusion over 1D data per glyph | pixel-space L1, high-frequency weighted | L1 + perceptual + adversarial |
| **pretrained prior** | — | pretrained VAE + ResNet18 | often none, but 1M samples |

## 2. What CENT v3 does instead, and why each difference is fatal

| dimension | literature | CENT v3 | consequence |
|---|---|---|---|
| layout | explicit, supervised, per-character boxes | `M3` exists but **`lambda_layout = 0.0` in every run** — never trained | model must invent global placement; nothing anchors glyph positions |
| content | character embedding **dictionary** / template images | 2680-class embedding read through a Gaussian mask | glyph structure must be learned from ~30k lines, from scratch |
| style | multi-scale encoder, injected at every block | **one pooled vector, added once** to the transcript memory | measured: style swap changes error by **0.03** vs **1.5** for text — 50× too weak |
| supervision | pixel/visual loss on the rendered line | **none** (image loss removed after the frame-mismatch discovery) | nothing in the objective requires the render to look like a character |
| prior | pretrained VAE / 1M character samples | from scratch, 30,302 lines | too little data for 2680 glyph classes |
| output parameterisation | glyph images or point sequences with pen states | 78 cubic Béziers × 8 params, MDN | no smoothness prior, no pixel prior; a local objective can be minimised while the glyph is wrong |
| style references | same modality, same writer | **scans**, while targets are **rendered trajectories of a different writing session** | we measured IoU(scan, trajectory render) = 0.008 — the style signal is from a different distribution than the target |

### 2.1 The measured failure, restated

- Single line, 14k steps, NLL −23: free-run error **0.072** — the model memorises one line.
- Four lines, 6k steps, NLL **−1.66**: every one of **16** (text × style) generations is closest to
  **the same** ground truth (GT #2). Diagonal never selected. Style effect ~0.03.

So the model does not read either condition; it emits one attractor. Under-trained *and*
architecturally under-conditioned — the two are currently confounded (§3 of
`V3_CONDITIONING_ANALYSIS.md`).

### 2.2 Why an autoregressive Bézier MDN is the wrong instrument here

No published Chinese line method uses it. The reasons are visible in our own numbers:

1. **No visual prior.** Chinese glyph correctness is a *global spatial* property. An AR model with
   a per-stroke likelihood can be locally perfect and globally illegible; we saw exactly that
   (NLL −23 for one line, and the 4-line model collapsing).
2. **Error accumulation.** Even with the recurrence verified to 5.96e-08 and the dominant-component
   decode, free-running needs teacher-forced parity per line; stochastic sampling diverges to
   error 8.8.
3. **Data inefficiency.** 78 strokes/line × 30k lines is ~2.4M stroke targets, but only 30k
   *lines*; the glyph dictionary in OLHWG and the 1M character images in MetaScript exist precisely
   because characters are where the data is.

## 3. The gap CENT can still own

Both OLHWG and TGC-Diff generate **per character** (OLHWG states the independence assumption
explicitly; TGC-Diff's layout handles spacing but connectors are not modelled). OLHWG's own
limitations section says character-independent generation "encounters difficulties in mimicking
styles with extensive cursive connections between characters… an end-to-end text line generation
method might be required."

That is CENT's thesis, and it remains unclaimed by the literature. The mistake was to make
connectivity the *substitute* for layout, glyph and style rather than a layer **on top** of a
working line generator.

## 4. Recommended rebuild — CENT v4

**Stage 0 — adopt the proven frame.** Two viable bases:

- **(A) Online/trajectory route** (closest to our data and to CENT's thesis): OLHWG-style layout
  generator + **per-character 1D diffusion** over strokes conditioned on
  `(character embedding, style tokens, box)`. Keeps stroke output, so connectivity is expressible.
- **(B) Offline/image route**: TGC-Diff-style cascaded diffusion with a pretrained VAE. Stronger
  appearance prior, but connectivity becomes an image-space property and our stroke-level claim
  weakens.

Recommendation: **(A)**, because our corpus is trajectories and the paper's contribution is
stroke-level continuity.

**Stage 1 — layout, supervised.** Train `M3` on the corpus boxes with `lambda_layout = 1.0`
(parametrisation from OLHWG: `[h, w, y_centre, Δx]`). Acceptance: box L1 comparable to OLHWG's
reported margin, and generated boxes visibly track the text.

**Stage 2 — glyph, with a content prior.** Two additions, both cheap:
(i) a **character embedding dictionary** initialised from rendered prototype glyphs (prototypes can
be synthesised from a font, as MetaScript does), and (ii) **pixel-space supervision** on the
rendered line — L1 plus a **high-frequency-weighted** term as in TGC-Diff. This restores the
constraint that killed every CENT run: that the output must *look like* the text.

**Stage 3 — style, densely.** Replace the single pooled vector with a **multi-scale style encoder**
(whole line + per-character crops) whose output modulates **every decoder block** via AdaIN/FiLM.
Acceptance: style swap moves the output appreciably (target: style distance ≳ text distance), and
each output is closest to its own writer's ground truth.

**Stage 4 — connectivity, as the novel layer.** Only after stages 1–3 give recognisable lines:
per-boundary **M4** (join decision + exit/entry anchors + bridge) with the boundary contract we
already implemented (`start = exit + gap·(1−ζ)`), conditioned on the character pair and the style
code. Acceptance: join-rate error beats the trivial baselines (0.204 lift / 0.796 join) on **sampled**
output, and a recogniser still reads the line (connectivity must not destroy legibility).

**Stage 5 — evaluation with a recogniser in the loop.** TGC-Diff reports accuracy/FID/SSIM with a
trained recogniser on generated lines. We already adopted a frozen CASIA HTR
(`Teklia/pylaia-casia-hwdb2`, MIT); until it is wired in, "legible" is an opinion. Report ACC, AR/CR,
FID/KID, and our join/continuity metrics together.

## 5. Data reality check

- Our corpus: 30,302 usable lines, 78 strokes/line including 14 connectors, 815 writers (41,710
  lines available before the char-split filter).
- OLHWG/TGC-Diff train on CASIA OLHWDB/HWDB2 line corpora of comparable or larger size **plus** a
  pretrained VAE; MetaScript uses 1M character images.
- Practical implication: with no pretrained prior, 30k lines is enough for layout and style, not for
  learning 2680 glyph classes from scratch. Either (i) add a font-rendered prototype dictionary,
  (ii) use a pretrained VAE, or (iii) train the glyph stage on **characters** (HWDB2.0–2.2 char
  images, ~1M) and the layout stage on lines. All three are standard in the literature and none
  requires new data collection.

## 6. Diagnosis in one paragraph

CENT v3 failed not because of a bug but because it asked one 27M-parameter autoregressive model,
trained from scratch on 30k rendered-trajectory lines with **no image supervision, no trained
layout, a single pooled style vector, no character-shape prior, and style references drawn from a
different modality (scans) than its targets**, to learn layout, glyph shape, writer style and
inter-character connectivity simultaneously. The literature solves those as four separate problems
with dedicated components and pretrained priors. The single-line result (free-run 0.072) shows the
plumbing works; the four-line collapse shows the conditioning has nothing to latch onto. Fix the
frame — supervised layout, a character prior, dense style injection, pixel supervision — and CENT's
remaining contribution (the boundary contract for connectivity) becomes testable rather than
carrying the whole burden.
