# CENT — Prior-Art Survey

**Scope.** 13 repositories and 15 PDFs supplied in `repos/` and `papers/`, plus the 2025–2026 literature recovered online. The question for every method: *how does it generate, and can it produce a line of Chinese characters?*

**Target task (CENT).** Chinese handwriting generation of **text lines**, with (i) per-writer style mimicry, (ii) free cursive continuity between characters (连笔) instead of hard layout + character pasting, (iii) explicit handling of the train/infer gap between clean character-level corpora and cursive line-level inference.

**Evidence rules used in this document.**
- Paper claims cite `p.N` from the page markers in `survey/_pdftext/*.txt`.
- Code claims cite `repos/<repo>/<file>:<lines>`.
- Three evidence tiers are marked explicitly: **[V]** = verified first-hand by the lead (paper text or code read directly); **[A]** = taken from a specialist sub-agent record (`survey/_records/*.md`) which cites page/file lines; **[W]** = recovered from the web only and **not** verified against a supplied PDF.
- Nothing below is inferred silently. Where a source is missing or wrong, that is stated rather than filled in.

---

## 0. Data-integrity findings (read this first)

Two of the fifteen supplied PDFs are **not the papers their filenames claim**. Both defects were found independently (agent report, then re-verified by the lead by reading page 1 of each extracted text) **[V]**:

| File | Filename claims | Actually contains |
|---|---|---|
| `papers/FontDiffuser_AAAI2024.pdf` | FontDiffuser (AAAI 2024) | **CUDC: A Curiosity-Driven Unsupervised Data Collection Method … for Offline Reinforcement Learning** (Sun, Qian, Miao; arXiv:2312.12191) — an offline-RL paper with zero font/diffusion content |
| `papers/HWT_ICCV2021.pdf` | Handwriting Transformers (ICCV 2021) | **A Quantification of a Besicovitch Nonlinear Projection Theorem via Multiscale Analysis** (Davey & Taylor; arXiv:2104.00826 `[math.CA]`) — a pure-mathematics paper |

Consequences: **no page-citable claim exists for FontDiffuser or HWT.** Their records below rest on the repositories plus abstracts retrieved online, and are labelled **[A]**/**[W]** accordingly. Re-download both before writing the paper — the HWT repo is intact and the ICCV 2021 camera-ready is on the CVF open-access site.

A third file, `papers/WriteFreely_ICCV2026.pdf`, was missing. **It exists**: `https://iccvm.org/2026/files/papers/109.pdf`. The site blocks scripted downloads (TLS handshake reset over `schannel`, `urllib`, and `curl`), but the full text was recovered through a retrieval service **[W]**. It is recorded below.

A fourth note, about this survey rather than your data: the diffusion cluster's record was delivered late (`B_diffusion.md`) and is code-level like the others; I additionally cross-checked its two load-bearing claims against the paper text and `repos/DiffInk/utils/mask.py` before using them (§3.2, §3.3). One of those claims changes how DiffInk's numbers must be read — see §3.2.

---

## 1. The verdict in one screen

Every method splits along **three axes**, and the interesting cells are almost empty:

| Axis | Values present in the literature |
|---|---|
| Representation | **Online** = pen trajectory `(Δx, Δy, pen-state)` → SVG/raster; **Offline** = raster image |
| Unit | **Character** (most); **Word** (Latin); **Line** (few) |
| Style source | **None**; **Writer ID** (closed set); **Reference image(s)** (few-shot); **Reference trajectory** (line models) |

The decisive finding: **character-level Chinese generation is a solved, crowded problem; line-level Chinese generation has exactly two online methods (OLHWG, DiffInk) and a thin offline set, and *none* of them generate a contested, cursive boundary between characters.**

Answer to "can it generate a line of Chinese?" — sorted:

- **YES, natively (online trajectory):** `OLHWG` (ICLR 2025), `DiffInk` (ICLR 2026).
- **YES, natively (offline image), but Chinese only as characters:** `One-DM` (ECCV 2024), `DiffBrush` (ICCV 2025), `InkSpire` (ICLR 2026), `HistStyle`-line (IJDAR 2025).
- **PARTIAL — composes a line from per-character output, no boundary model:** `GANwriting` (Latin words), `HWT` (Latin words, fixed grid), `IJDAR-2025` (LLM layout + per-character images).
- **NO:** `Drawing RNN` 2016, `FontRNN` 2019, `zi2zi`, `CalliGAN` 2020, `WriteLikeYou` 2021, `SDT` 2023, `Elegantly Written` 2024, `HiSAC` 2026, `DNA` 2026, `WriteFreely` 2026, `FontDiffuser`.
- **Different task altogether:** `DiffRender` (ECCV 2026) — word-level online↔offline bridge.

---

## 2. Master table

`Cursive@boundary` = does the model represent what happens *between* two characters at all.
`Chinese` = is Chinese evaluated as a first-class script (not a demo)?

| Method | Venue | Repr. | Unit | Chinese | Line? | Cursive@boundary | Style input | Weights |
|---|---|---|---|---|---|---|---|---|
| Drawing & Recognizing RNN | arXiv'16 / TPAMI'18 | online | char | yes | no | no | none | — |
| FontRNN / PG-FontRNN | PG 2019 | online | font glyph | yes (font) | no | no | none (per-style model) | — |
| zi2zi | code only | offline | glyph | yes (font) | no | no | font ID | TF1, 27 fonts |
| CalliGAN | arXiv/CVPR-AI4CC'20 | offline | char | yes | no | no | ref image | none |
| GANwriting | ECCV'20 | offline | word ≤10ch | **no** (Latin) | partial | implicit only | **writer ID** | no |
| WriteLikeYou | CGF'21 | online | char | yes | no | no | K refs (meta) | data only |
| Handwriting Transformers | ICCV'21 | offline | word/line-image | **no** (Latin) | partial (grid) | within word only | 15 word refs | yes |
| SDT | CVPR'23 | online | char | yes | no | no | K=15 refs | **yes** |
| One-DM | ECCV'24 | offline | line | char-level only | **yes** (EN) | inherited from renderer | 1 ref | **yes** |
| Elegantly Written | ECCV'24 | online | char | yes | no | no | 3 glyphs | no |
| OLHWG | ICLR'25 | online | **line** | **yes** | **yes** | **no (by construction)** | 1 line prefix | no |
| DiffBrush | ICCV'25 | offline | line | partial | **yes** (EN) | implicit | ref + masks | yes (announced) |
| DiffInk | ICLR'26 | online | **line** | **yes** | **yes†** | **implicit only** | ref trajectory | **yes** |
| HiSAC | ECCV'26 | both | char 64² | yes | no | no | 5 glyphs | no |
| DNA | WACV'26 | online | char | yes (trad.) | no | no | ref + open-set | — |
| WriteFreely | ICCV'26 wksp | online | char (recon.) | yes | no | simulated (Bézier) | K refs | no |
| InkSpire | ICLR'26 | offline | line/multi-line | **bilingual** | **yes** | implicit | in-context | — |
| DiffRender | ECCV'26 | both | word | no | no | n/a | ref | — |
| CASHG / BoundInk | arXiv'26 | online | sentence | glyph-level only | **yes** (EN) | **explicit** | writer-conditioned | code |
| Yao et al. | IJDAR'25 | offline | **line** | **yes** | **yes** | **no (LLM layout)** | multi-level | — |

† DiffInk's line-level numbers are **prefix-conditioned**: the first ~30 % of the target line is fed clean, so the model generates only the remainder (§3.2). Treat it as prefix-continuation, not as text→line generation.

‡ OLHWG and DiffInk are the only two Chinese line-level methods; everything else in the "line" column is either Latin or a composition of per-character output.

---

## 3. Methods, grouped by what they teach CENT

### 3.1 The online Chinese character lineage — the code CENT will actually reuse

This is where the reusable modules, weights and conventions live. All of it is **character-level and stays character-level**.

**Drawing and Recognizing Chinese Characters with RNN** (Zhang et al., arXiv 2016 / TPAMI 2018). MDN-RNN over pen points with a 3-state pen model (`pen-down`, `pen-up`, `end-of-char`), content from a learned 3,755-class embedding, **no style control**. Line-level: **NO** — generation halts at end-of-char, and the paper names cursive text synthesis as future work ("an interesting future direction is to utilize the proposed method as building blocks for the synthesis of cursive handwritten Chinese texts", p.13) **[A]**. Value to CENT: the canonical pen-state convention and the cost-sensitive loss weights `[1,5,100]` that almost every later method inherits.

**FontRNN / PG-FontRNN** (Pacific Graphics 2019). Not a handwriting-style method: it generates **Chinese font glyphs** for thousands of characters from a small hand-designed set — skeleton, then U-Net contour prediction, then a 300×300 bitmap — using monotonic attention over a reference trajectory (the `std_*` convention). Critically, there is **no style encoder**: one model per style, and "each network needs to be trained from scratch" (p.3) **[A]**. Line-level: **NO**. Value to CENT: a documented failure mode worth citing — "the error will accumulate to a large extent if the sequence is long" (p.10) **[A]** — which is precisely the argument that chaining per-character generators cannot yield a coherent line.

**WriteLikeYou** (Tang & Lian, CGF 2021). Metric-based meta-learning with a style-attention module and angular centre loss; strongest few-shot style code in this lineage (0.604 zero-shot → 0.937 with 100 glyphs). Line-level: **NO**, stated outright: "we only discuss the isolated Chinese handwritten characters in this paper. We are planing to explore the generation of the coherent handwritten text segments in our future work" (p.10) **[A]**. Value: `StyleLuongAttention` and the writer-separation loss are directly reusable for CENT's style branch.

**SDT** (Dai et al., CVPR 2023). Style-disentangled Transformer: dual-head contrastive encoder separating **writer-level** from **character-level** style, content encoder, autoregressive decoder with GMM + pen state. Chinese/Japanese/English/Indic. Line-level: **NO** — one content image in, one character out; there is **no text-line, SynthText or concatenation mechanism anywhere in the paper (main text or appendices) or in the code** — this was checked twice, independently, because an early project note assumed one existed and it does not **[V]**. The authors' line-level method is a later, separate paper (DiffBrush). Value: the only shipped Chinese **style scorer** checkpoint among these repos, plus a strong char-level baseline. Its own README notes that lines can only be obtained by exporting to TTF and adjusting spacing in third-party software — i.e. hard layout by construction **[V]**.

**Elegantly Written** (Liu et al., ECCV 2024). Represents a glyph as a command sequence `(M/L, x, y)` — Bézier strokes, ~180 tokens — with a sequence encoder/decoder and a **stroke/component decomposition** table (`CharacterData/decomposition.json`). Line-level: **NO** in the released code: the sample unit is one glyph's command sequence, every glyph is min-shifted and centred into its own 224×224 canvas, and the decoder is fed the user's own trace, so there is no text-code pathway at all **[A]**. Value to CENT: **the most reusable representation artifact in the whole set** — a connection stroke is just an `L` command that runs across a boundary, which is exactly what a pixel model cannot express. Its successor (Elegantly Written V2, "Next-Scale Prediction") exists and is the basis of WriteFreely.

**HiSAC** (Hierarchical Style Aggregation, ECCV 2026, Xidian). Few-shot Chinese character generation at 64×64, per-character reference set, online+offline. Line-level: **NO** — and it names cursive as future work itself (p.15) **[A]**. Value: the best available **third-party character-level comparison table** for CENT's baseline section — it reports OLHWG at DTW 0.9173 / Style 91.28 / Content 90.11 versus HiSAC 0.8224 / 97.01 / 96.68 (Table 1, p.10), and it had to *retrain* the line-level method down to characters to compare **[A]**.

**DNA** (WACV 2026, NYCU). **Open-set** online generation: both the writer and the characters are unseen at test time. Style branch models stroke attributes ("writing direction, spacing, placement, and flow"); content branch decomposes characters into **IDS structural/component** encodings to generalise to unseen glyphs. Traditional Chinese (SCUT-COUCH2009) + Japanese, characters rendered at 64×64. Line-level: **NO** **[V]** (abstract + method read directly). Value: the strongest argument that *character content* can be decoupled from class identity — useful if CENT wants open-vocabulary lines.

**WriteFreely** (ICCV 2026; PDF recovered from the conference site `iccvm.org`) **[W]**. Not a generator: a **trajectory reconstruction/refinement** method — it takes an already-written, jittery electronic trace plus the writer's drawing parameters and re-renders a cleaner, style-preserving trajectory, via **next-scale prediction** (intra-block parallel, inter-block autoregressive) and a frequency-domain style module that perturbs amplitude while freezing phase. Character-level, CASIA-OLHWDB 1.0–1.2 (~3.7 M samples, 1,020 writers) + GIAHCC-UCAS2024. Line-level: **NO**. Its one direct relevance to CENT is a warning: to manufacture cursive it **joins a random subset of strokes with quadratic Bézier curves** ("stroke concatenation … reflecting speed-induced cursive effects"). That is the naive shortcut CENT must beat, and it is now published and citable as such.

### 3.2 The online line-level pair — CENT's direct competition

**OLHWG — "Decoupling Layout from Glyph in Online Chinese Handwriting Generation"** (Ren, Zhang, Chen; ICLR 2025; CASIA). The first serious Chinese text-line generator. It factorises

`p(X, Layout | C, X_ref) = p(Layout | C, X_ref) · Π_i p(x_i | c_i, X_ref, Box_i)`

i.e. **it assumes character independence given the bounding box** — stated in the paper as an assumption about the writing process ("when the bounding box of a character is given, the generation of the character is independent of other characters", p.4) **[V]**. Layout is a **2-layer LSTM (hidden 128)** emitting four numbers per character `[height, width, vertical centre, Δx]` from the previous box plus a 32-d embedding of six per-class box statistics (`train_box_generator.py:36-98`), conditioned in-context on the first ~10 true boxes of a reference line, and trained on lines of up to **25** characters (`train_box_generator.py:124`); each glyph is denoised independently by a 1D U-Net diffusion model and then **rescaled to exactly fill its box** and concatenated (`bonibox_gen/simplebox.py:92-97`, `generate_line.py:212-217`) **[A]**.

Three facts matter more than its numbers:
1. **The boundary is ink-free by arithmetic.** Because each glyph is normalised to `max−min` before scaling, no glyph can leave its box; boxes may overlap in the ground truth but the ink never does. There is no connection loss, no ligature state, no endpoint-matching term in the repository **[A]**.
2. **The authors concede CENT's thesis verbatim:** "due to the fact that each character is generated independently, our approach encounters difficulties in mimicking styles with extensive cursive connections between characters" (p.10) **[V]**. Appendix A.3 (p.17) goes further: an end-to-end variant *does* produce connective strokes between characters, but "as the number of characters increases, structural errors tend to occur" **[W]**.
3. **Its line metrics cannot see continuity.** The evaluation is eight inter-character *geometric* distances (∇1–∇8: centroid gaps, bound gaps, height/width ratios) plus recogniser AR/CR (0.852 / 0.857) — a layout metric that is blind to whether strokes join **[A]**.
4. **It trains its style reference on synthetic lines and infers on real ones.** The character-level style classifier is trained on **synthetic 15-character concatenations** (`style_classifier.py:126-155`) while inference is driven by a real line prefix (`generate_line.py:97`) — a train/infer mismatch inside the method itself **[A]**.

**DiffInk — "Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation"** (Pan et al., SCUT; ICLR 2026). **The strongest competitor and the mandatory baseline.** It is the first latent-diffusion-Transformer for **full-line** online handwriting: `InkVAE` (1D conv VAE, 8× temporal compression, GMM-20 decoder, 3-class pen state) regularised by an **OCR loss** (glyph accuracy) and a **style-classification loss**; then `InkDiT` denoises the full-line latent conditioned on text + reference style, DDIM sampling **[V]**.

Numbers (Table 1, p.7) **[V]**: AR 94.38 / CR 94.58 / Style 77.38 / DTW 1.049 / **58.47 chars/s** vs OLHWG 91.48 / 91.71 / 44.74 / 1.326 / 0.07. Data: CASIA-OLHWDB 2.0–2.2, 500 writers sampled (400 train / 100 test), **augmented to ~67,000 lines over 2,648 characters**. The augmentation is the key detail for CENT: they sample **isolated characters written by the same author** from OLHWDB 1.0–1.2, pick a real text line as a **layout template**, and paste the sampled characters into that line's original character BBoxes (Appendix A, p.16) **[V]**. Also evaluated on IAM-OnDB English lines (221 writers, "substantial cursive") **[V]**.

**How those numbers were obtained — read this before comparing against them.** The benchmark is **prefix-conditioned on the target line itself**, not a genuine full-line generation:
- `repos/DiffInk/utils/mask.py:1-45` builds a mask with `prefix_ratio=0.3`: the first ~30 % of the trajectory is marked as given (its latent mask is zeroed) and the first ~30 % of characters are excluded from the CTC head. The DiT therefore *generates only the last ~70 %* of every line **[V — read in code]**.
- The paper confirms the reference is the line's own beginning: "In the main experiments, we use a continuous reference segment from each text line as the style reference" (p.23, D.2), and "line-level metrics … are computed with each text line conditioned on its own reference segment" (p.24) **[V]**.

So the headline 94.38 AR describes *continuation from a true 30 % prefix*, not generation of a line from text + a writer. Its one-shot/短-reference regime exists only qualitatively (App. D.5, Fig. 17) **[A]**. **CENT should report prefix-free numbers and a cross-line style reference, and say so explicitly** — no published Chinese line-level number is comparable on that basis.

Its own framing hands CENT its opening: "character layout and connections cannot be fully disentangled, as they remain inherently context-dependent" (p.7) **[V]** — a *conclusion* that joint modelling is needed, without an explicit boundary mechanism.

### 3.3 Offline image models — mostly font transfer, not handwriting

**zi2zi** (code only; no paper). pix2pix-style conditional GAN, 256×256, one glyph per canvas, style = a **font ID embedding** from a closed catalogue (`embedding_num=40`), no style image, no writer. Line-level: **NO**. Critical distinction for CENT: this is **font-style transfer**, and its released weights encode font identity — unusable for handwriting **[A]**.

**CalliGAN** (arXiv 2020). Style/structure-aware Chinese calligraphy generator, 256×256, attention over a component/structure parser. Line-level: **NO**; one character per canvas. Note a paper↔code discrepancy: the paper describes an LSTM component encoder, the code runs an 8-head Transformer (`models/unet_onehot_cns_font_attention.py:142-186`) **[A]**. No licence, no weights.

**FontDiffuser** (AAAI 2024) — **paper not in the supplied set** (mis-filed, see §0). From the repo: two-stage diffusion font-style transfer with content/style encoders and a component-aware SCR module; unconditionally one character per canvas (`dataset/font_dataset.py:55-59` fixes one style image). Line-level: **NO**; font transfer, not writer mimicry **[A]**. Weights released, non-commercial.

**GANwriting** (ECCV 2020). The only method here that models **writers**: content from a text string, style from K reference word images, and a **writer ID used as supervision** (a classification head), never as a generator input. Output 64×216, **≤10 characters**. Line-level: **PARTIAL** — a fixed-length character strip, a "soft grid", with uniform per-character embedding repetition (`modules_tro.py:242-259`); its `tt.word_ladder.py` probes show boundaries *do* respond to content (`sired`→`fired` ligature, p.12) but **no boundary variable exists** — there is no pen state and no inter-glyph geometry **[A]**. Two paper↔code discrepancies to avoid importing into a comparison table: code uses **K=50** stacked style images vs the paper's 15, and the paper's additive style noise `Z` is absent from the code **[A]**.

**Handwriting Transformers** (ICCV 2021) — **paper not in the supplied set** (mis-filed, see §0); record from repo **[A]**. Transformer encoder-decoder producing a **fixed-size image strip** `32 × 16L` pixels, with an OCR network in the loop; Latin only (`params.py:22-23` `ALPHABET`), 15 word-image references, and **layout is a fixed uniform 16-px grid** tiling independently generated words, up to 16 words per line (`models/model.py:463`) **[A]**. Line-level for Chinese: **NO** on two independent counts (no CJK content channel; grid positions). For Latin: partial, by tiling.

**DiffBrush** (ICCV 2025, SCUT — the SDT authors' line-level successor) **[W]**. Diffusion over **line images**, with content-decoupled style learning via column/row masking and line+word discriminators; IAM/CVL. Not in the supplied folder, and it is required related work for CENT.

**One-DM** (ECCV 2024). One-shot style diffusion from a single reference, with a **Laplacian high-frequency branch** (`L_lapNCE` contrastive) and a gate; explicitly targets "character slant and letter joining" **(p.1)**, and the ablation states the mechanism outright: "the independent addition of the Laplacian branch helps the model **learn cursive connections** and other [style patterns]" (p.11-12) **[V]**. **Line-level for English** (IAM/CVL); its **Chinese evaluation is character-level** — Table 5 reports "styled handwritten character generation on Chinese and Japanese" on the ICDAR-2013 database (p.8) **[V]**, and the shipped Chinese assets (`assets/casia_v4.png`, `assets/Chinese.png`) are isolated-character grids with **no CJK config, data or checkpoint** in the repo **[A]**. The structural point for CENT: the content condition is a **rasterised text image assembled from per-character font bitmaps on a fixed advance** (code: a 94-symbol ASCII alphabet, 16×16 bitmaps; `data_loader/loader.py`) **[A]** — so character **order and advance are decided by a renderer**, while only the fine spacing and joins are learned. The model restyles a grid; it does not predict a layout. And the content channel is *vocabulary-free*, which is the one design here that would scale to a 20 k-character Chinese vocabulary.

**InkSpire** (ICLR 2026, CASIA) **[W]**. DiT that unifies style, content and noise in one latent space with no separate encoders; trained on **raw text-line images** with **multi-line masked infilling** and a revised positional encoding for arbitrary-length multi-line synthesis; **bilingual Chinese–English**; evaluated on IAM and ICDAR2013. Offline, line-level, image. This is the current offline state of the art and belongs in CENT's related work. Its sibling line — **Template-Guided Cascaded Diffusion for Stylized Handwritten Chinese Text-Line Generation** (Wang, Ren, Zhang, Yin, Liu; LNCS/ICONIP 2025) — is the same CASIA group applying cascaded diffusion to Chinese text lines, and should be cited alongside it **[W]**.

**Yao et al., IJDAR 2025** (Tsinghua) **[W]**. Offline **Chinese text-line** generation by a two-step recipe: style-controlled character images (radical-level contrastive pre-training, multi-level style control) + **text-line layout generated by prompt-engineering a large language model**. Chinese line-level: **YES**, but again as composition — an LLM decides positions, character images are generated separately. Third layout paradigm after learned-LSTM (OLHWG) and fixed grid (HWT).

**DiffRender** (ECCV 2026) **[V]**. Bridges online and offline by **differentiable physical rendering**: a compact **physical brush model** with six interpretable parameters `θ = {w_base, k_spread, ρ_ink, σ_sharp, p_min, p_max}` (width-to-pressure, ink density, edge falloff) plus a differentiable renderer that turns stroke trajectory + a **kinematic pressure proxy** into ink (`O_t = 1 − exp(−ρ_ink p_t)`); the trajectory generator is an LSTM with **K=20** GMM pen-position mixture and pen-state logits, and the rendered result is refined by diffusion. It also demonstrates robotic calligraphy. Evaluated at **word** level (IAM Words, CVL Words); the paper states that extending to coherent long-form sentence generation remains future work. English only. Value to CENT: a principled way to reconcile raster and trajectory supervision — exactly the machinery a train/infer gap needs — plus an ink-realism head; but it is word-level and has no Chinese.

### 3.4 The English boundary-aware line — CENT's methodological competitor in spirit

Not in the supplied folder; recovered online. These matter because they already contain **most of the conceptual machinery** for contribution (ii), just not for Chinese **[W]**:

- **CASHG / BoundInk** (arXiv 2026). Treats **inter-character boundaries as explicit generation units**: a Character Context Encoder, a **bigram-aware sliding-window Transformer** that supervises predecessor→current transitions, **connectivity-aware supervision with a dedicated `CursiveEOC` pen-state label for pen-down character continuation**, a Vertical Drift Loss, a three-stage curriculum (glyphs → bigrams → sentences), and a **Connectivity and Spacing Metrics (CSM)** suite (F1_Cursive, CRE, KGS, SSS). Its Chinese experiments are *glyph-level only* under the SDT protocol, and it states that under the OLHWG Chinese protocol "F1Cursive/CRE/SSS are not meaningful in this dataset (**no cursive positives** and no whitespace runs)".
- **Samsung / ICASSP 2026**: post-hoc **ligature synthesis** — a lightweight BiGRU classifier predicts head/tail/body/isolated per point, then a trace-deformation optimisation (connection distance + smoothness + displacement, solved as a tridiagonal system in O(N)) joins adjacent glyphs. Non-generative, character-independent, Latin/Cyrillic.
- **"Style-Preserving English Handwriting Synthesis"** (Zhou et al.): a **cursiveness parameter 0–1**, with connection probability factorised into a writer-independent prior × the writer's cursiveness, and ligature generation by polynomial fitting. This is the conceptual ancestor of any "cursive slider" and must be cited — it means *control over cursiveness is not by itself novel*.

---

## 4. What the prior art does and does not allow CENT to claim

### Contribution (i) — per-writer style mimicry: **not novel alone**

Few-shot/one-shot writer mimicry is solved and heavily benchmarked: SDT (K=15), WriteLikeYou (meta-learning), One-DM (**one** reference), OLHWG (a ~10-box line prefix), DiffInk (a reference trajectory — in practice a segment of the target line itself, §3.2), DNA (open-set, unseen writer **and** unseen characters). HiSAC and Elegantly Written are per-glyph.

**Where a gap remains:** style is always split across *channels* — OLHWG needs **one 1,000-point trace for glyph style** and **a separate line for layout style**; DiffInk conditions on a reference **trajectory** and therefore needs the user to have written a line already; nothing represents a writer's **continuity habit** (how often, where, and how hard they connect). A single writer code that jointly captures glyph shape + layout habits + **ligature propensity** is defensible — and it is measurable (boundary statistics per writer).

### Contribution (ii) — free cursive continuity instead of hard layout: **partly open, and the hardest to defend**

The Chinese line-level space has exactly two occupants:

- **OLHWG**: cannot do it, **by construction and by admission** (§3.2). The cleanest possible motivation citation, pp.10 + 17.
- **DiffInk**: does it **implicitly** — a full-line trajectory model can in principle emit a stroke across a boundary, and its English IAM-OnDB experiment shows it handles "substantial cursive". But there is **no explicit boundary mechanism**: no ligature state, no boundary loss, no control, and its Chinese training lines are pasted from character-level data (§3.2), so cursive coverage in its supervision is whatever CASIA-OLHWDB 2.0–2.2 happens to contain — which third-party evaluation indicates is **essentially none**. Decisively for any comparison table: its reported lines are produced **from the target line's own first ~30 %** (§3.2), so *no published Chinese number measures text→line generation without a prefix*.

So CENT cannot claim "first line-level Chinese generator" (OLHWG, DiffInk) nor "first end-to-end" (DiffInk). What remains defensible:

1. **Explicit boundary modelling** — a boundary/ligature variable with its own supervision and its own decode step, rather than hoping a full-line denoiser learns it. This directly contradicts OLHWG's independence assumption and matches what CASHG/BoundInk had to add for English.
2. **Controllable continuity** — a continuous degree of connection decoupled from content and from writer identity (InkSyn shows the concept exists for English with hand-built geometry; doing it learned, style-conditioned, for Chinese is new).
3. **A continuity metric for Chinese** — nothing in the Chinese literature evaluates the join (OLHWG measures only geometric distances and recognition). CASIA's line test sets have **no cursive positives**, so a Chinese continuity benchmark has to be built, not borrowed. This is both a contribution and a reviewer risk.

**Caveat that must appear in the paper.** The *idea* of learning cursive joins is not new: One-DM attributes its cursive ability to a high-frequency branch (§3.3), and InkSyn exposes an explicit cursiveness parameter. CENT's novelty is therefore **not** "we learn joins" — it is *learned, controllable, boundary-explicit joins for a script whose training lines contain almost none*. State it that way or a reviewer will cite One-DM and InkSyn.

### Contribution (iii) — train on clean character data, infer with cursive continuity: **genuinely open**

This is the most honest and least occupied of the three. Current practice does one of four things, none of which addresses the shift:

| Strategy | Who | Problem |
|---|---|---|
| Train on real lines only | DiffInk (400 writers, CASIA 2.0–2.2) | line corpora are ~500 writers, ~2.6k chars, and virtually no cursive |
| Train chars, paste at inference | OLHWG, IJDAR-2025, character baselines + layout modules | boundary is ink-free; stitching artifacts are the visible failure |
| Synthesise lines by pasting chars into template boxes | DiffInk augmentation (§3.2) | inherits the discontinuity it is meant to fix; no ligature supervision |
| Simulate defects/joins with Bézier curves | WriteFreely (§3.1) | geometrically fake joins; published as a shortcut CENT can cite as insufficient |

**Nobody frames the problem as a distribution shift between the training (clean, isolated, pen-lifted) and inference (continuous, cursive) manifolds, and nobody trains for it.** The supporting evidence is unusually strong: the character corpora are large (CASIA-OLHWDB/HWDB 1.x ≈ 3.7–3.9 M samples, 1,020 writers, 3,755 GB2312 level-1 characters) while the line corpora are small and ligature-poor; and an independent 2026 paper (CASHG) explicitly reports that "no cursive positives" exist in the Chinese line benchmark it evaluated on. A method that *pairs* character-level supervision with a learned ligature prior — trained from the few cursive examples that exist, or from cross-script/word-level line data — is new for Chinese.

---

## 5. Borrowing plan (code + weights)

Ranked by value per unit of integration effort, from the repositories actually present:

| Source | What to take | Why |
|---|---|---|
| **SDT** (`repos/SDT`) | Chinese online generator weights + **Content/style scorer checkpoints** + the 3-state GMM pen decoder | Only shipped Chinese style scorer in the set; gives CENT a char-level prior and free evaluation metrics |
| **DiffInk** (`repos/DiffInk`) | `model/vae.py` (InkVAE), `model/gmm.py`, `model/ocr.py`, `model/writer.py`, `dataset/vae_dataset.py`, the DiT (`model/dit.py`) + **released weights (MIT code)** | The only line-level Chinese stack with weights; its OCR-in-latent and style-classifier losses are directly reusable regularisers |
| **OLHWG** (`repos/OLHWG`) | `bonibox_gen/` (LSTM box generator + the 8 geometric features in `evaluate_box.py`), `style_classifier.py` | The layout baseline to beat, and a ready-made layout metric; **note: released `generate_line`/`evaluate_line` do not run as written, and no licence is shipped** |
| **Elegantly Written** (`repos/ElegantlyWritten`) | `decomposition.json` (stroke/component table), the `(M/L,x,y)` command format, `seqDec.py` | A boundary-crossing connection stroke is expressible only in this representation |
| **One-DM** (`repos/One-DM`) | High-frequency style branch, OCR model (`vae_HTR138.pth`), 10,400-class style ResNet | One-shot style extraction + a strong pre-trained Chinese/production OCR critic |
| **DNA** (`repos` not present; paper) | IDS structural/component encoding idea | Open-vocabulary content conditioning for unseen characters |
| **GANwriting** (`repos/GANwriting`) | Recogniser-in-the-loop loss (`recognizer/`), content-swap invariance, word-ladder diagnostic | Cheap content-fidelity supervision and a boundary diagnostic protocol |

Do **not** borrow: zi2zi / FontDiffuser / CalliGAN weights (they encode *font* identity, not a hand), and CalliGAN has no licence.

---

## 6. Risks and decisions this survey forces

**R1 — DiffInk is a direct, recent, strong competitor** with code and weights (ICLR 2026). CENT's contribution (ii) cannot be "line-level Chinese"; it must be the *boundary mechanism* DiffInk lacks. Budget for it as the primary baseline and write the novelty paragraph against its p.7 conclusion. **But do not chase its numbers as if they were comparable:** they are prefix-conditioned on the target line's own first 30 % (§3.2). The honest framing is that CENT reports the harder setting — generation from text + a writer reference with no prefix — and that DiffInk's own appendix confines the one-shot regime to qualitative figures.

**R2 — Contribution (ii) may read as incremental** against CASHG/BoundInk (explicit boundary units, `CursiveEOC`, CSM metrics — for English). CENT must make the Chinese case: no whitespace structure, thousands of classes, ligatures that are stylistic rather than mandatory, and a far worse supervision deficit.

**R3 — Cursive Chinese supervision is scarce and, worse, unmeasured.** This is the crux of contribution (iii). A dedicated data search (→ **`survey/cursive_chinese_line_data.md`**) found that **no dataset is purpose-built for cursive Chinese lines**, but cursive material exists in at least six corpora and, critically, **nobody has measured how much cursive the standard benchmark contains**. The notable find is **DCOH-120K** (83,142 Chinese online lines, 314 writers, **pressure + timestamps**, informal register) — larger in lines than CASIA-OLHWDB 2.x and far better suited to a connection-stroke study, yet it does not appear in any generation paper found. Three routes, and the choice is yours: (a) measure first, then learn from what exists (DCOH + OLIWER's 674 k boundary-spanning phrases); (b) simulate joins with a learned prior — weak, since WriteFreely already publishes the naive Bézier version; (c) cross-script transfer from Latin/Cyrillic/Arabic sentence data (CASHG, InkSyn, Samsung). **Do not** treat 行书-font synthesis as a substitute: it gives glyph-level cursive, not connection strokes with pen dynamics.

**R3a — The measurement is the decision.** Before any architecture work, run the pilot in `cursive_chinese_line_data.md` §5: compute the pen-lift gap (`Δt`, `Δs`) for every adjacent character pair in CASIA-OLHWDB 2.x and DCOH-120K. If a near-zero mode exists, contribution (iii) becomes *long-tail, class-imbalanced* supervision rather than *no data* — a much stronger and more defensible claim. If not, R3(a) dies and CENT must use (b) or (c).

**Caution if you start with (a):** synthesised pairs are not free supervision. DiffRender's own ablations report that renderer-generated online–offline pairs *degrade* an image-space model (One-DM FID 52.26 vs 29.00 real; CVL CER 0.2877 → 0.6285) **[A]**. Any synthetic-join strategy for CENT needs an explicit real-vs-synthetic ablation before it can be claimed as a contribution.

**R4 — You need a continuity metric before you can claim continuity.** CASIA line sets have no cursive positives. Build the evaluation set and the metric (a Chinese analogue of F1_Cursive/CRE, plus a human study) *before* training, or the claim is unfalsifiable.

**R5 — Character-level Chinese data is abundant (≈3.9 M samples, 1,020 writers); line-level is scarce (~500 writers, ~2.6 k chars).** Any design that needs paired line supervision inherits the scarcity; any design that uses only character data inherits the gap CENT claims to solve. The architecture has to sit on the character side and *import* boundary knowledge.

**R6 — Two PDFs in `papers/` are the wrong papers** (§0). Fix before the related-work section is drafted.

---

## 7. Where the full evidence lives

| File | Contents |
|---|---|
| `survey/_records/A_online.md` | Full 12-field records: Drawing RNN, FontRNN, WriteLikeYou, SDT, WriteFreely |
| `survey/_records/B_diffusion.md` | Full 12-field records: One-DM, DiffInk, DNA, DiffRender (code-level; the lead re-verified its two load-bearing claims) |
| `survey/cursive_chinese_line_data.md` | **Dataset survey**: cursive Chinese line corpora, licences, modality, and the pilot experiment that decides contribution (iii). **§9 holds the key finding: SCUT-COUCH documents cross-character 连笔 and annotates per-character stroke indices.** |
| `survey/architecture_boundary_connection.md` | **Architecture note**: how to learn the join without a layout, the existing formalisation (CASHG/BoundInk `CursiveEOC`), three design options, failure modes, and the boundary-only probe |
| `cent/` (+ `scripts/`, `configs/`) | **Architecture implementation**: InkVAE + BoundaryDiT with interleaved boundary slots, AR baseline, losses, metrics, DDIM with cursive control. Verified by `scripts/smoke_test.py` (no training) |
| `datasets/README.md` | SCUT-COUCH access status (archives are AES-encrypted until the lab approves the application), and which categories CENT needs |
| `survey/_records/C_gan.md` | Full records: zi2zi, CalliGAN, GANwriting, FontDiffuser (+ GMM/MDN cross-check) |
| `survey/_records/D_line.md` | Full records: OLHWG, HWT, HiSAC, Elegantly Written |
| `survey/_pdftext/*.txt` | Page-marked text of every supplied PDF (15 files) |

A/B/C/D records carry `p.N` or `file:lines` for every claim, plus their own source-defect notes; A, C and D are code-level and cite repositories directly. This document is the synthesis to write the paper from: §1 for the claim, §2 for the comparison table, §3 for related work, §4 for the novelty argument, §5 for the build, §6 for the decisions still open.

---

*Prepared as a research conclusion, not an architecture proposal. No CENT design is specified here.*
