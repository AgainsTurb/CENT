# Peer-method survey for the Chinese handwritten **text-line** baseline table

**Purpose.** The comparison table in the manuscript currently has four baselines (DiffBrush, TGC-Diff,
DiffInk, OLHWG — the last cite-only). This survey retrieves and verifies the peer-method space for
*generating handwritten text*, ranks what should be **added** and in which **mode** (native line
generator vs. composed character/word generator), and extracts the experimental protocol of the two
most relevant papers so ours can be written to match.

**Method and honesty contract.**
- Every paper, repo, licence and number below was **retrieved** with `exa_search` / `exa_contents` and
  `tavily_search` / `tavily_extract`, or read directly from the repositories already cloned under
  `A:\Papers\CENT\baselines\`. No built-in web search was used.
- Repo licences were cross-checked two ways: the `LICENSE` file inside our local clone, and the GitHub
  REST API (`license.spdx_id`) on 2026-11-22.
- Anything I could not retrieve is marked **`UNVERIFIED`** or **`NOT RETRIEVED`** and is never filled in
  by inference. Where a number comes from a *third party's* re-implementation rather than the paper's own
  authors, it says so.
- This document does **not** re-run anything. It records literature facts and protocol text.

Retrieval window: 2026-11-22. Query sets used: `Chinese handwritten text line synthesis`, `full-line
handwriting generation`, `text-line handwriting diffusion`, `手写文本行生成`, plus per-method title
queries for all 16 named methods and for the two named Chinese-specific papers.

---

## 0. Ranked recommendation — what to ADD, and in which mode

**Mode vocabulary used throughout** (this vocabulary is the contribution of this document to the table):

| Code | Meaning | Why it matters for CENT |
|---|---|---|
| **N** | **Native line model** — one model, line-level output; layout and inter-character relationships are produced by the same generator. | The only mode that *can* produce cursive connectivity as a by-product. Comparable on connectivity metrics. |
| **ND** | **Native with decoupled layout** — output is a line, but glyphs are generated independently and placed by a layout planner. | Cannot express inter-character connectivity *by construction*; the paper itself says so (OLHWG). |
| **C** | **Composed** — character/word generator plus external assembly. No layout model, no connectivity. | Not comparable on connectivity; comparable on character fidelity only. Must be labelled if used. |
| **NR** | **Not runnable** — no code, no weights, no licence, or broken entry point. | Cite published numbers as *not re-run*. |

### Ranked list

| # | Method | Mode to use | Verdict | Why | Cost / what changes if omitted |
|---|---|---|---|---|---|
| 1 | **DiffInk** (ICLR 2026) | **N**, online | **ADD now** | Only peer method that is *natively line-level, Chinese, and reports a throughput number* (`Avg. char/s`), so it anchors the efficiency claim that ours must beat. Already cloned with 42/42 adapter checks; MIT; weights advertised as released (README update 2026/3/21). **Must be reported as prefix-conditioned line completion, not text-only** (see §3.4). | Without it the table has no online-family *native* baseline and the efficiency comparison has no anchor. |
| 2 | **One-DM** (ECCV 2024) | **C** (line-as-one-string; no layout model) | **ADD** | The field's standard one-shot reference, MIT, weights released, and — decisively — **DiffBrush itself uses One-DM as its only Chinese text-line comparison** (Fig. 11). Adding it lets our table reproduce a row that already exists in the literature. Its Chinese capability is verified at *character* granularity on ICDAR-2013 (Table 5, Chinese FID 27.24). | Omitting it makes our Chinese table incomparable with DiffBrush's own Chinese figure. **Caveat:** the released checkpoint is the English word model; the README promises Chinese later. Whether Chinese weights actually shipped is `UNVERIFIED` — check before promising the row. |
| 3 | **TGC-Diff** (ICDAR 2025) | **N**, offline, Chinese | **KEEP** (already in) | Chinese-native line model, MIT, weights released, and its code ships the Chinese pipeline. Confirmed correct in the brief. Its *own* metrics are behind Springer's paywall and were **`NOT RETRIEVED`** — do not quote its numbers from the paper until the PDF is obtained. | — |
| 4 | **DiffBrush** (ICCV 2025) | **N**, offline; Chinese needs a code change | **KEEP** (already in) | Chinese support is a *code* change, not a data gap: the release ships `files/unifont.pickle` with 20,992 CJK glyphs (16×16), and the model consumes glyph *images*, never ids. Confirmed. | — |
| 5 | **Emuru** (CVPR 2025) | **N**, offline, **Latin only** | **ADD only if an English line table exists** | Genuine native line generator, arbitrary length, zero-shot, MIT, weights on HF; the one model that can be run **without training**, so it is cheap. But it is trained on synthetic Latin fonts and cannot produce Chinese. | If the paper stays Chinese-only, cite it as prior art rather than a row. |
| 6 | **TS-GAN** / **CSA-GAN** (BMVC 2020 / TPAMI 2022) | **N**, offline, **Latin only** | **CITE, do not run** | They are the *first* line-level generators and are the two baselines any reviewer will name. But both are alphabet one-hot models — **Chinese support: none** — and TS-GAN's licence is **CC BY-NC-SA 4.0** (non-commercial) while CSA-GAN has **no released code**. | Reviewers may ask why they are absent; answer in one sentence in the text, not with a row. |
| 7 | **InkSpire** (ICLR 2026) | **N**, offline, Chinese native | **CITE-ONLY** (stays as is) | Reports the strongest Chinese line numbers in the retrieved literature (ICDAR-2013: FID 10.98 / KID 11.45 / HWD 0.41 / CR 92.92 / AR 91.56) but the public repo is **1 KB, no source, no licence** — verified via API. | It is the current SOTA row in the table; it must be quoted with published numbers and marked not re-run. |
| 8 | **OLHWG** (ICLR 2025) | **ND**, online, Chinese | **CITE-ONLY** (stays as is) | No licence, no generator weights, and the released evaluation entry point cannot execute (verified by AST in `docs/BASELINE_AUDIT.md`). Its own conclusion concedes the connectivity failure — quotable. | — |
| 9 | **HWT**, **GANwriting**, **VATr/VATr++**, **DiffusionPen** | **C** | **DO NOT ADD to the line table**; keep in the archived composed/character table | All four are word-level, Latin, no layout. They are already adapted (54/25/94/1787 checks). Useful only as a *character-fidelity* ablation family. | Adding them to the main table invites the objection that they were never line models. |
| 10 | **MetaScript**, **SDT**, **HiSAC** | **C** | **CITE** | Chinese-native but **character-level** (SDT online character; MetaScript offline character + "Typewriter" composer; HiSAC character-level style aggregation). None is a line method. | — |
| 11 | **HCT-GAN** (ICPR 2022) | **N** (claimed) | **CITE, flag as un-obtainable** | Closest Chinese line prior art before 2025 — a components-regulated CGAN generating "entire lines of Chinese handwriting" in arbitrary length. **No code found; metrics behind IEEE paywall (`NOT RETRIEVED`).** | A Chinese-line paper that omits HCT-GAN looks like it did not search. One citation suffices. |
| 12 | **IJDAR multi-level style control** (2025) | **C** (character + LLM layout) | **CITE** | Explicitly claims to be *the* offline Chinese handwritten **text-line** method: "style-controlled character image generation and text-line layout generation". No code found. | — |
| 13 | **HandwritingAgent** (arXiv 2026) | **C** (LLM agent, per-character SVG) | **CITE with caution** | A 2026 entrant that reports a Chinese-line comparison against **DiffBrush's published numbers** on CASIA-HWDB1.1 and admits it could not get DiffBrush to run: "we compare performance against the reported scores." No code found. | Worth one sentence: it confirms that DiffBrush is hard to reproduce — which supports our own reproducibility note. |

**Bottom line for the table.** Today: four named baselines, of which **three run** (DiffBrush,
TGC-Diff, DiffInk) and one is cite-only (OLHWG). After this survey: **four runnable rows** — the same
three plus **One-DM as the composed row** — and **four cited rows** (InkSpire, OLHWG, HCT-GAN, and
TS-GAN/CSA-GAN as one English-line group). That gives a defensible CVPR table: two modalities ×
(native, composed) × (runnable, cited), with the mode column making the comparison honest. If an
English-line table is added, Emuru becomes a fifth runnable row at near-zero training cost.

---

## 1. Confirmations and corrections to the brief

| Item in the brief | Status | Evidence |
|---|---|---|
| DiffBrush = ICCV 2025, line-level, Chinese | **Confirmed** | ICCV 2025 OA, pp. 19054–19064; Appendix E is CASIA-HWDB 2.0–2.2. |
| TGC-Diff = ICDAR 2025, Chinese lines | **Confirmed** (and it is the **Best Paper**) | ICDAR 2025, LNCS pp. 149–166; repo README banner. |
| DiffInk = ICLR 2026 | **Confirmed** | ICLR 2026 proceedings; README "[2026/1/29] DiffInk is accepted by ICLR 2026". |
| OLHWG = ICLR 2025, cite-only | **Confirmed** | ICLR 2025 proceedings; repo has **no licence file** (API `license: NONE`). |
| HWT venue | **CORRECTED: ICCV 2021**, not CVPR | ICCV 2021 OA, pp. 1086–1094. |
| CSA-GAN venue | **CORRECTED: IEEE TPAMI 2022**, 44(12):8846–8860 — not a conference paper | DOI `10.1109/TPAMI.2021.3122572`; arXiv 2204.05539. |
| **MetaScript venue** | **CORRECTED: it is an arXiv preprint (2312.16251), not AAAI 2024** | The arXiv record carries no venue; the repo says "Project of AI3604 Computer Vision, 2023 Fall, SJTU"; the 2025 *Pattern Recognition* survey cites it as "2023, arXiv preprint arXiv:2312.16251". The earlier note in `docs/BASELINE_CHANGES.md` calling it AAAI 2024 is wrong. |
| "DiffPen" and "DiffusionPen" in the brief | **Same method** — "DiffPen" is the short name used by InkSpire/Emuru for DiffusionPen (ECCV 2024). | InkSpire: "DiffPen (Nikolaidou et al., 2024)"; Emuru tables use "DiffPen"; the paper itself writes "DiffusionPen (DiffPen)". |
| HiSAC | **Identified**: *Hierarchical Style Aggregation for Versatile Chinese Handwriting Generation*, **ECCV 2026**, Chinese-native, **character-level**. Repo is Apache-2.0 but **5 KB** → still no source. | ECCV 2026 PDF + Springer DOI + API size. |
| One-DM "word-level only" | **Partly corrected**: word-level by design, but it **does** report Chinese, at **character** granularity (ICDAR-2013, FID 27.24). DiffBrush additionally runs it as a **Chinese text-line** comparison. | One-DM Table 5 and §4.5; DiffBrush Fig. 11. |
| TGC-Diff's reported metrics | **`NOT RETRIEVED`** — the ICDAR chapter is paywalled and only the abstract + reference list were extractable. | Springer page returned references only. |

---

## 2. Full table

### 2.1 Line-level methods (can be scored as a line baseline)

| Method | Venue / year | Paper URL (resolved) | Code URL | Licence | Modality | Granularity | Chinese | Mode | Metrics reported (by its own authors) | Datasets / splits used |
|---|---|---|---|---|---|---|---|---|---|---|
| **DiffBrush** | ICCV 2025 (pp. 19054–19064) | https://openaccess.thecvf.com/content/ICCV2025/html/Dai_Beyond_Isolated_Words_Diffusion_Brush_for_Handwritten_Text-Line_Generation_ICCV_2025_paper.html | https://github.com/dailenson/DiffBrush | **MIT** (LICENSE © 2025 Gang Dai; API MIT) | offline | line | **native** (Unifont content encoder; needs a Chinese dataloader — code change, not data gap) | **N** | HWD, D_CER, D_WER, FID, IS, GS (English); HWD, FID, D_AR, D_CR (Chinese) | IAM: 13,353 lines / 657 writers → **496 train / 161 test** (CSA-GAN protocol). CVL English: 11,007 lines → **283 / 27**. CASIA-HWDB 2.0–2.2: 52,230 lines / 1,019 writers → **816 / 203**. Resolution: height 64; **64×1024**. |
| **TGC-Diff** | ICDAR 2025, LNCS pp. 149–166 (Best Paper) | https://doi.org/10.1007/978-3-032-04614-7_9 | https://github.com/newbie2niubility/TGC-Diff | **MIT** (LICENSE © 2025 Honglie Wang; API MIT) | offline | line | **native** (Chinese pipeline shipped) | **N** | **`NOT RETRIEVED`** (paywall). Third-party (InkSpire, ICDAR-2013): FID 23.43 / KID 13.85 / HWD 0.63 / CR 89.99 / AR 89.13; style score 60.46 % — *as evaluated by InkSpire, not by the authors* | **`NOT RETRIEVED`** from the paper. Code expects `data_root/data_ch/{train,test}/<writer>/<passage>/<stem>.png` + mirrored `data_temp_ch/.../<stem>_template.png` (verified in `baselines/tgc_diff/PATCHES.md`, commit `4d4058d`) |
| **InkSpire** | ICLR 2026 | https://proceedings.iclr.cc/paper_files/paper/2026/file/963ccebfefe8e349cad24e3ed4a92907-Paper-Conference.pdf | https://github.com/newbie2niubility/InkSpire — **1 KB, empty** | **none** (API `license: NONE`) | offline | line + multi-line | **native** (bilingual CN–EN) | **N (but NR)** | FID, KID, HWD, CR, AR (Chinese); FID, KID, HWD, ΔCER (English); style score | IAM 13,353 lines → **496 / 161**; **train on CASIA-HWDB2.0-2.2** (52,230 lines, 1,019 writers); **test on ICDAR-2013** (3,432 lines, 60 writers). 64×1024 normalisation for the style classifier |
| **DiffInk** | ICLR 2026 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/8547df6ea717b4aad8ae4e52b46a4a4f-Abstract-Conference.html | https://github.com/awei669/DiffInk | **MIT** (LICENSE © 2026 Wei Pan) | **online** (trajectories) | line | **native** | **N** (prefix-conditioned — see §3.4) | AR, CR, Style (writer-classifier accuracy), DTW (normalised), **Avg. char/s** | CASIA-OLHWDB 2.0–2.2, 500 writers sampled (400 train / 100 test); training augmented to **67,000 lines, 2,648 characters**; test **4,780 lines** |
| **OLHWG** | ICLR 2025 | https://proceedings.iclr.cc/paper_files/paper/2025/file/0e0df370cd1ce52c09bb5c1aa584625d-Paper-Conference.pdf (arXiv 2410.02309) | https://github.com/singularityrms/OLHWG | **none** (no licence file; API `NONE`) | **online** | line (layout + per-character glyph) | **native** | **ND** | Layout: ∇1–∇8 binary geometric feature deltas. Readability: **AR / CR**. Character-level: DTW, Content Score, Style Score | CASIA-OLHWDB 2.0–2.2 ≈ 52,000 lines / **1,200 writers → 1,000 train / 200 test**. Characters: CASIA-OLHWDB 1.0–1.2 (≈3.7 M chars, 1,020 writers) → test ICDAR-2013 (60 writers × 3,755 chars) |
| **TS-GAN** (Davis et al.) | BMVC 2020 | https://www.bmva-archive.org.uk/bmvc/2020/assets/papers/0815.pdf (arXiv 2009.00678) | https://github.com/herobd/handwriting_line_generation | **CC BY-NC-SA 4.0** (LICENSE read in full; API reports `Other`) | offline | **line** | **none** (one-hot Latin alphabet) | **N** (English only) | FID, GS; human study (Amazon Mechanical Turk, **14,875 responses**, 52.2 % accuracy) | IAM **6,161 train / 1,840 val** lines; RIMES lines. Height 64 |
| **CSA-GAN** (Kang et al.) | **TPAMI 2022**, 44(12):8846–8860 | https://arxiv.org/abs/2204.05539 (DOI 10.1109/TPAMI.2021.3122572) | **no code found** (the GitHub repo named `CSA-GAN` is an unrelated text-to-image model) | n/a | offline | **line** | **none** (alphabet) | **N** (English only) | HTR **CER / WER** on IAM test lines (Table VII); defines the IAM line split others copy | IAM lines (defines the **496 / 161** writer split that DiffBrush follows) |
| **Emuru** | CVPR 2025 | https://openaccess.thecvf.com/content/CVPR2025/papers/Pippi_Zero-Shot_Styled_Text_Image_Generation_but_Make_It_Autoregressive_CVPR_2025_paper.pdf | https://github.com/aimagelab/Emuru-autoregressive-text-img | **MIT** (API MIT) | offline | line (arbitrary length) | **none** (Latin; trained on synthetic fonts) | **N** (English only) | FID, **BFID**, KID, **ΔCER** (TrOCR-Base), HWD | IAM words **and** lines, CVL lines, RIMES lines, own `Karaoke` set. **Trained only on synthetic data**; zero-shot on all real sets |
| **HCT-GAN** (Li et al.) | ICPR 2022, pp. 1378–1385 | https://doi.org/10.1109/ICPR56361.2022.9956698 | **no code found** | n/a | offline | line ("arbitrary length") | **native** | **N** (claimed) | **`NOT RETRIEVED`** (IEEE paywall; only abstract obtained) | **`NOT RETRIEVED`** |
| **Multi-level Style Control** (Yao et al.) | IJDAR 2025, 28(3):429–440 | https://doi.org/10.1007/s10032-025-00533-x | **no code found** | n/a | offline | **line, by composition** (character generation → LLM-prompted layout) | **native** | **C** | Character-image metrics "comparable or even better … than diffusion model-based methods"; line samples used to train recognisers. Exact metric list `NOT RETRIEVED` | **`NOT RETRIEVED`** |
| **HandwritingAgent** | arXiv 2606.18788 (2026) | https://arxiv.org/abs/2606.18788 | **no code found** | n/a | offline (SVG stroke sequences) | character → composed line | **native** (via an LLM) | **C** | SSIM, ΔCER, FID, HWD (Chinese: CASIA-HWDB1.1, N=250) | 250 samples per condition: IAM (8 writers), CASIA-HWDB1.1 (Chinese), IAM-LINES, CROHME 2014, EDU-CHEMC, physics notes |

### 2.2 Character / word-level methods (composed only)

| Method | Venue / year | Paper URL (resolved) | Code URL | Licence | Modality | Granularity | Chinese | Mode | Metrics reported | Datasets / splits |
|---|---|---|---|---|---|---|---|---|---|---|
| **One-DM** | ECCV 2024 | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf | https://github.com/dailenson/One-DM | **MIT** | offline | word / arbitrary string | **per-character Chinese** verified (ICDAR-2013, FID 27.24) | **C** | FID, GS; user study (supplementary) | IAM 62,857 words / 500 writers → **339 / 161**; CVL 84,514 words → **283 / 27**; Chinese: ICDAR-2013; Japanese: UP_Kuchibue |
| **GANwriting** | ECCV 2020 | https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680273.pdf | https://github.com/omni-us/research-GANwriting | **MIT** | offline | word | **none** | **C** | FID (IV-S/IV-U/OOV-S/OOV-U); human eval (200 examiners) | IAM 62,857 words, 500 writers, **160 held-out test writers**; K = 15 style samples |
| **HWT** | **ICCV 2021**, pp. 1086–1094 | https://openaccess.thecvf.com/content/ICCV2021/html/Bhunia_Handwriting_Transformers_ICCV_2021_paper.html | https://github.com/ankanbhunia/Handwriting-Transformers | **MIT** | offline | word (arbitrary length) | **none** | **C** | FID, GS, human-based evaluation | IAM 9,862 lines / 62,857 words / 500 writers → **340 train / 160 test**; P = 15 style images; height 64 |
| **VATr / VATr++** | VATr CVPR 2023; VATr++ TPAMI 2024 | https://arxiv.org/abs/2303.15269 ; https://arxiv.org/abs/2402.10798 | https://github.com/aimagelab/VATr | **MIT** | offline | word | **none** as trained (Unifont archetypes are script-agnostic in principle) | **C** | FID, GS, CER (long-tail characters, digits) | IAM 62,857 words → **340 / 160**; P = 15; height 64; 79-character alphabet |
| **DiffusionPen** ("DiffPen") | ECCV 2024, pp. 417–434 | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11492.pdf (arXiv 2409.06065) | https://github.com/koninik/DiffusionPen | **MIT** | offline | word | **none** | **C** | CER, WER (HTR trained on synthetic, tested on IAM), FID, GS, style-variation tables | IAM; **k = 5** style images; MobileNetV2 style encoder; Canine-C text encoder |
| **MetaScript** | **arXiv 2312.16251 (2023)** — *not* AAAI | https://arxiv.org/abs/2312.16251 | https://github.com/xxyQwQ/metascript | **MIT** | offline | **character** + "Typewriter" composition | **native** | **C** | RA (recognition accuracy), IS, FID | CASIA-HWDB-1.1 ≈ 1 M images, 300 writers, 3,755 chars, **128×128**; test from CASIA-HWDB-1.0 (1,000 chars × 10 writers) |
| **SDT** | CVPR 2023, pp. 5977–5986 | https://openaccess.thecvf.com/content/CVPR2023/html/Dai_Disentangling_Writer_and_Character_Styles_for_Handwriting_Generation_CVPR_2023_paper.html | https://github.com/dailenson/SDT | **MIT** | **online** | **character** | **native** | **C** | DTW, Content Score, Style Score, user preference % | CASIA-OLHWDB 1.0–1.2 (≈3.7 M chars, 1,020 writers) → ICDAR-2013 test (60 writers) |
| **HiSAC** | **ECCV 2026**, LNCS pp. 104–122 | https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/3526.pdf (DOI 10.1007/978-3-032-37490-5_6) | https://github.com/IIP-Lab-XDU/HiSAC — **5 KB, no source** | **Apache-2.0** (API) | offline **and** online | **character** | **native** | **C (and NR as code)** | FID and content scores; exact metric set `NOT RETRIEVED` beyond the retrieved table fragments | `NOT RETRIEVED` |

**Reading the "Mode" column.** `N` rows are the only ones that can be compared to CENT on a
connectivity metric. `ND` (OLHWG) is a line model that *cannot* join characters — its own conclusion says
so. `C` rows must never sit in the same table block as `N` rows without the label.

---

## 3. Protocol of the two most relevant papers, quoted verbatim

### 3.1 DiffBrush (ICCV 2025) — datasets, splits, resolution

> "Evaluation dataset. To evaluate our DiffBrush in generating handwritten text-line, we use the widely
> adopted handwriting datasets IAM [36] and CVL [26]. IAM contains 13,353 English text-line images
> belonging to 657 unique writers. Following the protocol of CSA-GAN [23], we use text-lines from 496
> writers for training and the remaining 161 writers for testing. CVL dataset consists of handwritten
> text-lines from 310 writers in both English and German. For our experiments, we use the English
> portion, consisting of 11,007 text-lines, and follow the standard CVL split, with 283 writers for
> training and 27 for testing. In all experiments, we resize the images to a height of 64 pixels while
> preserving their aspect ratio, as done in previous works[7, 9, 23]. To manage varying widths, images
> with a width smaller than 1024 pixels are padded, whereas those exceeding 1024 are resized to a fixed
> size of 64 × 1024. We also evaluate DiffBrush on popular Chinese dataset CASIA HWDB (2.0–2.2) [32] (cf.
> Appendix E for more details.)"

Chinese split (Appendix E):

> "Dataset. CASIA-HWDB (2.0-2.2) consists of 52,230 Chinese text-lines belonging to 1,019 different
> writers. Following the standard split of CASIA-HWDB (2.0–2.2), we use text-lines from 816 writers for
> training and remaining 203 writers for testing. As mentioned in Appendix B, since Unifont [7, 44] can
> encode all Unicode characters, we still employ it to convert Chinese strings into textual content
> images. Similarly, in our experiments, all images are adjusted to 64 × 1024 pixels."

### 3.2 DiffBrush — metrics (verbatim, and the definition that matters)

> "Evaluation metrics. 1) We use the newly proposed Handwriting Distance (HWD) [45], specifically
> designed for handwriting style evaluation. HWD computes the Euclidean distance between features
> extracted by a VGG16 network pre-trained on a large corpus of handwritten text images. 2) For content
> evaluation, we follow latest studies [7, 40, 41] by using the generated training sets from each method
> to train an OCR system [48] and report its recognition performance on the real test set in terms of CER
> and WER. We name these new evaluation metrics DCER and DWER, with further discussions provided in
> Appendix C. 3) We use Frechet Inception Distance (FID) [15], Inception Score (IS) [51], and Geometry
> Score (GS) [24] to measure the visual quality of generated handwritten images. 4) We also conduct user
> studies to quantify the subjective quality of the generated handwritten text-line images in Appendix
> D."

Chinese metrics:

> "Similarly, we use HWD and FID [13, 15] to evaluate style imitation and visual quality of generated
> handwritten images, respectively. We still use the OCR system [48] to measure the content quality of
> generated results. Unlike English handwriting generation, we evaluate the OCR system's recognition
> performance on a real test set using two widely adopted metrics in Chinese text-line recognition [5,
> 54, 57]: Accuracy Rate (AR) and Correctness Rate (CR). Similarly, we name them DAR and DCR."

> **This is the single most important protocol fact in this survey.** `D_CER`/`D_WER`/`D_AR`/`D_CR` are
> **not** the recogniser's error on the generated images. They are the error of a recogniser **trained on
> generated data and evaluated on the real test set**. It measures *did the synthetic data teach a
> recogniser* — not *is the generated text readable*. OLHWG does the opposite (a fixed, real-trained
> recogniser applied to generated lines). The two are not interchangeable and must not share a column
> without a footnote.

### 3.3 DiffBrush — implementation details, baselines, ablations, user study, inference time

> "Implementation details. In all experiments, we use a randomly selected text-line sample as the style
> reference. In DiffBrush, both the style and content encoders are based on a ResNet18, followed by 2
> transformer encoder layers. Line discriminator uses three 3D convolution layers, and word
> discriminator has three 2D convolution layers. The model is trained for 800 epochs on eight RTX 4090
> GPUs using the AdamW optimizer with a learning rate of 10−4. We set masking ratio ρ to 0.5 after a grid
> search (cf. Table 5 in Appendix). We randomly drop the condition c with the probability of 0.1 for
> classifier-free training. During inference, we adopt a classifier-free guidance scale of 0.2 and use
> DDIM with 50 steps to accelerate the process."

**Baselines compared** (its Table 1, verbatim row labels): `TS-GAN` (one-shot), `CSA-GAN` (few-shot),
`VATr` (few-shot), `DiffusionPen` (few-shot), `One-DM` (one-shot), `Ours` (one-shot). For Chinese
(Fig. 11) the only comparison is **One-DM**. Note that **HWT is absent** from DiffBrush's table.

**Ablation dimensions** (§5.3 + Appendix F):

> "In this section, we conduct ablation studies to analyze our DiffBrush. We provide more analyses in
> Appendix, including generalization evaluation on various style backgrounds, failure case analysis,
> ablation results on masking ratio, enriching datasets to train recognizer, style interpolation results,
> style evaluation results in terms of WIER [12, 61], discussions about fine-grained style learning."

> "1) Comparing the content discriminators and CTC recognizer. 2) More visual ablation results of the
> style module and content discriminators, 3) More visual ablation results of vertical enhancing and
> horizontal enhancing heads, 4) More ablation results on discriminators architecture, and 5) The effect
> of masking ratio ρ."

**User study:** yes — Appendix D ("we also conduct user studies to quantify the subjective quality").
**Number of raters / question design: `NOT RETRIEVED`.**

**Inference time:** **not reported** in the retrieved text (`UNVERIFIED` — do not assert either way).
**Batch size:** **`NOT RETRIEVED`** (epochs, GPU count and optimiser are given; batch size is not in the
retrieved text).

### 3.4 DiffInk's protocol note that constrains how it may be reported

> "the input consists of a reference trajectory Xref and a concatenated text combining Tref and Tgen, and
> the goal is to generate a trajectory Xgen that renders Tgen while preserving the handwriting style in
> Xref."

Our own audit of the released code (`docs/BASELINE_AUDIT.md`, commit `65055545fbef`) found the
conditioning includes a clean 30 % prefix of the ground-truth trajectory, which cannot be switched off
(`prefix_ratio=0` still keeps one character). **Report it as prefix-conditioned line completion**, and
say so in the table footnote. Its own baseline table also documents how the field composes lines:

> "Baseline methods generate isolated characters and compose text lines through a shared layout
> prediction module from OLHWG."

### 3.5 OLHWG (ICLR 2025) — datasets, splits, metrics

> "For single character generation, following previous work, the CASIA-OLHWDB (1.0-1.2) is adopted as the
> training set, which contains about 3.7 million online Chinese handwritten characters produced by 1,020
> writers. The ICDAR-2013 competition database (Yin et al., 2013b) is adopted as the test set, which
> contains 60 writers, with each contributing the 3,755 most frequently used characters set of GB2312-80.
> For layout and text line generation, we adopt CASIA-OLHWDB (2.0-2.2) which consists of approximately
> 52,000 text lines written by 1,200 authors, totaling 1.3 million characters. We take 1,000 writers as
> the training set and the left 200 writers as the test set."

> "Readability Assessment. To assess the structural correctness performance, AR (Accurate Rate) and CR
> (Correct Rate) are two widely used metrics for Chinese string recognition (Su et al., 2009; Wang et
> al., 2011). We adopt the methods proposed in (Chen et al., 2023) as the text line recognizer, which can
> get 0.962 AR and 0.958 CR on the test set. We report the recognition metrics of the recognizer on our
> synthesized text lines, where higher values indicate more complete and accurate geometric structure."

**Direction of the recogniser here is the opposite of DiffBrush's**: fixed recogniser trained on *real*
data (0.962 AR / 0.958 CR on the real test set), then applied to *generated* lines.

**Layout metrics** are eight binary geometric feature deltas between adjacent characters (∇1–∇8,
"calculated between every two adjacent characters"), e.g. "Mean vertical distance between geometric
centers", "Mean distance between upper bounds".

**Baselines:** only two, both ablations of its own layout planner — `Gaussian` (per-class bounding-box
Gaussians sampled from the training set) and `Unconditional` (its layout generator with a zero-length
prefix). For characters it compares DeepImitator, FontRNN, WriteLikeU (style-transfer), SDT.

**User study:** yes — "Over 85% of the responses are 'yes,' indicating that the distinction between real
and synthesized samples is difficult to perceive." (Rater count and question wording: `NOT RETRIEVED`.)
**Reference length:** "The default reference text line length is set as 10."

### 3.6 OLHWG — implementation details, inference time, and its own connectivity admission

> "We implement our model in Pytorch and run experiments on NVIDIA TITAN RTX 24G GPUs. Both training and
> testing are completed on a single GPU. For training the layout planner, the optimizer is Adam with an
> initial learning rate of 0.01 and the batch size is 32. For training the diffusion character
> synthesizer, the initial learning rate is 0.001, the gradient clipping is 1.0, learning rate decay for
> each batch is 0.9998. We train the whole model with 400K iterations with the batch size of 64, which
> takes about 4 days."

> "For the hyperparameters of DDPM, we adopt the linear noising schedule proposed by (Ho et al., 2021):
> {T = 1000; α0 = 1 −1e−4; αT = 1 −2e−2}."

**Image resolution for lines: `NOT RETRIEVED`.** The appendix normalises trajectory height to 1
("we maintain the aspect ratio and normalize the height of each sample to 1") — that is trajectory
normalisation, not an image raster size.

**Inference time: not reported by OLHWG.** The only throughput figure in the literature is *DiffInk's*
measurement of OLHWG: **0.07 chars/s** (DiffInk Table 1) — a third-party number, not OLHWG's claim.

**Its own admission of the connectivity failure — quotable, and it is our motivation:**

> "However, due to the fact that each character is generated independently, our approach encounters
> difficulties in mimicking styles with extensive cursive connections between characters. An end-to-end
> text line generation method might be required to address this issue, as discussed in Appendix A.3."

---

## 4. What each metric actually requires

Definitions are as used by the papers retrieved above; the parenthetical is the resource you must have
installed and frozen to report the number.

| Metric | Requires | Notes / traps |
|---|---|---|
| **FID** | **Inception-v3 pretrained on ImageNet** (feature extractor; distribution distance). | DiffBrush, One-DM, GANwriting, HWT, VATr, DiffusionPen, Emuru, InkSpire all use it. Emuru notes FID is *insensitive to word alignment* — "models that generate single words and stitch them together" can score well, which is exactly our composed-baseline problem. Emuru adds **BFID** (FID on binarised images) to discount colour/texture. |
| **KID** | **Inception features** (MMD instead of Fréchet). | Reported by Emuru and InkSpire. InkSpire multiplies by 10³. |
| **HWD** | **VGG16 pretrained on handwriting** (a style-feature backbone), images resized to **height 32**; Euclidean distance between the *averaged per-writer feature vectors* of real and generated sets. | Original paper: *HWD: A Novel Evaluation Score for Styled Handwritten Text Generation*, **BMVC 2023** (Pippi, Quattrini, Cascianelli, Cucchiara). Reference code: https://github.com/aimagelab/HWD. **Licence: `NOASSERTION` — terms not stated; check before vendoring.** DiffBrush describes it as "features extracted by a VGG16 network pre-trained on a large corpus of handwritten text images". |
| **IS** | **Inception-v3 classifier** (ImageNet). | Reported by DiffBrush and MetaScript (MetaScript uses an EfficientNetV2 classifier trained on CASIA-HWDB-1.1 instead — *not* comparable to an Inception IS). |
| **GS** (Geometry Score) | **No pretrained network** — topological statistic of the generated set. | Reported by DiffBrush, One-DM, GANwriting, HWT, VATr, TS-GAN. Cheap, but not comparable across different image sizes. |
| **AR / CR** | A **frozen Chinese text-line recogniser**. AR = (N − D − S − I)/N, CR = (N − D − S)/N via DP string matching (the ICDAR-2013 / Yin et al. convention). | OLHWG uses Chen et al. (2023), "0.962 AR and 0.958 CR on the test set". DiffInk uses "a pretrained text-line OCR model … following the (Yin et al., 2013) OLHCTR benchmark". InkSpire uses the same convention. **Our repo has no frozen HTR yet** — `docs/BASELINE_AUDIT.md` names `Teklia/pylaia-casia-hwdb2` as the candidate; until it is wired, AR/CR must be `NOT_RUN`. |
| **D_CER / D_WER / D_AR / D_CR** | An **OCR system that is trained on the generated data** and then evaluated on the **real** test set. | DiffBrush's definition (quoted in §3.2). Requires a *trainable* recogniser per baseline, i.e. one full training run **per comparison row**. Expensive. Named differently from plain CER/WER for a reason. |
| **CER / WER (plain)** | A **fixed HTR/OCR applied to the generated images** (e.g. CNN-LSTM+CTC in DiffusionPen; the IAM HTR in CSA-GAN/VATr). | The opposite direction from `D_CER`. DiffusionPen reports both directions (Table 2 = synthetic-trained HTR on real IAM; Table 3 = synthetic added to real). |
| **ΔCER** | **TrOCR-Base** (Emuru): ΔCER = CER on reference images − CER on generated images. | Needs TrOCR-Base weights. Interpretation: "as readable as the reference, but not more" — guards against style collapse to a hyper-readable style. |
| **Style Score** | A **writer classifier trained on real lines** of the target script. | InkSpire uses Swin-T (~86.8 M params) trained per language, 94 % on a 60-way Chinese val set and 91 % on a 161-way English set. SDT/OLHWG use their own style-score networks. **Not comparable across papers.** |
| **DTW** | No network — normalised dynamic time warping between generated and real trajectories. **Online only.** | DiffInk normalised DTW; SDT DTW. Meaningless for offline images. |
| **Avg. char/s** | Wall-clock on a stated GPU. | Only DiffInk reports it. If we report speed we must state the GPU and batch size, or it is not a comparison. |

---

## 5. Protocol we should copy

A hybrid, with the conflicts resolved explicitly — that resolution is itself a contribution.

### 5.1 The split conflict must be resolved in the text

The literature uses **three incompatible Chinese-line protocols**:

| Protocol | Train | Test | Used by |
|---|---|---|---|
| **A** | CASIA-HWDB2.0–2.2, **816 writers** | same corpus, **203 writers** | DiffBrush (Appendix E) |
| **B** | CASIA-HWDB2.0–2.2, **1,000 writers** | same corpus, **200 writers** | OLHWG (online analogue: OLHWDB2.0–2.2, 1,000 / 200) |
| **C** | CASIA-HWDB2.0–2.2 (all 1,019 writers) | **ICDAR-2013, 3,432 lines / 60 writers** | InkSpire |

**Recommendation: adopt A** (816 / 203, offline) as the headline split, because (i) DiffBrush is the
primary baseline and the only one whose Chinese numbers we can cite against a matching split; (ii) it is
a *writer-disjoint* split inside one corpus, which is required for a style-generalisation claim; (iii)
ICDAR-2013 (protocol C) can then be used as a **second, harder test set** — a held-out *corpus*, which
is the stronger generalisation claim and costs nothing extra once the recogniser exists.

State all three in a footnote and say which rows are cross-protocol. Do not silently compare a
protocol-C number to a protocol-A number; that is the single easiest way for a reviewer to reject the
table.

### 5.2 Fixed protocol to freeze

- **Image resolution: height 64, width padded/resized to 1024 → 64 × 1024.** Matches DiffBrush exactly
  ("64 × 1024") and matches InkSpire's normalisation ("proportional resizing to height 64 followed by
  width-wise cropping or white padding"). Adopting anything else makes every FID/KID/HWD incomparable.
- **Style references: k lines from the target writer, never characters.** DiffBrush: "a randomly
  selected text-line sample as the style reference" (one-shot). OLHWG: "The default reference text line
  length is set as 10" for layout in-context prompting. CENT is a k-line method — report k = 1 and k = 5,
  and say which rows are one-shot vs few-shot, since the field mixes them (DiffBrush's own table has a
  **Shot** column: one/few).
- **Metrics, in two families, never collapsed:**
  - *style/realism*: **FID, KID, HWD** (+ BFID if we want to discount ink colour and paper texture).
  - *content*: **AR / CR** with a **frozen, real-trained** recogniser — and separately, if budget allows,
    **D_AR / D_CR** with a recogniser **trained on generated data**. The two must be labelled.
- **The connectivity metrics are ours and must be reported beside AR/CR, never instead of them**
  (threat T4 in `phase2_literature/text_line_synthesis.md`). No retrieved paper reports a connectivity or
  pen-lift number — that absence is now *verified* across 20+ methods, which strengthens the claim. Say
  it that way: "none of the methods we surveyed reports a connectivity metric", not "nobody has".
- **User study:** run one, because both anchor papers do (DiffBrush App. D; OLHWG reports ">85 %" of
  responses answering "yes"). For the wording, copy **TS-GAN's** question, which is the one quoted in
  the retrieved text: "participants viewed a single image at a time and were asked if the image was
  written by a human or a computer." Report the accuracy **plus the deliberately-bad control condition**,
  as TS-GAN does (52.2 % overall vs 89.5 % on poorly-generated images) — that control is what makes the
  number credible. OLHWG's exact question wording and rater count: `NOT RETRIEVED`.
- **Inference time:** report it. Neither anchor paper does (DiffBrush: not in the retrieved text; OLHWG:
  not reported), and DiffInk's `Avg. char/s` is the only throughput number in the field — which means a
  throughput column is currently **open ground** and cheap to own. State GPU, batch size, and whether the
  figure includes the recogniser.

### 5.3 Ablation dimensions to mirror

Copy DiffBrush's five, since they are the ones a reviewer of that paper will expect:

1. content supervision: discriminators **vs** a CTC recogniser loss (DiffBrush Table 3: 1.41 HWD with
   discriminators vs 1.69 with CTC at the same λ) — for us: connectivity loss vs an AR/CR-style proxy;
2. masking / condition-drop ratio (DiffBrush grid-searched ρ = 0.5);
3. classifier-free guidance scale (DiffBrush used 0.2) and DDIM steps (50);
4. style-reference count k (MetaScript's Table 1 is the model for this: k ∈ {1,2,4,8});
5. reference *length* (OLHWG Fig. 6 varies the prefix length and shows horizontal layout features improve
   with longer references) — directly relevant to CENT's k-line style claim.

### 5.4 What we should *not* copy

- **Eight-page-table padding with `IS` and `GS`.** DiffBrush reports them, but they are weakly motivated
  for handwriting and one of them (IS) is computed with a classifier trained on a different corpus in at
  least one paper (MetaScript). Keep them out, or put them in the supplement.
- **Reporting a composed baseline as if it were native.** Emuru explicitly warns that stitched words can
  score well on FID while having broken alignment — the exact failure our connectivity metric is designed
  to expose. Use that as a sentence of support, not as a silent comparison.

---

## 6. Unverified / not-retrieved register

Everything below is a gap, not a finding. Do not let these become numbers in the manuscript.

| Item | Status |
|---|---|
| TGC-Diff's own reported metrics, dataset splits, resolution, implementation details, user study, inference time | **`NOT RETRIEVED`** (Springer paywall; only abstract + references extractable). Third-party numbers from InkSpire are labelled as such. |
| HCT-GAN metrics, dataset, resolution, whether code exists | **`NOT RETRIEVED`** / no code found. |
| IJDAR multi-level style control: exact metrics, splits, implementation | **`NOT RETRIEVED`** (abstract + bibliographic record only). |
| HiSAC: exact metric set, splits | **`NOT RETRIEVED`** beyond retrieved table fragments. |
| OLHWG: line image resolution; inference time; user-study size | **`NOT RETRIEVED`** / not reported. |
| DiffBrush: batch size; inference time; user-study protocol and rater count | **`NOT RETRIEVED`**; inference time not found in the retrieved text. |
| DiffInk: whether the pretrained weights actually contain the 3,755-class vocabulary | **`UNVERIFIED`** — README announces "Code and pretrained weights are released" (2026/3/21) and links Drive/Baidu folders; nothing was downloaded. |
| One-DM: whether a Chinese checkpoint was ever released | **`UNVERIFIED`** — README (2024/09/07) says "Later versions supporting Chinese and Japanese will be released soon". |
| CSA-GAN: official code | **No code found.** A GitHub repo named `CSA-GAN` exists but is an unrelated text-to-image method. |
| Emuru / DiffusionPen / HWT / VATr repo weights availability | Repo/README claim HF or Releases; **not verified by download**. |
| HWD evaluation kit licence | **`NOASSERTION`** (GitHub API). Terms must be read before vendoring the backbone. |
| TS-GAN licence | **CC BY-NC-SA 4.0** — read from the LICENSE file. Non-commercial; fine for research, not for a commercial release. |
| Any 2025–2026 Chinese-line method beyond DiffBrush / TGC-Diff / InkSpire / DiffInk / HCT-GAN / IJDAR / HandwritingAgent | **None found** in the retrieved literature. This is an absence of evidence, not proof of absence. |

---

## 7. Sources retrieved

Papers: ICCV 2025 OA (DiffBrush paper + supplemental), arXiv 2508.03256, ICLR 2025 proceedings (OLHWG),
arXiv 2410.02309, ICLR 2026 proceedings (DiffInk, InkSpire), ECCV 2024 ECVA (One-DM, DiffusionPen),
ECCV 2020 ECVA (GANwriting), BMVC 2020 archive (TS-GAN paper + supplement), ICCV 2021 OA (HWT), CVPR 2023
OA (SDT), CVPR 2025 OA (Emuru paper + supplement), TPAMI/arXiv 2204.05539 (CSA-GAN), arXiv 2312.16251
(MetaScript), arXiv 2303.15269 + 2402.10798 (VATr / VATr++), BMVC 2023 (HWD), ECCV 2026 PDF (HiSAC),
Springer chapter (TGC-Diff, HiSAC), IEEE/DOI records (HCT-GAN, IJDAR), arXiv 2606.18788 (HandwritingAgent).

Code and licences: GitHub REST API for `dailenson/DiffBrush`, `newbie2niubility/TGC-Diff`, `awei669/DiffInk`,
`singularityrms/OLHWG`, `dailenson/One-DM`, `koninik/DiffusionPen`, `ankanbhunia/Handwriting-Transformers`,
`xxyQwQ/metascript`, `aimagelab/Emuru-autoregressive-text-img`, `IIP-Lab-XDU/HiSAC`,
`newbie2niubility/InkSpire`, `dailenson/SDT`, `aimagelab/VATr`, `omni-us/research-GANwriting`,
`herobd/handwriting_line_generation`, `aimagelab/HWD`; plus the `LICENSE` files inside the sixteen clones
under `A:\Papers\CENT\baselines\` and their `git` remotes.
