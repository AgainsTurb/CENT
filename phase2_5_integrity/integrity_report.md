# Integrity Verification Report — CENT

**Role:** integrity_verification_agent (Mode 1 — Stage 2.5, pre-review)
**Manuscript:** `paper/main.tex` (IEEEtran conference, 6 pp.), `paper/refs.bib` (42 entries)
**Date:** 2026-11-21
**Registry/retrieval sources used:** Crossref `/works/{DOI}` (live DOI records, backoff on HTTP 429), arXiv export API `id_list=` (live records), Exa/Tavily full-text retrieval of publisher/proceedings pages, local file inspection.
**Independence note:** every reference was re-resolved *from source in this session*. I did **not** accept `refs/verified_refs.json` as evidence; where it disagreed with the live record, the live record wins and the disagreement is reported (§F).

---

## 0. Verdict

| Verdict | Rule applied | Result |
|---|---|---|
| **FAIL** | PASS = 0 SERIOUS + 0 MEDIUM · PASS WITH NOTES = 0 SERIOUS/MEDIUM but ≥1 MINOR/note · **FAIL = any SERIOUS or MEDIUM** | **3 MEDIUM, 0 SERIOUS, 13 MINOR** |

> **REVISED VERDICT after Correction round 1 (§9): PASS WITH NOTES — 0 SERIOUS, 0 MEDIUM, 6 MINOR open + 3 nits.**
> All three MEDIUMs and 8 of 13 MINORs were re-verified as fixed in the current files. The remaining items are
> attached in §9. The only fix I regard as must-do-before-Stage-4.5 is the derived `refs/verification_report.md`
> (it now prints the title-squatting DOI as `fid`'s identifier; see §9.2 m11) — that is a defect in an artefact,
> not in the manuscript, which is why it does not hold the gate.

**First, the thing that matters most: the manuscript's central honesty claim holds.**

The paper claims (abstract, §I.C, §IV, §VI) that it reports a method, a feasibility analysis the authors computed themselves, and a frozen protocol — **no comparative experimental results, and no number without attribution or reproduction.** I checked this as a claim, not as a formality:

- There is **no fabricated or borrowed experimental result anywhere in the manuscript.** No accuracy, FID, CER, DTW, SSIM, or baseline number appears. No table reports an outcome.
- Every quantitative assertion traces either to `analysis/lexicon_stats.py` (reproducible; Table I matches `lexicon_stats.json` **exactly**, all 18 cells) or to a cited work I re-read.
- The 42-entry bibliography contains **zero fabrications**: 42/42 VERIFIED, 0 NOT_FOUND, 0 MISMATCH. The `fid` entry correctly cites arXiv:1706.08500 and carries **no DOI field**; the title-squatting DOI `10.18034/ajase.v8i1.9` appears in the file **only inside the evidence comment that documents its rejection**, and I confirmed that DOI is indeed a squatter: same title, but *Asian Journal of Applied Science and Engineering*, 2019, single author "Naresh Babu Bynagari" — not Heusel et al.. That rejection was correct and is the strongest evidence in the whole run that the reference pipeline was actually doing work.

The FAIL is therefore **not** for fabrication or for fabricated results. It is for three MEDIUM defects, all in internal consistency / bibliographic completeness, all fixable by editing text without new verification:

1. **M1** — `dp_font` prints 5 authors; the paper has 6 (Jianguo Sun omitted). Protocol A2: omitted co-author = MEDIUM.
2. **M2** — §V.B's "860/60/100 … respecting the released partition where one exists" is contradicted by the CASIA release's 4:1 writer partition (816/204 of 1,020).
3. **M3** — §III.B says corpus-disagreement exclusions "are excluded … and counted in the reported coverage statistic"; they are not counted, and cannot be (no corpus was obtained), and Table I's "covered" column measures lexicon presence only.

---

## 1. Phase A — Reference verification (all 42 entries)

Codes for "how resolved": **CR** = live Crossref DOI record; **AX** = live arXiv export-API record (title + full author list + date); **EX** = publisher/proceedings page retrieved (Exa/Tavily); **LOC** = local artefact inspected.

| # | key | Status | Resolved via | What was searched / confirmed | Defect |
|---|---|---|---|---|---|
| 1 | `diffusionpen` | **VERIFIED** | CR | `10.1007/978-3-031-73013-9_24` → *DiffusionPen: Towards Controlling the Style of Handwritten Text Generation*, Nikolaidou/Retsinas/Sfikas/Liwicki, LNCS — *Computer Vision – ECCV 2024*, 2024 | venue imprecise in bib (m8) |
| 2 | `one_dm` | **VERIFIED** | CR | `10.1007/978-3-031-73636-0_24` → *One-DM*, Dai/Zhang/Ke/Guo/Huang, LNCS — ECCV 2024, 2024 | venue imprecise (m8) |
| 3 | `vatr` | **VERIFIED** | CR | `10.1109/cvpr52729.2023.02151` → *Handwritten Text Generation from Visual Archetypes*, Pippi/Cascianelli/Cucchiara, CVPR 2023 | — |
| 4 | `vatr_pp` | **VERIFIED** | AX | `2402.10798v1` → *VATr++…*, Vanherle/Pippi/Cascianelli/Michiels/Van Reeth/Cucchiara, 2024 | — |
| 5 | `hwt` | **VERIFIED** | AX | `2104.03964v1` → *Handwriting Transformers*, Bhunia et al. (6 authors), 2021 | — |
| 6 | `ganwriting` | **VERIFIED** | AX | `2003.02567v2` → *GANwriting…*, Kang/Riba/Wang/Rusiñol/Fornés/Villegas, 2020 | — |
| 7 | `wordstylist` | **VERIFIED** | AX | `2303.16576v2` → *WordStylist…*, Nikolaidou et al. (8 authors), 2023 | — |
| 8 | `metascript` | **VERIFIED** | AX | `2312.16251v1` → *MetaScript…*, Xue/Wang/Bu/Li/Zhang, 2023 | — |
| 9 | `layout_glyph` | **VERIFIED** | AX + EX | `2410.02309v3` → *Decoupling Layout from Glyph…*, Ren/Zhang/Chen, 2024; ICLR 2025 proceedings page confirms | — |
| 10 | `sdt` | **VERIFIED** | AX | `2303.14736v2` → *Disentangling Writer and Character Styles…*, Dai et al. (7), 2023 | — |
| 11 | `rcgan` | **VERIFIED** | CR | `10.31577/cai_2024_2_482` → *Radical Constraint-Based GAN…*, Ye/Zhang/Yang/Du, *Computing and Informatics*, 2024 | — |
| 12 | `casia` | **VERIFIED** | CR + EX | `10.1109/icdar.2011.17` → *CASIA Online and Offline Chinese Handwriting Databases*, Liu/Yin/Wang/Wang, ICDAR 2011; **NLPR PDF retrieved** and both quoted phrases checked verbatim (Phase B) | — |
| 13 | `icdar2013zh` | **VERIFIED** | CR | `10.1109/icdar.2013.218` → *ICDAR 2013 Chinese Handwriting Recognition Competition*, Yin/Wang/Zhang/Liu, ICDAR 2013 | uncited where the 60-writer fact is used (m4) |
| 14 | `ddpm` | **VERIFIED** | AX | `2006.11239v2` → Ho/Jain/Abbeel, 2020 | **never cited** (m1) |
| 15 | `ldm` | **VERIFIED** | AX | `2112.10752v2` → Rombach et al., 2021 | **never cited** (m1) |
| 16 | `dit` | **VERIFIED** | CR | `10.1109/iccv51070.2023.00387` → Peebles/Xie, ICCV 2023 | **never cited** (m1) |
| 17 | `graves2013` | **VERIFIED** | AX | `1308.0850v5` → Graves, 2013 | — |
| 18 | `calligan` | **VERIFIED** | AX | `2005.12500v1` → Wu/Yang/Hsu, 2020 | **never cited** (m1) |
| 19 | `cffont` | **VERIFIED** | CR | `10.1109/cvpr52729.2023.00185` → Wang/Zhou/Ge/Jiang/Bao/Xu, CVPR 2023 | — |
| 20 | `fontdiffuser` | **VERIFIED** | AX | `2312.12142v1` → Yang et al. (6), 2023 (AAAI 2024 in print; preprint citation legitimate) | — |
| 21 | `emuru` | **VERIFIED** | AX | `2503.17074v2` → Pippi/Quattrini/Cascianelli/Tonioni/Cucchiara, 2025 | — |
| 22 | `scriptvit` | **VERIFIED** | AX | `2511.18307v1` → Acharya/Baskota, 2025 | — |
| 23 | `crnn` | **VERIFIED** | AX | `1507.05717v1` → Shi/Bai/Yao, 2015 | **never cited** (m1) |
| 24 | `cfg` | **VERIFIED** | AX | `2207.12598v1` → Ho/Salimans, 2022 | **never cited** (m1) |
| 25 | `diffvg` | **VERIFIED** | CR | `10.1145/3414685.3417871` → Li/Lukáč/Gharbi/Ragan-Kelley, *ACM TOG*, 2020 | — |
| 26 | `softras` | **VERIFIED** | AX | `1904.01786v1` → Liu/Li/Chen/Li, 2019 | — |
| 27 | `lffont` | **VERIFIED** | CR | `10.1609/aaai.v35i3.16340` → Park/Chun/Cha/Lee/Shim, AAAI 2021 | — |
| 28 | `dgfont` | **VERIFIED** | AX | `2104.03064v2` → Xie/Chen/Sun/Lu, 2021 | — |
| 29 | `inception` | **VERIFIED** | CR | `10.1109/cvpr.2016.308` → Szegedy/Vanhoucke/Ioffe/Shlens/Wojna, CVPR 2016 | — |
| 30 | `hi_gan` | **VERIFIED** | CR | `10.1609/aaai.v35i9.16917` → Gan/Wang, AAAI 2021 | — |
| 31 | `fid` | **VERIFIED** | AX | `1706.08500v6` → **Heusel/Ramsauer/Unterthiner/Nessler/Hochreiter**, 2017; NeurIPS 2017 (in bib as `booktitle`), `note = arXiv:1706.08500`, **no `doi` field**. Squatter `10.18034/ajase.v8i1.9` independently confirmed (Crossref): same title, *Asian J. Applied Science and Engineering* 2019, author N. B. Bynagari. **Correctly rejected; the bib does not cite it.** | — |
| 32 | `hisac` | **VERIFIED** | CR + EX | `10.1007/978-3-032-37490-5_6` → *Hierarchical Style Aggregation for Versatile Chinese Handwriting Generation*, Wang/Gao/Wang, LNCS — *Computer Vision – ECCV 2026*, 2026; ECCV 2026 poster + project repo corroborate | venue imprecise (m8) |
| 33 | `diffwriter` | **VERIFIED** | CR + EX | `10.1007/978-981-99-8141-0_7` → *Diff-Writer…*, Ren/Zhang/Wang/Yin/Liu, CCIS 1964 (ICONIP 2023), 2023 | venue imprecise (m8) |
| 34 | `difffont` | **VERIFIED** | AX | `2212.05895v3` → *Diff-Font: Diffusion Model for Robust One-Shot Font Generation*, He et al. (7), 2022 | — |
| 35 | `dsd` | **VERIFIED** | AX | `2008.11354v2` → *Generating Handwriting via Decoupled Style Descriptors*, Kotani/Tellex/Tompkin, 2020 | no store record (m11) |
| 36 | `boundink` | **VERIFIED** | AX | `2604.02103v3` → *BoundInk: Boundary-Aware Online Handwriting Generation*, Shin/Hong/Bak, 2026 | no store record (m11) |
| 37 | `inkspire` | **VERIFIED** | EX | ICLR 2026 proceedings (`proceedings.iclr.cc/…/963cceb…-Abstract-Conference.html`), ICLR virtual poster 10010623, ML Anthology: **Honglie Wang, Yan-Ming Zhang, Wangzi Yao, Fei Yin, Cheng-Lin Liu**, 2026 | store says UNMATCHED (m11) |
| 38 | `dp_font` | **VERIFIED with defect** | EX | IJCAI 2024 proceedings 863 + **BibTeX** `ijcai.org/proceedings/2024/bibtex/863`: *DP-Font…*, doi 10.24963/ijcai.2024/863, pages 7796–7804, **authors: Zhang, Zhu, Benarab, Ma, Dong, *Sun*** | **M1: 6th author Jianguo Sun omitted** |
| 39 | `scfont` | **VERIFIED** | CR + EX | `10.1609/aaai.v33i01.33014015` → Jiang/Lian/Tang/Xiao, AAAI 2019 (pp. 4015–4022); AAAI OJS page + abstract retrieved | — |
| 40 | `stroke2font` | **VERIFIED** | EX | MDPI `mdpi.com/1999-4893/19/3/231` → *Stroke2Font…*, Li/Bian/Chai, *Algorithms* 19(3):231, 2026-03-18; abstract confirms "stroke element decomposition and Bézier curve parameterization … decouples structural representation from style rendering" | no volume/pages/DOI in bib (m10) |
| 41 | `hanzi` | **VERIFIED** | EX + LOC | `github.com/skishore/makemeahanzi` COPYING + project page: **graphics.txt is derived from Arphic PL KaitiM GB and Arphic PL UKai and is redistributable under the Arphic Public License**; dictionary.txt is separate LGPL-3.0+. Local `data/lexicon/graphics.txt` = **30,778,076 bytes, 9,574 lines** — exactly the figures in the bib's evidence comment and `lexicon_stats.json` | no store record (m11); year "2024" not verified |
| 42 | `lvgh` | **VERIFIED with defect** | AX + EX | `2511.11119v1` → *Stroke Modeling Enables Vectorized Character Generation with Large Vectorized Glyph Model*, **Xinyue Zhang, Haolong Li, Jiawei Ma, Chen Ye**, 2025-11-14 | **m9: all 4 authors omitted; recorded reason is false** |

### A2 bibliographic accuracy — item-by-item

Author lists were compared field-by-field against the retrieved record for **all 42**. Result: **40/42 exact, 1 incomplete by one author (`dp_font`), 1 with no authors at all (`lvgh`)**. Title, year, and venue were compared for all 42: **no wrong year, no wrong venue, no wrong DOI**. Venue *imprecision* (series name only, conference dropped) affects 4 entries (m8).

**On the two items the dispatch asked me to confirm specifically:**
- `fid` — confirmed: arXiv:1706.08500 is genuinely Heusel et al., NeurIPS 2017 (5 authors), and the bib entry carries no DOI. The squatter string is present in the file only in the rejection comment (a naive `grep` will match it — read the field, not the string).
- `lvgh` — the bib does **not** print "Anonymous". `make_bib.py` sets `"author": "Anonymous"` but also `omit_authors: True`, so the emitted entry has **no author field at all**. The recorded justification ("Author list not machine-verified, so authors are omitted rather than guessed") is *honest in intent but factually wrong now*: the arXiv export API returns the four authors, as does the abs page. Omission avoided a guess; it did not require one.

### A3 ghost-citation check

- Dangling citations (cited, absent from bib): **0**.
- Orphan entries (in bib, never cited): **6** — `ddpm`, `ldm`, `dit`, `calligan`, `crnn`, `cfg`. Confirmed twice: by parsing all `\cite{}` keys (36 distinct) and by text-searching the compiled `paper/CENT-draft.pdf`, where none of the six titles appears (BibTeX emits only cited keys, so the printed list has **36** entries, matching). → MINOR m1.

---

## 2. Phase B — Citation context

**Coverage achieved: 19 of 36 cited keys in depth (53%) — above the 30% floor — and 100% of the citations that carry a specific factual or quantitative claim.** Results:

| Citation | Manuscript claim | Source evidence | Verdict |
|---|---|---|---|
| §II.A `diffusionpen` | "hybrid metric-learning plus classification style encoder conditions **better than either alone**" | DiffPen-class 68.04 / DiffPen-triplet 67.50 / DiffPen 70.31 writer-classification acc.; CER 7.12 / 7.13 / 6.94 — **but FID 22.23 / 22.06 / 22.54** and the paper itself writes "This is not the case for the other metrics, but the differences are nonsignificant" | **MINOR_DISTORTION (m5)** — true for style classification and CER, false for FID; an unqualified "better" is not what the source says |
| §II.C `diffwriter` | "argues sequential decoding struggles with global structure" | Abstract, verbatim: "Most of the previous methods are based on autoregressive models… However, this often makes it difficult to capture the global structure of the handwriting data" | **VERIFIED** (exact) |
| §II.C `sdt` | "the two representations carry information at different frequency scales" | Abstract, verbatim: "the two learned style representations provide information at **different frequency magnitudes**"; §4.2/Fig. 7: writer-wise = low-frequency, character-wise = high-frequency | **VERIFIED** (exact) |
| §II.D `dp_font` | "encodes stroke order as a **36-dimensional** conditioning vector" | "each character is encoded into a 36-dimensional vector (the maximum stroke count in commonly used Chinese characters is 36)" | **VERIFIED** (exact) |
| §II.D `dp_font` | "removing it **produces** characters with stroke errors" | "when generating characters with difficult structures, the DP-Font without explicit encoding of stroke order and quantity **may** generate characters with stroke errors" (see Fig. 6 ablation: none / one-bit / count / order) | **MINOR_DISTORTION (m6)** — hedge "may", and the qualifier "characters with difficult structures", dropped |
| §II.B `one_dm` | "reports a Chinese setting on the ICDAR-2013 competition data" | One-DM ECCV-2024 **supplementary §C.4**: "For the Chinese handwritten character generation task, we use the ICDAR-2013 competition database [31], which contains **60 writers and 3755 different Chinese characters**" | **VERIFIED** |
| §II.B `rcgan` | "adding a radical-consistency loss that reduces stroke error" | RC-GAN: radical constraint module "to constrain the radical sequence … to avoid stroke errors"; human-counted stroke errors 42 vs 62 and 70 vs 220 vs NICE-GAN | **VERIFIED** |
| §II.B `inkspire` | "unifies style, content and noise in one diffusion-transformer latent and trains bilingually" | ICLR 2026 PDF: "a diffusion transformer based model that unifies style, content, and noise within a shared latent space"; trained on IAM **and** CASIA-HWDB2.0–2.2 + ICDAR2013 | **VERIFIED** |
| §II.D `scfont` | "transfers writing trajectories with separated strokes from a reference font" | AAAI abstract: "learn how to transfer the writing trajectories with separated strokes in the reference font style into those in the target style" | **VERIFIED** (near-verbatim) |
| §II.D `stroke2font` | "stroke element graph with Bézier parameterisation, decouples structure from style" | MDPI abstract: "decouples structural representation from style rendering through stroke element decomposition and Bézier curve parameterization" | **VERIFIED** |
| §II.A `vatr` | "replaced one-hot character identifiers with rendered visual archetypes, showing geometric similarity transfers to rare ones" | arXiv 2303.15269: "Unlike existing approaches … independent one-hot-encoded characters … geometric similarities among the archetypes are reflected in styled characters. These similarities can be exploited for generating long-tail characters" | **VERIFIED** |
| §II.A `vatr_pp` | "regularised the inputs" | VATr++: "we propose generally applicable strategies for input preparation and training regularization" | **VERIFIED** |
| §II.A `dsd` | "factorised style explicitly" | DSD: decouples writer and character style into separate vectors (C-matrix, writer-DSD w) | **VERIFIED** |
| §II.C `layout_glyph` | "extends this to lines" | ICLR 2025: layout/glyph decoupling for text-line generation | **VERIFIED** |
| §II.C/§III.C `hisac` | lines + "hierarchical (radical, component, stroke) style aggregation"; "finds stroke-level correspondence informative" | ECCV 2026: three-level decomposition (radicals/components/strokes) + HiSA cross-scale matching incl. stroke-level cosine similarity | **VERIFIED** |
| §VI `hisac` | "the Chinese online literature finds **cursive connections hardest to imitate**" | Conclusion: "learning efficient representations for characters with extremely complex structures or severe stylistic deformations (e.g., running script or cursive handwriting) remains a challenging direction for future work" | **MINOR_DISTORTION (m7)** — "remains challenging" is not the comparative "hardest" |
| §V.A `casia` | quoted: "using Anoto pen on papers for obtaining both online trajectory data and offline images" | **Verbatim substring** of the CASIA abstract/NLPR PDF | **VERIFIED** (exact quote) |
| §V.A `casia` | quoted: "the same partitioning of writers … between online and offline datasets" | **Verbatim**: "The same partitioning of writers is taken between online and offline datasets and between isolated characters and handwritten texts datasets." | **VERIFIED** (exact quote, ellipsis honest) |
| §V.A `casia` | "produced by 1,020 writers"; corpus names OLHWDB1.0–1.2 / HWDB1.0–1.2 | Paper: 1,220 writers contributed overall; **the released DB1.0–1.2 contain 1,020 writers**; released names OLHWDB1.0–1.2 / HWDB1.0–1.2 | **VERIFIED** |
| §I `fid`,`inception` | FID "computed on Inception features trained on natural photographs" | Correct description of FID/Inception-v3 | **VERIFIED** |
| §IV `hanzi` | "the public stroke-order lexicon (9,574 characters; Arphic Public License)" | Local file 9,574 records; COPYING confirms Arphic PL | **VERIFIED** |
| §II.D `lvgh` | "treats a glyph as an ordered sequence of Bézier strokes" | Abstract: "special sequences composed of ordered strokes"; SVGs; "predicting the next stroke" with an LLM | **VERIFIED in substance**; the *Bézier* descriptor was not confirmed from the abstract (SVG stroke paths are Bézier in practice) — **not verified** |

**Not sampled (17 of 36 keys):** `ganwriting`, `hwt`, `wordstylist`, `metascript`, `graves2013`, `emuru`, `scriptvit`, `hi_gan`, `cffont`, `fontdiffuser`, `lffont`, `dgfont`, `difffont`, `diffvg`, `softras`, `icdar2013zh`, `boundink`. Each carries only a title-level or tool-level attribution ("generates pixels", "is the few-shot font counterpart", "is a differentiable rasteriser", "moved the backbone to attention") — low misrepresentation risk, but **not checked**. Stage 4.5 must cover them at 100%.

No case of serious misrepresentation, no cherry-picked number, and **no citation cited for a factual claim its source does not make.** The three MINOR distortions all drop a qualifier rather than invent one.

---

## 3. Phase C2 — Internal consistency

### C2.1 Abstract ↔ body ↔ Table I
- Abstract: "3,755/3,755 … median 10 strokes, 99th percentile 19" → §I contributions (same three numbers) → §IV ("3,755/3,755", "median of 10"). **Consistent.**
- No other number appears in the abstract. **Consistent.**
- Table II and Table III contain no numbers to check (metric/ablation names only).

### C2.2 Table I vs `analysis/lexicon_stats.json` — **exact match, 18/18 cells**

| Cell | Paper | JSON | |
|---|---|---|---|
| L1 chars / covered | 3,755 / 3,755 | 3755 / 3755 | ✓ |
| L1 mean / p99 / max / >20 | 9.77 / 19 / 24 / 0.37% | 9.766 / 19 / 24 / 0.37 | ✓ |
| L2 chars / covered | 3,008 / 3,008 | 3008 / 3008 | ✓ |
| L2 mean / p99 / max / >20 | 11.69 / 22 / 30 / 1.80% | 11.688 / 22 / 30 / 1.8 | ✓ |
| Union chars / covered | 6,763 / 6,763 | 6763 / 6763 | ✓ |
| Union mean / p99 / max / >20 | 10.62 / 20 / 30 / 1.01% | 10.62 / 20 / 30 / 1.01 | ✓ |

Also verified reproducible: 3,755 + 3,008 = **6,763** ✓; the GB2312 level split in `lexicon_stats.py` (`hi <= 0xD7` → level-1) is correct; `n_inventory` is the true GB2312 level-1/level-2 cardinality (3,755 / 3,008) ✓.

### C2.3 The two derived ratios — computed
- "level-2 characters are **20% more complex**": 11.688 / 9.766 = **1.19680** → **+19.68%**. "20%" is correct to the stated precision. ✓
- "**nearly five times** as likely to exceed 20 strokes": 1.80 / 0.37 = **4.865** (bounds from rounding: 4.79–4.93). "Nearly five" is accurate. ✓

### C2.4 "192 scalars" vs Table I max = 24, and vs the §III.D token formula
- 24 strokes × 8 scalars (one cubic = 4 control points × 2 coords) = **192** ✓ — so the 192 follows from the table's max of 24.
- **But §III.D states the token tensor is "N_c × (4m+1) scalars"**, which at m = 1 is **5 per stroke → 120**, and if the pen state is included the correct expression is (8m+1) → **216**. Neither 120 nor 216 is 192. **The formula and the headline number are mutually inconsistent.** → MINOR **m3**.
- "4,096 pixels for a 64×64 image": 64² = 4,096 ✓.

### C2.5 Figure / table cross-references
All four `\ref` targets (`sec:analysis`, `sec:protocol`, `tab:lexicon`, `tab:ablations`) have matching `\label`s; no dangling reference is possible. The built PDF contains **0 occurrences of "??" and 0 of "[?]"**. ✓
- **But** `fig:pipeline` (Figure 1) and `tab:metrics` (Table II) are **never referenced from the text** → MINOR **m2** (uncited float; also means Table II is never discussed).

### C2.6 New MEDIUM findings (internal consistency)

**M2 — the writer split cannot "respect the released partition".** §V.B: "Writer-disjoint: 860/60/100 train/validation/test writers, respecting the released partition where one exists." The CASIA paper states: "We previously partitioned the OLHWDB1.0 dataset into a training subset of 350 writers and a test subset of 70 writers. For uniform ratio in different datasets, we later reset the ratio of training writers and test writers as **4:1**. The same partitioning of writers is taken between online and offline datasets…" With 1,020 released writers that is **816 train / 204 test**. An 860-writer training set **exceeds the released training pool**, so 860/60/100 is a *re-split*, not a respected release. (860+60+100 = 1,020 ✓ internally, so the arithmetic is fine; the qualifier is what is false.)

**M3 — the coverage statistic cannot include corpus-disagreement exclusions.** §III.B: "Characters absent from the lexicon, **or whose lexicon stroke count disagrees with the corpus majority, are excluded from training and counted in the reported coverage statistic.**" Table I's "covered" column is `lexicon ∩ GB2312` only (`lexicon_stats.py` never touches a corpus), and §I.C states the CASIA corpora **were not obtained**. So the second exclusion class is neither computed nor counted, and "the coverage is complete for the target inventory" (end of §III.B) conflates *topology availability* with *trainable coverage*. This is the one internal inconsistency that makes a headline number look stronger than it is.

---

## 4. Phase D — AI research failure-mode checklist

| # | Mode | Outcome | Evidence |
|---|---|---|---|
| 1 | **Fabricated citations** | **CLEAR** | 42/42 VERIFIED against live registries/publisher pages; 0 NOT_FOUND, 0 MISMATCH, 0 dangling citations. The one title-squatting record was caught and rejected. (The `dp_font` author omission is a bibliographic completeness defect, not fabrication — see M1.) |
| 2 | **Implementation bugs** | **CLEAR (with two notes)** | The only code is `analysis/lexicon_stats.py`. Re-read line by line: GB2312 level split correct; levels match the canonical 3,755/3,008/6,763 cardinalities; `>20` matches the table header; output is deterministic (no sampling/randomness). **Notes:** (a) `strokes_p99` = `counts[int(0.99*(n-1))]` is a nearest-rank variant, not interpolated — immaterial here, but the percentile convention is undisclosed; (b) `CMD_ARGS` is dead code. Neither affects any reported number. |
| 3 | **Hallucinated results** | **NOT APPLICABLE** | The declaration holds: the manuscript reports no comparative result anywhere (verified by full read, not by trusting the declaration), and no number exists that is neither reproducible from `lexicon_stats.py` nor attributable to a source I re-read. There is no result to hallucinate. Note that the *surrounding* pipeline artefacts also contain no invented numbers — every figure in `phase2_literature/synthesis.md` §2 that I could trace (1,020 writers, the two quotes, ≈3.9M samples, 7,356 classes) is verbatim in the CASIA PDF. |
| 4 | **Shortcut reliance** | **SUSPECTED (narrow, specific)** | Two concrete instances, both in the *reference pipeline*, not the prose: (a) 12 of the 42 bibliography entries bypass `verified_refs.json` and enter through a **hand-typed `OVERRIDES` dict** in `refs/make_bib.py` with no stored retrieval record — and that shortcut is exactly where the `dp_font` author list lost its 6th author; (b) `refs/verification_report.md` (generated 08:15) was never regenerated after the 08:20 re-run, so it still declares 9 works "excluded from the manuscript" that the manuscript cites. The shortcut did not produce a false reference (I verified all 42 independently), but it removed the mechanism that would have caught M1. |
| 5 | **Bug-as-insight** | **NOT APPLICABLE** | No experimental result exists, so no result can be reinterpreted as a finding. The only computed artefact (Table I) is a feasibility count with no interpretive overreach beyond "the constraint is definable", which the data supports. |
| 6 | **Methodology fabrication** | **CLEAR (with a disclosure gap)** | Nothing is described as executed that was not: §IV's "we measured" is backed by a script and its output; the corpora are explicitly *not* used; the design is written in present tense as a design (normal for a proposal) and §VI puts code/weights in the future ("We will release"). **Gap (m13):** the manuscript never says outright whether CENT has been *implemented*. One sentence in §I.C would remove the only remaining way a reader could mistake §III for an implemented system. |
| 7 | **Pipeline-level frame-lock** | **CLEAR on documentary evidence** | The frame was interrogated and the record shows it: `devils_advocate_checkpoint1.md` returns REVISE with six items; C2 (paired supervision) was resolved by retrieving the supplier's documentation rather than assuming; C4 (OOV path) changed the method; C3 changed the metric (order-equivalence classes appear in §V.C); C5 changed the ethics text (explicit "no technical watermark … access control"); C6 changed the genre. A falsifiability statement and a negative-result commitment are in §II.E/§V.D. The frame that *cannot* be checked from artefacts is whether "no results" has become self-justifying — that is a quality judgement, outside this gate. |

---

## 5. Phase E — Bounded claim check (no registry built, per scope)

Classification: **(i)** attributed to a cited reference · **(ii)** reproduced from `analysis/lexicon_stats.py` · **(iii)** UNSUPPORTED. Sub-classification used for the three (iii) items: **(iii-a)** unverifiable or contradicted → SERIOUS; **(iii-b)** true and traceable to a source already in the reference list, but uncited at the point of use → **MINOR** (citation-placement fix). I state this rule explicitly rather than silently; under a mechanical "any (iii) = SERIOUS" rule, items E-14 and E-20 would move up, and I do not think that is the right reading of a fabricated-result audit.

| ID | Location | Assertion | Class | Note |
|---|---|---|---|---|
| E-1 | Abs, §I, §IV, Tbl I | 3,755/3,755 GB2312 level-1 covered | (ii) | exact |
| E-2 | Abs, §I, §IV | median 10 strokes | (ii) | exact |
| E-3 | Abs, §I | 99th percentile 19 | (ii) | exact |
| E-4 | §I, Tbl I | 192 scalars (24 strokes × 8) | (ii)+arithmetic | see m3 (formula clash) |
| E-5 | §I | 4,096 pixels for 64×64 | arithmetic | ✓ |
| E-6 | §IV, Tbl I | mean 9.77 / 11.69 / 10.62; p99 22 / 20; max 24 / 30 / 30; 0.37% / 1.80% / 1.01% | (ii) | 18/18 exact |
| E-7 | §IV | level-2 "20% more complex" | (ii) derived | 1.19680 → +19.7% ✓ |
| E-8 | §IV | "nearly five times as likely to exceed 20 strokes" | (ii) derived | 4.865 ✓ |
| E-9 | §IV | lexicon 9,574 characters | (ii)/LOC | 9,574 lines, 30,778,076 bytes ✓ |
| E-10 | §IV, Tbl I | lexicon licensed Arphic Public License | (i) `hanzi` | COPYING confirms ✓ |
| E-11 | §V.A | 1,020 writers, corpora names | (i) `casia` | ✓ |
| E-12 | §V.A | both quoted phrases | (i) `casia` | verbatim ✓ |
| E-13 | §II.B | One-DM reports a Chinese setting on ICDAR-2013 | (i) `one_dm` | ✓ supp. §C.4 |
| E-14 | §V.D (T3) | ICDAR-2013 test set has 60 writers | **(iii-b)** | True (ICDAR-2013 paper's data; independently stated by One-DM supp. and HiSAC), **uncited here** → add `\cite{icdar2013zh}`. MINOR. |
| E-15 | §II.A | DiffusionPen hybrid encoder | (i) `diffusionpen` | m5 (over-general "better") |
| E-16 | §II.C | Diff-Writer global-structure argument | (i) `diffwriter` | ✓ exact |
| E-17 | §II.C | SDT different frequency scales | (i) `sdt` | ✓ exact |
| E-18 | §II.D | DP-Font 36-dim stroke-order vector | (i) `dp_font` | ✓ exact |
| E-19 | §II.D/§VI | DP-Font stroke conditioning helps | (i) `dp_font` | ✓ ablation Table 3 |
| E-20 | §VI | HiSAC: cursive "hardest to imitate" | (i) `hisac` — **overstated** | m7 |
| E-21 | §II.B | RC-GAN radical loss reduces stroke error | (i) `rcgan` | ✓ |
| E-22 | §II.B | InkSpire unified latent, bilingual | (i) `inkspire` | ✓ |
| E-23 | §II.D | SCFont separated-stroke trajectories | (i) `scfont` | ✓ |
| E-24 | §II.D | Stroke2Font stroke-element graph + Bézier | (i) `stroke2font` | ✓ |
| E-25 | §II.A | VATr archetypes → long-tail | (i) `vatr` | ✓ |
| E-26 | §II.A | VATr++ regularised inputs | (i) `vatr_pp` | ✓ |
| E-27 | §II.A | DSD factorised style | (i) `dsd` | ✓ |
| E-28 | §II.C | layout_glyph extends to lines | (i) `layout_glyph` | ✓ |
| E-29 | §I | FID uses Inception features on natural photographs | (i) `fid`,`inception` | ✓ |
| E-30 | §I | "FID is close to blind to glyph identity" | **(iii-lite)** | Hedged, standard, argued — but **uncited**. Recommend citing the FID/inception line and softening to "insensitive to glyph identity by construction of its feature space". Not counted SERIOUS: not empirical data, hedged, and true of FID's construction. |
| E-31 | §III.C | "the five basic categories: horizontal, vertical, left-falling, right-falling, turning" | definitional | Standard Chinese stroke typology; **also stated by DP-Font** ("five most basic types of strokes") — optional `\cite{dp_font}`. Note only. |
| E-32 | §V.B | 860/60/100 writers | protocol parameter | Sums to 1,020 ✓, but see **M2** |
| E-33 | §V.B | 3 seeds; 95% bootstrap over items | protocol parameter | no external claim |
| E-34 | §V.C | order-equivalence classes as primary statistic | protocol design | ✓ (answers DA C3) |
| E-35 | §V.D | "95% interval excluding zero"; FID band | protocol rule | no external claim |
| E-36 | §VI | "no number without provenance — run ledger" | process commitment | consistent with the no-results state |

**Result: 0 × (iii-a) SERIOUS. 2 × (iii-b)/overstatement (E-14, E-20) at MINOR. 1 × (iii-lite) (E-30) as a recommendation.** **No unsupported empirical number exists anywhere in the manuscript.** This is the clean result the paper was trying to earn, and it earned it.

---

## 6. Phase A/B/C/D/E — correction list

Severity order. "Fix" is the exact edit.

| ID | Sev | Item | Exact fix |
|---|---|---|---|
| **M1** | **MEDIUM** | `refs.bib` `dp_font` lists 5 authors; the paper has 6 (Liguo Zhang, Yalong Zhu, Achref Benarab, Yusen Ma, Yuxin Dong, **Jianguo Sun**) | Add `and Jianguo Sun`. Also add `doi = {10.24963/ijcai.2024/863}` and `pages = {7796--7804}` from the IJCAI record. Fix the `OVERRIDES["dp_font"]` author string in `refs/make_bib.py`, not just the .bib. |
| **M2** | **MEDIUM** | §V.B "860/60/100 … respecting the released partition where one exists" is incompatible with CASIA's released 4:1 partition (816/204 of 1,020) | Either (a) drop the clause: "Writer-disjoint: 860/60/100 train/validation/test writers (a re-split; the released 4:1 partition is not preserved — \cite{casia})"; or (b) adopt 816/84/120 or 816/102/102 and say which. State it once; §V.A already says the partition must "be stated exactly". |
| **M3** | **MEDIUM** | §III.B claims corpus-disagreement exclusions "are … counted in the reported coverage statistic"; Table I counts lexicon presence only, and no corpus was obtained | Rewrite: "Characters absent from the lexicon are excluded and counted (Table I). A second exclusion — lexicon stroke count disagreeing with the corpus majority — **cannot be computed before the corpus is obtained** and will be reported as a corpus-vs-lexicon disagreement rate at that point." Then Table I's "covered" column means what it says. |
| m1 | MINOR | 6 uncited bib entries (`ddpm`, `ldm`, `dit`, `calligan`, `crnn`, `cfg`) | Delete, or cite where they belong (e.g. `ddpm`/`ldm` in §III.D's diffusion framing, `crnn` at §V.C's frozen HTR recogniser). The built PDF already omits them, so this is bibliography hygiene, not a reader-facing defect. |
| m2 | MINOR | `fig:pipeline` and `tab:metrics` are never `\ref`'d | Add "Figure~\ref{fig:pipeline}" in §III.A and "Table~\ref{tab:metrics}" in §V.C. |
| m3 | MINOR | §III.D "N_c × (4m+1) scalars" contradicts "192 scalars" at m=1 (5 vs 8 per stroke; 9 with pen state) | Replace with the true count: `N_c × (8m+1)` (geometry + pen state) and reconcile the 192: "192 scalars of geometry plus 24 pen bits for the most complex level-1 character". |
| m4 | MINOR | §V.D T3 "(60 writers)" uncited | Append `\cite{icdar2013zh}` (and optionally `one_dm`, whose supp. states it). |
| m5 | MINOR | §II.A "conditions better than either alone" | Qualify: "…gives the best writer-classification accuracy and CER of its ablations, at a small non-significant cost in FID". |
| m6 | MINOR | §II.D "removing it produces characters with stroke errors" | Restore the source's hedge: "reports that without explicit stroke-order encoding, models **may** produce stroke errors on characters with difficult structures". |
| m7 | MINOR | §VI HiSAC "cursive connections hardest to imitate" | "identifies cursive/running-script deformation as the case that remains hard". |
| m8 | MINOR | 4 entries print the series only, with no conference (`diffusionpen`, `one_dm`, `hisac` → "Lecture Notes in Computer Science"; `diffwriter` → "Communications in Computer and Information Science") | Add the conference to `booktitle`: "Computer Vision – ECCV 2024", "Computer Vision – ECCV 2026", "Neural Information Processing (ICONIP 2023)". |
| m9 | MINOR | `lvgh` has no authors at all, and the stated reason is false | Add "Xinyue Zhang and Haolong Li and Jiawei Ma and Chen Ye" (verified via arXiv export API and abs page). Remove `omit_authors`. |
| m10 | MINOR | `stroke2font` has no volume/issue/article number/DOI | Add `journal = {Algorithms}`, `volume = {19}`, `number = {3}`, `pages = {231}`, `year = {2026}` (verified on the MDPI page). I did **not** verify a DOI for this article — take it from the MDPI page rather than from me. |
| m11 | MINOR | Verification artefacts contradict the bibliography: `verified_refs.json` counts (30+4+0+1=35 ≠ total 40), 5 statuses use "UNMATCHED" outside its own vocabulary, 12/42 bib entries are not ACCEPTED (7 have no record at all: `dsd`, `boundink`, `dp_font`, `scfont`, `stroke2font`, `hanzi`, `lvgh`); `verification_report.md` is stale and lists 9 cited works as "excluded from the manuscript"; `synthesis.md` §1's "only if ACCEPTED" is false for 12 entries | Regenerate both artefacts from the store **after** adding a record for every OVERRIDE (with its DOI/arXiv/URL and retrieval timestamp), fix the counts, and stop claiming "excluded" for anything in the bib. Without this, a Stage 4.5 re-run started from these files will mis-audit the paper. |
| m12 | MINOR | Acknowledgement "all cited works were resolved against bibliographic registries" | Say what actually happened: "resolved against Crossref/arXiv records, publisher proceedings pages, or the work's own full text; one title-squatting record was rejected." `hanzi` is a project repository, not a registry. |
| m13 | MINOR | §III never states whether CENT is implemented | One sentence in §I.C: "The architecture in §III is specified, not yet implemented; no artefact was trained for this draft." |
| n1 | NOTE | `pipeline_state.json` says `current_stage: 1`, `awaiting_checkpoint_1`, and lists "empirical scope: protocol-only vs reduced-scale pilot" as a **pending author decision** — while the manuscript has already been written as protocol-only, and `integrity_gates.2.5` still reads `not_run` | Close the author decision and advance the state before Stage 3; otherwise the paper's genre rests on an unrecorded decision. Not a manuscript defect. |
| n2 | NOTE | `lexicon_stats.py` uses a nearest-rank percentile without disclosure, and `Table I` reports no median column although median is quoted in text | Optional: state the percentile convention in the caption; consider adding the median column (10 / 11 / 10) since the abstract quotes it. |

---

## 7. What I did NOT check (coverage boundaries)

Stated so no reader mistakes this report for an all-integrity guarantee.

1. **17 of 36 cited keys were not context-checked** (listed in §2). Their attributions are title-level or tool-level; I did not open those papers. **Semantic extraction of "all citations" is not claimed and was not attempted.**
2. **I did not read the full text of most cited works.** For 30+ entries the evidence is the registry/publisher record (title, authors, venue, year) plus, where a specific claim is made, the abstract/supplementary or the exact passage. "The reference exists and says what is attributed to it" is bounded by the passages I retrieved; it is not a full-text audit.
3. **I did not verify the content of `boundink` (arXiv:2604.02103) or `scriptvit` (2511.18307) beyond registry metadata.** Both are recent preprints; existence, title, authors and date are confirmed, nothing else.
4. **`hanzi` year (2024)** is unverified; the project page is undated. Immaterial to usage; the licence and record count are verified.
5. **`stroke2font` DOI** — I did not verify one; add it from the MDPI page, not from this report.
6. **`lvgh` "Bézier"** descriptor unverified (see §2). Everything else about that citation is confirmed.
7. **I did not compile the manuscript.** Reference/figure resolution was checked statically (all `\label`/`\ref` pairs) and against the existing `paper/CENT-draft.pdf` (0 × "??", 0 × "[?]", 36 printed references). Any LaTeX error introduced after that PDF build is outside this check.
8. **No statistical review** of the frozen protocol's adequacy (power, multiplicity, whether 3 seeds suffice), and **no judgement of the method's merit, novelty or publishability** — those belong to the reviewers, not this gate.
9. **Licence compliance is checked only as attribution**: `graphics.txt` is correctly attributed as Arphic PL, and the paper states it uses only topology. I did not audit redistribution obligations for a derived stroke table (the blueprint asserts they exist and are Arphic PL — plausible, not audited here).
10. **Same-source risk (declared).** I am an AI verifying references against live registries; that mitigates same-source hallucination for *existence*, but my reading of whether a paraphrase is faithful (§2's MINOR distortions) is still a same-family judgement. Three such calls are the ones a human should re-read: `diffusionpen` (m5), `dp_font` (m6), `hisac` (m7).

---

## 8. Bottom line for the author

The paper's honesty claim survives contact with the evidence. Nothing in it is invented: no result, no borrowed number, no fake reference, and the one known trap in this project (the FID title squatter) was correctly rejected and is visible as rejected. The feasibility numbers reproduce exactly from the released script, and the two derived ratios in §IV are arithmetically right.

What fails the gate is smaller and duller: a reference that lost an author, a writer split that cannot be what it says it is, and a coverage statistic described as containing something it never computed. Fix those three sentences and the bibliography line, regenerate the verification artefacts, and this manuscript should pass a re-run of this gate at zero SERIOUS / zero MEDIUM.

---

# 9. Correction round 1 — re-verification of the fix set

**Scope discipline:** I re-verified *only* the claimed fixes, against the current files (`paper/main.tex`, `paper/refs.bib`, `refs/overrides.json`, `refs/make_bib.py`, `refs/verification_report.md`, `state/pipeline_state.json`). **Phase A was not re-run**; the 42/42 VERIFIED result and its per-entry evidence stand as recorded in §1, and the two entries that changed (`dp_font`, `lvgh`) were re-read in full. I did not take the lead's summary as evidence for anything.

## 9.1 Revised verdict

| | |
|---|---|
| **Verdict** | **PASS WITH NOTES** |
| Counts | **0 SERIOUS · 0 MEDIUM · 6 MINOR open · 3 nits** |
| Rule | Zero SERIOUS/MEDIUM but ≥1 MINOR → release, with the MINOR list and notes attached. Verified: no new SERIOUS or MEDIUM entered with the fixes. |

## 9.2 Item-by-item

| Item | Claimed | Verified (how) | Result |
|---|---|---|---|
| **M1** dp_font 6th author | override moved to JSON, Sun added | `refs/overrides.json` #dp_font author string ends `… and Yusen Ma and Yuxin Dong and **Jianguo Sun**`; `paper/refs.bib` `@inproceedings{dp_font}` prints the same six names; evidence comment records why | **FIXED** |
| **M2** writer splits | rewritten, 816/204 | §V.B now: "following the released four-to-one partition of the 1,020 writers: 816 train and 204 test writers. A 60-writer validation subset is carved from the 816 and is used *only* for the gate; it never contributes to a reported number." Arithmetic: 816 + 204 = **1,020** ✓; 1,020 at 4:1 = **816 / 204** ✓ (matches the CASIA sentence "we later reset the ratio … as 4:1"). Full-text grep: **no "860" anywhere**, and "respecting the released partition" is gone | **FIXED** |
| **M3** coverage claim | rewritten | §III.B now splits the two exclusions: lexicon-absent characters "are excluded"; the corpus-majority check "cannot be run until the corpora are obtained; it is applied at training time … Table I itself measures lexicon coverage of the inventory and nothing else." Cross-checked against §IV, which now reads "total … every one of the 3,755 level-1 characters has a defined stroke count and order, so no character is un-generatable **for lack of topology**" — the claim and the table now measure the same thing | **FIXED** |
| m1 orphans | 42/42 cited | My own regex over every `\cite{}` in `main.tex`: **42 distinct keys, 42 bib entries, 0 orphans, 0 dangling.** New homes confirmed by reading: `ddpm,ldm,dit` → §III.D; `crnn` → Table II ("CER of a frozen HTR recogniser"); `cfg` → §III.F ("Sampling uses classifier-free guidance \cite{cfg}"); `calligan` → §II.B | **FIXED** |
| m2 uncited floats | both referenced | `fig:pipeline` → §III.A ("Figure~\ref{fig:pipeline} summarises the architecture"); `tab:metrics` → §V.C ("Table~\ref{tab:metrics} lists the six instruments"). All 6 `\ref` targets have labels; 6 refs / 6 labels | **FIXED** |
| m3 scalars | 8 per stroke | §III.D: "N_c × 8 scalars (four control points per stroke, two coordinates each, so 192 for the most complex level-1 character)". 24 × 8 = **192** ✓; "(4m+1)" no longer appears anywhere in the file | **FIXED** — one residual nit, below |
| m5 DiffusionPen | softened | §II.A now: "DiffusionPen combines metric learning and classification in a single style encoder in order to capture both seen and unseen styles" — matches the paper's own abstract ("capture both textual and stylistic characteristics of seen and unseen words and styles"); "better than either alone" is gone | **FIXED** |
| m6 DP-Font hedge | "can" | §II.D: "shows in ablation that models without explicit stroke-order encoding *can* generate characters with stroke errors" — "can" ≈ the source's "may" | **FIXED** (the "characters with difficult structures" qualifier is still dropped; optional) |
| m9 lvgh authors | 4 names added | `refs.bib` `@article{lvgh}` now `Xinyue Zhang and Haolong Li and Jiawei Ma and Chen Ye`. **Independent confirmation, re-queried this round:** arXiv export API `id_list=2511.11119` → *Stroke Modeling Enables Vectorized Character Generation with Large Vectorized Glyph Model*; authors **Xinyue Zhang; Haolong Li; Jiawei Ma; Chen Ye**; published 2025-11-14. Match | **FIXED** |
| — overrides as data | JSON, evidence per entry | `refs/make_bib.py` lines 30–33: `OVERRIDES_PATH = ROOT / 'overrides.json'` and `OVERRIDES = json.loads(OVERRIDES_PATH.read_text(...))`. **No entry remains a code literal.** All **12** entries carry an `evidence` string: fid, hisac, diffwriter, difffont, dsd, boundink, insprire, dp_font, scfont, stroke2font, lvgh, hanzi | **FIXED** (12/12, not the 6 checked) |
| — report derived | `render_report.py` | Header now names the generator and its three inputs; provenance classes (registry / full-text / rejected) are separated; the stale "Unmatched — excluded from the manuscript" block is gone; every cited work shows `in manuscript: yes`; the Coverage boundary states "42 entries ship in the manuscript; all 42 are cited"; one rejected match (`fid`). **But the table has two wrong rows and omits seven entries** — see m11 below | **PARTLY FIXED** |
| — pipeline_state | no longer stage 1 | `current_stage: "2.5"`; stage 1 `complete`; stage 2 `draft_complete`; stage 2.5 `running`; `integrity_gates.2.5: "running"` | **FIXED** |
| m13 implementation | — | Already in the file, no action needed: §I.C now reads "*CENT is specified here, not implemented*: no reference implementation exists as of this draft, so no number below is a measurement of a running system except the lexicon analysis in Section IV." | **FIXED** |

## 9.3 Residual MINOR items (do not hold the gate)

| ID | Sev | Item | Fix |
|---|---|---|---|
| **m11** | **MINOR — must fix before Stage 4.5** | The derived report is still wrong in two rows and incomplete in seven. **(a)** `fid` row prints identifier `10.18034/ajase.v8i1.9`, venue *Asian Journal of Applied Science and Engineering*, year **2019** — i.e. the squatter, listed as this citation's identity by the very table whose "Rejected matches" section six lines below says that record was rejected and superseded by arXiv:1706.08500. **(b)** `inkspire` row prints `10.1016/j.image.2023.116958`, *Signal Processing: Image Communication* 2023 — the rejected Crossref near-miss ("Stylized image denoising…"), not the real ICLR 2026 record. **(c)** The table has 40 rows for 42 shipped entries: `dp_font`, `dsd`, `boundink`, `lvgh`, `scfont`, `stroke2font`, `hanzi` are absent (they have no `verified_refs.json` record, and the renderer appears to iterate the store), so the entry that lost an author and the entry that lost its authors are both invisible in the file whose job is to catch that. Root cause: for full-text overrides the renderer merges the *stale store's* `matched_title`/`doi`/`venue` into the row instead of the override's own record. | Render from `overrides.json` for every override key (never from the store's near-miss fields); iterate the union of store keys + override keys + bib keys; and for `fid` print `arXiv:1706.08500` / *NeurIPS 2017* with the squatter only in the Rejected block. **Do not hand this file to Stage 4.5 as it stands — a re-audit started from it would re-import the squatter.** |
| m4+ | MINOR | Two corpus facts still asserted without a local citation: §V.D "the ICDAR-2013 test set is small (**60 writers**)" and the newly added §V.B "the released **four-to-one** partition". Both are true (60 writers is corroborated by One-DM's supplementary §C.4 and HiSAC; the 4:1 ratio is in the CASIA PDF) and `casia`/`icdar2013zh` are cited in §V.A, but not at these sentences. | Add `\cite{icdar2013zh}` at T3 and `\cite{casia}` at the 4:1 clause. |
| m7 | MINOR | §VI still reads "the Chinese online literature finds cursive connections **hardest to imitate** \cite{hisac}". HiSAC's conclusion says severe cursive deformation "remains a challenging direction for future work" — "remains challenging" is not the comparative "hardest". | "…identifies cursive/running-script deformation as the case that remains hard." |
| m8 | MINOR | Venue imprecision unchanged: `diffusionpen`, `one_dm`, `hisac` print only "Lecture Notes in Computer Science"; `diffwriter` only "Communications in Computer and Information Science". No conference named, so a reader cannot locate the paper. | Name ECCV 2024 / ECCV 2026 / ICONIP 2023 in `booktitle` (and in `overrides.json`, which is now the source). |
| m10 | MINOR | `stroke2font` still carries no volume, issue, article number or DOI; the evidence line says so deliberately ("DOI not recorded here; the article page is the evidence"). | Add `volume = {19}, number = {3}, pages = {231}` from the MDPI page; take the DOI from the page too (I did not verify a DOI for this article). |
| m12 | MINOR | Acknowledgment unchanged: "all cited works were resolved against bibliographic registries". For the 12 full-text overrides the resolution was a proceedings PDF / DOI record / project repository, and `hanzi` is a software repository, not a registry. | "…resolved against Crossref/arXiv records, publisher proceedings pages, or the work's own full text; one title-squatting record was rejected." |
| m14 *(new)* | MINOR | New text in §III.F: "Sampling uses classifier-free guidance \cite{cfg} with the character condition dropped at a fixed rate, so that the amount of **style** conditioning can be traded at inference." The mechanism is right, the direction is not: both branches keep $S_w$, so what the guidance weight trades is **character/content** adherence against the unconditional prior — not style. | "…so that adherence to the character condition can be traded at inference." |

**Nits (not counted):** (n-a) `\cite{ddpm,ldm,dit}` supports "a one-dimensional U-Net"; DiT's point is that a transformer *replaces* the U-Net, so the third citation argues against the sentence's architecture — either drop `dit` or say "diffusion backbones". (n-b) §III.D's "N_c × 8 scalars" omits the per-stroke pen state $d_i$ that §III.A defines, so the token count is 8m+1 if pen state is in the tensor; the 192 stays arithmetically right for geometry, so this is a one-clause clarification, not a contradiction. (n-c) `state/pipeline_state.json` has UTF-8/GBK mojibake in several string values (`"REVISE 鈥? C4 fixed"`, `"搂I.C"`, `"搂IV"`) — the JSON parses, but the record's text is corrupted and will be re-read by later stages; rewrite it as UTF-8.

## 9.4 Answers to the two questions

1. **Implementation disclosure — is it blocking? No, and you have already closed it.** §I.C now states "*CENT is specified here, not implemented*: no reference implementation exists as of this draft, so no number below is a measurement of a running system except the lexicon analysis in Section IV." That is exactly the sentence I would have asked for, in the right place. **Do not add a second one.** My only remaining disclosure item is the Acknowledgement wording (m12, MINOR).
2. **`pipeline_state.json`** — confirmed: no longer stage 1 (`current_stage: 2.5`, stage 1 complete, 2.5 running). One note: it now contains mojibake (n-c).

## 9.5 What this round did not re-check

Phase A (42 entries) was **not** re-run; the §1 verdict for 40 unchanged entries rests on this session's earlier live lookups. The two changed entries (`dp_font`, `lvgh`) were re-read in full and both are now correct. Phase B was **not** re-sampled beyond the sentences that changed; the three MINOR_DISTORTION calls from §2 stand except where noted (m5, m6, m7 — two fixed, one open). The manuscript was **not** recompiled: §9 checks are against the source, so the `paper/CENT-draft.pdf` on disk predates these edits and any LaTeX error introduced by them is outside this round. The eight unchanged `analysis/` figures were not recomputed — Table I is untouched and still matches `lexicon_stats.json` cell for cell.

**Standing conclusion:** nothing in the corrected manuscript violates its central claim. Every number is still either computed by the released script (Table I, exact) or attributed to a source I read. The gate now opens at PASS WITH NOTES; the one thing I would not let travel to Stage 4.5 is the derived reference report (m11).
