# Cursive Chinese text-line data — what actually exists

**Question.** Is there a dataset of Chinese *lines of text* with *cursive continuity between characters* (连笔), sufficient to train and evaluate CENT's contributions (ii) and (iii)?

**Answer in one line.** No dataset is purpose-built for it, and no published work has *measured* how much cursive any Chinese line corpus contains — but cursive material does exist in at least six corpora, and one online corpus (DCOH-120K) is far better suited than the one everyone uses (CASIA-OLHWDB 2.x). The decisive next step is a measurement, not a search.

**Most suitable — revised after the wide search, and after you ruled out DCOH-120K.** The best-available corpus for CENT is **SCUT-COUCH2009**, for a reason that took the full search to establish: it is the only Chinese trajectory corpus whose own publication **documents cursive writing that crosses character boundaries**, and its word data carries **per-character bounding boxes and per-character stroke indices**, which is exactly the supervision a boundary/connection model needs (§9). Use **TextLine_NU** (8,809 unconstrained lines, >190 writers) for line-level training and **Word8888 / Word17366 / Word44208** for boundary-dense training, with **CASIA-OLHWDB 2.0–2.2** kept because every baseline number in the literature is quoted on it. DCOH-120K is dropped (no access, and your visual inspection found ordinary unconnected writing — consistent with it being tablet-captured with a stylus). Names + links are in §8.

---

## 0. Two different things get called "cursive" — keep them apart

This distinction decides the dataset choice, and prior papers conflate them:

| | What it is | Where it lives | What CENT needs from it |
|---|---|---|---|
| **(a) 行书 / 行草 script style** | The *script* is running/cursive: simplified strokes, glyphs merged inside a character, sometimes character boundaries dissolved | Calligraphy corpora, historical manuscripts, 行书 fonts | Glyph-shape prior; extreme cases where character boundaries vanish |
| **(b) 连笔 across characters in ordinary handwriting** | Ordinary 楷书/行楷 writing where the pen does not lift between two characters — a *connection stroke* crosses the boundary | Ordinary line corpora, naturalistic writing | **The actual supervision signal for contribution (ii)** |

CENT's contribution (ii) is about **(b)**. Contribution (iii) is the gap between data that has (b) and models trained on data that does not. A calligraphy dataset of (a) does **not** solve (b): a 草书 masterpiece has no per-writer, per-line, modern-handwriting statistics, and no trajectories.

---

## 1. Online (trajectory) line corpora — the only modality that can supervise a connection stroke

| Dataset | Lines | Writers | Unit | Pressure/time | Access | Notes |
|---|---|---|---|---|---|---|
| **CASIA-OLHWDB 2.0–2.2** | **52,220** | **1,019** | page + line + per-character stroke indices | x,y only (30 Hz, no pressure) | free for research, application form | The de-facto benchmark (OLHWG, DiffInk). Copied news text ⇒ mostly formal 楷书 |
| **DCOH-120K** (DCOH-Chinese) | **83,142** | **314** | line | **x, y, stroke idx, timestamp, tilt, pressure** | CC BY-NC-ND 4.0; application + 1–2 publications | Modern informal corpus (news, forums, comments); tablet+stylus. Paper: EGO-LM, *Pattern Recognition* 2025 |
| **DCOH-E** (extended, used in OLIWER) | 87,759 | 567 | line | same | via DOLPHIN release | 5,100 extra lines / 255 extra writers |
| **SCUT-COUCH Textline_NU** | 8,809 | >157 | line | x,y (Unipen `.SEGMENT`/`.PEN_DOWN`) | free for research w/ letter; **US$2,000 commercial** | Explicitly *unconstrained*; 17 % collected in Nantes |
| **OLIWER** (derived) | **674,017 phrases (2–5 chars)** | **1,731** | phrase | x,y + pressure (p=1 where absent) | GitHub `SCUT-DLVCLab/DOLPHIN` | Built *by segmenting lines at character boundaries*, so a phrase **spans real boundaries with real continuity** |

**Why this table matters.** Everyone in the generation literature trains on CASIA-OLHWDB 2.x. DCOH-120K is **1.6× larger in lines**, has **pressure**, is **informal-register** (people writing comments and forum posts connect strokes far more than people copying a news paragraph), and — decisively — carries **timestamps**, which make the pen-lift gap between two characters directly measurable.

**The measurable definition of (b) that nobody has published.** For each adjacent character pair in a line, compute

- `Δt` = timestamp of first point of char *i+1* minus last timestamp of char *i*
- `Δs` = Euclidean distance between those two points

A pair with `Δt ≈ 0` and small `Δs` **is** a connection (the pen never left the paper). Every pure-x,y dataset (CASIA, COUCH) can still support the `Δs` half, and CASIA's per-point character labels (confirmed by the OLIWER paper: "the character annotations, i.e., the character belonging to each point, are provided") let you detect a single stroke that spans two character labels — an exact, if conservative, definition.

**So the prevalence of cursive in the standard benchmark is an afternoon of scripting away, and no paper has done it.** That measurement is CENT's first ablation and, if the number is small, its central motivation.

---

## 2. Offline (image) line and page corpora — natural handwriting, no trajectories

| Dataset | Lines / pages | Writers | Why it matters | Access |
|---|---|---|---|---|
| **HIT-MW** | 8,664 lines / 853 forms | **>780** | **Naturally written, mailed in, no pre-printed boxes**, "real handwriting phenomena such as miswriting and erasing" — the closest thing to how people actually write | Free download (documents + textlines), non-commercial; **textual ground truth is charged/on request**; authorisation letter required to publish |
| **SCUT-HCCDoc** | 116,629 lines / 12,253 images | — | Camera-captured handwriting **from social media** (Weibo, WeChat, Instagram) ⇒ informal, likely cursive | on request |
| **SCUT-EPT** | 50,000 lines | 2,986 volunteers | Real **exam papers**: erasures, supplements, dense grids | non-commercial; application + encrypted archive |
| **CASIA-HWDB 2.0–2.2** | 52,220 lines (= the online pages, scanned) | 1,019 | Same writers as OLHWDB ⇒ **online↔offline pairing per writer** | free for research |
| **CASIA-HWDB-T** | **56,469 touching 2+-char strings** | — | Hand-annotated **touching characters** with touching points — direct evidence that inter-character contact exists in CASIA, and a ready-made boundary-hard case set | on request |

HIT-MW is the most under-used resource here: 780+ writers, *genuinely* natural writing, and a per-form writer identity. It is image-only, which is precisely why a DiffRender-style bridge (§4) becomes attractive.

---

## 3. Genuinely cursive *line* material — but historical, and image-only

| Source | Cursive content | Size | Access |
|---|---|---|---|
| **EvaHan 2026 Dataset C** | **5,000 train + 200 test hand-written line images "in regular (kaishu) and cursive (xingshu) scripts"** — the single most on-target cursive-line statement found | ~5,200 lines | Registration by e-mail to evahan2026@gmail.com; training set released 2026-01-01, registration closed 2026-01-30. **Likely too late for the official split — ask the organisers directly.** Traditional characters, historical Buddhist-canon material |
| **UniCalli dataset** (HF `TSXu/UniCalli_dataset`) | Chinese calligraphy, **95+ masters, 5 script styles**, with bounding boxes + modern transcriptions + author + style labels; **column-level** generation | 1.13 GB | public on HuggingFace; research use |
| **Chronicles-OCR** | 2,800 images, **400 per script**, seven scripts incl. 行书 / 草书, paragraph-level parsing | 2,800 | research only |
| **Ancient-Bench** | 2,700 images, 9 media, 7 scripts incl. Running / Cursive | 2,700 | HF `hiuyi/Ancient-Bench` |
| **Traditional-Chinese OCR synthetic** (Lin 2025) | **4,103,595 line images**, varied calligraphic renderings + degradation | 4.1 M | HuggingFace |
| **UltraGlyph** (HF `ENC-PSL/evahan-ultraglyph`) | 129,386 synthetic lines mixing real glyphs with **TTF-rendered Regular / Running / Cursive** fonts | 152 k rows | public |

**Synthetic cursive is real and available.** UNICALLI (ICLR 2026) explicitly renders with "three TrueType Font (TTF) files for Regular, Running, and Cursive scripts", and UltraGlyph does the same at scale. The catch is the one CENT must not paper over: a 行书 font gives *glyph-level* cursive (a) and **no connection stroke between characters with real pen dynamics** (b). It cannot supply trajectory supervision for a join.

---

## 4. What no dataset provides

1. **No dataset is labelled for continuity.** There is no `pen_connected(i, i+1)` flag, no ligature annotation, anywhere. Every candidate metric must be computed or annotated. (Consistent with §3.4 of the main survey: no Chinese paper evaluates the join.)
2. **No modern cursive line corpus with writer identity *and* trajectories.** (b)-rich material is either image-only (HIT-MW) or historical (EvaHan), and (b)-capable trajectories (CASIA, DCOH) are largely formal writing whose cursive content is **unmeasured**.
3. **No open-set cursive evaluation line set.** So a Chinese continuity benchmark is itself a contribution — and unlike most "new benchmark" claims, this one is defensible because the need is documented by two independent 2026 papers (CASHG reporting "no cursive positives"; OLHWG conceding it cannot mimic "extensive cursive connections").

---

## 5. The decisive pilot (do this before designing CENT)

Cheap, bounded, and it settles contribution (iii):

**P1. Measure pen-lift gaps.** On CASIA-OLHWDB 2.0–2.2 and DCOH-120K, compute `Δt` and `Δs` for every adjacent character pair. Plot the joint distribution. Report the fraction of pairs with `Δt < ε` (DCOH only) and `Δs < δ` (both). *Expected outcome either way is informative*: a bimodal distribution with a distinct near-zero mode means the supervision exists and is merely unused; a unimodal distribution with no near-zero mass confirms the scarcity claim quantitatively.

**P2. Audit the benchmark's own claim.** Verify CASHG's "no cursive positives" by the stroke-spanning definition (§1). If CASIA-OLHWDB 2.x has 5–15 % of boundaries connected, CENT can learn joins from it and contribution (iii) shifts from *no data* to *long-tail, class-imbalanced data* — a much easier and more honest story.

**P3. Check the writer × register interaction.** DCOH is informal, CASIA is transcribed news. Compare (b) rates between them. If informality raises cursive by a large factor, then "register mismatch" — not "missing data" — is the real framing of contribution (iii), and that is a stronger, more measurable claim than scarcity.

---

## 6. Recommendation

A **two-source strategy**, in this order:

1. **Supervise joins where they are measurable** — DCOH-120K (83 k lines, 314 writers, pressure + timestamps, informal register) as the primary online training corpus, with CASIA-OLHWDB 2.x retained for comparability with OLHWG/DiffInk. Use OLIWER's 674 k phrases as a boundary-dense auxiliary set, since each phrase deliberately spans 2–5 real character boundaries.
2. **Build a small, honest continuity benchmark** — a few hundred curated cursive lines (hand-picked 行楷/行书 from HIT-MW + EvaHan C if shared + UniCalli columns), annotated for connection, used only for evaluation. That is enough for a CVPR claim and is a genuinely new artifact.
3. **Do not** base training on 行书-font synthesis. It gives (a), not (b), and a reviewer who knows WriteFreely's Bézier shortcut will say so.

**Licensing reality check.** Nothing here is free for unrestricted use: DCOH-120K is CC BY-NC-ND with an application; CASIA is research-only by agreement; SCUT-EPT/HCCDoc are application + encrypted; COUCH has a US$2,000 commercial price; HIT-MW's *text* costs money. For a CVPR paper this is all fine (research use), but every dataset needs its agreement in place before training, and the paper must state licenses.

---

## 7. Bottom line for the discussion
The scarcity premise is **half right**. There is no "cursive Chinese line dataset" to download — but there is more than enough material to *measure*, *learn from*, and *evaluate* cursive joins, and the best-suited corpus (DCOH-120K) does not appear in any generation paper I found (the generation literature uses CASIA by default). The open question is no longer "is there data?" but "**how much cursive is in it, and is it enough for a boundary-explicit model to learn from?**" — which P1–P3 answer in days, not months.

---

## 8. Access list — name + link

Primary candidates are bolded. "Verified" means the link resolved during this search; unverified rows say what to do instead.

### Online (trajectories) — the modality CENT generates

| Dataset | Link | Access |
|---|---|---|
| **DCOH-120K** | `https://github.com/SCUT-DLVCLab/DCOH-120K` | Application form + 1–2 publications → e-mail to `lianwen.jin@gmail.com` / `eelwjin@scut.edu.cn`; CC BY-NC-ND 4.0. JSON per line: `writer`, `label`, `points` (x, y, stroke index, timestamp, tilt x, tilt y, pressure) |
| **CASIA-OLHWDB 2.0–2.2** | `https://nlpr.ia.ac.cn/databases/handwriting/Home.html` · download `https://nlpr.ia.ac.cn/databases/handwriting/Download.html` | Free for academic research under an agreement (application form on the site) |
| SCUT-COUCH Textline_NU | `http://www.hcii-lab.net/data/scutcouch/` (English: `/EN/`) | Free for research with a signed letter of commitment; US$2,000 commercial |
| OLIWER (derived phrases) | `https://github.com/SCUT-DLVCLab/DOLPHIN` | Code + build scripts; you must obtain its three source corpora yourself |
| DCOH-E (extended) | via the DOLPHIN release above | same as DCOH |

### Offline (images) — natural cursive without trajectories

| Dataset | Link | Access |
|---|---|---|
| **HIT-MW** | `https://code.google.com/archive/p/hit-mw-database` · `http://hitmwdb.blogspot.com/` · NIST catalogue `https://tsapps.nist.gov/BDBC/Search/Details/548` | Documents/textlines free (non-commercial); **textual ground truth is charged/on request**; authorisation letter needed to publish |
| CASIA-HWDB 2.0–2.2 (line images) | same NLPR site | research agreement |
| CASIA-HWDB2-line (pre-cut, HF mirror) | `https://huggingface.co/datasets/Teklia/CASIA-HWDB2-line` | Ready-to-load line images + transcripts; **check the mirror's terms against the original agreement** |
| SCUT-EPT | `https://github.com/HCIILAB/SCUT-EPT_Dataset_Release` | Non-commercial; application + publications; training archive is encrypted |
| SCUT-HCCDoc | *no verified URL* — request via HCII Lab, SCUT (paper: *Pattern Recognition* 108, 2020, doi `10.1016/j.patcog.2020.107559`) | on request |
| CASIA-HWDB-T (touching 2+-char strings) | *no verified URL* — request via NLPR/CASIA (ICFHR 2012 paper) | on request |

### Cursive-script material (historical / calligraphic)

| Dataset | Link | Access |
|---|---|---|
| **EvaHan 2026 Dataset C** (kaishu + xingshu line images) | `https://circse.github.io/LT4HALA/2026/EvaHan.html` · `https://github.com/GoThereGit/EvaHan` | Registration by e-mail `evahan2026@gmail.com`; registration closed 2026-01-30 — **ask the organisers whether the split can still be shared** |
| UniCalli | `https://huggingface.co/datasets/TSXu/UniCalli_dataset` · `https://github.com/EnVision-Research/UniCalli` · project `https://envision-research.github.io/UniCalli/` | Public download; research use; column-level calligraphy with transcriptions |
| Chronicles-OCR | `https://huggingface.co/datasets/jinichol/Chronicles-OCR` · `https://github.com/VirtualLUOUCAS/Chronicles-OCR` | Public; research only (400 images per script, incl. 行书 / 草书) |
| Ancient-Bench | `https://huggingface.co/datasets/hiuyi/Ancient-Bench` | Public; research use |
| UltraGlyph (synthetic lines) | `https://huggingface.co/datasets/ENC-PSL/evahan-ultraglyph` | Public; fonts + recombined glyphs, incl. Regular/Running/Cursive renders |
| Traditional-Chinese OCR synthetic (Lin 2025) | *name reported, exact repository id not verified* — search HuggingFace for the 4.1 M-line traditional-Chinese synthetic OCR set before citing it | public |

### One caveat that affects how you use each

CASIA-OLHWDB 2.x carries **per-character stroke indices and per-point character labels**, which is what makes boundary/cursive analysis possible. **DCOH-120K gives a line transcription + stroke index, but not character-level boundaries** — fine for text→line generation, less good for layout/boundary statistics. That is the one practical reason to use both rather than replacing CASIA with DCOH.

---

## 9. The find that matters: SCUT-COUCH documents 连笔 *across* characters

The wide search turned up the single most useful piece of evidence in this whole survey, and it was sitting in a 2011 IJDAR paper that nobody in the generation literature cites for this purpose. From the SCUT-COUCH2009 description (`http://www.hcii-lab.net/data/scutcouch/download/SCUT-COUCH2009-A Comprehensive Online Unconstrained Chinese Handwriting Database and Benchmark Evaluation.pdf`), verbatim:

> "**many Chinese characters are written cursively, with two or more strokes connected**, since the average stroke number of the handwritten samples is much less than that of standard characters."

> "**some Chinese words are even cursively written in one stroke.**"

> "**Over-connected stroke:** In the writing process, **even some strokes in different characters were written connectively**. Fig. 12(f) shows over-connected-stroke samples '主题', '运行', '朋友' …"

That is a published, 2011, official description of **cross-character connection in a public online Chinese dataset**. It contradicts nothing in CASIA; it simply shows the two corpora are not interchangeable.

**And the supervision is annotated.** The same paper: "we segmented the 8,888 Chinese word samples into single characters and manually labeled each character … record the **bounding box of every single character** as well as the **indices of the strokes in each character**" — explicitly "to offer information on both the **spatial and temporal relationships between character members** of a Chinese word."

Why this matters for CENT, concretely:

| Need | Does SCUT-COUCH provide it? |
|---|---|
| Chinese characters in sequence | Yes — Word8888 / Word17366 / Word44208 (2–4 character words) |
| Trajectories | Yes — online, Unipen-style strokes |
| **Boundary labels** | **Yes** — per-character stroke indices mean "did a stroke span two characters" is directly computable, no heuristic |
| Writer identity | Yes — >190 writers |
| Cursive at the boundary | **Yes — documented by the authors, with example words** |
| Full lines | Yes — TextLine_NU, 8,809 lines, explicit `.PEN_DOWN` / `.PEN_UP` |
| Unconstrained register | Yes — collected on PDAs/phones, explicitly "unconstrained" |

**Caveats, stated plainly.** The word data was captured on PDAs/smartphones, so sampling is coarser than CASIA's Anoto pen, and stroke capture can fail on very fast strokes (the paper documents "disconnected stroke" artefacts). The corpus is word-level rather than paragraph-level, and it is ~2011 hardware. None of this disqualifies it for learning *boundary dynamics*, but the paper must disclose it, and glyph-quality comparisons should still be reported on CASIA.

### Other candidates the wide search added

| Dataset | What it gives | Link |
|---|---|---|
| **CASIA-HWDB-T** (touching characters) | **Public download**, no application: 50,157 all-Chinese touching 2+-character strings, each annotated with character classes **and the touching points**. The perfect hard-case evaluation set for a boundary model | `https://nlpr.ia.ac.cn/databases/handwriting/Touching_Characters_Databases.html` |
| **CASIA-onDo** | 2,012 online **documents**, 841,159 strokes, 200 writers, **pressure + pen-state + time**, InkML, 5,046 text lines. Real documents, not transcribed news | `https://nlpr.ia.ac.cn/databases/CASIA-onDo/index.html` (application form) |
| **CASIA online writer-ID DB** | 187 Chinese writers, sentences ~50 words, **x, y, timestamp, button status, azimuth, altitude, pressure** — richer per-point features than OLHWDB 2.x | `http://biometrics.idealtest.org` (registration) |
| **OpenHandWrite / stroke-level DB (2026)** | 42 speakers × 1,200 characters, 200 Hz Wacom, stroke-level; **public on OSF**. Character-level, but clean modern trajectories | `https://osf.io/rn2ck/` (paper: *Behavior Research Methods* 58(6):162, 2026) |
| **Mobao** (calligraphy, 2024) | 1,929,393 calligraphy images, 6 scripts incl. **running script = 668,040 images**, 2,681 calligraphers, 4,660 characters | described in *Moyun* (arXiv 2410.07618) |
| **Arabic strictly-cursive corpora** (transfer) | altecOnDB: **1,001 writers, 31,124 lines**, script is cursive by definition; ADAB (173 writers); Calliar (2,500 online sentences, stroke/char/word/sentence annotation) | see AltecOnDB (arXiv 1412.7626) and Calliar (NCAA 2022) |

### The one prior-art flag from this round

**Chen, Lin, Jian, Shen, Chen, "Data-driven Handwriting Synthesis in a Conjoined Manner", *Computer Graphics Forum* 2015** (NTU Taiwan) computes a **"cursive probability" for every character pair** — from the pair's observed conjoined frequency, backed off to start/finish-position groups — and then **jointly optimises glyph shape and the connecting trajectory** so that a character is deformed *in order to* connect, plus paragraph layout imitation. It is not a neural generator, but the concept "learn per-pair connection probability, then jointly optimise the join" predates CENT by a decade and is Chinese. **CENT must cite it and differentiate on learning, style conditioning, and scale** — not on the idea of a learned connection decision.

A Microsoft patent (US 7,983,478) formalises the same mechanism as **"imaginary strokes"**: connect every pair of consecutive real strokes into one continuous trajectory and add a **binary connection/no-connection indicator**. That is the oldest statement of the problem, and it is worth one sentence in related work.
