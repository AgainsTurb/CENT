# Baseline set — change record

## Revision 3 (2026-11-21, task corrected to text lines)

**The task changed**: CENT now generates Chinese handwritten **text lines with cursive
inter-character connectivity**, not isolated characters. The baseline set was rebuilt
for the third time as a consequence.

### Archived (not deleted) — `baselines/_archive_character_level/`
GANwriting, HWT (ICCV 2021, not CVPR), VATr, DiffusionPen, MetaScript (arXiv 2312.16251, not AAAI 2024), SDT. All were adapted and all passed
their smoke tests (54, 25, 94, 1787, 17 and 13 checks). They are out of scope now: a
character or word generator has **no layout and no inter-character connectivity**, so
it cannot be compared to a line generator on connectivity metrics. Kept for a possible
character-level ablation and because the adapters are real work.

### Line-level set (cloned at pinned commits)

| Baseline | Modality | Commit | Licence | Adapter |
|---|---|---|---|---|
| DiffBrush (ICCV 2025) | offline lines | `da9addc1140b` | MIT | not yet written |
| TGC-Diff (ICDAR 2025 Best Paper) | offline lines | `4d4058d29a12` | MIT | not yet written |
| DiffInk (ICLR 2026) | online lines | `65055545fbef` | MIT | not yet written |
| OLHWG (ICLR 2025) | online lines | `e1bed55e0a66` | **no licence file** | not yet written |

OLHWG ships no licence file; that is recorded in the run sheet rather than assumed
permissive.

### Excluded
- **InkSpire** (ICLR 2026): public repo, one file, no source. Cite published numbers as not re-run.
- **One-DM** (ECCV 2024): word-level; used only as a style-transfer reference point.

### Evidence
All four cloned; commits pinned via `git rev-parse`; `ars_experiment plan-check` →
ok. The DiffBrush clone initially failed (`invalid index-pack output`) and was completed
with a blob filter plus `git reset --hard`, leaving 2,946 files.

---

## Revision 2 (earlier, superseded) — character-level set

The baseline set was rebuilt after a wide retrieval sweep (Exa + Tavily) for methods that
actually generate **Chinese offline handwriting** and actually ship **runnable code**. Two
baselines were removed, two were added, and four exclusions are now recorded in
`experiments/run_sheet.json` rather than quietly dropped.

## Removed

| Baseline | Commit | Why it was removed | Evidence |
|---|---|---|---|
| One-DM | `dde2205a70a2` | The released code is an **English word generator**. Four independent blockers: hardcoded 81-symbol vocabulary, `data/unifont.pickle` glyph atlas absent from the repo, English checkpoints, and a word-shaped canvas (`32·len(text)` px). Making it Chinese would make it a different method. | adapter report; `baselines/one_dm` (now removed) |
| HiSAC | `5e3e582d77b7` | Public repository contains **only `LICENSE`**. No source. | `git clone` → 1 file, 0 MB |

Both are now cited with their **published** numbers marked *not re-run*.

## Added

| Baseline | Commit | Why it qualifies | Notes |
|---|---|---|---|
| **MetaScript** | `7ec444ffac44` | Chinese-native, **offline**, **character-level**, few-shot from reference characters, trained on CASIA-HWDB-1.1, MIT | Closest in setting to CENT: isolated characters, not lines |
| **SDT** | `ba1da572c5c9` | Chinese-native, MIT, first-class Chinese config (`configs/CHINESE_CASIA.yml`), and a documented **offline** path ("offline-to-offline framework") | Online-first; the offline extension is the comparable branch |

## Kept (language-agnostic, already adapted)

GANwriting, HWT, VATr, DiffusionPen. They are Latin-origin but language-agnostic glyph
generators, already adapted to character-level Chinese. They are kept as a **separate family**
because they test a different question from the Chinese-native ones: whether a
structure-agnostic generator reaches the same place. One-DM's own Chinese experiment compares
against GANwriting/HWT/VATr, so the comparison is field-standard.

## Excluded, with reasons (not silently dropped)

| Work | Repo state | Reason |
|---|---|---|
| **InkSpire** (ICLR 2026) | public repo, 1 file, **no Python** | code not released; also text-line granularity |
| **TGC-Diff** (ICDAR 2025 Best Paper) | real code, 22 Python files, MIT | generates Chinese handwritten **text lines**, not isolated characters — not comparable on a per-character stroke metric |
| **DiffInk** (ICLR 2026) | real code, MIT, weights released | **online** trajectory generation (T2OHG), not offline images |
| **OLHWG / WriteLikeYou / HCCG-CycleGAN** | code exists | online-only (first two), or Torch7/Lua from 2018 (HCCG) — not runnable in this environment |

InkSpire, TGC-Diff and DiffInk are cited as related work; the granularity/modality difference is
stated in §II of the manuscript.

## Consequences carried into the paper

- §II.B now separates the **character-level** line from the **text-line** line and says why the
  latter are cited but not compared.
- §V.D lists the two baseline families and records all four exclusions.
- The run sheet's `baselines[]` holds six entries and `excluded_baselines[]` holds four, each with
  repo, commit and reason. `ars_experiment plan-check` → **ok, no findings**.

## What this does *not* fix

A Chinese-native offline baseline at the **same** granularity with a *hard* stroke constraint still
does not exist in the literature we retrieved — which is the paper's gap, and also the reason the
comparison set is two families rather than one. If that reading is wrong, the correction is a new
retrieval pass, not a new claim.

---

## Revision 3b (2026-11-22) — working-Chinese policy

Triggered by a rule from the author: **if a repo cannot generate Chinese, it is not a
baseline; cite it only.** `baselines/` now contains only methods that can produce Chinese
characters or Chinese lines.

### Removed from the working set (kept in `baselines/_cite_only/`)

| Method | Why | Evidence |
|---|---|---|
| One-DM (ECCV 2024) | **Chinese-impossible as released** | alphabet is an 81-character Latin string literal at `data_loader/loader.py:25`; a Chinese transcript raises `KeyError` before a tensor exists. The alphabet is not in the weights (no `nn.Embedding` in `models/`), so a patch is possible — but **no Chinese checkpoint was published**: the model zoo has one English IAM model and the README says Chinese is "released soon" while the paper shows Chinese figures. |
| OLHWG (ICLR 2025) | **Chinese-capable, unobtainable** | no licence file, no generator weights (only its HF *dataset* is linked), and `evaluate_line.py` raises before computing a metric. Its layout planner's weights are present, so that half is runnable. Move back if the authors release `unetddpm290000.pth` + `style_encoder290000.pth`. |

### Added to the working set

| Method | Mode | Why it qualifies | Weights |
|---|---|---|---|
| SDT (CVPR 2023) | composed | official implementation of a Chinese character generator, online trajectories | README links Google Drive + Baidu |
| MetaScript (arXiv 2312.16251) | composed | Chinese character generator, MIT, ships its own stitcher | download advertised; Baidu-only per third-party report — verify before relying |

Both are **character** models, so they enter the table as composed, assembled by our
composer (`benchmark/compose_line.py`, `benchmark/compose_trajectory.py`) with the layout
named. Their connector rate is **zero by construction** and is reported as a declared
property, not a measurement.

### Consequence worth stating in the paper

Once a Chinese-impossible model is out of the comparison, the reviewed claim gets sharper:
the remaining composed baselines *do* generate Chinese glyphs, so any gap in the
connectivity columns is a gap in connectivity, not in Chinese support.

### Also corrected

- `sdt` was recorded as an arXiv preprint; the authors' official repo states **CVPR 2023**,
  so the override now says CVPR with the arXiv id as a note (authors reused unchanged from
  the verified record).
- MetaScript's venue was wrong in our own change record (AAAI 2024); it is arXiv 2312.16251.

---

## Revision 3c (2026-11-22) — OLHWG restored

**Reversal of revision 3b.** OLHWG was moved to `_cite_only/` on the belief that it released
no weights. It does: `unetddpm450000.pth` and `style_encoder450000.pth`, plus the
pre-processed line data `all_datas.npy` and `dictionary.npy`, are on the Hugging Face
repository `Immortalman12/OLHWD`, whose card declares MIT.

- **Restored** to `baselines/olhwg/` as a runnable **ND**-mode baseline (weights + data to be
  downloaded on the server; code patches documented).
- **Not restored:** One-DM. That verdict is independent and still holds — the released loader
  hard-codes an 81-character Latin alphabet and no Chinese checkpoint was ever published. It
  stays in `_cite_only/`.
- **Adapter to be fixed:** its prerequisite check inherited `evaluate_line.py`'s stale
  `290000` filenames; the line path (`generate_line.py`) loads `450000`, which is what ships.

Why this is worth the patch: it converts OLHWG's own admission about cursive connections into
a measured connectivity number, which is exactly the comparison this paper exists to make.
