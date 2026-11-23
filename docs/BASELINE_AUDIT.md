# Baseline audit — code, methods, and what each can actually run

Audited 2026-11-21 against the line benchmark (CASIA-HWDB2.x / OLHWDB2.x test lines,
AR/CR + FID + connectivity). Every claim below was verified by reading the cloned code and
running each adapter's smoke test; nothing was taken from a README where the code
disagreed.

| Baseline | Commit | Licence | Chinese lines as shipped? | Adapter | Verdict |
|---|---|---|---|---|---|
| DiffBrush | `da9addc1140b` | MIT | **no** — IAM-only loader | `adapt_casia_lines.py`, 25 checks PASS | reproducible **after** code change + weights |
| TGC-Diff | `4d4058d29a12` | MIT | **yes** — Chinese pipeline shipped | `adapt_casia_lines.py`, 10 checks PASS | reproducible, weights missing locally |
| DiffInk | `65055545fbef` | MIT | yes, but **prefix-conditioned** | `adapt_casia_lines.py`, 42/42 + 1 blocked | reproducible only with its release vocabulary |
| OLHWG | `e1bed55e0a66` | **none** | yes, but unshippable | `adapt_casia_lines.py`, 60/60 | **cite-only** |

## Cross-cutting finding: no baseline ships a metric

All four write images or trajectories and stop. TGC-Diff's `test_ch.py` emits PNGs;
DiffBrush's `generate.py` emits PNGs; DiffInk's `val_dit.py` emits trajectories; OLHWG's
`evaluate_line.py` cannot execute at all. **The benchmark scorer is therefore ours**
(`benchmark/score_lines.py`): AR/CR by DP edit distance over predicted transcripts, plus
FID and connectivity where the instruments exist. This is also why the paper can claim
that connectivity is unreported: the codebases contain no metric to report it with.

## DiffBrush (ICCV 2025)

- **Method:** latent diffusion over line images; content-decoupled style learning with
  row/column masking; line- and word-level discriminators.
- **What the code contains:** an IAM-only dataset (`data_loader/IAMDataset.py`), an
  80-character Latin alphabet, and English wikitext prompts filtered to 35–61
  characters. `generate.py` dispatches on the config filename containing `IAM`.
- **Important correction to my earlier reading:** the release **does** ship Chinese glyph
  data — `files/unifont.pickle` holds 55,659 glyph records, 20,992 of them CJK
  (U+4E00–U+9FFF), 16×16 boolean. The model consumes glyph *images*, never character
  ids, so Chinese support is a **code change, not a data gap**. The adapter supplies it
  (`data_loader/ChineseDataset.py`, `configs/Chinese_lines.yml`).
- **Untested here:** `generate.py` never ran (no `diffusers`, no checkpoint, no VAE, no
  GPU); the 64×1024 output shape is arithmetic, not measurement.

## TGC-Diff (ICDAR 2025 Best Paper)

- **Method:** cascaded — Layout-Diff models spatial style, Imitating-Diff imitates glyph
  style through a Template; two stages, two checkpoints.
- **Contract it reads** (reverse-engineered, with line citations in its `PATCHES.md`):
  `<data_root>/data_ch/<split>/<writer_dir>/<passage>/<stem>.png` plus a mirrored
  `data_temp_ch/.../<stem>_template.png`. Writer-directory naming is **asymmetric**:
  `train` uses `int(x)`, `test` uses `int(x[1:])`, so a test writer dir must carry a
  non-digit prefix or construction raises. Only `train`/`test` exist — there is no `val`
  branch, so our 60-writer validation subset must be presented as a `test` directory.
  `dataset.reference_num` is read and never used: the style pool is the writer's whole
  directory, one line drawn at random.
- **Blockers:** the `_template.png` files are a separate download produced by their
  Layout-Diff, which `test_ch.py` never runs, so templates are required inputs; the
  adapter synthesises them from a font and records that in a manifest so they can never
  be mistaken for real ones. Missing locally: the SD-3.5 VAE and `weight/weight.pt`;
  `nccl` also makes it Linux/CUDA-only.

## DiffInk (ICLR 2026)

- **Method:** latent diffusion transformer over pen trajectories; glyph- and
  style-aware; generates whole lines.
- **Contract:** HDF5 via `h5py`, one group per line with `point_seq [T,5]`,
  `char_points_idx`, `line_text`, `writer_id`. The 5-wide row is `(x, y)` plus a 3-class
  one-hot whose third column is **end-of-character**, not end-of-sequence.
- **Two findings that change how it may be reported:**
  1. **It is prefix-conditioned line completion, not text-only generation.** The style
     input is not a writer id; `infer_diffink` conditions on a clean 30 % prefix of the
     ground-truth trajectory and mixes it back in. That cannot be switched off
     (`prefix_ratio=0` still keeps one char). A "DiffInk text-only" number is not a thing
     the release can produce.
  2. `num_text_embedding` is **data-derived**, so our 3,755-class charset yields a
     different embedding width and `load_state_dict(strict=True)` fails. The release
     vocabulary (`All_zi.json`) is in their weight bundle, not the repo. Forcing our own
     vocabulary with `strict=False` leaves text conditioning untrained — **never report
     that as a DiffInk score.**
- **Blocked locally:** `h5py` is absent, so `import dataset` fails; the adapter ships a
  pure-stdlib HDF5 writer (`h5mini.py`) so the contract is verified without it, and
  flags that h5py acceptance is unverified.

## OLHWG (ICLR 2025) — cite-only

Three independent blockers, each sufficient:
1. **No licence file and no licence statement anywhere** in the tracked tree → all rights
   reserved. Do not redistribute, vendor, or embed it.
2. **No generator weights.** The character generator's checkpoints
   (`models/unetddpm290000.pth`, `style_encoder290000.pth`) are absent, and the
   4,055-row label space lives *inside* them, so a substitute vocabulary would silently
   mis-condition the model. The layout planner's weights **are** present
   (`bonibox_gen/boxlstm3000.pth`), so that half is runnable.
3. **The released evaluation entry point cannot execute**: `generate_line()` returns a
   scalar while `evaluate_line.py` unpacks two values, and
   `unconditional_generate_boxes()` is declared with zero parameters but called with
   five. Both proven by AST, independent of weights.

It is cited with its published numbers, marked not re-run.

## Consequences for the paper and the plan

- §V.D's baseline paragraph stays accurate, but "reproduced" must be qualified per
  system: DiffBrush after a code change, TGC-Diff when weights are fetched, DiffInk only
  with its vocabulary **and** as prefix-conditioned completion, OLHWG not at all.
- The run sheet now records a `reproducibility` field per baseline so a reader cannot
  mistake "cloned and adapted" for "ran and scored".
- The benchmark's AR/CR needs a frozen recogniser. The obvious candidate is the released
  Chinese HTR trained on this corpus (`Teklia/pylaia-casia-hwdb2`); until it is wired and
  its identity recorded in the ledger, AR/CR is reported `NOT_RUN`, never proxied.

---

# Revision 5 (2026-11-22): wider survey, mode taxonomy, One-DM verdict

Triggered by "four peer methods is not enough". Full survey in `PEER_METHOD_SURVEY.md`;
protocol alignment in `PROTOCOL_ALIGNMENT.md`.

## The mode taxonomy that the table now uses

| Code | Meaning | Why it matters |
|---|---|---|
| **N** | native line model | the only mode where connectivity can arise as a by-product |
| **ND** | native output, decoupled layout | connectivity impossible **by construction** |
| **C** | composed (glyph model + our composer) | comparable on glyph fidelity, never on connectivity |
| **NR** | not runnable here | cited with published numbers, marked not re-run |

OLHWG is **ND**, not N: its own report concedes it "encounters difficulties in mimicking
styles with extensive cursive connections between characters" because each character is
generated independently. That admission is our motivation, and the taxonomy is what stops
us from claiming a connectivity win over a model that cannot express connectivity at all.

## One-DM (ECCV 2024) — adapted, not runnable for Chinese

Verified in the clone at `dde2205a70a2`:

- the alphabet is an **81-character Latin string literal** (`data_loader/loader.py:25`);
  a Chinese transcript raises `KeyError` before any tensor exists;
- the alphabet is **not** in the weights (no `nn.Embedding` in `models/`), so extending it
  changes no checkpoint shape — the blocker is not architecture;
- **no Chinese checkpoint is published.** The model zoo has one English IAM model, and the
  README says Chinese is "released soon" while the paper shows Chinese figures;
- `test.py:98` also builds its canvas from the 3-D style tensor's width instead of height,
  producing a 384 px-tall canvas where training used 64. Patched, and recorded.

So the composed harness works and this baseline cannot populate it. The paper says
"released artefact is English-only; adaptation feasible, weights unavailable" rather than
quoting a number we did not produce. DiffBrush's Chinese comparison against One-DM therefore
cannot be reproduced from public artefacts, and we note that.

## Corrections to our own records

- **MetaScript** is arXiv 2312.16251, not AAAI 2024 (a course project, per a 2025 survey).
- **HWT** is ICCV 2021, not CVPR.
- **CSA-GAN** is TPAMI 2022, 44(12):8846–8860.
- **DiffPen = DiffusionPen** (same ECCV 2024 method); do not list both.
- **TS-GAN: CC BY-NC-SA 4.0** (non-commercial) and Latin-only; **CSA-GAN has no released
  code**. Both are cited, neither is run, and the paper says why in one sentence.

## Two protocol traps now documented

1. **AR/CR and D_AR/D_CR run in opposite directions.** AR/CR = a fixed, real-trained
   recogniser applied to *generated* lines. D_AR/D_CR = a recogniser *trained on generated*
   lines, tested on *real* ones. They never share a column.
2. **The literature uses three incompatible Chinese-line splits**: 816/203 (DiffBrush),
   1,000/200 (OLHWG), and train-HWDB2 / test-ICDAR-2013. We adopt 816/203 and state the
   conflict rather than silently picking the flattering one.

---

# CORRECTION (2026-11-22): OLHWG does release its weights

This document, and `docs/BASELINE_CHANGES.md` revision 3, recorded OLHWG as
**cite-only** on three grounds. Two of them were wrong. The error and its cause are kept
here rather than edited away, because the same mistake could be made again.

## What was wrong

| Claim I made | Reality | How I got it wrong |
|---|---|---|
| "no generator weights" | **`unetddpm450000.pth` (164,622,792 B) and `style_encoder450000.pth` (31,760,219 B) are published** on `huggingface.co/datasets/Immortalman12/OLHWD`, together with the pre-processed **text-line** data `all_datas.npy` (1,676,792,545 B, from CASIA-OLHWDB 2.0-2.2) and `dictionary.npy` (61,268 B). | I read the repository README, saw it call that URL a *dataset* page, and treated it as data-only. I never listed the files inside it. |
| "no licence" | The HF **dataset card declares `license: mit`** (`tags: ["license:mit"]`, `cardData.license = "mit"`). The **code** repository still has no LICENSE file, so the two artefacts carry different statements. | I checked the GitHub tree and grepped for licence text there, which was correct but incomplete: it says nothing about the released weights and data. |

The third claim survives: **the released evaluation entry point still raises**. And a detail
that the earlier audit half-caught is now decisive — `generate_line.py:280,284` already loads
the **450000** pair, i.e. the line-generation path expects exactly what was released, while
`evaluate_line.py:32,35` still names `290000`, a stale filename. My adapter inherited the
stale name and therefore reported "checkpoints missing" even against the real release.

## Consequence

OLHWG moves from **cite-only** to a **runnable ND-mode baseline**: a decoupled-layout online
line generator whose weights, line data and licence exist, and whose evaluation entry point
needs the two patches already established by AST (`generate_line` returns a scalar that
`evaluate_line` unpacks as two values; `unconditional_generate_boxes` is declared with zero
parameters but called with five).

That matters more for this paper than for most comparisons. OLHWG is the closest
decoupled-layout peer, and its own report concedes that independently generated glyphs cannot
carry "extensive cursive connections between characters". With its weights in hand, that
concession becomes a **number we can measure** — connector rate and pen-lift agreement close
to zero by construction — instead of a sentence we quote.

## Process lesson

An audit that reads a README and not the *file listing* of the repository that README links
to is not an audit. The check that would have caught this is one API call
(`/api/datasets/<id>/tree/<rev>`), and it is now part of the checklist in
`docs/PROTOCOL_ALIGNMENT.md`.
