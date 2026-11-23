# CENT — Structure-First Chinese Handwriting **Line** Generation

**C**hin**E**se ha**N**dwri**T**ing: generate a whole handwritten **line** — character stroke graphs *and* the cursive connectors between them — not a bag of glyphs.

Inside a character, stroke count and order are fixed by a public stroke-order lexicon, so a
glyph cannot be the wrong character. Between characters, every boundary carries an explicit
**connector latent** — join or lift, and if joined, the stroke that carries the pen across.
That is what makes fast writing look fast. One latent decodes to a cursive trajectory and to
an offline line image.

> **Status: no empirical results exist. The line-level model now runs end to end on a synthetic fixture.**
> **Contribution, after the prior-art audit:** continuity at character boundaries as a
> construction (connector control points anchored to neighbour endpoints) rather than a
> pen-state supervision target; connection metrics that stay defined in the Chinese regime;
> and an image-only connectivity proxy validated against trajectories. `docs/CURSIVE_CONNECTIVITY_PRIOR_ART.md`.
>
> The manuscript reports a method, a feasibility analysis and a frozen protocol. What exists:
> line readers, a canonical line builder, CENT's line model (connector tokens, layout tokens,
> k-line style), three of four connectivity metrics, AR/CR from transcripts, the benchmark
> scorer, four adapted baselines, 66 core tests and ~140 adapter checks. What does not exist:
> any training on real data, AR/CR from images (no frozen HTR), and the speed metric (no
> timestamps). `DEVIATIONS.md` lists every gap.

---


## Layout

```
paper/                 manuscript (XeLaTeX, IEEE two-column), refs.bib, compiled PDF
phase1_scoping/        RQ brief, methodology blueprint, Devil's Advocate checkpoint
phase2_literature/     synthesis.md (character-level) + text_line_synthesis.md (the corrected map)
phase2_5_integrity/    Stage 2.5 gate report
experiments/           frozen run sheet (line-level) + harness spec, hash-bound
analysis/              lexicon feasibility study — the paper's only measured numbers
refs/                  reference verification: candidates, store, overrides, bib generator
common/                CASIA readers (.gnt/.pot/.dgrl/.wptt), canonical builders, lexicon, CLIs
cent/                  CENT: line model (connector/layout tokens, k-line style), 64x1024 canvas, train/eval
benchmark/recogniser.py     frozen HTR slot (Teklia/pylaia-casia-hwdb2); AR/CR stays NOT_RUN until calibrated
benchmark/compose_line.py   glyph -> line assembly (uniform vs style layout)
benchmark/metrics.py        FID/KID/HWD/AR-CR math; features are declared, not assumed
benchmark/connectivity.py   join F1, join-rate error, continuity error, offline proxy + validation
benchmark/run_composed.py   run a character/word baseline as a composed line
baselines/             four line-level repos cloned; _archive_character_level/ holds the old set
tests/                 66 synthetic-fixture tests (no corpus needed)
docs/RUNBOOK.md        the GPU-server workflow
DEVIATIONS.md          paper claims vs implementation, gap by gap
```

## Quick start (no corpus needed)

```bash
python tests/test_casia.py             # 14 checks: character-level .gnt/.pot layout
python tests/test_canonical.py         # 16 checks: character canonical builder
python tests/test_casia_lines.py       # 21 checks: .dgrl/.wptt readers + .pot size-field discovery
python tests/test_canonical_lines.py   # 15 checks: line canonical builder, PNG round-trip, connectors
```

Then the line model end to end, on the fixture, on CPU:

```bash
python -m cent.train_lines --config cent/configs/line_smoke.json --dry-run --device cpu
python -m cent.train_lines --config cent/configs/line_smoke_train.json --device cpu
python -m cent.eval_lines  --config .fixture/runs/line_smoke_train/config.json \
    --ckpt .fixture/runs/line_smoke_train/ckpt_last.pt --split val
```

Expected: 144 checks pass. They prove the readers are internally consistent on fixtures
built to the published tables — **not** that the real corpora parse. The line model runs
and reports connectivity, with `NOT_RUN` for speed/AR/CR/FID.

## Full pipeline (GPU server, corpus required)

```bash
# 1. probe before parsing: a mis-framed record yields plausible garbage
python -m common.probe_lines dir /data/casia/HWDB2/Gnt2.0 --kind dgrl
python -m common.probe_lines dir /data/casia/OLHWDB2/Pot2.0 --kind wptt
python -m common.probe dir /data/casia/OLHWDB1 --kind pot        # character-level, 2-byte header

# 2. build the line canonical set (images + transcripts + inferred connectors)
python -c "from common.canonical_lines import build_lines; from common.lexicon import load_lexicon; \
build_lines('/data/casia/HWDB2/Gnt2.0','/data/casia/OLHWDB2/Pot2.0','data/canonical_lines/v1', \
lexicon=load_lexicon('data/lexicon/graphics.txt'))"
```

See `docs/RUNBOOK.md`. **No GPU is rented without the author's permission**, and nothing
here downloads the corpus.

## What the code refuses to do

- `common/canonical_lines.py` marks every per-character stroke split as **inferred**, and
  reports connector counts only with that caveat — the corpus does not store the split.
- `common/probe_lines.py` refuses to let a build follow a failed probe.
- `cent/eval.py` reports `NOT_RUN` for instruments it does not own rather than
  substituting a proxy.

## Baseline status

**Policy: `baselines/` holds only what can generate Chinese.** A method that cannot write
Chinese, or whose weights cannot be obtained, is cited with published numbers, marked *not
re-run*, and moved to `baselines/_cite_only/` — never deleted, because the audit that
established why is evidence.

| Baseline | Modality | Mode | Clone | Adapter |
|---|---|---|---|---|
| DiffBrush | offline lines | native | `da9addc` | yes, 25 checks |
| TGC-Diff | offline lines | native | `4d4058d` | yes, 10 checks |
| DiffInk | online lines | native | `6505554` | yes, 42/42; prefix-conditioned |
| SDT | online characters | composed | `dailenson/SDT` | yes, 36/36; weights via Drive |
| MetaScript | offline characters | composed | `xxyQwQ/metascript` | yes, 73/73; weights Baidu-only |
| OLHWG | online lines | native (decoupled layout) | `e1bed55` | yes, 60/60; weights + line data on HF (MIT card) |

Cite-only (`baselines/_cite_only/`): **One-DM** (81-char Latin alphabet literal in the
loader; no Chinese checkpoint ever released). Further cited rows in the paper: InkSpire (no source),
HCT-GAN (no code), TS-GAN / CSA-GAN (Latin or no code), Emuru (Latin), HiSAC and
Diff-Writer (no source).

## Data and licensing

- Stroke lexicon: Make Me A Hanzi `graphics.txt`, **Arphic Public License** (not CC-BY;
  the dictionary file is separately LGPL-3.0).
- CASIA corpora: not redistributed here; access is by application and the terms are
  respected — no mirror was used.
