# Archived: character-level baselines (out of scope after the 2026-11-21 correction)

These six repositories were cloned and adapted when CENT was framed as an **isolated-character**
generator. The task is now **Chinese text-line generation with cursive connection**, so a
character generator is not a comparable baseline: it emits one glyph with no layout and no
inter-character connectivity.

Kept rather than deleted because the adapters are real work (~2,000 passing checks) and may be
relevant again for a character-level ablation. Do NOT run them as line-level baselines.

| Repo | Why it does not fit the line task |
|---|---|
| ganwriting, hwt, vatr, diffusionpen | word/character image generators; no line layout, no connective strokes |
| metascript | character-level Chinese, offline; no connectivity |
| sdt | character-level (online + offline extension); no inter-character connector model |

Line-level baselines live in aselines/: DiffBrush, TGC-Diff, DiffInk, OLHWG.
