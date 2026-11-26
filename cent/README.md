# CENT — Chinese handwriting generation of text **lines**

Architecture implementation for the CENT paper target: full text-line Chinese
handwriting generation with (i) per-writer style mimicry, (ii) **free cursive
continuity between characters** instead of hard layout, and (iii) explicit
handling of the train/infer gap between clean character-level corpora and
cursive line-level inference.

Research conclusions live in `survey/`; this folder is the model code.

## Pipeline (run on the GPU box)

Everything is self-fetching: the data layer detects what is missing, downloads the
official WPTT archives (no password), verifies the zips, extracts, and caches
writer-disjoint splits. Every stage writes `metrics.json` **and** inspection PNGs,
because metrics alone are not an acceptance check.

```bash
python scripts/run_all.py                  # full pipeline, stage by stage
python scripts/run_all.py --smoke          # 2-step plumbing check (~5 s)
python scripts/run_all.py --stop-after stage3
```

| Stage | Script | Produces | Visual check to look at |
|---|---|---|---|
| 0 | `setup_data.py` | `datasets/CASIA-OLHWDB2.0/`, `datasets/processed/` | page/line counts |
| 1 (optional) | `train_vae.py --stage 1` | char-level VAE | `recon_*.png` |
| 2 | `train_vae.py --stage 2` | `vae.pt` | `recon_*.png`, `loss_curves.png` |
| 3 | `derive_boundaries.py` | `align.npz`, `join_stats.json` | `alignment_*.png` (guides must land between characters), `join_rate_per_writer.png` |
| 4/5 | `train_dit.py` | `dit.pt`, control curve | `samples_control.png` (does a join look like one stroke, or a bent glyph?) |
| 6 | `evaluate.py` | `metrics.json` | `generated_vs_real.png` (black = real, red = generated) |

Stage 1 skips itself cleanly when no `.pot` corpus is present. Gates print
PASS/FAIL per stage; `--strict` makes a failure stop the run, `--force` continues.

## Layout

```
cent/                     # the package
  data/
    format_wptt.py        # CASIA-OLHWDB 2.0 parser (byte-exact, self-checking)
    format_scut.py        # SCUT-COUCH readers (locked corpus; kept for reference)
    format_pot.py         # OLHWDB 1.x characters (optional stage 1)
    tokens.py             # deltas + 4-way pen state + boundary labels
    synthetic.py          # fixtures in the REAL formats (smoke tests)
    dataset.py            # Dataset, collate, frame<->char alignment
  models/
    inkvae.py             # sequence VAE: 8x latent, GMM-20 decoder, 4-way pen state
    boundary_dit.py       # BoundaryDiT: interleaved boundary slots  <-- the paper
    baseline_ar.py        # Option A: AR stream + CursiveEOC (ablation baseline)
    text_encoder.py       # character sequence embeddings
    diffusion.py          # DDPM schedule + DDIM sampler
  pipeline/               # the runnable stages
    fetch.py              # detect/download/verify/extract datasets
    cache.py              # WPTT -> lines.npz, writer-disjoint splits
    align.py              # CTC Viterbi -> point-level boundaries -> join labels
    stage_vae.py          # stages 1-2
    stage_boundaries.py   # stage 3
    stage_dit.py          # stages 4-5
    stage_eval.py         # stage 6
    run_all.py            # driver with gates + visual checkpoints
  viz.py                  # figures: matplotlib (CJK font auto-detected) with a
                          # dependency-free numpy fallback if mpl is absent
  losses.py               # L_vae, L_kl, L_ctc, L_style, L_diff, L_bnd
  metrics.py              # boundary F1, CRE, gap stats, DTW
  model.py                # CENTModel: assembly + generate(control=...)
  train.py                # legacy joint loop (superseded by pipeline/)
scripts/                  # thin wrappers: setup_data / train_vae / derive_boundaries
                          #                 train_dit / evaluate / run_all
configs/cent_tiny.yaml
datasets/README.md        # data status, capabilities per corpus, licences
TRAINING_PLAN.md          # datasets + staged procedure + gates
```

## The idea in one paragraph

A layout module is not needed: in trajectory space the layout *is* the
coordinate stream. The only thing prior work fails to model is the **boundary
event** — whether the pen lifts between two characters. CENT makes it explicit
twice over: as a 4-way pen state (`PEN_MOVE`, `PEN_UP`, `EOC`, **`CURSIVE_EOC`**)
and as a **boundary slot** inserted between character latents,
`[c1 | B1 | c2 | B2 | … | cn]`. The DiT denoises the whole line jointly, so the
boundary decision conditions the tail of character *i* and the head of
character *i+1* — which is exactly what "generate the glyphs, then join them"
(post-hoc Bézier / trace deformation) cannot do.

## Run

```bash
python scripts/smoke_test.py                     # verifies everything below
python scripts/inspect_dataset.py --fixture
python scripts/measure_boundary_stats.py --fixture --tau 0.06
```

`train.py` is complete but must not be run until the SCUT-COUCH password is
issued (see `datasets/README.md`).

## What the smoke test verifies

1. fixtures written in the real on-disk formats
2. Unipen and binary `.IDX/.DAT` parsers round-trip (incl. GB decode)
3. join/lift label derivation (normalised-gap rule + same-stroke rule)
4. Dataset/collate contract (masks, char alignment)
5. CENTModel forward → finite losses; backward → non-zero gradients
6. sampling runs and the cursive-control knob spans lift-only ↔ join-only
7. AR baseline (Option A) forwards
8. decoder latent → points
9. metrics compute

## Status / honesty notes

* **No training was run.** The smoke test performs one forward and one backward
  pass to prove gradients flow; no optimizer step, no checkpoint.
* **Real data is not yet readable**: the SCUT-COUCH archives are AES-encrypted
  and the password requires an approved application. All parsers are written
  against the official reader, so real data will load unchanged.
* The control knob's *mechanism* is verified (it moves the decision), but its
  *calibration* is meaningless on random weights — that needs training.
