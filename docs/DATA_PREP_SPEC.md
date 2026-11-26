# Data preparation specification (CENT v5, DiffInk-style)

_Purpose: turn the existing `canonical_lines_v1` corpus into one packed array file that a
training job can memmap. No new external data is required for this step._

## 1. What the trainer needs per line

One record per line, fixed length `P = 384` points:

| field | shape / dtype | content | source |
|---|---|---|---|
| `xy` | [N,384,2] f16 | pen trajectory, per-line normalised to [0,1] | `reconstruct(cont)` of the 78-stroke sequence, resampled to 384 |
| `pen` | [N,384] u8 | 0 = pen down, 1 = pen-up at a stroke end | stroke boundaries |
| `text` | [N,32] i16 | character ids | `item["text"]` |
| `text_len` | [N] i16 | number of characters | `item["n_chars"]` |
| `writer` | [N] i16 | writer id | `records[i]["writer"]` |
| `boxes` | [N,32,4] f16 | per-character (x, y, w, h) | char spans in the trajectory |
| `n_chars` | [N] i16 | characters with strokes | `item["n_chars"]` |
| `join` | [N,31] u8 | 1 if the pen did not lift at the boundary | P3(end of char i) to P0(start of i+1), canvas units, threshold `cfg.join_tau` |
| `gap` | [N,31,2] f16 | boundary displacement vector, normalised | same measurement |
| `refs` | [N,5] i32 | 5 lines by the same writer, excluding self | built once, deterministic |
| `split` | [N] u8 | 0 train / 1 val / 2 test, **by writer** | 400/50/50 writers |

Total size at N = 30,302: **≈ 70 MB** (xy 46 MB, pen 12 MB, boxes 8 MB, rest < 4 MB).

## 2. Procedure

1. **Enumerate** every usable line exactly once (`StrokeDatasetV3` with `limit=None`), and record
   `stable_id = (writer, page, line)` for provenance.
2. **Per line**: build the 78-stroke payload, resample to 384 points, compute `pen`, character
   spans, boxes, joins and gaps — all in **canvas units before normalisation**, then normalise.
3. **Reference indices**: for each line, 5 others from the same writer (skip writers with < 6 lines).
4. **Split by writer**, never by line (a line-level split leaks a writer's style into test).
5. **Write** one HDF5 with chunking (`chunks=(1024,384,2)`) and gzip level 4; one `.json` sidecar
   with counts, source sha256, script git hash, and the join-label statistics.
6. **Validate before shipping** — fail loudly:
   - count check: every source line appears once, `n_chars` matches the source;
   - resampling check: `max |resampled - original|` under 1 px at 1024 wide;
   - a 5-line visual sheet (GT vs resampled) written beside the file;
   - `mean(join)` reported, with the histogram of `‖gap‖` — this is the number that decides whether
     the boundary module is learnable (it was **0.018** in the last run, i.e. almost all "not joined";
     that must be explained before training, not after).

Expected build time: 1–2 h single-core, ~15 min with 8 workers.

## 3. How much data for the full pipeline

| stage | lines needed | we have |
|---|---|---|
| VAE must fit (reconstruction row legible) | ≥ 10,000 | **30,302** usable |
| VAE good | 20,000–30,000 | 30,302 |
| Diffusion over latents | 10,000–30,000 | 30,302 |
| per-character coverage | — | median 83 occurrences/char; 99.8 % of tokens from chars seen ≥ 20× |

So our corpus is **sufficient**; the limit is not volume but the 56 % annotation coverage per line
(`char_strokes` labels only the annotated prefix, measured and documented in `data_v3.py`).

## 4. What is NOT needed, and what would be

**Not needed from outside:** trajectories, text, writers, character spans, reference lines — all
already inside `canonical_lines_v1`. The obsolete `cache/` folder in the bucket was built for the
v3 token format and must not be reused.

**Only if we want more than 30k lines:**
- **CASIA-OLHWDB 2.0–2.2 full** — official request required; our current extraction covers 41,710
  stroke-bearing lines, and the remaining pages are not in our pipeline.
- **CASIA-HWDB 2.0–2.2 character images (~1M)** — would only be needed for a font/character prior;
  the census shows we do not need it for the dictionary.

DiffInk trains on ~67k lines (it augments with frequency-aware synthesis). At 30k we are within a
factor of two, which is acceptable for a first working CENT; augmentation is a later lever.

## 5. Training budget implied (for the record)

- VAE: 30k lines, batch 64, ~30–50 epochs → **≈ 15–25 min** on a 3090.
- DiT: latents [30k, 96, 64], batch 32, 100–200 epochs → **≈ 2–3 h**.
- Total **≈ 3–4 h ≈ $0.8** at $0.20/h, plus the one-time cache build.
