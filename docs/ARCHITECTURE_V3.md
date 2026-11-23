# CENT v3 — architecture and module responsibility

Status: **smoke-tested on CPU, not trained.** No GPU was rented. A synthetic fixture
(`tools/make_fixture_corpus.py`) drives `tools/smoke_v3.py` through dataset -> collate -> M1-M7
forward -> all losses -> backward -> autoregressive decode -> render -> M8, and the trainer
entry point is exercised with `--dry-run`. Both pass with exit code 0 at 136/138 parameters
receiving gradient. Nothing has been trained on real data; the §5 gates still apply.

## 1. Why the architecture changed

CENT v2 was a diffusion model over **absolute** Bézier payloads with an auxiliary **join
classifier**, trained for ten runs. The scan of where it failed:

| hypothesis | test | outcome |
|---|---|---|
| parsers / canonical build | native parsers, real-file probes, 144 tests | sound |
| inverted image convention | ink/paper mismatch | fixed |
| timestep coverage | `T` 1000 -> 200 | not the limit |
| style conditioning | encoder pooled paper pixels, 0.05% writer differentiation | fixed (42%, writer loss 5.09 -> 1.11) |
| text-stroke alignment | hard local window added | no visible effect |
| image supervision | scan is a different writing session; IoU 0.008 even after bbox cropping | target made self-consistent (IoU 1.0), still flat |
| data volume | 8,631 -> 30,302 lines | falsified: diffusion 0.1093 -> 0.1005, image unchanged |

The formulation was the outlier, not the bugs: Cursive Transformer reaches realistic cursive
on **3,500 words** autoregressively, while 30,302 lines of diffusion produced nothing legible.
v3 switches to autoregressive stroke generation with a structural continuity contract.

## 2. Module table

| # | Module | File | Input | Output | ~params |
|---|---|---|---|---|---|
| M1 | TranscriptEncoder | `cent/models/v3.py` | char ids | `H ∈ R^{C×D}` | 2M |
| M2 | StyleEncoder | `cent/models/style.py` (ink fix) | K reference lines | `z_g, z_l, z_κ` | 3M |
| M3 | LayoutPlanner | `cent/models/v3.py` | `H`, `z_g` | boxes `(h,w,y_c,Δx)` | 0.5M |
| M4 | JoinPlanner | `cent/models/v3.py` | pair `(c_i,c_{i+1})`, boxes, prior, `z_κ` | join logit + exit/entry anchors + bridge control | 1M |
| M5 | StrokeDecoder | `cent/models/v3.py` | prefix strokes, `H` | relative payload (8) + pen + stop | 20M |
| M6 | gaussian_memory_mask | `cent/models/v3.py` | stroke→char index | additive text mask | 0 |
| M7 | WriterHead | `cent/models/v3.py` | `z_g` | writer logits | 0.1M |
| M8 | refine_boundaries | `cent/models/v3.py` | line + joins | C¹-smoothed line | 0 |

Generator (M1–M6) is ~27M parameters, sequence length ~100 stroke tokens per line.

## 3. Responsibility matrix

| task aspect | module | explicitly not |
|---|---|---|
| what to write (content) | M1 | M2, M3, M5 |
| who writes it (style) | M2 | M1, M3, M4 |
| writer identification (supervision) | **M7** | M2 (encodes only) |
| where characters sit (spacing, slant, baseline rhythm) | M3 | M4, M5 |
| **whether two characters connect, and through what bridge** | **M4** | M1, M3, M5, M7 |
| stroke geometry, order, intra-character structure | M5 | M3, M4 |
| pen-path continuity (no jumps) | **M5, structural** | — |
| which character a stroke belongs to | **M6** | M1, M5 |
| C¹ smoothness at generation | M8 | M5 |
| legibility measurement | frozen HTR (external) | the generator |

### The three senses of coherence, kept separate

| sense | mechanism | module |
|---|---|---|
| pen-path continuity | relative offsets + pen-state | M5, structural |
| inter-character connection | explicit join decision + boundary contract | M4, planned |
| line-level coherence (rhythm, drift) | autoregressive layout with style context | M3, learned |

v2 collapsed all three into one join-rate number. That is why a v2 checkpoint could match the
corpus join rate (0.1404 vs 0.1333) while rendering a smear: the metric was satisfiable by
collapsing strokes into a band.

## 4. The boundary contract

This is the claim, in one formula. For boundary *i*:

```
join_i  ~ Bernoulli(σ(M4(H_i, H_{i+1}, B_i, B_{i+1}, P(c_i,c_{i+1}), z_κ)))
gap_i   = entry_i - exit_i                     (M4)
start   = exit_i + gap_i * (1 - join_i)        (forced on the first stroke of char i+1)
```

If `join_i = 1` the start offset is **exactly zero**; if `join_i = 0` the stroke begins at the
planned lift distance. Continuity is a coordinate the decoder must hit, not a label it guesses.
This replaces v2's join classifier, whose F1 (0.153) was below the always-join baseline (0.339).

## 5. Staged build and falsification gates

| stage | build | gate | cost |
|---|---|---|---|
| S0 | tokeniser only | decode(encode(line)) must re-render bit-identically | CPU, minutes |
| S1 | M1 + M5 + M6 | **overfit one line**: loss -> ~0 and a legible sample. v2 never passed this (0.090 at best) | ~30 min |
| S2 | + M3 | box L1 comparable to OLHWG's reported ∇₁–∇₈ | ~1 h |
| S3 | + M4 | join-rate error on **sampled** output beats 0.204 (lift) / 0.796 (join) | ~1 h |
| S4 | + M8 + HTR | AR/CR on generated lines; Gaussian-vs-uniform mask ablation | ~2 h |

S1 is the gate. If it fails, the representation is wrong and no amount of GPU time fixes it —
that is the single transferable lesson of v2.

## 6. Files added or changed

| file | role |
|---|---|
| `cent/config_v3.py` | v3 config; unknown keys rejected |
| `cent/models/v3.py` | M1, M3–M8 |
| `cent/data_v3.py` | per-char grouping, relative payloads, boxes, joins, priors |
| `cent/train_v3.py` | staged trainer, `--stage s1|s2|s3`, `--limit`, `--dry-run` |
| `benchmark/sample_v3.py` | AR decoder with the boundary contract, contact sheet |
| `tools/build_pair_prior.py` | per-pair join prior for M4 (codepoint-keyed) |
| `docs/ARCHITECTURE_V3.md` | this file |

Unchanged and still valid: parsers (`common/casia_native.py`), the canonical corpus, the
rasteriser, the metrics, and `tools/sync_box.py`.

## 7. Known gaps before S1 can be trusted

1. `data_v3` requires `char_strokes` in the corpus; the drop counters report coverage. If
   coverage is low the dataset will be small and that must be checked before training.
2. `_sample_mdn` uses a diagonal Gaussian mixture; correlated control points may need a fuller
   covariance if samples look jittery.
3. The sampler decodes character by character but the teacher-forced training pass sees the
   whole line; a per-character termination test (stop BCE) is the only guard against runaway
   generation, and it is unverified.
4. `plan_joins` is fed a constant 0.2 prior at inference in `sample_v3`; it should be fed the
   same per-pair table used in training.

## 8. What the smoke test caught (and why it mattered)

The first execution of this code found eight defects, four of which were silent:

| # | defect | consequence if shipped |
|---|---|---|
| 1 | M5 cross-attended to raw **text ids** instead of the M1 encoding, and the mask was one column too wide | trains, produces nothing meaningful; the alignment module would have been inert |
| 2 | `z_κ` was the base encoder's feature tensor (wrong style class instantiated) | M4 crashed immediately |
| 3 | M4's input width computed as `2*dim+12` instead of `3*dim+9` | M4 crashed |
| 4 | per-sample boundary tensors had **variable width** | batches misaligned silently across samples |
| 5 | `min()` on numpy 0-d arrays | dataset raised `ValueError` |
| 6 | collate rejected numpy `refs` from the base dataset | crash |
| 7 | **`z_l` and `style_tokens` received no gradient at all** | local style conditioning was dead weight; found only by counting parameters with gradients |
| 8 | sampler fed a constant 0.2 prior of the wrong width | crash |

Items 1, 4 and 7 are the instructive ones: none of them would have raised an error. A model
that trains but whose alignment path is inert, or whose batch is misaligned per sample, looks
like "the model needs more epochs" -- which is precisely the trap ten v2 runs fell into.

The gradient census is now part of the smoke test, and it is cheap: `sum(p.grad is not None)`
over named parameters. Any future component that nothing consumes is reported by name.


## 12. M3' — soft anchors (replaces the box planner)

**Rule, stated once: an anchor is a conditioning signal, never a constraint.**

`plan_layout` now emits five numbers per character, `a_i = (x, y, w, h, dx)`, in canvas units.
They are soft: no rect is handed to the decoder, strokes are never clipped, and the boundary
contract `start(next) = exit_i + gap_i (1 - zeta_i)` is unchanged.

`memory_and_mask(H, tmask, zg, stage)` builds the neighbourhood
`(a_{i-1}, a_i, a_{i+1})`, projects it with `anchor_ctx` (15 -> dim) and **adds** it to the
transcript memory. Strokes therefore see where their character sits and where its neighbours
sit, and may still overshoot all three, which is what keeps cursive connection expressible.

### Why not OLHWG's factorisation

OLHWG uses `p(x_i | c_i, X_ref, Box_i)`: conditional independence given the box. That
independence is the reason a join cannot arise, and they say so themselves. CENT keeps the plan
and drops the independence: the connector is generated *between* two anchors (M4), not inside
either glyph.

### Objective

```
L = L_stroke + L_pen + L_stop + L_abs + L_writer
  + lambda_layout * ||a - a_hat||_1
  + lambda_cont   * ||Delta^2 a_hat||_1        # anchor jerk: handwriting advances smoothly
  + lambda_count  * CE(count)
```

`lambda_layout` defaults to **1.0** (it was 0.0 in every previous run, i.e. layout was never
trained). `lambda_cont` defaults to 0.5. The continuity term penalises jerk in the *anchor
sequence*, not the stroke shapes, so it smooths global placement without flattening glyphs.

### Verified locally (CPU smoke, `tools/smoke_v3.py`)

- SMOKE PASS, all seven stages.
- `anchor_ctx` and `layout_head` now receive gradient; 144/146 parameters trainable (the two
  without gradient are the auxiliary `style.writer_head`, as expected when the writer loss is
  off). An earlier patch rewrote only the probe decoder call, leaving the real loss path
  anchor-free and `anchor_ctx` gradient-starved - the same "inert conditioning" failure that
  made style a no-op, caught here by the gradient census rather than by a loss curve.
- Legacy `benchmark/sample_v3.py` maps the 5-dim anchors back to the 4-dim box parametrisation
  the M4 join path was trained on; `benchmark/decode_v3.py` is the current decoder.
