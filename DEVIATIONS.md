# DEVIATIONS — paper claims vs. what the code actually does (line-level, revision 4)

Updated 2026-11-22 after aligning the protocol to the two most relevant peer papers
(`docs/PROTOCOL_ALIGNMENT.md`) and adding the composed-line comparison mode. Nothing here
is softened: a row says NOT IMPLEMENTED only if nothing runs.

| # | Paper says | Code does | Status |
|---|---|---|---|
| 1 | Line = character stroke graphs + explicit connector latents | `cent/models/line.py::LineDenoiser` + `cent/data_lines.py` build the interleaved sequence; the join decision is a real binary head. | **IMPLEMENTED**, fixtures only |
| 2 | Stroke count/order fixed per character by a lexicon | `common/lexicon.py`; the line dataset pads/truncates to the lexicon count. | IMPLEMENTED |
| 3 | Join/lift supervised by an inferred split | `common/casia_lines.infer_char_strokes` (declared, conservative); `stroke_char` persisted so the flag is trainable. | IMPLEMENTED (data side) |
| 4 | Reads `.dgrl` and `.wptt` | `common/casia_lines.py`, 21 tests; `.pot` size field discovered (2 or 4 bytes). | IMPLEMENTED |
| 5 | One latent decodes to trajectory and line image | `decode_trajectory` + `BezierRenderer`; line canvas placed by the generated layout. | IMPLEMENTED |
| 6 | Style from k reference **lines** | `LineStyleEncoder` over k line images: global, per-type, connector style. | IMPLEMENTED |
| 7 | Connectivity metrics that survive the Chinese regime | `benchmark/connectivity.py`: join F1, join-rate error (defined with zero positives), gap similarity, continuity error, and an image-only proxy with a validation routine. Speed profile NOT_RUN (no timestamps). 25 checks. | IMPLEMENTED |
| 8 | Baselines: native + composed, grouped | `baselines/` holds only Chinese-capable methods: DiffBrush, TGC-Diff, DiffInk (native) + OLHWG (decoupled layout) + SDT, MetaScript (composed), each with an adapter and a passing smoke (25/10/42/36/73 checks). Only One-DM is excluded, as Chinese-impossible. | IMPLEMENTED as adapters; no weights downloaded yet |
| 9 | AR/CR by DP string matching | `benchmark/metrics.py::ar_cr` verified (hand-built operation counts, then corpus aggregate); `benchmark/score_lines.py` applies it. From **images** it is NOT_RUN without a frozen recogniser. | **PARTIAL** |
| 10 | FID, KID, HWD | `benchmark/metrics.py` implements all three as real math (Fréchet closed-form test, MMD growth test, HWD shift test — 15 checks). **Features are not computed**: no Inception, no handwriting VGG16. | IMPLEMENTED (math), NOT_RUN (features) |
| 11 | 64×1024 padded lines | `common/canonical_lines.py` pads and **raises rather than crops**; cropping would desynchronise image and transcript. | IMPLEMENTED |
| 12 | Content images shared by all offline baselines | `common/content_render.py` renders from the 55,659-glyph Unifont table DiffBrush ships (20,992 CJK). | IMPLEMENTED |
| 13 | User study, inference time | Not run; no code. | NOT IMPLEMENTED |
| 14 | No comparative results reported | True. Nothing trained on real data; every number in the repo is from a 24-line synthetic fixture. | UPHELD |

| 15 | Contributions repositioned after the prior-art audit | `docs/CURSIVE_CONNECTIVITY_PRIOR_ART.md`: the connection idea is prior art (Lin & Wan 2007; Wang et al. 2005; Chen et al. 2015; BoundInk 2026; UniCalli 2025). The paper now claims continuity-by-construction, Chinese-regime metrics, a validated offline proxy, and style-conditioned join rate --- and explicitly disclaims novelty for modelling connectivity. | CORRECTED |
| 16 | Plan decisions implemented (canvas 1024, max_chars 32, verifier, paired-only, real writer ids, CFG dropout, tolerant parser, published HTR slot) | Renderer is non-square and verified 64x1024; aspect preserved by one scale; `paired_only` counts what it drops; verifier loss logged as `legib`; `benchmark/recogniser.py` fails closed and names Teklia/pylaia-casia-hwdb2. 144 tests. | IMPLEMENTED |
## Verified by running, not by reading

- `cent.train_lines --dry-run` / one epoch / `cent.eval_lines` — four losses, checkpoint,
  sampling, connectivity metrics, explicit `NOT_RUN` rows.
- `benchmark/metrics.py` — 15 checks including closed-form Fréchet and HWD identities.
- `benchmark/run_composed.py --stub` in both layouts — 3 composed lines each, manifest
  flags STUB and refuses to emit a self-scored transcript.
- Core suite: 81 checks (14 + 16 + 21 + 15 + 15).

**The numbers these runs produce are fixture numbers and are not results.**

## Still missing before a real run

- A frozen Chinese line recogniser (AR/CR, D_CER/D_WER) and its own accuracy on real lines.
- Inception features (FID/KID) and the handwriting VGG16 behind HWD.
- Speed profile: needs raw pen timing, which the canonical layer does not store.
- The baselines' own checkpoints and VAEs.
- User study and timing runs.


Updated 2026-11-21 after implementing CENT's line model and auditing the four baselines.
Nothing here is softened: a row says NOT IMPLEMENTED only if nothing runs.
