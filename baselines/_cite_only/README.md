# _cite_only — Chinese-capable or not, these CANNOT be a working baseline

A method lives here when it cannot run a real experiment on Chinese text lines, even
with unlimited compute. It still appears in the paper, cited with published numbers and
marked *not re-run*. It is kept here (not deleted) because the audit that established why
it cannot run is itself evidence, and because a release could change the verdict.

| Folder | Why it is not a working baseline |
|---|---|
| `onedm` | **Chinese-impossible.** The released loader hard-codes an 81-character Latin alphabet (`data_loader/loader.py:25`), so any Chinese transcript raises `KeyError` before a tensor exists; and no Chinese checkpoint was ever published — the README promises one "soon" while the paper shows Chinese figures. The adapter, alphabet patch and composed-runner wiring are all present and tested; they simply have nothing to drive. |

Everything else in `baselines/` is a baseline we intend to run: DiffBrush, TGC-Diff,
DiffInk, OLHWG (native line models) and SDT, MetaScript (Chinese character models, composed
into lines by our composer).

**Correction (2026-11-22).** OLHWG was briefly listed here on the false conclusion that it
released no weights. It does: `unetddpm450000.pth` and `style_encoder450000.pth`, plus the
pre-processed line data, sit on the Hugging Face repository `Immortalman12/OLHWD`, whose card
declares MIT. It has been moved back to `baselines/olhwg/`. The lesson is recorded rather
than hidden: an audit that reads a repository README and not the file listing of the
repository it links to can produce exactly this error.
