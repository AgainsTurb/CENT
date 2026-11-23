# CENT — runbook for the GPU server (text-line task)

Everything here runs on a rented GPU box. Nothing in this repository downloads the corpus
or rents compute by itself.

> **Standing constraint.** No GPU is rented without the author's explicit permission.
> **Task:** Chinese handwritten **text lines** with cursive inter-character connectivity,
> style inferred from k reference **lines**.

---

## 0. What exists on the server vs. here

| Thing | Where | Notes |
|---|---|---|
| Source, readers, builders, CENT components | this repo | line readers + builder are done and tested |
| Line model (connector tokens, layout, k-line style) | **not written** | see `DEVIATIONS.md` rows 1, 6, 7, 9 |
| CASIA-HWDB2.x (offline lines) | server only | 52,230 lines, 1,019 writers, `.dgrl` |
| CASIA-OLHWDB2.x (online lines) | server only | 52,220 lines, `.wptt` |
| ICDAR-2013 lines | server only | 3,432 lines, 60 writers |
| Stroke lexicon | `data/lexicon/` | 30.8 MB, Arphic Public License |

## 1. Probe before you parse

A parser that mis-frames a record still produces line-shaped garbage. Probe first:

```bash
python -m common.probe_lines dir /data/casia/HWDB2/Gnt2.0 --kind dgrl --limit 10
python -m common.probe_lines dir /data/casia/OLHWDB2/Pot2.0 --kind wptt
python -m common.probe dir /data/casia/OLHWDB1 --kind pot      # character level, 2-byte header
```

Require `ok: true` on every file. The `.pot` probe matters: the earlier character reader
assumed a 4-byte size field and the supplier's table says 2 — `iter_pot_compat` discovers
the width, and the probe reports which held.

**Connector sanity check.** If `.wptt` lines never report more strokes than the sum of
their characters' lexicon counts, there are no connectors to learn and the paper's central
claim cannot be tested on that data. Check this before committing compute:

```bash
python - <<'PY'
from common.casia_lines import iter_wptt, infer_char_strokes
from common.lexicon import load_lexicon
lex = load_lexicon('data/lexicon/graphics.txt')
page = iter_wptt('/data/casia/OLHWDB2/Pot2.0/001-P01.wptt')
for ln in page.lines[:5]:
    counts = [lex.get(c, 0) for c in ln.text]
    groups, conns = infer_char_strokes(ln, lexicon_counts=counts if all(counts) else None)
    print(ln.text, len(ln.stroke_indices), counts, 'connectors=', len(conns))
PY
```

## 2. Build the line canonical set

```bash
python - <<'PY'
from common.canonical_lines import build_lines
from common.lexicon import load_lexicon
meta = build_lines(
    '/data/casia/HWDB2/Gnt2.0',          # offline pages -> line images + transcripts
    '/data/casia/OLHWDB2/Pot2.0',        # online pages  -> trajectories + connectors
    'data/canonical_lines/v1',
    lexicon=load_lexicon('data/lexicon/graphics.txt'),
)
print(meta['counts'], 'paired lines:', meta['paired_lines'])
PY
```

The build reports `paired_lines`: lines whose image and trajectory were joined on
(writer, page, line index) **with matching text**. Lines that do not pair keep their image
and lose their trajectory — they are still usable offline, and the number is reported
rather than hidden.

Then check the connector statistics the paper depends on:

- connector rate per writer (fraction of boundaries with an inferred connector),
- how many lines have no connectors at all,
- `lexicon.agreement` from the character build, which measures T2 (writers deviating from
  canonical stroke counts).

Record these in the ledger. They are findings, not knobs.

## 3. Then, and only then, the model

The line model does not exist yet. Before training anything it must implement:
connector tokens in the denoising sequence, layout tokens, k-line style encoding, the four
connectivity metrics, and AR/CR. Until then `cent/` trains a character model and cannot
produce a line — do not run it and call the output a line result.

## 4. Baselines

| Baseline | Modality | Clone | Adapter |
|---|---|---|---|
| DiffBrush | offline lines | `baselines/diffbrush` @ `da9addc` | **not written** |
| TGC-Diff | offline lines | `baselines/tgc_diff` @ `4d4058d` | **not written** |
| DiffInk | online lines | `baselines/diffink` @ `6505554` | **not written** |
| OLHWG | online lines | `baselines/olhwg` @ `e1bed55` | **not written**, and no licence file |

InkSpire and One-DM are excluded with reasons in `docs/BASELINE_CHANGES.md`.

## 5. Freeze before you run

`experiments/run_sheet.json` is the frozen plan (hash in `run_sheet.sha256`, `plan-check`
ok). The gate reads the **validation** writers only; the reported test partitions are
evaluated once, after a gate PASS. Changing the harness voids the sheet: amend, do not edit.

**Decision rule, restated:** content accuracy is a *gate* (AR within 1 point of the best
baseline), not the headline. The claim is about connector rate and pen-lift agreement
being closer to the writer's own at matched accuracy. If connectivity improves while
text accuracy falls, the rule says that is not an improvement.

## 6. Ledger

Every run leaves, under `runs/<experiment_id>/`: config, the git commit of each baseline
actually used, seeds, and the provider job id. A number without a ledger line is not
reportable.

## 7. Deliberately absent

No corpus, no canonical arrays, no checkpoints, and no GPU provisioning.
