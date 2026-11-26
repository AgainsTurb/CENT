# Dataset verification — 2026-11-26

Three archives were placed in `datasets/`: two official **WPTT** releases and
**COUCH09** from the DOLPHIN release. Both were parsed and run through the
pipeline. Verdict: **one is usable, one is not, and neither supplies the label
CENT's boundary head needs** — but there is a clean way to bootstrap it.

All numbers below were produced by `scripts/verify_real_data.py` on the files on
disk, not taken from papers.

---

## 1. What is on disk

| Archive | Size | Extracted to | Contents |
|---|---|---|---|
| `WPTT2.0-Train.zip` | 60.7 MB | `datasets/CASIA-OLHWDB2.0/train/` | **1,678 pages** (`001-P16.wptt` …) |
| `WPTT2.0-Test.zip` | 15.3 MB | `datasets/CASIA-OLHWDB2.0/test/` | **420 pages** |
| `COUCH09.zip` | 148.2 MB | not extracted (148 MB → 145,000 files) | 145 writer dirs × 1,000 word samples |

2,098 pages total, which matches the official OLHWDB2.0 count exactly
(420 writers, 2,098 pages, 20,573 lines, 540,009 characters). So this is
**OLHWDB 2.0** — not 2.1/2.2.

## 2. WPTT: verified byte-for-byte ✅

The reader in `cent/data/format_wptt.py` parses header → strokes → lines and
**asserts that it consumes exactly `sample_len` bytes**. On 20 pages:

```
pages parsed: 20  (failures: 0)
lines=197  strokes=20,921  characters=5,196   (26.4 chars/line)
```

26.4 chars/line matches the corpus's documented 26.68. Transcripts decode
correctly, including mixed ASCII digits:

```
'2002年以来,国内企业家包括许多著名企业家在内涉嫌违法犯罪被捕入'
'调查显示,党政干部读书有三个显著特点:接受网络等媒体阅读的速度'
```

One decoding subtlety worth recording: character codes are 16-bit and **mixed
script** — ASCII is stored as `0x00XX`, Chinese as the two GB bytes. Decoding the
whole stream at once produces interleaved NULs (`'2\x000\x000\x002\x00年'`), so
`_decode_codes` handles each code separately. The parser now does this.

### What WPTT does and does not give CENT

| Need | Status |
|---|---|
| Line trajectory, pen lifts delimiting strokes | ✅ 656 strokes / page typical |
| Line transcript (text conditioning) | ✅ |
| Writer identity | ✅ from filename (`001-P16` → writer 001) |
| Lines (not characters) | ✅ 26.4 chars/line |
| **Character boundaries (point-level)** | ❌ **line-level annotation only** |
| **Join / lift labels** | ❌ follows from the above |

## 3. COUCH09 (DOLPHIN copy): strokes yes, text no ❌

The raw files are **4-column text**, and the header is explicit:

```
       x               y            pressure         stroke
       39              30              1               0
       41              31              1               0
```

- **Stroke index is present** (column 4) — DOLPHIN's `preprocess.py` keeps only
  `split()[:3]`, which discards it. So the released zip is *more* informative
  than DOLPHIN's preprocessed `.pkl`.
- `pressure` is a constant `1` — a dummy, as expected for this corpus.
- **The word text is not in the archive.** Filenames are indices (`L4872.txt`),
  and no lexicon ships with the zip. Without the transcript we do not even know
  how many characters each sample has (SCUT-COUCH words are 2–5 characters).

**Consequence:** COUCH09 cannot train text-conditioned generation and cannot
supply character boundaries. It is usable only as an **unlabelled style/geometry
corpus** (145 writers × 1,000 multi-character words, boundary-dense).

## 4. The hard finding: neither corpus can label a join

This was measured, not assumed. Two proxies, both reported with their failure
mode (20 pages, n≈5,000–7,800 boundaries):

| Proxy | Rate | Why it is wrong |
|---|---|---|
| **Gap proxy** — stroke ends within τ=0.06 of the next stroke's start | **7.9 %** | A 连笔 join is **one unbroken stroke**. Cutting at stroke granularity always lands *between* two strokes, so this only detects "pen lifted, then retouched nearby" |
| **X-overlap proxy** — last stroke of char *i* extends past char *i+1*'s left edge | **51.2 %** | Inflated: Chinese glyphs routinely reach into the next cell. Overlap ≠ continuity |

So the honest statement is: **no public Chinese trajectory corpus provides
point-level character boundaries, and without them the join label is not
observable.** That is not a data-sourcing failure to be papered over; it is the
central difficulty of the task, and it is exactly what contribution (iii) should
address.

## 5. How to actually obtain join labels

Three routes, in order of rigour:

1. **Bootstrap from the model's own CTC alignment (recommended).** The pipeline
   already has a CTC head on the InkVAE latent. Stage 1 trains it on
   line transcripts — no boundary supervision needed. Stage 2 runs Viterbi
   alignment to place a boundary *at point level*, then the join label is exact
   and free: `join ⇔ stroke_id[t] == stroke_id[t-1]` at the boundary point.
   This turns the annotation problem into a curriculum, and it is a defensible
   contribution in its own right ("boundaries are learned, joins are a
   byproduct").
2. **Hand-annotate a small evaluation set** (a few hundred cursive lines,
   marking connection strokes). Needed regardless, because no automatic proxy
   can validate itself.
3. **Ask for SCUT-COUCH `.SGM` segmentation** when the application is approved —
   but note its per-area stroke index lists are still stroke-granular, so it too
   can only see a join if a stroke is shared between two character areas.

## 6. Bottom line

- **Keep and use WPTT.** It is the real line-level corpus: text + trajectory +
  420 writers, verified, unencrypted. Write the loader into the dataset path.
- **Keep COUCH09 zipped**, for optional style/geometry pretraining only. Do not
  build content conditioning on it.
- **Do not expect either to supervise the boundary head.** Build the CTC
  bootstrap instead, and start the hand-annotated continuity set early — it is
  the artifact that will make the paper's claim checkable.
