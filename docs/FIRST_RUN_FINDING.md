# First real-data run: what blocked it, and what it means for the method

Instance: RTX 3090 (x16, 25.6 GB/s), 32 GB RAM, 40 GB disk, Switzerland, **$0.256/h**.
Ran ~75 minutes, spent **≈$0.32**. Stopped before training. Nothing below is inferred from
documentation — every number was measured on the supplier's actual files.

## 1. What worked

| Step | Result |
|---|---|
| Rent + boot | 3090, `pcie.link.width 16`, sm_86, torch 2.5.1+cu124 preinstalled |
| Download CASIA line data | NLPR throttles **150 KB/s per connection** but scales linearly; `aria2c -x16` reached **38 MB/s**. 2.6 GB fetched in ~2 min once parallelised |
| `.dgrl` parser | **fixed** — now reads real pages ("这双解放初期沂蒙山区…"), 5 lines/130 chars per page |
| `.wptt` parser | **fixed** — reads 8–10 lines/page, clean transcripts ("调查显示,党政干部读书有三个显著特点…") |
| Canonical build | 815 writers, **41,710 stroke-bearing lines**, **11,778 paired**, 2,680 classes, 391 MB, 13,768 line images |
| Token cache | 7,843 paired train lines (max 341 tokens), 846 val, primed in ~4 min |

Both format bugs are documented and fixed in `common/casia_native.py`; the old readers were built
from a second-hand description and were wrong about endianness, label ordering and bitmap depth.

## 2. What blocked the run — the decisive number

**Join rate = 0.0000 over 178,674 boundaries.** Not one character boundary in 7,843 paired lines
carries a connector under the count-based inference.

The cause is not a bug in the inference; it is the data:

| | measured |
|---|---|
| characters per line | 25.9 |
| strokes the lexicon expects | **163.4** |
| strokes actually present | **96.0** |
| ratio actual / lexicon | **0.597** |
| lines where actual >= lexicon (the rule's precondition) | **3 / 400 = 0.8%** |

Real writers produce **~40% fewer strokes than the lexicon order** for the same text: strokes are
merged, abbreviated and joined *within* the pen path. With fewer strokes than the lexicon demands,
`infer_char_strokes` can only split greedily by count — every stroke is consumed by characters, no
stroke is ever left over to be labelled a connector, and the character boundaries themselves are
guesses.

## 3. Why this is a method problem, not a pipeline problem

CENT's token sequence assumes, per character, a **lexicon-fixed stroke count and order**, with a
**separate connector token per boundary**. Both assumptions break on this corpus:

1. **The intra-character topology oracle is false in practice.** The paper's constraint ("the
   stroke graph of a character has the lexicon's count") does not hold for 99.2% of lines. A
   reviewer with access to this corpus would find that immediately.
2. **Connections are not separate strokes.** In cursive Chinese the connection is usually the pen
   path continuing out of character A into character B — there is no extra stroke to label. A
   connector *token* has nothing to supervise it; the phenomenon lives inside strokes, not between
   them.
3. **Per-character localisation is not identifiable** from this data: the format gives line-level
   stroke indices and line-level transcripts, but no per-character stroke assignment.

So the blocker is not that training fails — it is that the supervision the design needs does not
exist in the corpus. Training a 30-epoch model now would teach it a decomposition that is wrong
about 99% of lines, and the contact sheets would show misassigned strokes. I stopped rather than
produce a number that looks like a result.

## 4. Options, with their costs

**A. Re-formulate at stroke level (recommended).** Drop the per-character split for line data.
The line becomes a sequence of strokes with a line-level transcript; a "join" becomes a geometric
property (a stroke that starts near where the previous one ended), not a labelled token. This is
supervisable from exactly this corpus, keeps the continuity-by-construction renderer, and turns the
central claim into "we model where the pen does not lift". Cost: dataset + model head changes,
~1–2 days; paper §III needs rewriting around stroke-level joins.

**B. Keep character decomposition, get per-character data.** Use the **isolated character** corpora
(HWDB1.x/OLHWDB1.x), where each sample is one character with its own strokes, to learn the character
generator; use line data only for layout and for the *emergent* join statistics. Cost: extra
downloads (~8 GB character data), a two-stage model, and the line-level join claim weakens to an
emergent property.

**C. Use ICDAR-2013 online text**, which may carry finer annotation than CASIA's 2.x line files.
Cost: unzip and inspect (already downloaded, 13 MB) — cheap to check, uncertain payoff.

**D. Report the negative result.** "Count-based character decomposition is not identifiable in
Chinese text-line corpora; here is the measurement (0.597 ratio, 0.8% of lines admissible)." That is
a genuine contribution to the community and is publishable as a short paper or a section, but it is
not the CVPR line-generation paper originally planned.

## 5. What I would do next, in order

1. **Check C first** (10 minutes, data already on the instance): if the ICDAR-2013 online text
   carries per-character stroke indices, option B collapses into the current design.
2. If not, take **A**: redesign the token sequence at stroke level. The renderer, the metrics
   (join-rate error is already defined geometrically for the image-only proxy), and the style
   encoder all survive unchanged — it is the dataset and one head that change.
3. Keep the parser fixes and the cache builder; they are correct and reusable either way.

## 6. State left behind

- Instance **stopped** (GPU billing off; disk still allocated at $0.03/h). Canonical set, caches and
  the downloaded corpus are on it; it can be restarted in seconds.
- The 30-epoch run was **not** started. The $1 budget is unspent beyond $0.32 of setup and data.
- Fixed and local: `common/casia_native.py` (both parsers), `common/casia_lines.py` (delegates),
  `cent/cache_lines.py` + `cent/train_lines.py` (cache path), `IMG_W_MAX` raised to 4096 for storage
  while the model canvas stays 1024.
