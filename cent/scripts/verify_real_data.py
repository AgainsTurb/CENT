"""Verify the architecture on the REAL corpora that are now on disk.

Checks, in order:
  1. WPTT pages parse and the byte accounting closes (parser self-check)
  2. transcripts decode to Chinese
  3. per-line samples convert to tokens (char ids estimated, see caveat)
  4. the join (连笔) rate measured on real CASIA-OLHWDB2.0 lines
  5. CENTModel forward runs on a real batch with finite losses
  6. COUCH09 status is reported (stroke column present, transcript absent)

Run:  python scripts/verify_real_data.py --pages 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.data.dataset import CouchDataset, build_vocab_from_samples, collate  # noqa: E402
from cent.data.format_wptt import read_wptt, page_to_samples  # noqa: E402
from cent.data.tokens import to_tokens  # noqa: E402
from cent.model import CENTModel  # noqa: E402

OK, BAD = "\u2713", "\u2717"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.06)
    args = ap.parse_args()

    t0 = time.time()
    train_dir = ROOT / "datasets" / "CASIA-OLHWDB2.0" / "train"
    test_dir = ROOT / "datasets" / "CASIA-OLHWDB2.0" / "test"
    fails = []

    def check(cond, msg):
        print(f"  {OK if cond else BAD} {msg}")
        if not cond:
            fails.append(msg)

    print("\n[1] corpus on disk")
    n_tr = len(list(train_dir.glob("*.wptt"))) if train_dir.exists() else 0
    n_te = len(list(test_dir.glob("*.wptt"))) if test_dir.exists() else 0
    print(f"  CASIA-OLHWDB2.0: {n_tr} train pages, {n_te} test pages")
    check(n_tr > 0, "WPTT train pages present")

    print(f"\n[2] parse {args.pages} pages (stride {args.stride})")
    pages = sorted(train_dir.glob("*.wptt"))[: args.pages * args.stride : args.stride]
    parsed, samples, bad = [], [], []
    for p in pages:
        try:
            pg = read_wptt(p)
        except Exception as e:  # noqa: BLE001
            bad.append((p.name, str(e)[:60]))
            continue
        parsed.append(pg)
        samples.extend(page_to_samples(pg))
    print(f"  pages parsed: {len(parsed)}  (failures: {len(bad)})")
    for name, err in bad[:3]:
        print(f"    {name}: {err}")
    check(len(bad) == 0, "every page parsed and byte accounting closed")

    n_lines = sum(p.n_lines for p in parsed)
    n_strokes = sum(len(p.strokes) for p in parsed)
    n_chars = sum(len("".join(p.line_text)) for p in parsed)
    print(f"  lines={n_lines}  strokes={n_strokes}  characters={n_chars}  "
          f"({n_chars/max(1,n_lines):.1f} chars/line)")
    print(f"  usable line samples after segmentation: {len(samples)}")
    check(n_chars > 0 and n_strokes > 0, "non-trivial content recovered")

    print("\n[3] transcripts decode")
    for p in parsed[:2]:
        for t in p.line_text[:2]:
            print(f"    {t[:48]!r}")
    check(all(any("\u4e00" <= c <= "\u9fff" for c in t) for p in parsed[:2]
              for t in p.line_text[:2]), "Chinese characters decode (not mojibake)")

    print("\n[4] continuity statistics on real lines")
    all_lab = []
    for s in samples:
        vocab0 = build_vocab_from_samples([s])
        tok = to_tokens(s, vocab0)
        if tok is not None:
            all_lab.extend(tok["boundary_labels"].tolist())
    gap_rate = float(np.mean(all_lab)) if all_lab else float("nan")

    # Second proxy: a join means the pen kept moving into the next character's
    # territory, i.e. a stroke of character i reaches past the left edge of
    # character i+1. This is the one that can actually see a connection stroke.
    in_stroke, n_b = 0, 0
    for s in samples:
        co = s.stroke_char
        if not co:
            continue
        c2s: dict[int, list[int]] = {}
        for si, c in enumerate(co):
            c2s.setdefault(c, []).append(si)
        keys = sorted(c2s)
        for a, b in zip(keys, keys[1:]):
            last = c2s[a][-1]
            first_next = c2s[b][0]
            if len(s.strokes[last]) == 0 or len(s.strokes[first_next]) == 0:
                continue
            next_left = float(s.strokes[first_next][:, 0].min())
            reaches = float(s.strokes[last][:, 0].max()) > next_left
            in_stroke += int(reaches)
            n_b += 1
    reach_rate = in_stroke / n_b if n_b else float("nan")

    print(f"  gap proxy        : {gap_rate:.3f}  (n={len(all_lab)})  "
          f"[stroke ends within tau={args.tau} of the next stroke's start]")
    print(f"  x-overlap rate   : {reach_rate:.3f}  (n={n_b})  "
          f"[last stroke of char i extends past char i+1's left edge]")
    print("  IMPORTANT — neither number is a join rate, and here is why:")
    print("    * A 连笔 join is ONE unbroken stroke. Cutting at stroke granularity always")
    print("      lands between two strokes, so the gap proxy can only see")
    print("      'pen lifted, then retouched nearby' (~8%), never a true connection.")
    print("    * The x-overlap proxy is inflated (~51%) because Chinese glyphs routinely")
    print("      reach into the next cell; overlap is not continuity.")
    print("    => WPTT alone cannot supply join labels. They must be derived from")
    print("       point-level character boundaries (CTC alignment) or hand-annotated.")
    check(len(all_lab) > 0, "boundary labels produced")

    print("\n[5] model forward on a real batch")
    vocab = build_vocab_from_samples(samples)
    ds = CouchDataset(vocab, samples=samples, max_points=2500, min_chars=2)
    print(f"  vocabulary size: {len(vocab.chars)} unique chars, dataset {len(ds)} samples")
    batch = collate([ds[i] for i in range(min(4, len(ds)))])
    B, T = batch["dxy"].shape[:2]
    print(f"  batch: B={B} T={T} chars={int(batch['char_mask'].sum())} "
          f"boundaries={int(batch['bnd_mask'].sum())}")

    model = CENTModel(n_chars=max(64, len(vocab)), n_writers=32, d=64, z_dim=32,
                      vae_d=64, stride=8, dit_depth=2, dit_heads=4, n_mix=6, n_steps=100)
    model.train()
    table = {w: i for i, w in enumerate(sorted({b for b in batch["writers"]}))}
    losses = model(batch, writer_table=table)
    shown = {k: float(v.detach()) for k, v in losses.items()
             if torch.is_tensor(v) and v.dim() == 0}
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in shown.items()))
    check(all(np.isfinite(v) for v in shown.values()), "losses finite on real data")
    losses["total"].backward()
    gn = sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  grad-norm: {gn:.3f}")
    check(gn > 0, "gradients flow on real data")

    print("\n[6] COUCH09 (DOLPHIN copy)")
    probe = ROOT / "datasets" / "_probe" / "couch"
    txts = sorted(probe.glob("*.txt")) if probe.exists() else []
    if txts:
        head = txts[0].read_text(encoding="utf-8", errors="replace").splitlines()[0]
        ncols = len(txts[0].read_text(encoding="utf-8").splitlines()[1].split())
        print(f"  columns={ncols}  header={head.split()}")
        print("  verdict: x, y, pressure, STROKE are present; the WORD TEXT is not in the")
        print("           archive (filenames are indices, no lexicon shipped), so it cannot")
        print("           train text-conditioned generation. Usable only for style/geometry.")
        check(ncols == 4, "COUCH09 keeps the stroke column")
    else:
        print("  (no probe files extracted)")

    print(f"\n{'='*64}")
    print(f"DONE in {time.time()-t0:.1f}s  |  failures: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    print("=" * 64)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
