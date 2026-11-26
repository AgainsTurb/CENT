"""Smoke test — verifies the architecture end to end on tiny synthetic data.

No training. Checks, in order:
  1. synthetic fixtures are written in the real SCUT-COUCH formats
  2. both parsers round-trip (Unipen + binary .IDX/.DAT)
  3. the boundary (join/lift) label is derived correctly
  4. the Dataset/collate contract holds (masks, alignment)
  5. CENTModel forward produces finite losses and backward produces grads
  6. sampling runs, and the cursive control knob actually changes join rate
  7. the AR baseline (Option A) forwards
  8. the decoder turns latents back into points

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.data.dataset import CouchDataset, build_vocab_from_samples, collate  # noqa: E402
from cent.data.format_scut import (  # noqa: E402
    load_scutcouch_textline,
    load_scutcouch_word_dir,
    read_unipen,
)
from cent.data.synthetic import build_fixture_corpus  # noqa: E402
from cent.data.tokens import CURSIVE_EOC, PEN_NAMES  # noqa: E402
from cent.metrics import boundary_f1, continuation_rate_error, dtw_distance  # noqa: E402
from cent.model import CENTModel  # noqa: E402
from cent.models.baseline_ar import ARLineModel  # noqa: E402

OK = "\u2713"
BAD = "\u2717"
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {OK if cond else BAD} {msg}")
    if not cond:
        failures.append(msg)


def main() -> int:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)

    print("\n[1] build fixtures in the real on-disk formats")
    man = build_fixture_corpus(ROOT / "datasets" / "_smoke", n_lines=10, n_words=14,
                               seed=1, cursive_rate=0.4)
    print(f"  root: {man['root']}")
    check(Path(man["unipen_dir"]).exists(), "Unipen textline dir written")
    check(Path(man["word_dir"]).exists(), "binary WORD dir written")

    print("\n[2] parser round-trip")
    uni = list(load_scutcouch_textline(man["unipen_dir"]))
    wr = list(load_scutcouch_word_dir(man["word_dir"]))
    check(len(uni) == man["n_lines"], f"Unipen files parsed ({len(uni)})")
    check(len(wr) == man["n_words"], f"binary words parsed ({len(wr)})")
    check(all(len(s.text) >= 2 for s in uni + wr), "transcripts non-empty")
    check(all(s.n_strokes >= 1 for s in uni + wr), "strokes recovered")
    # check the GB round-trip survives
    same_text = sum(1 for s in wr if s.text and all("\u4e00" <= c <= "\u9fff" for c in s.text))
    check(same_text == len(wr), "GB codes decode back to Chinese")

    print("\n[3] pen-state / boundary labels")
    from cent.data.tokens import CharVocab, to_tokens
    v0 = CharVocab.build([s.text for s in wr])
    all_lab, n_join, n_bnd = [], 0, 0
    for s in wr:
        tok = to_tokens(s, v0)
        if tok is None:
            continue
        all_lab.extend(tok["boundary_labels"].tolist())
        pen = tok["pen"]
        n_join += int((pen == CURSIVE_EOC).sum())
        n_bnd += int((tok["char_ids"][1:] != tok["char_ids"][:-1]).sum())
    rate = float(np.mean(all_lab)) if all_lab else float("nan")
    print(f"  boundaries={len(all_lab)}  joins={int(sum(all_lab)) if all_lab else 0}  "
          f"join-rate={rate:.3f} (generator target 0.4)")
    check(len(all_lab) > 0, "boundaries exist")
    check(0.15 < rate < 0.75, "join rate in the expected range")
    print(f"  pen vocabulary: {', '.join(PEN_NAMES)}")

    print("\n[4] dataset + collate")
    vocab = build_vocab_from_samples(uni + wr)
    ds = CouchDataset(vocab, samples=uni + wr, max_points=2000, min_chars=2)
    check(len(ds) > 0, f"dataset built ({len(ds)} samples)")
    batch = collate([ds[i] for i in range(min(4, len(ds)))])
    B, T = batch["dxy"].shape[:2]
    print(f"  batch: B={B} T={T} L={batch['char_seq'].shape[1]} "
          f"boundaries={int(batch['bnd_mask'].sum())}")
    check(batch["point_mask"].sum() > 0, "point mask populated")
    check(batch["bnd_mask"].sum() > 0, "boundary mask populated")
    check(bool((batch["char_ids"][batch["point_mask"]] >= 0).all()),
          "every valid point has a char id")

    print("\n[5] CENTModel forward / backward")
    model = CENTModel(n_chars=max(64, len(vocab)), n_writers=32, d=64, z_dim=32,
                      vae_d=64, stride=8, dit_depth=2, dit_heads=4, n_mix=6,
                      n_steps=100)
    model.train()
    losses = model(batch, writer_table={w: i for i, w in enumerate(sorted({b for b in batch['writers']}))})
    shown = {k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.dim() == 0}
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in shown.items()))
    check(all(np.isfinite(v) for v in shown.values()), "all losses finite")
    losses["total"].backward()
    gnorm = sum(float(p.grad.norm()) ** 2 for p in model.parameters()
                if p.grad is not None) ** 0.5
    print(f"  grad-norm after backward: {gnorm:.4f}")
    check(gnorm > 0, "gradients flow")

    print("\n[6] sampling + cursive control")
    model.eval()
    with torch.no_grad():
        style = model.style_vector(batch)
        rates = {}
        for c in (0.0, 0.5, 1.0):
            out = model.generate(batch["char_seq"], batch["char_mask"], style,
                                 control=c, steps=6, n_chars_per_frame=2)
            valid = out["bnd_valid"]
            committed = out["committed"]
            jr = float((committed[valid] > 0.5).float().mean()) if valid.any() else float("nan")
            rates[c] = jr
            print(f"  control={c:.1f} -> committed join-rate={jr:.3f} "
                  f"(slots={int(valid.sum())})")
        check(set(np.round(list(rates.values()), 3)) >= {0.0, 1.0},
              "control knob spans lift-only to join-only")

    print("\n[7] decoder")
    with torch.no_grad():
        dec = model.decode(out["z"])
    check(dec["points"].shape[:2] == (B, out["z"].shape[1] * 8),
          f"decoded points shape {tuple(dec['points'].shape)}")
    check(torch.isfinite(dec["points"]).all(), "decoded points finite")

    print("\n[8] AR baseline (Option A) forward")
    ar = ARLineModel(d=64, depth=2, n_heads=4, n_mix=6, n_chars=max(64, len(vocab)))
    with torch.no_grad():
        style_d = torch.randn(B, 64)
        ar_out = ar(batch["dxy"], batch["pen"], batch["char_ids"], style_d)
    print(f"  gmm={tuple(ar_out['gmm'].shape)} pen={tuple(ar_out['pen_logits'].shape)}")
    check(ar_out["gmm"].shape[:2] == batch["dxy"].shape[:2], "AR gmm matches T")
    check(torch.isfinite(ar_out["gmm"]).all(), "AR output finite")

    print("\n[9] metrics")
    f1 = boundary_f1(losses["dit_out"]["bnd_logits"], *__import__(
        "cent.losses", fromlist=["align_boundary_labels"]
    ).align_boundary_labels(batch["bnd_labels"], batch["bnd_mask"],
                            losses["dit_out"]["bnd_valid"]))
    print(f"  boundary F1={f1['f1']:.3f} on {f1['n']} slots (untrained: ~chance)")
    cre = continuation_rate_error([0, 1, 0], [0, 1, 1])
    check(abs(cre - 1 / 3) < 1e-6, f"CRE computes correctly ({cre:.3f})")
    d = dtw_distance(np.zeros((3, 2)), np.ones((4, 2)))
    check(np.isfinite(d), f"DTW computes ({d:.3f})")

    print(f"\n{'='*62}")
    if failures:
        print(f"SMOKE TEST FAILED — {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"SMOKE TEST PASSED — all checks green in {time.time()-t0:.1f}s")
    print(f"{'='*62}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
