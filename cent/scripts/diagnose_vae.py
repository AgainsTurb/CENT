"""Diagnose the InkVAE locally: train briefly, then measure what actually broke.

Four numbers, each aimed at a failure the loss curve hid:

  CER          character error rate (edit distance) — replaces the positional
               `zip` accuracy that reported 0.000 for any length mismatch
  EPE          endpoint error of the reconstruction, normalised by line height
               (the loss improved while the drawing collapsed; this catches it)
  straightness ratio of reconstructed path length to its endpoint distance.
               ~1.0 means "a straight line", which is what a collapsed decoder
               outputs; real handwriting is far above 1
  z-sensitivity EPE change when the latent is swapped for another line's.
               ~0 means the decoder ignores the latent (posterior collapse)

    python scripts/diagnose_vae.py --pages 3 --steps 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.data.tokens import CharVocab  # noqa: E402
from cent.losses import vae_losses  # noqa: E402
from cent.models.inkvae import InkVAE  # noqa: E402
from cent.pipeline import cache as cache_mod  # noqa: E402
from cent.pipeline.common import gmm_to_points, points_to_strokes  # noqa: E402
from cent.viz import figure_panels, render_strokes  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def cer(pred_ids, target_ids) -> float:
    """Levenshtein distance / target length — the honest character metric."""
    if not target_ids:
        return 0.0
    prev = list(range(len(target_ids) + 1))
    for i, p in enumerate(pred_ids, 1):
        cur = [i]
        for j, t in enumerate(target_ids, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != t)))
        prev = cur
    return prev[-1] / len(target_ids)


def greedy(logits) -> list[int]:
    ids, prev = [], -1
    for k in logits.argmax(-1).tolist():
        if k != 0 and k != prev:
            ids.append(k)
        prev = k
    return ids


def straightness(pts) -> float:
    if len(pts) < 3:
        return 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    span = float(np.linalg.norm(pts[-1] - pts[0]))
    return seg / max(span, 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-ctc", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--single", action="store_true",
                    help="train AND evaluate on one fixed line (true overfit test)")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cache_mod.build_cache(Path(a.root), limit_pages=a.pages)
    store = cache_mod.LineStore(Path(a.root), "train")
    vocab = cache_mod.build_vocab([store])
    ds = cache_mod.LineDataset(store, vocab)
    print(f"lines={len(ds)} vocab={len(vocab)+1} device={dev}")

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True,
                    collate_fn=cache_mod.collate_lines)
    model = InkVAE(d=a.d, z_dim=a.z_dim, stride=a.stride, n_mix=20,
                   n_chars=len(vocab) + 1, n_writers=8).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    fixed = None
    if a.single:
        b0 = cache_mod.collate_lines([ds[0]])
        fixed = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b0.items()}
    step = 0
    while step < a.steps:
        for batch in ([fixed] if fixed is not None else dl):
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            x = torch.cat([batch["dxy"],
                           torch.nn.functional.one_hot(batch["pen"], 4).float()], -1)
            out = model(x, batch["pen"], batch["point_mask"], sample=True)
            widx = torch.zeros(batch["dxy"].shape[0], dtype=torch.long, device=dev)
            losses = vae_losses(out, batch["dxy"], batch["pen"], batch["char_seq"],
                                batch["char_mask"], widx, batch["point_mask"],
                                w_ctc=a.w_ctc)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"  step {step:4d} total={float(losses["total"].detach()):.3f} "
                      f"gmm={float(losses["gmm"].detach()):.3f} ctc={float(losses["ctc"].detach()):.3f}")
            if step >= a.steps:
                break

    # ---------------- evaluation
    model.eval()
    ev = fixed if fixed is not None else next(iter(dl))
    ev = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ev.items()}
    x = torch.cat([ev["dxy"], torch.nn.functional.one_hot(ev["pen"], 4).float()], -1)
    with torch.no_grad():
        out = model(x, ev["pen"], ev["point_mask"], sample=False)
        z = out["mu"]
        # gmm_to_points returns per-step DELTAS; positions need a cumsum
        recon = torch.cumsum(gmm_to_points(out["gmm"]), dim=1)   # deltas -> points
        # latent swap: decode with another line's latent
        z_swap = z.roll(1, dims=0)
        out_swap = model.decode(z_swap)
        if isinstance(out_swap, torch.Tensor) and out_swap.dim() == 4:
            recon_swap = torch.cumsum(gmm_to_points(out_swap), dim=1)
        elif isinstance(out_swap, dict):
            recon_swap = out_swap.get("points")
        else:
            recon_swap = out_swap

    # gmm_to_points returns cumulative positions (see common.gmm_to_points)
    r = recon.cpu().numpy()
    truth = np.cumsum(ev["dxy"].cpu().numpy(), axis=1)
    mask = ev["point_mask"].cpu().numpy()

    epis, straights_t, straights_r, cers = [], [], [], []
    for b in range(r.shape[0]):
        t = int(mask[b].sum())
        if t < 8:
            continue
        h = float(truth[b, :t, 1].max() - truth[b, :t, 1].min()) or 1.0
        epis.append(float(np.abs(r[b, :t] - truth[b, :t]).mean()) / h)
        straights_t.append(straightness(truth[b, :t]))
        straights_r.append(straightness(r[b, :t]))
        cers.append(cer(greedy(out["ctc_logits"][b, : int(out["latent_mask"][b].sum())]),
                        ev["char_seq"][b, : int(ev["char_mask"][b].sum())].tolist()))

    zsens = None
    if isinstance(recon_swap, torch.Tensor):
        rs = recon_swap.cpu().numpy()
        diffs = [float(np.abs(rs[b, : int(mask[b].sum())] - r[b, : int(mask[b].sum())]).mean())
                 for b in range(min(4, len(r)))]
        zsens = float(np.mean(diffs))

    print("\n================ diagnosis ================")
    print(f"  CER (edit distance)      : {np.mean(cers):.3f}   (1.0 = nothing decoded)")
    print(f"  recon EPE / line height  : {np.mean(epis):.3f}   (0 = perfect, >0.2 = broken)")
    print(f"  straightness  real       : {np.mean(straights_t):.2f}")
    print(f"  straightness  recon      : {np.mean(straights_r):.2f}   "
          f"(~1.0 => decoder emits a straight line)")
    if zsens is not None:
        print(f"  z-sensitivity (EPE)      : {zsens:.4f}  (~0 => latent ignored)")
    print("===========================================")
    ok = np.mean(cers) < 0.5 and np.mean(epis) < 0.2 and np.mean(straights_r) > 2.0
    print("VERDICT:", "looks learnable — continue training" if ok
          else "still broken — fix the architecture before renting a GPU")

    panels = []
    for b in range(min(3, len(r))):
        t = int(mask[b].sum())
        panels.append((f"real line {b} (straightness {straights_t[b]:.1f})",
                       render_strokes(points_to_strokes(truth[b, :t], ev["pen"][b, :t].cpu().numpy()))))
        panels.append((f"reconstruction (straightness {straights_r[b]:.1f}, "
                       f"EPE {epis[b]:.2f})",
                       render_strokes(points_to_strokes(r[b, :t], ev["pen"][b, :t].cpu().numpy()),
                                      colour=(170, 40, 40))))
    figure_panels(panels, "diagnose_vae_recon.png",
                  title=f"InkVAE after {a.steps} steps — geometry check")
    print("figure -> diagnose_vae_recon.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
