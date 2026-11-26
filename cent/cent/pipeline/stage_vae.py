"""Stage 1/2 — train the InkVAE (reconstruction + CTC) on characters or lines.

Stage 1 (optional): isolated characters, if an OLHWDB 1.x corpus is present.
Stage 2 (required): text lines from CASIA-OLHWDB 2.0.

Each run writes checkpoints, ``metrics.json`` and **inspection images**. The
gate is the CTC greedy-decode character accuracy, because that is what decides
whether Stage 3 can align boundaries at all.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.tokens import CharVocab
from ..losses import vae_losses
from ..models.inkvae import InkVAE
from ..viz import (backend_name, figure_panels, plot_bars, plot_curves,
                   render_strokes, save_png, vstack_padded)
from . import cache as cache_mod
from . import fetch, perf
from .common import (banner, gate, gmm_to_points, pick_device, points_to_strokes,
                     seed_everything, visual_check, write_metrics)


def _ctc_accuracy(model, loader, device, max_batches: int = 4) -> float:
    """Character accuracy by edit distance (1 - CER), not positional matching.

    The earlier version compared decoded and target sequences position by
    position with zip(), which reports 0.0 for any length mismatch even when the
    decode is largely correct. That is why CTC read exactly 0.000 for 33 epochs.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            x = torch.cat([batch["dxy"],
                           torch.nn.functional.one_hot(batch["pen"], 4).float()], -1)
            out = model(x, batch["pen"], batch["point_mask"], sample=False)
            pred = out["ctc_logits"].argmax(-1)
            for b in range(pred.shape[0]):
                ids, prev = [], -1
                for k in pred[b].tolist():
                    if k != 0 and k != prev:
                        ids.append(k)
                    prev = k
                tgt = batch["char_seq"][b, : int(batch["char_mask"][b].sum())].tolist()
                if not tgt:
                    continue
                # Levenshtein / len(target)
                prevrow = list(range(len(tgt) + 1))
                for ii, pp in enumerate(ids, 1):
                    cur = [ii]
                    for jj, tt in enumerate(tgt, 1):
                        cur.append(min(prevrow[jj] + 1, cur[jj - 1] + 1,
                                       prevrow[jj - 1] + (pp != tt)))
                    prevrow = cur
                # clamp: insertions can push Levenshtein past len(target)
                scores.append(max(0.0, 1.0 - prevrow[-1] / len(tgt)))
    model.train()
    return float(sum(scores) / len(scores)) if scores else 0.0


def _recon_visuals(model, loader, device, out_dir: Path, n: int = 3):
    """Original vs reconstruction, side by side."""
    model.eval()
    paths = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            x = torch.cat([batch["dxy"],
                           torch.nn.functional.one_hot(batch["pen"], 4).float()], -1)
            out = model(x, batch["pen"], batch["point_mask"], sample=False)
            rec = gmm_to_points(out["gmm"]).cpu().numpy()
            pen_rec = out["pen_logits"].argmax(-1).cpu().numpy()
            dxy0 = batch["dxy"].cpu().numpy()
            pen0 = batch["pen"].cpu().numpy()
            mask = batch["point_mask"].cpu().numpy()
            for b in range(min(n, rec.shape[0])):
                t = int(mask[b].sum())
                ot = dxy0[b, :t]
                orig_pts = np.stack([np.cumsum(ot[:, 0]), np.cumsum(ot[:, 1])], axis=-1)
                r_pts = rec[b, :t]        # absolute (x, y)
                p = out_dir / f"recon_{b}.png"
                figure_panels([
                    ("input (real writer)",
                     render_strokes(points_to_strokes(orig_pts, pen0[b, :t]))),
                    ("VAE reconstruction",
                     render_strokes(points_to_strokes(r_pts, pen_rec[b, :t]))),
                ], p, title=f"reconstruction — validation sample {b}")
                paths.append(p)
            break
    model.train()
    return paths


def run(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = pick_device(args.device)
    root = Path(args.root)
    out = Path(args.out)
    pinfo = perf.setup_perf(device, amp=args.amp)

    banner(f"STAGE {args.stage} — InkVAE ({'characters' if args.stage == 1 else 'text lines'})")
    print(f"  device: {device}   root: {root}   out: {out}")
    print(f"  figure backend: {backend_name()}")
    print(f"  perf: {perf.describe(pinfo)}")

    if args.stage == 1:
        print("\n[data] character corpus is optional in this build.")
        chars_dir = fetch.find_char_dataset(root)
        if chars_dir is None:
            print("  no .pot corpus found — SKIPPING Stage 1.")
            print("  (Stage 2 will train the VAE from scratch on lines; this is fine,")
            print("   but record in the paper that char-level pretraining was skipped.)")
            return 0
        print(f"  character corpus: {chars_dir}")
        print("  NOTE: the .pot reader is implemented but has not been exercised on")
        print("        real files in this environment — verify the first batch visually.")
        from ..data.format_pot import PotCharDataset, read_pot_dir  # noqa: F401
        samples = read_pot_dir(chars_dir, limit=args.limit_chars)
        if not samples:
            print("  no parsable .pot samples — SKIPPING Stage 1.")
            return 0
        vocab = CharVocab.build([s["text"] for s in samples])
        dataset = PotCharDataset(samples, vocab)
    else:
        fetch.ensure_wptt(root, allow_download=not args.no_download)
        meta = cache_mod.build_cache(root, limit_pages=args.limit_pages,
                                     force=args.rebuild_cache)
        stores = {s: cache_mod.LineStore(root, s) for s in ("train", "val")}
        vocab = cache_mod.build_vocab([stores["train"], stores["val"]])
        dataset = cache_mod.LineDataset(stores["train"], vocab, limit=args.limit_lines)
        val_ds = cache_mod.LineDataset(stores["val"], vocab, limit=min(64, len(stores["val"])))

    print(f"  vocabulary: {len(vocab.chars)} characters")
    writers = sorted({b["writer"] for b in [dataset[i] for i in range(min(200, len(dataset)))]})
    writer_table = {w: i for i, w in enumerate(writers)}

    lk = perf.loader_kwargs(device, args.workers)
    lk_ns = {k: v for k, v in lk.items() if k not in ("shuffle", "drop_last")}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=cache_mod.collate_lines, **lk_ns)
    val_loader = DataLoader(dataset if args.stage == 1 else val_ds,
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=cache_mod.collate_lines, **lk_ns)

    model = InkVAE(d=args.d, z_dim=args.z_dim, stride=args.stride, n_mix=args.n_mix,
                   n_chars=len(vocab), n_writers=max(8, len(writer_table))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.init and Path(args.init).exists():
        sd = torch.load(args.init, map_location=device)
        model.load_state_dict(sd["model"])
        print(f"  initialised from {args.init}")

    out.mkdir(parents=True, exist_ok=True)
    history = {"total": [], "gmm": [], "ctc": [], "val_ctc_acc": []}
    step = 0
    t0 = time.time()
    max_steps = args.steps

    for epoch in range(args.epochs):
        for batch in loader:
            batch = perf.to_device(batch, device)
            with perf.autocast(device, pinfo["amp"]):
                x = torch.cat([batch["dxy"],
                               torch.nn.functional.one_hot(batch["pen"], 4).float()], -1)
                outv = model(x, batch["pen"], batch["point_mask"], sample=True)
            widx = torch.tensor([writer_table.get(w, 0) for w in batch["writers"]],
                                device=device)
            losses = vae_losses(outv, batch["dxy"], batch["pen"], batch["char_seq"],
                                batch["char_mask"], widx, batch["point_mask"])
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            for k in ("total", "gmm", "ctc"):
                history[k].append(float(losses[k].detach()))
            if step % args.log_every == 0:
                acc = _ctc_accuracy(model, val_loader, device)
                history["val_ctc_acc"].append(acc)
                print(f"  ep{epoch} step{step:5d} total={losses['total'].item():.3f} "
                      f"gmm={losses['gmm'].item():.3f} ctc={losses['ctc'].item():.3f} "
                      f"val_ctc_acc={acc:.3f} "
                      f"({(time.time()-t0)/max(1,step)*1000:.0f} ms/step)")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    acc = _ctc_accuracy(model, val_loader, device, max_batches=8)
    ckpt = out / "vae.pt"
    torch.save({"model": model.state_dict(), "vocab": vocab.chars,
                "n_writers": max(8, len(writer_table)),
                "stage": args.stage, "args": vars(args)}, ckpt)

    vis = _recon_visuals(model, val_loader, device, out / "visuals")
    plot_curves({"total": history["total"], "gmm nll": history["gmm"],
                 "ctc": history["ctc"]},
                out / "visuals" / "loss_curves.png",
                title=f"Stage {args.stage} training loss", smooth=25)
    if history["val_ctc_acc"]:
        plot_bars(history["val_ctc_acc"], out / "visuals" / "val_ctc_acc.png",
                  title="CTC greedy-decode character accuracy (validation)",
                  ylabel="accuracy", vmax=1.0,
                  hline=0.90 if args.stage == 1 else 0.85,
                  hline_label="gate threshold")
    vis += [out / "visuals" / "loss_curves.png"]

    write_metrics(out / "metrics.json", {
        "stage": args.stage, "steps": step, "vocab": len(vocab.chars),
        "val_ctc_acc": acc, "final_loss": history["total"][-1] if history["total"] else None,
        "checkpoint": str(ckpt),
    })

    gate_name = "G1" if args.stage == 1 else "G2"
    threshold = 0.90 if args.stage == 1 else 0.85
    ok = acc >= threshold
    visual_check(vis, f"stage {args.stage}")
    return 0 if gate(gate_name, ok,
                     f"CTC char accuracy {acc:.3f} vs target {threshold:.2f}", strict=args.strict) else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CENT stage 1/2: InkVAE training")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--out", default="runs/stage2_vae")
    ap.add_argument("--init", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=0, help="cap steps (0 = no cap)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--n-mix", type=int, default=20)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--limit-chars", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                    help="bf16 autocast (default on)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--compile", action="store_true", help="torch.compile the VAE")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on gate failure")
    ap.add_argument("--force", action="store_true", help="continue past gate failure")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.limit_pages = args.limit_pages or None
    args.limit_lines = args.limit_lines or None
    if args.force:
        args.strict = False
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
