"""Stage 3 -derive point-level character boundaries and join labels.

Loads the Stage-2 VAE, Viterbi-aligns every line against its transcript, and
writes ``align.npz`` plus join statistics. The statistics are the paper's first
hard measurement of cursive continuity in the standard Chinese benchmark, so the
script also draws the evidence: alignments with boundary guides, and the
per-writer join-rate distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.inkvae import InkVAE
from ..viz import backend_name, figure_strokes, plot_bars, plot_hist
from . import cache as cache_mod
from . import fetch
from .align import derive_store_boundaries, save_alignment
from .common import banner, gate, pick_device, seed_everything, visual_check, write_metrics


def _alignment_visuals(model, store, vocab, out_dir: Path, *, stride: int,
                       device, n: int = 4) -> list[Path]:
    """Draw lines with vertical guides where the alignment put char boundaries."""
    from .cache import line_to_tensor
    from .align import labels_to_points, viterbi_ctc

    model.eval()
    paths = []
    picked = 0
    with torch.no_grad():
        for i in range(len(store)):
            line = store.get(i)
            t = line_to_tensor(line, vocab)
            if t is None:
                continue
            x = torch.cat([torch.from_numpy(t["dxy"])[None],
                           torch.nn.functional.one_hot(
                               torch.from_numpy(t["pen"])[None], 4).float()], -1).to(device)
            pen = torch.from_numpy(t["pen"])[None].to(device)
            mask = torch.ones(1, t["dxy"].shape[0], dtype=torch.bool, device=device)
            out = model(x, pen, mask, sample=False)
            lp = out["ctc_logits"][0].log_softmax(-1).cpu().numpy()
            lab, conf = viterbi_ctc(lp, t["char_seq"].tolist())
            pts_lab = labels_to_points(lab, len(t["dxy"]), stride)

            strokes = []
            cur, sid = [], line["stroke_ids"]
            for k in range(len(t["dxy"])):
                if k > 0 and sid[k] != sid[k - 1] and cur:
                    strokes.append(np.asarray(cur))
                    cur = []
                cur.append(np.cumsum(t["dxy"][: k + 1], axis=0)[-1])
            if cur:
                strokes.append(np.asarray(cur))

            guides = []
            x_abs = np.cumsum(t["dxy"][:, 0])
            for k in range(1, len(pts_lab)):
                if pts_lab[k] != pts_lab[k - 1]:
                    guides.append(float(x_abs[k]))
            p = out_dir / f"alignment_{picked}.png"
            figure_strokes(strokes, p, boundary_x=guides,
                           title=(f"alignment conf={conf:.2f} · {len(guides)} boundaries "
                                  f"for {len(t['char_seq'])} chars · {line['text'][:28]}"))
            paths.append(p)
            picked += 1
            if picked >= n:
                break
    return paths


def run(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = pick_device(args.device)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    banner("STAGE 3 -boundary + join label derivation")
    print(f"  device: {device}   vae: {args.vae}   out: {out}")
    print(f"  figure backend: {backend_name()}")

    fetch.ensure_wptt(root, allow_download=not args.no_download)
    cache_mod.build_cache(root, limit_pages=args.limit_pages, force=args.rebuild_cache)

    ckpt_path = Path(args.vae)
    if not ckpt_path.exists():
        print(f"  checkpoint missing: {ckpt_path} -run stage 2 first")
        return 2
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = cache_mod.build_vocab([cache_mod.LineStore(root, "train")])
    if ckpt.get("vocab"):
        from ..data.tokens import CharVocab
        chars = list(ckpt["vocab"])
        vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})

    train_store = cache_mod.LineStore(root, "train")
    writers = sorted({int(w) for w in train_store.writers})
    model = InkVAE(d=args.d, z_dim=args.z_dim, stride=args.stride, n_mix=args.n_mix,
                   n_chars=len(vocab), n_writers=int(ckpt.get("n_writers") or max(8, len(writers)))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    limit = args.limit_lines or None
    print(f"  aligning {limit or len(train_store)} lines ...")
    result = derive_store_boundaries(model, train_store, vocab, stride=args.stride,
                                     device=device, limit=limit,
                                     min_conf=args.min_conf)

    stats = save_alignment(result, train_store,
                           root / "processed" / "train" / "align.npz",
                           out / "join_stats.json")

    # visuals
    vis = _alignment_visuals(model, train_store, vocab, out / "visuals",
                             stride=args.stride, device=device)
    per_writer = list(stats["per_writer"].values())
    plot_bars(per_writer, out / "visuals" / "join_rate_per_writer.png",
              title="Join rate per writer (derived labels)",
              labels=list(stats["per_writer"].keys()), ylabel="join rate",
              vmax=max(0.5, max(per_writer) if per_writer else 1.0))
    plot_hist(result["boundary_conf"], out / "visuals" / "alignment_confidence.png",
              title="Alignment confidence per boundary",
              xlabel="mean frame posterior")
    vis += [out / "visuals" / "join_rate_per_writer.png",
            out / "visuals" / "alignment_confidence.png"]

    write_metrics(out / "metrics.json", stats)
    print("\n  join statistics")
    print(f"    boundaries        : {stats['n_boundaries']}")
    print(f"    joins             : {stats['n_joins']}  ({stats['join_rate']:.4f})")
    print(f"    lines with a join : {stats['lines_with_a_join']}/{stats['n_lines']}")
    print(f"    writers measured  : {len(stats['per_writer'])}")

    visual_check(vis, "stage 3")
    print("\n  read the images before believing the rate: guides should land between")
    print("  characters, and a 'join' should look like an unbroken stroke across the guide.")

    ok = stats["n_boundaries"] > 0 and stats["join_rate"] > 0.0
    return 0 if gate("G3", ok,
                     f"join rate {stats['join_rate']:.4f} over {stats['n_boundaries']} "
                     f"boundaries", strict=args.strict) else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CENT stage 3: derive boundaries/joins")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--vae", default="runs/stage2_vae/vae.pt")
    ap.add_argument("--out", default="runs/stage3_boundaries")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--n-mix", type=int, default=20)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--force", action="store_true")
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

