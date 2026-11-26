"""Stage 4/5 -train BoundaryDiT on frozen VAE latents, then calibrate control.

Supervision comes from Stage 3: point-level character ids give the frame layout,
and the derived join labels supervise the boundary head. The curriculum follows
the plan -2- character phrases first (boundary-dense), then whole lines.

An honest limitation, printed at runtime: training uses the *aligned* frame
layout, while inference has only the text and therefore uses an even split. That
mismatch is itself an ablation (`--layout even` makes training match inference).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.tokens import CharVocab
from ..losses import boundary_loss, masked_mean
from ..model import CENTModel
from .common import gmm_to_points, points_to_strokes
from ..viz import (backend_name, figure_panels, plot_bars, plot_curves, render_strokes,
                   save_png,
                   render_strokes, save_png, vstack_padded)
from . import cache as cache_mod
from . import fetch, perf
from .align import load_alignment
from .common import (banner, gate, pick_device, points_to_strokes, seed_everything,
                     visual_check, write_metrics)


# ---------------------------------------------------------------- data


class DitDataset(Dataset):
    def __init__(self, root: Path, split: str, vocab: CharVocab, *, phase: str = "lines",
                 limit: int | None = None, seed: int = 0):
        self.store = cache_mod.LineStore(root, split)
        self.align = load_alignment(root, split)
        if self.align is None:
            raise RuntimeError("align.npz missing -run stage 3 first")
        self.vocab = vocab
        self.phase = phase
        self.rng = np.random.default_rng(seed)
        self.lines = []
        for i in range(len(self.store)):
            lo, hi = int(self.store.offsets[i]), int(self.store.offsets[i + 1])
            cid = self.align["char_ids"][lo:hi]
            if (cid >= 0).sum() < 8:
                continue
            self.lines.append(i)
        if limit:
            self.lines = self.lines[:limit]

    def __len__(self) -> int:
        return len(self.lines)

    def _boundaries(self, i: int):
        m = self.align["boundary_line"] == i
        return self.align["boundary_points"][m], self.align["boundary_join"][m]

    def __getitem__(self, k: int) -> dict:
        i = self.lines[k]
        lo, hi = int(self.store.offsets[i]), int(self.store.offsets[i + 1])
        pts = self.store.points[lo:hi].astype(np.float32)
        sid = self.store.stroke_ids[lo:hi].astype(np.int64)
        cid = self.align["char_ids"][lo:hi].astype(np.int64)
        bpts, bjoin = self._boundaries(i)
        text = str(self.store.texts[i])

        c0 = 0
        c1 = int(cid.max()) + 1 if (cid >= 0).any() else 0
        if self.phase == "phrases" and c1 > 3:
            span = int(self.rng.integers(2, min(5, c1 - 1) + 1))
            start = int(self.rng.integers(0, max(1, c1 - span)))
            starts = [int(np.argmax(cid >= c)) for c in range(c1)]
            a = starts[start]
            b = starts[start + span] if start + span < c1 else len(pts)
            pts, sid, cid = pts[a:b], sid[a:b], cid[a:b] - start
            keep = (bpts >= a) & (bpts < b)
            bpts, bjoin = bpts[keep] - a, bjoin[keep]
            c0, c1 = 0, span
            text = text[start : start + span + 1]

        p = pts - pts.min(axis=0, keepdims=True)
        h = float(p[:, 1].max() - p[:, 1].min()) or 1.0
        p = p / h
        dxy = np.zeros_like(p)
        dxy[1:] = p[1:] - p[:-1]
        pen = np.full(len(p), 0, dtype=np.int64)
        pen[1:] = np.where(sid[1:] == sid[:-1], 0, 1)

        return {"dxy": dxy.astype(np.float32), "pen": pen, "char_ids": cid,
                "char_seq": np.asarray(self.vocab.encode(text), dtype=np.int64),
                "bnd_points": bpts.astype(np.int64),
                "bnd_join": bjoin.astype(np.float32),
                "text": text, "writer": int(self.store.writers[i]), "index": i}


def collate_dit(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["dxy"].shape[0] for b in batch)
    L = max(b["char_seq"].shape[0] for b in batch)
    K = max(len(b["bnd_points"]) for b in batch)
    dxy = torch.zeros(B, T, 2)
    pen = torch.zeros(B, T, dtype=torch.long)
    cid = torch.full((B, T), -1, dtype=torch.long)
    pm = torch.zeros(B, T, dtype=torch.bool)
    cs = torch.zeros(B, L, dtype=torch.long)
    cm = torch.zeros(B, L, dtype=torch.bool)
    bp = torch.full((B, K), -1, dtype=torch.long)
    bj = torch.zeros(B, K)
    bm = torch.zeros(B, K, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["dxy"].shape[0]
        dxy[i, :t] = torch.from_numpy(b["dxy"])
        pen[i, :t] = torch.from_numpy(b["pen"])
        cid[i, :t] = torch.from_numpy(b["char_ids"])
        pm[i, :t] = True
        l = b["char_seq"].shape[0]
        cs[i, :l] = torch.from_numpy(b["char_seq"])
        cm[i, :l] = True
        k = len(b["bnd_points"])
        bp[i, :k] = torch.from_numpy(b["bnd_points"])
        bj[i, :k] = torch.from_numpy(b["bnd_join"])
        bm[i, :k] = True
    return {"dxy": dxy, "pen": pen, "char_ids": cid, "point_mask": pm,
            "char_seq": cs, "char_mask": cm, "bnd_points": bp, "bnd_join": bj,
            "bnd_mask": bm, "texts": [b["text"] for b in batch],
            "writers": [b["writer"] for b in batch], "index": [b["index"] for b in batch]}


def _layout(cid: torch.Tensor, pmask: torch.Tensor, bnd_points: torch.Tensor,
            bnd_join: torch.Tensor, bnd_mask: torch.Tensor, stride: int):
    """Point-level char ids + boundaries -> frame layout and boundary targets."""
    B, T = cid.shape
    T_lat = max(1, (T + stride - 1) // stride)
    fchar = torch.full((B, T_lat), -1, dtype=torch.long)
    fmask = torch.zeros(B, T_lat, dtype=torch.bool)
    # boundary slots live *between* frames, so there are T_lat-1 of them
    labels = torch.zeros(B, max(1, T_lat - 1), dtype=torch.float32)
    lmask = torch.zeros(B, max(1, T_lat - 1), dtype=torch.bool)
    for b in range(B):
        t = int(pmask[b].sum())
        for f in range(min(T_lat, (t + stride - 1) // stride)):
            lo, hi = f * stride, min(t, (f + 1) * stride)
            if lo >= hi:
                break
            chunk = cid[b, lo:hi]
            chunk = chunk[chunk >= 0]
            if chunk.numel() == 0:
                continue
            vals, counts = torch.unique(chunk, return_counts=True)
            fchar[b, f] = vals[torch.argmax(counts)]
            fmask[b, f] = True
        for k in range(int(bnd_mask[b].sum())):
            p = int(bnd_points[b, k])
            f = min(max(0, T_lat - 2), p // stride)
            labels[b, f] = bnd_join[b, k]
            lmask[b, f] = True
    dev = cid.device
    return (fchar.to(dev), fmask.to(dev), labels.to(dev), lmask.to(dev))


# ---------------------------------------------------------------- train


def run(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = pick_device(args.device)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    banner("STAGE 4 -BoundaryDiT (frozen VAE)")
    print(f"  device: {device}   phase: {args.phase}   layout: {args.layout}")
    print(f"  figure backend: {backend_name()}")
    print("  NOTE: training uses the aligned frame layout; sampling uses an even split.")
    print("        This train/infer mismatch is an ablation (--layout even), not a detail.")

    fetch.ensure_wptt(root, allow_download=not args.no_download)
    cache_mod.build_cache(root, limit_pages=args.limit_pages, force=args.rebuild_cache)

    vae_ckpt = Path(args.vae)
    if not vae_ckpt.exists():
        print(f"  missing {vae_ckpt} -run stage 2 first")
        return 2
    ck = torch.load(vae_ckpt, map_location="cpu")
    chars = list(ck.get("vocab") or [])
    vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})

    ds = DitDataset(root, "train", vocab, phase=args.phase,
                    limit=args.limit_lines or None, seed=args.seed)

    writers = sorted({int(w) for w in cache_mod.LineStore(root, "train").writers})
    model = CENTModel(n_chars=max(64, len(vocab)), n_writers=int(ck.get("n_writers") or max(8, len(writers))),
                      d=args.d, z_dim=args.z_dim, vae_d=args.vae_d, stride=args.stride,
                      dit_depth=args.depth, dit_heads=args.heads, n_mix=args.n_mix,
                      n_steps=1000).to(device)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    if args.resume and not args.eval_only:
        rs0 = torch.load(args.resume, map_location="cpu")
        if rs0.get("optimizer"):
            opt.load_state_dict(rs0["optimizer"])
        print(f"  [resume] continuing from step {start_step}")
    start_step = 0
    if args.resume:
        rs = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(rs["model"])
        start_step = int(rs.get("step") or 0)
        print(f"  [resume] loaded {args.resume} (step {rs.get('step')})")
        if args.eval_only:
            ds_v = DitDataset(root, "train", vocab, phase="lines", limit=args.limit_lines or 6)
            vis, curve = _sample_visuals(model, ds_v, device, out / "visuals", args)
            print("  [eval-only] control curve:", curve)
            visual_check(vis, "stage 4 (eval-only)")
            return 0
    pinfo = perf.setup_perf(device, amp=args.amp)
    print(f"  vocab={len(vocab)} writers={len(writers)} lines={len(ds)} "
          f"trainable params={sum(p.numel() for p in trainable)/1e6:.2f} M")
    print(f"  perf: {perf.describe(pinfo)}")

    lk = perf.loader_kwargs(device, args.workers)
    if args.cache_latents:
        cache = _build_latent_cache(model, ds, device, layout=args.layout)
        print(f"  [cache] {len(cache)} lines encoded once (VAE is frozen)")
        dl = DataLoader(cache, batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate_cached, **lk)
    else:
        cache = None
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_dit, **lk)

    # held-out writers, used only for early stopping
    val_cache = None
    if args.eval_every:
        try:
            ds_val = DitDataset(root, "val", vocab, phase="lines",
                                limit=args.limit_val or 400, seed=args.seed)
            val_cache = _build_latent_cache(model, ds_val, device, log_every=0, layout=args.layout)
            print(f"  [val] {len(val_cache)} held-out lines for F1 checkpointing")
        except Exception as e:  # noqa: BLE001
            print(f"  [val] skipped ({type(e).__name__}: {e})")

    hist = {"total": [], "diff": [], "bnd": [], "bnd_acc": []}
    best_f1 = float("-inf")
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        for batch in dl:
            batch = perf.to_device(batch, device)
            if cache is not None:
                z0 = batch["z0"]
                fchar, fmask = batch["fchar"], batch["fmask"]
                labels, lmask = batch["labels"], batch["lmask"]
                ctx, style = batch["ctx"], batch["style"]
            else:
                x = torch.cat([batch["dxy"],
                               torch.nn.functional.one_hot(batch["pen"], 4).float()], -1)
                with torch.no_grad():
                    mu, _, lm = model.vae.encode(x, batch["point_mask"])
                z0 = mu
                if args.layout == "even":
                    nch = batch["char_mask"].sum(1).tolist()
                    fcs, fms = [], []
                    for b in range(mu.shape[0]):
                        fc1, fm1 = _even_frame_char(int(nch[b]), mu.shape[1], device,
                                                    model.stride)
                        fcs.append(fc1[0])
                        fms.append(fm1[0])
                    fchar = torch.stack(fcs)
                    fmask = torch.stack(fms)
                    labels = torch.zeros(fchar.shape[0], max(1, fchar.shape[1] - 1),
                                         device=device)
                    lmask = torch.zeros(fchar.shape[0], max(1, fchar.shape[1] - 1),
                                        dtype=torch.bool, device=device)
                else:
                    fchar, fmask, labels, lmask = _layout(
                        batch["char_ids"], batch["point_mask"], batch["bnd_points"],
                        batch["bnd_join"], batch["bnd_mask"], args.stride)
                with perf.autocast(device, pinfo["amp"]):
                    ctx = model.text(batch["char_seq"], batch["char_mask"])
                style = model.style_proj(
                    (z0 * lm.unsqueeze(-1)).sum(1)
                    / lm.sum(1).clamp_min(1.0).unsqueeze(-1))

            t = torch.randint(0, model.n_steps, (z0.shape[0],), device=device)
            noise = torch.randn_like(z0)
            a = model.alphas_cumprod[t].view(-1, 1, 1)
            z_t = a.sqrt() * z0 + (1 - a).sqrt() * noise
            with perf.autocast(device, pinfo["amp"]):
                outd = model.dit(z_t, t, frame_char=fchar, frame_mask=fmask,
                                 char_ctx=ctx, style=style)
            l_diff = masked_mean((outd["x0"].float() - z0) ** 2, fmask)
            # the DiT exposes one boundary slot per frame *transition*; crop/align
            # defensively in case the layout and the VAE rounding differ by one
            bv = outd["bnd_valid"]
            W = bv.shape[1]
            lab, lm = labels[:, :W], lmask[:, :W]
            if lm.shape[1] < W:
                pad = W - lm.shape[1]
                lab = torch.cat([lab, torch.zeros(lab.shape[0], pad, device=lab.device)], 1)
                lm = torch.cat([lm, torch.zeros(lm.shape[0], pad, dtype=torch.bool,
                                                device=lm.device)], 1)
            mask = lm & bv
            l_bnd = boundary_loss(outd["bnd_logits"].float(), lab, mask,
                                  pos_weight=args.pos_weight)
            total = l_diff + args.w_bnd * l_bnd

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1

            acc = float(((outd["bnd_logits"][mask] > 0).float()
                         == lab[mask]).float().mean()) if mask.any() else float("nan")
            for k, v in (("total", total), ("diff", l_diff), ("bnd", l_bnd)):
                hist[k].append(float(v.detach()))
            hist["bnd_acc"].append(acc)

            if args.eval_every and step % args.eval_every == 0:
                f1 = _eval_boundary_f1(model, val_cache, device)
                hist.setdefault("val_f1", []).append(float(f1))
                # always keep a rolling checkpoint, so stopping early never
                # throws the run away even when no validation split is available
                torch.save({"model": model.state_dict(), "vocab": chars,
                            "n_writers": int(ck.get("n_writers") or len(writers)),
                            "optimizer": opt.state_dict(), "step": step,
                            "val_f1": float(f1) if f1 == f1 else None,
                            "args": vars(args)}, out / "dit_last.pt")
                better = (f1 > best_f1) if (f1 == f1 and best_f1 == best_f1) else (f1 == f1)
                if better:
                    best_f1 = f1
                    torch.save({"model": model.state_dict(), "vocab": chars,
                                "n_writers": int(ck.get("n_writers") or len(writers)),
                                "optimizer": opt.state_dict(),
                                "step": step, "val_f1": float(f1),
                                "args": vars(args)}, out / "dit_best.pt")
                _quick_sample(model, val_cache or cache, device,
                              out / "visuals" / f"sample_step{step:06d}.png")
                print(f"  [eval] step {step:6d}  val boundary F1={f1:.4f}"
                      f"  best={best_f1:.4f}  ({'saved' if better else 'kept'})")
            if step % args.log_every == 0:
                print(f"  ep{epoch} step{step:5d} total={float(total):.3f} "
                      f"diff={float(l_diff):.3f} bnd={float(l_bnd):.3f} "
                      f"acc={acc:.3f} ({(time.time()-t0)/step*1000:.0f} ms/step)")
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break

    ckpt = out / "dit.pt"
    torch.save({"model": model.state_dict(), "vocab": chars, "args": vars(args)}, ckpt)

    # ---- visuals: samples at three control strengths + control curve
    vis, curve = _sample_visuals(model, ds, device, out / "visuals", args)
    plot_curves({"diffusion (latent MSE)": hist["diff"], "boundary BCE": hist["bnd"]},
                out / "visuals" / "loss_curves.png", title="Stage 4 training loss",
                smooth=20)
    accs = [a for a in hist["bnd_acc"] if a == a][-200:]
    plot_bars(accs, out / "visuals" / "boundary_acc.png",
              title="Boundary head accuracy (recent steps)", ylabel="accuracy",
              vmax=1.0, hline=0.5, hline_label="chance")
    vis += [out / "visuals" / "loss_curves.png", out / "visuals" / "boundary_acc.png"]

    last_acc = [a for a in hist["bnd_acc"] if a == a]
    write_metrics(out / "metrics.json", {
        "stage": 4, "steps": step, "final_total": hist["total"][-1] if hist["total"] else None,
        "final_diff": hist["diff"][-1] if hist["diff"] else None,
        "final_bnd": hist["bnd"][-1] if hist["bnd"] else None,
        "boundary_acc": last_acc[-1] if last_acc else None,
        "control_curve": curve, "checkpoint": str(ckpt),
    })
    visual_check(vis, "stage 4")
    print("\n  inspect: do joins look like strokes that continue, or like glyphs bent")
    print("  to touch? The latter is the failure mode this architecture exists to avoid.")

    ok = bool(last_acc) and last_acc[-1] > 0.5
    return 0 if gate("G4", ok, f"boundary acc {last_acc[-1] if last_acc else float('nan'):.3f}",
                     strict=args.strict) else 1


def _sample_visuals(model, ds, device, out_dir: Path, args):
    """Render generated lines for control in {0, 0.5, 1} + the calibration curve."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    paths_out, curve = [], {}
    n = min(3, len(ds))
    with torch.no_grad():
        for k in range(n):
            item = ds[k]
            cs = torch.from_numpy(item["char_seq"])[None].to(device)
            cm = torch.ones_like(cs, dtype=torch.bool)
            style = torch.randn(1, model.style_proj.out_features, device=device)
            opts = []
            for c in (0.0, 0.5, 1.0):
                out = model.generate(cs, cm, style, control=c, steps=args.sample_steps,
                                     n_chars_per_frame=args.n_chars_per_frame)
                dec = model.decode(out["z"])
                pts = dec["points"][0].cpu().numpy()
                pen = dec["pen"][0].cpu().numpy()
                opts.append(render_strokes(points_to_strokes(pts, pen)))
                if k == 0:
                    v = out["bnd_valid"]
                    committed = out["committed"]
                    rate = float((committed[v] > 0.5).float().mean()) if v.any() else float("nan")
                    curve[str(c)] = rate
            p = out_dir / f"samples_control_{k}.png"
            figure_panels([("control=0.0 — never join", opts[0]),
                           ("control=0.5 — model prior", opts[1]),
                           ("control=1.0 — always join", opts[2])],
                          p, title=f"generated line {k}: cursiveness sweep")
            paths_out.append(p)
    plot_bars(list(curve.values()), out_dir / "control_curve.png",
              title="realised join rate vs control", labels=list(curve.keys()),
              ylabel="realised join rate", vmax=1.0, hline=0.5,
              hline_label="requested 0.5")
    model.train()
    return paths_out + [out_dir / "control_curve.png"], curve



def _even_frame_char(n_chars: int, T_lat: int, device, stride: int = 8):
    """Character id per latent frame, evenly split.

    This is the layout `generate()` can reproduce at inference time. Training on
    the Viterbi-aligned layout instead was the mismatch that produced scribbles.
    """
    fc = torch.full((1, T_lat), -1, dtype=torch.long, device=device)
    fm = torch.zeros(1, T_lat, dtype=torch.bool, device=device)
    if n_chars <= 0:
        return fc, fm
    per = T_lat / float(n_chars)
    idx = (torch.arange(T_lat, device=device) / per).long().clamp(0, n_chars - 1)
    fc[0] = idx
    fm[0] = True
    return fc, fm


def _build_latent_cache(model, ds, device, *, batch_size: int = 32, log_every: int = 200,
                        layout: str = "aligned"):
    """Encode every training line once (VAE frozen) and cache the DiT inputs.

    Removes the per-step VAE encode *and* the per-step python layout building,
    which dominate wall-clock for this model size. Numerics are unchanged: the
    cached tensors are exactly what the loop computed before (mu, not a sample).
    """
    items = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            raw = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            batch = collate_dit(raw)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            x = torch.cat([batch["dxy"],
                               torch.nn.functional.one_hot(batch["pen"], 4).float()], -1)
            with perf.autocast(device, True):
                mu, _, lm = model.vae.encode(x, batch["point_mask"])
            if layout == "even":
                nch = batch["char_mask"].sum(1).tolist()
                fcs, fms = [], []
                for b in range(mu.shape[0]):
                    fc1, fm1 = _even_frame_char(int(nch[b]), mu.shape[1], device, model.stride)
                    fcs.append(fc1[0]); fms.append(fm1[0])
                fchar = torch.stack(fcs); fmask = torch.stack(fms)
                labels = torch.zeros(fchar.shape[0], max(1, fchar.shape[1] - 1), device=device)
                lmask = torch.zeros(fchar.shape[0], max(1, fchar.shape[1] - 1),
                                    dtype=torch.bool, device=device)
            else:
                fchar, fmask, labels, lmask = _layout(
                batch["char_ids"], batch["point_mask"], batch["bnd_points"],
                batch["bnd_join"], batch["bnd_mask"], model.stride)
            style = model.style_proj(
                (mu * lm.unsqueeze(-1)).sum(1) / lm.sum(1).clamp_min(1.0).unsqueeze(-1))
            with perf.autocast(device, True):
                ctx = model.text(batch["char_seq"], batch["char_mask"])
            for b in range(mu.shape[0]):
                tl = int(fmask[b].sum())
                items.append({
                    "z0": mu[b, :tl].float().cpu(),
                    "fchar": fchar[b, :tl].cpu(),
                    "fmask": fmask[b, :tl].cpu(),
                    "labels": labels[b].float().cpu(),
                    "lmask": lmask[b].cpu(),
                    "ctx": ctx[b].float().cpu(),
                    "style": style[b].float().cpu(),
                })
            if log_every and start % (log_every * batch_size) == 0:
                print(f"  [cache] {min(start + batch_size, len(ds))}/{len(ds)} lines")
    model.train()
    return items


def _collate_cached(batch: list[dict]) -> dict:
    B = len(batch)
    Tl = max(b["z0"].shape[0] for b in batch)
    L = max(b["ctx"].shape[0] for b in batch)
    z_dim = batch[0]["z0"].shape[1]
    d = batch[0]["ctx"].shape[1]
    z0 = torch.zeros(B, Tl, z_dim)
    fchar = torch.full((B, Tl), -1, dtype=torch.long)
    fmask = torch.zeros(B, Tl, dtype=torch.bool)
    Tb = max(1, Tl - 1)          # boundary slots live between frames
    labels = torch.zeros(B, Tb, dtype=torch.float32)
    lmask = torch.zeros(B, Tb, dtype=torch.bool)
    ctx = torch.zeros(B, L, d)
    style = torch.zeros(B, d)
    for i, b in enumerate(batch):
        t = b["z0"].shape[0]
        z0[i, :t] = b["z0"]
        fchar[i, :t] = b["fchar"]
        fmask[i, :t] = b["fmask"]
        kl = min(b["labels"].shape[0], Tb)
        labels[i, :kl] = b["labels"][:kl]
        lmask[i, :kl] = b["lmask"][:kl]
        l = b["ctx"].shape[0]
        ctx[i, :l] = b["ctx"]
        style[i] = b["style"]
    return {"z0": z0, "fchar": fchar, "fmask": fmask, "labels": labels,
            "lmask": lmask, "ctx": ctx, "style": style}



def _eval_boundary_f1(model, val_cache, device, *, batch_size=16, n_batches=12):
    """Boundary F1 on held-out writers. The DiT is trained on a noised latent, so
    evaluate at a small fixed t with no noise for a stable, comparable signal."""
    if not val_cache:
        return float("nan")
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for start in range(0, min(len(val_cache), batch_size * n_batches), batch_size):
            batch = _collate_cached(val_cache[start:start + batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            z0 = batch["z0"]
            B = z0.shape[0]
            t = torch.zeros(B, dtype=torch.long, device=device)
            out = model.dit(z0, t, frame_char=batch["fchar"], frame_mask=batch["fmask"],
                            char_ctx=batch["ctx"], style=batch["style"])
            mask = batch["lmask"] & out["bnd_valid"]
            pred = (out["bnd_logits"][mask] > 0)
            lab = batch["labels"][mask] > 0.5
            tp += int((pred & lab).sum())
            fp += int((pred & ~lab).sum())
            fn += int((~pred & lab).sum())
    model.train()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec)



@torch.no_grad()
def _quick_sample(model, cache, device, path, *, steps=20):
    """One line, DDIM from noise, rendered. Uses the cached real style + layout."""
    if not cache:
        return None
    item = _collate_cached(cache[:1])
    item = {k: v.to(device) for k, v in item.items()}
    z = torch.randn(item["z0"].shape, device=device)
    ts = torch.linspace(model.n_steps - 1, 0, steps).long().clamp(0, model.n_steps - 1)
    for i, t in enumerate(ts):
        tb = torch.full((1,), int(t), dtype=torch.long, device=device)
        out = model.dit(z, tb, frame_char=item["fchar"], frame_mask=item["fmask"],
                        char_ctx=item["ctx"], style=item["style"])
        a = model.alphas_cumprod[t]
        an = (model.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts)
              else torch.tensor(1.0, device=device))
        eps = (z - a.sqrt() * out["x0"]) / (1 - a).sqrt().clamp_min(1e-6)
        z = an.sqrt() * out["x0"] + (1 - an).sqrt() * eps
    gmm, penl = model.vae.decode(z)
    pts = torch.cumsum(gmm_to_points(gmm), dim=1)[0].cpu().numpy()
    pen = penl[0, : len(pts)].argmax(-1).cpu().numpy()
    return save_png(render_strokes(points_to_strokes(pts, pen), height=120), path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CENT stage 4: BoundaryDiT")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--vae", default="runs/stage2_vae/vae.pt")
    ap.add_argument("--out", default="runs/stage4_dit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--phase", default="lines", choices=["lines", "phrases"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--vae-d", type=int, default=128,
                    help="width of the frozen VAE; independent of the DiT --d")
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-mix", type=int, default=20)
    ap.add_argument("--w-bnd", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--n-chars-per-frame", type=int, default=2)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    # measured: bf16 HURTS this model (14.7 -> 19.4 ms/step at depth 3, d=128)
    # because it is latency-bound, not compute-bound. Off by default.
    ap.add_argument("--amp", dest="amp", action="store_true", default=False)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--cache-latents", dest="cache_latents", action="store_true",
                    default=True, help="encode each line once (default on)")
    ap.add_argument("--no-cache-latents", dest="cache_latents", action="store_false")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--layout", default="even", choices=["even", "aligned"])
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--eval-only", dest="eval_only", action="store_true")
    ap.add_argument("--limit-val", type=int, default=400)
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

