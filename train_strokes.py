"""Train CENT v2 at stroke level on real Chinese line data.

Formulation (see docs/FIRST_RUN_FINDING.md for why the character-decomposed version was abandoned):

    a line = an ordered stroke sequence (Bézier payloads) + its transcript
    a join = the geometric fact that stroke i+1 starts where stroke i ended

Losses
  * diffusion noise MSE on the stroke payloads
  * BCE on the per-boundary connection decision (N-1 decisions per line)
  * image L1 through the splatting renderer against the real scanned line
  * writer classification on the style vector

The per-character legibility term is deliberately absent: without per-character segmentation there
is no way to verify one character at a time. Line-level legibility will come from the frozen HTR at
evaluation time (benchmark/recogniser.py), which is a measurement, not a training signal.

Usage
  python -m cent.train_strokes --config cent/configs/sh_r1.json --dry-run
  python -m cent.train_strokes --config cent/configs/sh_r1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CENTConfig
from .data_strokes import StrokeLineDataset, collate_strokes
from .models.diffusion import Diffusion
from .models.line import LineStyleEncoder
from .models.line_stroke import StrokeDenoiser
from .models.stroke_graph import render_strokes_splat


def run(cfg: CENTConfig) -> int:
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")
    logf = (out / "train_log.jsonl").open("a", encoding="utf-8")

    ds = StrokeLineDataset(cfg.canonical_root, cfg.split_train, k_refs=cfg.k_refs,
                           img_h=cfg.img_size, canvas_w=cfg.canvas_width,
                           max_strokes=cfg.max_tokens, tau=getattr(cfg, "join_tau", 0.35),
                           paired_only=cfg.paired_only, limit=cfg.limit, seed=cfg.seed)
    print(f"[data] {len(ds)} lines, writers={len(ds.writer2idx)} "
          f"skipped: unpaired={ds.n_unpaired} too_short={ds.n_short}")
    if len(ds) == 0:
        raise RuntimeError("no usable lines")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_strokes,
                        num_workers=cfg.num_workers, drop_last=False)

    style = LineStyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types,
                             n_heads=cfg.n_heads).to(device)
    den = StrokeDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads,
                         n_classes=cfg.n_classes, k_refs=cfg.k_refs,
                         max_strokes=cfg.max_tokens).to(device)
    whead = torch.nn.Linear(cfg.dim, cfg.n_writers).to(device)
    diff = Diffusion(T=cfg.T).to(device)

    params = list(style.parameters()) + list(den.parameters()) + list(whead.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    for epoch in range(cfg.epochs):
        for b in loader:
            t0 = time.time()
            tok = b["tokens"].to(device)
            mask = b["mask"].to(device)
            join = b["join"].to(device)
            text = b["text"].to(device)
            tmask = b["text_mask"].to(device)
            refs = b["refs"].to(device)
            rmask = b["ref_mask"].to(device)
            target = b["target_img"].to(device)
            writer = b["writer_idx"].to(device)

            z_g, z_l, z_k, _ = style(refs, rmask)
            st = torch.cat([z_g.unsqueeze(1), z_l, z_k.unsqueeze(1)], dim=1)

            B = tok.shape[0]
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(tok)
            xt = diff.q_sample(tok, t, noise)
            eps = den(xt, t, mask, st, text, tmask,
                      pos=b["pos"].to(device), scalars=b["scalars"].to(device))
            # divide by the number of VALID tokens, not by B*N: padding diluted the loss and
            # made its value incomparable across batches of different length
            se = F.mse_loss(eps, noise, reduction="none").mul(mask.unsqueeze(-1)).sum()
            loss_diff = se / mask.sum().clamp_min(1.0) / eps.shape[-1]

            a = diff.alpha_bar.to(device)[t].view(-1, 1, 1)
            x0 = ((xt - (1 - a).sqrt() * eps) / a.sqrt().clamp_min(1e-5)).clamp(-2, 2)

            # ---- connectivity, supervised as geometry rather than as a classifier ----
            # A per-boundary classifier on the noisy hidden state sat at its prior in three runs,
            # and a supervised probe showed the label was barely predictable from the tokens. But
            # connectivity is simply a property of where a stroke ENDS and the next one STARTS, so
            # we train it directly: joined boundaries are pulled together, lifted ones are pushed
            # past a floor. The label decides which regime applies; it is not a prediction target.
            p0 = x0[..., 0:2]                       # stroke start  (Bezier P0)
            p3 = x0[..., 6:8]                       # stroke end    (Bezier P3)
            gap_abs = (p3[:, :-1] - p0[:, 1:]).norm(dim=-1)      # [B, N-1] canvas units
            # SCALE NORMALISATION. With absolute gaps the loss has a degenerate minimum: shrink
            # every stroke until all gaps are small, and the joined term goes to zero while the
            # lifted term never triggers. Measured: gap_J and gap_L fell together (0.135 / 0.144).
            # Dividing by the line's mean stroke chord removes that shortcut -- shrinking the whole
            # line leaves the ratio unchanged -- so the only way down is to separate the two cases.
            seg = (p3 - p0).norm(dim=-1)                          # [B, N] stroke chord lengths
            chord = ((seg * mask).sum(1) / mask.sum(1).clamp_min(1.0)).clamp_min(1e-3)
            gap = gap_abs / chord[:, None]
            joined = join[:, :-1]                                # 1 = the writer joined here
            valid = (mask[:, :-1] * mask[:, 1:])
            floor = float(getattr(cfg, "continuity_floor", 0.5))   # in units of the mean chord
            wj = joined * valid
            wl = (1.0 - joined) * valid
            loss_join = ((gap.pow(2) * wj).sum() / wj.sum().clamp_min(1.0)
                         + (torch.relu(floor - gap).pow(2) * wl).sum() / wl.sum().clamp_min(1.0))

            img = render_strokes_splat(x0, mask, cfg.img_size, cfg.canvas_width)
            # CONVENTION: the corpus stores ink=0 / paper=1; the rasteriser returns ink=1 / paper=0.
            # Comparing them directly compares INVERTED images: a blank render scores 0.015 while
            # the loss sits at ~0.98 and can never fall (observed in run 1).
            ink = (target < 0.5).float()
            lw = 1.0 + float(getattr(cfg, "ink_weight", 20.0)) * ink
            l1 = ((1.0 - img - target).abs() * lw).mean() / lw.mean()
            p_in = img.flatten(1)
            t_in = ink.flatten(1)
            dice = (2 * (p_in * t_in).sum(1) + 1.0) / (p_in.sum(1) + t_in.sum(1) + 1.0)
            per_row = ((1.0 - img - target).abs().mean(dim=(1, 2, 3))
                       + float(getattr(cfg, "dice_weight", 0.5)) * (1.0 - dice))
            # The image term is only meaningful where x0 is a real estimate. At t=190/200 the
            # predicted clean strokes are noise, so the render cannot match anything and the loss
            # plateaus at the high-noise average -- measured: overfitting ONE line left it at 0.52
            # while the vector loss fell to 0.09. Weight it to low noise, where the geometry exists.
            w_img = (t < 50).float()
            loss_img = ((per_row * w_img).sum() / w_img.sum().clamp_min(1.0)
                        if float(w_img.sum()) > 0 else per_row.mean() * 0.0)
            loss_w = F.cross_entropy(whead(z_g), writer)

            loss = (loss_diff + cfg.lambda_image * loss_img + loss_join
                    + cfg.lambda_writer * loss_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            rec = {"step": step, "epoch": epoch, "loss": float(loss.detach()),
                   "diff": float(loss_diff.detach()), "join": float(loss_join.detach()),
                   "img": float(loss_img.detach()), "writer": float(loss_w.detach()),
                   "gap_joined": float((gap * wj).sum() / wj.sum().clamp_min(1.0)),
                   "gap_lifted": float((gap * wl).sum() / wl.sum().clamp_min(1.0)),
                   "gap_joined_abs": float((gap_abs * wj).sum() / wj.sum().clamp_min(1.0)),
                   "gap_lifted_abs": float((gap_abs * wl).sum() / wl.sum().clamp_min(1.0)),
                   "join_rate_gt": float((joined * valid).sum() / valid.sum().clamp_min(1.0)),
                   "sec": round(time.time() - t0, 3), "formulation": "stroke-level"}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % 10 == 0:
                print(f"[{epoch:02d} {step:05d}] loss={rec['loss']:.4f} diff={rec['diff']:.4f} "
                      f"cont={rec['join']:.4f} img={rec['img']:.4f} "
                      f"gap_J={rec['gap_joined']:.4f} gap_L={rec['gap_lifted']:.4f}")
            step += 1
            if cfg.dry_run:
                print("[dry-run] one batch complete")
                return 0
        torch.save({"style": style.state_dict(), "denoiser": den.state_dict(),
                    "writer_head": whead.state_dict(), "config": cfg.__dict__,
                    "epoch": epoch, "step": step, "formulation": "stroke-level+endpoint-continuity"},
                   out / "ckpt_last.pt")
        print(f"[epoch {epoch}] -> {out/'ckpt_last.pt'}")
    logf.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = CENTConfig.load(a.config)
    if a.dry_run:
        cfg.dry_run = True
    if a.device:
        cfg.device = a.device
    if a.limit is not None:
        cfg.limit = a.limit
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
