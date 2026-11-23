"""Training loop for CENT.

Loss = diffusion noise MSE
     + lambda_image      * L1( render(x0_hat), target image )      [dual-decode supervision]
     + lambda_legibility * CE( verifier(x0_hat), char )             [training-time only]
     + lambda_writer     * CE( writer_head(z_g), writer )           [style encoder supervision]

x0_hat is the one-step denoising estimate from the predicted noise, so all three
auxiliary terms act on geometry the denoiser actually produces.

The image target is a real glyph of the same character by the same writer but a
DIFFERENT act of writing (see cent/data.py). The image loss is therefore a soft
style-consistency objective, not a reconstruction; it is weighted below the
diffusion term by design.

Usage
  python -m cent.train --config cent/configs/cent_r1.json
  python -m cent.train --config ... --dry-run        # shapes only, no checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CENTConfig
from .data import PairedStrokeDataset, collate
from .models.diffusion import Diffusion, StrokeVerifier
from .models.stroke_graph import BezierRenderer
from .models.style import StrokeDenoiser, StyleEncoder


def build(cfg: CENTConfig, device: str):
    style = StyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types, n_heads=cfg.n_heads).to(device)
    denoiser = StrokeDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads,
                              n_classes=cfg.n_classes, n_types=cfg.n_types).to(device)
    verifier = StrokeVerifier(n_classes=cfg.n_classes, dim=cfg.dim // 2).to(device)
    renderer = BezierRenderer(size=cfg.img_size).to(device)
    diffusion = Diffusion(T=cfg.T).to(device)
    writer_head = torch.nn.Linear(cfg.dim, cfg.n_writers).to(device)
    return style, denoiser, verifier, renderer, diffusion, writer_head


def style_tokens_from(z_g: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
    return torch.cat([z_g.unsqueeze(1), z_l], dim=1)     # [B,1+n_types,D]


def x0_from_eps(xt: torch.Tensor, eps: torch.Tensor, ab,
                t: torch.Tensor) -> torch.Tensor:
    a = ab[t].view(-1, 1, 1)
    return ((xt - (1 - a).sqrt() * eps) / a.sqrt().clamp_min(1e-5)).clamp(-2, 2)


def train(cfg: CENTConfig) -> int:
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")

    ds = PairedStrokeDataset(cfg.canonical_root, cfg.split_train, k_refs=cfg.k_refs,
                             img_size=cfg.img_size, seed=cfg.seed)
    if cfg.limit:
        ds.keys = ds.keys[: cfg.limit]
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=cfg.num_workers, drop_last=True)
    print(f"[data] split={cfg.split_train} pairs={len(ds)} classes={len(ds.charset)} "
          f"img={ds.img_size}")

    style, denoiser, verifier, renderer, diffusion, writer_head = build(cfg, device)
    params = (list(style.parameters()) + list(denoiser.parameters()) +
              list(verifier.parameters()) + list(writer_head.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs * len(loader)))

    step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            t0 = time.time()
            tok = batch["tokens"].to(device)                 # [B,N,8]
            mask = batch["stroke_mask"].to(device)           # [B,N]
            char = batch["char_idx"].to(device)
            refs = batch["refs"].to(device)                  # [B,k,1,S,S]
            rmask = batch["ref_mask"].to(device)
            target = batch["target_img"].to(device)          # [B,1,S,S]
            has_t = batch["has_target"].to(device)

            z_g, z_l, _ = style(refs, rmask)
            st = style_tokens_from(z_g, z_l)

            # diffusion term
            B = tok.shape[0]
            t = torch.randint(0, diffusion.T, (B,), device=device)
            noise = torch.randn_like(tok)
            xt = diffusion.q_sample(tok, t, noise)
            eps_hat = denoiser(xt, t, char, st, mask)
            loss_diff = F.mse_loss(eps_hat, noise, reduction="none").mul(mask.unsqueeze(-1)).mean()

            # auxiliary terms on the one-step estimate
            x0 = x0_from_eps(xt, eps_hat, diffusion.alpha_bar.to(device), t)
            img = renderer(x0, mask)
            loss_img = (F.l1_loss(img, target, reduction="none").mean(dim=(1, 2, 3)) * has_t).mean()
            logits = verifier(x0, mask)
            loss_leg = F.cross_entropy(logits, char)
            # The verifier must also see REAL stroke graphs, or it learns to
            # recognise the denoiser's artefacts instead of the character.
            loss_ver = F.cross_entropy(verifier(tok, mask), char)
            loss_w = F.cross_entropy(writer_head(z_g), batch["writer"].to(device))

            loss = (loss_diff + cfg.lambda_image * loss_img
                    + cfg.lambda_legibility * loss_leg
                    + cfg.lambda_verifier_real * loss_ver
                    + cfg.lambda_writer * loss_w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()

            rec = {"step": step, "epoch": epoch, "loss": float(loss.detach()),
                   "diff": float(loss_diff.detach()), "img": float(loss_img.detach()),
                   "leg": float(loss_leg.detach()), "ver": float(loss_ver.detach()),
                   "wr": float(loss_w.detach()),
                   "sec": round(time.time() - t0, 3)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 20 == 0:
                print(f"[{epoch:02d} {step:06d}] loss={rec['loss']:.4f} diff={rec['diff']:.4f} "
                      f"img={rec['img']:.4f} leg={rec['leg']:.3f} wr={rec['wr']:.3f}")
            step += 1
            if cfg.dry_run:
                print("[dry-run] one batch complete; no checkpoint written")
                return 0

        ckpt = {"style": style.state_dict(), "denoiser": denoiser.state_dict(),
                "verifier": verifier.state_dict(), "writer_head": writer_head.state_dict(),
                "config": cfg.__dict__, "epoch": epoch, "step": step}
        torch.save(ckpt, out / f"ckpt_e{epoch:02d}.pt")
        torch.save(ckpt, out / "ckpt_last.pt")
        print(f"[epoch {epoch}] checkpoint -> {out/'ckpt_last.pt'}")
    log.close()
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
    return train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
