"""Train CENT on paired Chinese text lines.

Decisions this file encodes (from docs/CENT_TRAINING_PLAN.md, author-approved):

  canvas 64 x 1024          the line is rendered non-square, so the image target is the line
  max_chars = 32            measured: 40 characters of level-2 glyphs need 547 tokens
  paired lines only         first campaign: exact connectivity supervision + an image target
  verifier enabled          per-character legibility, trained on REAL stroke groups
  lambda_image = 1.0        the offline objective stays at full weight
  published HTR             content metrics come from an external recogniser (benchmark/recogniser.py)

Objectives
  * diffusion noise MSE on token payloads
  * BCE on the per-boundary join decision (the paper's central variable)
  * image L1, rendered at the line canvas, against the paired real line
  * writer classification on the line style vector (real ids, not a constant)
  * per-character legibility: a verifier over each character's stroke group

Usage
  python -m cent.train_lines --config cent/configs/casia_r1.json --dry-run
  python -m cent.train_lines --config cent/configs/casia_r1.json
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
from .data_lines import LineTokenDataset, collate_lines
from .models.diffusion import Diffusion, StrokeVerifier
from .models.groups import build_char_groups
from .models.line import KIND_CHAR_STROKE, KIND_CONNECTOR, LineDenoiser, LineStyleEncoder
from .models.stroke_graph import BezierRenderer


def style_tokens(z_g, z_l, z_k):
    return torch.cat([z_g.unsqueeze(1), z_l, z_k.unsqueeze(1)], dim=1)


def run(cfg: CENTConfig) -> int:
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")
    logf = (out / "train_log.jsonl").open("a", encoding="utf-8")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(getattr(cfg, "amp", "none"))
    amp_ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if (amp_dtype and device == "cuda") \
        else (lambda: torch.autocast("cpu", enabled=False))

    from common.lexicon import load_lexicon
    lexicon = load_lexicon(cfg.lexicon_path)
    ds = LineTokenDataset(cfg.canonical_root, cfg.split_train, lexicon=lexicon,
                          k_refs=cfg.k_refs, img_h=cfg.img_size, canvas_w=cfg.canvas_width,
                          max_chars=cfg.max_chars, max_tokens=cfg.max_tokens,
                          paired_only=cfg.paired_only, limit=cfg.limit, seed=cfg.seed)
    print(f"[data] usable={len(ds)} writers={len(ds.writers)} "
          f"skipped: unpaired={ds.n_unpaired} too_long={ds.n_too_long}")
    if len(ds) == 0:
        raise RuntimeError("no usable lines: need PAIRED lines with strokes within max_chars "
                           f"(paired_only={cfg.paired_only}, max_chars={cfg.max_chars})")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_lines, drop_last=False,
                        num_workers=cfg.num_workers)

    style = LineStyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types,
                             n_heads=cfg.n_heads).to(device)
    den = LineDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads,
                       n_classes=cfg.n_classes, max_tokens=cfg.max_tokens).to(device)
    renderer = BezierRenderer(size=cfg.img_size, width_size=cfg.canvas_width).to(device)
    diff = Diffusion(T=cfg.T).to(device)
    whead = torch.nn.Linear(cfg.dim, cfg.n_writers).to(device)
    verifier = StrokeVerifier(n_classes=cfg.n_classes).to(device) if cfg.lambda_legibility > 0 else None

    params = list(style.parameters()) + list(den.parameters()) + list(whead.parameters())
    if verifier is not None:
        params += list(verifier.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    for epoch in range(cfg.epochs):
        for b in loader:
            t0 = time.time()
            tok = b["tokens"].to(device)
            kind = b["kind"].to(device)
            cpos = b["char_pos"].to(device)
            slot = b["char_slot"].to(device)
            mask = b["mask"].to(device)
            join = b["join"].to(device)
            n_conn = b["n_conn"].to(device)
            refs = b["refs"].to(device)
            rmask = b["ref_mask"].to(device)
            target = b["target_img"].to(device)
            writer = b["writer_idx"].to(device)
            char_ids = b["char_ids"].to(device)

            # classifier-free guidance: drop the character condition on a fraction of rows
            if cfg.drop_char_p > 0:
                drop = torch.rand(tok.shape[0], device=device) < cfg.drop_char_p
                if drop.any():
                    cpos = torch.where(drop.unsqueeze(1), torch.full_like(cpos, cfg.n_classes), cpos)

            with amp_ctx():
                z_g, z_l, z_k, _ = style(refs, rmask)
            z_g, z_l, z_k = z_g.float(), z_l.float(), z_k.float()
            st = style_tokens(z_g, z_l, z_k)

            B = tok.shape[0]
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(tok)
            xt = diff.q_sample(tok, t, noise)
            with amp_ctx():
                eps, join_logits = den(xt, t, kind, cpos, st, mask, connector_mask=mask.bool())
            eps = eps.float()
            join_logits = join_logits.float() if join_logits is not None else None
            loss_diff = F.mse_loss(eps, noise, reduction="none").mul(mask.unsqueeze(-1)).mean()

            # ---- join BCE over real boundaries ----
            conn_mask = torch.zeros_like(join, dtype=torch.bool)
            join_pred = torch.full_like(join, -1e4)
            if join_logits is not None:
                for i in range(B):
                    n = int(n_conn[i])
                    if n == 0:
                        continue
                    idx = ((kind[i] == KIND_CONNECTOR) & mask[i].bool()).nonzero(as_tuple=True)[0]
                    if len(idx):
                        vals = join_logits[i][idx][:n]
                        join_pred[i, : len(vals)] = vals
                        conn_mask[i, : len(vals)] = True
            sel = conn_mask & (join_pred > -1e3)
            loss_join = (F.binary_cross_entropy_with_logits(join_pred[sel], join[sel])
                         if sel.any() else tok.new_zeros(()))

            # ---- one-step estimate x0, then the offline image loss on the line canvas ----
            a = diff.alpha_bar.to(device)[t].view(-1, 1, 1)
            x0 = ((xt - (1 - a).sqrt() * eps) / a.sqrt().clamp_min(1e-5)).clamp(-2, 2)
            img = renderer(x0, mask)
            loss_img = F.l1_loss(img, target)

            # ---- writer classification with REAL ids ----
            loss_w = F.cross_entropy(whead(z_g), writer)

            # ---- per-character legibility on the estimated stroke groups ----
            loss_leg = tok.new_zeros(())
            if verifier is not None:
                G, gm, labels = build_char_groups(x0, kind, slot, mask, char_ids)
                if G is not None:
                    with amp_ctx():
                        leg_logits = verifier(G, gm)
                    loss_leg = F.cross_entropy(leg_logits.float(), labels)

            loss = (loss_diff + cfg.lambda_image * loss_img + loss_join
                    + cfg.lambda_writer * loss_w + cfg.lambda_legibility * loss_leg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            rec = {"step": step, "epoch": epoch, "loss": float(loss.detach()),
                   "diff": float(loss_diff.detach()), "join": float(loss_join.detach()),
                   "img": float(loss_img.detach()), "writer": float(loss_w.detach()),
                   "legib": float(loss_leg.detach()), "sec": round(time.time() - t0, 3),
                   "amp": getattr(cfg, "amp", "none"),
                   "canvas": [cfg.img_size, cfg.canvas_width]}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % 10 == 0:
                print(f"[{epoch:02d} {step:05d}] loss={rec['loss']:.4f} diff={rec['diff']:.4f} "
                      f"join={rec['join']:.4f} img={rec['img']:.4f} leg={rec['legib']:.4f}")
            step += 1
            if cfg.dry_run:
                print("[dry-run] one batch complete; no checkpoint written")
                return 0
        torch.save({"style": style.state_dict(), "denoiser": den.state_dict(),
                    "writer_head": whead.state_dict(),
                    "verifier": verifier.state_dict() if verifier is not None else None,
                    "config": cfg.__dict__, "epoch": epoch, "step": step},
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
