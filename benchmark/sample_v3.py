"""Sample lines from a CENT v3 checkpoint and write a contact sheet.

Decoding is autoregressive over strokes. The boundary contract is enforced explicitly: for
each character boundary, M4 predicts a join decision and an anchor gap, and the first stroke of
the next character is *forced* to begin at `exit + gap * (1 - join)`. A joined boundary
therefore has a zero start offset by construction rather than by hope -- this is the property
the paper claims, and it is visible in this file as a single line of code.

Usage
    python -m benchmark.sample_v3 --config runs/v3/config.json \
        --ckpt runs/v3/ckpt_last.pt --split val --n 3 --out runs/v3/sheet.png

Not yet executed; see docs/ARCHITECTURE_V3.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3, collate_v3
from cent.models.stroke_graph import render_strokes_splat
from cent.models.v3 import CenterModel, refine_boundaries


def _sample_mdn(pi, mu, ls, temperature=1.0, generator=None):
    """One 8-vector from the mixture. temperature <= 0 -> deterministic mixture mean."""
    if temperature <= 0:
        return (mu * pi.exp().unsqueeze(-1)).sum(1)
    pi = pi / max(temperature, 1e-3)
    comp = torch.multinomial(pi.exp().clamp_min(1e-9), 1, generator=generator).squeeze(-1)
    idx = comp.view(pi.shape[0], 1, 1).expand(-1, 1, mu.shape[-1])
    m = mu.gather(1, idx).squeeze(1)
    s = torch.exp(ls.gather(1, idx).squeeze(1)) * max(temperature, 1e-3) ** 0.5
    eps = torch.randn(m.shape, dtype=m.dtype, device=m.device, generator=generator)
    return m + s * eps


@torch.no_grad()
def decode_line(model, text, tmask, refs, ref_mask, n_chars, cfg, generator=None,
                tf_prefix=None):
    """Return (absolute payloads [N,8], joins per boundary [C-1], boxes [C,4])."""
    device = text.device
    H, zg, z_l, z_k = model.encode(text, tmask, refs, ref_mask)
    # plan_layout now returns 5-dim soft anchors (x,y,w,h,dx); the M4 join path was
    # trained on the legacy 4-dim box parametrisation, so map back explicitly.
    anchors = model.plan_layout(H, zg, tmask)
    boxes = torch.stack([anchors[..., 3], anchors[..., 2],
                         anchors[..., 1], anchors[..., 4]], dim=-1)
    J = max(int(tmask.shape[1]) - 1, 1)
    if cfg.stage == "s1":
        # M4 is untrained at stage s1: use no contract, so the visual check tests M5 alone
        join_p = torch.zeros(H.shape[0], J, device=H.device)
        gap = torch.zeros(H.shape[0], J, 2, device=H.device)
    else:
        prior = torch.full((H.shape[0], J), 0.2, device=H.device)   # TODO: training table
        jd = model.plan_joins(H, boxes, z_k, zg, prior, tmask)
        join_p = torch.sigmoid(jd["logit"])
        gap = (jd["entry"] - jd["exit"]) * (1.0 - (join_p > 0.5).float()).unsqueeze(-1)

    B = text.shape[0]
    # stroke count per character: no threshold, no class imbalance, no 1-stroke collapse
    counts_pred = model.plan_counts(H, zg, tmask).argmax(-1) + 1     # [B,C]
    conts, pens, chars = [], [], []
    prev_end = torch.zeros(B, 2, device=device)
    boundary_stroke = []
    for ci in range(n_chars):
        if len(conts) >= cfg.max_strokes_line:
            break
        n_this = int(counts_pred[:, ci].max().clamp(1, cfg.max_strokes_char))
        for k in range(n_this):
            # full prefix, with a BOS at the front: a 1-token prefix (the earlier bug) leaves a
            # causal decoder with no history and it can only emit the marginal stroke
            if conts:
                pc = torch.stack(conts, 1)
                pp = torch.stack(pens, 1)
                pch = torch.stack(chars, 1)
            else:
                pc = torch.zeros(B, 0, 8, device=device)
                pp = torch.zeros(B, 0, dtype=torch.long, device=device)
                pch = torch.zeros(B, 0, dtype=torch.long, device=device)
            full_c = torch.cat([torch.zeros(B, 1, 8, device=device), pc], dim=1)
            full_p = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), pp], dim=1)
            full_ch = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), pch], dim=1)
            mk = torch.ones(full_c.shape[0], full_c.shape[1], device=device)
            out = model.decoder(full_c, full_p, full_ch, mk, H, tmask, sigma=cfg.gauss_sigma)
            c_new = _sample_mdn(out["pi"][:, -1], out["mu"][:, -1], out["ls"][:, -1],
                                cfg.temperature, generator)
            # anchor the stroke absolutely, then express it as an offset so the
            # downstream recurrence stays exact
            if "abs" in out:
                a = out["abs"][:, -1]
                c_new = c_new.clone()
                c_new[:, 0:2] = a - prev_end
            if tf_prefix is not None and len(conts) < tf_prefix.shape[1]:
                c_new = tf_prefix[:, len(conts)]      # teacher-forced reconstruction
            if k == 0 and ci > 0 and gap.shape[1] >= ci:
                # BOUNDARY CONTRACT: joined boundary -> zero offset, lifted -> planned gap
                c_new = c_new.clone()
                c_new[:, 0:2] = gap[:, ci - 1]
            stop = torch.sigmoid(out["stop"][:, -1])
            prev_end = (prev_end + c_new[:, 0:2]) if "abs" not in out else \
                (out["abs"][:, -1])
            prev_end = prev_end + c_new[:, 6:8]
            conts.append(c_new)
            pens.append((torch.sigmoid(out["pen"][:, -1]) > 0.5).long())
            chars.append(torch.full((B,), ci, dtype=torch.long, device=device))
            if k == 0 and ci > 0:
                boundary_stroke.append(len(conts) - 1)
            if len(conts) >= cfg.max_strokes_line:      # hard cap: stop may never fire
                break

    cont = torch.stack(conts, 1)                       # [B,N,8] relative
    # rebuild absolute Béziers by cumulative summation
    start = cont[..., 0:2].cumsum(1)                   # pen position at each stroke start
    ctrl = cont[..., 2:8].view(*cont.shape[:2], 3, 2) + start.unsqueeze(-2)
    payload = torch.cat([start.unsqueeze(-2), ctrl], dim=-2)     # [B,N,4,2]
    joins = (join_p > 0.5).long()
    return payload, joins, boxes, boundary_stroke


def _render(payload, mask, h=128, w=2048):
    return 1.0 - render_strokes_splat(payload.reshape(payload.shape[0], payload.shape[1], 8),
                                      mask, h, w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="sheet_v3.png")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = V3Config.load(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = CenterModel(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads,
                        n_classes=cfg.n_classes, style_dim=cfg.style_dim,
                        style_depth=cfg.style_depth, n_style_types=cfg.n_style_types,
                        max_strokes=cfg.max_strokes_line, max_chars=cfg.max_chars,
                        n_comp=cfg.mdn_components, n_writers=2048).to(device).eval()
    # strict=False: a checkpoint from stage s1 predates modules M3/M4 use
    missing = model.load_state_dict(ck["model"], strict=False)
    if missing.missing_keys:
        print(f"  loaded with {len(missing.missing_keys)} missing keys (s1 checkpoint)")

    ds = StrokeDatasetV3(cfg.canonical_root, a.split, k_refs=cfg.k_refs,
                         img_h=cfg.img_height, canvas_w=cfg.canvas_width,
                         max_chars=cfg.max_chars, max_strokes_char=cfg.max_strokes_char,
                         max_strokes_line=cfg.max_strokes_line, join_tau=cfg.join_tau,
                         n_classes=cfg.n_classes, pair_prior_path=cfg.pair_prior_path,
                         limit=max(a.n, 2))
    b = collate_v3([ds[i] for i in range(min(a.n, len(ds)))])
    text, tmask = b["text"].to(device), b["text_mask"].to(device)
    refs, rmask = b["refs"].to(device), b["ref_mask"].to(device)

    g = torch.Generator(device=device).manual_seed(a.seed)
    n_chars = int(b["n_chars"].max())
    payload, joins, boxes, boundary_stroke = decode_line(model, text, tmask, refs, rmask,
                                                         n_chars, cfg, g)

    rows, meta = [], []
    for i in range(payload.shape[0]):
        n = int(b["n_strokes"][i])
        mask = torch.zeros(1, payload.shape[1], device=device)
        mask[0, :n] = 1.0
        img = _render(payload[i:i + 1], mask)[0, 0].cpu().numpy()
        gt = b["cont"][i:i + 1].to(device)
        gt_payload = torch.zeros(1, cfg.max_strokes_line, 4, 2, device=device)
        _s = gt[..., 0:2].cumsum(1)
        _c = gt[..., 2:8].view(1, cfg.max_strokes_line, 3, 2) + _s.unsqueeze(-2)
        gt_payload[0] = torch.cat([_s.unsqueeze(-2), _c], dim=-2)
        gmask = torch.zeros(1, cfg.max_strokes_line, device=device)
        gmask[0, :n] = 1.0
        gt_img = _render(gt_payload, gmask)[0, 0].cpu().numpy()
        pred_join = float(joins[i, : int(b["n_chars"][i]) - 1].float().mean()) \
            if int(b["n_chars"][i]) > 1 else float("nan")
        gt_join = float(b["join"][i, : max(int(b["n_chars"][i]) - 1, 1)].mean())
        rows.append(("", [gt_img, img]))
        meta.append({"chars": int(b["n_chars"][i]), "strokes": n,
                     "pred_join_rate": pred_join, "gt_join_rate": gt_join})

    from benchmark.visualise import contact_sheet
    out = contact_sheet(rows, a.out, glyphs_path="")
    print(json.dumps({"sheet": out, "stage": cfg.stage, "samples": meta},
                     indent=2, ensure_ascii=False))
    print("row order per sample: [GT strokes rendered, GENERATED]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
