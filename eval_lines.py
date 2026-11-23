"""Evaluate CENT on text lines.

Reports the paper's four connectivity instruments plus image agreement, and refuses to
invent the ones it cannot own:

  connector_rate        fraction of boundaries the model joins; compared with the
                        writer's own rate on the same lines
  pen_lift_agreement    per-boundary agreement between predicted and reference joins
  continuity_error      at joined boundaries, |connector start - previous exit| plus a
                        tangent mismatch; measures how smooth the join is
  speed_profile         NOT_RUN: our canonical lines carry no timestamps, so there is no
                        velocity to measure. Reporting one would be fabrication.
  AR, CR                NOT_RUN without a frozen HTR. These are the paper's content
                        metrics and are the benchmark harness's job, not the model's.
  FID                   NOT_RUN below 512 samples and without Inception.
  image_l1 / ssim       measured against the paired real line.

Usage
  python -m cent.eval_lines --config .fixture/runs/line_smoke/config.json \
      --ckpt .fixture/runs/line_smoke/ckpt_last.pt --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.connectivity import continuity_error, cursive_rate_error
from .config import CENTConfig
from .data_lines import LineTokenDataset, collate_lines
from .models.diffusion import Diffusion
from .models.line import KIND_CONNECTOR, LineDenoiser, LineStyleEncoder
from .models.stroke_graph import BezierRenderer
from .train_lines import style_tokens


def _not_run(reason: str) -> dict:
    return {"status": "NOT_RUN", "reason": reason}


@torch.no_grad()
def sample_line(den, diff, shape, kind, cpos, st, mask, device, guidance=1.0):
    """Reverse diffusion on the token sequence, returning x0 and join logits."""
    x = torch.randn(shape, device=device)
    for step in range(diff.T - 1, -1, -1):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        eps, _ = den(x, t, kind, cpos, st, mask, connector_mask=mask.bool())
        ab = diff.alpha_bar.to(device)[step]
        ab_prev = diff.alpha_bar.to(device)[max(step - 1, 0)]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-5)).clamp(-2, 2)
        if step > 0:
            var = (1 - ab_prev) / (1 - ab).clamp_min(1e-8) * (1 - ab / ab_prev).clamp_min(1e-8)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps + var.sqrt() * torch.randn_like(x)
        else:
            x = x0
        x = x * mask.unsqueeze(-1)
    eps, join_logits = den(x, torch.zeros(shape[0], device=device, dtype=torch.long),
                           kind, cpos, st, mask, connector_mask=mask.bool())
    return x, join_logits


def _gather_connectors(join_logits, kind, mask, n_conn):
    out = []
    for i in range(join_logits.shape[0]):
        n = int(n_conn[i])
        idx = ((kind[i] == KIND_CONNECTOR) & mask[i].bool()).nonzero(as_tuple=True)[0]
        vals = join_logits[i][idx][:n] if len(idx) else join_logits.new_zeros(0)
        out.append(vals)
    return out


@torch.no_grad()
def evaluate(cfg: CENTConfig, ckpt: str, split: str = "val", max_lines: int = 32) -> dict:
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    style = LineStyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types,
                             n_heads=cfg.n_heads).to(device)
    den = LineDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads,
                       n_classes=cfg.n_classes).to(device)
    style.load_state_dict(ck["style"]); den.load_state_dict(ck["denoiser"])
    style.eval(); den.eval()
    renderer = BezierRenderer(size=cfg.img_size, width_size=cfg.canvas_width).to(device)
    diff = Diffusion(T=cfg.T).to(device)

    from common.lexicon import load_lexicon
    ds = LineTokenDataset(cfg.canonical_root, split, lexicon=load_lexicon(cfg.lexicon_path),
                          k_refs=cfg.k_refs, img_h=cfg.img_size, canvas_w=cfg.canvas_width,
                          max_chars=cfg.max_chars, max_tokens=cfg.max_tokens,
                          paired_only=cfg.paired_only, limit=max_lines)
    if len(ds) == 0:
        return {"split": split, "n": 0, "error": "no usable lines in this split"}
    loader = torch.utils.data.DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False,
                                         collate_fn=collate_lines)

    rep: dict = {"split": split, "n": len(ds), "ckpt": str(ckpt)}
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "cont": [], "l1": [], "ssim": [],
           "pred_flags": [], "gt_flags": []}
    for b in loader:
        tok, kind, cpos = b["tokens"].to(device), b["kind"].to(device), b["char_pos"].to(device)
        mask, join, n_conn = b["mask"].to(device), b["join"].to(device), b["n_conn"].to(device)
        refs, rmask = b["refs"].to(device), b["ref_mask"].to(device)
        z_g, z_l, z_k, _ = style(refs, rmask)
        st = style_tokens(z_g, z_l, z_k)
        gen, join_logits = sample_line(den, diff, tok.shape, kind, cpos, st, mask, device)

        # ---- connectivity ----
        pred_list = _gather_connectors(join_logits, kind, mask, n_conn)
        for i, pred in enumerate(pred_list):
            n = int(n_conn[i])
            ref = join[i, :n]
            p = (pred > 0).float()
            tot["tp"] += int(((p == 1) & (ref == 1)).sum())
            tot["fp"] += int(((p == 1) & (ref == 0)).sum())
            tot["fn"] += int(((p == 0) & (ref == 1)).sum())
            tot["tn"] += int(((p == 0) & (ref == 0)).sum())
            tot["pred_flags"].extend([bool(v) for v in p.tolist()])
            tot["gt_flags"].extend([bool(v) for v in ref.tolist()])

        # ---- continuity: distance between a connector's first point and the previous
        # character's last stroke end, at boundaries the model chose to join
        for i in range(gen.shape[0]):
            kinds_i = kind[i].tolist()
            conn_pos = [k for k, kk in enumerate(kinds_i) if kk == KIND_CONNECTOR]
            stroke_pos = [k for k, kk in enumerate(kinds_i) if kk == 0]
            for c in conn_pos[: int(n_conn[i])]:
                prev = [s for s in stroke_pos if s < c]
                if not prev:
                    continue
                ctrl_c = gen[i, c, :8].view(4, 2)
                ctrl_s = gen[i, prev[-1], :8].view(4, 2)
                tot["cont"].append(float(torch.linalg.norm(ctrl_c[0] - ctrl_s[3])))

        # ---- image agreement ----
        img = renderer(gen, mask)
        tgt = b["target_img"].to(device)[:, :, :, : img.shape[-1]]
        tot["l1"].append(float(F.l1_loss(img, tgt)))
        mu_a, mu_b = img.mean(dim=(1, 2, 3)), tgt.mean(dim=(1, 2, 3))
        va = ((img - mu_a.view(-1, 1, 1, 1)) ** 2).mean(dim=(1, 2, 3))
        vb = ((tgt - mu_b.view(-1, 1, 1, 1)) ** 2).mean(dim=(1, 2, 3))
        cov = ((img - mu_a.view(-1, 1, 1, 1)) * (tgt - mu_b.view(-1, 1, 1, 1))).mean(dim=(1, 2, 3))
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
            (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
        tot["ssim"].append(float(ssim.mean()))

    tp, fp, fn, tn = tot["tp"], tot["fp"], tot["fn"], tot["tn"]
    n_bound = tp + fp + fn + tn
    pred_rate = (tp + fp) / n_bound if n_bound else float("nan")
    ref_rate = (tp + fn) / n_bound if n_bound else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    rep["connectivity"] = {
        "status": "OK" if n_bound else "NOT_RUN",
        "boundaries": n_bound,
        "connector_rate_pred": round(pred_rate, 4) if n_bound else None,
        "connector_rate_ref": round(ref_rate, 4) if n_bound else None,
        "pen_lift_agreement": round((tp + tn) / n_bound, 4) if n_bound else None,
        "join_precision": round(prec, 4) if n_bound else None,
        "join_recall": round(rec, 4) if n_bound else None,
    }
    pred_flags = [bool(x) for x in tot.get("pred_flags", [])]
    gt_flags = [bool(x) for x in tot.get("gt_flags", [])]
    if pred_flags and len(pred_flags) == len(gt_flags):
        rep["join_rate_error"] = cursive_rate_error(pred_flags, gt_flags)
    else:
        rep["join_rate_error"] = {"status": "NOT_RUN",
                                  "reason": "predicted/reference boundary lists do not align"}

    cont = tot["cont"]
    rep["continuity_error"] = ({"status": "OK", "value": round(sum(cont) / len(cont), 4)}
                              if cont else _not_run("the model joined no boundary in this split"))
    rep["speed_profile"] = _not_run("canonical lines carry no timestamps; there is no velocity "
                                    "to measure. A speed metric needs the raw pen timing.")
    rep["image_l1"] = {"status": "OK", "value": round(sum(tot["l1"]) / len(tot["l1"]), 4)}
    rep["ssim"] = {"status": "OK", "value": round(sum(tot["ssim"]) / len(tot["ssim"]), 4)}
    rep["AR"] = _not_run("no frozen HTR: character accuracy needs a recogniser, which is the "
                         "benchmark harness's instrument, not the model's")
    rep["CR"] = _not_run("no frozen HTR (see AR)")
    rep["FID"] = _not_run(f"only {len(ds)} lines and no Inception features; FID is not meaningful here")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-lines", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = evaluate(CENTConfig.load(a.config), a.ckpt, a.split, a.max_lines)
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
