"""CENT v3 trainer (autoregressive stroke model).

Staged by design, because every v2 run failed late on something a cheap test could have
caught:

    --stage s1   strokes only            gate: overfit a single line (loss -> ~0, legible)
    --stage s2   + layout planner        gate: boxes L1 against the corpus
    --stage s3   + join planner          gate: join-rate error on *sampled* output

`--limit 1 --stage s1` is the S1 falsification test from docs/ARCHITECTURE_V3.md. If the model
cannot memorise one line, the representation is wrong and no amount of GPU time fixes it.

No smoke test has been run on this file; see the status note in docs/ARCHITECTURE_V3.md.
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

from .config_v3 import V3Config
from .data_v3 import StrokeDatasetV3, collate_v3
from .models.v3 import CenterModel


def build(cfg: V3Config, device: str):
    model = CenterModel(
        dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads, n_classes=cfg.n_classes,
        style_dim=cfg.style_dim, style_depth=cfg.style_depth, n_style_types=cfg.n_style_types,
        max_strokes=cfg.max_strokes_line, max_chars=cfg.max_chars,
        n_comp=cfg.mdn_components, n_writers=2048, dropout=cfg.dropout,
    ).to(device)
    return model


def step_loss(model, b, cfg: V3Config, device: str, stage: str):
    cont = b["cont"].to(device)
    pen = b["pen"].to(device)
    char_idx = b["char_idx"].to(device)
    mask = b["mask"].to(device)
    stop = b["stop"].to(device)
    text = b["text"].to(device)
    tmask = b["text_mask"].to(device)

    H, zg, z_l, z_k = model.encode(text, tmask, b["refs"].to(device), b["ref_mask"].to(device))

    # AUTOREGRESSIVE SHIFT. The decoder must not see the token it is asked to predict.
    # Before this line existed, input[t] == target[t] and a causal decoder could solve the task
    # by copying its input: measured teacher-forced error 0.0043 with stroke NLL -40.7, while the
    # true next-stroke error was 0.11-0.17. The loss looked excellent and measured an identity map.
    # STANDARDISE the 8 dims (Graves normalised pen offsets for exactly this reason).
    _sm, _ss = getattr(cfg, "stroke_mean", None), getattr(cfg, "stroke_std", None)
    if _sm is not None:
        _mu = torch.tensor(_sm, device=device, dtype=cont.dtype)
        _sd = torch.tensor(_ss, device=device, dtype=cont.dtype)
        cont_t = (cont - _mu) / _sd
    else:
        cont_t = cont
    B = cont.shape[0]
    bos_c = cont.new_zeros(B, 1, cont_t.shape[-1])
    bos_i = cont.new_zeros(B, 1, dtype=torch.long)
    inp_c = torch.cat([bos_c, cont_t[:, :-1]], dim=1)
    inp_p = torch.cat([bos_i, pen[:, :-1]], dim=1)
    # CARRIER: position k carries the character of stroke k (the one being predicted).
    # With the lagged label the model could never know which character it was starting.
    inp_ci = char_idx
    inp_m = torch.cat([mask.new_ones(B, 1), mask[:, :-1]], dim=1)
    # SCHEDULED SAMPLING. Teacher forcing alone never shows the model its own errors, so at
    # inference a 64-stroke line drifts (measured: next-stroke error 0.0087 given a TRUE prefix,
    # but free-running output degenerates). Replace a fraction of the input tokens with the
    # model's own mixture mean, detached, so it learns to continue from imperfect history.
    # PERFORMANCE: the anchor memory is built ONCE. It used to be rebuilt for the probe and
    # again for the loss forward, and plan_layout is a full transformer pass over the text.
    mem, mmask = model.memory_and_mask(H, tmask, zg, stage)
    p_ss = float(getattr(cfg, "scheduled_sampling", 0.0))
    gstep = int(getattr(model, "_gstep", 0))
    if model.training and p_ss > 0 and gstep % max(int(getattr(cfg, "ss_probe_every", 4)), 1) == 0:
        # PERFORMANCE: this probe is a full extra decoder forward; refresh it every few steps.
        # The mixture means between refreshes are one step stale, which cannot matter for a
        # random drop mask.
        with torch.no_grad():
            probe = model.decoder(inp_c, inp_p, inp_ci, inp_m, mem, mmask,
                                  sigma=cfg.gauss_sigma)
            model._mu_self = (probe["mu"] * probe["pi"].exp().unsqueeze(-1)).sum(2).detach()
    if model.training and p_ss > 0 and getattr(model, "_mu_self", None) is not None:
        drop = (torch.rand(inp_c.shape[:2], device=inp_c.device) < p_ss).unsqueeze(-1)
        inp_c = torch.where(drop, model._mu_self, inp_c)
    model._gstep = gstep + 1
    out = model.decoder(inp_c, inp_p, inp_ci, inp_m, mem, mmask, sigma=cfg.gauss_sigma)
    l_stroke = model.decoder.nll(out, cont_t, mask)
    l_pen = F.binary_cross_entropy_with_logits(out["pen"], pen.float(), weight=mask)
    l_stop = F.binary_cross_entropy_with_logits(out["stop"], stop, weight=mask)
    # absolute start target, by the same recurrence the sampler uses
    # PERFORMANCE + exactness. start_k = start_{k-1} + P3_{k-1} + Delta_k, so the whole
    # recurrence is ONE cumsum. A cumsum of Delta alone was the bug that once produced
    # y = -11.85; the P3 term is what makes this version exact (checked against the loop).
    _d = cont[..., 0:2]
    _p3 = cont[..., 6:8]
    abs_t = torch.cumsum(torch.cat([_d[:, :1], _d[:, 1:] + _p3[:, :-1]], dim=1), dim=1)
    l_abs = (((out["abs"] - abs_t).abs().sum(-1)) * mask).sum() / mask.sum().clamp_min(1.0)

    # CURVE LOSS: sample both cubics and compare points. Differentiable, cheap, and
    # the only term whose value depends on where the control points sit.
    _curve_term = 0.0
    if float(getattr(cfg, "lambda_curve", 0.0)) > 0:
        _pi = out["pi"].exp().unsqueeze(-1)
        _m = (out["mu"] * _pi).sum(2)          # expected token [B,L,8]
        _p0 = out["abs"]
        _p1 = _p0 + _m[..., 2:4]; _p2 = _p0 + _m[..., 4:6]; _p3 = _p0 + _m[..., 6:8]
        _g0 = abs_t
        _g1 = _g0 + cont[..., 2:4]; _g2 = _g0 + cont[..., 4:6]; _g3 = _g0 + cont[..., 6:8]
        _tt = torch.linspace(0, 1, 8, device=device).view(1, 1, 8, 1)
        def _bez(a, b, c, d):
            return ((1 - _tt) ** 3 * a.unsqueeze(-2) + 3 * (1 - _tt) ** 2 * _tt * b.unsqueeze(-2)
                    + 3 * (1 - _tt) * _tt ** 2 * c.unsqueeze(-2) + _tt ** 3 * d.unsqueeze(-2))
        _l = (_bez(_p0, _p1, _p2, _p3) - _bez(_g0, _g1, _g2, _g3)).abs().mean(-1).mean(-1)
        l_curve = (_l * mask).sum() / mask.sum().clamp_min(1.0)
        _curve_term = float(getattr(cfg, "lambda_curve", 0.0)) * l_curve
        logs["curve"] = float(l_curve.detach())
    loss = (cfg.lambda_stroke * l_stroke + cfg.lambda_pen * l_pen
            + cfg.lambda_stop * l_stop + float(getattr(cfg, "lambda_abs", 1.0)) * l_abs
            + _curve_term)
    logs = {"stroke": float(l_stroke.detach()), "pen": float(l_pen.detach()),
            "stop": float(l_stop.detach())}
    logs["abs"] = float(l_abs.detach())

    # M7 writer identification — auxiliary, keeps z_g informative
    if cfg.lambda_writer > 0:
        wl = F.cross_entropy(model.writer_head(zg), b["writer"].to(device))
        loss = loss + cfg.lambda_writer * wl
        logs["writer"] = float(wl.detach())

    if stage in ("s2", "s3"):
        box_t = b["boxes"].to(device)
        bmask = b["box_mask"].to(device)
        # ---- M3' soft anchors ---------------------------------------------------
        pred = getattr(model, "_last_anchors", None)
        if pred is None:
            pred = model.plan_layout(H, zg, tmask)             # [B,C,5]
        bx = torch.cumsum(box_t[..., 3], dim=1)                # x from dx advances
        anchor_t = torch.stack([bx, box_t[..., 2], box_t[..., 1],
                                box_t[..., 0], box_t[..., 3]], dim=-1)
        l_cont = pred.new_zeros(())
        l_box = (((pred - anchor_t).abs().sum(-1)) * bmask).sum() / bmask.sum().clamp_min(1.0)
        loss = loss + cfg.lambda_layout * l_box
        logs["layout"] = float(l_box.detach())
        # continuity: handwriting advances smoothly, so penalise jerk in the anchors
        # rather than only error against the corpus boxes
        if pred.shape[1] > 2:
            adv = pred[..., 0].diff(dim=1)
            jerk = (adv[:, 1:] - adv[:, :-1]).abs()
            mc = bmask[:, 1:] if bmask.shape[1] == jerk.shape[1] else bmask[:, 2:]
            l_cont = (jerk * mc).sum() / mc.sum().clamp_min(1.0)
            ys = pred[..., 1]
            smooth = (ys[:, 2:] - 2 * ys[:, 1:-1] + ys[:, :-2]).abs()
            ms = mc[:, :smooth.shape[1]]
            l_cont = l_cont + (smooth * ms).sum() / ms.sum().clamp_min(1.0)
        loss = loss + float(getattr(cfg, "lambda_cont", 0.5)) * l_cont
        logs["cont"] = float(l_cont.detach())
        cl = model.plan_counts(H, zg, tmask)                       # [B,C,13]
        ctgt = (b["counts"].to(device) - 1).clamp(0, 12)
        l_cnt = F.cross_entropy(cl.reshape(-1, 13), ctgt.reshape(-1), reduction="none")
        l_cnt = (l_cnt * bmask.reshape(-1)).sum() / bmask.sum().clamp_min(1.0)
        loss = loss + float(getattr(cfg, "lambda_count", 1.0)) * l_cnt
        logs["count"] = float(l_cnt.detach())

    if stage == "s3":
        jd = model.plan_joins(H, b["boxes"].to(device), z_k, zg, b["prior"].to(device), tmask)
        jm = (tmask[:, :-1] * tmask[:, 1:]) if tmask.shape[1] > 1 else tmask[:, :0]
        if float(jm.sum()) > 0:
            l_join = F.binary_cross_entropy_with_logits(jd["logit"], b["join"].to(device)[:, :jd["logit"].shape[1]],
                                                       weight=jm)
            gap = b["gap"].to(device)[:, :jd["entry"].shape[1]]
            l_gap = (((jd["entry"] - jd["exit"] - gap).abs().sum(-1)) * jm).sum() / jm.sum().clamp_min(1.0)
            loss = loss + cfg.lambda_join * l_join + cfg.lambda_bridge * l_gap
            logs["join"] = float(l_join.detach())
            logs["bridge"] = float(l_gap.detach())
    return loss, logs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="override cfg.stage: s1|s2|s3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="one batch, then exit")
    a = ap.parse_args()

    cfg = V3Config.load(a.config)
    if a.stage:
        cfg.stage = a.stage
    if a.limit is not None:
        cfg.limit = a.limit
    if a.epochs is not None:
        cfg.epochs = a.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    ds = StrokeDatasetV3(cfg.canonical_root, "train", k_refs=cfg.k_refs,
                         img_h=cfg.img_height, canvas_w=cfg.canvas_width,
                         max_chars=cfg.max_chars, max_strokes_char=cfg.max_strokes_char,
                         max_strokes_line=cfg.max_strokes_line, join_tau=cfg.join_tau,
                         paired_only=cfg.paired_only, limit=cfg.limit,
                         n_classes=cfg.n_classes, pair_prior_path=cfg.pair_prior_path,
                         prior_smoothing=cfg.prior_smoothing)
    print(f"[data] {len(ds)} lines  dropped={ds.dropped}", flush=True)
    if getattr(cfg, "stroke_mean", None) is None:
        import numpy as _np
        _v = []
        for _i in range(min(len(ds), 200)):
            _it = ds[_i]
            _v.append(_it["cont"][:int(_it["n_strokes"])].reshape(-1, 8).numpy())
        _V = _np.concatenate(_v, 0)
        cfg.stroke_mean = _V.mean(0).tolist()
        cfg.stroke_std = (_V.std(0) + 1e-6).tolist()
        print("[std] mean", [round(x, 5) for x in cfg.stroke_mean], flush=True)
        print("[std] std ", [round(x, 5) for x in cfg.stroke_std], flush=True)
    # prove which labelling the trainer actually sees (connectors included = 78, not 64)
    print(f"[data] first item: strokes={int(ds[0]['n_strokes'])} "
          f"chars={int(ds[0]['n_chars'])} counts_sum={int(ds[0]['counts'].sum())}", flush=True)
    if len(ds) == 0:
        raise SystemExit("no usable lines: check char_strokes coverage in the canonical corpus")

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                    num_workers=cfg.num_workers, collate_fn=collate_v3,
                    pin_memory=(device == "cuda"))

    model = build(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(ds) // cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(total, 1),
                                                pct_start=0.05)
    run = Path(cfg.out_dir) / cfg.run_name
    run.mkdir(parents=True, exist_ok=True)
    cfg.save(run / "config.json")
    log = (run / "train_log.jsonl").open("w", encoding="utf-8")

    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for b in dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg.amp and device == "cuda")):
                loss, logs = step_loss(model, b, cfg, device, cfg.stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if step < total - 1:
                sched.step()
            rec = {"step": step, "epoch": epoch, "loss": float(loss.detach()), "stage": cfg.stage,
                   "sec": time.time() - t0, **logs}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 20 == 0:
                print(f"[{epoch:02d} {step:05d}] loss={float(loss.detach()):.4f} " +
                      " ".join(f"{k}={v:.4f}" for k, v in logs.items()), flush=True)
            step += 1
            if a.dry_run:
                print("[dry-run] one batch complete", flush=True)
                return 0
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch},
                   run / "ckpt_last.pt")
        print(f"[epoch {epoch}] -> {run/'ckpt_last.pt'}", flush=True)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
