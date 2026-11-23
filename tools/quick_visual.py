"""Two-row visual: ground truth, and what the model generates. Nothing else.

Everything else the run learns - error, out-of-frame count, per-character counts, the
difference image - is printed to the log, not drawn. The user inspects two rows only.

    python tools/quick_visual.py --config cent/configs/quick.json --steps 4000 --lines 2
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3, collate_v3
from cent.train_v3 import build, step_loss
from benchmark.decode_v3 import decode, reconstruct, draw_payload

W, H = 2048, 128


def write_two_row(path, gt, gen, lab_gt, lab_gen):
    """Own renderer: identical tone for both rows and legible labels."""
    band = 26
    im = Image.new("L", (W, 2 * H + 3 * band), 255)
    d = ImageDraw.Draw(im)
    for i, (img, lab) in enumerate(((gt, lab_gt), (gen, lab_gen))):
        y = band + i * (H + band)
        d.text((8, y - band + 6), lab, fill=0)
        im.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), (0, y))
    im.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lines", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/two_row.png")
    a = ap.parse_args()

    cfg = V3Config.load(a.config)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.stage = "s2"
    torch.manual_seed(cfg.seed)
    ds = StrokeDatasetV3(cfg.canonical_root, "train", k_refs=cfg.k_refs, img_h=cfg.img_height,
                         canvas_w=cfg.canvas_width, max_chars=cfg.max_chars,
                         max_strokes_char=cfg.max_strokes_char,
                         max_strokes_line=cfg.max_strokes_line, join_tau=cfg.join_tau,
                         n_classes=cfg.n_classes, limit=a.lines)
    print(f"[data] {len(ds)} lines | strokes={int(ds[0]['n_strokes'])} "
          f"counts_sum={int(ds[0]['counts'].sum())}", flush=True)
    batch = collate_v3([ds[i] for i in range(len(ds))])
    gt = reconstruct(batch["cont"].to(dev))[:, :int(batch["n_strokes"][0])]
    gt_img, _, gt_ooc = draw_payload(gt, W, H)

    model = build(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                                total_steps=max(a.steps, 1), pct_start=0.1)
    t0 = time.time()
    for step in range(a.steps):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(cfg.amp and dev == "cuda")):
            loss, logs = step_loss(model, batch, cfg, dev, "s2")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step < a.steps - 1:
            sched.step()
        if step % max(a.steps // 4, 1) == 0 or step == a.steps - 1:
            model.eval()
            with torch.no_grad():
                _H, _zg, _, _ = model.encode(batch["text"].to(dev), batch["text_mask"].to(dev),
                                            batch["refs"].to(dev), batch["ref_mask"].to(dev))
                anchors = model.plan_layout(_H, _zg, batch["text_mask"].to(dev))
                res = decode(model, cfg, batch["text"].to(dev), batch["text_mask"].to(dev),
                             batch["refs"].to(dev), batch["ref_mask"].to(dev),
                             int(batch["n_chars"][0]), mode="free", temperature=0.0)
                free = res["payload"][:, :res["n_strokes"][0]]   # drop the batch pad
                g, _, ooc = draw_payload(free, W, H)
                M = min(gt.shape[1], free.shape[1])
                err = float((free[0, :M] - gt[0, :M]).abs().mean())
                diff = float(np.abs(gt_img - g).mean())
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 3),
                              "free_err": round(err, 4), "n_gen": int(res["n_strokes"][0]),
                              # unit evidence: anchors are canvas units, strokes are [0,1]
                              "anchor_absmax": round(float(anchors.abs().max()), 2),
                              "stroke_absmax": round(float(free.abs().max()), 2),
                              "n_gt": int(gt.shape[1]), "ooc": ooc, "diff": round(diff, 4),
                              "counts_pred": res["counts"][0, :int(batch["n_chars"][0])].tolist(),
                              "counts_true": batch["counts"][0, :int(batch["n_chars"][0])].tolist(),
                              **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
            model.train()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    write_two_row(str(out), gt_img, g, "GT", f"MODEL  err={err:.3f}  ooc={ooc}")
    print(f"[done] {a.steps} steps in {time.time() - t0:.1f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
