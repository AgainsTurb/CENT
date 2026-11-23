import argparse, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3, collate_v3
from cent.train_v3 import build, step_loss
from benchmark.decode_v3 import decode, reconstruct, draw_payload

W, H = 2048, 128
VARIANTS = [("exact counts", dict(count_mode="exact", clamp=None)),
            ("batchmax counts", dict(count_mode="batchmax", clamp=None)),
            ("exact + page clamp", dict(count_mode="exact", clamp="page"))]

def sheet(path, rows):
    band = 24
    im = Image.new("L", (W, len(rows) * (H + band) + band), 255); d = ImageDraw.Draw(im)
    for i, (lab, img) in enumerate(rows):
        y = band + i * (H + band)
        d.text((8, y - band + 5), lab, fill=0)
        im.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), (0, y))
    im.save(path)

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lines", type=int, default=2)
a = ap.parse_args()
cfg = V3Config.load(a.config); dev = "cuda"; cfg.stage = "s2"; torch.manual_seed(cfg.seed)
ds = StrokeDatasetV3(cfg.canonical_root, "train", k_refs=cfg.k_refs, img_h=cfg.img_height,
                     canvas_w=cfg.canvas_width, max_chars=cfg.max_chars, max_strokes_char=cfg.max_strokes_char,
                     max_strokes_line=cfg.max_strokes_line, join_tau=cfg.join_tau,
                     n_classes=cfg.n_classes, limit=a.lines)
b = collate_v3([ds[i] for i in range(len(ds))]); t0 = time.time()
gt = reconstruct(b["cont"].to(dev))[:, :int(b["n_strokes"][0])]
gt_img, _, _ = draw_payload(gt, W, H)
m = build(cfg, dev); opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=a.steps, pct_start=0.1)
for step in range(a.steps):
    m.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, logs = step_loss(m, b, cfg, dev, "s2")
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip); opt.step()
    if step < a.steps - 1: sch.step()
print(json.dumps({"trained": a.steps, "sec": round(time.time() - t0, 1),
                  "loss": round(float(loss), 2), **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
m.eval(); rows = [("GT", gt_img)]; best = None
with torch.no_grad():
    for lab, kw in VARIANTS:
        r = decode(m, cfg, b["text"].to(dev), b["text_mask"].to(dev), b["refs"].to(dev),
                   b["ref_mask"].to(dev), int(b["n_chars"][0]), mode="free", temperature=0.0, **kw)
        free = r["payload"][:, :r["n_strokes"][0]]
        img, _, ooc = draw_payload(free, W, H)
        M = min(gt.shape[1], free.shape[1])
        err = float((free[0, :M] - gt[0, :M]).abs().mean())
        print(json.dumps({"variant": lab, "err": round(err, 4), "ooc": ooc,
                          "n": int(free.shape[1]), "n_gt": int(gt.shape[1])}), flush=True)
        rows.append((f"{lab}  err={err:.3f} ooc={ooc}", img))
        if best is None or (ooc, err) < best[0]: best = ((ooc, err), img, lab, err, ooc)
sheet("runs/ab.png", rows)
sheet("runs/two_row.png", [("GT", gt_img), (f"MODEL {best[2]}  err={best[3]:.3f} ooc={best[4]}", best[1])])
print("[done]", flush=True)
