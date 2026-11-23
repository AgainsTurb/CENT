"""Sample a trained CENT checkpoint and write a contact sheet for visual inspection.

Each sample row shows the ground-truth line, the generated image, the generated trajectory and
the reference lines the style was taken from — so a reader can judge joins, spacing and style
without opening tensors.

Usage
    python -m benchmark.visualise_run --config runs/casia_r1/config.json \
        --ckpt runs/casia_r1/ckpt_last.pt --split val --n 6 --out runs/casia_r1/sheet_val.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark.visualise import contact_sheet, trajectory_image
from cent.config import CENTConfig
from cent.data_lines import LineTokenDataset, collate_lines
from cent.eval_lines import sample_line
from cent.models.diffusion import Diffusion
from cent.models.line import KIND_CHAR_STROKE, KIND_CONNECTOR, LineDenoiser, LineStyleEncoder
from cent.models.stroke_graph import BezierRenderer, bezier_sample
from cent.train_lines import style_tokens
from common.lexicon import load_lexicon


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="sheet.png")
    ap.add_argument("--guidance", type=float, default=1.0)
    a = ap.parse_args()

    cfg = CENTConfig.load(a.config)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)

    style = LineStyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types,
                             n_heads=cfg.n_heads).to(device)
    den = LineDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads,
                       n_classes=cfg.n_classes, max_tokens=cfg.max_tokens).to(device)
    style.load_state_dict(ck["style"])
    den.load_state_dict(ck["denoiser"])
    style.eval(); den.eval()
    renderer = BezierRenderer(size=cfg.img_size, width_size=cfg.canvas_width).to(device)
    diff = Diffusion(T=cfg.T).to(device)

    ds = LineTokenDataset(cfg.canonical_root, a.split, lexicon=load_lexicon(cfg.lexicon_path),
                          k_refs=cfg.k_refs, img_h=cfg.img_size, canvas_w=cfg.canvas_width,
                          max_chars=cfg.max_chars, max_tokens=cfg.max_tokens,
                          paired_only=cfg.paired_only, limit=max(a.n, 2))
    if len(ds) == 0:
        raise RuntimeError(f"no usable lines in split {a.split!r}")
    loader = torch.utils.data.DataLoader(ds, batch_size=min(a.n, len(ds)), shuffle=False,
                                         collate_fn=collate_lines)
    batch = next(iter(loader))
    tok, kind = batch["tokens"].to(device), batch["kind"].to(device)
    cpos, mask = batch["char_pos"].to(device), batch["mask"].to(device)
    n_conn = batch["n_conn"].to(device)
    refs, rmask = batch["refs"].to(device), batch["ref_mask"].to(device)

    z_g, z_l, z_k, _ = style(refs, rmask)
    st = style_tokens(z_g, z_l, z_k)
    gen, join_logits = sample_line(den, diff, tok.shape, kind, cpos, st, mask, device,
                                   guidance=a.guidance)
    img = renderer(gen, mask)

    rows = []
    spoken = []
    for i in range(min(a.n, gen.shape[0])):
        text = batch["text"][i]
        gt = batch["target_img"][i, 0].cpu().numpy()
        pred = img[i, 0].cpu().numpy()
        # polylines from runs of contiguous character strokes (same geometry the renderer sees)
        runs, cur = [], []
        for k in range(int(mask[i].sum())):
            if kind[i, k].item() == KIND_CHAR_STROKE:
                cur.append(gen[i, k].reshape(4, 2))
            elif cur:
                runs.append(torch.stack(cur)); cur = []
        if cur:
            runs.append(torch.stack(cur))
        polys = []
        for r in runs:
            pts = bezier_sample(r.unsqueeze(0), 16)[0]
            polys.extend([p.cpu().numpy() for p in pts])
        traj = trajectory_image(polys, cfg.img_size, cfg.canvas_width)

        ref_imgs = [refs[i, j, 0].cpu().numpy() for j in range(cfg.k_refs)
                    if rmask[i, j] > 0.5]
        n_join = int((torch.sigmoid(join_logits[i][
            (kind[i] == KIND_CONNECTOR) & mask[i].bool()]) > 0.5).sum()) if join_logits is not None else 0
        rows.append((f"{text}  joins {n_join}", [gt, pred, traj]))
        rows.append(("references", ref_imgs))
        spoken.append({"text": text, "joins_pred": n_join, "n_boundaries": int(n_conn[i])})

    out = contact_sheet(rows, a.out)
    print(f"sheet: {out}")
    print(json.dumps({"split": a.split, "n": len(spoken), "samples": spoken},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
