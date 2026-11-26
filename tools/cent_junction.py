"""CENT on top of DiffInk: junction baseline + M8 continuity refinement.

Runs on the GPU box, inside /workspace/DiffInk, with the released checkpoints.

Phase A: for a handful of validation lines, decode DiffInk's generated trajectory and
         measure what happens AT CHARACTER JUNCTIONS:
             join rate, |gap| between the end of char i and the start of char i+1,
             tangent discontinuity dtheta, and the number of junctions.
Phase B: apply our M8 C1 refinement at every joined junction and re-measure dtheta.

Character boundaries are exact: DiffInk's 5th channel is an end-of-character marker.
"""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/DiffInk")
from utils.utils import load_config_from_yaml, set_seed
from dataset import build_test_datasets_and_loaders
from model.vae import VAE
from model.dit import DiT
from model.diffusion import Diffusion
from model.gmm import get_mixture_coef
from utils.utils import ModelConfig
from trainer.dit_trainer import build_prefix_mask_from_char_points

DEV = "cuda"
CFG = "./configs/dit_val_config.yaml"
NLINE = int(os.environ.get("NLINE", 8))


def load_models(cfg_dict):
    mcfg = ModelConfig(cfg_dict)
    vae = VAE(mcfg).to(DEV)
    ck = torch.load(cfg_dict["vae_model_path"], map_location=DEV)
    vae.load_state_dict(ck["model_state_dict"], strict=False)
    vae.eval()
    dit = DiT(mcfg).to(DEV)
    ck = torch.load(cfg_dict["dit_resume_ckpt"], map_location=DEV)
    dit.load_state_dict(ck["model_state_dict"], strict=False)
    dit.eval()
    return vae, dit


def junctions(pts, pen_ch):
    """Indices where the end-of-character flag fires (pen channel index 2)."""
    return np.where(pen_ch > 0.5)[0]


def tangent(pts, i, before=True, k=3):
    if before:
        a, b = max(i - k, 0), i
    else:
        a, b = i, min(i + k, len(pts) - 1)
    if b - a < 1:
        return None
    v = pts[b] - pts[a]
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def angle(u, v):
    if u is None or v is None:
        return float("nan")
    c = float(np.clip(np.dot(u, v), -1, 1))
    return math.degrees(math.acos(c))


def refine(pts, pen_ch, tau=0.02):
    """M8: at each joined junction, blend the two touching points to share one tangent."""
    pts = pts.copy()
    n_fix = 0
    for i in junctions(pts, pen_ch):
        if i < 2 or i >= len(pts) - 2:
            continue
        gap = np.linalg.norm(pts[i] - pts[i - 1])
        if gap > tau:
            continue
        mid = 0.5 * (pts[i - 1] + pts[i])
        pts[i - 1] = 0.5 * pts[i - 1] + 0.5 * mid
        pts[i] = 0.5 * pts[i] + 0.5 * mid
        n_fix += 1
    return pts, n_fix


def stats(traj, pen, tag, rows):
    js = junctions(traj, pen)
    gaps, angs = [], []
    for i in js:
        if i < 2 or i >= len(traj) - 2:
            continue
        gaps.append(float(np.linalg.norm(traj[i] - traj[i - 1])))
        angs.append(angle(tangent(traj, i, True), tangent(traj, i, False)))
    if not gaps:
        return
    rows.append({"tag": tag, "n": len(gaps),
                 "join_rate": round(float(np.mean([g < 0.02 for g in gaps])), 3),
                 "gap_mean": round(float(np.mean(gaps)), 4),
                 "dtheta_mean": round(float(np.nanmean(angs)), 1),
                 "dtheta_p90": round(float(np.nanpercentile(angs, 90)), 1)})


def main():
    cfg = load_config_from_yaml(CFG)
    set_seed(42)
    loader, cfg = build_test_datasets_and_loaders(cfg)
    vae, dit = load_models(cfg)
    diff = Diffusion(noise_steps=1000, schedule_type="cosine", device=DEV)
    rows, K = [], 0
    with torch.no_grad():
        for p, batch in enumerate(loader):
            data, mask, text_idx, char_points_idx, _ = batch
            data, mask, text_idx = data.to(DEV), mask.to(DEV), text_idx.to(DEV)
            data = data.permute(0, 2, 1)
            feat = vae.encode(data)[0].permute(0, 2, 1)
            lm, lpm, _ = build_prefix_mask_from_char_points(char_points_idx, mask,
                                                            compression_factor=8, prefix_ratio=0.3)
            fm = (lm * lpm).unsqueeze(1)
            x_pred = diff.ddim_sample(dit, data.size(0), feat, text_idx, lm, lpm,
                                      sampling_timesteps=20, eta=0.0, cfg_scale=1.0)
            x_mix = x_pred.permute(0, 2, 1) * fm + feat.permute(0, 2, 1) * (1 - fm)
            out = vae.decode(x_mix)
            pi, mu1, mu2, s1, s2, corr, pen, _ = get_mixture_coef(out, num_mixture=20)
            for i in range(data.size(0)):
                L = int(mask[i].sum().item())
                m = pi[i].argmax(-1)[:L].cpu().numpy()
                xs = mu1[i].cpu().numpy()[np.arange(L), m]
                ys = mu2[i].cpu().numpy()[np.arange(L), m]
                pn = pen[i].cpu().numpy()[:L]
                gen = np.stack([xs, ys], 1)
                gt = data[i].permute(1, 0).cpu().numpy()[:L]
                gt_pen = gt[:, 4] if gt.shape[1] >= 5 else np.zeros(L)
                stats(gt[:, :2], gt_pen, "GT", rows)
                stats(gen, pn, "DiffInk", rows)
                ref, nf = refine(gen, pn)
                stats(ref, pn, "DiffInk+M8", rows)
                rows[-1]["fixed"] = nf
            K += 1
            if K >= max(1, NLINE // max(1, data.size(0))):
                break
    for r in rows:
        print(json.dumps(r), flush=True)
    print(json.dumps({"lines": K, "rows": len(rows)}))


main()
