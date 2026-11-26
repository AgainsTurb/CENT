"""CENT JGT (v1): C1 continuity enforcement at character junctions, with visuals.

For each junction (end-of-character index k):
  * exit tangent  t1  from the last points of character i
  * entry tangent t2  from the first points of character i+1
  * rotate the FIRST window of character i+1 about its entry point so that t2 -> t1,
    with a decaying weight, so the character body is preserved but the hand enters
    along the outgoing direction (that is what "cursive continuity" looks like).

Renders GT / DiffInk / DiffInk+JGT for the same line and reports junction dtheta.
"""
import json, math, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/DiffInk")
from utils.utils import load_config_from_yaml, set_seed, ModelConfig
from dataset import build_test_datasets_and_loaders
from model.vae import VAE
from model.dit import DiT
from model.diffusion import Diffusion
from model.gmm import get_mixture_coef
from trainer.dit_trainer import build_prefix_mask_from_char_points

DEV = "cuda"
CFG = "./configs/dit_val_config.yaml"
NLINE = int(os.environ.get("NLINE", 3))
WIN = int(os.environ.get("WIN", 10))


def load_models(cfg):
    m = ModelConfig(cfg)
    vae = VAE(m).to(DEV)
    vae.load_state_dict(torch.load(cfg["vae_model_path"], map_location=DEV)["model_state_dict"], strict=False)
    vae.eval()
    dit = DiT(m).to(DEV)
    dit.load_state_dict(torch.load(cfg["dit_resume_ckpt"], map_location=DEV)["model_state_dict"], strict=False)
    dit.eval()
    return vae, dit


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def junction_angles(pts, js, k=3):
    out = []
    for i in js:
        if i < k + 1 or i + k >= len(pts):
            continue
        t1 = unit(pts[i] - pts[i - k])
        t2 = unit(pts[i + k] - pts[i])
        if np.allclose(t1, 0) or np.allclose(t2, 0):
            continue
        out.append(math.degrees(math.acos(float(np.clip(np.dot(t1, t2), -1, 1)))))
    return np.array(out) if out else np.array([np.nan])


def c1_fix(pts, js, win=WIN):
    """Rotate the head of each next character toward the exit tangent, decaying."""
    pts = pts.copy()
    n_fix, changed = 0, []
    for k in js:
        if k < 3 or k + 2 >= len(pts):
            continue
        t1 = unit(pts[k] - pts[k - 3])
        w = min(win, len(pts) - k - 2)
        if w < 2 or np.allclose(t1, 0):
            continue
        t2 = unit(pts[k + w] - pts[k + 1])
        if np.allclose(t2, 0):
            continue
        ang = math.acos(float(np.clip(np.dot(t1, t2), -1, 1)))
        if ang < math.radians(25):
            continue
        sign = 1.0 if (t1[0] * t2[1] - t1[1] * t2[0]) < 0 else -1.0
        for i in range(1, w + 1):
            a = (1.0 - i / (w + 1.0)) ** 1.5
            pts[k + i] = pts[k] + rot(pts[k + i] - pts[k], sign * a * ang)
        n_fix += 1
        changed.append(k)
    return pts, n_fix


def main():
    cfg = load_config_from_yaml(CFG)
    set_seed(42)
    loader, cfg = build_test_datasets_and_loaders(cfg)
    vae, dit = load_models(cfg)
    diff = Diffusion(noise_steps=1000, schedule_type="cosine", device=DEV)
    rows, got = [], 0
    with torch.no_grad():
        for p, batch in enumerate(loader):
            data, mask, text_idx, cpi, _ = batch
            data, mask, text_idx = data.to(DEV), mask.to(DEV), text_idx.to(DEV)
            data = data.permute(0, 2, 1)
            feat = vae.encode(data)[0].permute(0, 2, 1)
            lm, lpm, _ = build_prefix_mask_from_char_points(cpi, mask, compression_factor=8, prefix_ratio=0.3)
            fm = (lm * lpm).unsqueeze(1)
            x_pred = diff.ddim_sample(dit, data.size(0), feat, text_idx, lm, lpm,
                                      sampling_timesteps=20, eta=0.0, cfg_scale=1.0)
            x_mix = x_pred.permute(0, 2, 1) * fm + feat.permute(0, 2, 1) * (1 - fm)
            out = vae.decode(x_mix)
            pi, mu1, mu2, s1, s2, corr, pen, _ = get_mixture_coef(out, num_mixture=20)
            for i in range(data.size(0)):
                L = int(mask[i].sum().item())
                pi_i, m1, m2, pe = pi[i].cpu(), mu1[i].cpu(), mu2[i].cpu(), pen[i].cpu()
                if pi_i.shape[0] < pi_i.shape[1]:
                    pi_i, m1, m2 = pi_i.T, m1.T, m2.T
                    if pe.dim() == 2 and pe.shape[0] < pe.shape[1]:
                        pe = pe.T
                m = pi_i[:L].argmax(-1).numpy()
                gen = np.stack([m1[:L].numpy()[np.arange(L), m], m2[:L].numpy()[np.arange(L), m]], 1)
                pn = pe[:L].numpy()
                if pn.ndim == 2:
                    pn = pn[:, -1]
                gt = data[i].permute(1, 0).cpu().numpy()[:L]
                js = np.where(pn > 0.5)[0]
                gt_js = np.where(gt[:, 4] > 0.5)[0]
                gen_fix, nfix = c1_fix(gen, js)
                rows.append({"gt": gt[:, :2], "gen": gen, "fix": gen_fix,
                             "gt_js": gt_js, "js": js,
                             "d_gt": float(np.nanmean(junction_angles(gt[:, :2], gt_js))),
                             "d_gen": float(np.nanmean(junction_angles(gen, js))),
                             "d_fix": float(np.nanmean(junction_angles(gen_fix, js))),
                             "nfix": nfix})
                got += 1
                if got >= NLINE:
                    break
            if got >= NLINE:
                break

    print(json.dumps({"lines": got,
                      "dtheta_gt": round(float(np.nanmean([r["d_gt"] for r in rows])), 1),
                      "dtheta_diffink": round(float(np.nanmean([r["d_gen"] for r in rows])), 1),
                      "dtheta_jgt": round(float(np.nanmean([r["d_fix"] for r in rows])), 1),
                      "junctions_fixed": int(sum(r["nfix"] for r in rows))}, indent=1), flush=True)

    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 2.1 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        ax.plot(r["gt"][:, 0], -r["gt"][:, 1], "-", lw=1.6, color="#111111", label="GT (human)")
        ax.plot(r["gen"][:, 0], -r["gen"][:, 1] - 60, "-", lw=1.4, color="#1f77b4", label="DiffInk")
        ax.plot(r["fix"][:, 0], -r["fix"][:, 1] - 120, "-", lw=1.4, color="#d62728", label="CENT +JGT")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        ax.set_title(f"dtheta at junctions: GT {r['d_gt']:.0f}deg | DiffInk {r['d_gen']:.0f}deg | "
                     f"+JGT {r['d_fix']:.0f}deg  ({r['nfix']} junctions adjusted)",
                     fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig("/workspace/cent_jgt.png", dpi=130, bbox_inches="tight")
    print("[saved] /workspace/cent_jgt.png", flush=True)


main()
