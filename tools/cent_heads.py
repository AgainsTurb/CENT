"""CENT heads on a frozen DiffInk backbone (Phase C).

x0_hat (DiffInk latent, frozen)
   ├── M3' anchor head   : per-character (x, y, w, h)          L1 + jerk
   └── M4 junction head  : per-boundary (zeta, gx, gy)         weighted BCE + L1
         └── contract residual  || g - g_true * (1 - zeta) ||

Labels come from the end-of-character channel of the 5-channel trajectory, so they are exact.
Gate: zeta AUC >= 0.75 on held-out lines, junction gap error reduced >= 30 % vs a mean predictor.
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/DiffInk")
from utils.utils import load_config_from_yaml, set_seed, ModelConfig
from dataset import build_test_datasets_and_loaders
from model.vae import VAE
from model.dit import DiT
from model.diffusion import Diffusion
from trainer.dit_trainer import build_prefix_mask_from_char_points

DEV = "cuda"
CFG = "./configs/dit_val_config.yaml"
N_TRAIN_BATCH = int(os.environ.get("N_TRAIN", 12))
N_VAL_BATCH = int(os.environ.get("N_VAL", 6))
LAT_D = None


def load_models(cfg):
    m = ModelConfig(cfg)
    vae = VAE(m).to(DEV)
    vae.load_state_dict(torch.load(cfg["vae_model_path"], map_location=DEV)["model_state_dict"], strict=False)
    vae.eval()
    dit = DiT(m).to(DEV)
    dit.load_state_dict(torch.load(cfg["dit_resume_ckpt"], map_location=DEV)["model_state_dict"], strict=False)
    dit.eval()
    for p in list(vae.parameters()) + list(dit.parameters()):
        p.requires_grad_(False)
    return vae, dit


def collect(vae, dit, diff, loader, nbatch):
    """Return list of samples: latent x0_hat, char start indices (latent), labels."""
    out = []
    with torch.no_grad():
        for p, batch in enumerate(loader):
            if p >= nbatch:
                break
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
            lat = x_mix.permute(0, 2, 1)                      # [B, T_lat, D]
            cpi = char_points_idx.cpu().numpy()
            for b in range(data.size(0)):
                L = int(mask[b].sum().item())
                gt = data[b].permute(1, 0).cpu().numpy()[:L]   # [T,5]
                starts = [int(v) for v in np.atleast_1d(cpi[b]) if 0 <= int(v) < L]
                starts = sorted(set(starts))
                if len(starts) < 3:
                    continue
                boxes, gaps = [], []
                for j, s0 in enumerate(starts):
                    s1 = starts[j + 1] if j + 1 < len(starts) else L
                    seg = gt[s0:s1, :2]
                    if len(seg) < 2:
                        boxes.append(None); continue
                    lo, hi = seg.min(0), seg.max(0)
                    boxes.append(np.concatenate([lo, np.maximum(hi - lo, 1e-6)]))
                    if j + 1 < len(starts):
                        gap = gt[starts[j + 1], :2] - gt[s1 - 1, :2]
                        gaps.append(gap)
                keep = [j for j, bx in enumerate(boxes) if bx is not None]
                if len(keep) < 3:
                    continue
                boxes = np.stack([boxes[j] for j in keep])
                li = np.clip((np.array([starts[j] for j in keep]) / 8.0).astype(int), 0, lat.shape[1] - 1)
                junc = np.stack(gaps) if gaps else np.zeros((0, 2))
                out.append({"lat": lat[b].cpu(), "li": li, "box": boxes, "gap": junc})
    return out


class Heads(nn.Module):
    def __init__(s, d, w=256):
        super().__init__()
        s.anc = nn.Sequential(nn.Linear(d, w), nn.GELU(), nn.Linear(w, 4))
        s.bnd = nn.Sequential(nn.Linear(2 * d, w), nn.GELU(), nn.Linear(w, 3))

    def forward(s, f, fnext):
        return s.anc(f), s.bnd(torch.cat([f, fnext], -1))


def probe(lat, li):
    return lat[li]


def build_batch(samples, idx):
    anc_f, anc_next, anc_t = [], [], []
    bnd_f, bnd_t = [], []
    for i in idx:
        sm = samples[i]
        f = probe(sm["lat"], torch.tensor(sm["li"]))
        anc_f.append(f)
        anc_t.append(torch.tensor(sm["box"], dtype=torch.float32))
        if len(f) > 1:
            bnd_f.append(torch.cat([f[:-1], f[1:]], -1))
            g = torch.tensor(sm["gap"], dtype=torch.float32)
            bnd_t.append(g)
    return (torch.stack(anc_f), torch.stack(anc_t),
            torch.stack(bnd_f), torch.stack(bnd_t))


def main():
    cfg = load_config_from_yaml(CFG)
    set_seed(42)
    loader, cfg = build_test_datasets_and_loaders(cfg)
    vae, dit = load_models(cfg)
    diff = Diffusion(noise_steps=1000, schedule_type="cosine", device=DEV)

    print("[collect] train...", flush=True)
    tr = collect(vae, dit, diff, loader, N_TRAIN_BATCH)
    print(f"[collect] {len(tr)} train lines", flush=True)
    print("[collect] val...", flush=True)
    va = collect(vae, dit, diff, loader, N_TRAIN_BATCH + N_VAL_BATCH)[len(tr):]
    print(f"[collect] {len(va)} val lines", flush=True)

    d = tr[0]["lat"].shape[1]
    allg = torch.cat([torch.tensor(s["gap"]) for s in tr])
    TAU = float(torch.quantile(allg.norm(dim=1), 0.31))    # match the human join rate 31 %
    print(f"[labels] gap tau {TAU:.4f}  join rate {float((allg.norm(dim=1) < TAU).float().mean()):.3f}",
          flush=True)

    heads = Heads(d).to(DEV)
    opt = torch.optim.AdamW(heads.parameters(), lr=2e-3)
    A_f, A_t, B_f, B_t = build_batch(tr, list(range(len(tr))))
    pos_weight = 2.2
    for s in range(600):
        pa, pb = heads(A_f, A_f)
        la = F.l1_loss(pa, A_t)
        if len(A_f) > 1:
            la = la + 0.5 * (pa[:, 1:, 0] - 2 * pa[:, :-1, 0] + pa[:, :-2, 0]).abs().mean()
        zt = torch.sigmoid(pb[..., 0])
        y = (B_t.norm(dim=-1) < TAU).float()
        w = 1.0 + (pos_weight - 1.0) * y
        lb = F.binary_cross_entropy(zt, y, reduction="none") * w
        lb = lb.mean() + 0.5 * F.l1_loss(pb[..., 1:], B_t)
        loss = la + lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 200 == 0:
            print(f"[heads {s}] anchor {la.item():.4f} bnd {lb.item():.4f}", flush=True)

    with torch.no_grad():
        V_f, V_t, Vb_f, Vb_t = build_batch(va, list(range(len(va))))
        pa, pb = heads(V_f, V_f)
        anc_err = (pa - V_t).abs().mean().item()
        zt = torch.sigmoid(pb[..., 0])
        y = (Vb_t.norm(dim=-1) < TAU).float()
        pred = (zt > 0.5).float()
        acc = float((pred == y).float().mean())
        base_acc = float((torch.zeros_like(y) == y).float().mean())
        gap_pred = pb[..., 1:]
        gap_err = (gap_pred - Vb_t).abs().mean().item()
        mean_pred = Vb_t.mean(dim=0, keepdim=True).expand_as(Vb_t)
        gap_err_mean = (mean_pred - Vb_t).abs().mean().item()
        auc = 0.0
        try:
            s_pos = zt[y == 1]; s_neg = zt[y == 0]
            if len(s_pos) and len(s_neg):
                auc = float((s_pos[:, None] > s_neg[None, :]).float().mean())
        except Exception:
            pass
        contract = ((gap_pred - Vb_t * (1 - zt).unsqueeze(-1)).abs().mean()).item()

    res = {"val_lines": len(va), "tau": round(TAU, 4), "anchor_l1": round(anc_err, 4),
           "zeta_acc": round(acc, 3), "zeta_majority_acc": round(base_acc, 3), "zeta_auc": round(auc, 3),
           "gap_err_head": round(gap_err, 4), "gap_err_mean_pred": round(gap_err_mean, 4),
           "gap_reduction": round(1 - gap_err / max(gap_err_mean, 1e-9), 3),
           "contract_residual": round(contract, 4)}
    print("=== CENT HEADS ===")
    print(json.dumps(res, indent=1))
    torch.save({"heads": heads.state_dict(), "tau": TAU, "latent_dim": d}, "/workspace/cent_heads.pt")
    print("[saved] /workspace/cent_heads.pt")


main()
