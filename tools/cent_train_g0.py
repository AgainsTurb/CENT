"""CENT model: reference-conditioned full-line generation + soft anchors + junction head.

Builds ON TOP of DiffInk's released weights without changing their function at init:

  InputEmbedding.proj   old: Linear(384 + 512 -> 896)          cat(x_t, text)
                        new: Linear(384+384+512+4 -> 896)      cat(x_t, text, ref, A)
                        the new columns are zero-initialised, so at step 0 the network
                        computes exactly what it did before (backbone preserved).

  * x_t      : noised TARGET latent
  * ref      : VAE latent of ANOTHER line by the same writer   (style reference)
  * A        : soft anchors, one (cx,cy,w,h) per latent step, interpolated between
               consecutive character boxes  -- a conditioning signal, never a clip box
  * head     : on the predicted latent, per junction: (zeta, gx, gy, kappa)
  * decode   : GMM offsets -> CUMSUM -> trajectory   (offsets, not absolute positions)

Gate G0: overfit ONE batch of 8 lines. If the picture is not characters, wiring is wrong.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/DiffInk")
from utils.utils import load_config_from_yaml, set_seed, ModelConfig
from dataset import build_test_datasets_and_loaders
from model.vae import VAE
from model.dit import DiT
from model.gmm import get_mixture_coef, sample_from_params

DEV = "cuda"
STEPS = int(os.environ.get("STEPS", 2000))
LR = float(os.environ.get("LR", 1e-4))
CFG = "./configs/dit_val_config.yaml"
TDIFF = 1000
LAM_J = 0.1


def ca(t, s0=0.008):
    f = math.cos(((t / TDIFF) + s0) / (1 + s0) * math.pi / 2) ** 2
    return f / math.cos(s0 / (1 + s0) * math.pi / 2) ** 2


def build_anchors(gt, e, starts, L, Llat):
    """Soft anchors [Llat, 4]: (cx, cy, w, h) per latent step, lerped between characters."""
    xy = gt[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1e-6)
    boxes, pos = [], []
    n = min(len(e), len(starts))
    for k in range(n):
        s0, s1 = int(starts[k]), min(int(e[k]) + 1, len(gt))
        seg = xy[s0:s1]
        if len(seg) < 2:
            continue
        b0, b1 = seg.min(0), seg.max(0)
        c = (b0 + b1) / 2
        boxes.append(np.concatenate([(c - lo) / span, np.maximum(b1 - b0, 1e-6) / span]))
        pos.append(((s0 + s1) / 2.0) / max(1, len(gt)) * (Llat - 1))
    if not boxes:
        return None
    boxes = np.asarray(boxes); pos = np.asarray(pos)
    grid = np.arange(Llat)
    A = np.stack([np.interp(grid, pos, boxes[:, d]) for d in range(4)], 1)
    return A.astype(np.float32)


def patch_input_embedding(dit, latent_dim=384, text_dim=512, extra=384 + 4):
    """Grow InputEmbedding.proj, zero-initialising the new columns."""
    old = dit.input_embed.proj
    new = nn.Linear(old.in_features + extra, old.out_features).to(old.weight.device)
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias)
        new.weight[:, :old.in_features].copy_(old.weight)
    dit.input_embed.proj = new
    return dit


def fwd_cent(dit, x, ref, A, text, t):
    """Forward with reference latent and soft anchors appended to the text embedding."""
    te = dit.text_embed(text, x.shape[1])
    inp = torch.cat([x, ref, te, A], dim=-1)
    h = dit.input_embed.proj(inp)
    h = dit.input_embed.conv_pos_embed(h) + h
    tt = dit.time_embed(t if t.ndim else t.repeat(x.shape[0]))
    rope = dit.rotary_embed.forward_from_seq_len(x.shape[1])
    for blk in dit.transformer_blocks:
        h = blk(h, tt, mask=None, rope=rope)
    h = dit.norm_out(h, tt)
    return dit.proj_out(h)


class JHead(nn.Module):
    def __init__(s, d=384, w=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2 * d, w), nn.GELU(), nn.Linear(w, 4))

    def forward(s, z, jidx):
        f = z[:, jidx]                                   # [B, C, d]
        a, b = f[:, :-1], f[:, 1:]
        return s.net(torch.cat([a, b], -1))              # [B, C-1, 4]


def main():
    set_seed(0)
    cfg = load_config_from_yaml(CFG)
    loader, cfg = build_test_datasets_and_loaders(cfg)
    mc = ModelConfig(cfg)
    vae = VAE(mc).to(DEV)
    vae.load_state_dict(torch.load(cfg["vae_model_path"], map_location=DEV)["model_state_dict"], strict=False)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    dit = DiT(mc).to(DEV)
    dit.load_state_dict(torch.load(cfg["dit_resume_ckpt"], map_location=DEV)["model_state_dict"], strict=False)
    dit = patch_input_embedding(dit)

    data, mask, text_idx, cpi, wid = next(iter(loader))   # ONE batch, reused -> overfit
    data, mask, text_idx = data.to(DEV), mask.to(DEV), text_idx.to(DEV)
    B = data.size(0)
    print(f"[G0] overfitting {B} lines, T={data.shape[1]}", flush=True)

    with torch.no_grad():
        lat = vae.encode(data.permute(0, 2, 1))[0].permute(0, 2, 1)   # [B, L, 384]
    Llat = lat.shape[1]
    ref = torch.roll(lat, shifts=1, dims=0)               # ANOTHER line as style reference

    anchors, jidx, jtgt = [], [], []
    for b in range(B):
        Lp = int(mask[b].sum().item())
        gt = data[b].permute(1, 0).cpu().numpy()[:Lp]
        e = np.where(gt[:, 4] > 0.5)[0]
        if len(e) < 3:
            anchors.append(np.zeros((Llat, 4), np.float32)); jidx.append([0, 1]); jtgt.append(np.zeros((1, 4)))
            continue
        starts = np.r_[0, e[:-1] + 1]
        A = build_anchors(gt, e, starts, Lp, Llat)
        anchors.append(A if A is not None else np.zeros((Llat, 4), np.float32))
        ji, jg = [], []
        n = min(len(e), len(starts))
        for k in range(n - 1):
            nxt = int(starts[k + 1])
            jj = int(round(e[k] * (Llat - 1) / max(1, Lp - 1)))
            up = float(gt[nxt, 3])
            gap = gt[nxt, :2] - gt[int(e[k]), :2]
            span = np.maximum(gt[:, :2].max(0) - gt[:, :2].min(0), 1e-6)
            ji.append(jj)
            jg.append([0.0 if up > 0.5 else 1.0, gap[0] / span[0], gap[1] / span[1]])
        if ji:
            jidx.append(ji); jtgt.append(np.asarray(jg, np.float32))
        else:
            jidx.append([0, 1]); jtgt.append(np.zeros((1, 3), np.float32))
    A_t = torch.tensor(np.stack(anchors)).to(DEV)

    head = JHead().to(DEV)
    prm = [p for p in dit.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(prm, lr=LR)
    t0 = time.time()
    for s in range(STEPS):
        i = torch.arange(B, device=DEV)
        z0 = lat[i]
        tt = torch.randint(0, TDIFF, (B,), device=DEV)
        a = torch.tensor([ca(float(v)) for v in tt], device=DEV).view(-1, 1, 1)
        eps = torch.randn_like(z0)
        zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
        pred = fwd_cent(dit, zt, ref, A_t, text_idx, tt.float() / TDIFF)   # predicts z0
        vm = (torch.arange(Llat, device=DEV)[None, :] < mask.sum(1, keepdim=True).clamp_max(Llat)).float()
        rec = (((pred - z0) ** 2).mean(-1) * vm).sum() / vm.sum()
        pj = head(pred, torch.arange(min(jidx[0].__len__() + 1, Llat), device=DEV))
        jl = 0.0
        for b in range(B):
            ji = jidx[b]; jg = torch.tensor(jtgt[b], device=DEV)
            if len(ji) < 2:
                continue
            pb = head(pred[b:b + 1], torch.tensor([0] + ji, device=DEV))[0]
            zt_ = torch.sigmoid(pb[:, 0]); gp = pb[:, 1:3]
            jl = jl + F.binary_cross_entropy(zt_, jg[:len(zt_), 0]) * 0.3 \
                 + ((gp - jg[:len(gp), 1:3]).abs().mean()) + 0.5 * ((gp - jg[:len(gp), 1:3] * (1 - zt_).unsqueeze(-1)).abs().mean())
        loss = rec + LAM_J * jl / B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prm, 1.0)
        opt.step()
        if s % 200 == 0:
            print(f"[g0 {s}] rec {rec.item():.5f} junc {(jl/B).item():.4f}  {time.time()-t0:.0f}s", flush=True)
    print(f"[g0] trained in {time.time()-t0:.0f}s", flush=True)

    # ---- sample: pure noise -> DDIM -> decode (cumsum) ----
    with torch.no_grad():
        z = torch.randn(B, Llat, lat.shape[-1], device=DEV)
        ts = list(np.linspace(TDIFF - 1, 0, 25).astype(int))
        for k, tv in enumerate(ts):
            t = torch.full((B,), tv, device=DEV)
            e = fwd_cent(dit, z, ref, A_t, text_idx, torch.full((B,), tv / TDIFF, device=DEV))
            z0h = e
            eps = (z - math.sqrt(ca(tv)) * z0h) / max(math.sqrt(1 - ca(tv)), 1e-6)
            if k + 1 < len(ts):
                z = math.sqrt(ca(ts[k + 1])) * z0h + math.sqrt(1 - ca(ts[k + 1])) * eps
            else:
                z = z0h
        out = vae.decode(z.permute(0, 2, 1))
        pi, mu1, mu2, s1, s2, corr, pen, _ = get_mixture_coef(out, num_mixture=20)

    fig, axes = plt.subplots(B, 2, figsize=(15, 1.15 * B))
    for b in range(B):
        Lp = int(mask[b].sum().item())
        gt = data[b].permute(1, 0).cpu().numpy()[:Lp]
        pk = [pi[b].cpu(), mu1[b].cpu(), mu2[b].cpu(), s1[b].cpu(), s2[b].cpu(), corr[b].cpu(), pen[b].cpu()]
        if pk[0].shape[0] > pk[0].shape[1]:
            pk = [t.T if t.dim() == 2 else t for t in pk]
        st = sample_from_params(pk, temp=0.1, max_seq_len=Lp, greedy=True)
        gen = np.cumsum(st[:, :2], axis=0)
        for ax, pts, flag, title, col in (
                (axes[b, 0], gt[:, :2], (gt[:, 3] > 0.5) | (gt[:, 4] > 0.5), "GT", "#111"),
                (axes[b, 1], gen, st[:, 4] > 0.5, "CENT (G0 overfit)", "#d62728")):
            lo, hi = pts.min(0), pts.max(0)
            q = (pts - lo) / np.maximum(hi - lo, 1e-6)
            q = q.copy(); q[flag] = np.nan
            ax.plot(q[:, 0], -q[:, 1], lw=1.1, color=col)
            ax.set_aspect("auto"); ax.axis("off"); ax.set_title(title, fontsize=8, loc="left")
    plt.subplots_adjust(hspace=0.08, wspace=0.02, left=0.01, right=0.99, top=0.97, bottom=0.01)
    plt.savefig("/workspace/cent_g0.png", dpi=130, facecolor="white")
    print("[saved] /workspace/cent_g0.png", flush=True)
    torch.save({"dit": dit.state_dict(), "head": head.state_dict()}, "/workspace/cent_g0.pt")


main()
