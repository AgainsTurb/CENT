"""Objectives.

Naming follows the paper: ``L_vae`` (reconstruction), ``L_kl``, ``L_ctc``
(latent must stay character-decodable), ``L_style`` (writer classification),
``L_diff`` (masked latent MSE) and ``L_bnd`` (the boundary objective — lift vs
join, the term that makes contribution (ii) explicit).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = mask.to(x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    return (x * m).sum() / m.sum().clamp_min(eps)


def align_boundary_labels(bnd_labels: torch.Tensor, bnd_mask: torch.Tensor,
                          bnd_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Place per-sample boundary labels into the model's valid boundary slots.

    The dataset stores boundaries in character order; the DiT exposes one slot
    per *frame* boundary. Both are ordered, so a sequential fill is correct.
    """
    B, T1 = bnd_valid.shape
    out = torch.zeros(B, T1, device=bnd_labels.device)
    msk = torch.zeros(B, T1, dtype=torch.bool, device=bnd_labels.device)
    for b in range(B):
        labs = bnd_labels[b][bnd_mask[b]].tolist()
        k = 0
        for f in range(T1):
            if not bool(bnd_valid[b, f]):
                continue
            if k < len(labs):
                out[b, f] = labs[k]
                msk[b, f] = True
            k += 1
    return out, msk


def vae_losses(out: dict, dxy: torch.Tensor, pen: torch.Tensor,
               char_seq: torch.Tensor, char_mask: torch.Tensor,
               writer_idx: torch.Tensor, mask: torch.Tensor, *,
               w_gmm: float = 1.0, w_pen: float = 2.0, w_kl: float = 1e-3,
               w_ctc: float = 1.0, w_style: float = 0.5,
               w_aux: float = 2.0) -> dict:
    """VAE objective.

    All terms are computed in **fp32 even under bf16 autocast**: the GMM
    log-likelihood and CTC are numerically delicate, so the speed comes from the
    convolutions and only the loss arithmetic is upcast.
    """
    from .models.inkvae import InkVAE

    # (dx, y) parameterisation: y is absolute and free, so per-step vertical
    # noise cannot integrate into a wandering baseline. x stays a delta.
    x_abs = torch.cumsum(dxy[..., 0], dim=1)
    y_abs = torch.cumsum(dxy[..., 1], dim=1)
    geo = torch.stack([x_abs, y_abs], dim=-1)
    nll = InkVAE.gmm_nll(out["gmm"].float(), geo.float())     # (B,T)
    l_gmm = masked_mean(nll, mask)

    pen_loss = F.cross_entropy(
        out["pen_logits"].float().reshape(-1, 4), pen.reshape(-1), reduction="none"
    ).view_as(pen)
    l_pen = masked_mean(pen_loss, mask)

    l_kl = 0.5 * (out["mu"].float().pow(2) + out["logvar"].float().exp()
                  - 1 - out["logvar"].float())
    l_kl = masked_mean(l_kl, out["latent_mask"])

    # CTC: latent frames must decode to the character sequence (fp32)
    logp = out["ctc_logits"].float().log_softmax(-1).transpose(0, 1)  # (T_lat, B, C)
    in_len = out["latent_mask"].sum(1).long().clamp(min=1)
    tgt = char_seq[:, : max(1, char_seq.shape[1])]
    tgt_len = char_mask.sum(1).long().clamp(min=1)
    Lmax = int(tgt_len.max())
    if int(in_len.min()) < Lmax:
        tgt = tgt[:, : int(in_len.min())]
        tgt_len = tgt_len.clamp(max=int(in_len.min()))
    l_ctc = F.ctc_loss(logp, tgt, in_len, tgt_len, blank=0, zero_infinity=True)

    l_style = F.cross_entropy(out["writer_logits"].float(), writer_idx)

    # Explicit geometry term. The GMM likelihood alone rewards a *probable*
    # mean displacement, not a correct one, so the decoder can satisfy it with a
    # smooth mean trajectory (the "straight line" collapse). This L1 term on the
    # mixture expectation forces the mean to track the real pen movement.
    pi = F.softmax(out["gmm"][..., 0], dim=-1)
    pred_delta = (pi.unsqueeze(-1) * out["gmm"][..., 1:3]).sum(2)
    l_aux = masked_mean((pred_delta.float() - geo.float()).abs().sum(-1), mask)

    total = (w_gmm * l_gmm + w_pen * l_pen + w_kl * l_kl
             + w_ctc * l_ctc + w_style * l_style + w_aux * l_aux)
    return {
        "total": total, "gmm": l_gmm, "pen": l_pen, "kl": l_kl,
        "ctc": l_ctc, "style": l_style, "aux": l_aux,
    }


def boundary_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                  *, pos_weight: float = 4.0) -> torch.Tensor:
    """BCE over boundary slots; ``pos_weight`` compensates join being rare."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask], pos_weight=pw
    )


def dit_loss(out: dict, z0: torch.Tensor, bnd_labels: torch.Tensor,
             bnd_mask: torch.Tensor, frame_mask: torch.Tensor, *,
             w_diff: float = 1.0, w_bnd: float = 1.0, pos_weight: float = 4.0) -> dict:
    l_diff = masked_mean((out["x0"] - z0).pow(2), frame_mask)
    labels, msk = align_boundary_labels(bnd_labels, bnd_mask, out["bnd_valid"])
    l_bnd = boundary_loss(out["bnd_logits"], labels, msk, pos_weight=pos_weight)
    return {"total": w_diff * l_diff + w_bnd * l_bnd, "diff": l_diff, "bnd": l_bnd}
