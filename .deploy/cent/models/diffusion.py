"""Content verifier and the DDPM wrapper.

The verifier is a training-time legibility regulariser: it is trained on REAL
stroke graphs to map a stroke graph to a character, and its cross-entropy on
GENERATED graphs is added to the loss. It is never used at inference, and it is
not a hard constraint -- the hard constraint is the lexicon topology. Ablation
A6 is what decides whether it earns its keep.

The diffusion module is a minimal DDPM (Ho et al., 2020) with classifier-free
guidance at sampling time (Ho & Salimans, 2022).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stroke_graph import COORDS, N_MAX


class StrokeVerifier(nn.Module):
    """Stroke graph -> character logits."""

    def __init__(self, n_classes: int = 3755, dim: int = 128, depth: int = 3):
        super().__init__()
        self.inp = nn.Linear(COORDS, dim)
        self.pos = nn.Parameter(torch.randn(N_MAX, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(tokens) + self.pos[: tokens.shape[1]].unsqueeze(0)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        for blk in self.blocks:
            h = h + blk(h)
        if mask is not None:
            w = mask.unsqueeze(-1)
            pooled = (h * w).sum(1) / w.sum(1).clamp_min(1e-6)
        else:
            pooled = h.mean(1)
        return self.head(pooled)


def cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * torch.pi / 2) ** 2
    ab = f / f[0]
    return ab.clamp(1e-5, 1.0).float()          # alpha_bar[0..T], decreasing


class Diffusion(nn.Module):
    """DDPM over stroke-graph tokens."""

    def __init__(self, T: int = 1000):
        super().__init__()
        self.T = T
        self.register_buffer("alpha_bar", cosine_schedule(T))

    # ---- training ----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def p_loss(self, model: nn.Module, x0: torch.Tensor, mask: torch.Tensor,
               char_idx: torch.Tensor | None, style_tokens: torch.Tensor,
               drop_char_p: float = 0.1) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        # classifier-free guidance: drop the character condition on a fraction of tokens
        drop = torch.rand(B, device=x0.device) < drop_char_p
        pred = model(xt, t, char_idx, style_tokens, mask, drop_char=False)
        if drop.any():
            pred_dropped = model(xt, t, char_idx, style_tokens, mask, drop_char=True)
            pred = torch.where(drop.view(-1, 1, 1), pred_dropped, pred)
        return F.mse_loss(pred, noise, reduction="none").mul(mask.unsqueeze(-1)).mean()

    # ---- sampling ----
    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, int, int], mask: torch.Tensor,
               char_idx: torch.Tensor | None, style_tokens: torch.Tensor,
               guidance: float = 2.0, device: str | torch.device = "cpu") -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for step in range(self.T - 1, -1, -1):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            if guidance != 1.0 and char_idx is not None:
                e_cond = model(x, t, char_idx, style_tokens, mask)
                e_uncond = model(x, t, char_idx, style_tokens, mask, drop_char=True)
                eps = e_uncond + guidance * (e_cond - e_uncond)
            else:
                eps = model(x, t, char_idx, style_tokens, mask)
            ab = self.alpha_bar.to(device)[step]
            ab_prev = self.alpha_bar.to(device)[max(step - 1, 0)]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-5)
            x0 = x0.clamp(-2, 2)
            if step > 0:
                var = (1 - ab_prev) / (1 - ab).clamp_min(1e-8) * (1 - ab / ab_prev).clamp_min(1e-8)
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps + var.sqrt() * torch.randn_like(x)
            else:
                x = x0
            x = x * mask.unsqueeze(-1)
        return x
