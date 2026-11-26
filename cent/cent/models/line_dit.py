"""LineDiT — text-line diffusion with NO layout, but WITH order.

A vanilla positional embedding is not enough here: the DiT denoises every frame
jointly and non-causally, so nothing forces character l to land anywhere in
particular, and the fastest descent is "predict the average line".

What is added is *order*, never *width*:

    score(t, l) = q_t·k_l/√d  −  λ_h · (u_t − v_l)²
    u_t = t/(T−1),  v_l = l/(L−1),  λ_h = softplus(learnable, per head)

This is a Gaussian band along the diagonal. It constrains only that characters
appear in sequence, not how many frames each occupies, so unequal glyph widths
stay free and nothing is pre-generated at train or inference time. λ is learned
and soft: heads may override it, which is what allows dynamic behaviour.

Joins are not decided here. They live in the decoder's pen state (MOVE vs UP).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .boundary_dit import AdaLNBlock, sinusoidal_embedding


class MonotoneCrossAttention(nn.Module):
    """Cross-attention with a learned monotone-order penalty on the logits."""

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.hd = d // n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        # per-head strength of the order prior; starts almost free (λ≈0.10)
        self.lam = nn.Parameter(torch.full((n_heads,), -2.2))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k = k.view(B, L, self.h, self.hd).transpose(1, 2)
        v = v.view(B, L, self.h, self.hd).transpose(1, 2)

        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hd)   # B,H,T,L
        u = torch.arange(T, device=x.device, dtype=x.dtype) / max(1, T - 1)
        w = torch.arange(L, device=x.device, dtype=x.dtype) / max(1, L - 1)
        lam = torch.nn.functional.softplus(self.lam).view(1, self.h, 1, 1)
        s = s - lam * (u.view(1, 1, T, 1) - w.view(1, 1, 1, L)) ** 2

        s = s.masked_fill(~ctx_mask.view(B, 1, 1, L), float("-inf"))
        a = torch.softmax(s, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, T, self.h * self.hd)
        return self.out(o)


class LineDiT(nn.Module):
    def __init__(self, *, z_dim: int = 64, d: int = 384, depth: int = 12,
                 n_heads: int = 8, max_tokens: int = 512) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.d = d
        self.max_tokens = max_tokens

        self.in_proj = nn.Linear(z_dim, d)
        self.pos_emb = nn.Embedding(max_tokens, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.style_proj = nn.Linear(d, d)

        self.xattn = MonotoneCrossAttention(d, n_heads)
        self.xnorm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([AdaLNBlock(d, n_heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(d)
        self.frame_head = nn.Linear(d, z_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, *,
                char_ctx: torch.Tensor, char_mask: torch.Tensor,
                style: torch.Tensor) -> dict:
        B, T, _ = z_t.shape
        pos = torch.arange(T, device=z_t.device)
        h = self.in_proj(z_t) + self.pos_emb(pos)[None]

        h = h + self.xattn(self.xnorm(h), char_ctx, char_mask)

        cond = self.t_mlp(sinusoidal_embedding(t, self.d)) + self.style_proj(style)
        for blk in self.blocks:
            h = blk(h, cond)
        return {"x0": self.frame_head(self.norm_out(h)), "lam": self.xattn.lam}
