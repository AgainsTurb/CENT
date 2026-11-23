"""CENT's style encoder and stroke-graph denoiser.

Style: k reference images -> z_g (global) + z_l (per-stroke-type local).

Note on the local-style mechanism. The design in the paper says "one local
vector per stroke type (5 basic categories)". Stroke type is not available at
inference -- we generate geometry, not labelled strokes -- and our lexicon
provides topology but not types. So the implementation uses **five learned type
anchors** and lets each stroke attend to them with a soft query. The anchors are
free parameters, not supervised by stroke-type labels. This is a deviation from
the paper's wording and is recorded here and in DEVIATIONS.md; ablation A3 still
measures whether the local branch earns its place, which is what the claim needs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stroke_graph import COORDS, N_MAX


class StyleEncoder(nn.Module):
    """k reference glyphs -> global + local style codes."""

    def __init__(self, dim: int = 256, k_max: int = 15, n_types: int = 5, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.k_max = k_max
        self.n_types = n_types
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, dim, 3, 2, 1), nn.GroupNorm(8, dim), nn.GELU(),
        )
        self.proj = nn.Linear(dim, dim)
        self.type_queries = nn.Parameter(torch.randn(n_types, dim) * 0.02)
        self.cross = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.writer_head = nn.Linear(dim, 1)   # 1-d placeholder, replaced by n_writers at use

    def forward(self, refs: torch.Tensor, ref_mask: torch.Tensor | None = None):
        """refs [B,k,1,S,S] -> (z_g [B,D], z_l [B,n_types,D], feats [B,k,D])."""
        B, K = refs.shape[:2]
        x = refs.reshape(B * K, *refs.shape[2:])
        f = self.trunk(x).mean(dim=(-2, -1))            # [B*K,D]
        f = self.proj(f).reshape(B, K, self.dim)
        if ref_mask is None:
            ref_mask = torch.ones(B, K, device=f.device, dtype=f.dtype)
        w = ref_mask / ref_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z_g = (f * w.unsqueeze(-1)).sum(dim=1)          # [B,D]
        q = self.norm_q(self.type_queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.norm_kv(f)
        z_l, _ = self.cross(q, kv, kv, key_padding_mask=(ref_mask < 0.5))  # [B,n_types,D]
        return z_g, z_l, f


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock1D(nn.Module):
    """AdaLN-conditioned 1-D residual block."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.cond = nn.Linear(cond_dim, dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.conv2(self.norm2(h).transpose(1, 2)).transpose(1, 2)
        out = x + self.drop(h)
        return out * mask.unsqueeze(-1) if mask is not None else out


class StrokeDenoiser(nn.Module):
    """Predicts the noise on stroke tokens [B,N_MAX,8].

    Conditioning: timestep (AdaLN), character (additive token), and style through
    cross-attention to [z_g; z_l] (n_types+1 tokens).
    """

    def __init__(self, dim: int = 256, depth: int = 4, n_heads: int = 4,
                 n_classes: int = 3755, n_types: int = 5, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.inp = nn.Linear(COORDS, dim)
        self.pos = nn.Parameter(torch.randn(N_MAX, dim) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.char_emb = nn.Embedding(n_classes, dim)
        self.char_null = nn.Parameter(torch.randn(dim) * 0.02)
        self.style_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ResBlock1D(dim, dim) for _ in range(depth)])
        self.cross = nn.ModuleList(
            [nn.MultiheadAttention(dim, n_heads, batch_first=True) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, COORDS)

    def forward(self, x: torch.Tensor, t: torch.Tensor, char_idx: torch.Tensor | None,
                style_tokens: torch.Tensor, mask: torch.Tensor | None = None,
                drop_char: bool = False) -> torch.Tensor:
        """x [B,N,8], t [B], char_idx [B] or None, style_tokens [B,1+n_types,D] -> eps [B,N,8]."""
        B, N = x.shape[:2]
        h = self.inp(x) + self.pos[:N].unsqueeze(0)
        cond = self.t_mlp(timestep_embedding(t, self.dim))
        if char_idx is not None and not drop_char:
            cond = cond + self.char_emb(char_idx)
        elif char_idx is None or drop_char:
            cond = cond + self.char_null
        kv = self.style_norm(style_tokens)
        for blk, xa in zip(self.blocks, self.cross):
            h = blk(h, cond, mask)
            q = h
            a, _ = xa(q, kv, kv)
            h = h + a
            if mask is not None:
                h = h * mask.unsqueeze(-1)
        h = self.out_norm(h)
        eps = self.out(h)
        return eps * mask.unsqueeze(-1) if mask is not None else eps
