"""CENT's line model: a denoiser over an interleaved token sequence.

Sequence layout (one line):

    [ char_stroke × N_c1 ][ layout ][ connector ][ char_stroke × N_c2 ][ layout ] ... [ connector ]

Every item is a payload of 8 scalars (a cubic Bezier's control points; for layout,
(dx, dy, theta) padded), plus a kind id and the index of the character it belongs to.
The connector tokens carry the join decision: a binary head reads the connector
token's hidden state and predicts P(join). That is the object the paper claims as its
contribution, and it is a real tensor here, not a diagram.

Why one sequence rather than separate per-character models: a connector depends on the
exit of the previous character and the entry of the next, and on the writer's habit, so
the geometry has to attend across characters. A 1-D U-Net with style cross-attention
does that in one pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .style import ResBlock1D, StyleEncoder, timestep_embedding

PAYLOAD = 8
KIND_CHAR_STROKE, KIND_LAYOUT, KIND_CONNECTOR = 0, 1, 2
N_KINDS = 3


class LineDenoiser(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 4, n_heads: int = 4,
                 n_classes: int = 3755, n_kinds: int = N_KINDS,
                 max_tokens: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.inp = nn.Linear(PAYLOAD, dim)
        self.kind_emb = nn.Embedding(n_kinds, dim)
        self.char_emb = nn.Embedding(n_classes + 1, dim)   # +1 for "no character"
        self.pos = nn.Parameter(torch.randn(max_tokens, dim) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.style_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ResBlock1D(dim, dim, dropout) for _ in range(depth)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(dim, n_heads, batch_first=True)
                                   for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, PAYLOAD)
        self.join_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, kind: torch.Tensor,
                char_pos: torch.Tensor, style_tokens: torch.Tensor,
                mask: torch.Tensor | None = None,
                connector_mask: torch.Tensor | None = None):
        """x [B,T,8], kind [B,T], char_pos [B,T], style [B,1+ntypes,D].
        Returns (eps [B,T,8], join_logits [B,C]) where C = max connectors per line."""
        B, T = x.shape[:2]
        h = self.inp(x) + self.kind_emb(kind) + self.char_emb(char_pos).double().to(x.dtype)
        h = h + self.pos[:T].unsqueeze(0)
        cond = self.t_mlp(timestep_embedding(t, self.dim))
        kv = self.style_norm(style_tokens)
        for blk, xa in zip(self.blocks, self.cross):
            h = blk(h, cond, mask)
            a, _ = xa(h, kv, kv)
            h = h + a
            if mask is not None:
                h = h * mask.unsqueeze(-1)
        h = self.out_norm(h)
        eps = self.out(h)
        if mask is not None:
            eps = eps * mask.unsqueeze(-1)

        join_logits = None
        if connector_mask is not None:
            # gather connector positions and score them
            sel = (kind == KIND_CONNECTOR) & connector_mask.bool()
            if sel.any():
                logits_flat = self.join_head(h).squeeze(-1)          # [B,T]
                join_logits = torch.where(sel, logits_flat,
                                          torch.full_like(logits_flat, -1e4))
        return eps, join_logits


class LineStyleEncoder(StyleEncoder):
    """Style encoder over reference LINE images (any width; the CNN pools globally)."""

    def __init__(self, dim: int = 256, k_max: int = 5, n_types: int = 5, n_heads: int = 4):
        super().__init__(dim=dim, k_max=k_max, n_types=n_types, n_heads=n_heads)
        # connector style: one extra query vector attending over the same features
        self.connector_query = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, refs: torch.Tensor, ref_mask: torch.Tensor | None = None):
        z_g, z_l, feats = super().forward(refs, ref_mask)
        q = self.norm_q(self.connector_query).view(1, 1, -1).expand(z_g.shape[0], -1, -1)
        kv = self.norm_kv(feats)
        z_k, _ = self.cross(q, kv, kv,
                            key_padding_mask=(ref_mask < 0.5) if ref_mask is not None else None)
        return z_g, z_l, z_k.squeeze(1), feats
