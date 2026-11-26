"""Option A baseline: autoregressive line stream with CursiveEOC pen states.

This is the ablation that shows *why* the boundary-slot DiT (Option B) is
needed: an AR model can represent a join (it is just a pen state), but its
boundary decision is entangled with every previous point, and error accumulates
along the line. Kept deliberately small and dependency-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ARLineModel(nn.Module):
    def __init__(self, *, d: int = 128, depth: int = 3, n_heads: int = 4,
                 n_mix: int = 20, n_chars: int = 500, n_pen: int = 4,
                 max_len: int = 2048) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2 + n_pen, d)
        self.char_emb = nn.Embedding(n_chars, d, padding_idx=0)
        self.style_proj = nn.Linear(d, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.n_mix = n_mix
        self.head = nn.Linear(d, 6 * n_mix + n_pen)

    def forward(self, dxy: torch.Tensor, pen: torch.Tensor, char_ids: torch.Tensor,
                style: torch.Tensor, mask: torch.Tensor | None = None):
        """dxy (B,T,2), pen (B,T) long, char_ids (B,T) long, style (B,d)."""
        B, T, _ = dxy.shape
        pen_oh = torch.nn.functional.one_hot(pen, num_classes=4).float()
        x = torch.cat([dxy, pen_oh], dim=-1)
        h = self.in_proj(x)
        h = h + self.char_emb(char_ids.clamp(min=0))
        h = h + self.style_proj(style).unsqueeze(1)
        pos = self.pos(torch.arange(T, device=dxy.device)).unsqueeze(0)
        h = h + pos

        causal = torch.triu(
            torch.ones(T, T, device=dxy.device, dtype=torch.bool), diagonal=1
        )
        out = self.enc(h, mask=causal)
        y = self.head(out)
        M = self.n_mix
        gmm = y[..., : 6 * M].view(B, T, M, 6)
        pen_logits = y[..., 6 * M:]
        return {"gmm": gmm, "pen_logits": pen_logits}
