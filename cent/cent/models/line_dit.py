"""LineDiT — text-line diffusion with NO layout of any kind.

Contribution (ii) is preserved and sharpened: there is no `frame_char`, no
per-character frame allocation, no boundary token, and no head predicting which
character sits where. The character sequence conditions the latent through
**cross-attention only**, so where each glyph lands and whether neighbours are
joined is decided by the model, per boundary, from context — dynamic continuity
rather than an all-or-nothing rule.

Joins are not classified here. They live where they are observable: in the
decoder's 4-way pen state (PEN_MOVE / PEN_UP / EOC / CURSIVE_EOC). See
`decode_pen` for the decode-time cursiveness knob.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .boundary_dit import AdaLNBlock, sinusoidal_embedding


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

        # text conditioning: frames attend to the character sequence.
        # No frame->character map exists anywhere in this model.
        self.xattn = nn.MultiheadAttention(d, n_heads, batch_first=True)
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

        q = self.xnorm(h)
        a, _ = self.xattn(q, char_ctx, char_ctx,
                          key_padding_mask=~char_mask, need_weights=False)
        h = h + a

        cond = self.t_mlp(sinusoidal_embedding(t, self.d)) + self.style_proj(style)
        for blk in self.blocks:
            h = blk(h, cond)
        return {"x0": self.frame_head(self.norm_out(h))}
