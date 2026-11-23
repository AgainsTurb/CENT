"""Stroke-level CENT denoiser: strokes in, strokes out, transcript as cross-attention.

Why this replaces the character-decomposed model: CASIA text lines cannot be split into
per-character stroke groups (measured 0.597 strokes per lexicon stroke; 0.8% of lines admissible),
so per-token character indexing and a separate connector token have nothing to attach to.

What is kept:
  * the same Bézier payload (8 scalars per stroke) and the same differentiable renderer;
  * style conditioning from k reference lines;
  * a per-boundary decision head -- now "does this stroke connect to the previous one?", which is a
    geometric fact of the data rather than an inferred label.

What changes:
  * no per-token character embedding; the line transcript enters through cross-attention, so one
    line's text conditions the whole stroke sequence;
  * there is NO join classifier. Connectiviity is supervised geometrically on the predicted clean
    strokes (see cent/train_strokes.py): a per-boundary classifier trained on noisy hidden states
    sat at its prior in three runs, and a supervised probe showed the label was barely predictable
    from the tokens. The phenomenon is a property of where the endpoints land, so it is trained and
    measured there.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .line import PAYLOAD


class TextEncoder(nn.Module):
    """Encodes the line transcript as a sequence the denoiser can cross-attend to."""

    def __init__(self, n_classes: int, dim: int = 256, depth: int = 2, n_heads: int = 4,
                 max_chars: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, dim, padding_idx=n_classes)
        self.pos = nn.Parameter(torch.randn(max_chars, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim * 2, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, depth)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # ids: [B, C] with n_classes as the padding id; mask: [B, C] float
        L = min(ids.shape[1], self.pos.shape[0])
        ids = ids[:, :L]
        if mask is not None:
            mask = mask[:, :L]
        x = self.emb(ids) + self.pos[:L]
        pad = None if mask is None else (mask < 0.5)
        out = self.enc(x, src_key_padding_mask=pad)
        return out


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device))
                      * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class StrokeDenoiser(nn.Module):
    """1-D denoiser over a stroke sequence with style and transcript conditioning."""

    def __init__(self, dim: int = 256, depth: int = 4, n_heads: int = 4, n_classes: int = 2680,
                 k_refs: int = 5, max_strokes: int = 512, text_depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(PAYLOAD, dim)
        self.pos = nn.Parameter(torch.randn(max_strokes, dim) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text = TextEncoder(n_classes, dim, text_depth, n_heads)
        # alignment features: where this stroke sits in the line, and the line's stroke/char budget
        self.feat = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.style_proj = nn.Linear(dim, dim * max(k_refs, 1))
        self.text_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim * 2, batch_first=True, norm_first=True,
                                       dropout=dropout)
            for _ in range(depth)])
        self.out = nn.Linear(dim, PAYLOAD)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor,
                style_tokens: torch.Tensor, text_ids: torch.Tensor,
                text_mask: torch.Tensor | None = None, pos: torch.Tensor | None = None,
                scalars: torch.Tensor | None = None, local_window: bool = True):
        B, N = x.shape[:2]
        h = self.inp(x) + self.pos[:N] + self.t_mlp(timestep_embedding(t, self.inp.out_features))[:, None]
        # style summary added as a bias (the k style tokens are already aligned with tokens elsewhere
        # in the project; here a pooled bias is sufficient and keeps the head count low)
        st = self.style_proj(style_tokens.mean(1))
        h = h + st.view(B, -1, h.shape[-1]).mean(1)[:, None]
        txt = self.text(text_ids, text_mask)
        if pos is not None:
            # ALIGNMENT. Without this the model had no way to bind a stroke to a character, and it
            # answered with the conditional mean of many placements -- rendered as a smear.
            sc = scalars if scalars is not None else x.new_zeros(x.shape[0], 2)
            f = torch.stack([pos, sc[:, 0:1].expand(-1, N), sc[:, 1:2].expand(-1, N)], dim=-1)
            h = h + self.feat(f)
        pad = mask < 0.5
        attn_mask = None
        if local_window and pos is not None and text_ids is not None and text_ids.shape[1] > 1:
            C = text_ids.shape[1]
            centre = pos * (C - 1)                                  # [B,N] expected char index
            idx = torch.arange(C, device=x.device, dtype=x.dtype).view(1, 1, -1)
            # half-window ~ 3 characters, widened when strokes are long and chars few
            half = min(max(3.0 * C / max(N, 1), 2.0), float(C))
            allow = (idx - centre.unsqueeze(-1)).abs() <= half      # [B,N,C]
            if text_mask is not None:
                allow = allow & (text_mask > 0.5).unsqueeze(1)
            # A row of the mask must never be empty: an all-False row makes softmax produce NaN,
            # which surfaces as a device-side assert. Two fallbacks, in order -- keep the valid
            # characters, then allow everything -- both expressed with `where` (no scatter, which
            # is where the bool edge case lived).
            empty = ~allow.any(-1, keepdim=True)
            if bool(empty.any()):
                keep_valid = (text_mask > 0.5).unsqueeze(1).expand_as(allow) if text_mask is not None \
                    else torch.ones_like(allow)
                allow = torch.where(empty, keep_valid, allow)
                still = ~allow.any(-1, keepdim=True)
                if bool(still.any()):
                    allow = torch.where(still, torch.ones_like(allow), allow)
            attn_mask = (~allow).repeat_interleave(self.blocks[0].self_attn.num_heads, 0)
        for blk in self.blocks:
            # attend over strokes, then over the transcript, keeping the stroke axis
            h = blk(h, src_key_padding_mask=pad)
            a, _ = self.text_attn(
                h, txt, txt,
                key_padding_mask=None if text_mask is None else (text_mask < 0.5),
                attn_mask=attn_mask)
            h = h + a
        return self.out(h)
