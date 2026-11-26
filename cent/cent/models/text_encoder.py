"""Content text encoder: character sequence -> embeddings.

Deliberately thin. For a large Chinese vocabulary (3.7k-7.2k classes) an
embedding table is enough; the paper's claim is about the *boundary* mechanism,
not about the text encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, n_chars: int, d: int = 128, depth: int = 2, n_heads: int = 4,
                 max_len: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_chars, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.d = d
        self.max_len = max_len

    def forward(self, char_seq: torch.Tensor, char_mask: torch.Tensor) -> torch.Tensor:
        """(B, L) ids, (B, L) mask -> (B, L, d) contextual embeddings."""
        B, L = char_seq.shape
        L = min(L, self.max_len)
        char_seq = char_seq[:, :L]
        char_mask = char_mask[:, :L]
        x = self.emb(char_seq)
        pos = self.pos(torch.arange(L, device=char_seq.device)).unsqueeze(0)
        x = x + pos
        # transformer wants True = ignore
        pad = ~char_mask
        h = self.enc(x, src_key_padding_mask=pad)
        return h

    def gather_chars(self, h: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        """(B,L,d) + (B,T) char ids (-1 pad) -> (B,T,d)."""
        B, T = char_ids.shape
        L = h.shape[1]
        idx = char_ids.clamp(min=0, max=max(0, L - 1))
        out = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        out = out * (char_ids >= 0).unsqueeze(-1).to(out.dtype)
        return out
