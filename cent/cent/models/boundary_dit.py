"""BoundaryDiT — the CENT model.

Core idea (contribution ii): the join between two characters is not a
post-processing step. It is a **slot** in the latent sequence,

    [ frames(char 1) | B1 | frames(char 2) | B2 | ... | frames(char n) ]

where ``Bi`` is a boundary token that (a) is predicted as lift / join by a
dedicated head, and (b) *attends to both neighbours and is attended to by
them* on every denoising step. Because the DiT is bidirectional and denoises
the whole line jointly, the decision conditions the shape of the tail of
character i and the head of character i+1 — which is exactly the property that
"generate the glyphs, then join them" cannot have.

No layout module exists anywhere in this file: position is carried by the
frame latents themselves.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Linear(int(d * mlp_ratio), d)
        )
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate1.unsqueeze(1) * a
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class BoundaryDiT(nn.Module):
    def __init__(self, *, z_dim: int = 64, d: int = 128, depth: int = 3,
                 n_heads: int = 4, n_chars: int = 500, max_tokens: int = 512,
                 n_pen: int = 4) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.d = d
        self.n_pen = n_pen

        self.in_proj = nn.Linear(z_dim, d)
        self.type_emb = nn.Embedding(2, d)                 # 0 frame, 1 boundary
        self.bnd_emb = nn.Parameter(torch.zeros(d))        # boundary seed token
        self.label_emb = nn.Embedding(2, d)                # committed lift/join
        self.pos_emb = nn.Embedding(max_tokens, d)
        self.char_proj = nn.Linear(d, d)

        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.style_proj = nn.Linear(d, d)

        self.blocks = nn.ModuleList([AdaLNBlock(d, n_heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(d)
        self.frame_head = nn.Linear(d, z_dim)
        self.bnd_head = nn.Linear(d, 1)                    # lift vs join logit
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    @staticmethod
    def build_layout(frame_char: torch.Tensor, frame_mask: torch.Tensor):
        """Interleave frames and boundary tokens — fully vectorised.

        The previous implementation looped over batch x frames in Python, which
        dominated step time (~80 ms/step at batch 16). This version uses only
        cumsum/scatter, so it costs a few tensor ops instead of ~1000 Python
        iterations, with identical output.

        Returns
          token_type  (B, N) long   0 frame / 1 boundary
          token_char  (B, N) long   char id for frames, -1 for boundary
          token_left  (B, N) long   left char (boundary only)
          token_right (B, N) long   right char (boundary only)
          tok_mask    (B, N) bool
          frame_slot  (B, T_lat) long  token index of each frame (-1 if pad)
          bnd_slot    (B, T_lat-1) long token index of boundary between f,f+1
        """
        B, T = frame_char.shape
        dev = frame_char.device
        valid = frame_mask & (frame_char >= 0)

        b = torch.zeros(B, T, dtype=torch.bool, device=dev)
        if T > 1:
            b[:, :-1] = (valid[:, :-1] & valid[:, 1:]
                         & (frame_char[:, :-1] != frame_char[:, 1:]))

        n_valid = valid.sum(1)
        n_bnd = b.sum(1)
        N = int((n_valid + n_bnd).max().item()) if B else 1
        N = max(N, 1)

        frame_rank = torch.cumsum(valid.to(torch.long), 1) - 1
        bnd_before = torch.cumsum(b.to(torch.long), 1) - b.to(torch.long)
        frame_slot = torch.where(valid, frame_rank + bnd_before,
                                 torch.full_like(frame_rank, -1))

        tok_type = torch.zeros(B, N, dtype=torch.long, device=dev)
        tok_char = torch.full((B, N), -1, dtype=torch.long, device=dev)
        tok_left = torch.full((B, N), -1, dtype=torch.long, device=dev)
        tok_right = torch.full((B, N), -1, dtype=torch.long, device=dev)
        tok_mask = torch.zeros(B, N, dtype=torch.bool, device=dev)

        fs = frame_slot.clamp(min=0)
        tok_char.scatter_(1, fs, torch.where(valid, frame_char,
                                             torch.zeros_like(frame_char)))
        tok_mask.scatter_(1, fs, valid)

        bnd_slot_full = torch.full((B, T), -1, dtype=torch.long, device=dev)
        if T > 1:
            bnd_slot_full[:, :-1] = torch.where(
                b[:, :-1], frame_slot[:, :-1] + 1,
                torch.full_like(frame_slot[:, :-1], -1))
        bs = bnd_slot_full.clamp(min=0)
        tok_type.scatter_(1, bs, b.to(torch.long))

        left = torch.zeros(B, T, dtype=torch.long, device=dev)
        right = torch.zeros(B, T, dtype=torch.long, device=dev)
        if T > 1:
            left[:, :-1] = frame_char[:, :-1]
            right[:, :-1] = frame_char[:, 1:]
        tok_left.scatter_(1, bs, torch.where(b, left, torch.full_like(left, -1)))
        tok_right.scatter_(1, bs, torch.where(b, right, torch.full_like(right, -1)))

        return {
            "token_type": tok_type,
            "token_char": tok_char,
            "token_left": tok_left,
            "token_right": tok_right,
            "tok_mask": tok_mask | (tok_type == 1),
            "frame_slot": frame_slot,
            "bnd_slot": bnd_slot_full[:, : max(1, T - 1)],
        }

    # ------------------------------------------------------------------
    def forward(self, z_t: torch.Tensor, t: torch.Tensor, *,
                frame_char: torch.Tensor, frame_mask: torch.Tensor,
                char_ctx: torch.Tensor, style: torch.Tensor,
                bnd_commit: torch.Tensor | None = None):
        """z_t: (B,T_lat,z) noisy frames -> x0_pred, bnd_logits, layout."""
        B, T_lat, _ = z_t.shape
        lay = self.build_layout(frame_char, frame_mask)
        N = lay["token_type"].shape[1]

        # per-frame char embedding, gathered from the text encoder context
        #   char_ctx: (B, L, d) -> per-frame (B, T_lat, d)
        fchar = frame_char.clamp(min=0)
        L = char_ctx.shape[1]
        idx = fchar.clamp(max=max(0, L - 1))
        frame_ce = torch.gather(
            char_ctx, 1, idx.unsqueeze(-1).expand(-1, -1, char_ctx.shape[-1])
        )
        frame_ce = frame_ce * (frame_char >= 0).unsqueeze(-1).to(frame_ce.dtype)
        frame_ce = self.char_proj(frame_ce)

        # scatter into token positions (buffers must follow the activation dtype:
        # under bf16 autocast the text encoder output is bf16, not fp32)
        frame_tok = self.in_proj(z_t)
        dt = frame_tok.dtype
        ce_tok = torch.zeros(B, N, self.d, device=z_t.device, dtype=dt)
        b_idx = lay["frame_slot"].clamp(min=0)
        ce_tok.scatter_(1, b_idx.unsqueeze(-1).expand(-1, -1, self.d),
                        frame_ce.to(dt))
        # boundary tokens: mean of the two neighbouring character embeddings
        left = lay["token_left"].clamp(min=0)
        right = lay["token_right"].clamp(min=0)
        left_ce = torch.gather(
            char_ctx, 1, left.clamp(max=L - 1).unsqueeze(-1).expand(-1, -1, self.d)
        )
        right_ce = torch.gather(
            char_ctx, 1, right.clamp(max=L - 1).unsqueeze(-1).expand(-1, -1, self.d)
        )
        bnd_mask_tok = (lay["token_type"] == 1).unsqueeze(-1).to(dt)
        ce_tok = ce_tok + bnd_mask_tok * 0.5 * (left_ce + right_ce).to(dt)

        # base tokens
        tok = torch.zeros(B, N, self.d, device=z_t.device, dtype=dt)
        f_idx = lay["frame_slot"].clamp(min=0)
        tok.scatter_(1, f_idx.unsqueeze(-1).expand(-1, -1, self.d), frame_tok)
        tok = tok + bnd_mask_tok * self.bnd_emb.to(dt).view(1, 1, -1)

        pos = torch.arange(N, device=z_t.device).unsqueeze(0).expand(B, -1)
        tok = (tok + self.type_emb(lay["token_type"]).to(dt)
               + ce_tok.to(dt)
               + self.pos_emb(pos.clamp(max=self.max_tokens - 1)).to(dt))

        if bnd_commit is not None:
            lab = bnd_commit.clamp(min=0, max=1).long()
            has = (bnd_commit >= 0).unsqueeze(-1).to(tok.dtype)
            b_slot = lay["bnd_slot"].clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d)
            emb = has.unsqueeze(-1) * self.label_emb(lab).to(tok.dtype).unsqueeze(2)
            emb = emb.squeeze(2)
            tok.scatter_add_(1, b_slot, emb)

        cond = self.t_mlp(sinusoidal_embedding(t, self.d)) + self.style_proj(style)
        pad = ~lay["tok_mask"]
        for blk in self.blocks:
            tok = blk(tok, cond, key_padding_mask=pad)
        tok = self.norm_out(tok)

        # frame predictions
        f_tok = torch.gather(tok, 1, f_idx.unsqueeze(-1).expand(-1, -1, self.d))
        x0 = self.frame_head(f_tok)
        x0 = x0 * frame_mask.unsqueeze(-1).to(x0.dtype)

        # boundary logits
        b_tok = torch.gather(tok, 1, lay["bnd_slot"].clamp(min=0).unsqueeze(-1)
                             .expand(-1, -1, self.d))
        logits = self.bnd_head(b_tok).squeeze(-1)
        valid = (lay["bnd_slot"] >= 0)
        logits = logits.masked_fill(~valid, 0.0)

        return {"x0": x0, "bnd_logits": logits, "bnd_valid": valid, "layout": lay}
