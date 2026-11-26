"""Vectorised character-group assembly for the per-character verifier.

The obvious implementation loops over (batch, character) and boolean-indexes the token axis:

    for b in range(B):
        for j in range(n_chars):
            sel = (kind[b] == KIND_CHAR_STROKE) & (slot[b] == j) & mask[b].bool()
            groups.append(x0[b][sel])

At B=16 and 32 characters that is 512 iterations of ~10 small GPU ops each: measured **143 ms
per step**, more than the denoiser forward pass, and it grows with the batch. It is pure launch
overhead --- the arithmetic it performs is negligible.

This module does the same work in a fixed number of kernels. It relies on one invariant the
dataset guarantees: within a line, tokens are emitted in character order, so the (sample,
character) keys of the selected tokens are non-decreasing, hence contiguous runs. That makes the
within-group rank computable with `unique_consecutive` + `repeat_interleave`, no Python loop.
"""
from __future__ import annotations

import torch

from .line import KIND_CHAR_STROKE


def build_char_groups(x0: torch.Tensor, kind: torch.Tensor, slot: torch.Tensor,
                      mask: torch.Tensor, char_ids: torch.Tensor,
                      max_strokes: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group stroke tokens by character without a Python loop.

    Args:
        x0: [B, T, 8] payloads (control points) at the current estimate.
        kind: [B, T] token kinds.
        slot: [B, T] character position within the line.
        mask: [B, T] token validity.
        char_ids: [B, n_chars] class id per character, -1 for padding.
        max_strokes: cap on strokes per character (24 is the observed maximum).
    Returns:
        (groups [G, S, 8], group_mask [G, S], labels [G]) with G = samples x characters.
    """
    B, T = kind.shape
    n_chars = char_ids.shape[1]
    dev = x0.device

    sel = (kind == KIND_CHAR_STROKE) & mask.bool()
    if not bool(sel.any()):
        return x0.new_zeros(0, 1, x0.shape[-1]), x0.new_zeros(0, 1), torch.zeros(0, dtype=torch.long, device=dev)

    bi, ti = sel.nonzero(as_tuple=True)              # [N]
    jj = slot[bi, ti]                                # [N]
    key = bi * n_chars + jj

    # contiguous runs of equal keys -> within-run rank, no loop
    _, counts = torch.unique_consecutive(key, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=counts.dtype, device=dev), counts.cumsum(0)[:-1]])
    rank = torch.arange(key.numel(), device=dev) - torch.repeat_interleave(starts, counts)

    S = int(counts.max().item())
    S = max(1, min(S, max_strokes))

    clean = rank < S                                 # drop overflow, keep tensors rectangular
    bi, jj, rank, ti = bi[clean], jj[clean], rank[clean], ti[clean]

    G = x0.new_zeros(B, n_chars, S, x0.shape[-1])
    G[bi, jj, rank] = x0[bi, ti]
    gm = x0.new_zeros(B, n_chars, S)
    gm[bi, jj, rank] = 1.0

    has = gm.any(-1)                                 # [B, n_chars] characters actually present
    labels = char_ids[has]
    return G[has], gm[has], labels
