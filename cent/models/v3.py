"""CENT v3 — autoregressive stroke model.

One module, one responsibility. The mapping is deliberately explicit because the paper's
claims are attributed to named modules:

    M1 TranscriptEncoder   what to write            (content)
    M2 StyleEncoder        who writes it            (writer identity)
    M3 LayoutPlanner       where characters sit     (spacing, slant, baseline rhythm)
    M4 JoinPlanner         whether/how two chars connect  (inter-character continuity)
    M5 StrokeDecoder       stroke geometry + order  (intra-character structure, pen path)
    M6 GaussianMemoryMask  which character a stroke belongs to (alignment)
    M7 WriterHead          writer identification    (auxiliary supervision only)
    M8 refine_boundaries   C1 smoothness at joins   (inference-time, closed form)

The pen path is continuous **by construction**: M5 emits each stroke as an offset from the
current pen position, and M4 supplies a boundary contract that fixes the offset of the first
stroke of every character. A join is therefore a coordinate the decoder must hit, not a label
it must guess.

Nothing here is verified on hardware yet; see docs/ARCHITECTURE_V3.md for the staged build and
the falsification test that gates each stage.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .line import LineStyleEncoder as StyleEncoder  # returns (z_g, z_l, z_kappa)
from .style import timestep_embedding  # noqa: F401  (re-exported for callers)

CONT = 8          # dx,dy + three control offsets relative to the stroke start
N_PEN = 2         # pen-state embedding table
EPS = 1e-6


# --------------------------------------------------------------------------------------
# M6 — alignment
# --------------------------------------------------------------------------------------
def gaussian_memory_mask(char_idx: torch.Tensor, n_chars: int,
                         sigma: float = 1.0) -> torch.Tensor:
    """[B, L] stroke->character assignment  ->  additive mask [B, L, n_chars].

    TrInk (EMNLP 2025) reports CER 8.5 with a Gaussian memory mask against 14.9 for a
    uniform local window and 14.1 for exponential decay, and 70.1 with no mask at all.
    An earlier revision of CENT used exactly the uniform-window variant, the worst of the
    three, which is why this function exists as a named module rather than an inline tweak.
    """
    j = torch.arange(n_chars, device=char_idx.device, dtype=torch.float32).view(1, 1, -1)
    mu = char_idx.to(torch.float32).unsqueeze(-1)
    return -((j - mu) ** 2) / (2.0 * max(sigma, EPS) ** 2)


# --------------------------------------------------------------------------------------
# M5 + M6 — stroke decoder
# --------------------------------------------------------------------------------------
class StrokeDecoder(nn.Module):
    """Causal decoder over stroke tokens, cross-attending to the transcript under M6."""

    def __init__(self, dim: int, depth: int, n_heads: int, n_classes: int,
                 max_strokes: int, max_chars: int, n_comp: int = 6, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_comp = n_comp
        self.max_chars = max_chars
        self.cont = nn.Linear(CONT, dim)
        self.pen = nn.Embedding(N_PEN, dim)
        self.char = nn.Embedding(max_chars + 1, dim)
        # +1 for the BOS token prepended at inference and in the shifted training input
        self.pos = nn.Parameter(torch.zeros(1, max_strokes + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.start = nn.Parameter(torch.zeros(1, 1, dim))

        layer = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.text_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        self.mdn_pi = nn.Linear(dim, n_comp)
        self.mdn_mu = nn.Linear(dim, n_comp * CONT)
        self.mdn_ls = nn.Linear(dim, n_comp * CONT)
        # ABSOLUTE START. Predicting only the offset from the previous stroke end makes every
        # error shift all later strokes: measured free-running drift to y=5.0 on a [0,1]
        # canvas after 64 strokes. An absolute head bounds each stroke independently.
        self.abs_head = nn.Linear(dim, 2)
        self.pen_head = nn.Linear(dim, 1)
        self.stop_head = nn.Linear(dim, 1)

    # ---- forward one teacher-forced pass -------------------------------------------
    def forward(self, cont: torch.Tensor, pen: torch.Tensor, char_idx: torch.Tensor,
                mask: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor | None,
                sigma: float = 1.0):
        """cont [B,L,8], pen/char_idx/mask [B,L], mem [B,C,D] = M1 transcript encoding."""
        B, L, _ = cont.shape
        h = self.cont(cont) + self.pen(pen) + self.char(char_idx.clamp(max=self.max_chars))
        h = h + self.pos[:, :L] + self.start
        causal = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=causal, src_key_padding_mask=(mask < 0.5))

        # M6: Gaussian memory mask, additive over text logits
        C = mem.shape[1]
        am = gaussian_memory_mask(char_idx.clamp(max=C - 1), C, sigma)
        am = am.unsqueeze(1).expand(-1, self.text_attn.num_heads, -1, -1)
        am = am.reshape(B * self.text_attn.num_heads, L, C)
        a, _ = self.text_attn(h, mem, mem, attn_mask=am, need_weights=False,
                              key_padding_mask=None if mem_mask is None else (mem_mask < 0.5))
        h = self.norm(h + a)

        pi = F.log_softmax(self.mdn_pi(h), dim=-1)
        mu = self.mdn_mu(h).view(B, L, self.n_comp, CONT)
        ls = self.mdn_ls(h).view(B, L, self.n_comp, CONT).clamp(-8.0, 4.0)
        return {"pi": pi, "mu": mu, "ls": ls,
                "abs": torch.sigmoid(self.abs_head(h)),
                "pen": self.pen_head(h).squeeze(-1),
                "stop": self.stop_head(h).squeeze(-1)}

    # ---- losses ---------------------------------------------------------------------
    @staticmethod
    def nll(out: dict, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pi, mu, ls = out["pi"], out["mu"], out["ls"]
        inv = torch.exp(-ls)
        d2 = ((target.unsqueeze(2) - mu) * inv) ** 2
        logp = -0.5 * (d2.sum(-1) + 2.0 * ls.sum(-1) + CONT * math.log(2 * math.pi))
        return -(torch.logsumexp(pi + logp, dim=-1) * mask).sum() / mask.sum().clamp_min(1.0)


class CenterModel(nn.Module):
    """M1..M7 assembled. M8 is a closed-form function, not a layer."""

    def __init__(self, dim: int = 384, depth: int = 8, n_heads: int = 8, n_classes: int = 2680,
                 style_dim: int = 256, style_depth: int = 4, n_style_types: int = 5,
                 max_strokes: int = 320, max_chars: int = 64, n_comp: int = 6,
                 n_writers: int = 1024, dropout: float = 0.0):
        super().__init__()
        # M1 transcript encoder
        self.char_emb = nn.Embedding(n_classes + 1, dim)
        self.char_pos = nn.Parameter(torch.zeros(1, max_chars + 1, dim))
        nn.init.trunc_normal_(self.char_pos, std=0.02)
        enc = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.transcript = nn.TransformerEncoder(enc, 2, enable_nested_tensor=False)

        # M2 style encoder (reads INK, see cent/models/style.py)
        self.style = StyleEncoder(dim=style_dim, k_max=5, n_types=n_style_types, n_heads=4)
        self.style_to_dim = nn.Linear(style_dim, dim)
        self.style_tokens = nn.Linear(style_dim, dim)
        self.style_k = nn.Linear(style_dim, dim)   # z_kappa lives in style_dim

        # M3 layout planner
        self.layout = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, n_heads, 2 * dim, dropout, batch_first=True,
                                       norm_first=True, activation="gelu"),
            2, enable_nested_tensor=False)
        # M3' soft anchors: (x_center, y_center, w, h, dx_advance). Five numbers, no clip
        # region: strokes may leave every anchor, which is what keeps cursive overshoot
        # and inter-character connection expressible.
        self.layout_head = nn.Linear(dim, 5)
        self.anchor_ctx = nn.Linear(5 * 3, dim)       # a_{i-1}, a_i, a_{i+1}
        # SLOT POSITIONAL CODE (soft layout, no boxes): a Fourier code of the character's
        # position along the line, added to the transcript memory. Position becomes a
        # property of identity rather than of a chained prediction, and no layout stage or
        # clip rectangle is needed. Gated by `use_slot_pos` so it can be A/B'd.
        self.slot_pos = nn.Linear(8, dim)
        # CHARACTER DICTIONARY (A.1): a learned per-character prototype of its strokes,
        # in token space. The decoder predicts a RESIDUAL on top of this, so glyph shape
        # has a content prior instead of being inferred from the transcript alone.
        self.char_dict = nn.Embedding(n_classes, 64)
        self.char_dict_head = None      # built lazily: 64 -> max_strokes*8
        self.use_slot_pos = False
        self.count_head = nn.Linear(dim, 13)          # strokes for this character, 1..13

        # M4 join planner
        # features are [H_i, H_j, box_i, box_j, prior, z_kappa] = 2*dim + 8 + 1 + dim
        self.join = nn.Sequential(nn.Linear(3 * dim + 9, 2 * dim), nn.GELU(),
                                  nn.Linear(2 * dim, 1 + 2 + 2 + 4))
        # M5 (+M6)
        self.decoder = StrokeDecoder(dim, depth, n_heads, n_classes, max_strokes,
                                     max_chars, n_comp, dropout)
        # M7 writer head (auxiliary)
        self.writer_head = nn.Linear(dim, n_writers)

    # ---- encoding -------------------------------------------------------------------
    def encode(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
               refs: torch.Tensor, ref_mask: torch.Tensor):
        B, C = text_ids.shape
        h = self.char_emb(text_ids) + self.char_pos[:, :C]
        H = self.transcript(h, src_key_padding_mask=(text_mask < 0.5))
        z_g, z_l, z_k, _feats = self.style(refs, ref_mask)
        zg = self.style_to_dim(z_g)
        # Local style tokens must reach the loss, otherwise z_l is dead weight: the smoke test
        # caught exactly that (style.type_queries and style_tokens had no gradient at all, because
        # nothing downstream consumed z_l). Fold them into the transcript memory.
        H = H + self.style_tokens(z_l.mean(dim=1)).unsqueeze(1)
        return H, zg, z_l, self.style_k(z_k)

    # ---- M3 -------------------------------------------------------------------------
    def memory_and_mask(self, H, tmask, zg, stage, use_anchors=True):
        """Condition the transcript memory on anchor neighbourhoods, then hand the
        decoder (memory, mask). Anchors only ADD information here; nothing is clipped.
        """
        if stage in ("s2", "s3") and use_anchors:
            a = self.plan_layout(H, zg, tmask)
            nb = torch.cat([torch.nn.functional.pad(a[:, :-1], (0, 0, 1, 0)),
                            a,
                            torch.nn.functional.pad(a[:, 1:], (0, 0, 0, 1))], dim=-1)
            H = H + self.anchor_ctx(nb)
        if getattr(self, "use_slot_pos", False):
            C = H.shape[1]
            u = torch.arange(C, device=H.device, dtype=H.dtype) / max(C - 1, 1)
            fr = torch.stack([torch.sin(3.14159265 * u * (2 ** k)) for k in range(4)]
                             + [torch.cos(3.14159265 * u * (2 ** k)) for k in range(4)], -1)
            H = H + self.slot_pos(fr).unsqueeze(0)
            self._last_anchors = a          # reused by the loss: plan_layout is a
                                            # full transformer pass, do not run it twice
        return H, tmask

    def plan_layout(self, H: torch.Tensor, zg: torch.Tensor, text_mask: torch.Tensor):
        B, C, D = H.shape
        x = H + zg.unsqueeze(1)
        causal = torch.triu(torch.ones(C, C, device=H.device, dtype=torch.bool), diagonal=1)
        x = self.layout(x, mask=causal, src_key_padding_mask=(text_mask < 0.5))
        return self.layout_head(x)                     # [B,C,4] in canvas units

    def stroke_base(self, ci: torch.Tensor, j: torch.Tensor):
        """Per-stroke prototype token for (character, stroke-index-in-character)."""
        K = self.max_strokes_char
        if self.char_dict_head is None:
            self.char_dict_head = torch.nn.Linear(64, K * 8).to(ci.device)
        e = self.char_dict(ci)                       # [B,L,64]
        v = self.char_dict_head(e).view(*ci.shape, K, 8)
        idx = j.clamp(0, K - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 8)
        return v.gather(2, idx).squeeze(2)           # [B,L,8]

    def plan_counts(self, H: torch.Tensor, zg: torch.Tensor, text_mask: torch.Tensor):
        """Per-character stroke count logits, 1..13 (class 0 = one stroke)."""
        return self.count_head(H + zg.unsqueeze(1))

    # ---- M4 -------------------------------------------------------------------------
    def plan_joins(self, H: torch.Tensor, boxes: torch.Tensor, z_k: torch.Tensor,
                   zg: torch.Tensor, prior: torch.Tensor, text_mask: torch.Tensor):
        """Per boundary i (between char i and i+1). Returns dict of tensors [B,C-1,...]."""
        B, C, D = H.shape
        if C < 2:
            z = H.new_zeros(B, 0)
            return {"logit": z, "exit": z.unsqueeze(-1).expand(-1, -1, 2),
                    "entry": z.unsqueeze(-1).expand(-1, -1, 2),
                    "ctrl": z.unsqueeze(-1).expand(-1, -1, 4)}
        h_i, h_j = H[:, :-1], H[:, 1:]
        b_i, b_j = boxes[:, :-1], boxes[:, 1:]
        zk = z_k.unsqueeze(1).expand(-1, C - 1, -1)
        pr = prior[:, :C - 1].unsqueeze(-1)
        feat = torch.cat([h_i, h_j, b_i, b_j, pr, zk], dim=-1)
        out = self.join(feat)
        return {"logit": out[..., 0], "exit": out[..., 1:3],
                "entry": out[..., 3:5], "ctrl": out[..., 5:9]}


# --------------------------------------------------------------------------------------
# M8 — geometric refinement (inference only, closed form)
# --------------------------------------------------------------------------------------
def refine_boundaries(strokes: list[torch.Tensor], joins: list[int],
                      tau_deg: float = 35.0) -> list[torch.Tensor]:
    """C1-smooth every joined boundary by rotating the incoming control point.

    `strokes` are cubic Béziers [4,2] in canvas coordinates, in writing order. At a joined
    boundary the outgoing tangent of stroke k and the incoming tangent of stroke k+1 should
    be nearly collinear; where they differ by more than `tau_deg`, P1 of the outgoing stroke
    is rotated towards the continuation direction. Endpoints are never moved, so continuity
    (which is structural in the representation) is preserved exactly by this step.
    """
    out = [s.clone() for s in strokes]
    for i, join in enumerate(joins):
        if not join or i + 1 >= len(out):
            continue
        a, b = out[i], out[i + 1]
        t_out = a[3] - a[2]
        t_in = b[1] - b[0]
        na, nb = t_out.norm(), t_in.norm()
        if float(na) < EPS or float(nb) < EPS:
            continue
        cos = float((t_out / na * (t_in / nb)).sum().clamp(-1.0, 1.0))
        ang = math.degrees(math.acos(cos))
        if ang <= tau_deg:
            continue
        c, s = float(cos), float(math.sin(math.radians(ang)))
        rot = torch.tensor([[c, -s], [s, c]], dtype=a.dtype, device=a.device)
        a[1] = a[0] + rot @ (a[1] - a[0])
    return out
