"""CENTModel — assembles InkVAE + TextEncoder + BoundaryDiT + Diffusion.

Two entry points matter:
  * ``forward(batch)``  -> training losses (never called in this session)
  * ``generate(...)``   -> sampling with a *cursive control* knob

The control knob is the paper's contribution (ii) in its operational form:
``control=0`` reproduces the writer's own join rate, ``control=1`` forces every
reachable boundary to join, ``control=0`` forces none to. Because boundary
slots are committed back into the DiT during denoising, the tails of the
adjoining characters are regenerated to *fit* the decision rather than patched
afterwards.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .losses import dit_loss, vae_losses
from .models.boundary_dit import BoundaryDiT
from .models.inkvae import InkVAE
from .models.text_encoder import TextEncoder
from .data.dataset import frame_char_ids


class CENTModel(nn.Module):
    def __init__(self, *, n_chars: int = 500, n_writers: int = 32, d: int = 128,
                 z_dim: int = 64, vae_d: int = 128, stride: int = 8,
                 dit_depth: int = 3, dit_heads: int = 4, n_mix: int = 20,
                 n_steps: int = 1000, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.stride = stride
        self.z_dim = z_dim
        self.device_name = str(device)

        self.vae = InkVAE(d=vae_d, z_dim=z_dim, stride=stride, n_mix=n_mix,
                          n_chars=n_chars, n_writers=n_writers)
        self.text = TextEncoder(n_chars=n_chars, d=d)
        self.dit = BoundaryDiT(z_dim=z_dim, d=d, depth=dit_depth, n_heads=dit_heads,
                               n_chars=n_chars)
        self.style_proj = nn.Linear(z_dim, d)

        # diffusion constants are registered as plain tensors (moved with .to())
        betas = torch.linspace(1e-4, 2e-2, n_steps)
        ac = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", ac)
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    def vae_input(self, batch: dict) -> torch.Tensor:
        pen_oh = torch.nn.functional.one_hot(batch["pen"], num_classes=4).float()
        return torch.cat([batch["dxy"], pen_oh], dim=-1)

    def style_vector(self, batch: dict) -> torch.Tensor:
        """Pooled latent of the *reference* sample (here: the same sample)."""
        out = self.vae(self.vae_input(batch), batch["pen"], batch["point_mask"])
        z = out["z"]
        lm = out["latent_mask"].unsqueeze(-1)
        pooled = (z * lm).sum(1) / lm.sum(1).clamp_min(1.0)
        return self.style_proj(pooled)
    def writer_index(self, writers: list[str], table: dict[str, int]) -> torch.Tensor:
        return torch.tensor([table.get(w, 0) for w in writers], dtype=torch.long)

    # ------------------------------------------------------------------
    def forward(self, batch: dict, *, writer_table: dict[str, int] | None = None) -> dict:
        """Full training loss (VAE + DiT). Not executed in this session."""
        x = self.vae_input(batch)
        v = self.vae(x, batch["pen"], batch["point_mask"])
        widx = self.writer_index(batch["writers"], writer_table or {})

        lv = vae_losses(v, batch["dxy"], batch["pen"], batch["char_seq"],
                        batch["char_mask"], widx, batch["point_mask"])

        # latent layout
        T_lat = v["z"].shape[1]
        fchar, fmask = frame_char_ids(batch["char_ids"], self.stride,
                                      batch["point_mask"], T_lat)

        style = self.style_vector(batch)
        char_ctx = self.text(batch["char_seq"], batch["char_mask"])

        z0 = v["z"].detach()
        t = torch.randint(0, self.n_steps, (z0.shape[0],), device=z0.device)
        noise = torch.randn_like(z0)
        a = self.alphas_cumprod[t].view(-1, 1, 1)
        z_t = a.sqrt() * z0 + (1 - a).sqrt() * noise

        out = self.dit(z_t, t, frame_char=fchar, frame_mask=fmask,
                       char_ctx=char_ctx, style=style)
        ld = dit_loss(out, z0, batch["bnd_labels"], batch["bnd_mask"], fmask)

        total = lv["total"] + ld["total"]
        return {"total": total, **{f"vae_{k}": val for k, val in lv.items() if k != "total"},
                **{f"dit_{k}": val for k, val in ld.items() if k != "total"},
                "z": v["z"], "frame_char": fchar, "frame_mask": fmask,
                "char_ctx": char_ctx, "style": style, "dit_out": out}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, char_seq: torch.Tensor, char_mask: torch.Tensor,
                 style: torch.Tensor, *, control: float | None = None,
                 steps: int = 12, n_chars_per_frame: int = 3,
                 force_labels: torch.Tensor | None = None):
        """Sample a line.

        control: None -> model prior; 0.0 -> force lift; 1.0 -> force join;
                 anything in between -> bias the logits by logit(control).
        force_labels: optional (B, T_lat-1) hard labels override.
        """
        B, L = char_seq.shape
        n_frames = int(char_mask.sum(1).max().item()) * n_chars_per_frame
        T_lat = max(8, n_frames)
        z = torch.randn(B, T_lat, self.z_dim, device=char_seq.device)

        # provisional char layout: split frames evenly across characters
        fchar = torch.full((B, T_lat), -1, dtype=torch.long, device=char_seq.device)
        fmask = torch.zeros(B, T_lat, dtype=torch.bool, device=char_seq.device)
        for b in range(B):
            n = int(char_mask[b].sum().item())
            if n == 0:
                continue
            per = max(1, T_lat // n)
            for c in range(n):
                lo = c * per
                hi = T_lat if c == n - 1 else min(T_lat, (c + 1) * per)
                fchar[b, lo:hi] = c
                fmask[b, lo:hi] = True

        char_ctx = self.text(char_seq, char_mask)
        commit = None
        logits = None
        ts = torch.linspace(self.n_steps - 1, 0, steps).long()
        for i, td in enumerate(ts):
            t = td.expand(B).to(char_seq.device)
            out = self.dit(z, t, frame_char=fchar, frame_mask=fmask,
                           char_ctx=char_ctx, style=style, bnd_commit=commit)
            x0 = out["x0"].clamp(-4, 4)
            logits = out["bnd_logits"]
            a_t = self.alphas_cumprod[td]
            eps = (z - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp_min(1e-8)
            t_next = ts[i + 1] if i + 1 < len(ts) else torch.zeros_like(td)
            a_n = self.alphas_cumprod[t_next]
            z = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps

            # commit a decision for the remaining steps
            if i == 0:
                if force_labels is not None:
                    commit = force_labels.float()
                elif control is not None:
                    c = min(max(control, 1e-4), 1 - 1e-4)
                    bias = math.log(c / (1 - c))
                    lbl = ((logits + bias) > 0).float()
                    lbl = lbl * out["bnd_valid"].float()
                    commit = torch.where(out["bnd_valid"], lbl, torch.full_like(lbl, -1.0))
                else:
                    commit = torch.where(out["bnd_valid"], (logits > 0).float(),
                                         torch.full_like(logits, -1.0))

        return {
            "z": z, "bnd_logits": logits, "bnd_valid": out["bnd_valid"],
            "committed": commit, "x0": out["x0"],
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        """Latent frames -> points, pen states, strokes."""
        gmm, pen_logits = self.vae.decode(z)
        pen = pen_logits.argmax(-1)
        pi = torch.softmax(gmm[..., 0], dim=-1)
        best = pi.argmax(-1)
        b = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        mu = torch.gather(gmm[..., 1:3], 2, b).squeeze(2)         # (B,T,2)
        pts = mu.cumsum(1)
        return {"points": pts, "pen": pen, "dxy": mu}
