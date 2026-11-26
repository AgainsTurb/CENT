"""InkVAE — sequence VAE over pen trajectories.

Input per point: (dx, dy, one-hot pen state)  -> 6 channels.
Latent: (T/8, z_dim), i.e. 8x temporal compression (kept deliberately coarse so
that a *boundary slot* can sit between two characters without being swamped).

Decoder head parameterises a 20-component GMM over (dx, dy) plus a 4-way pen
state, so the reconstruction loss is the standard MDN negative log-likelihood.

Two regularisers shape the latent (both borrowed in spirit from DiffInk, but
re-implemented here):
  * CTC over the character sequence  -> the latent must stay char-decodable
  * writer classification            -> the latent must carry style
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



class Residual(nn.Module):
    """DiffInk's residual block: GN -> GELU -> conv3 -> GN -> GELU -> conv1 + skip."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [Residual(in_channels, hidden_channels) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class InkVAE(nn.Module):
    def __init__(self, *, in_dim: int = 6, d: int = 128, z_dim: int = 64,
                 stride: int = 8, n_mix: int = 20, n_chars: int = 500,
                 n_writers: int = 32, n_pen: int = 4) -> None:
        super().__init__()
        self.stride = stride
        self.z_dim = z_dim
        self.n_mix = n_mix
        self.n_pen = n_pen

        n_down = 3  # 2^3 = 8
        assert 2 ** n_down == stride

        self.enc_in = nn.Conv1d(in_dim, 128, kernel_size=3, stride=2, padding=1)
        self.enc_rs1 = ResidualStack(128, 64, 4)
        self.enc_down2 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_rs2 = ResidualStack(256, 128, 4)
        self.enc_down3 = nn.Conv1d(256, 384, kernel_size=3, stride=2, padding=1)
        self.enc_rs3 = ResidualStack(384, 192, 4)
        self.enc_out = nn.Conv1d(384, 2 * z_dim, kernel_size=3, padding=1)

        dd = 128
        self.dd = dd
        self.dec_in = nn.Conv1d(z_dim, 384, kernel_size=3, padding=1)
        self.dec_rs1 = ResidualStack(384, 192, 4)
        self.dec_up1 = nn.ConvTranspose1d(384, 256, kernel_size=4, stride=2, padding=1)
        self.dec_rs2 = ResidualStack(256, 128, 4)
        self.dec_up2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_rs3 = ResidualStack(128, 64, 4)
        self.dec_up3 = nn.ConvTranspose1d(128, dd, kernel_size=4, stride=2, padding=1)

        head_dim = 6 * n_mix + n_pen
        self.head_dim = head_dim

        # positional conditioning at the latent resolution
        self.pos_dim = 32
        self.pos_mlp = nn.Sequential(nn.Linear(self.pos_dim, dd), nn.GELU(),
                                     nn.Linear(dd, z_dim))
        self.skip_drop = 0.0

        # global context before the head (DiffInk: 4 layers, d=256, FFN 1024)
        self.post = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dd, nhead=4, dim_feedforward=4 * dd,
                                       batch_first=True, norm_first=True, dropout=0.0),
            num_layers=4)
        self.post_proj = nn.Linear(dd, head_dim)

        # Positional conditioning.
        # Without it the decoder minimises the GMM likelihood by emitting the
        # sequence-mean delta at every step, i.e. a straight line: it has no way
        # to know which part of the line it is drawing. Measured on a single
        # overfit line: straightness 1.1 (straight) vs 6.5 for real handwriting.
        self.pos_dim = 32
        self.pos_mlp = nn.Sequential(nn.Linear(self.pos_dim, d), nn.GELU(),
                                     nn.Linear(d, z_dim))   # add before the first conv
        self.skip_drop = 0.3

        # latent auxiliaries
        self.ctc_head = nn.Linear(z_dim, n_chars)
        self.writer_head = nn.Linear(z_dim, n_writers)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, mask: torch.Tensor, *, return_skips: bool = False):
        """x: (B, T, 6) -> mu, logvar, latent_mask[, skips]."""
        B, T, C = x.shape
        pad = (-T) % self.stride
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        h = x.transpose(1, 2)
        h = self.enc_rs1(self.enc_in(h))
        h = self.enc_rs2(self.enc_down2(h))
        h = self.enc_rs3(self.enc_down3(h))
        h = self.enc_out(h)                        # (B, 2z, T_lat)
        mu, logvar = h.chunk(2, dim=1)
        mu = mu.transpose(1, 2)
        logvar = logvar.transpose(1, 2).clamp(-8.0, 8.0)
        T_lat = mu.shape[1]
        idx = torch.arange(T_lat, device=mask.device).unsqueeze(0)
        lm = ((idx * self.stride) < mask.sum(dim=1, keepdim=True)).float()
        if return_skips:
            return mu, logvar, lm, []
        return mu, logvar, lm

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor,
                       sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(mu)

    @staticmethod
    def _sin_pe(n: int, d: int, device, dtype) -> torch.Tensor:
        """Sinusoidal position encoding (1, n, d)."""
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
        div = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * i / d)
        pe = torch.zeros(n, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
        return pe.unsqueeze(0).to(dtype)

    def _pos(self, n: int, device, dtype) -> torch.Tensor:
        """Sinusoidal position features over the latent frames -> (1, d, n)."""
        u = torch.linspace(0.0, 1.0, n, device=device).unsqueeze(-1)
        freqs = torch.exp(torch.linspace(0.0, 6.9, self.pos_dim // 2, device=device))
        ang = u * freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        pe = self.pos_mlp(feats.to(dtype))
        return pe.transpose(0, 1).unsqueeze(0)

    def decode(self, z: torch.Tensor, skips: list | None = None):
        """z: (B, T_lat, z_dim) -> gmm (B, T, M, 6), pen logits (B, T, 4).

        Structure follows DiffInk: ResidualStack -> deconv, three times, then a
        Transformer over the whole sequence before the GMM head. Conv-only
        decoders reconstruct handwriting as a smooth average.
        """
        h = z.transpose(1, 2)
        h = h + self._pos(h.shape[-1], z.device, h.dtype)
        h = self.dec_in(h)
        h = self.dec_up1(self.dec_rs1(h))
        h = self.dec_up2(self.dec_rs2(h))
        h = self.dec_up3(self.dec_rs3(h))
        h = h.transpose(1, 2)
        h = h + self._sin_pe(h.shape[1], h.shape[2], h.device, h.dtype)
        h = self.post(h)
        h = self.post_proj(h)
        M = self.n_mix
        gmm = h[..., : 6 * M].view(*h.shape[:2], M, 6)
        pen = h[..., 6 * M:]
        return gmm, pen

    def forward(self, x: torch.Tensor, pen: torch.Tensor, mask: torch.Tensor,
                *, sample: bool = True, n_chars: int | None = None):
        mu, logvar, lm, skips = self.encode(x, mask, return_skips=True)
        z = self.reparameterise(mu, logvar, sample=sample)
        gmm, pen_logits = self.decode(z, skips=skips)

        # the decoder reconstructs ceil(T/8)*8 points; crop back to T
        T = mask.shape[1]
        if gmm.shape[1] > T:
            gmm = gmm[:, :T]
            pen_logits = pen_logits[:, :T]
        elif gmm.shape[1] < T:
            pad = T - gmm.shape[1]
            gmm = F.pad(gmm, (0, 0, 0, 0, 0, pad))
            pen_logits = F.pad(pen_logits, (0, 0, 0, pad))

        B, T_lat, _ = z.shape
        ctc_logits = self.ctc_head(z)                                  # (B, T_lat, C)
        pooled = (z * lm.unsqueeze(-1)).sum(1) / lm.sum(1).clamp_min(1.0).unsqueeze(-1)
        writer_logits = self.writer_head(pooled)                       # (B, n_writers)

        return {
            "mu": mu, "logvar": logvar, "z": z, "latent_mask": lm,
            "gmm": gmm, "pen_logits": pen_logits,
            "ctc_logits": ctc_logits, "writer_logits": writer_logits,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def gmm_nll(gmm: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """gmm: (B,T,M,6) [pi_logit, mux, muy, sx_logit, sy_logit, rho], target: (B,T,2)."""
        pi_logit, mux, muy, sx_logit, sy_logit, rho = gmm.unbind(-1)
        pi = F.log_softmax(pi_logit, dim=-1)
        sx = torch.exp(sx_logit).clamp_min(1e-4)
        sy = torch.exp(sy_logit).clamp_min(1e-4)
        rho = torch.tanh(rho)
        dx = target[..., 0:1]
        dy = target[..., 1:2]
        zx = (dx - mux) / sx
        zy = (dy - muy) / sy
        om = 1 - rho ** 2
        z = zx ** 2 + zy ** 2 - 2 * rho * zx * zy
        log_norm = -torch.log(2 * torch.pi * sx * sy * om.sqrt().clamp_min(1e-6))
        log_prob = log_norm - z / (2 * om.clamp_min(1e-6))
        return -(torch.logsumexp(pi + log_prob, dim=-1))
