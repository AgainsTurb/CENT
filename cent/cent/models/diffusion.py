"""Minimal DDPM schedule + DDIM sampler (used by BoundaryDiT)."""

from __future__ import annotations

import torch


class Diffusion:
    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, device: str | torch.device = "cpu") -> None:
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]]
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.alphas_cumprod[t].view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, *, steps: int = 20, device="cpu",
                    eta: float = 0.0, **cond):
        """``model_fn(x_t, t_scalar_tensor, **cond) -> x0_pred``."""
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.n_steps - 1, 0, steps, device=device).long()
        for i, t in enumerate(ts):
            t_batch = t.expand(shape[0])
            x0 = model_fn(x, t_batch, **cond)
            t_next = ts[i + 1] if i + 1 < len(ts) else torch.zeros_like(t)
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[t_next]
            x0 = x0.clamp(-4.0, 4.0)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp_min(1e-8)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x
