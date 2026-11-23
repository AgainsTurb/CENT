"""Stroke-graph representation: polyline <-> fixed-length tensors, and a
differentiable Bezier renderer.

Representation used by the model
--------------------------------
A character has N_c strokes (fixed by the lexicon). Each stroke is approximated
by ONE cubic Bezier, so a stroke is 8 scalars (4 control points x (x, y)). A
sample is therefore a tensor [N_MAX, 8] plus a validity mask [N_MAX].

The single-cubic fit is an approximation chosen for a tractable, fixed-size
sequence: real strokes need 2-4 cubics. `fit_stroke` uses the chord-length
endpoints and quarter-point positions, which is exact for straight and gently
curved strokes and visibly coarse for hooks. This is recorded as a limitation,
not hidden: a stroke that bends sharply will render softer than the writer drew
it, and the ablation A4 (renderer) is where that cost shows up.

Renderer
--------
`render_strokes` maps control points to a soft ink map differentiably. Every
pixel's coverage is computed from its distance to the sampled Bezier polyline
through a sigmoid, so gradients flow to the control points. Implemented in
chunks to keep peak memory bounded on long sequences.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MAX = 40          # max strokes per character (GB2312 max is 30; margin for OOV)
COORDS = 8          # 4 control points * (x, y)
N_SEG = 4           # polyline samples per cubic used for rendering


def fit_stroke(points: np.ndarray) -> np.ndarray:
    """Approximate one polyline (T,2) by one cubic Bezier -> (4,2).

    Endpoints are exact. P1/P2 are placed on the straight line between them at
    the positions that the quarter- and three-quarter-length sample points
    would occupy, which reproduces straight strokes exactly and mildly
    under-fits curvature.
    """
    p = np.asarray(points, np.float32).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros((4, 2), np.float32)
    if len(p) == 1:
        return np.repeat(p, 4, axis=0)
    p0, p3 = p[0], p[-1]
    # sample at 1/4 and 3/4 of arc length
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-8:
        return np.repeat(p0[None], 4, axis=0)
    s /= s[-1]
    q1 = np.array([np.interp(0.25, s, p[:, 0]), np.interp(0.25, s, p[:, 1])], np.float32)
    q2 = np.array([np.interp(0.75, s, p[:, 0]), np.interp(0.75, s, p[:, 1])], np.float32)
    # cubic at t=1/4 and 3/4 with control points on the chord: P1 = P0 + a, P2 = P3 - b
    p1 = p0 + (q1 - p0) * 2.0
    p2 = p3 + (q2 - p3) * 2.0
    return np.stack([p0, p1, p2, p3]).astype(np.float32)


def strokes_to_tensor(strokes: list[np.ndarray], n_max: int = N_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    """List of polylines -> (tokens [n_max, 8], mask [n_max])."""
    tok = np.zeros((n_max, COORDS), np.float32)
    msk = np.zeros((n_max,), np.float32)
    for i, s in enumerate(strokes[:n_max]):
        tok[i] = fit_stroke(s).reshape(-1)
        msk[i] = 1.0
    return torch.from_numpy(tok), torch.from_numpy(msk)


def tensor_to_strokes(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> list[np.ndarray]:
    """(N,8) tokens -> list of (4,2) control-point arrays."""
    t = tokens.detach().cpu().numpy()
    out = []
    for i in range(t.shape[0]):
        if mask is not None and float(mask[i]) <= 0:
            break
        out.append(t[i].reshape(4, 2))
    return out


def bezier_sample(ctrl: torch.Tensor, n: int = N_SEG) -> torch.Tensor:
    """Sample a cubic Bezier. ctrl [B,N,4,2] -> points [B,N,n,2]."""
    t = torch.linspace(0.0, 1.0, n, device=ctrl.device, dtype=ctrl.dtype).view(1, 1, n, 1)
    p0, p1, p2, p3 = (ctrl[:, :, i, :].unsqueeze(2) for i in range(4))
    u = 1.0 - t
    return u ** 3 * p0 + 3 * u ** 2 * t * p1 + 3 * u * t ** 2 * p2 + t ** 3 * p3


def render_strokes(ctrl: torch.Tensor, mask: torch.Tensor | None = None, size: int = 64,
                   width: float = 0.035, tau: float = 0.01, chunk: int = 8,
                   width_size: int | None = None) -> torch.Tensor:
    """Differentiable soft rasteriser.

    Args:
        ctrl: [B, N, 4, 2] control points in [0,1].
        mask: [B, N] stroke validity.
        size: output resolution.
        width: ink half-width in normalised units (~2-3 px at 64).
        tau: sigmoid temperature; smaller = crisper, noisier gradients.
    Returns:
        [B, 1, size, size] coverage in [0,1] (1 = ink).
    """
    B, N = ctrl.shape[0], ctrl.shape[1]
    if mask is None:
        mask = torch.ones(B, N, device=ctrl.device, dtype=ctrl.dtype)
    W = int(width_size) if width_size else size
    ys = torch.linspace(0, 1, size, device=ctrl.device, dtype=ctrl.dtype)
    xs = torch.linspace(0, 1, W, device=ctrl.device, dtype=ctrl.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")          # [H,W]

    out = ctrl.new_zeros(B, 1, size, W)
    for b in torch.split(torch.arange(B, device=ctrl.device), chunk):
        c = ctrl[b]                                          # [b,N,4,2]
        m = mask[b].view(-1, N, 1, 1)                        # [b,N,1,1]
        m5 = m.unsqueeze(-1)                                 # [b,N,1,1,1] for the 5-D distance map
        pts = bezier_sample(c, N_SEG + 1)                    # [b,N,n,2]
        # segment-level distances to every pixel.
        # a/e are [b,N,n-1,1,1,2] so they broadcast against the pixel grid [1,1,1,S,S,2].
        a = pts[:, :, :-1, :].unsqueeze(3).unsqueeze(3)
        e = pts[:, :, 1:, :].unsqueeze(3).unsqueeze(3)
        P = torch.stack([gx, gy], dim=-1).view(1, 1, 1, size, W, 2)
        ap = P - a
        ae = e - a
        denom = (ae ** 2).sum(-1).clamp_min(1e-9)
        tt = (ap * ae).sum(-1) / denom
        tt = tt.clamp(0, 1)
        proj = a + tt.unsqueeze(-1) * ae
        d = torch.linalg.norm(P - proj, dim=-1)              # [b,N,n-1,S,S]
        d = d * m5
        # a stroke covers a pixel if ANY segment is close: take the min over segments
        dmin = d.amin(dim=2)                                 # [b,N,S,S]
        cov = torch.sigmoid((width - dmin) / tau)
        cov = cov * m
        img = 1.0 - torch.prod(1.0 - cov.clamp(0, 1), dim=1)  # union of strokes
        out[b] = img.unsqueeze(1)
    return out


def decode_trajectory(ctrl: torch.Tensor, mask: torch.Tensor | None = None,
                      n_points: int = 32, size: int = 64) -> list[dict]:
    """Decode a stroke graph to an online trajectory.

    This is the other half of the dual decode: same control points, sampled at
    uniform t, scaled to pixel coordinates, with a pen-up event after every
    stroke except the last. The renderer and this function therefore agree by
    construction -- they read the same geometry, they do not learn to agree.

    Returns a list of dicts: {"x": [n], "y": [n], "pen_up": bool, "t": [n]}, one
    per stroke, plus "order" = stroke index (the drawing order is the graph's).
    """
    if ctrl.dim() == 3 and ctrl.shape[-1] == COORDS:
        ctrl = ctrl.reshape(ctrl.shape[0], ctrl.shape[1], 4, 2)
    B, N = ctrl.shape[:2]
    pts = bezier_sample(ctrl, n_points)                    # [B,N,n,2]
    if mask is None:
        mask = torch.ones(B, N, device=ctrl.device, dtype=ctrl.dtype)
    out: list[dict] = []
    for b in range(B):
        strokes = []
        valid = [i for i in range(N) if float(mask[b, i]) > 0.5]
        for rank, i in enumerate(valid):
            p = pts[b, i].detach().cpu().numpy() * float(size)
            strokes.append({
                "order": rank,
                "x": p[:, 0].round(2).tolist(),
                "y": p[:, 1].round(2).tolist(),
                "t": np.linspace(0, 1, n_points).round(4).tolist(),
                "pen_up": rank != len(valid) - 1,
            })
        out.append({"strokes": strokes, "n_strokes": len(strokes)})
    return out


def render_strokes_splat(ctrl: torch.Tensor, mask: torch.Tensor | None = None,
                         size: int = 64, width_size: int = 1024, n_seg: int = 8,
                         sigma: float = 1.0) -> torch.Tensor:
    """Splatting rasteriser: O(points x kernel), not O(strokes x pixels).

    The distance renderer materialises a [B, N, n, H, W] tensor to compare every stroke
    segment against every pixel. At character scale (40 strokes, 64x64) that is ~10 MB; at
    line scale (470 strokes, 64x1024) it is ~316 GB, so the line model cannot use it.

    This rasteriser evaluates each Bezier at n_seg+1 points, maps them to pixel coordinates
    and bilinearly scatters a small Gaussian footprint into the canvas. Cost is
    B*N*(n_seg+1)*k for a k-tap kernel --- a few million entries instead of 10^10 --- and it
    stays differentiable through the control points, because the scatter weights are
    functions of the (continuous) point positions.

    Args:
        ctrl: [B, N, 4, 2] or [B, N, 8] control points in [0,1].
        mask: [B, N] stroke validity.
        size, width_size: canvas height and width in pixels.
        n_seg: Bezier subdivisions per stroke.
        sigma: footprint falloff in pixels (1.0 ~ a 3x3 splat).
    Returns:
        [B, 1, size, width_size] ink coverage in [0,1].
    """
    if ctrl.dim() == 3:
        ctrl = ctrl.view(ctrl.shape[0], ctrl.shape[1], 4, 2)
    B, N = ctrl.shape[0], ctrl.shape[1]
    H, W = size, width_size
    if mask is None:
        mask = torch.ones(B, N, device=ctrl.device, dtype=ctrl.dtype)

    pts = bezier_sample(ctrl, n_seg + 1)                      # [B,N,n,2] in [0,1]
    xs = (pts[..., 0].clamp(0, 1) * (W - 1)).reshape(B, -1)   # [B, N*n]
    ys = (pts[..., 1].clamp(0, 1) * (H - 1)).reshape(B, -1)
    w = mask.repeat_interleave(n_seg + 1, dim=1).reshape(B, -1)   # [B, N*n]

    canvas = ctrl.new_zeros(B, 1, H * W)
    x0 = torch.floor(xs)
    y0 = torch.floor(ys)
    fx = xs - x0
    fy = ys - y0
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        wx = fx if dx else (1 - fx)
        wy = fy if dy else (1 - fy)
        wt = wx * wy * w
        xi = (x0 + dx).clamp(0, W - 1).long()
        yi = (y0 + dy).clamp(0, H - 1).long()
        idx = (yi * W + xi).view(B, 1, -1)
        canvas.scatter_add_(2, idx, wt.view(B, 1, -1))

    if sigma and sigma > 0 and n_seg > 1:
        # a light blur turns 1-px polylines into strokes of realistic thickness
        k = max(3, int(2 * round(sigma) + 1))
        pad = k // 2
        c = canvas.view(B, 1, H, W)
        ker = torch.exp(-0.5 * (torch.arange(k, device=ctrl.device, dtype=ctrl.dtype) - pad) ** 2
                        / max(sigma, 1e-3) ** 2)
        ker = (ker / ker.sum()).view(1, 1, 1, k)
        c = F.conv2d(F.pad(c, (pad, pad, 0, 0), mode="replicate"), ker)
        c = F.conv2d(F.pad(c, (0, 0, pad, pad), mode="replicate"), ker.view(1, 1, k, 1))
        canvas = c.view(B, 1, H * W)

    return canvas.clamp(0, 1).view(B, 1, H, W)


class BezierRenderer(nn.Module):
    """Thin module wrapper so the renderer can sit in an nn.Sequential / be
    registered, and so its hyper-parameters live in a config."""

    def __init__(self, size: int = 64, width: float = 0.035, tau: float = 0.01,
                 width_size: int | None = None):
        super().__init__()
        self.size, self.width, self.tau = size, width, tau
        self.width_size = width_size

    def forward(self, ctrl: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Accept either flat tokens [B,N,8] (what the denoiser emits) or
        # explicit control points [B,N,4,2].
        if ctrl.dim() == 3 and ctrl.shape[-1] == COORDS:
            ctrl = ctrl.reshape(ctrl.shape[0], ctrl.shape[1], 4, 2)
        return render_strokes(ctrl, mask, self.size, self.width, self.tau,
                              width_size=self.width_size)
