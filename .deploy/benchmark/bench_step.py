"""Benchmark one CENT training step at production shape, on whatever GPU is present.

Measures wall-clock seconds and peak VRAM for a forward+backward pass with every loss active
(diffusion, join BCE, splat image L1, writer CE, per-character verifier CE). The point is to
replace guesswork in the GPU sizing table with a number measured on real hardware for the real
tensor shapes.

Usage
  python -m benchmark.bench_step --batch 16 --chars 32 --strokes 11 --canvas-w 1024 --gpu
  python -m benchmark.bench_step --batch 64 --chars 32 --strokes 17 --canvas-w 1024 --gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.models.diffusion import Diffusion, StrokeVerifier  # noqa: E402
from cent.models.groups import build_char_groups  # noqa: E402
from cent.models.line import (  # noqa: E402
    KIND_CHAR_STROKE, KIND_CONNECTOR, KIND_LAYOUT, LineDenoiser, LineStyleEncoder,
)
from cent.models.stroke_graph import render_strokes_splat  # noqa: E402

PAYLOAD = 8
N_CLASSES = 2703
N_WRITERS = 1019


def make_batch(B, n_chars, strokes, H, W, k_refs, device):
    """Synthetic batch with the real shapes.

    A line of `n_chars` characters with `strokes` strokes each costs
    n_chars*strokes + n_chars (layout) + (n_chars-1) (connectors) tokens.
    """
    T = n_chars * strokes + n_chars + (n_chars - 1)
    g = torch.Generator(device="cpu").manual_seed(0)
    tok = torch.rand(B, T, PAYLOAD, generator=g) * 0.6 + 0.2
    kind = torch.zeros(B, T, dtype=torch.long)
    slot = torch.zeros(B, T, dtype=torch.long)
    pos = 0
    for j in range(n_chars):
        for _ in range(strokes):
            kind[:, pos] = KIND_CHAR_STROKE
            slot[:, pos] = j
            pos += 1
        kind[:, pos] = KIND_LAYOUT
        slot[:, pos] = j
        pos += 1
        if j < n_chars - 1:
            kind[:, pos] = KIND_CONNECTOR
            slot[:, pos] = j
            pos += 1
    n_conn = n_chars - 1
    join = torch.zeros(B, n_chars)
    join[:, :n_conn] = (torch.rand(B, n_conn, generator=g) < 0.35).float()
    refs = torch.rand(B, k_refs, 1, H, W, generator=g)
    target = torch.rand(B, 1, H, W, generator=g)
    char_ids = torch.randint(0, N_CLASSES, (B, n_chars), generator=g)
    return {
        "tokens": tok.to(device), "kind": kind.to(device), "slot": slot.to(device),
        "mask": torch.ones(B, T, device=device),
        "join": join.to(device), "n_conn": torch.full((B,), n_conn, device=device, dtype=torch.long),
        "refs": refs.to(device), "ref_mask": torch.ones(B, k_refs, device=device),
        "target_img": target.to(device),
        "writer_idx": torch.randint(0, N_WRITERS, (B,), generator=g).to(device),
        "char_ids": char_ids.to(device), "cpos": torch.randint(0, N_CLASSES, (B, T), generator=g).to(device),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chars", type=int, default=32)
    ap.add_argument("--strokes", type=int, default=11)
    ap.add_argument("--canvas-h", type=int, default=64)
    ap.add_argument("--canvas-w", type=int, default=1024)
    ap.add_argument("--k-refs", type=int, default=5)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    device = "cuda" if (a.gpu and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(0)
    B, H, W = a.batch, a.canvas_h, a.canvas_w

    style = LineStyleEncoder(dim=a.dim, k_max=a.k_refs, n_types=5, n_heads=4).to(device)
    den = LineDenoiser(dim=a.dim, depth=a.depth, n_heads=4, n_classes=N_CLASSES,
                       max_tokens=1024).to(device)
    ver = StrokeVerifier(n_classes=N_CLASSES).to(device)
    whead = torch.nn.Linear(a.dim, N_WRITERS).to(device)
    diff = Diffusion(T=1000).to(device)
    params = (list(style.parameters()) + list(den.parameters())
              + list(ver.parameters()) + list(whead.parameters()))
    opt = torch.optim.AdamW(params, lr=2e-4)
    b = make_batch(B, a.chars, a.strokes, H, W, a.k_refs, device)
    T = b["tokens"].shape[1]

    def one_step():
        z_g, z_l, z_k, _ = style(b["refs"], b["ref_mask"])
        st = torch.cat([z_g.unsqueeze(1), z_l, z_k.unsqueeze(1)], dim=1)
        t = torch.randint(0, diff.T, (B,), device=device)
        noise = torch.randn_like(b["tokens"])
        xt = diff.q_sample(b["tokens"], t, noise)
        eps, jl = den(xt, t, b["kind"], b["cpos"], st, b["mask"], connector_mask=b["mask"].bool())
        loss = F.mse_loss(eps, noise)
        pred = torch.full_like(b["join"], -1e4)
        for i in range(B):
            n = int(b["n_conn"][i])
            idx = ((b["kind"][i] == KIND_CONNECTOR) & b["mask"][i].bool()).nonzero(as_tuple=True)[0]
            if len(idx):
                pred[i, :n] = jl[i][idx][:n]
        sel = pred > -1e3
        loss = loss + F.binary_cross_entropy_with_logits(pred[sel], b["join"][sel])
        ab = diff.alpha_bar.to(device)[t].view(-1, 1, 1)
        x0 = ((xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-5)).clamp(-2, 2)
        img = render_strokes_splat(x0, b["mask"], H, W)
        loss = loss + F.l1_loss(img, b["target_img"])
        loss = loss + 0.1 * F.cross_entropy(whead(z_g), b["writer_idx"])
        G, gm, labels = build_char_groups(x0, b["kind"], b["slot"], b["mask"], b["char_ids"])
        if G is not None:
            loss = loss + 0.1 * F.cross_entropy(ver(G, gm), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())

    for _ in range(a.warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(a.steps):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / a.steps

    rep = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "gpu": device == "cuda",
        "batch": B, "tokens": T, "chars": a.chars, "strokes_per_char": a.strokes,
        "canvas": [H, W], "k_refs": a.k_refs,
        "sec_per_step": round(dt, 4),
        "lines_per_hour": round(3600 / dt * B, 1),
        "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                         if device == "cuda" else None),
        "gpu_total_gb": (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                         if device == "cuda" else None),
    }
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
