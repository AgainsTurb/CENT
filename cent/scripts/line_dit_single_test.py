"""Does LineDiT work at all? Overfit one line, then sample from noise.

With no layout, binding text to ink is done purely by cross-attention, so this
test asks the hard question first: can the model reproduce a line it has seen,
with character sequence as its only text signal?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cent.model import CENTModel
from cent.models.line_dit import LineDiT
from cent.pipeline.common import gmm_to_points, points_to_strokes
from cent.viz import figure_panels, render_strokes
from train_dit_v2 import batch_from, load_cache
from train_line_dit import cursiveness, ddim

D, DEPTH, HEADS = 384, 6, 8
STEPS = 1500


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load("runs_box/vae_ctc60k.pt", map_location="cpu", weights_only=False)
    import hashlib
    h = hashlib.sha1()
    for k in sorted(ck["model"].keys()):
        v = ck["model"][k]
        if torch.is_tensor(v):
            h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    expect = {"vae_hash": h.hexdigest()[:16], "stride": 8, "z_dim": 64,
              "dit_d": D, "text_vocab": len(ck.get("vocab") or []) + 1}

    model = CENTModel(n_chars=max(64, len(ck.get("vocab") or []) + 1),
                      n_writers=int(ck.get("n_writers") or 8), d=D, z_dim=64,
                      vae_d=128, stride=8, dit_depth=DEPTH, dit_heads=HEADS,
                      n_mix=20, n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)
    model.dit = LineDiT(z_dim=64, d=D, depth=DEPTH, n_heads=HEADS).to(dev)
    print(f"LineDiT params {sum(p.numel() for p in model.dit.parameters())/1e6:.2f} M")

    tr = load_cache("train", expect)
    alphas = model.alphas_cumprod

    # pick one mid-length line
    idx = 200                      # 209 frames, 14.8% pen lifts: has real structure
    sel = torch.tensor([idx])
    z0, char, cmask, _l, _m, style, _t = batch_from(tr, sel, dev)
    tlat = int(_t[0])
    with torch.no_grad():
        ctx = model.text(char, cmask)   # text encoder is frozen in this test
    print(f"line {idx}: T_lat={z0.shape[1]}  chars={int(cmask.sum())}")

    params = list(model.dit.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)
    model.train()
    g = torch.Generator(device=dev).manual_seed(0)
    for step in range(1, STEPS + 1):
        t = torch.randint(0, model.n_steps, (1,), device=dev, generator=g)
        aa = alphas[t].view(-1, 1, 1)
        z_t = aa.sqrt() * z0 + (1 - aa).sqrt() * torch.randn_like(z0)
        x0 = model.dit(z_t, t, char_ctx=ctx, char_mask=cmask, style=style)["x0"]
        loss = (x0 - z0).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 250 == 0:
            print(f"  step {step:4d} x0_l1={float(loss):.4f}")

    model.eval()
    with torch.no_grad():
        tx = torch.zeros(1, dtype=torch.long, device=dev)
        d0 = model.dit(z0, tx, char_ctx=ctx, char_mask=cmask, style=style)["x0"]
        zs = ddim(model.dit, z0.shape, ctx, cmask, style, alphas, dev, steps=40)
        outs = {"DiT@t=0": d0, "DDIM from noise": zs}
        res = {}
        for k, z in outs.items():
            gmm, penl = model.vae.decode(z[:, :tlat])
            pts = torch.cumsum(gmm_to_points(gmm), dim=1)[0].cpu().numpy()
            pen = penl[0, : len(pts)].argmax(-1).cpu().numpy()
            res[k] = (pts, pen)
        print("cursiveness (alpha=0):",
              {k: round(v, 3) for k, v in cursiveness(model.vae, zs, 0.0).items()})

    # target points, reconstructed from the VAE decode of the true latent
    with torch.no_grad():
        gt, gpl = model.vae.decode(z0[:, :tlat])
        tpts = torch.cumsum(gmm_to_points(gt), dim=1)[0].cpu().numpy()
        tpen = gpl[0, : len(tpts)].argmax(-1).cpu().numpy()

    def epe(p):
        n = min(len(p), len(tpts))
        return float(np.abs(p[:n] - tpts[:n]).mean())

    print(f"EPE  DiT@t=0        : {epe(res['DiT@t=0'][0]):.3f}")
    print(f"EPE  DDIM from noise: {epe(res['DDIM from noise'][0]):.3f}")

    figure_panels([
        ("TARGET (VAE decode of true latent)", render_strokes(points_to_strokes(tpts, tpen), height=110)),
        (f"LineDiT @t=0   EPE={epe(res['DiT@t=0'][0]):.2f}",
         render_strokes(points_to_strokes(*res["DiT@t=0"]), height=110, colour=(30, 90, 160))),
        (f"LineDiT DDIM from noise   EPE={epe(res['DDIM from noise'][0]):.2f}",
         render_strokes(points_to_strokes(*res["DDIM from noise"]), height=110, colour=(40, 140, 70))),
    ], "runs_box/line_dit_single.png",
        title=f"LineDiT: no layout, text via cross-attention only ({STEPS} steps, 1 line)")
    print("figure -> runs_box/line_dit_single.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
