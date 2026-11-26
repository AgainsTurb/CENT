"""Gate C: does text bind to ink without layout, on UNSEEN text?

Train LineDiT on 12 lines, then generate a 13th line whose text was never in
training. No frame_char, no layout head, no boundary head: the character
sequence reaches the model only by cross-attention.
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
N_TRAIN, STEPS = 12, 3000


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load("runs_box/vae_ctc60k.pt", map_location="cpu", weights_only=False)
    chars = list(ck.get("vocab") or [])
    model = CENTModel(n_chars=max(64, len(chars) + 1),
                      n_writers=int(ck.get("n_writers") or 8), d=D, z_dim=64,
                      vae_d=128, stride=8, dit_depth=DEPTH, dit_heads=HEADS,
                      n_mix=20, n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)
    model.dit = LineDiT(z_dim=64, d=D, depth=DEPTH, n_heads=HEADS).to(dev)
    print(f"LineDiT {sum(p.numel() for p in model.dit.parameters())/1e6:.2f} M")

    cache = torch.load("stage4_cache/train.pt", map_location="cpu", weights_only=False)
    tlat = cache["tlat"]
    # 12 training lines of moderate length + 1 held-out line
    pool = [i for i in range(len(tlat)) if 110 <= int(tlat[i]) <= 200]
    train_idx = pool[:N_TRAIN]
    hold = pool[N_TRAIN + 40]
    print(f"train lines {train_idx}\nheld-out line {hold} (tlat={int(tlat[hold])})")

    def prep(i):
        z, ch, cm, _l, _m, st, tl = batch_from(cache, torch.tensor([i]), dev)
        with torch.no_grad():
            ctx = model.text(ch, cm)
        return z, ctx, cm, st, int(tl[0])

    train = [prep(i) for i in train_idx]
    zh, ctxh, cmh, sth, tlh = prep(hold)
    # style must not leak the target line: borrow a training line's style
    style_h = train[0][3]

    params = list(model.dit.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4)
    model.train()
    g = torch.Generator(device=dev).manual_seed(0)
    alphas = model.alphas_cumprod
    for step in range(1, STEPS + 1):
        z, ctx, cm, st, tl = train[step % len(train)]
        t = torch.randint(0, model.n_steps, (1,), device=dev, generator=g)
        aa = alphas[t].view(-1, 1, 1)
        z_t = aa.sqrt() * z + (1 - aa).sqrt() * torch.randn_like(z)
        x0 = model.dit(z_t, t, char_ctx=ctx, char_mask=cm, style=st)["x0"]
        loss = (x0 - z).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:4d} x0_l1={float(loss):.4f}")

    model.eval()
    with torch.no_grad():
        zs = ddim(model.dit, zh.shape, ctxh, cmh, style_h, alphas, dev, steps=40)

        def dec(z, t):
            gmm, penl = model.vae.decode(z[:, :t])
            pts = torch.cumsum(gmm_to_points(gmm), dim=1)[0].cpu().numpy()
            pen = penl[0, : len(pts)].argmax(-1).cpu().numpy()
            return pts, pen

        tgt, tpen = dec(zh, tlh)
        gen, gpen = dec(zs, tlh)
        tr_z, tr_ctx, tr_cm, tr_st, tr_t = train[0]
        rec, rpen = dec(ddim(model.dit, tr_z.shape, tr_ctx, tr_cm, tr_st, alphas, dev, 40), tr_t)

    n = min(len(gen), len(tgt))
    epe = float(np.abs(gen[:n] - tgt[:n]).mean())
    cur = cursiveness(model.vae, zs, 0.0)
    print(f"\nEPE(gen vs held-out target): {epe:.3f}")
    print(f"cursiveness: MOVE={cur['MOVE']:.3f} UP={cur['UP']:.3f} "
          f"(data lift rate for this line: see target)")

    figure_panels([
        ("TARGET (held-out text, VAE decode)", render_strokes(points_to_strokes(tgt, tpen), height=100)),
        (f"GENERATED from noise, unseen text   EPE={epe:.2f}",
         render_strokes(points_to_strokes(gen, gpen), height=100, colour=(40, 140, 70))),
        ("RECALL of a training line (upper bound)", render_strokes(points_to_strokes(rec, rpen), height=100, colour=(30, 90, 160))),
    ], "runs_box/gateC.png",
        title=f"Gate C: 12 lines, {STEPS} steps — does unseen text produce its characters?")
    print("figure -> runs_box/gateC.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
