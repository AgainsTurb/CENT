"""Stage 4 v3 — line diffusion, no layout, no boundary head.

Removed vs v2:
  * LayoutHead (frames-per-character prediction)          -> no layout at all
  * BoundaryDiT frame_char / boundary slots / bnd_head     -> text via cross-attention
  * boundary BCE on frame-level slots                      -> joins live in pen state

Kept (contributions):
  * per-writer style from a real reference latent
  * whole-line latent generation, so continuity is emergent and per-boundary
  * decode-time cursiveness knob (bias CURSIVE_EOC vs EOC), measured not trained
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cent.model import CENTModel
from cent.models.line_dit import LineDiT
from cent.pipeline.common import gmm_to_points, points_to_strokes
from train_dit_v2 import batch_from, load_cache

PEN_NAMES = {0: "MOVE", 1: "UP", 2: "EOC", 3: "CURSIVE_EOC"}


@torch.no_grad()
def ddim(dit, shape, ctx, cmask, style, alphas, device, steps=24):
    z = torch.randn(shape, device=device)
    ts = torch.linspace(len(alphas) - 1, 0, steps).long().clamp(0, len(alphas) - 1)
    for i, t in enumerate(ts):
        tb = torch.full((shape[0],), int(t), dtype=torch.long, device=device)
        x0 = dit(z, tb, char_ctx=ctx, char_mask=cmask, style=style)["x0"]
        a = alphas[t]
        an = alphas[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        eps = (z - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-6)
        z = an.sqrt() * x0 + (1 - an).sqrt() * eps
    return z


@torch.no_grad()
def cursiveness(vae, z, alpha, tlat=None):
    """Decode-time cursiveness knob over the classes that actually exist.

    `tlat` is MANDATORY in practice: decoding the zero-padded tail fabricates
    ink (measured: 93 of 273 frames for a 180-frame line), which is what made
    earlier renders look like smears.

    CASIA-OLHWDB records only pen-down/pen-up: the data contains MOVE and UP,
    never EOC or CURSIVE_EOC. Continuity between characters is exactly "the pen
    stayed down", so the knob biases MOVE against UP.
    """
    if tlat is not None:
        z = z[:, :tlat]
    gmm, penl = vae.decode(z)
    logits = penl.clone()
    logits[..., 0] = logits[..., 0] + alpha      # MOVE = stays down (join)
    logits[..., 1] = logits[..., 1] - alpha      # UP   = lifts      (break)
    pen = logits.argmax(-1)
    tot = pen.numel()
    out = {PEN_NAMES[k]: float((pen == k).sum()) / tot for k in range(4)}
    out["lift_rate"] = out["UP"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vae", default="runs_box/vae_ctc60k.pt")
    ap.add_argument("--out", default="runs/stage4_v3")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                       ) if a.device == "auto" else torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.vae, map_location="cpu", weights_only=False)
    import hashlib
    h = hashlib.sha1()
    for k in sorted(ck["model"].keys()):
        v = ck["model"][k]
        if torch.is_tensor(v):
            h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    expect = {"vae_hash": h.hexdigest()[:16], "stride": 8, "z_dim": 64,
              "dit_d": a.d, "text_vocab": len(ck.get("vocab") or []) + 1}

    model = CENTModel(n_chars=max(64, len(ck.get("vocab") or []) + 1),
                      n_writers=int(ck.get("n_writers") or 8), d=a.d, z_dim=64,
                      vae_d=128, stride=8, dit_depth=a.depth, dit_heads=a.heads,
                      n_mix=20, n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)

    # replace the layout/boundary DiT with the no-layout one
    model.dit = LineDiT(z_dim=64, d=a.d, depth=a.depth, n_heads=a.heads).to(dev)
    print(f"  trainable DiT {sum(p.numel() for p in model.dit.parameters())/1e6:.2f} M "
          f"(no layout, no boundary head)")

    tr = load_cache("train", expect)
    va = load_cache("val", expect)
    alphas = model.alphas_cumprod

    params = (list(model.dit.parameters()) + list(model.text.parameters())
              + list(model.style_proj.parameters()))
    opt = torch.optim.AdamW(params, lr=a.lr)
    start = 0
    if a.resume:
        rs = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.dit.load_state_dict(rs["dit"]); model.text.load_state_dict(rs["text"])
        model.style_proj.load_state_dict(rs["style"])
        if rs.get("optimizer"):
            opt.load_state_dict(rs["optimizer"])
        start = int(rs.get("step") or 0)
        print(f"  resumed at {start}")

    def save(step):
        torch.save({"dit": model.dit.state_dict(), "text": model.text.state_dict(),
                    "style": model.style_proj.state_dict(), "step": step,
                    "optimizer": opt.state_dict(), "args": vars(a)}, out / "line_dit.pt")

    model.train()
    t0 = time.time()
    n = tr["z0"].shape[0]
    for step in range(start + 1, a.steps + 1):
        idx = torch.randint(0, n, (a.batch,))
        z0, char, cmask, _lab, _lm, style, _tl = batch_from(tr, idx, dev)
        ctx = model.text(char, cmask)
        t = torch.randint(0, model.n_steps, (z0.shape[0],), device=dev)
        aa = alphas[t].view(-1, 1, 1)
        z_t = aa.sqrt() * z0 + (1 - aa).sqrt() * torch.randn_like(z0)
        x0 = model.dit(z_t, t, char_ctx=ctx, char_mask=cmask, style=style)["x0"]
        loss = (x0 - z0).abs().mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % a.log_every == 0:
            print(f"  step {step:6d}  x0_l1={float(loss):.4f}  "
                  f"({(time.time()-t0)/max(1,step-start)*1000:.0f} ms/step)")

        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vidx = torch.arange(min(16, va["z0"].shape[0]))
                vz, vc, vcm, _l, _m, vst, _t = batch_from(va, vidx, dev)
                vctx = model.text(vc, vcm)
                vt = torch.full((vz.shape[0],), 300, dtype=torch.long, device=dev)
                vaa = alphas[300].view(-1, 1, 1)
                vzt = vaa.sqrt() * vz + (1 - vaa).sqrt() * torch.randn_like(vz)
                vx0 = model.dit(vzt, vt, char_ctx=vctx, char_mask=vcm, style=vst)["x0"]
                vmse = float(((vx0 - vz) ** 2).mean())
                zs = ddim(model.dit, vz[:1].shape, vctx[:1], vcm[:1], vst[:1],
                          alphas, dev)
                tl0 = int(batch_from(va, torch.arange(1), dev)[6][0])
                cur = {al: cursiveness(model.vae, zs, al, tl0) for al in (-6.0, 0.0, 6.0)}
            model.train()
            print(f"  [eval] {step:6d}  val x0 MSE={vmse:.4f}")
            for al, c in cur.items():
                print(f"         alpha={al:+.1f}  MOVE={c['MOVE']:.3f}  "
                      f"UP={c['UP']:.3f}  lift_rate={c['lift_rate']:.3f}")
        if step % a.ckpt_every == 0:
            save(step)
    save(a.steps)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
