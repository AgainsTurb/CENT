"""Stage 4 sizing and speed check: are the parameters appropriate, and is the
training loop free of slow Python work?"""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.model import CENTModel  # noqa: E402
from cent.models.boundary_dit import BoundaryDiT  # noqa: E402


def count(mod):
    return sum(p.numel() for p in mod.parameters())


def main() -> int:
    # ---------- 1. our parameter budget
    m = CENTModel(n_chars=1198, n_writers=8, d=128, z_dim=64, vae_d=128,
                  stride=8, dit_depth=3, dit_heads=4, n_mix=20, n_steps=1000)
    parts = {
        "InkVAE (frozen at stage 4)": count(m.vae),
        "  encoder": sum(count(getattr(m.vae, n)) for n in
                         ("enc_in", "enc_rs1", "enc_down2", "enc_rs2",
                          "enc_down3", "enc_rs3", "enc_out")),
        "  decoder": sum(count(getattr(m.vae, n)) for n in
                         ("dec_in", "dec_rs1", "dec_up1", "dec_rs2", "dec_up2",
                          "dec_rs3", "dec_up3", "post", "post_proj")),
        "  ctc head": count(m.vae.ctc_head),
        "text encoder": count(m.text),
        "style proj": count(m.style_proj),
        "BoundaryDiT (trained at stage 4)": count(m.dit),
    }
    print("== parameter budget ==")
    for k, v in parts.items():
        print(f"  {k:<34} {v/1e6:8.2f} M")
    trained = parts["BoundaryDiT (trained at stage 4)"] + parts["text encoder"] + parts["style proj"]
    print(f"  {'TRAINED AT STAGE 4':<34} {trained/1e6:8.2f} M")

    # ---------- 2. peer comparison
    print("\n== peer comparison (DiffInk DiT) ==")
    try:
        sys.path.insert(0, str(ROOT / "repos" / "DiffInk"))
        from model.dit import DiT as TheirDiT  # type: ignore
        sig = inspect.signature(TheirDiT.__init__)
        print("  their DiT signature:", sig)
        defaults = {k: v.default for k, v in sig.parameters.items()
                    if v.default is not inspect.Parameter.empty}
        print("  defaults:", {k: v for k, v in defaults.items() if isinstance(v, (int, float))})
        try:
            theirs = TheirDiT(**{k: v for k, v in defaults.items()})
            print(f"  their DiT params (defaults): {count(theirs)/1e6:.2f} M")
        except Exception as e:  # noqa: BLE001
            print("  could not instantiate with defaults:", type(e).__name__, str(e)[:120])
    except Exception as e:  # noqa: BLE001
        print("  DiffInk import failed:", type(e).__name__, str(e)[:120])

    # ---------- 3. step speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    B, T_lat, L = 16, 64, 26
    z0 = torch.randn(B, T_lat, 64, device=device)
    fchar = (torch.arange(T_lat, device=device).repeat(B, 1) // 2)
    fmask = torch.ones(B, T_lat, dtype=torch.bool, device=device)
    ctx = torch.randn(B, L, 128, device=device)
    style = torch.randn(B, 128, device=device)
    opt = torch.optim.AdamW([p for p in m.dit.parameters()], lr=1e-4)
    t = torch.randint(0, 1000, (B,), device=device)

    def step():
        a = m.alphas_cumprod[t].view(-1, 1, 1)
        z_t = a.sqrt() * z0 + (1 - a).sqrt() * torch.randn_like(z0)
        out = m.dit(z_t, t, frame_char=fchar, frame_mask=fmask, char_ctx=ctx, style=style)
        loss = ((out["x0"] - z0) ** 2 * fmask.unsqueeze(-1)).sum() / fmask.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 30
    for _ in range(N):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1000
    print(f"\n== step speed (device={device}, batch {B}, T_lat {T_lat}) ==")
    print(f"  {ms:.1f} ms/step  (forward+backward+opt)")
    for steps in (60_000, 120_000, 200_000):
        print(f"  {steps:>7} steps -> {steps*ms/1000/60:6.1f} min")

    # ---------- 4. look for python loops in the hot path
    print("\n== hot-path inspection ==")
    dl = inspect.getsource(m.dit.forward)
    print(f"  BoundaryDiT.forward: {len(dl.splitlines())} lines, "
          f"python for-loops: {sum(1 for l in dl.splitlines() if l.strip().startswith('for '))}")
    import cent.pipeline.stage_dit as sd
    src = inspect.getsource(sd)
    looplines = [l.strip() for l in src.splitlines()
                 if l.strip().startswith("for ") and "in range(len(" not in l]
    print(f"  stage_dit python loops (outside caching): {len(looplines)}")
    for l in looplines[:8]:
        print("    ", l[:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
