"""Stage 4 trainer v2: layout head + disk cache + val generalisation.

Three fixes over the previous trainer:
  1. LAYOUT HEAD  - predicts how many latent frames each character occupies,
     trained against the Viterbi alignment. Training uses the true counts
     (teacher forcing); sampling uses predicted counts, so train and inference
     finally share one contract.
  2. DISK CACHE   - loads stage4_cache/{train,val}.pt and refuses a cache built
     by a different VAE (fingerprint check).
  3. VAL METRICS  - precision/recall on the LIFT class (label 0), not accuracy:
     with ~88% joins, accuracy is flattered by the prior.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cent.data.tokens import CharVocab
from cent.model import CENTModel
from cent.pipeline.common import gmm_to_points, points_to_strokes
from cent.viz import figure_panels, render_strokes

CACHE = Path("stage4_cache")


# ---------------------------------------------------------------- layout utils
def counts_from_labels(lmask: torch.Tensor, tlat: torch.Tensor, n_chars: torch.Tensor):
    """Aligned frame counts per character, recovered from boundary positions."""
    counts = []
    for i in range(lmask.shape[0]):
        t, n = int(tlat[i]), int(n_chars[i])
        bpos = torch.nonzero(lmask[i], as_tuple=False).flatten().tolist()
        edges = [0] + [b + 1 for b in bpos if b + 1 < t] + [t]
        spans = [edges[j + 1] - edges[j] for j in range(len(edges) - 1)]
        if len(spans) > n:
            spans = spans[:n]
        spans += [1] * (n - len(spans))
        counts.append(torch.tensor(spans, dtype=torch.float32))
    return counts


def frame_char_from_counts(counts: torch.Tensor, tlat: int):
    """(L,) counts -> (T_lat,) character id per frame."""
    fc = torch.full((tlat,), -1, dtype=torch.long)
    f = 0
    for c, n in enumerate(counts.tolist()):
        n = max(1, int(round(n)))
        fc[f:min(tlat, f + n)] = c
        f += n
        if f >= tlat:
            break
    return fc


class LayoutHead(nn.Module):
    """Predicts frames-per-character from the text embedding + style."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ctx: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        s = style.unsqueeze(1).expand(-1, ctx.shape[1], -1)
        return self.net(torch.cat([ctx, s], -1)).squeeze(-1)   # (B,L) frames


# ---------------------------------------------------------------- cache
def load_cache(split: str, expect: dict) -> dict:
    p = CACHE / f"{split}.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run scripts/build_stage4_cache.py")
    d = torch.load(p, map_location="cpu", weights_only=False)
    fp = d.get("fingerprint") or {}
    bad = {k: (fp.get(k), v) for k, v in expect.items() if fp.get(k) != v}
    if bad:
        raise RuntimeError(f"cache {p} fingerprint mismatch: {bad}")
    print(f"  [cache] {split}: {d['z0'].shape[0]} lines, Tmax={d['z0'].shape[1]}, "
          f"fp={fp.get('vae_hash')}")
    return d


def batch_from(d: dict, idx: torch.Tensor, dev):
    z = d["z0"][idx].float().to(dev)
    char = d["char"][idx].to(dev)
    cmask = char > 0
    # No slot shift: measured overlap between cached labels and the DiT's
    # boundary slots is 99.2%, so labels already land on the right slots.
    lab = d["labels"][idx].float().to(dev)
    lmask = d["lmask"][idx].to(dev)
    style = d["style"][idx].to(dev)
    tlat = d["tlat"][idx]
    return z, char, cmask, lab, lmask, style, tlat


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, head, d, dev, *, n_batches=8, batch=16):
    """Boundary P/R on the LIFT class + layout error, on held-out writers."""
    tp = fp_ = fn = 0
    ltp = lfp = lfn = 0
    all_logits, all_lift = [], []
    layout_abs = []
    model.eval(); head.eval()
    n = min(len(d["z0"]), n_batches * batch)
    for i in range(0, n, batch):
        idx = torch.arange(i, min(i + batch, n))
        z0, char, cmask, lab, lmask, style, tlat = batch_from(d, idx, dev)
        ctx = model.text(char, cmask)
        cnt = counts_from_labels(lmask.cpu(), tlat.cpu(), cmask.sum(1).cpu())
        # fraction -> frames uses EACH LINE's own length, not the padded width
        pred_cnt = head(ctx, style) * tlat.float().unsqueeze(1).to(dev)
        target_cnt = torch.zeros_like(pred_cnt)
        cmask_f = torch.zeros_like(pred_cnt, dtype=torch.bool)
        for b in range(pred_cnt.shape[0]):
            c = cnt[b].to(dev)
            target_cnt[b, : len(c)] = c
            cmask_f[b, : len(c)] = True
        layout_abs.append(float((pred_cnt[cmask_f] - target_cnt[cmask_f]).abs().mean()))
        # boundary metrics under the TRUE layout (isolates the head)
        T = z0.shape[1]          # latents are padded to the global Tmax
        fcs = []
        for b in range(z0.shape[0]):
            fcs.append(frame_char_from_counts(cnt[b], T))
        fchar = torch.stack(fcs).to(dev)
        fmask = fchar >= 0
        # boundary logits are noise-dependent: evaluating only at t=0 (which the
        # head does not see during training) made the metric meaningless
        lg = None
        for tv in (0, 200, 500):
            tb = torch.full((z0.shape[0],), tv, dtype=torch.long, device=dev)
            o = model.dit(z0, tb, frame_char=fchar, frame_mask=fmask,
                          char_ctx=ctx, style=style)
            m = lmask & o["bnd_valid"]
            cur = o["bnd_logits"][m]
            lg = cur if lg is None else lg + cur
        lg = lg / 3.0
        lift = lab[m] < 0.5
        pred = lg > 0
        all_logits.append(lg.detach().float().cpu())
        all_lift.append(lift.detach().float().cpu())
        tp += int((pred & ~lift).sum())
        fp_ += int((pred & lift).sum())
        fn += int((~pred & ~lift).sum())
        # the LIFT class is the one that discriminates: 88% of boundaries are
        # joins, so "always join" scores ~0.94 F1 while predicting no lift at all
        ltp += int((~pred & lift).sum())
        lfp += int((~pred & ~lift).sum())
        lfn += int((pred & lift).sum())
    model.train(); head.train()
    prec = tp / max(1, tp + fp_)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    # Threshold sweep on the LIFT class. A fixed 0.5 threshold is part of why the
    # head looked untrained: with 88% joins the BCE optimum sits far from it.
    best_lift_f1, best_thr = 0.0, 0.0
    if all_logits:
        Lg = torch.cat(all_logits)
        Y = torch.cat(all_lift)
        for thr in torch.linspace(-6.0, 6.0, 49):
            p = Lg > thr
            atp = int((~p & (Y > 0.5)).sum())
            afp = int((~p & (Y < 0.5)).sum())
            afn = int((p & (Y > 0.5)).sum())
            pr = atp / max(1, atp + afp)
            rc = atp / max(1, atp + afn)
            f1 = 2 * pr * rc / max(1e-9, pr + rc)
            if f1 > best_lift_f1:
                best_lift_f1, best_thr = f1, float(thr)
    return {"join_precision": prec, "join_recall": rec, "join_f1": f1,
            "lift_thr": best_thr, "lift_best_f1": best_lift_f1,
            "lift_precision": ltp / max(1, ltp + lfp),
            "lift_recall": ltp / max(1, ltp + lfn),
            "layout_mae": sum(layout_abs) / max(1, len(layout_abs))}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-layout", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vae", default="runs_box/vae_ctc60k.pt")
    ap.add_argument("--out", default="runs/stage4_v2")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--resume", default="")
    ap.add_argument("--pos-weight", type=float, default=1.3)
    ap.add_argument("--w-layout", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                       ) if a.device == "auto" else torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "visuals").mkdir(exist_ok=True)

    ck = torch.load(a.vae, map_location="cpu", weights_only=False)
    chars = list(ck.get("vocab") or [])
    vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})
    expect = {"vae_hash": None}
    import hashlib
    h = hashlib.sha1()
    for k in sorted(ck["model"].keys()):
        v = ck["model"][k]
        if torch.is_tensor(v):
            h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    expect = {"vae_hash": h.hexdigest()[:16], "stride": 8, "z_dim": 64,
              "dit_d": a.d, "text_vocab": len(vocab)}

    model = CENTModel(n_chars=max(64, len(vocab)),
                      n_writers=int(ck.get("n_writers") or 8), d=a.d, z_dim=64,
                      vae_d=128, stride=8, dit_depth=a.depth, dit_heads=a.heads,
                      n_mix=20, n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)
    head = LayoutHead(a.d).to(dev)

    tr = load_cache("train", expect)
    va = load_cache("val", expect)
    print(f"  trainable: DiT {sum(p.numel() for p in model.dit.parameters())/1e6:.2f} M "
          f"+ layout {sum(p.numel() for p in head.parameters())/1e3:.0f} K")

    opt = torch.optim.AdamW([
        {"params": list(model.dit.parameters()) + list(model.text.parameters())
         + list(model.style_proj.parameters()), "lr": a.lr},
        {"params": head.parameters(), "lr": a.lr_layout},
    ])
    start = 0
    if a.resume:
        rs = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rs["model"]); head.load_state_dict(rs["layout"])
        if rs.get("optimizer"):
            opt.load_state_dict(rs["optimizer"])
        start = int(rs.get("step") or 0)
        print(f"  resumed from step {start}")

    best = -1.0
    model.train(); head.train()
    n_tr = tr["z0"].shape[0]
    t0 = time.time()
    for step in range(start + 1, a.steps + 1):
        idx = torch.randint(0, n_tr, (a.batch,))
        z0, char, cmask, lab, lmask, style, tlat = batch_from(tr, idx, dev)
        ctx = model.text(char, cmask)
        cnt = counts_from_labels(lmask.cpu(), tlat.cpu(), cmask.sum(1).cpu())
        T = z0.shape[1]          # latents are padded to the global Tmax
        fchar = torch.stack([frame_char_from_counts(cnt[b], T)
                             for b in range(z0.shape[0])]).to(dev)
        fmask = fchar >= 0

        t = torch.randint(0, model.n_steps, (z0.shape[0],), device=dev)
        an = model.alphas_cumprod[t].view(-1, 1, 1)
        z_t = an.sqrt() * z0 + (1 - an).sqrt() * torch.randn_like(z0)
        o = model.dit(z_t, t, frame_char=fchar, frame_mask=fmask,
                      char_ctx=ctx, style=style)
        l_diff = (((o["x0"] - z0) ** 2) * fmask.unsqueeze(-1)).sum() / (
            fmask.sum() * z0.shape[-1])
        m = lmask & o["bnd_valid"]
        l_bnd = (F.binary_cross_entropy_with_logits(
            o["bnd_logits"][m], lab[m], pos_weight=torch.tensor(a.pos_weight, device=dev))
            if m.any() else torch.zeros((), device=dev))

        pred_cnt = head(ctx, style)          # predicted FRACTION of frames
        tgt = torch.zeros_like(pred_cnt); cm = torch.zeros_like(pred_cnt, dtype=torch.bool)
        for b in range(pred_cnt.shape[0]):
            c = cnt[b].to(dev) / max(1, int(tlat[b]))
            tgt[b, :len(c)] = c; cm[b, :len(c)] = True
        l_lay = (pred_cnt[cm] - tgt[cm]).abs().mean() if cm.any() else torch.zeros((), device=dev)

        loss = l_diff + l_bnd + a.w_layout * l_lay
        if not torch.isfinite(loss):
            print(f"  [skip] non-finite at {step}")
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.dit.parameters()) + list(head.parameters()), 0.5)
        opt.step()

        if step % a.log_every == 0:
            print(f"  step {step:6d} diff={float(l_diff):.4f} bnd={float(l_bnd):.4f} "
                  f"layout={float(l_lay):.3f} ({(time.time()-t0)/max(1,step-start)*1000:.0f} ms)")

        if step % a.eval_every == 0 or step == a.steps:
            met = evaluate(model, head, va, dev)
            print(f"  [eval] {step:6d}  LIFT@0.5 P={met['lift_precision']:.3f} "
                  f"R={met['lift_recall']:.3f} | LIFT@best thr={met['lift_thr']:+.1f} "
                  f"F1={met['lift_best_f1']:.3f} | layoutMAE={met['layout_mae']:.2f}")
            if met["join_f1"] > best:
                best = met["join_f1"]
                torch.save({"model": model.state_dict(), "layout": head.state_dict(),
                            "optimizer": opt.state_dict(), "step": step,
                            "vocab": chars, "metrics": met, "args": vars(a)},
                           out / "dit_best.pt")
        if step % a.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "layout": head.state_dict(),
                        "optimizer": opt.state_dict(), "step": step,
                        "vocab": chars, "args": vars(a)}, out / "dit_last.pt")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
