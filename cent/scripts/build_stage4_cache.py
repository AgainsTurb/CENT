"""Build the Stage 4 latent cache locally (VAE frozen) and save it to disk.

Stored per line: z0 (fp16 latent), char ids, aligned boundary labels + mask,
real style vector, writer, T_lat. `ctx` is NOT stored — the text encoder is
small and cheap, so training recomputes it, which keeps the file ~3x smaller.

A fingerprint (VAE weights hash + stride/z_dim/d) is stored so training can
refuse a cache built by a different model or code version.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cent.data.tokens import CharVocab  # noqa: E402
from cent.model import CENTModel  # noqa: E402
from cent.pipeline import cache as cache_mod  # noqa: E402
from cent.pipeline import stage_dit as sd  # noqa: E402

D, DEPTH, HEADS, ZD, STRIDE = 384, 12, 8, 64, 8
VAE = "runs_box/vae_ctc60k.pt"
OUT = Path("stage4_cache")


def vae_fingerprint(ck) -> str:
    h = hashlib.sha1()
    for k in sorted(ck["model"].keys()):
        v = ck["model"][k]
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def build(split: str, model, vocab, ck, dev, *, batch=1):
    # batch=1 ON PURPOSE. InkVAE uses nn.GroupNorm(1, C), which normalises over
    # the whole padded temporal extent, so a line encoded in a batch of 32 gets
    # different statistics than the same line encoded alone. Inference encodes
    # one line at a time; the cache must match that, so 1 it is.
    store = cache_mod.LineStore(Path("datasets"), split)
    al_path = Path("datasets/processed") / split / "align.npz"
    al = np.load(al_path, allow_pickle=True) if al_path.exists() else None
    print(f"[{split}] lines={len(store)} align={'yes' if al is not None else 'NO'}")

    ds = sd.DitDataset(Path("datasets"), split, vocab, phase="lines")
    zs, cs, labs, lms, styles, writers, tlats = [], [], [], [], [], [], []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            raw = [ds[j] for j in range(i, min(i + batch, len(ds)))]
            b = sd.collate_dit(raw)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            x = torch.cat([b["dxy"], torch.nn.functional.one_hot(b["pen"], 4).float()], -1)
            mu, _, lm = model.vae.encode(x, b["point_mask"])
            style = model.style_proj((mu * lm.unsqueeze(-1)).sum(1)
                                     / lm.sum(1).clamp_min(1.0).unsqueeze(-1))
            fc, fm, lb, lmask = sd._layout(b["char_ids"], b["point_mask"],
                                           b["bnd_points"], b["bnd_join"],
                                           b["bnd_mask"], STRIDE)
            for k in range(mu.shape[0]):
                t = int(fm[k].sum())
                zs.append(mu[k, :t].half().cpu())
                cs.append(b["char_seq"][k][b["char_mask"][k]].cpu())
                labs.append(lb[k].float().cpu())
                lms.append(lmask[k].cpu())
                styles.append(style[k].float().cpu())
                writers.append(int(b["writers"][k]))
                tlats.append(t)
            if i % (batch * 20) == 0:
                print(f"  {min(i + batch, len(ds))}/{len(ds)}  "
                      f"{(time.time()-t0)/max(1, i+batch)*1000:.0f} ms/line")

    Tmax = max(tlats)
    Lmax = max(int(c.shape[0]) for c in cs)
    Kmax = max(int(l.shape[0]) for l in labs)
    Z = torch.zeros(len(zs), Tmax, ZD, dtype=torch.float16)
    C = torch.zeros(len(cs), Lmax, dtype=torch.long)
    LB = torch.zeros(len(labs), Kmax)
    LM = torch.zeros(len(lms), Kmax, dtype=torch.bool)
    for i in range(len(zs)):
        Z[i, : zs[i].shape[0]] = zs[i]
        C[i, : cs[i].shape[0]] = cs[i]
        LB[i, : labs[i].shape[0]] = labs[i]
        LM[i, : lms[i].shape[0]] = lms[i]
    return {"z0": Z, "char": C, "labels": LB, "lmask": LM,
            "style": torch.stack(styles), "writer": torch.tensor(writers),
            "tlat": torch.tensor(tlats)}


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(VAE, map_location="cpu", weights_only=False)
    chars = list(ck.get("vocab") or [])
    vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})
    model = CENTModel(n_chars=max(64, len(vocab)), n_writers=int(ck.get("n_writers") or 8),
                      d=D, z_dim=ZD, vae_d=128, stride=STRIDE, dit_depth=DEPTH,
                      dit_heads=HEADS, n_mix=20, n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for p in model.vae.parameters():
        p.requires_grad_(False)

    fp = {"vae_hash": vae_fingerprint(ck), "stride": STRIDE, "z_dim": ZD,
          "dit_d": D, "text_vocab": len(vocab), "n_writers": int(ck.get("n_writers") or 8)}
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        try:
            data = build(split, model, vocab, ck, dev)
        except Exception as e:  # noqa: BLE001
            print(f"[{split}] skipped: {type(e).__name__}: {e}")
            continue
        data["fingerprint"] = fp
        p = OUT / f"{split}.pt"
        torch.save(data, p)
        print(f"[{split}] saved {p}  {p.stat().st_size/1e6:.0f} MB  "
              f"Tmax={data['z0'].shape[1]}  n={data['z0'].shape[0]}")
    (OUT / "fingerprint.json").write_text(__import__("json").dumps(fp, indent=2), encoding="utf-8")
    print("fingerprint:", fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
