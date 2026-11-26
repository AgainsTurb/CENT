"""Align the validation split, then build its Stage 4 cache.

Stage 3 only ever ran on `train`. Without a val alignment there are no boundary
labels on held-out writers, so Stage 4 generalisation cannot be measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_stage4_cache as B  # noqa: E402
from cent.data.tokens import CharVocab  # noqa: E402
from cent.models.inkvae import InkVAE  # noqa: E402
from cent.pipeline import align as A  # noqa: E402
from cent.pipeline import cache as cache_mod  # noqa: E402


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("datasets")
    ck = torch.load(B.VAE, map_location="cpu", weights_only=False)
    chars = list(ck.get("vocab") or [])
    vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})

    vae = InkVAE(d=128, z_dim=64, stride=B.STRIDE, n_mix=20,
                 n_chars=max(64, len(vocab)),
                 n_writers=int(ck.get("n_writers") or 8)).to(dev)
    vae.load_state_dict(ck["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    store = cache_mod.LineStore(root, "val")
    print(f"[val] lines={len(store)}  aligning ...")
    res = A.derive_store_boundaries(vae, store, vocab, stride=B.STRIDE,
                                    device=dev, limit=None, min_conf=0.0)
    out_align = root / "processed" / "val" / "align.npz"
    A.save_alignment(res, store, out_align)
    n_b = len(res.get("boundary_points", [])) if isinstance(res, dict) else -1
    print(f"[val] boundaries={n_b}")
    p = root / "processed" / "val" / "align.npz"
    print(f"[val] align.npz exists={p.exists()} size={p.stat().st_size/1e6 if p.exists() else 0:.2f} MB")

    # ---- rebuild only the val cache
    model = B.CENTModel(n_chars=max(64, len(vocab)),
                        n_writers=int(ck.get("n_writers") or 8), d=B.D,
                        z_dim=B.ZD, vae_d=128, stride=B.STRIDE,
                        dit_depth=B.DEPTH, dit_heads=B.HEADS, n_mix=20,
                        n_steps=1000).to(dev)
    model.vae.load_state_dict(ck["model"])
    for q in model.vae.parameters():
        q.requires_grad_(False)

    data = B.build("val", model, vocab, ck, dev)
    data["fingerprint"] = {"vae_hash": B.vae_fingerprint(ck), "stride": B.STRIDE,
                           "z_dim": B.ZD, "dit_d": B.D,
                           "text_vocab": len(vocab),
                           "n_writers": int(ck.get("n_writers") or 8)}
    out = B.OUT / "val.pt"
    torch.save(data, out)
    print(f"[val] saved {out}  {out.stat().st_size/1e6:.0f} MB  n={data['z0'].shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
