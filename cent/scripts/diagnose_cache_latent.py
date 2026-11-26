"""Is the cached latent the same object as a freshly encoded one?

If the cache truncated z0 (or normalised the input differently), decoding it
gives a flat stroke while everything downstream looks 'correct'.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cent.model import CENTModel  # noqa: E402
from cent.pipeline import stage_dit as sd  # noqa: E402
from cent.pipeline.common import gmm_to_points  # noqa: E402
from train_dit_v2 import load_cache  # noqa: E402

D = 384
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

from cent.data.tokens import CharVocab
chars = list(ck.get("vocab") or [])
vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})
model = CENTModel(n_chars=max(64, len(chars) + 1),
                  n_writers=int(ck.get("n_writers") or 8), d=D, z_dim=64,
                  vae_d=128, stride=8, dit_depth=3, dit_heads=8, n_mix=20,
                  n_steps=1000).to(dev)
model.vae.load_state_dict(ck["model"])
model.eval()

cache = load_cache("train", expect)
ds = sd.DitDataset(Path("datasets"), "train", vocab, phase="lines")

i = 8546
zc = cache["z0"][i].float()[None].to(dev)
t_cached = int(cache["tlat"][i])
print(f"cached: shape={tuple(zc.shape)}  tlat={t_cached}  "
      f"zero_frames={(zc.abs().sum(-1)[0] == 0).sum().item()}")

raw = sd.collate_dit([ds[i]] if hasattr(ds, "__getitem__") else [ds[0]])
raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
x = torch.cat([raw["dxy"], torch.nn.functional.one_hot(raw["pen"], 4).float()], -1)
with torch.no_grad():
    mu, _, lm = model.vae.encode(x, raw["point_mask"])
print(f"fresh : shape={tuple(mu.shape)}  valid_frames={int(lm.sum())}  "
      f"points={int(raw['point_mask'].sum())}")

with torch.no_grad():
    g1, _ = model.vae.decode(zc)
    g2, _ = model.vae.decode(mu)
p1 = torch.cumsum(gmm_to_points(g1), dim=1)[0].cpu().numpy()
p2 = torch.cumsum(gmm_to_points(g2), dim=1)[0].cpu().numpy()

print(f"decoded cached: n={len(p1)}  x[{p1[:,0].min():.3f},{p1[:,0].max():.3f}] "
      f"y[{p1[:,1].min():.3f},{p1[:,1].max():.3f}]")
print(f"decoded fresh : n={len(p2)}  x[{p2[:,0].min():.3f},{p2[:,0].max():.3f}] "
      f"y[{p2[:,1].min():.3f},{p2[:,1].max():.3f}]")
n = min(len(p1), len(p2))
print(f"EPE cached-vs-fresh: {np.abs(p1[:n]-p2[:n]).mean():.3f}")
T = min(mu.shape[1], zc.shape[1])
print(f"latent L1 cached-vs-fresh (first {T} frames): "
      f"{(zc[0,:T]-mu[0,:T]).abs().mean().item():.4f}")
