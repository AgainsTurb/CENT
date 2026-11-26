"""Gate A: cached latent honoured, decode sliced, and does it render as text?"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cent.data.tokens import CharVocab  # noqa: E402
from cent.model import CENTModel  # noqa: E402
from cent.pipeline import stage_dit as sd  # noqa: E402
from cent.pipeline.common import gmm_to_points, points_to_strokes  # noqa: E402
from cent.viz import figure_panels, render_strokes  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load("runs_box/vae_ctc60k.pt", map_location="cpu", weights_only=False)
chars = list(ck.get("vocab") or [])
vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})
model = CENTModel(n_chars=max(64, len(chars) + 1),
                  n_writers=int(ck.get("n_writers") or 8), d=384, z_dim=64,
                  vae_d=128, stride=8, dit_depth=3, dit_heads=8, n_mix=20,
                  n_steps=1000).to(dev)
model.vae.load_state_dict(ck["model"])
model.eval()

cache = torch.load("stage4_cache/train.pt", map_location="cpu", weights_only=False)
ds = sd.DitDataset(Path("datasets"), "train", vocab, phase="lines")

def dec(z, t):
    with torch.no_grad():
        g, p = model.vae.decode(z[:, :t])          # SLICE: no zero-padded frames
    pts = torch.cumsum(gmm_to_points(g), dim=1)[0].cpu().numpy()
    pen = p[0, : len(pts)].argmax(-1).cpu().numpy()
    return pts, pen

panels = []
for i in [8546, 200, 5000]:
    zc = cache["z0"][i].float()[None].to(dev)
    t = int(cache["tlat"][i])
    raw = sd.collate_dit([ds[i]])
    raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
    x = torch.cat([raw["dxy"], torch.nn.functional.one_hot(raw["pen"], 4).float()], -1)
    with torch.no_grad():
        mu, _, lm = model.vae.encode(x, raw["point_mask"])
    l1 = float((zc[0, :t] - mu[0, :t]).abs().mean())
    pen_states, counts = torch.unique(raw["pen"][0][: int(raw["point_mask"][0].sum())],
                                      return_counts=True)
    dist = {int(k): float(v) / counts.sum() for k, v in zip(pen_states, counts)}
    pts, pen = dec(zc, t)
    print(f"line {i}: tlat={t}  L1(cached,fresh)={l1:.6f}  "
          f"pts={len(pts)}  y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]")
    print(f"   pen states in data: {dist}")
    panels.append((f"line {i}  L1={l1:.4f}  n={len(pts)}",
                   render_strokes(points_to_strokes(pts, pen), height=90)))

figure_panels(panels, "runs_box/gateA_decode.png",
              title="Gate A: cached latent, decode sliced to tlat — is this text?")
print("figure -> runs_box/gateA_decode.png")
