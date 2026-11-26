"""Does the cursiveness knob act on a class that exists?"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cent.model import CENTModel  # noqa: E402
from train_line_dit import cursiveness  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load("runs_box/vae_ctc60k.pt", map_location="cpu", weights_only=False)
chars = list(ck.get("vocab") or [])
model = CENTModel(n_chars=max(64, len(chars) + 1),
                  n_writers=int(ck.get("n_writers") or 8), d=384, z_dim=64,
                  vae_d=128, stride=8, dit_depth=3, dit_heads=8, n_mix=20,
                  n_steps=1000).to(dev)
model.vae.load_state_dict(ck["model"])
model.eval()

cache = torch.load("stage4_cache/train.pt", map_location="cpu", weights_only=False)
print(f"{'alpha':>7} {'MOVE':>7} {'UP':>7}   (real latents, lines 200/5000/8546)")
for i in (200, 5000, 8546):
    z = cache["z0"][i].float()[None].to(dev)[:, : int(cache["tlat"][i])]
    row = []
    for al in (-6.0, -3.0, 0.0, 3.0, 6.0):
        c = cursiveness(model.vae, z, al)
        row.append(f"a={al:+.0f}: UP={c['UP']:.3f}")
    print(f"  line {i:5d}  " + "  ".join(row))
