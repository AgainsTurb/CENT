"""Fix verification: decode only real frames, and is the encoder batch-invariant?"""
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
from train_dit_v2 import load_cache  # noqa: E402

D = 384
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load("runs_box/vae_ctc60k.pt", map_location="cpu", weights_only=False)
chars = list(ck.get("vocab") or [])
vocab = CharVocab(chars=chars, stoi={c: i + 1 for i, c in enumerate(chars)})
model = CENTModel(n_chars=max(64, len(chars) + 1),
                  n_writers=int(ck.get("n_writers") or 8), d=D, z_dim=64,
                  vae_d=128, stride=8, dit_depth=3, dit_heads=8, n_mix=20,
                  n_steps=1000).to(dev)
model.vae.load_state_dict(ck["model"])
model.eval()

ds = sd.DitDataset(Path("datasets"), "train", vocab, phase="lines")
i = 8546


def enc(batch_raw):
    b = sd.collate_dit(batch_raw)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    x = torch.cat([b["dxy"], torch.nn.functional.one_hot(b["pen"], 4).float()], -1)
    with torch.no_grad():
        mu, _, lm = model.vae.encode(x, b["point_mask"])
    return mu, lm, b


mu1, lm1, b1 = enc([ds[i]])                                  # alone
mu32, lm32, b32 = enc([ds[j] for j in range(i - 8, i + 24)])  # in a batch of 32
k = 8                                                         # our line's slot
t1 = int(lm1[0].sum())
print(f"alone  : T={mu1.shape[1]} valid={t1}")
print(f"batched: T={mu32.shape[1]} valid={int(lm32[k].sum())} (slot {k})")
T = min(mu1.shape[1], mu32.shape[1])
print(f"L1 alone-vs-batched (first {T}): "
      f"{(mu1[0,:T]-mu32[k,:T]).abs().mean().item():.5f}")

with torch.no_grad():
    g1, p1 = model.vae.decode(mu1[:, :t1])
pts1 = torch.cumsum(gmm_to_points(g1), dim=1)[0].cpu().numpy()
pen1 = p1[0, : len(pts1)].argmax(-1).cpu().numpy()

mu32s = mu32[k:k + 1, : int(lm32[k].sum())]
with torch.no_grad():
    g2, p2 = model.vae.decode(mu32s)
pts2 = torch.cumsum(gmm_to_points(g2), dim=1)[0].cpu().numpy()
pen2 = p2[0, : len(pts2)].argmax(-1).cpu().numpy()

# the broken path: cached latent decoded WITHOUT slicing
cache = load_cache("train", {"vae_hash": __import__("hashlib").sha1(b"".join(
    ck["model"][k2].detach().float().cpu().numpy().tobytes()
    for k2 in sorted(ck["model"]) if torch.is_tensor(ck["model"][k2]))).hexdigest()[:16],
    "stride": 8, "z_dim": 64, "dit_d": D, "text_vocab": len(chars) + 1})
zc = cache["z0"][i].float()[None].to(dev)
with torch.no_grad():
    g3, _ = model.vae.decode(zc)              # unsliced: includes zero padding
pts3 = torch.cumsum(gmm_to_points(g3), dim=1)[0].cpu().numpy()

print(f"decoded alone   : n={len(pts1)}  x[{pts1[:,0].min():.2f},{pts1[:,0].max():.2f}] "
      f"y[{pts1[:,1].min():.2f},{pts1[:,1].max():.2f}]")
print(f"decoded batched : n={len(pts2)}  x[{pts2[:,0].min():.2f},{pts2[:,0].max():.2f}] "
      f"y[{pts2[:,1].min():.2f},{pts2[:,1].max():.2f}]")
print(f"decoded UNsliced: n={len(pts3)}  y_max={pts3[:,1].max():.2f}")

figure_panels([
    (f"fresh encode, sliced to tlat={t1}", render_strokes(points_to_strokes(pts1, pen1), height=110)),
    (f"batched encode, sliced  (n={len(pts2)})", render_strokes(points_to_strokes(pts2, pen2), height=110, colour=(30, 90, 160))),
    (f"cached latent UNsliced (n={len(pts3)}) — the bug", render_strokes(points_to_strokes(pts3, np.zeros(len(pts3), dtype=int)), height=110, colour=(180, 60, 60))),
], "runs_box/decode_fix.png", title="Decode only real frames: does the line become text?")
print("figure -> runs_box/decode_fix.png")
