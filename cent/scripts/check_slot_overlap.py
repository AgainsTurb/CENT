"""How many boundary slots are actually scored? The number that decides everything."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_dit_v2 import counts_from_labels, frame_char_from_counts  # noqa: E402
from cent.models.boundary_dit import BoundaryDiT  # noqa: E402

d = torch.load("stage4_cache/train.pt", map_location="cpu", weights_only=False)

for tag, shift in (("as cached", 0), ("shifted -1", 1)):
    tot_lab = tot_slot = tot_over = 0
    join_share = []
    for i in range(50):
        lm = d["lmask"][i].clone()
        lb = d["labels"][i].clone()
        if shift:
            lm = torch.cat([lm[1:], torch.zeros(1, dtype=lm.dtype)])
            lb = torch.cat([lb[1:], torch.zeros(1, dtype=lb.dtype)])
        n_chars = int((d["char"][i] > 0).sum())
        cnts = counts_from_labels(lm.unsqueeze(0), d["tlat"][i].unsqueeze(0),
                                 torch.tensor([n_chars]))[0]
        fc = frame_char_from_counts(cnts, int(d["tlat"][i])).unsqueeze(0)
        lay = BoundaryDiT.build_layout(fc, fc >= 0)
        valid = (lay["bnd_slot"][0] >= 0)          # 1-D, batch of 1
        W = valid.shape[0]
        m = lm[:W] & valid
        tot_lab += int(lm.sum())
        tot_slot += int(valid.sum())
        tot_over += int(m.sum())
        if m.any():
            join_share.append(float(lb[:W][m].mean()))
    print(f"[{tag}] labels={tot_lab}  valid_slots={tot_slot}  SCORED={tot_over} "
          f"({tot_over/max(1,tot_lab)*100:.1f}% of labels)  "
          f"join_share={sum(join_share)/max(1,len(join_share)):.3f}")
