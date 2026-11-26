"""Is the cached boundary label misaligned with the rebuilt layout?"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_dit_v2 import counts_from_labels, frame_char_from_counts  # noqa: E402

d = torch.load("stage4_cache/train.pt", map_location="cpu", weights_only=False)
lmask, lab = d["lmask"], d["labels"]
char, tlat = d["char"], d["tlat"]
n = min(300, lmask.shape[0])

exact = off_by = other = 0
mismatch_examples = []
for i in range(n):
    lm = lmask[i]
    pos_label = torch.nonzero(lm, as_tuple=False).flatten().tolist()
    n_chars = int((char[i] > 0).sum())
    T = int(tlat[i])
    cnts = counts_from_labels(lm.unsqueeze(0), tlat[i].unsqueeze(0),
                              torch.tensor([n_chars]))[0]
    fc = frame_char_from_counts(cnts, T)
    # boundary positions implied by the rebuilt layout
    pos_rebuilt = (torch.nonzero(fc[1:] != fc[:-1], as_tuple=False).flatten() + 1).tolist()
    if pos_label == pos_rebuilt:
        exact += 1
    else:
        diff = (len(pos_label) - len(pos_rebuilt))
        if len(pos_rebuilt) and pos_label[:len(pos_rebuilt)] == pos_rebuilt:
            off_by += 1
        else:
            other += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append((i, len(pos_label), len(pos_rebuilt),
                                          pos_label[:6], pos_rebuilt[:6]))

print(f"lines checked: {n}")
print(f"  exact match        : {exact}")
print(f"  prefix match (off) : {off_by}")
print(f"  mismatched         : {other}")
print(f"  n_chars vs labels  : {int((char[0] > 0).sum())} vs {int(lmask[0].sum())}")
for e in mismatch_examples:
    print("  example line", e[0], "labels", e[1], "rebuilt", e[2], "\n    lab", e[3], "\n    reb", e[4])
