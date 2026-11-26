"""Paired dataset for CENT.

CENT needs, for one training example:
  * a stroke graph          -- from the ONLINE corpus (OLHWDB .pot)
  * a rendered target image -- from the OFFLINE corpus (HWDB .gnt)
  * k style references      -- offline or online, same writer, DIFFERENT characters
  * a character index       -- into the lexicon charset

The join is on (writer, character). Pairing granularity is exactly that: the
trajectory we read and the raster we supervise are different acts of writing by
the same writer, so the image is a style-consistent target, not a pixel-exact
rendering. This is stated in the paper as a limitation, and it is why the
rendering loss is a soft objective rather than a reconstruction.

If the corpus supplies no online samples, this dataset raises: a pixel-only
corpus cannot train CENT's stroke branch, and silently falling back would make
the method something other than what the paper describes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .models.stroke_graph import N_MAX, strokes_to_tensor
from .lexicon_bridge import topology_target, fit_strokes_to_count


class PairedStrokeDataset(Dataset):
    def __init__(self, canonical_root: str | Path, split: str = "train", k_refs: int = 5,
                 img_size: int = 64, seed: int = 0, version: str | None = None,
                 topology_mode: str = "lexicon"):
        if topology_mode not in ("lexicon", "observed"):
            raise ValueError("topology_mode must be 'lexicon' (hard constraint) or "
                             "'observed' (ablation A2)")
        self.topology_mode = topology_mode
        self.root = Path(canonical_root) / version if version else Path(canonical_root)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.img_size = meta["img_size"] if img_size is None else img_size
        self.k_refs = k_refs
        self.split = split
        self.rng = np.random.default_rng(seed)

        d = self.root / split
        self.images = np.load(d / "images.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy")
        self.writers = np.load(d / "writers.npy")
        self.source = np.load(d / "source.npy")
        # Canonical stroke counts from the lexicon. Without this file the "hard
        # constraint" would be the writer's observed count in disguise.
        lex_path = d / "lexicon_strokes.npy"
        self.lexicon_strokes = (np.load(lex_path) if lex_path.exists()
                                else np.full(len(self.labels), -1, np.int16))

        on_path = d / "online.npz"
        if not on_path.exists():
            raise RuntimeError(
                f"{on_path} missing: this split has no online samples. CENT trains on stroke "
                "graphs, so an offline-only corpus cannot be used. Build the canonical set with "
                "both --offline and --online directories."
            )
        on = dict(np.load(on_path))
        self._online_index = on["index"]            # positions in `images` that have strokes
        self._stroke_starts = on["stroke_starts"]
        self._pt_starts = on["pt_starts"]
        self._points = on["points"]
        if len(self._online_index) == 0:
            raise RuntimeError("online.npz contains zero samples")

        # (writer, label) -> positions, for online samples and for image targets
        self.online_by_key: dict[tuple[int, int], list[int]] = {}
        for pos in self._online_index.tolist():
            key = (int(self.writers[pos]), int(self.labels[pos]))
            self.online_by_key.setdefault(key, []).append(pos)
        self.image_by_key: dict[tuple[int, int], list[int]] = {}
        for pos in range(len(self.labels)):
            key = (int(self.writers[pos]), int(self.labels[pos]))
            self.image_by_key.setdefault(key, []).append(pos)
        # writer -> positions usable as style references
        self.by_writer: dict[int, list[int]] = {}
        for pos in range(len(self.labels)):
            self.by_writer.setdefault(int(self.writers[pos]), []).append(pos)
        self.keys = sorted(self.online_by_key)
        self.charset = meta["charset"]

    def __len__(self) -> int:
        return len(self.keys)

    def strokes_for(self, pos: int) -> list[np.ndarray]:
        k = int(np.searchsorted(self._online_index, pos))
        s0, s1 = int(self._stroke_starts[k]), int(self._stroke_starts[k + 1])
        out = []
        for si in range(s0, s1):
            a, b = int(self._pt_starts[si]), int(self._pt_starts[si + 1])
            out.append(self._points[a:b])
        return out

    def _refs(self, writer: int, exclude_label: int) -> tuple[np.ndarray, np.ndarray]:
        pool = [p for p in self.by_writer.get(writer, []) if int(self.labels[p]) != exclude_label]
        if not pool:
            pool = self.by_writer.get(writer, [])
        pick = self.rng.choice(len(pool), size=min(self.k_refs, len(pool)), replace=False)
        idx = [pool[int(i)] for i in np.atleast_1d(pick)]
        refs = np.zeros((self.k_refs, self.img_size, self.img_size), np.float32)
        mask = np.zeros((self.k_refs,), np.float32)
        for j, p in enumerate(idx[: self.k_refs]):
            refs[j] = 1.0 - np.asarray(self.images[p], np.float32) / 255.0   # 1 = ink
            mask[j] = 1.0
        return refs, mask

    def __getitem__(self, i: int) -> dict:
        writer, label = self.keys[i]
        pos = self.online_by_key[(writer, label)][0]
        strokes = self.strokes_for(pos)
        observed = len(strokes)
        lex = int(self.lexicon_strokes[pos]) if pos < len(self.lexicon_strokes) else -1
        target_n, mismatch = topology_target(observed, lex if lex > 0 else None,
                                             mode=self.topology_mode)
        strokes, action = fit_strokes_to_count(strokes, target_n)
        tok, smask = strokes_to_tensor(strokes, N_MAX)
        refs, rmask = self._refs(writer, label)

        target = np.zeros((1, self.img_size, self.img_size), np.float32)
        has_target = 0.0
        cand = [p for p in self.image_by_key.get((writer, label), []) if int(self.source[p]) == 0]
        if cand:
            p = cand[self.rng.integers(len(cand))]
            target[0] = 1.0 - np.asarray(self.images[p], np.float32) / 255.0
            has_target = 1.0

        return {
            "tokens": tok,                                   # [N_MAX, 8]
            "stroke_mask": smask,                            # [N_MAX]
            "char_idx": torch.tensor(label, dtype=torch.long),
            "char": self.charset[label],
            "writer": torch.tensor(writer, dtype=torch.long),
            "refs": torch.from_numpy(refs).unsqueeze(1),     # [k,1,S,S]
            "ref_mask": torch.from_numpy(rmask),             # [k]
            "target_img": torch.from_numpy(target),          # [1,S,S]
            "has_target": torch.tensor(has_target),
            "n_strokes": torch.tensor(target_n, dtype=torch.long),
            "count_mismatch": torch.tensor(float(mismatch)),
            "fit_action": action,                            # exact | truncated | padded
        }


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        if k in ("char",):
            out[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out
