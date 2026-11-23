"""Line-level training data for CENT.

Turns a canonical line into the token sequence the denoiser consumes:

    [char strokes of c1][layout c1][connector c1|c2][char strokes of c2][layout c2]...

Design decisions encoded here (see docs/CENT_TRAINING_PLAN.md):

* **Canvas 64x1024, aspect preserved.** Coordinates are mapped into the canvas frame with a
  SINGLE scale (`s = min(W/span_x, H/span_y)`), so a 1:16 line stays 1:16. Normalising x and
  y independently would square up the line and make the renderer's image target wrong.
* **`max_chars=32`, `max_tokens=1024`.** Measured: level-2 glyphs average 11.69 strokes, so
  40 characters need 547 tokens and would be truncated mid-character at 512. Lines longer
  than 32 characters are dropped and counted (`n_too_long`), never silently truncated.
* **`paired_only`** restricts the set to lines carrying BOTH an image and a trajectory, which
  is the first campaign: connectivity supervision is exact and the image target exists.
* **Style references** are k other lines from the same writer, downscaled to the canvas
  (not truncated: truncation would show the model only the first few characters).
* **Writer ids** are returned so the writer head can learn something (it used to be trained
  against a constant).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .models.line import KIND_CHAR_STROKE, KIND_CONNECTOR, KIND_LAYOUT, PAYLOAD
from .models.stroke_graph import fit_stroke

MAX_CHARS = 32
MAX_TOKENS = 1024


def _fit(a: np.ndarray) -> np.ndarray:
    """Polyline -> 8-vector cubic payload."""
    return fit_stroke(a).reshape(-1)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize to (h, w); keeps the WHOLE line, never crops it."""
    ih, iw = img.shape[:2]
    if (ih, iw) == (h, w):
        return img
    rows = (np.arange(h) * ih / h).astype(np.int32).clip(0, ih - 1)
    cols = (np.arange(w) * iw / w).astype(np.int32).clip(0, iw - 1)
    return img[rows][:, cols]


class LineTokenDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", lexicon: dict[str, int] | None = None,
                 k_refs: int = 5, img_h: int = 64, canvas_w: int = 1024,
                 max_chars: int = MAX_CHARS, max_tokens: int = MAX_TOKENS,
                 paired_only: bool = False, limit: int = 0, seed: int = 0):
        self.root = Path(root)
        self.split = split
        self.k_refs = k_refs
        self.img_h = img_h
        self.canvas_w = canvas_w
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.paired_only = paired_only
        self.rng = np.random.default_rng(seed)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.charset = meta["charset"]
        self.char2idx = {c: i for i, c in enumerate(self.charset)}
        self.no_char = len(self.charset)          # the "condition dropped" index
        d = self.root / split
        self.records = [json.loads(l) for l in
                        (d / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        tp = d / "trajectories.npz"
        self.traj = dict(np.load(tp)) if tp.exists() else None
        if lexicon is None:
            raise ValueError("LineTokenDataset needs the lexicon: the stroke count is the constraint")
        self.lexicon = lexicon

        writers = sorted({int(r["writer"]) for r in self.records})
        self.writer2idx = {w: i for i, w in enumerate(writers)}
        self.writers = set(writers)

        self.usable: list[int] = []
        self.n_too_long = 0
        self.n_unpaired = 0
        for i, r in enumerate(self.records):
            if not r.get("has_strokes"):
                continue
            if self.paired_only and not r.get("paired"):
                self.n_unpaired += 1
                continue
            if not (0 < len(r["text"]) <= max_chars):
                self.n_too_long += 1
                continue
            self.usable.append(i)
        if limit:
            self.usable = self.usable[:limit]
        self.by_writer: dict[int, list[int]] = {}
        for i in self.usable:
            self.by_writer.setdefault(int(self.records[i]["writer"]), []).append(i)

    def __len__(self) -> int:
        return len(self.usable)

    def _strokes(self, i: int) -> tuple[list[np.ndarray], list[int]]:
        if self.traj is None:
            return [], []
        a = int(self.traj["line_stroke_off"][i])
        b = int(self.traj["line_stroke_off"][i + 1])
        pts, chars = [], []
        for s in range(a, b):
            p0, p1 = int(self.traj["stroke_off"][s]), int(self.traj["stroke_off"][s + 1])
            pts.append(np.asarray(self.traj["points"][p0:p1], np.float32))
            chars.append(int(self.traj["stroke_char"][s]))
        return pts, chars

    def _full_image(self, i: int) -> np.ndarray | None:
        r = self.records[i]
        if not r.get("image"):
            return None
        from .common_bridge import read_png
        p = self.root / self.split / "images" / r["image"]
        if not p.exists():
            return None
        img = read_png(p).astype(np.float32) / 255.0
        return _resize(img, self.img_h, self.canvas_w)

    def _refs(self, writer: int, exclude: int) -> tuple[np.ndarray, np.ndarray]:
        pool = [i for i in self.by_writer.get(writer, []) if i != exclude]
        if not pool:
            pool = self.usable[:1]
        pick = self.rng.choice(len(pool), size=min(self.k_refs, len(pool)), replace=False)
        refs = np.ones((self.k_refs, 1, self.img_h, self.canvas_w), np.float32)   # 1 = paper
        mask = np.zeros((self.k_refs,), np.float32)
        for j, p in enumerate(np.atleast_1d(pick)[: self.k_refs]):
            img = self._full_image(int(pool[int(p)]))
            if img is None:
                continue
            refs[j, 0] = img
            mask[j] = 1.0
        return refs, mask

    def __getitem__(self, k: int) -> dict:
        i = self.usable[k]
        r = self.records[i]
        strokes, schar = self._strokes(i)
        text = r["text"][: self.max_chars]

        # ---- aspect-preserving normalisation into the canvas frame ----
        allp = np.concatenate(strokes) if strokes else np.zeros((1, 2), np.float32)
        lo, hi = allp.min(0), allp.max(0)
        span = np.maximum(hi - lo, 1e-6)
        scale = min(self.canvas_w / span[0], self.img_h / span[1])   # ONE scale, not two

        def norm(a: np.ndarray) -> np.ndarray:
            q = np.asarray(a, np.float32) - lo
            q = q * np.array([scale / self.canvas_w, scale / self.img_h], np.float32)
            return q

        tokens, kinds, cpos, slots = [], [], [], []
        join_flags: list[float] = []
        for j, ch in enumerate(text):
            n_lex = self.lexicon.get(ch, 0)
            own = [s for s, c in zip(strokes, schar) if c == j]
            n = n_lex if n_lex > 0 else max(1, len(own))
            for si in range(min(n, self.max_tokens - len(tokens))):
                if si < len(own):
                    tokens.append(_fit(norm(own[si])))
                else:
                    tokens.append(np.zeros(PAYLOAD, np.float32))    # zero-length padding
                kinds.append(KIND_CHAR_STROKE)
                cpos.append(self.char2idx.get(ch, self.no_char))
                slots.append(j)
            # layout token: this character's ink centroid, in canvas coordinates
            if own:
                c0 = norm(np.concatenate(own)).mean(0)
                dx, dy = float(c0[0]), float(c0[1])
            else:
                dx = dy = 0.0
            lay = np.zeros(PAYLOAD, np.float32)
            lay[0], lay[1] = dx, dy
            tokens.append(lay); kinds.append(KIND_LAYOUT); cpos.append(self.no_char); slots.append(j)
            if j < len(text) - 1:
                conn = [s for s, c in zip(strokes, schar) if c == -1]
                join_flags.append(1.0 if conn else 0.0)
                payload = _fit(norm(conn[0])) if conn else np.zeros(PAYLOAD, np.float32)
                tokens.append(payload); kinds.append(KIND_CONNECTOR); cpos.append(self.no_char); slots.append(j)
            if len(tokens) >= self.max_tokens:
                break

        T = len(tokens)
        tok = torch.from_numpy(np.asarray(tokens, np.float32))
        kind = torch.tensor(kinds, dtype=torch.long)
        cp = torch.tensor(cpos, dtype=torch.long)
        slot = torch.tensor(slots, dtype=torch.long)
        mask = torch.ones(T)
        n_conn = len(join_flags)
        join = torch.zeros(self.max_chars)
        if n_conn:
            join[:n_conn] = torch.tensor(join_flags, dtype=torch.float32)
        refs, rmask = self._refs(int(r["writer"]), i)
        img = self._full_image(i)
        target = torch.ones(1, self.img_h, self.canvas_w)
        if img is not None:
            target[0] = torch.from_numpy(img)
        return {"tokens": tok, "kind": kind, "char_pos": cp, "char_slot": slot, "mask": mask,
                "join": join, "n_conn": torch.tensor(n_conn, dtype=torch.long),
                "refs": torch.from_numpy(refs), "ref_mask": torch.from_numpy(rmask),
                "target_img": target, "text": text,
                "writer_idx": torch.tensor(self.writer2idx[int(r["writer"])], dtype=torch.long),
                "char_ids": torch.tensor([self.char2idx.get(c, self.no_char) for c in text],
                                         dtype=torch.long)}


def collate_lines(batch: list[dict]) -> dict:
    T = max(b["tokens"].shape[0] for b in batch)
    tokens, kinds, cpos, slots, masks = [], [], [], [], []
    for b in batch:
        t = b["tokens"]
        pad = T - t.shape[0]
        tokens.append(torch.cat([t, torch.zeros(pad, PAYLOAD)]))
        kinds.append(torch.cat([b["kind"], torch.zeros(pad, dtype=torch.long)]))
        cpos.append(torch.cat([b["char_pos"], torch.zeros(pad, dtype=torch.long)]))
        slots.append(torch.cat([b["char_slot"], torch.zeros(pad, dtype=torch.long)]))
        masks.append(torch.cat([b["mask"], torch.zeros(pad)]))
    n_chars = max(b["char_ids"].shape[0] for b in batch)
    char_ids = torch.full((len(batch), n_chars), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        char_ids[i, : b["char_ids"].shape[0]] = b["char_ids"]
    return {
        "tokens": torch.stack(tokens), "kind": torch.stack(kinds),
        "char_pos": torch.stack(cpos), "char_slot": torch.stack(slots),
        "mask": torch.stack(masks),
        "join": torch.stack([b["join"] for b in batch]),
        "n_conn": torch.stack([b["n_conn"] for b in batch]),
        "refs": torch.stack([b["refs"] for b in batch]),
        "ref_mask": torch.stack([b["ref_mask"] for b in batch]),
        "target_img": torch.stack([b["target_img"] for b in batch]),
        "writer_idx": torch.stack([b["writer_idx"] for b in batch]),
        "char_ids": char_ids,
        "text": [b["text"] for b in batch],
    }
