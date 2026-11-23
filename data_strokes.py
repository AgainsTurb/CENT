"""Stroke-level line dataset (CENT v2).

The character-decomposed formulation failed on real data: CASIA line corpora contain ~40% fewer
strokes than the lexicon order (measured 96 vs 163 strokes per line, 0.8% of lines admissible), so
per-character stroke groups cannot be identified and a separate connector token has nothing to
supervise it.

This module treats a line as what the corpus actually stores: an ordered sequence of strokes plus a
line transcript. A "join" is then a **geometric** property of consecutive strokes:

    gap(i) = || end(stroke i) - start(stroke i+1) || / stroke_scale
    join(i) = 1  iff  gap(i) < tau

where `stroke_scale` is the median stroke bounding-box diagonal in the line, so the threshold is
scale-free across writers. This is measurable on every line, needs no character localisation, and
matches the quantity the metrics already report (continuity error at a join).

Conditioning is on the whole transcript: the denoiser cross-attends to a text encoding of the line's
characters instead of indexing a per-token character slot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .models.line import KIND_CHAR_STROKE, PAYLOAD
from .models.stroke_graph import fit_stroke, render_strokes_splat

DEFAULT_TAU = 0.35      # in units of stroke_scale; calibrated on real data (see below)


def _fit(a: np.ndarray) -> np.ndarray:
    return fit_stroke(a).reshape(-1)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if (ih, iw) == (h, w):
        return img
    rows = (np.arange(h) * ih / h).astype(np.int32).clip(0, ih - 1)
    cols = (np.arange(w) * iw / w).astype(np.int32).clip(0, iw - 1)
    return img[rows][:, cols]


def stroke_scale(strokes: list[np.ndarray]) -> float:
    """Scale-free unit: the median stroke bounding-box diagonal."""
    diags = []
    for s in strokes:
        if len(s) < 2:
            continue
        span = s.max(0) - s.min(0)
        diags.append(float(np.hypot(*span)))
    diags = [d for d in diags if d > 0]
    return float(np.median(diags)) if diags else 1.0


def join_flags(strokes: list[np.ndarray], tau: float = DEFAULT_TAU) -> np.ndarray:
    """Geometric connection between consecutive strokes, in [0,1]."""
    n = len(strokes)
    if n < 2:
        return np.zeros(0, np.float32)
    sc = max(stroke_scale(strokes), 1e-6)
    out = np.zeros(n - 1, np.float32)
    for i in range(n - 1):
        a, b = strokes[i], strokes[i + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        gap = float(np.hypot(*(a[-1] - b[0]))) / sc
        out[i] = 1.0 if gap < tau else 0.0
    return out


def join_gaps(strokes: list[np.ndarray]) -> np.ndarray:
    """The raw normalised gaps, for calibration and analysis."""
    n = len(strokes)
    if n < 2:
        return np.zeros(0, np.float32)
    sc = max(stroke_scale(strokes), 1e-6)
    return np.array([float(np.hypot(*(strokes[i][-1] - strokes[i + 1][0]))) / sc
                     for i in range(n - 1)], np.float32)


class StrokeLineDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", k_refs: int = 5,
                 img_h: int = 64, canvas_w: int = 1024, max_strokes: int = 512,
                 tau: float = DEFAULT_TAU, paired_only: bool = False, limit: int = 0,
                 seed: int = 0, render_missing: bool = True, max_text: int = 64):
        self.root = Path(root)
        self.split = split
        self.k_refs, self.img_h, self.canvas_w = k_refs, img_h, canvas_w
        self.max_strokes, self.tau, self.max_text = max_strokes, tau, max_text
        self.render_missing = render_missing
        self.rng = np.random.default_rng(seed)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.charset = meta["charset"]
        self.char2idx = {c: i for i, c in enumerate(self.charset)}
        self.pad_id = len(self.charset)
        d = self.root / split
        self.records = [json.loads(l) for l in
                        (d / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.traj = dict(np.load(d / "trajectories.npz"))
        writers = sorted({int(r["writer"]) for r in self.records})
        self.writer2idx = {w: i for i, w in enumerate(writers)}

        self.usable, self.n_unpaired, self.n_short = [], 0, 0
        for i, r in enumerate(self.records):
            if not r.get("has_strokes"):
                continue
            if paired_only and not r.get("paired"):
                self.n_unpaired += 1
                continue
            if int(r.get("n_strokes", 0)) < 2:
                self.n_short += 1
                continue
            self.usable.append(i)
        if limit:
            self.usable = self.usable[:limit]
        self.by_writer: dict[int, list[int]] = {}
        for i in self.usable:
            self.by_writer.setdefault(int(self.records[i]["writer"]), []).append(i)

    def __len__(self) -> int:
        return len(self.usable)

    def strokes_of(self, i: int) -> list[np.ndarray]:
        z = self.traj
        a, b = int(z["line_stroke_off"][i]), int(z["line_stroke_off"][i + 1])
        out = []
        for s in range(a, b):
            p0, p1 = int(z["stroke_off"][s]), int(z["stroke_off"][s + 1])
            out.append(np.asarray(z["points"][p0:p1], np.float32))
        return out

    def _image(self, i: int) -> np.ndarray | None:
        r = self.records[i]
        if not r.get("image"):
            return None
        p = self.root / self.split / "images" / r["image"]
        if not p.exists():
            return None
        from .common_bridge import read_png
        img = read_png(p).astype(np.float32) / 255.0
        # FRAME FIX. Canonical images are padded strips: measured, the ink of a line occupied cols
        # 1-268 of a 1024-wide canvas, while the trajectory is normalised to fill the whole width.
        # The two were in different frames, IoU(render, scan) was 0.003-0.009, and the image loss
        # could never fall -- it sat at 0.60 in eight runs.
        # Cropping to the ink bbox before resizing puts the scan in the same frame as the strokes.
        ink = img < 0.5
        if bool(ink.any()):
            rows = np.where(ink.any(axis=1))[0]
            cols = np.where(ink.any(axis=0))[0]
            m = 1
            r0, r1 = max(int(rows[0]) - m, 0), min(int(rows[-1]) + m, img.shape[0] - 1)
            c0, c1 = max(int(cols[0]) - m, 0), min(int(cols[-1]) + m, img.shape[1] - 1)
            img = img[r0:r1 + 1, c0:c1 + 1]
        return _resize(img, self.img_h, self.canvas_w)

    def _render(self, strokes: list[np.ndarray]) -> np.ndarray:
        from benchmark.visualise import trajectory_image
        return _resize(trajectory_image(strokes, self.img_h, self.canvas_w).astype(np.float32) / 255.0,
                       self.img_h, self.canvas_w)

    def __getitem__(self, k: int) -> dict:
        i = self.usable[k]
        r = self.records[i]
        strokes = self.strokes_of(i)[: self.max_strokes]
        text = r["text"]

        # uniform-scale normalisation into the canvas frame (aspect preserved)
        allp = np.concatenate(strokes)
        lo, hi = allp.min(0), allp.max(0)
        span = np.maximum(hi - lo, 1e-6)
        scale = min(self.canvas_w / span[0], self.img_h / span[1])

        def norm(a):
            q = (np.asarray(a, np.float32) - lo) * scale
            return q / np.array([self.canvas_w, self.img_h], np.float32)

        payloads, mask = [], []
        for s in strokes:
            if len(s) < 2:
                payloads.append(np.zeros(PAYLOAD, np.float32)); mask.append(0.0)
            else:
                payloads.append(_fit(norm(s))); mask.append(1.0)
        tok = torch.from_numpy(np.asarray(payloads, np.float32))
        join = torch.from_numpy(join_flags(strokes, self.tau))
        text_ids = torch.tensor([self.char2idx.get(c, self.pad_id)
                                 for c in text[: self.max_text]], dtype=torch.long)

        # TARGET IMAGE = a render of THIS line's own strokes.
        # The .dgrl scan and the .wptt trajectory are two SEPARATE writing sessions: the offline
        # corpus was written with an ordinary pen and scanned, the online corpus with an Anoto pen
        # on dot-pattern paper. They are paired by writer and text, not by strokes, so a scan is not
        # the image of this trajectory. Measured: cropping to the ink bbox (removing any frame
        # offset) left IoU(render, scan) at 0.008-0.013 -- the correspondence simply does not exist.
        # Using the scan as a pixel target therefore supervises the model towards something it never
        # drew, which is why the image loss never fell below ~0.60 in nine runs.
        # Scans remain useful as STYLE references (writer identity does transfer across renditions);
        # as pixel targets they are wrong.
        # The target must be rendered by the SAME rasteriser the loss uses, otherwise the loss
        # compares two different drawings: the numpy polyline plotter produced ink fraction 0.0124
        # while the splat renderer on the fitted payloads produced 0.0014, giving IoU 0.008 even for
        # a perfect reconstruction. Rendering the target through render_strokes_splat on the very
        # payloads the model regresses makes the target self-consistent by construction.
        if self.render_missing:
            _tk = torch.from_numpy(np.asarray(payloads, np.float32))[None]
            _mk = torch.from_numpy(np.asarray(mask, np.float32))[None]
            img = 1.0 - render_strokes_splat(_tk, _mk, self.img_h, self.canvas_w)[0, 0].numpy()
        else:
            img = np.ones((self.img_h, self.canvas_w), np.float32)

        n_st = len(payloads)
        pos = torch.from_numpy((np.arange(n_st) / max(n_st - 1, 1)).astype(np.float32))
        scalars = torch.tensor([n_st / float(self.max_strokes),
                                len(text) / float(self.max_text)], dtype=torch.float32)
        refs, rmask = self._refs(int(r["writer"]), i)
        return {"tokens": tok, "mask": torch.from_numpy(np.asarray(mask, np.float32)),
                "join": join, "text": text, "text_ids": text_ids,
                "target_img": torch.from_numpy(img)[None].float(),
                "refs": torch.from_numpy(refs), "ref_mask": torch.from_numpy(rmask),
                "writer_idx": torch.tensor(self.writer2idx[int(r["writer"])], dtype=torch.long),
                "pos": pos, "scalars": scalars, "pad_id": self.pad_id}

    def _refs(self, writer: int, exclude: int):
        pool = [i for i in self.by_writer.get(writer, []) if i != exclude]
        if not pool:
            pool = self.usable[:1]
        pick = self.rng.choice(len(pool), size=min(self.k_refs, len(pool)), replace=False)
        refs = np.ones((self.k_refs, 1, self.img_h, self.canvas_w), np.float32)
        mask = np.zeros((self.k_refs,), np.float32)
        for j, p in enumerate(np.atleast_1d(pick)[: self.k_refs]):
            img = self._image(int(pool[int(p)]))
            if img is None:
                img = self._render(self.strokes_of(int(pool[int(p)]))[: self.max_strokes])
            refs[j, 0] = img
            mask[j] = 1.0
        return refs, mask


def collate_strokes(batch: list[dict]) -> dict:
    N = max(b["tokens"].shape[0] for b in batch)
    C = max(b["text_ids"].shape[0] for b in batch)
    B = len(batch)
    tok = torch.zeros(B, N, PAYLOAD)
    mask = torch.zeros(B, N)
    join = torch.zeros(B, N)
    pad_id = int(batch[0].get("pad_id", 0))
    text = torch.full((B, C), pad_id, dtype=torch.long)
    pos_t = torch.zeros(B, N)
    scal = torch.zeros(B, 2)
    tmask = torch.zeros(B, C)
    for i, b in enumerate(batch):
        n = b["tokens"].shape[0]
        k = b["pos"].shape[0]
        pos_t[i, :k] = b["pos"]
        scal[i] = b["scalars"]
        tok[i, :n] = b["tokens"]
        mask[i, :n] = b["mask"]
        join[i, : n - 1] = b["join"]
        c = b["text_ids"].shape[0]
        text[i, :c] = b["text_ids"]
        tmask[i, :c] = 1.0
    return {"tokens": tok, "mask": mask, "join": join, "text": text, "text_mask": tmask,
            "pos": pos_t, "scalars": scal,
            "target_img": torch.stack([b["target_img"] for b in batch]),
            "refs": torch.stack([b["refs"] for b in batch]),
            "ref_mask": torch.stack([b["ref_mask"] for b in batch]),
            "writer_idx": torch.stack([b["writer_idx"] for b in batch]),
            "texts": [b["text"] for b in batch]}
