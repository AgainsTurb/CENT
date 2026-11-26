"""Processed cache + torch datasets for the CENT stages.

Raw WPTT pages are parsed once into a compact ``lines.npz`` per split, with
writer-disjoint train/val carved out of the released train pages. Both the VAE
stages and the boundary/diffusion stage read from this cache.

Pen states before alignment
---------------------------
Without point-level character ids (which arrive only in Stage 3) we can only
distinguish ``PEN_MOVE`` from ``PEN_UP``. That is deliberate: the VAE stage
learns stroke structure, and the character-boundary states (``EOC`` /
``CURSIVE_EOC``) become available — and are supervised — in Stage 4 once the
alignment exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.format_wptt import read_wptt
from ..data.tokens import CharVocab, PEN_MOVE, PEN_UP


# ---------------------------------------------------------------- build


def _pages_for(split_dir: Path, writers: set[str] | None, limit: int | None):
    pages = sorted(split_dir.glob("*.wptt"))
    if writers is not None:
        pages = [p for p in pages if p.stem.split("-")[0] in writers]
    if limit:
        pages = pages[:limit]
    return pages


def build_cache(root: Path = Path("datasets"), *, val_frac: float = 0.2,
                seed: int = 0, limit_pages: int | None = None,
                force: bool = False, log=print) -> dict:
    """Parse WPTT pages into ``datasets/processed/{train,val,test}/lines.npz``."""
    root = Path(root)
    src = root / "CASIA-OLHWDB2.0"
    out_root = root / "processed"
    meta_path = out_root / "meta.json"

    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if limit_pages is None or meta.get("limit_pages") == limit_pages:
            log(f"[cache] reusing {out_root} ({meta['counts']})")
            return meta

    if not (src / "train").exists():
        raise RuntimeError(f"missing {src/'train'} — run cent.pipeline.fetch first")

    train_writers = sorted({p.stem.split("-")[0] for p in (src / "train").glob("*.wptt")})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_writers))
    n_val = max(1, int(round(val_frac * len(train_writers))))
    val_set = {train_writers[i] for i in perm[:n_val]}
    trn_set = set(train_writers) - val_set

    splits = {
        "train": _pages_for(src / "train", trn_set, limit_pages),
        "val": _pages_for(src / "train", val_set, max(1, (limit_pages or 400) // 5)),
        "test": _pages_for(src / "test", None, limit_pages),
    }

    meta = {"counts": {}, "writers": {}, "seed": seed, "val_frac": val_frac,
            "limit_pages": limit_pages}
    for name, pages in splits.items():
        pts, sid, off, texts, wr, pg = [], [], [0], [], [], []
        for pi, path in enumerate(pages):
            try:
                page = read_wptt(path)
            except Exception as e:  # noqa: BLE001
                log(f"[cache] skip {path.name}: {e}")
                continue
            for li, (idx, text) in enumerate(zip(page.line_stroke_idx, page.line_text)):
                text = (text or "").strip()
                if len(text) < 2 or not idx:
                    continue
                strokes = [page.strokes[i] for i in idx if 0 <= i < len(page.strokes)]
                keep = [s for s in strokes if len(s) > 0]
                if len(keep) < 2:
                    continue
                local = []
                for si, s in enumerate(keep):
                    local.append(np.full(len(s), si, dtype=np.int32))
                arr = np.concatenate(keep, axis=0).astype(np.float32)
                pts.append(arr)
                sid.append(np.concatenate(local))
                off.append(off[-1] + len(arr))
                texts.append(text)
                wr.append(int(path.stem.split("-")[0]))
                pg.append(pi)

        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "lines.npz",
            points=np.concatenate(pts) if pts else np.zeros((0, 2), np.float32),
            stroke_ids=np.concatenate(sid) if sid else np.zeros(0, np.int32),
            offsets=np.asarray(off, dtype=np.int64),
            texts=np.asarray(texts, dtype=object),
            writers=np.asarray(wr, dtype=np.int32),
            pages=np.asarray(pg, dtype=np.int32),
        )
        meta["counts"][name] = len(texts)
        meta["writers"][name] = len({int(w) for w in wr})
        log(f"[cache] {name}: {len(texts)} lines from {len(pages)} pages, "
            f"{meta['writers'][name]} writers")

    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- store


class LineStore:
    """Read-only view over one split's cached lines."""

    def __init__(self, root: Path, split: str) -> None:
        d = Path(root) / "processed" / split / "lines.npz"
        if not d.exists():
            raise FileNotFoundError(f"{d} — run build_cache first")
        z = np.load(d, allow_pickle=True)
        self.points = z["points"]
        self.stroke_ids = z["stroke_ids"]
        self.offsets = z["offsets"]
        self.texts = list(z["texts"])
        self.writers = z["writers"]
        self.pages = z["pages"]

    def __len__(self) -> int:
        return len(self.texts)

    def get(self, i: int) -> dict:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return {"points": self.points[lo:hi].astype(np.float32),
                "stroke_ids": self.stroke_ids[lo:hi].astype(np.int64),
                "text": str(self.texts[i]),
                "writer": int(self.writers[i])}


def build_vocab(stores: list[LineStore], min_count: int = 1) -> CharVocab:
    texts = [t for s in stores for t in s.texts]
    return CharVocab.build(texts, min_count=min_count)


# ---------------------------------------------------------------- tensors


def line_to_tensor(line: dict, vocab: CharVocab) -> dict | None:
    pts = line["points"]
    if len(pts) < 4:
        return None
    sid = line["stroke_ids"]
    # per-line height normalisation; keep relative spacing (no per-char rescale)
    p = pts - pts.min(axis=0, keepdims=True)
    h = float(p[:, 1].max() - p[:, 1].min())
    p = p / (h if h > 1e-6 else 1.0)
    dxy = np.zeros_like(p)
    dxy[1:] = p[1:] - p[:-1]
    pen = np.full(len(p), PEN_MOVE, dtype=np.int64)
    pen[1:] = np.where(sid[1:] == sid[:-1], PEN_MOVE, PEN_UP)
    return {"dxy": dxy.astype(np.float32), "pen": pen,
            "char_seq": np.asarray(vocab.encode(line["text"]), dtype=np.int64),
            "text": line["text"], "writer": line["writer"]}


def collate_lines(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["dxy"].shape[0] for b in batch)
    L = max(b["char_seq"].shape[0] for b in batch)
    dxy = torch.zeros(B, T, 2)
    pen = torch.zeros(B, T, dtype=torch.long)
    pmask = torch.zeros(B, T, dtype=torch.bool)
    cs = torch.zeros(B, L, dtype=torch.long)
    cmask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["dxy"].shape[0]
        dxy[i, :t] = torch.from_numpy(b["dxy"])
        pen[i, :t] = torch.from_numpy(b["pen"])
        pmask[i, :t] = True
        l = b["char_seq"].shape[0]
        cs[i, :l] = torch.from_numpy(b["char_seq"])
        cmask[i, :l] = True
    return {"dxy": dxy, "pen": pen, "point_mask": pmask, "char_seq": cs,
            "char_mask": cmask,
            "writers": [int(b["writer"]) for b in batch],
            "texts": [b["text"] for b in batch]}


class LineDataset(Dataset):
    def __init__(self, store: LineStore, vocab: CharVocab, limit: int | None = None):
        self.store, self.vocab = store, vocab
        self.indices = list(range(len(store)))
        if limit:
            self.indices = self.indices[:limit]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> dict:
        i = self.indices[k]
        t = line_to_tensor(self.store.get(i), self.vocab)
        while t is None:            # skip degenerate lines deterministically
            i = (i + 1) % len(self.store)
            t = line_to_tensor(self.store.get(i), self.vocab)
        t["index"] = i
        return t
