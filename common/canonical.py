"""Canonical on-disk format + Dataset for the CASIA corpora.

Why a canonical layer at all: six baselines each expect a different input shape
(IAM word images + a 26-letter alphabet, a per-writer pickle of PIL images, ...).
Converting CASIA once into one documented layout, then writing one thin adapter
per baseline, keeps the dataset logic in a single tested place instead of six.

Layout written by `build`:

    <out>/meta.json                 charset, writers, splits, img size, counts
    <out>/<split>/images.npy        uint8  [N, S, S]   0 = ink, 255 = paper (glyph convention)
    <out>/<split>/labels.npy        int32  [N]         index into meta.charset
    <out>/<split>/writers.npy       int32  [N]         writer id (from file name)
    <out>/<split>/source.npy        int32  [N]         0 = offline image, 1 = online
    <out>/<split>/online.npz        only for online samples:
        n_strokes      int16 [M]
        stroke_starts  int32 [M+1]   index into the stroke table
        pt_starts      int32 [K+1]   index into `points`, per stroke
        points         float32 [P,2] image coordinates (y down)

`images.npy` for an online sample is rendered from its strokes by
`rasterise_strokes`, so a single baseline adapter can read either corpus.

The build is deterministic: writers are ordered, splits are a pure function of
the sorted writer list and the seed, and nothing is shuffled at build time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .casia import iter_gnt, iter_pot

IMG_SIZE = 64
DEFAULT_SEED = 20261121
VAL_WRITERS = 60


# --------------------------------------------------------------------------- #
# image handling
# --------------------------------------------------------------------------- #
def crop_to_ink(img: np.ndarray, ink_is_dark: bool = True) -> np.ndarray:
    """Crop to the tight ink bounding box. Returns the input if there is no ink."""
    ink = (img < 128) if ink_is_dark else (img >= 128)
    if not ink.any():
        return img
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    return img[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def fit_square(img: np.ndarray, size: int = IMG_SIZE, ink_is_dark: bool = True) -> np.ndarray:
    """Crop, pad to square preserving aspect, resize to `size`.

    Returns uint8 with **0 = ink, 255 = paper**, matching the corpus bitmap
    convention (a .gnt stores 255 as background). `CanonicalDataset` inverts this
    to the float contract 1 = ink; the arrays on disk are NOT inverted.
    """
    img = crop_to_ink(img, ink_is_dark)
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side), 255 if ink_is_dark else 0, np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top: top + h, left: left + w] = img
    # nearest-neighbour resize keeps the glyph crisp and avoids a cv2 dependency
    yi = (np.arange(size) * side / size).astype(np.int32).clip(0, side - 1)
    out = canvas[yi][:, yi]
    return out if ink_is_dark else 255 - out


def rasterise_strokes(strokes: Sequence[np.ndarray], size: int = IMG_SIZE, width: int = 2) -> np.ndarray:
    """Draw polylines into a uint8 image in normalised [0,1] space.

    Same convention as `fit_square`: 0 = ink, 255 = paper.
    """
    pts_all = np.concatenate([s for s in strokes if len(s)], axis=0) if strokes else None
    canvas = np.full((size, size), 255, np.uint8)
    if pts_all is None or len(pts_all) == 0:
        return canvas
    lo = pts_all.min(axis=0)
    hi = pts_all.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (size - 2 * width - 1) / span.max()

    def to_px(s: np.ndarray) -> np.ndarray:
        p = (s - lo) * scale + width
        return p

    for s in strokes:
        if len(s) < 2:
            continue
        p = to_px(s)
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            xs = np.linspace(x0, x1, n).round().astype(int)
            ys = np.linspace(y0, y1, n).round().astype(int)
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    xx = (xs + dx).clip(0, size - 1)
                    yy = (ys + dy).clip(0, size - 1)
                    canvas[yy, xx] = 0
    return canvas


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def writer_splits(writers: Sequence[int], seed: int = DEFAULT_SEED,
                  val_n: int = VAL_WRITERS) -> dict[str, list[int]]:
    """Writer-disjoint 4:1 train/test with `val_n` writers carved out of train.

    Deterministic in (writers, seed, val_n): the writer list is sorted first, so
    the result does not depend on file enumeration order.
    """
    ws = sorted(int(w) for w in writers)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ws))
    n_test = max(1, len(ws) // 5)
    test = [ws[i] for i in perm[:n_test]]
    train_pool = [ws[i] for i in perm[n_test:]]
    val = train_pool[:min(val_n, max(1, len(train_pool) // 10))]
    train = train_pool[len(val):]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _writer_id(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    if not m:
        raise ValueError(f"cannot read a writer id from {path.name}")
    return int(m.group(1))


def build(
    offline_dir: str | Path | None,
    online_dir: str | Path | None,
    out_dir: str | Path,
    charset: Sequence[str] | None = None,
    img_size: int = IMG_SIZE,
    limit_per_writer: int | None = None,
    seed: int = DEFAULT_SEED,
    pot_label_bytes: int = 4,
    lexicon: dict[str, int] | None = None,
) -> dict:
    """Convert .gnt/.pot files into the canonical layout. Returns the meta dict.

    When `lexicon` ({char: canonical stroke count}) is supplied, each split also
    gets `lexicon_strokes.npy` (canonical count per sample, -1 if unknown) and the
    meta records the agreement rate between observed and canonical counts. That
    rate is the paper's threat T2 ("do real writers conform to canonical
    topology?") and it is measured here rather than asserted.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: collect characters (so the charset is stable before writing) --
    observed: set[str] = set()
    n_scanned = 0
    for d, reader in ((offline_dir, "gnt"), (online_dir, "pot")):
        if not d:
            continue
        for f in sorted(Path(d).glob("*.gnt" if reader == "gnt" else "*.pot")):
            recs = iter_gnt(f) if reader == "gnt" else iter_pot(f, label_bytes=pot_label_bytes)
            n = 0
            for r in recs:
                n += 1
                n_scanned += 1
                if r.char:
                    observed.add(r.char)
                if limit_per_writer and n >= limit_per_writer:
                    break
    if charset is None:
        charset = sorted(observed)
    charset = list(charset)
    ch2i = {c: i for i, c in enumerate(charset)}

    # ---- pass 2: emit per-split arrays --------------------------------------
    raw: dict[str, list] = {"img": [], "lab": [], "wid": [], "src": [], "online": []}
    skipped_unknown_char = 0
    writers_seen: set[int] = set()

    for d, kind in ((offline_dir, "gnt"), (online_dir, "pot")):
        if not d:
            continue
        for f in sorted(Path(d).glob("*.gnt" if kind == "gnt" else "*.pot")):
            wid = _writer_id(f)
            writers_seen.add(wid)
            recs = iter_gnt(f) if kind == "gnt" else iter_pot(f, label_bytes=pot_label_bytes)
            n = 0
            for r in recs:
                n += 1
                if limit_per_writer and n > limit_per_writer:
                    break
                if not r.char or r.char not in ch2i:
                    skipped_unknown_char += 1
                    continue
                if kind == "gnt":
                    img = fit_square(r.bitmap, img_size)
                    online = None
                else:
                    img = rasterise_strokes(r.strokes, img_size)
                    online = r.strokes
                # The split depends on the full writer set, so samples are staged
                # first and partitioned after all writers are known.
                raw["img"].append(img)
                raw["lab"].append(ch2i[r.char])
                raw["wid"].append(wid)
                raw["src"].append(0 if kind == "gnt" else 1)
                raw["online"].append(online)

    if not raw["wid"]:
        raise RuntimeError(
            "no samples parsed. Check, in this order: (1) the directories exist and contain "
            f"*.gnt/*.pot files -- offline_dir={offline_dir!r}, online_dir={online_dir!r}; "
            f"(2) `python -m common.probe dir <dir> --kind gnt|pot` reports ok; "
            f"(3) the charset contains the decoded characters -- {skipped_unknown_char} records "
            f"were skipped as unknown, from {n_scanned} scanned. "
            f"First unknown examples: {sorted(observed)[:10] if observed else 'none observed'}"
        )

    splits = writer_splits(sorted(writers_seen), seed=seed)
    where = {}
    for name, ids in splits.items():
        for w in ids:
            where[w] = name

    stats = {s: 0 for s in splits}
    meta_chars = charset
    for s in splits:
        (out / s).mkdir(parents=True, exist_ok=True)

    # group indices by split
    idx_by_split: dict[str, list[int]] = {s: [] for s in splits}
    for i, w in enumerate(raw["wid"]):
        idx_by_split[where.get(w, "train")].append(i)

    agreement = {"n_paired": 0, "n_agree": 0, "rate": None, "mean_abs_diff": None}
    for s, idxs in idx_by_split.items():
        imgs = np.stack([raw["img"][i] for i in idxs]) if idxs else np.zeros((0, img_size, img_size), np.uint8)
        np.save(out / s / "images.npy", imgs)
        labs = np.asarray([raw["lab"][i] for i in idxs], np.int32)
        np.save(out / s / "labels.npy", labs)
        np.save(out / s / "writers.npy", np.asarray([raw["wid"][i] for i in idxs], np.int32))
        np.save(out / s / "source.npy", np.asarray([raw["src"][i] for i in idxs], np.int32))
        stats[s] = len(idxs)

        # Canonical stroke count per sample (-1 when the character is not in the
        # lexicon) and the count the writer actually used (-1 for offline samples,
        # which carry no trajectory). Their difference is the T2 statistic.
        if lexicon is not None:
            lex = np.asarray([lexicon.get(charset[lab], -1) for lab in labs], np.int16)
            obs = np.asarray(
                [-1 if raw["online"][i] is None else len(raw["online"][i]) for i in idxs],
                np.int16,
            )
            np.save(out / s / "lexicon_strokes.npy", lex)
            np.save(out / s / "observed_strokes.npy", obs)
            both = (lex > 0) & (obs > 0)
            if both.any():
                agree = (lex[both] == obs[both])
                agreement["n_paired"] += int(both.sum())
                agreement["n_agree"] += int(agree.sum())
                prev_n = agreement["n_paired"] - int(both.sum())
                prev_sum = (agreement["mean_abs_diff"] or 0.0) * prev_n
                agreement["mean_abs_diff"] = float(
                    (prev_sum + np.abs(lex[both].astype(np.int32) - obs[both]).sum())
                    / agreement["n_paired"]
                )
        on = [raw["online"][i] for i in idxs if raw["online"][i] is not None]
        if on:
            n_strokes = np.asarray([len(o) for o in on], np.int16)
            stroke_starts = np.concatenate([[0], np.cumsum(n_strokes)]).astype(np.int32)
            pt_starts = [0]
            pts = []
            for strokes in on:
                for st in strokes:
                    pts.append(st.reshape(-1, 2).astype(np.float32))
                    pt_starts.append(pt_starts[-1] + len(st))
            np.savez_compressed(
                out / s / "online.npz",
                n_strokes=n_strokes,
                stroke_starts=stroke_starts,
                pt_starts=np.asarray(pt_starts, np.int32),
                points=np.concatenate(pts) if pts else np.zeros((0, 2), np.float32),
                index=np.asarray([i for i in range(len(idxs)) if raw["online"][idxs[i]] is not None], np.int32),
            )

    if agreement["n_paired"]:
        agreement["rate"] = round(agreement["n_agree"] / agreement["n_paired"], 4)

    meta = {
        "format": "cent-canonical/1.0",
        "built_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "img_size": img_size,
        "seed": seed,
        "charset": meta_chars,
        "num_classes": len(meta_chars),
        "splits": {s: sorted(v) for s, v in splits.items()},
        "counts": stats,
        "writers_total": len(writers_seen),
        "records_scanned": n_scanned,
        "skipped_unknown_char": skipped_unknown_char,
        "offline_dir": str(offline_dir) if offline_dir else None,
        "online_dir": str(online_dir) if online_dir else None,
        "pot_label_bytes": pot_label_bytes,
        "lexicon": {
            "supplied": lexicon is not None,
            "characters_in_lexicon": (sum(1 for c in meta_chars if c in lexicon) if lexicon else 0),
            "agreement": agreement,
            "meaning": "agreement compares the writer's observed stroke count with the "
                       "lexicon's canonical count on online samples. It is threat T2 "
                       "(rigidity) measured, not assumed.",
        },
        "note": "Images are uint8 with 0 = ink and 255 = paper (the corpus convention). CanonicalDataset returns 1 = ink. Online samples are additionally "
                "rasterised into images so one adapter can read either corpus.",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CanonicalDataset:
    """Memmap-backed reader over the canonical layout.

    Returns dicts with keys: image (float32 [1,S,S] in [0,1], 1 = ink),
    label (int64), writer (int64), with_online and strokes when requested.
    """

    def __init__(self, root: str | Path, split: str = "train", version: str | None = None,
                 limit: int | None = None):
        # `version` is an optional subdirectory, e.g. CanonicalDataset("data/canonical", version="v1").
        # Omitting it reads `root` directly, which is what `build(out_dir=...)` writes.
        self.root = Path(root) / version if version else Path(root)
        self.split = split
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.charset: list[str] = meta["charset"]
        self.img_size: int = meta["img_size"]
        d = self.root / split
        self.images = np.load(d / "images.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy")
        self.writers = np.load(d / "writers.npy")
        self.source = np.load(d / "source.npy")
        if limit:
            self.images, self.labels = self.images[:limit], self.labels[:limit]
            self.writers, self.source = self.writers[:limit], self.source[:limit]
        self._online = None
        if (d / "online.npz").exists():
            self._online = dict(np.load(d / "online.npz"))

    def __len__(self) -> int:
        return len(self.labels)

    def font_map(self) -> dict[str, list[str]]:
        """writer -> list of characters, the shape most baselines actually want."""
        out: dict[str, list[str]] = {}
        for w, l in zip(self.writers.tolist(), self.labels.tolist()):
            out.setdefault(str(w), []).append(self.charset[l])
        return out

    def __getitem__(self, i: int) -> dict:
        img = np.asarray(self.images[i], np.float32)
        item = {
            "image": (1.0 - img / 255.0)[None, :, :],   # 1 = ink
            "label": int(self.labels[i]),
            "writer": int(self.writers[i]),
            "char": self.charset[int(self.labels[i])],
        }
        if self._online is not None:
            on = self._online
            pos = np.flatnonzero(on["index"] == i)
            if len(pos):
                k = int(pos[0])
                s0, s1 = on["stroke_starts"][k], on["stroke_starts"][k + 1]
                strokes = []
                for si in range(int(s0), int(s1)):
                    a, b = on["pt_starts"][si], on["pt_starts"][si + 1]
                    strokes.append(on["points"][a:b])
                item["strokes"] = strokes
                item["with_online"] = True
        return item
