"""Canonical **line-level** dataset: one record per handwritten text line.

Layout written by `build_lines`:

    <out>/meta.json
    <out>/<split>/index.jsonl        one JSON record per line
    <out>/<split>/images/<id>.png    uint8 L, 0 = ink, 255 = paper
    <out>/<split>/trajectories.npz   only when online lines were supplied:
        points           float32 [P,2]   per-stroke concatenated, page coordinates
        stroke_off       int32   [S+1]   slice into points, one entry per stroke
        line_stroke_off  int32   [R+1]   slice into strokes, one entry per record
        stroke_char      int32   [S]     character index, -1 = connector (inferred)

index.jsonl record:
    {"id","writer","page","line","text","image","modality","paired",
     "n_strokes","n_connectors","char_strokes","has_strokes"}

Two facts this module must not blur:
  * Images come from `.dgrl`, trajectories from `.wptt`. They are the same writers and
    texts but different artefacts, so a record is `paired` only when (writer, page, line
    index) AND the transcript agree.
  * The per-character stroke split is INFERRED (`casia_lines.infer_char_strokes`), never
    read from the corpus. `stroke_char == -1` means "treated as a connector" — our
    definition, not the supplier's.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .casia_lines import infer_char_strokes, iter_dgrl, iter_wptt
from .canonical import writer_splits

IMG_H = 64
IMG_W_MAX = 1024


def _writer_of(stem: str) -> int:
    m = re.match(r"(\d+)", stem)
    if not m:
        raise ValueError(f"cannot read a writer id from {stem!r}")
    return int(m.group(1))


def _resize_h(img: np.ndarray, h: int = IMG_H) -> np.ndarray:
    if img.shape[0] == h:
        return img
    rows = (np.arange(h) * img.shape[0] / h).astype(np.int32).clip(0, img.shape[0] - 1)
    cols = (np.arange(max(1, int(img.shape[1] * h / img.shape[0]))) * img.shape[0] / h).astype(np.int32)
    cols = cols.clip(0, img.shape[1] - 1)
    return img[rows][:, cols]


def _pad_w(img: np.ndarray, w_max: int) -> np.ndarray:
    """Pad a line to a fixed width with paper. NEVER crop.

    Cropping cuts characters off the end of a line while the transcript still lists
    them, which silently corrupts every content metric computed from that pair. If a
    line is wider than the canvas, raise the canvas.
    """
    if img.shape[1] > w_max:
        raise ValueError(
            f"line is {img.shape[1]}px wide, above the {w_max}px canvas. Raise the canvas "
            "rather than cropping: a cropped line no longer matches its transcript."
        )
    if img.shape[1] == w_max:
        return img
    return np.pad(img, ((0, 0), (0, w_max - img.shape[1])), constant_values=255)


def _save_png(path: Path, img: np.ndarray) -> None:
    import struct
    import zlib

    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9))
                     + chunk(b"IEND", b""))


def _read_png(path: Path) -> np.ndarray:
    import struct
    import zlib

    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(raw):
        (ln,) = struct.unpack(">I", raw[pos:pos + 4])
        tag = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if tag == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    buf = zlib.decompress(idat)
    out = np.zeros((h, w), np.uint8)
    stride = w + 1
    for r in range(h):
        row = buf[r * stride:(r + 1) * stride]
        if row[0] != 0:
            raise ValueError(f"unsupported PNG filter {row[0]} in {path}")
        out[r] = np.frombuffer(row[1:], np.uint8)
    return out


def build_lines(dgrl_dir: str | Path | None, wptt_dir: str | Path | None,
                out_dir: str | Path, lexicon: dict[str, int] | None = None,
                img_height: int = IMG_H, max_pages: int | None = None,
                seed: int = 20261121) -> dict:
    """Convert `.dgrl` / `.wptt` pages into the canonical line layout."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    writers: set[int] = set()
    charset: set[str] = set()
    online: dict[tuple[int, str, int], dict] = {}

    # ---- offline: image + transcript ----
    if dgrl_dir:
        pages = sorted(Path(dgrl_dir).glob("*.dgrl"))
        if max_pages:
            pages = pages[:max_pages]
        for page_path in pages:
            wid = _writer_of(page_path.stem)
            writers.add(wid)
            for ln in iter_dgrl(page_path):
                img = _pad_w(_resize_h(ln.bitmap, img_height), IMG_W_MAX)
                rid = f"{page_path.stem}-{ln.index:03d}"
                records.append({
                    "id": rid, "writer": wid, "page": page_path.stem, "line": ln.index,
                    "text": ln.text, "image": f"{rid}.png", "modality": "offline",
                    "paired": False, "n_strokes": 0, "n_connectors": 0,
                    "char_strokes": [], "has_strokes": False,
                    "_img": img, "_strokes": [], "_stroke_char": [],
                })
                charset.update(ln.text)

    # ---- online: trajectories + transcript ----
    if wptt_dir:
        pages = sorted(Path(wptt_dir).glob("*.wptt"))
        if max_pages:
            pages = pages[:max_pages]
        for page_path in pages:
            wid = _writer_of(page_path.stem)
            writers.add(wid)
            page = iter_wptt(page_path)
            for ln in page.lines:
                counts = None
                if lexicon and ln.text:
                    cc = [lexicon.get(c, -1) for c in ln.text]
                    counts = cc if all(c > 0 for c in cc) else None
                groups, connectors = infer_char_strokes(ln, lexicon_counts=counts)
                char_of = {}
                for ci, grp in enumerate(groups):
                    for si in grp:
                        char_of[si] = ci
                for si in connectors:
                    char_of[si] = -1
                strokes, schar = [], []
                for si in ln.stroke_indices:
                    if si < len(page.strokes):
                        strokes.append(page.strokes[si].points)
                        schar.append(char_of.get(si, -1))
                online[(wid, page_path.stem, ln.index)] = {
                    "text": ln.text, "strokes": strokes, "stroke_char": schar,
                    "n_connectors": len(connectors),
                    "char_strokes": [[int(g[0]), int(g[-1])] for g in groups if g],
                }
                charset.update(ln.text)

    # ---- join online onto offline where (writer, page, line, text) agree ----
    matched: set[tuple[int, str, int]] = set()
    for r in records:
        key = (r["writer"], r["page"], r["line"])
        on = online.get(key)
        if on is None or on["text"] != r["text"]:
            continue
        matched.add(key)
        r["paired"] = True
        r["modality"] = "paired"
        r["n_strokes"] = len(on["strokes"])
        r["n_connectors"] = on["n_connectors"]
        r["char_strokes"] = on["char_strokes"]
        r["has_strokes"] = bool(on["strokes"])
        r["_strokes"] = on["strokes"]
        r["_stroke_char"] = on["stroke_char"]

    # ---- online-only lines become records too (usable for training, no image) ----
    for key, on in online.items():
        if key in matched:
            continue
        wid, page, li = key
        records.append({
            "id": f"{page}-{li:03d}-on", "writer": wid, "page": page, "line": li,
            "text": on["text"], "image": None, "modality": "online",
            "paired": False, "n_strokes": len(on["strokes"]),
            "n_connectors": on["n_connectors"], "char_strokes": on["char_strokes"],
            "has_strokes": bool(on["strokes"]), "_img": None,
            "_strokes": on["strokes"], "_stroke_char": on["stroke_char"],
        })

    if not records:
        raise RuntimeError(
            f"no lines parsed: dgrl={dgrl_dir!r} wptt={wptt_dir!r}. Run "
            "`python -m common.probe_lines dir <dir> --kind dgrl|wptt` first."
        )

    splits = writer_splits(sorted(writers), seed=seed)
    where = {w: s for s, ids in splits.items() for w in ids}
    stats = {s: 0 for s in splits}
    for s in splits:
        (out / s / "images").mkdir(parents=True, exist_ok=True)

    for s in splits:
        recs = [r for r in records if where.get(r["writer"]) == s]
        idx, pts, stroke_off, line_off, schar = [], [], [0], [0], []
        for r in recs:
            if r.get("_img") is not None:
                _save_png(out / s / "images" / r["image"], r["_img"])
            n = 0
            for st in r.get("_strokes", []):
                pts.append(np.asarray(st, np.float32).reshape(-1, 2))
                stroke_off.append(stroke_off[-1] + len(pts[-1]))
                n += 1
            for c in r.get("_stroke_char", [])[:n]:
                schar.append(int(c))
            line_off.append(line_off[-1] + n)
            idx.append({k: v for k, v in r.items() if not k.startswith("_")})

        (out / s / "index.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in idx) + ("\n" if idx else ""),
            encoding="utf-8")
        if pts:
            np.savez_compressed(
                out / s / "trajectories.npz",
                points=np.concatenate(pts).astype(np.float32),
                stroke_off=np.asarray(stroke_off, np.int32),
                line_stroke_off=np.asarray(line_off, np.int32),
                stroke_char=np.asarray(schar, np.int32),
            )
        stats[s] = len(idx)

    meta = {
        "format": "cent-lines/1.0",
        "img_height": img_height,
        "num_classes": len(charset),
        "charset": sorted(charset),
        "writers_total": len(writers),
        "splits": {k: sorted(v) for k, v in splits.items()},
        "counts": stats,
        "paired_lines": sum(1 for r in records if r.get("paired")),
        "lines_with_strokes": sum(1 for r in records if r.get("has_strokes")),
        "dgrl_dir": str(dgrl_dir) if dgrl_dir else None,
        "wptt_dir": str(wptt_dir) if wptt_dir else None,
        "lexicon_supplied": lexicon is not None,
        "note": "Images are uint8 with 0 = ink, 255 = paper. Per-character stroke splits are "
                "INFERRED (casia_lines.infer_char_strokes), not read from the corpus.",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


class LineDataset:
    """Reads the canonical line layout: record + image + optional per-line strokes."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.charset = self.meta["charset"]
        self.char2idx = {c: i for i, c in enumerate(self.charset)}
        d = self.root / split
        self.records = [json.loads(l) for l in
                        (d / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.img_h = self.meta["img_height"]
        tp = d / "trajectories.npz"
        self.traj = dict(np.load(tp)) if tp.exists() else None

    def __len__(self) -> int:
        return len(self.records)

    def strokes_of(self, i: int) -> tuple[list[np.ndarray], list[int]]:
        """Per-line strokes and their character index (-1 = connector)."""
        if self.traj is None:
            return [], []
        a = int(self.traj["line_stroke_off"][i])
        b = int(self.traj["line_stroke_off"][i + 1])
        out, chars = [], []
        for s in range(a, b):
            p0, p1 = int(self.traj["stroke_off"][s]), int(self.traj["stroke_off"][s + 1])
            out.append(self.traj["points"][p0:p1])
            chars.append(int(self.traj["stroke_char"][s]))
        return out, chars

    def image_of(self, i: int) -> np.ndarray | None:
        r = self.records[i]
        if not r.get("image"):
            return None
        p = self.root / self.split / "images" / r["image"]
        return _read_png(p) if p.exists() else None

    def by_writer(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i, r in enumerate(self.records):
            out.setdefault(int(r["writer"]), []).append(i)
        return out

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        strokes, schar = self.strokes_of(i)
        return {"record": r, "image": self.image_of(i), "text": r["text"],
                "writer": int(r["writer"]), "strokes": strokes, "stroke_char": schar}
