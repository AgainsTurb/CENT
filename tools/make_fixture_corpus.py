"""Build a tiny synthetic canonical corpus so CENT v3 can be exercised without the real data.

The real corpus lives on the Drive backup (``gd:CENT/datasets``) and is ~391 MB. This fixture
writes the same on-disk layout (``meta.json`` + ``<split>/index.jsonl`` + ``<split>/trajectories.npz``)
with a handful of hand-drawn polylines, so every code path in the v3 dataset/model/trainer can
be executed on CPU. It is a plumbing fixture, never a training set.

    python tools/make_fixture_corpus.py --out data/_fixture/v1 --lines 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHARS = list("永和中人大小上下天日月明")


def _char_strokes(rng: np.random.Generator, n_strokes: int, x0: float, y0: float,
                  w: float, h: float) -> list[np.ndarray]:
    """Axis-aligned squiggles, one polylines per stroke, inside a character box."""
    out = []
    for _ in range(n_strokes):
        ax, ay = x0 + rng.uniform(0.05, 0.7) * w, y0 + rng.uniform(0.05, 0.7) * h
        bx, by = ax + rng.uniform(0.1, 0.5) * w, ay + rng.uniform(0.1, 0.5) * h
        t = np.linspace(0, 1, int(rng.integers(4, 9)), dtype=np.float32)[:, None]
        pts = np.concatenate([ax + (bx - ax) * t, ay + (by - ay) * t + 0.1 * h * np.sin(t * 3)], 1)
        out.append(pts.astype(np.float32))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/_fixture/v1")
    ap.add_argument("--lines", type=int, default=6)
    ap.add_argument("--writers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.out)
    rng = np.random.default_rng(a.seed)
    points, stroke_off, line_stroke_off = [], [0], [0]

    for split in ("train", "val"):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)

    for split, n_lines in (("train", a.lines), ("val", max(2, a.lines // 3))):
        recs = []
        for li in range(n_lines):
            writer = int(rng.integers(0, a.writers))
            n_chars = int(rng.integers(2, 5))
            text = "".join(CHARS[int(rng.integers(0, len(CHARS)))] for _ in range(n_chars))
            strokes, counts = [], []
            for ci in range(n_chars):
                k = int(rng.integers(2, 5))
                counts.append(k)
                strokes += _char_strokes(rng, k, 100.0 + ci * 300.0, 40.0, 240.0, 140.0)
            for s in strokes:
                points.append(s)
                stroke_off.append(stroke_off[-1] + len(s))
            line_stroke_off.append(line_stroke_off[-1] + len(strokes))
            recs.append({
                "id": f"{split}-{li:03d}", "writer": writer, "page": f"p{li//3}",
                "line": li, "text": text, "image": None, "modality": "online",
                "paired": False, "n_strokes": len(strokes), "n_connectors": 0,
                "char_strokes": counts, "has_strokes": True,
            })
        (root / split / "index.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        # points must be a FLAT [P,2] array: strokes are delimited by stroke_off, not by shape,
        # so concatenate it explicitly (asarray + reshape would stack the variable-length rows)
        np.savez(root / split / "trajectories.npz",
                 points=np.concatenate(points).astype(np.float32),
                 stroke_off=np.asarray(stroke_off, np.int64),
                 line_stroke_off=np.asarray(line_stroke_off, np.int64))

    (root / "meta.json").write_text(json.dumps(
        {"format": "cent-lines/1.0", "img_height": 64, "num_classes": len(CHARS),
         "charset": CHARS}, ensure_ascii=False), encoding="utf-8")
    print(f"fixture -> {root}  (charset {len(CHARS)}, {a.lines} train lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
