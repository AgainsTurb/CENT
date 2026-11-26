"""Stage 3: CTC Viterbi alignment -> point-level character ids -> join labels.

This is the stage that manufactures the supervision nothing ships: once a
boundary is known *at point level*, the join label is exact by definition —

    join  <=>  stroke_id[t] == stroke_id[t-1]      (the pen never lifted)

where ``t`` is the first point of the next character.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..metrics import join_rate
from .cache import LineStore, line_to_tensor


# ---------------------------------------------------------------- viterbi


def viterbi_ctc(log_probs: np.ndarray, target: list[int], blank: int = 0):
    """Best CTC path. Returns (frame_labels, mean_confidence).

    ``frame_labels[t]`` is the target character index emitted at frame ``t``,
    or ``-1`` for blank. Vectorised over the state axis.
    """
    T, C = log_probs.shape
    L = len(target)
    if L == 0 or T == 0:
        return np.full(T, -1, dtype=np.int32), 0.0

    S = 2 * L + 1
    ext = np.full(S, -1, dtype=np.int64)
    ext[1::2] = np.asarray(target, dtype=np.int64)

    NEG = -1e30
    alpha = np.full(S, NEG, dtype=np.float64)
    alpha[0] = log_probs[0, blank]
    if S > 1:
        alpha[1] = log_probs[0, ext[1]]
    back = np.full((T, S), -1, dtype=np.int32)

    idx = np.arange(S)
    for t in range(1, T):
        stay = alpha
        from1 = np.concatenate([[NEG], alpha[:-1]])
        from2 = np.full(S, NEG)
        if S > 2:
            ok = (ext[2:] >= 0) & (ext[2:] != ext[:-2])
            from2[2:] = np.where(ok, alpha[:-2], NEG)
        cand = np.stack([stay, from1, from2])
        arg = cand.argmax(axis=0)
        best = cand.max(axis=0)
        back[t] = arg
        emit = np.where(ext >= 0, log_probs[t, np.maximum(ext, 0)], log_probs[t, blank])
        alpha = best + emit

    end = [S - 1] + ([S - 2] if S >= 2 else [])
    s = max(end, key=lambda k: alpha[k])
    states = np.empty(T, dtype=np.int32)
    states[T - 1] = s
    for t in range(T - 1, 0, -1):
        states[t - 1] = {
            0: states[t],
            1: states[t] - 1,
            2: states[t] - 2,
        }[int(back[t, states[t]])]

    labels = np.where(states % 2 == 1, (states - 1) // 2, -1).astype(np.int32)
    conf = float(np.mean([np.exp(log_probs[t, states[t]]) for t in range(T)]))
    return labels, conf


def labels_to_points(frame_labels: np.ndarray, n_points: int, stride: int) -> np.ndarray:
    """Expand frame-level character labels to point level."""
    out = np.full(n_points, -1, dtype=np.int32)
    for f, lab in enumerate(frame_labels):
        lo = f * stride
        hi = min(n_points, (f + 1) * stride)
        if lo >= n_points:
            break
        out[lo:hi] = lab
    # forward-fill gaps, then backward-fill the head
    last = -1
    for t in range(n_points):
        if out[t] >= 0:
            last = out[t]
        elif last >= 0:
            out[t] = last
    nxt = -1
    for t in range(n_points - 1, -1, -1):
        if out[t] >= 0:
            nxt = out[t]
        elif nxt >= 0:
            out[t] = nxt
    return out


def derive_store_boundaries(model, store: LineStore, vocab, *, stride: int = 8,
                            device: str = "cpu", limit: int | None = None,
                            min_conf: float = 0.0):
    """Run alignment over a split and return per-line boundaries."""
    model.eval()
    n = len(store) if limit is None else min(limit, len(store))
    char_ids_all = []
    bnd_point, bnd_join, bnd_line, bnd_conf = [], [], [], []

    with torch.no_grad():
        for i in range(n):
            line = store.get(i)
            t = line_to_tensor(line, vocab)
            if t is None:
                char_ids_all.append(np.full(len(line["points"]), -1, np.int32))
                continue
            batch = {
                "dxy": torch.from_numpy(t["dxy"])[None].to(device),
                "pen": torch.from_numpy(t["pen"])[None].to(device),
                "point_mask": torch.ones(1, t["dxy"].shape[0], dtype=torch.bool,
                                         device=device),
                "char_seq": torch.from_numpy(t["char_seq"])[None].to(device),
                "char_mask": torch.ones(1, t["char_seq"].shape[0], dtype=torch.bool,
                                        device=device),
            }
            x = torch.cat([batch["dxy"],
                           torch.nn.functional.one_hot(batch["pen"], 4).float()], -1)
            out = model(x, batch["pen"], batch["point_mask"], sample=False)
            lp = out["ctc_logits"][0].log_softmax(-1).cpu().numpy()
            target = t["char_seq"].tolist()
            lab_frames, conf = viterbi_ctc(lp, target)
            pts = labels_to_points(lab_frames, len(t["dxy"]), stride)
            char_ids_all.append(pts)

            if conf < min_conf:
                continue
            sid = line["stroke_ids"]
            for k in range(1, len(pts)):
                if pts[k] != pts[k - 1]:
                    bnd_point.append(k)
                    bnd_join.append(1 if sid[k] == sid[k - 1] else 0)
                    bnd_line.append(i)
                    bnd_conf.append(conf)

    return {
        "char_ids": char_ids_all,
        "boundary_points": np.asarray(bnd_point, dtype=np.int32),
        "boundary_join": np.asarray(bnd_join, dtype=np.int8),
        "boundary_line": np.asarray(bnd_line, dtype=np.int32),
        "boundary_conf": np.asarray(bnd_conf, dtype=np.float32),
    }


def save_alignment(result: dict, store: LineStore, out_path: Path,
                   stats_path: Path | None = None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    char_ids = np.concatenate(result["char_ids"]) if result["char_ids"] else np.zeros(0, np.int32)
    np.savez_compressed(
        out_path,
        char_ids=char_ids.astype(np.int32),
        offsets=store.offsets,
        boundary_points=result["boundary_points"],
        boundary_join=result["boundary_join"],
        boundary_line=result["boundary_line"],
        boundary_conf=result["boundary_conf"],
    )

    joins = result["boundary_join"]
    lines_with = len({int(x) for x, j in zip(result["boundary_line"], joins) if j})
    by_writer: dict[int, list[int]] = {}
    for ln, j in zip(result["boundary_line"], joins):
        by_writer.setdefault(int(store.writers[ln]), []).append(int(j))
    stats = {
        "n_boundaries": int(len(joins)),
        "join_rate": float(joins.mean()) if len(joins) else float("nan"),
        "n_joins": int(joins.sum()) if len(joins) else 0,
        "lines_with_a_join": lines_with,
        "n_lines": len(store),
        "per_writer": {str(w): float(np.mean(v)) for w, v in sorted(by_writer.items())},
        "mean_conf": float(np.mean(result["boundary_conf"])) if len(joins) else float("nan"),
    }
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def load_alignment(root: Path, split: str):
    p = Path(root) / "processed" / split / "align.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}
