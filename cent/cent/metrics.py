"""Evaluation metrics for line-level Chinese handwriting generation.

The Chinese literature measures *layout* (OLHWG: eight geometric box distances)
and *recognition* (AR/CR). Neither can see the join. These helpers add the
continuity side, following the definitions used for English sentence-level
work (F1_Cursive / CRE) and adapting them to characters instead of words.
"""

from __future__ import annotations

import numpy as np
import torch


def boundary_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                threshold: float = 0.5) -> dict:
    """F1 for the join class (the metric the Chinese literature lacks)."""
    if mask.sum() == 0:
        return {"f1": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "n": 0}
    pred = (logits[mask] > threshold).float()
    true = labels[mask].float()
    tp = float(((pred == 1) & (true == 1)).sum())
    fp = float(((pred == 1) & (true == 0)).sum())
    fn = float(((pred == 0) & (true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "n": int(mask.sum())}


def continuation_rate_error(pred_labels, true_labels) -> float:
    """|mean(pred joins) - mean(true joins)| -- CRE."""
    p = float(np.mean(pred_labels)) if len(pred_labels) else 0.0
    t = float(np.mean(true_labels)) if len(true_labels) else 0.0
    return abs(p - t)


def gap_stats(points: np.ndarray, char_ids: np.ndarray, labels: np.ndarray) -> dict:
    """Inter-character gap for lift vs join boundaries (KGS-style)."""
    lifts, joins = [], []
    for t in range(1, len(char_ids)):
        if char_ids[t] != char_ids[t - 1]:
            g = float(np.linalg.norm(points[t] - points[t - 1]))
            (joins if labels and labels[len(joins) + len(lifts)] > 0.5 else lifts).append(g)
    return {
        "gap_lift_mean": float(np.mean(lifts)) if lifts else float("nan"),
        "gap_join_mean": float(np.mean(joins)) if joins else float("nan"),
        "n_lift": len(lifts), "n_join": len(joins),
    }


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Plain DTW (no window). Small sequences only."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    D = np.full((len(a) + 1, len(b) + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = d + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[len(a), len(b)] / max(len(a), len(b)))


def join_rate(labels) -> float:
    labels = list(labels)
    return float(np.mean(labels)) if labels else float("nan")
