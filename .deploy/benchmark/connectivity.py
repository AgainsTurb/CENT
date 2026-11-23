"""Connectivity metrics for text lines: per-boundary, online and offline.

WHY THIS FILE EXISTS
BoundInk (arXiv:2604.02103) introduced Connectivity and Spacing Metrics (CSM) ---
F1_Cursive, CRE, KGS, SSS --- and reports that under the Chinese (OLHWG-compatible)
protocol *only KGS is defined*, because that evaluation set has no cursive-positive
boundaries and no whitespace runs. Chinese lines have no word spaces at all, and their
connector ground truth must be inferred (see common/casia_lines.py). So the field's own
metric suite degenerates exactly where this paper works.

This module therefore provides:
  * per-boundary JOIN metrics that are well defined without whitespace or a
    cursive-positive test set:
      cursive_f1, cursive_rate_error, gap_similarity   (CSM's intent, our formulas)
      space_similarity                                  (reported only where defined)
  * continuity_error --- tangent and distance mismatch at a join. CSM has no smoothness
    term; this is the property a Bezier connector guarantees by construction.
  * offline_join_proxy --- connectivity estimated from an IMAGE alone, so image-only line
    generators (DiffBrush, TGC-Diff, InkSpire) can be scored at all.
  * validate_offline_proxy --- checks that proxy against trajectory-derived ground truth on
    paired lines, which is the only way an image-based connectivity number earns belief.

HONESTY NOTE ON NAMES: BoundInk's exact formulas are not retrievable from the public text
(the metric names and intent are; the definitions are not). The CSM-named functions here are
therefore OUR definitions of the same constructs, and the paper must say so rather than
present them as reproductions. `csm_note()` returns that sentence.
"""
from __future__ import annotations

import numpy as np

INK_THRESHOLD = 128          # below this (darker) counts as ink


def csm_note() -> str:
    return ("CSM-named metrics follow BoundInk's constructs (F1_Cursive, CRE, KGS, SSS) but "
            "are our own definitions: the released text does not state their formulas.")


# --------------------------------------------------------------------------- #
# per-boundary decisions (online or derived)
# --------------------------------------------------------------------------- #
def cursive_f1(pred_join: list[bool], gt_join: list[bool]) -> dict:
    """F1 over boundary join decisions. The core connectivity number."""
    if len(pred_join) != len(gt_join):
        raise ValueError(f"length mismatch: {len(pred_join)} predictions vs {len(gt_join)} boundaries")
    tp = sum(1 for p, g in zip(pred_join, gt_join) if p and g)
    fp = sum(1 for p, g in zip(pred_join, gt_join) if p and not g)
    fn = sum(1 for p, g in zip(pred_join, gt_join) if not p and g)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (tp + fp + fn) else None)
    return {"status": "OK", "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "F1_cursive": f1,
            "note": "undefined (None) when there is no positive boundary to score"}


def cursive_rate_error(pred_join: list[bool], gt_join: list[bool]) -> dict:
    """|predicted connector rate - reference connector rate|. Lower is better.

    Well defined even when there are no positive boundaries, which is the Chinese case:
    a model that joins nothing and a writer who joins nothing score 0 error, correctly.
    """
    if not gt_join:
        return {"status": "NOT_RUN", "reason": "no boundaries"}
    pr = sum(map(bool, pred_join)) / len(pred_join)
    gr = sum(map(bool, gt_join)) / len(gt_join)
    return {"status": "OK", "pred_rate": round(pr, 4), "ref_rate": round(gr, 4),
            "CRE": round(abs(pr - gr), 4)}


def gap_similarity(pred_gaps: list[float], gt_gaps: list[float]) -> dict:
    """KGS: 1 - normalized mean absolute gap error. 1.0 is perfect.

    Gaps are the horizontal distances between consecutive character ink boxes; they capture
    kerning, which is where decoupled layout most visibly fails.
    """
    if not gt_gaps or len(pred_gaps) != len(gt_gaps):
        return {"status": "NOT_RUN", "reason": "no comparable gaps"}
    p, g = np.asarray(pred_gaps, float), np.asarray(gt_gaps, float)
    scale = max(float(np.mean(np.abs(g))), 1e-6)
    return {"status": "OK", "KGS": round(float(max(0.0, 1.0 - np.mean(np.abs(p - g)) / scale)), 4)}


def space_similarity(pred: list[float], gt: list[float]) -> dict:
    """SSS: word-space similarity. NOT_RUN for Chinese, which has no word spaces.

    Reported explicitly rather than silently omitted, because BoundInk also reports it as
    undefined under the Chinese protocol.
    """
    if not gt:
        return {"status": "NOT_RUN",
                "reason": "Chinese text lines have no word spaces; SSS is undefined here "
                          "(BoundInk reports the same for the OLHWG-compatible protocol)"}
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    scale = max(float(np.mean(np.abs(g))), 1e-6)
    return {"status": "OK", "SSS": round(float(max(0.0, 1.0 - np.mean(np.abs(p - g)) / scale)), 4)}


# --------------------------------------------------------------------------- #
# continuity (the property a Bezier connector guarantees)
# --------------------------------------------------------------------------- #
def continuity_error(prev_stroke: np.ndarray, connector: np.ndarray,
                     next_stroke: np.ndarray, scale: float = 1.0) -> dict:
    """Distance and tangent mismatch where a connector meets its neighbours.

    distance  = |connector.start - prev.end| + |connector.end - next.start|
    tangent   = angle between the previous stroke's exit direction and the connector's
                entry direction, plus the same at the exit. Radians, 0 = perfectly smooth.

    A system that supervises a pen-down "join" label can still produce a kink here; a
    Bezier whose control points ARE the neighbours' endpoints cannot. That difference is
    invisible to CSM, which has no smoothness term.
    """
    prev, conn, nxt = (np.asarray(a, float) for a in (prev_stroke, connector, next_stroke))
    d_start = float(np.linalg.norm(conn[0] - prev[-1]))
    d_end = float(np.linalg.norm(conn[-1] - nxt[0]))
    # tangents from the last two points / first two points
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.zeros_like(v)
    t_prev = unit(prev[-1] - prev[-2]) if len(prev) >= 2 else None
    t_cin = unit(conn[1] - conn[0]) if len(conn) >= 2 else None
    t_cout = unit(conn[-1] - conn[-2]) if len(conn) >= 2 else None
    t_next = unit(nxt[1] - nxt[0]) if len(nxt) >= 2 else None
    def angle(u, v):
        if u is None or v is None:
            return None
        c = float(np.clip(np.dot(u, v), -1.0, 1.0))
        return float(np.arccos(c))
    a_in = angle(t_prev, t_cin)
    a_out = angle(t_cout, t_next)
    tang = [a for a in (a_in, a_out) if a is not None]
    return {"status": "OK",
            "distance": round((d_start + d_end) / max(scale, 1e-6), 4),
            "tangent_rad": round(float(np.mean(tang)), 4) if tang else None,
            "tangent_deg": round(float(np.degrees(np.mean(tang))), 2) if tang else None}


# --------------------------------------------------------------------------- #
# offline: connectivity from an image alone
# --------------------------------------------------------------------------- #
def _ink(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=0) if a.shape[0] in (1, 3) else a.mean(axis=2)
    return a < INK_THRESHOLD


def offline_join_proxy(image: np.ndarray, n_chars: int, band: int = 2,
                       min_cols: int = 2) -> dict:
    """Estimate which character boundaries are JOINED, from the image alone.

    Procedure (deliberately simple and stated, not learned):
      1. locate the ink bounding box;
      2. divide its width evenly into `n_chars` cells (the corpus gives no per-character
         boxes for offline lines, so this is the declared approximation);
      3. a boundary at x is JOINED if ink is present in the columns [x-band, x+band] AND
         that ink overlaps vertically with ink on both sides, i.e. a stroke runs through.

    Validity: it assumes roughly equal character advances, which holds for the written-page
    corpora but not for unusual layouts. `validate_offline_proxy` must be run against
    trajectory-derived ground truth before any number from this is reported.
    """
    ink = _ink(image)
    if ink.sum() == 0 or n_chars < 2:
        return {"status": "NOT_RUN", "reason": "no ink or fewer than two characters",
                "joins": [], "boundaries": []}
    cols = np.flatnonzero(ink.any(axis=0))
    x0, x1 = int(cols[0]), int(cols[-1])
    w = max(x1 - x0, n_chars)
    bounds = [x0 + int(round(w * k / n_chars)) for k in range(1, n_chars)]
    joins, detail = [], []
    for x in bounds:
        lo, hi = max(x0, x - band), min(x1 + 1, x + band + 1)
        b = ink[:, lo:hi]
        left = ink[:, max(x0, x - 4 * band - min_cols):lo]
        right = ink[:, hi:min(x1 + 1, x + 4 * band + min_cols)]
        through = bool(b.any()) and int(b.any(axis=0).sum()) >= min_cols
        # vertical overlap: ink in the band that shares rows with ink on both sides
        rows_b = np.flatnonzero(b.any(axis=1))
        overlap = False
        if through and rows_b.size:
            rows_l = set(np.flatnonzero(left.any(axis=1)).tolist())
            rows_r = set(np.flatnonzero(right.any(axis=1)).tolist())
            overlap = any((r in rows_l or r in rows_r) for r in rows_b.tolist())
        joined = bool(through and overlap)
        joins.append(joined)
        detail.append({"x": x, "through": through, "overlap": overlap})
    return {"status": "OK", "n_chars": n_chars, "joins": joins,
            "connector_rate": round(sum(joins) / max(len(joins), 1), 4),
            "boundaries": detail,
            "assumption": "equal character advances inside the line's ink bounding box"}


def validate_offline_proxy(samples: list[tuple[np.ndarray, int, list[bool]]]) -> dict:
    """Compare the offline proxy against trajectory-derived joins on paired lines.

    Each sample is (line_image, n_chars, gt_join_flags). Returns agreement statistics, so
    an offline connectivity number can be reported with a measured validity, not an
    assumed one.
    """
    tp = fp = fn = tn = 0
    used = 0
    for img, n, gt in samples:
        pred = offline_join_proxy(img, n)["joins"]
        if len(pred) != len(gt):
            continue
        used += 1
        for p, g in zip(pred, gt):
            if p and g:
                tp += 1
            elif p and not g:
                fp += 1
            elif not p and g:
                fn += 1
            else:
                tn += 1
    if used == 0:
        return {"status": "NOT_RUN", "reason": "no paired lines with matching boundary counts"}
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {"status": "OK", "lines": used, "boundaries": total,
            "accuracy": round((tp + tn) / total, 4),
            "precision": prec, "recall": rec, "F1": f1,
            "note": "this is the validity of the OFFLINE PROXY, not of a generator"}
