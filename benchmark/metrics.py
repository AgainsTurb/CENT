"""Benchmark metrics for line generation, matching how the field reports them.

The two protocols we align to (DiffBrush, ICCV 2025; OLHWG, ICLR 2025) use:

    FID, KID      visual realism       (Inception features)
    HWD           style distance       (VGG16 features pretrained on handwriting)
    AR, CR        content correctness  (a frozen Chinese line recogniser)
    IS, GS        visual quality       (DiffBrush only)

The MATH for FID, KID and HWD is implemented here and unit-tested. The FEATURES are not
computed here, because that requires a specific pretrained network; every function takes
features and the caller states which extractor produced them. A metric with no extractor
returns NOT_RUN with the exact reason, so a table can never quietly fill in a number that
was never measured.

AR/CR follow the Chinese line-recognition convention: computed by DP edit distance against
the ground-truth transcript, aggregated over the corpus, with
    AR = (N - D - S - I) / N        CR = (N - D - S) / N
where N is the number of reference characters. A recogniser's own accuracy bounds all of
this, which is why its identity and its own AR/CR on real lines must be reported.
"""
from __future__ import annotations

import numpy as np

__all__ = ["edit_counts", "ar_cr", "frechet_distance", "kid_mmd", "hwd_distance",
           "metric_requirements", "NOT_RUN"]


def NOT_RUN(reason: str) -> dict:
    return {"status": "NOT_RUN", "reason": reason}


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #
def edit_counts(hyp: str, ref: str) -> tuple[int, int, int, int]:
    """(N, deletions, substitutions, insertions) for hyp against ref.

    N is the reference length. Deletion = an extra character in the hypothesis (present in
    output, absent in reference). Insertion = a missing character.
    """
    a, b = list(hyp), list(ref)
    n, m = len(a), len(b)
    cost = np.zeros((n + 1, m + 1), np.int32)
    D = np.zeros((n + 1, m + 1), np.int32)
    S = np.zeros((n + 1, m + 1), np.int32)
    I = np.zeros((n + 1, m + 1), np.int32)
    cost[:, 0] = np.arange(n + 1)
    D[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    I[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cost[i, j] = cost[i - 1, j - 1]
                D[i, j], S[i, j], I[i, j] = D[i - 1, j - 1], S[i - 1, j - 1], I[i - 1, j - 1]
                continue
            sub = cost[i - 1, j - 1] + 1
            dele = cost[i - 1, j] + 1
            ins = cost[i, j - 1] + 1
            best = min(sub, dele, ins)
            cost[i, j] = best
            if best == sub:
                D[i, j], S[i, j], I[i, j] = D[i - 1, j - 1], S[i - 1, j - 1] + 1, I[i - 1, j - 1]
            elif best == dele:
                D[i, j], S[i, j], I[i, j] = D[i - 1, j] + 1, S[i - 1, j], I[i - 1, j]
            else:
                D[i, j], S[i, j], I[i, j] = D[i, j - 1], S[i, j - 1], I[i, j - 1] + 1
    return m, int(D[n, m]), int(S[n, m]), int(I[n, m])


def ar_cr(pairs: list[tuple[str, str]]) -> dict:
    """Aggregate AR/CR over (hypothesis, reference) pairs."""
    N = D = S = I = 0
    for hyp, ref in pairs:
        n, d, s, i = edit_counts(hyp, ref)
        N += n; D += d; S += s; I += i
    if N == 0:
        return NOT_RUN("no reference characters")
    return {"status": "OK", "N": N, "D": D, "S": S, "I": I,
            "AR": round(max(0.0, (N - D - S - I) / N), 4),
            "CR": round(max(0.0, (N - D - S) / N), 4),
            "lines": len(pairs),
            "caveat": "computed from recogniser output; the recogniser's own accuracy bounds this"}


# --------------------------------------------------------------------------- #
# realism / style
# --------------------------------------------------------------------------- #
def frechet_distance(fa: np.ndarray, fb: np.ndarray, eps: float = 1e-6) -> float:
    """FID between two feature sets, computed from features (not images)."""
    fa = np.asarray(fa, np.float64); fb = np.asarray(fb, np.float64)
    mu1, mu2 = fa.mean(0), fb.mean(0)
    c1 = np.cov(fa, rowvar=False) + eps * np.eye(fa.shape[1])
    c2 = np.cov(fb, rowvar=False) + eps * np.eye(fb.shape[1])
    d = mu1 - mu2
    # symmetric PSD square root via eigen-decomposition (scipy would be heavier)
    w, v = np.linalg.eigh(c1)
    w = np.clip(w, 0, None)
    sqrt_c1 = v @ np.diag(np.sqrt(w)) @ v.T
    m = sqrt_c1 @ c2 @ sqrt_c1
    w2, _ = np.linalg.eigh(m)
    tr_sqrt = np.sqrt(np.clip(w2, 0, None)).sum()
    return float(d @ d + np.trace(c1) + np.trace(c2) - 2 * tr_sqrt)


def kid_mmd(fa: np.ndarray, fb: np.ndarray) -> float:
    """KID: unbiased MMD with a cubic polynomial kernel, as in the KID paper."""
    fa = np.asarray(fa, np.float64); fb = np.asarray(fb, np.float64)
    d = fa.shape[1]
    gamma = 1.0 / d
    k = lambda x, y: (gamma * (x @ y.T) + 1.0) ** 3
    kaa, kbb, kab = k(fa, fa), k(fb, fb), k(fa, fb)
    na, nb = len(fa), len(fb)
    return float((kaa.sum() - np.trace(kaa)) / (na * (na - 1))
                 + (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1))
                 - 2 * kab.mean())


def hwd_distance(fa: np.ndarray, fb: np.ndarray) -> float:
    """HWD: Euclidean distance between mean handwriting features.

    Style distance only --- it needs a net pretrained on handwriting (as DiffBrush uses),
    so it is meaningless with ImageNet features. The caller must pass features from that
    net, and the paper must name the net.
    """
    fa = np.asarray(fa, np.float64); fb = np.asarray(fb, np.float64)
    return float(np.linalg.norm(fa.mean(0) - fb.mean(0)))


def metric_requirements() -> dict:
    """What each metric still needs before it can report a number."""
    return {
        "AR/CR": "a frozen Chinese line recogniser (identity + its own AR/CR on real lines)",
        "FID": "Inception-v3 features for generated and real lines (>=512 lines)",
        "KID": "same Inception features as FID; unbiased MMD, no size floor in principle",
        "HWD": "VGG16 features from a net pretrained on handwritten text images",
        "IS": "Inception softmax predictions (DiffBrush protocol)",
        "GS": "Geometry Score with a subsampled topological pipeline (DiffBrush protocol)",
        "connectivity": "generated trajectories: CENT reports it; image-only baselines cannot",
    }
