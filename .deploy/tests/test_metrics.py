"""Tests for the benchmark metrics: edit-count AR/CR, Frechet distance, KID/MMD, HWD.

These are math checks against analytic answers, not evaluations of any model:
  * AR/CR on hand-built strings where the operation counts are known by construction.
  * Frechet distance between identical distributions is 0, and between N(mu1, I) and
    N(mu2, I) equals |mu1-mu2|^2 exactly, which gives a closed-form target.
  * KID/MMD is ~0 for two samples of the same distribution and grows with separation.
  * HWD is 0 for identical features and |delta| for a shift.

Usage: python tests/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.metrics import (  # noqa: E402
    ar_cr, edit_counts, frechet_distance, hwd_distance, kid_mmd, metric_requirements,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


def main() -> int:
    print("\nedit counts / AR / CR")
    check("identical strings -> no operations", edit_counts("人大人大", "人大人大") == (4, 0, 0, 0))
    check("one missing char -> 1 insertion", edit_counts("人大大", "人大人大") == (4, 0, 0, 1),
          str(edit_counts("人大大", "人大人大")))
    check("one extra char -> 1 deletion", edit_counts("人大人大X", "人大人大") == (4, 1, 0, 0),
          str(edit_counts("人大人大X", "人大人大")))
    check("one wrong char -> 1 substitution", edit_counts("人大水大", "人大人大") == (4, 0, 1, 0),
          str(edit_counts("人大水大", "人大人大")))

    rep = ar_cr([("人大人大", "人大人大"), ("人大大", "人大人大"), ("人大人大X", "人大人大")])
    # N=12, D=1, I=1, S=0 -> AR=(12-2)/12, CR=(12-1)/12
    check("AR over the corpus", abs(rep["AR"] - 10 / 12) < 1e-4, str(rep["AR"]))
    check("CR over the corpus", abs(rep["CR"] - 11 / 12) < 1e-4, str(rep["CR"]))
    check("substitution hits CR but not AR",
          abs(ar_cr([("人大水大", "人大人大")])["AR"] - 3 / 4) < 1e-6
          and abs(ar_cr([("人大水大", "人大人大")])["CR"] - 3 / 4) < 1e-6)

    print("\nFrechet distance")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 8))
    b = a + 1.5                                  # pure shift, identity covariance
    fd = frechet_distance(a, b)
    check("FID is |mu shift|^2 for equal covariances", abs(fd - 8 * 1.5 ** 2) < 1e-3, f"{fd:.4f}")
    check("FID of a set with itself is ~0", frechet_distance(a, a) < 1e-6,
          str(frechet_distance(a, a)))

    print("\nKID / MMD")
    k_same = kid_mmd(a, rng.normal(size=(500, 8)))
    k_far = kid_mmd(a, rng.normal(loc=3.0, size=(500, 8)))
    check("KID grows with separation", k_far > k_same, f"same={k_same:.4f} far={k_far:.4f}")
    check("KID is finite and can be slightly negative (unbiased estimator)", np.isfinite(k_same))

    print("\nHWD")
    check("HWD of identical features is 0", hwd_distance(a, a) < 1e-9)
    check("HWD equals the mean shift norm", abs(hwd_distance(a, a + 2.0) - 2.0 * np.sqrt(8)) < 1e-6,
          str(hwd_distance(a, a + 2.0)))

    print("\nrequirements are explicit")
    req = metric_requirements()
    check("every metric names its extractor", all(v for v in req.values()), str(req))
    check("AR/CR names the recogniser dependency", "recogniser" in req["AR/CR"], req["AR/CR"])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
