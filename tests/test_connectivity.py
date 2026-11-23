"""Tests for connectivity metrics: join F1, rate error, gaps, continuity, offline proxy.

The offline proxy is the one part that could quietly mislead, so it is tested on synthetic
lines where the answer is known by construction: a stroke drawn across the boundary must be
detected as a join, and separated characters must not.

Usage: python tests/test_connectivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.connectivity import (  # noqa: E402
    continuity_error, csm_note, cursive_f1, cursive_rate_error, gap_similarity,
    offline_join_proxy, space_similarity, validate_offline_proxy,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


def line_image(n_chars: int = 3, joined: tuple[bool, ...] = (True, False),
               h: int = 32, cell: int = 20) -> np.ndarray:
    """Ink = 0 (canonical polarity). Each character is a vertical bar; a join draws a
    horizontal stroke through the boundary band between two cells."""
    img = np.full((h, n_chars * cell), 255, np.uint8)
    for i in range(n_chars):
        x = i * cell + cell // 2
        img[6:h - 6, x - 2:x + 2] = 0
    for b, j in enumerate(joined):
        if j:
            x0 = (b + 1) * cell - 4
            img[10:14, x0:x0 + 8] = 0          # stroke crossing the boundary
    return img


def main() -> int:
    print("\njoin F1")
    r = cursive_f1([True, True, False], [True, False, False])
    check("one TP, one FP, no FN", (r["TP"], r["FP"], r["FN"]) == (1, 1, 0), str(r))
    check("F1 is 2/3 here", abs(r["F1_cursive"] - 2 / 3) < 1e-6, str(r["F1_cursive"]))
    check("all correct gives F1=1", cursive_f1([True, False], [True, False])["F1_cursive"] == 1.0)
    check("no positives anywhere is undefined, not zero",
          cursive_f1([False, False], [False, False])["F1_cursive"] is None)

    print("\ncursive rate error (the Chinese-safe one)")
    r = cursive_rate_error([True, True, False, False], [False] * 4)
    check("rate error computed without any positive GT", r["status"] == "OK", str(r))
    check("CRE equals the rate gap", abs(r["CRE"] - 0.5) < 1e-9, str(r["CRE"]))
    check("both-silent is zero error",
          cursive_rate_error([False, False], [False, False])["CRE"] == 0.0)

    print("\ngaps and spaces")
    check("identical gaps give KGS 1", gap_similarity([4, 5], [4, 5])["KGS"] == 1.0)
    check("doubled gaps give KGS 0", gap_similarity([8, 10], [4, 5])["KGS"] == 0.0,
          str(gap_similarity([8, 10], [4, 5])))
    check("word-space similarity is NOT_RUN for Chinese, with the reason",
          space_similarity([], [])["status"] == "NOT_RUN"
          and "no word spaces" in space_similarity([], [])["reason"])

    print("\ncontinuity (the smoothness term CSM lacks)")
    prev = np.array([[0.0, 0.0], [10.0, 0.0]], float)
    nxt = np.array([[20.0, 0.0], [30.0, 0.0]], float)
    smooth = np.array([[10.0, 0.0], [13.0, 0.0], [16.0, 0.0], [20.0, 0.0]], float)
    kinked = np.array([[10.0, 9.0], [13.0, 9.0], [16.0, 9.0], [20.0, 9.0]], float)
    good = continuity_error(prev, smooth, nxt)
    bad = continuity_error(prev, kinked, nxt)
    check("a straight connector has ~0 tangent error", good["tangent_rad"] < 1e-6, str(good))
    check("a displaced connector has ~0 tangent but nonzero distance",
          bad["distance"] > 0 and bad["tangent_rad"] < 1e-6, str(bad))
    diag = np.array([[10.0, 0.0], [13.0, 3.0], [16.0, 3.0], [20.0, 0.0]], float)
    check("a kinked connector shows a large tangent error",
          continuity_error(prev, diag, nxt)["tangent_rad"] > 0.7,
          str(continuity_error(prev, diag, nxt)))

    print("\noffline proxy: joined vs separated, known by construction")
    j = offline_join_proxy(line_image(3, (True, False)), 3)
    check("proxy runs", j["status"] == "OK", str(j))
    check("proxy finds exactly two boundaries", len(j["joins"]) == 2, str(j["joins"]))
    check("proxy calls the drawn-through boundary JOINED", j["joins"][0] is True, str(j["boundaries"]))
    check("proxy does not call the empty boundary joined", j["joins"][1] is False, str(j["boundaries"]))
    all_join = offline_join_proxy(line_image(3, (True, True)), 3)
    check("both joins detected", all_join["joins"] == [True, True], str(all_join["joins"]))
    none_join = offline_join_proxy(line_image(3, (False, False)), 3)
    check("no false positives when nothing crosses", none_join["joins"] == [False, False],
          str(none_join["joins"]))
    check("connector rate reported", none_join["connector_rate"] == 0.0, str(none_join["connector_rate"]))
    blank = offline_join_proxy(np.full((16, 40), 255, np.uint8), 3)
    check("a blank line is NOT_RUN, not a silent zero", blank["status"] == "NOT_RUN", str(blank))

    print("\nproxy validation on paired lines")
    samples = [(line_image(3, (True, False)), 3, [True, False]),
               (line_image(3, (False, False)), 3, [False, False])]
    v = validate_offline_proxy(samples)
    check("validation returns accuracy over boundaries", v["status"] == "OK" and v["boundaries"] == 4, str(v))
    check("proxy agrees with ground truth on these clear cases", v["accuracy"] == 1.0, str(v))
    check("empty sample set is NOT_RUN", validate_offline_proxy([])["status"] == "NOT_RUN")

    print("\nthe CSM naming caveat is exposed, not buried")
    check("csm_note states the formulas are ours", "our own definitions" in csm_note(), csm_note())

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
