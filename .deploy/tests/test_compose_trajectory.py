"""Test the trajectory composer that makes composed online baselines comparable.

The point of these checks is the honesty property, not the geometry: a composed online
baseline must report connector_rate = 0 as a DECLARED consequence of composition, never as
a measured result.

Usage: python tests/test_compose_trajectory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.compose_trajectory import compose_trajectory, trajectory_to_polylines  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if (detail and not cond) else ''}")


def char(n_strokes: int = 2, h: float = 40.0) -> list[np.ndarray]:
    return [np.array([[0, i * 3.0], [10, i * 3.0], [20, i * 3.0 + h]], np.float32)
            for i in range(n_strokes)]


def main() -> int:
    chars = [char(2), char(3), char(1)]

    print("\ncomposition preserves every stroke, in reading order")
    strokes, meta = compose_trajectory(chars, height=64, mode="uniform")
    check("stroke count is the sum of the characters'", len(strokes) == 6, str(len(strokes)))
    check("meta counts characters", meta["n_chars"] == 3, str(meta["n_chars"]))
    xs = [s[:, 0].mean() for s in strokes]
    check("later characters sit to the right of earlier ones",
          xs[0] < xs[-1], f"{xs[0]:.1f} vs {xs[-1]:.1f}")

    print("\ndeclared connectivity, not measured")
    d = meta["declared"]
    check("connector rate declared as 0", d["connector_rate"] == 0.0, str(d["connector_rate"]))
    check("declared as construction, not measurement", d["source"] == "construction", d["source"])
    check("declaration explains itself", "pen lift" in d["why"], d["why"])

    print("\nstyle layout uses the reference statistics")
    layout = {"advance": 48.0, "baseline": 30.0, "jitter_adv": 2.0, "jitter_y": 1.0,
              "source": "3 reference lines"}
    s2, m2 = compose_trajectory(chars, layout, height=64, mode="style")
    check("style mode is recorded", m2["mode"] == "style", m2["mode"])
    check("style advance is near the reference", abs(m2["advance_mean"] - 48.0) < 6.0,
          f"{m2['advance_mean']:.1f}")
    check("no fallback when a layout exists", m2["fallback_reason"] is None, str(m2["fallback_reason"]))

    s3, m3 = compose_trajectory(chars, None, mode="style")
    check("style without a layout falls back and says why",
          m3["mode"] == "uniform" and m3["fallback_reason"] is not None, str(m3))

    print("\npolylines break at every pen lift")
    pl = trajectory_to_polylines(strokes)
    n_nan = int(np.isnan(pl[:, 0]).sum())
    check("one NaN separator between consecutive strokes", n_nan == len(strokes) - 1, str(n_nan))
    check("total rows = points + separators",
          len(pl) == sum(len(s) for s in strokes) + (len(strokes) - 1), str(len(pl)))

    print("\nedge cases")
    one, m1 = compose_trajectory([char(1)], height=64)
    check("a single character still composes", len(one) == 1 and m1["n_chars"] == 1)
    try:
        compose_trajectory([], height=64)
        check("empty input is refused", False)
    except ValueError:
        check("empty input is refused", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
