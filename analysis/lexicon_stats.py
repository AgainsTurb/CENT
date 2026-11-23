#!/usr/bin/env python3
"""Lexicon-side feasibility analysis for CENT.

Computes, from Make Me A Hanzi `graphics.txt` (Arphic Public License) and the
GB2312 code charts, the statistics the paper needs to state whether a
per-character stroke-topology constraint is *definable* for the target
inventory, and how complex that inventory is.

Deterministic: fixed input files, no sampling, no randomness.

Outputs: analysis/lexicon_stats.json, analysis/lexicon_stats.md
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRAPHICS = ROOT.parent / "data" / "lexicon" / "graphics.txt"
OUT = ROOT

# SVG path command -> number of coordinate pairs it consumes
CMD_ARGS = {"M": 1, "L": 1, "T": 1, "C": 3, "S": 2, "Q": 2, "A": 7, "H": 0.5, "V": 0.5, "Z": 0}


def parse_path(d: str) -> dict:
    """Count SVG path commands in one stroke. Robust to implicit repeats and
    compact/negative-number encodings."""
    tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    cmd_counts: Counter[str] = Counter()
    segments = 0
    cur = None
    for tok in tokens:
        if tok.isalpha():
            cur = tok.upper()
            cmd_counts[cur] += 1
        elif cur:
            segments += 1
    # number of numeric arguments consumed, as an upper bound on control points
    nums = sum(1 for t in tokens if not t.isalpha())
    return {
        "commands": dict(cmd_counts),
        "n_commands": sum(cmd_counts.values()),
        "n_numbers": nums,
        "has_curve": any(c in cmd_counts for c in ("C", "Q", "S", "T", "A")),
    }


def gb2312_levels() -> tuple[set[str], set[str]]:
    """GB2312 level-1 (3,755 chars) and level-2 (3,008 chars) code charts."""
    l1, l2 = set(), set()
    for hi in range(0xB0, 0xF8):
        for lo in range(0xA1, 0xFF):
            code = bytes([hi, lo])
            try:
                ch = code.decode("gb2312")
            except UnicodeDecodeError:
                continue
            if len(ch) != 1:
                continue
            (l1 if hi <= 0xD7 else l2).add(ch)
    return l1, l2


def main() -> int:
    lines = GRAPHICS.read_text(encoding="utf-8").splitlines()
    recs = []
    bad = 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1

    l1, l2 = gb2312_levels()
    by_char = {r["character"]: r for r in recs if "character" in r and r.get("strokes")}

    def stats(chars: set[str]) -> dict:
        present = [c for c in chars if c in by_char]
        counts = [len(by_char[c]["strokes"]) for c in present]
        if not counts:
            return {"n": 0}
        counts.sort()
        segs = []
        curved = 0
        for c in present:
            st = by_char[c]["strokes"]
            s = sum(parse_path(p)["n_commands"] for p in st)
            segs.append(s)
            if any(parse_path(p)["has_curve"] for p in st):
                curved += 1
        return {
            "n_inventory": len(chars),
            "n_covered": len(present),
            "coverage_pct": round(100 * len(present) / len(chars), 2),
            "strokes_mean": round(statistics.mean(counts), 3),
            "strokes_median": statistics.median(counts),
            "strokes_stdev": round(statistics.pstdev(counts), 3),
            "strokes_min": counts[0],
            "strokes_p90": counts[int(0.90 * (len(counts) - 1))],
            "strokes_p99": counts[int(0.99 * (len(counts) - 1))],
            "strokes_max": counts[-1],
            "pct_gt20_strokes": round(100 * sum(1 for x in counts if x > 20) / len(counts), 2),
            "pct_gt30_strokes": round(100 * sum(1 for x in counts if x > 30) / len(counts), 2),
            "mean_svg_commands_per_char": round(statistics.mean(segs), 2),
            "pct_with_bezier_curve": round(100 * curved / len(present), 2),
        }

    all_chars = set(by_char)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Make Me A Hanzi graphics.txt (Arphic Public License)",
        "license_note": "graphics.txt is derived from AR PL KaitiM GB / AR PL UKai and is "
        "distributed under the Arphic Public License. The dictionary file is a separate "
        "LGPL-3.0-or-later work.",
        "records_total": len(recs),
        "records_unparsable": bad,
        "characters_with_strokes": len(all_chars),
        "stroke_count_histogram": dict(
            sorted(Counter(len(by_char[c]["strokes"]) for c in all_chars).items())
        ),
        "gb2312_level1": stats(l1),
        "gb2312_level2": stats(l2),
        "gb2312_union": stats(l1 | l2),
        "all": stats(all_chars),
        "most_complex": sorted(
            ({"character": c, "strokes": len(by_char[c]["strokes"])} for c in all_chars),
            key=lambda r: -r["strokes"],
        )[:12],
        "has_medians_pct": round(
            100 * sum(1 for c in all_chars if by_char[c].get("medians")) / len(all_chars), 2
        ),
    }
    (OUT / "lexicon_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    a, b = report["gb2312_level1"], report["gb2312_level2"]
    md = [
        "# Lexicon feasibility analysis",
        "",
        f"Source: Make Me A Hanzi `graphics.txt` (Arphic Public License) · "
        f"generated {report['generated_at']}",
        "",
        f"- Records: {report['records_total']:,} ({report['records_unparsable']} unparsable)",
        f"- Characters with stroke data: {report['characters_with_strokes']:,}",
        f"- Fraction carrying stroke medians: {report['has_medians_pct']}%",
        "",
        "| Inventory | characters | covered | coverage | mean strokes | median | p99 | max | >20 strokes | >30 strokes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in (("GB2312 level-1", a), ("GB2312 level-2", b), ("GB2312 union", report["gb2312_union"])):
        md.append(
            f"| {name} | {s['n_inventory']:,} | {s['n_covered']:,} | {s['coverage_pct']}% | "
            f"{s['strokes_mean']} | {s['strokes_median']} | {s['strokes_p99']} | {s['strokes_max']} | "
            f"{s['pct_gt20_strokes']}% | {s['pct_gt30_strokes']}% |"
        )
    md += ["", "## Most complex characters in the lexicon", "",
           "| character | strokes |", "|---|---|"]
    for r in report["most_complex"]:
        md.append(f"| {r['character']} | {r['strokes']} |")
    (OUT / "lexicon_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("characters_with_strokes", "gb2312_level1", "gb2312_level2")}, indent=2, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
