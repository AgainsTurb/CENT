#!/usr/bin/env python3
"""Render refs/verification_report.md from the stored records.

Derived, never hand-written. Sources of truth, in priority order per key:
  1. overrides.json  -- full-text / DOI-verified records; these MUST be used
     as-is. An earlier revision merged the stale registry store into these rows
     and re-imported a rejected title-squatter DOI, which is exactly the class of
     error this file exists to expose.
  2. verified_refs.json -- registry resolutions (ACCEPTED / NO_MATCH / REJECTED).
  3. paper/refs.bib  -- what actually ships; also the key set to iterate.

Iterates the union of all three key sets so no shipped entry can be omitted.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIB = ROOT.parent / "paper" / "refs.bib"


def arxiv_from(text: str | None) -> str:
    if not text:
        return ""
    m = re.search(r"arXiv:([0-9]{4}\.[0-9]{4,5})", text)
    return f"arXiv:{m.group(1)}" if m else ""


def main() -> int:
    store_raw = json.loads((ROOT / "verified_refs.json").read_text(encoding="utf-8"))
    store = {e["key"]: e for e in store_raw["entries"]}
    overrides = json.loads((ROOT / "overrides.json").read_text(encoding="utf-8"))["entries"]
    shipped = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))

    rows = []
    for k in sorted(set(store) | set(overrides) | shipped):
        row = {"key": k, "klass": "-", "score": "-", "rid": "-", "venue": "-", "year": "-"}
        if k in overrides:                       # authoritative for this key
            o = overrides[k]
            row["klass"] = "full-text"
            row["rid"] = o.get("doi") or arxiv_from(o.get("journal") or o.get("note"))
            row["venue"] = o.get("booktitle") or o.get("journal") or "-"
            row["year"] = o.get("year", "-")
        elif k in store:
            e = store[k]
            if e.get("status") == "REJECTED":
                row["klass"] = "rejected"
                row["score"] = e.get("score") or "-"
                row["rid"] = "none (rejected)"
                row["venue"] = "rejected record, not cited"
                row["year"] = "-"
            else:
                row["klass"] = {"ACCEPTED": "registry"}.get(e.get("status"), "unresolved")
                row["score"] = e.get("score") or e.get("best_partial_score") or "-"
                row["rid"] = e.get("doi") or e.get("arxiv_id") or "-"
                row["venue"] = e.get("venue") or "-"
                row["year"] = e.get("year") or "-"
        else:
            row["klass"] = "bib-only"
        row["shipped"] = "yes" if k in shipped else "**no**"
        rows.append(row)

    def tbl(rs: list[dict]) -> list[str]:
        out = ["| key | class | score | identifier | venue | year | in manuscript |",
               "|---|---|---|---|---|---|---|"]
        for r in rs:
            out.append(
                f"| {r['key']} | {r['klass']} | {r['score']} | {r['rid']} | "
                f"{str(r['venue'])[:40]} | {r['year']} | {r['shipped']} |"
            )
        return out

    lines = [
        "# Reference verification report",
        "",
        f"Derived at {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`refs/render_report.py` from `overrides.json`, `verified_refs.json` and the shipped "
        "`paper/refs.bib`. Do not edit by hand; re-run the script.",
        "",
        "Three provenance classes, kept distinct:",
        "",
        "- **registry** -- a Crossref or arXiv title match at similarity >= 0.87.",
        "- **full-text** -- the paper itself, or its official DOI/proceedings record, was retrieved.",
        "- **rejected** -- a match that was examined and refused. A rejected record is never the",
        "  identifier of a cited work; the cited work is listed separately under its own key.",
        "",
        f"Shipped: {len(shipped)} entries in `paper/refs.bib`; "
        f"rows below: {len(rows)} (union of shipped, registry and override keys).",
        "",
    ]
    lines += tbl([r for r in rows if r["klass"] != "rejected"])

    rejected = [r for r in rows if r["klass"] == "rejected"]
    # A key that was rejected AND then re-established by a full-text record (fid)
    # takes the full-text row above, so collect rejections from the store directly:
    # the refusal must stay documented even when the citation is now correct.
    rejected_store = [e for e in store.values() if e.get("status") == "REJECTED"]
    if rejected_store:
        lines += ["", "## Rejected matches (never cited as themselves)", ""]
        for e in rejected_store:
            lines.append(f"- `{e['key']}` -- {e.get('rejection_reason', 'no reason recorded')}")
            if e.get("superseded_by"):
                lines.append(f"  - superseded by `{e['superseded_by']}`, which is the record "
                             "listed for this citation in the table above.")

    lines += [
        "",
        "## Coverage boundary",
        "",
        "This report establishes bibliographic resolution and provenance class. It does not "
        "establish that a cited work supports the sentence citing it -- that is the integrity "
        "gate's Phase B, which is recorded in `phase2_5_integrity/integrity_report.md`.",
        "",
    ]
    (ROOT / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report: {len(shipped)} shipped, {len(rows)} rows, {len(rejected_store)} rejected documented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
