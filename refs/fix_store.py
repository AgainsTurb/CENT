#!/usr/bin/env python3
"""Correct the reference store: mark the title-squatting match as REJECTED so the
record does not later read as a verified citation."""
from __future__ import annotations

import json
from pathlib import Path

P = Path(__file__).resolve().parent / "verified_refs.json"
d = json.loads(P.read_text(encoding="utf-8"))
for e in d["entries"]:
    if e["key"] == "fid":
        e["status"] = "REJECTED"
        e["rejection_reason"] = (
            "Crossref title search matched a title-squatting record "
            "(DOI 10.18034/ajase.v8i1.9, Asian Journal of Applied Science and Engineering) "
            "that is not Heusel et al.'s NeurIPS 2017 paper. Title similarity alone is not "
            "verification; the intended work is cited from its arXiv record (1706.08500) instead."
        )
        e["superseded_by"] = "arxiv:1706.08500"
counts = d["counts"]
counts["accepted"] = sum(1 for e in d["entries"] if e.get("status") == "ACCEPTED")
counts["rejected"] = sum(1 for e in d["entries"] if e.get("status") == "REJECTED")
P.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
print("fid marked REJECTED;", counts)
