#!/usr/bin/env python3
"""Bounded single-lookup verifier: one Crossref attempt, one arXiv attempt,
short timeouts, no long backoff. Used for a small set of load-bearing keys.

Usage: python refs/verify_one.py key1 key2 ...
Merges results into refs/verified_refs.json (never downgrades an ACCEPTED entry).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_refs as vr  # noqa: E402
from verify_refs import ACCEPT_RATIO, arxiv, crossref  # noqa: E402


def quick_fetch(url: str, timeout: int = 15, _retries: int = 2):
    """Bounded replacement for verify_refs.fetch: short timeout, at most 2 tries."""
    last = None
    for _ in range(_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": vr.UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
                return r.read(), None
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    return None, last


vr.fetch = quick_fetch  # crossref()/arxiv() resolve fetch through the module global
vr.CROSSREF_SLEEP = 0.6
vr.ARXIV_SLEEP = 1.5


ROOT = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    spec = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    by_key = {c["key"]: c for c in spec["candidates"]}
    store_path = ROOT / "verified_refs.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    entries = {e["key"]: e for e in store["entries"]}

    for key in argv:
        c = by_key.get(key)
        if not c:
            print(f"{key}: not in candidates")
            continue
        rec = entries.get(key, {})
        if rec.get("status") == "ACCEPTED":
            print(f"{key}: already ACCEPTED ({rec.get('score')})")
            continue
        cr, e1 = crossref(c["title"])
        time.sleep(0.6)
        best = cr if (cr and cr["score"] >= ACCEPT_RATIO) else None
        ax, e2 = None, None
        if best is None:
            ax, e2 = arxiv(c["title"])
            if ax and ax["score"] >= ACCEPT_RATIO:
                best = ax
        new = {
            "key": key,
            "query_title": c["title"],
            "hint": c.get("hint"),
            "kind": c.get("kind"),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if best:
            new["status"] = "ACCEPTED"
            for f in ("matched_title", "score", "doi", "arxiv_id", "venue", "year",
                      "authors", "source", "query_url", "type"):
                new[f] = best.get(f)
        else:
            new["status"] = "NO_MATCH"
            new["best_partial_score"] = (cr or {}).get("score")
            new["best_partial_title"] = (cr or {}).get("matched_title")
        entries[key] = new
        print(f"{key}: {new['status']} {new.get('score')} "
              f"{(new.get('matched_title') or '')[:70]} {new.get('doi') or new.get('arxiv_id') or ''}")

    ordered = [entries[c["key"]] for c in spec["candidates"] if c["key"] in entries]
    acc = [e for e in ordered if e.get("status") == "ACCEPTED"]
    store["entries"] = ordered
    store["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store["counts"] = {
        "total": len(ordered),
        "accepted": len(acc),
        "no_match": sum(1 for e in ordered if e.get("status") == "NO_MATCH"),
        "lookup_failed": sum(1 for e in ordered if e.get("status") == "LOOKUP_FAILED"),
    }
    store_path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\naccepted {store['counts']['accepted']}/{store['counts']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
