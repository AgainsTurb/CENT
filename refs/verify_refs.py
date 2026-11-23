#!/usr/bin/env python3
"""Resolve every candidate reference against Crossref and arXiv.

Deterministic given the same candidates and the same registry replies.

Three honest states, never conflated:
  ACCEPTED      a registry returned a title with normalised similarity >= ACCEPT_RATIO
  NO_MATCH      every registry answered, but nothing passed the threshold
  LOOKUP_FAILED every registry request errored (network/rate limit); NOT evidence of absence

Outputs: refs/verified_refs.json, refs/verification_report.md
Usage:   python refs/verify_refs.py [--resume]
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACCEPT_RATIO = 0.87
UA = "CENT-reference-verifier/1.1 (mailto:research@example.org)"
ATOM = "{http://www.w3.org/2005/Atom}"
CROSSREF_SLEEP = 1.6
ARXIV_SLEEP = 3.5
RETRIES = 5


def norm(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[\u2010-\u2015]", "-", t)
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def fetch(url: str, timeout: int = 40) -> tuple[bytes | None, str | None]:
    """GET with backoff. Returns (body, error). One of the two is always None."""
    delay = 2.0
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
                return r.read(), None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            return None, last
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(delay)
            delay *= 2
    return None, last


def crossref(title: str) -> tuple[dict | None, str | None]:
    q = urllib.parse.urlencode(
        {"query.bibliographic": title, "rows": 5, "mailto": "research@example.org"}
    )
    url = f"https://api.crossref.org/works?{q}"
    body, err = fetch(url)
    time.sleep(CROSSREF_SLEEP)
    if body is None:
        return None, err
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return None, f"json: {e}"
    best = None
    for item in payload.get("message", {}).get("items", []):
        titles = item.get("title") or []
        if not titles or not titles[0]:
            continue
        s = ratio(title, titles[0])
        if best is None or s > best["score"]:
            best = {
                "source": "crossref",
                "query_url": url,
                "matched_title": titles[0],
                "score": round(s, 4),
                "doi": item.get("DOI"),
                "type": item.get("type"),
                "venue": (item.get("container-title") or [None])[0],
                "year": ((item.get("issued", {}).get("date-parts") or [[None]])[0] or [None])[0],
                "authors": [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in (item.get("author") or [])[:8]
                ],
                "publisher": item.get("publisher"),
            }
    return best, None


def arxiv(title: str) -> tuple[dict | None, str | None]:
    q = urllib.parse.urlencode(
        {"search_query": f'ti:"{title}"', "max_results": 5, "sortBy": "relevance"}
    )
    url = f"http://export.arxiv.org/api/query?{q}"
    body, err = fetch(url)
    time.sleep(ARXIV_SLEEP)
    if body is None:
        return None, err
    try:
        root = ET.fromstring(body.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return None, f"xml: {e}"
    best = None
    for entry in root.findall(f"{ATOM}entry"):
        t_el = entry.find(f"{ATOM}title")
        if t_el is None or not t_el.text:
            continue
        cand = " ".join(t_el.text.split())
        s = ratio(title, cand)
        if best is None or s > best["score"]:
            aid = entry.find(f"{ATOM}id")
            pub = entry.find(f"{ATOM}published")
            names = [
                (a.find(f"{ATOM}name").text if a.find(f"{ATOM}name") is not None else "")
                for a in entry.findall(f"{ATOM}author")
            ]
            best = {
                "source": "arxiv",
                "query_url": url,
                "matched_title": cand,
                "score": round(s, 4),
                "arxiv_id": (aid.text.rsplit("/", 1)[-1] if aid is not None and aid.text else None),
                "venue": "arXiv preprint",
                "year": (pub.text[:4] if pub is not None and pub.text else None),
                "authors": [n for n in names[:8] if n],
                "doi": None,
                "type": "preprint",
            }
    return best, None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    resume = "--resume" in sys.argv
    spec = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    prior: dict[str, dict] = {}
    out_path = ROOT / "verified_refs.json"
    if resume and out_path.exists():
        for r in json.loads(out_path.read_text(encoding="utf-8"))["entries"]:
            if r.get("status") == "ACCEPTED":
                prior[r["key"]] = r

    entries = []
    for c in spec["candidates"]:
        if c["key"] in prior:
            entries.append(prior[c["key"]])
            continue
        rec = {
            "key": c["key"],
            "query_title": c["title"],
            "hint": c.get("hint"),
            "kind": c.get("kind"),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        errors = []
        cr, e1 = crossref(c["title"])
        ax, e2 = (None, None)
        best = cr if (cr and cr["score"] >= ACCEPT_RATIO) else None
        if best is None:
            ax, e2 = arxiv(c["title"])
            if ax and ax["score"] >= ACCEPT_RATIO:
                best = ax
        for e in (e1, e2):
            if e:
                errors.append(e)

        if best is not None:
            rec.update(
                {
                    "status": "ACCEPTED",
                    **{k: best.get(k) for k in (
                        "matched_title", "score", "doi", "arxiv_id", "venue", "year",
                        "authors", "source", "query_url", "type")},
                }
            )
        elif errors and (cr is None or ax is None):
            # at least one registry never answered; we cannot assert absence
            fallback = cr or ax
            rec.update(
                {
                    "status": "LOOKUP_FAILED",
                    "errors": errors,
                    "best_partial_score": (fallback or {}).get("score"),
                }
            )
        else:
            fb = cr or ax or {}
            rec.update(
                {
                    "status": "NO_MATCH",
                    "best_partial_score": fb.get("score"),
                    "best_partial_title": fb.get("matched_title"),
                    "errors": errors,
                }
            )
        entries.append(rec)
        print(f"{rec['status']:13s} {rec['key']:16s} {rec.get('score') or rec.get('best_partial_score')}")

    acc = [r for r in entries if r["status"] == "ACCEPTED"]
    nomatch = [r for r in entries if r["status"] == "NO_MATCH"]
    failed = [r for r in entries if r["status"] == "LOOKUP_FAILED"]
    out_path.write_text(
        json.dumps(
            {
                "accept_ratio": ACCEPT_RATIO,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "counts": {
                    "total": len(entries),
                    "accepted": len(acc),
                    "no_match": len(nomatch),
                    "lookup_failed": len(failed),
                },
                "entries": entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Reference verification report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"accept ratio ≥ {ACCEPT_RATIO} · Crossref → arXiv · "
        f"**{len(acc)}/{len(entries)} accepted, {len(nomatch)} no match, {len(failed)} lookup failed**",
        "",
        "| key | status | score | matched title | identifier | venue | year |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in entries:
        rid = r.get("doi") or r.get("arxiv_id") or ""
        lines.append(
            f"| {r['key']} | {r['status']} | "
            f"{r.get('score') or r.get('best_partial_score') or ''} | "
            f"{(r.get('matched_title') or r.get('query_title'))[:64]} | {rid} | "
            f"{(r.get('venue') or '')[:30]} | {r.get('year') or ''} |"
        )
    if nomatch or failed:
        lines += ["", "## Excluded from the manuscript", ""]
        for r in nomatch + failed:
            lines.append(
                f"- `{r['key']}` — “{r['query_title'][:70]}” → **{r['status']}** "
                f"(best {r.get('best_partial_score')}; {', '.join(r.get('errors') or []) or 'no error recorded'})"
            )
    (ROOT / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(acc)}/{len(entries)} accepted · {len(nomatch)} no match · {len(failed)} lookup failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
