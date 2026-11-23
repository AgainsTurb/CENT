#!/usr/bin/env python3
"""Print the top Crossref (and arXiv) candidates for a query so a human can judge
which record is the intended work. Read-only: writes nothing.

Usage: python refs/probe.py "some title" ["another title" ...]
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = "CENT-probe/1.0 (mailto:research@example.org)"
ATOM = "{http://www.w3.org/2005/Atom}"


def get(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        return r.read()


def main(queries: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    for q in queries:
        print(f"\n=== {q}")
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query.bibliographic": q, "rows": 5, "mailto": "research@example.org"}
        )
        try:
            d = json.loads(get(url).decode())
            for it in d["message"]["items"]:
                t = (it.get("title") or ["?"])[0]
                print(f"  CR  {t[:78]:78s} | {it.get('DOI')} | "
                      f"{(it.get('container-title') or [''])[0][:34]} | "
                      f"{((it.get('issued', {}).get('date-parts') or [[None]])[0] or [None])[0]} | "
                      f"{it.get('type')}")
        except Exception as e:  # noqa: BLE001
            print("  CR ERR", type(e).__name__, e)
        try:
            u2 = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {"search_query": f"ti:{q}", "max_results": 5}
            )
            root = ET.fromstring(get(u2).decode())
            for e in root.findall(f"{ATOM}entry"):
                t = " ".join((e.find(f"{ATOM}title").text or "").split())
                aid = e.find(f"{ATOM}id").text
                print(f"  ARX {t[:78]:78s} | {aid.rsplit('/', 1)[-1]}")
        except Exception as e:  # noqa: BLE001
            print("  ARX ERR", type(e).__name__, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
