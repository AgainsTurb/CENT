"""Dataset acquisition: detect → download → verify → extract.

Designed for a fresh GPU box: run any stage and the data layer will pull what it
needs. On a box that already has the files, every step short-circuits.

The official NLPR links served the WPTT archives over plain HTTP(S) with no
password (verified), so those are fetched automatically. SCUT-COUCH is *not*
fetched: its archives are AES-encrypted and require an approved application.
"""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

WPTT_URLS = {
    "train": "https://nlpr.ia.ac.cn/Download/Online/WPTTData/WPTT2.0-Train.zip",
    "test": "https://nlpr.ia.ac.cn/Download/Online/WPTTData/WPTT2.0-Test.zip",
}
EXPECTED_PAGES = {"train": 1678, "test": 420}
DOWNLOAD_PAGE = "https://nlpr.ia.ac.cn/databases/handwriting/Download.html"
UA = "Mozilla/5.0 (compatible; CENT-dataloader/0.1)"


def _log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def _http_get(url: str, dest: Path, *, resume: bool = True,
              timeout: int = 120, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        if resume and have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if (resume and have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                t0 = time.time()
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if total and have % (16 << 20) < (1 << 20):
                            pct = 100 * have / total
                            _log(f"  {dest.name}: {have/1e6:.0f}/{total/1e6:.0f} MB "
                                 f"({pct:.0f}%)")
            if dest.exists() and dest.stat().st_size > 0:
                return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            _log(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def discover_links() -> dict[str, list[str]]:
    """Scrape the official download page for archive links (best effort)."""
    out: dict[str, list[str]] = {"wptt": [], "pot": [], "other": []}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(DOWNLOAD_PAGE, headers={"User-Agent": UA}), timeout=60
        ) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        _log(f"could not read the download page ({e}); using built-in URLs")
        return out
    for href in re.findall(r'href="([^"]+\.zip)"', html):
        if "WPTT" in href.upper():
            out["wptt"].append(href)
        elif "POT" in href.upper() or "OLHWDB1" in href.upper():
            out["pot"].append(href)
        else:
            out["other"].append(href)
    return out


def _zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            return bad is None
    except Exception:  # noqa: BLE001
        return False


def _extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def ensure_wptt(root: Path = Path("datasets"), *, allow_download: bool = True) -> dict:
    """Make CASIA-OLHWDB 2.0 pages available. Returns a status dict."""
    base = root / "CASIA-OLHWDB2.0"
    status = {"root": str(base), "action": [], "pages": {}}

    for split in ("train", "test"):
        d = base / split
        have = len(list(d.glob("*.wptt"))) if d.exists() else 0
        status["pages"][split] = have
        if have >= int(EXPECTED_PAGES[split] * 0.95):
            status["action"].append(f"{split}: already extracted ({have} pages)")
            continue

        zip_path = root / f"WPTT2.0-{split.capitalize()}.zip"
        if not zip_path.exists() or not _zip_ok(zip_path):
            if not allow_download:
                raise RuntimeError(f"{split}: missing and downloads disabled")
            _log(f"{split}: downloading {WPTT_URLS[split]}")
            _http_get(WPTT_URLS[split], zip_path)
            if not _zip_ok(zip_path):
                raise RuntimeError(f"{split}: downloaded zip is corrupt")
            status["action"].append(f"{split}: downloaded")

        _log(f"{split}: extracting -> {d}")
        _extract(zip_path, d)
        have = len(list(d.glob("*.wptt")))
        status["pages"][split] = have
        status["action"].append(f"{split}: extracted ({have} pages)")

    return status


def find_char_dataset(root: Path = Path("datasets")) -> Path | None:
    """Locate an OLHWDB 1.x character corpus if the user supplied one.

    Not downloaded automatically: the archive names are not stable and the
    character stage is optional. Drop the `.pot` folders under
    ``datasets/CASIA-OLHWDB1.0-1.2`` (or any dir with a ``.pot`` file) and this
    finds them.
    """
    for cand in (root / "CASIA-OLHWDB1.0-1.2", root / "OLHWDB1", root):
        if not cand.exists():
            continue
        if next(cand.rglob("*.pot"), None) is not None:
            return cand
    return None


def status(root: Path = Path("datasets")) -> dict:
    base = root / "CASIA-OLHWDB2.0"
    pages = {s: len(list((base / s).glob("*.wptt"))) for s in ("train", "test")}
    return {
        "wptt": pages,
        "wptt_ready": pages["train"] > 0 and pages["test"] > 0,
        "chars": str(find_char_dataset(root) or ""),
        "cache": str(root / "processed"),
        "couch": str(root / "COUCH09.zip") if (root / "COUCH09.zip").exists() else "",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ensure CENT datasets are on disk")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root)
    _log("checking WPTT corpus")
    st = ensure_wptt(root, allow_download=not a.no_download)
    for act in st["action"]:
        _log(f"  {act}")
    chars = find_char_dataset(root)
    _log(f"character corpus: {chars if chars else 'not found (Stage 1 will be skipped)'}")
    _log(f"page counts: {st['pages']}")
    print("\nvisual/data status:")
    for k, v in status(root).items():
        print(f"  {k:12s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
