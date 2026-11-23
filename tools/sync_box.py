"""Make a hash manifest of the local CENT source tree, and verify it on the GPU box.

Three runs were wasted by local/box divergence: a file fixed directly on the box was then
overwritten by a stale local copy, and once a fix was reverted the same way. Discipline is not a
mechanism; a checksum comparison is.

Local:
    python tools/sync_box.py make            -> writes dist/cent_sync.tar.gz + dist/sync_hashes.json

Then upload both with the vastai MCP ssh_transfer, and on the box:
    tar xzf cent_sync.tar.gz -C /workspace/cent
    python tools/sync_box.py verify          -> prints MATCH / MISMATCH per file, exits 1 on any mismatch
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = ("cent", "common", "benchmark", "tests", "tools")
EXTS = (".py", ".json")
DIST = ROOT / "dist"


def hashes(root: Path) -> dict[str, str]:
    out = {}
    for t in TREES:
        d = root / t
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in EXTS and "__pycache__" not in p.parts:
                out[str(p.relative_to(root)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def make() -> int:
    DIST.mkdir(exist_ok=True)
    h = hashes(ROOT)
    (DIST / "sync_hashes.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
    bundle = DIST / "cent_sync.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        for t in TREES:
            if (ROOT / t).exists():
                tf.add(ROOT / t, arcname=t)
        tf.add(DIST / "sync_hashes.json", arcname="sync_hashes.json")
        if (ROOT / "tools").exists():
            tf.add(ROOT / "tools", arcname="tools")
    print(f"{len(h)} files, bundle {bundle.stat().st_size/1024:.0f} KB")
    print(f"upload: {bundle} -> /workspace/cent/cent_sync.tar.gz")
    print(f"        {DIST/'sync_hashes.json'} -> /workspace/sync_hashes.json")
    return 0


REQUIRED_MARKERS = {
    "cent/data_strokes.py": ["render_strokes_splat(_tk", "pos_t"],
    "cent/train_strokes.py": ["w_img", "mean(dim=(1, 2, 3))", "b[\"pos\"]"],
    "cent/models/line_stroke.py": ["local_window", "self.feat"],
    "cent/models/style.py": ["ink = (1.0 - refs)"],
}


def verify() -> int:
    want = json.loads((ROOT / "sync_hashes.json").read_text(encoding="utf-8"))
    have = hashes(ROOT)
    missing = [k for k in want if k not in have]
    extra = [k for k in have if k not in want]
    diff = [k for k in want if k in have and want[k] != have[k]]
    for k in diff:
        print(f"  MISMATCH {k}")
    for k in missing:
        print(f"  MISSING  {k}")
    for k in extra:
        print(f"  EXTRA    {k}")
    ok = not (diff or missing)
    print(f"{len(want)} expected, {len(have)} present, {len(diff)} differ, {len(missing)} missing")
    missing_feat = []
    for rel, marks in REQUIRED_MARKERS.items():
        txt = (ROOT / rel).read_text(encoding="utf-8", errors="ignore") if (ROOT / rel).exists() else ""
        for m in marks:
            if m not in txt:
                missing_feat.append(f"{rel}: {m}")
    for m in missing_feat:
        print(f"  FEATURE MISSING {m}")
    print("TREE MATCHES" if ok and not missing_feat else "TREE DIFFERS -- do not launch")
    return 0 if (ok and not missing_feat) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "make"
    sys.exit(make() if cmd == "make" else verify())
