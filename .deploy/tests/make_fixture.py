"""Build a tiny synthetic canonical dataset for adapter smoke tests.

Every baseline adapter can be exercised without the real corpus:

    python tests/make_fixture.py --out .fixture/canonical --writers 8 --chars 12

Writes the canonical layout described in common/canonical.py. The glyphs are
random blobs, not characters: this fixture proves data plumbing, never accuracy.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]      # A:\Papers\CENT
sys.path.insert(0, str(PROJ.parent))             # so `import CENT.common...` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent))

from CENT.common.canonical import build  # noqa: E402
from test_casia import write_gnt, write_pot  # noqa: E402

# 12 GB2312 level-1 characters, as real characters so the charset is plausible.
CHARS = list("\u65e5\u76ee\u5927\u4eba\u672a\u672b\u4e0a\u4e0b\u5c71\u6c34\u706b\u6728")


def blob(seed: int, size: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 255, np.uint8)
    for _ in range(4):
        r0, c0 = rng.integers(0, size - 6, 2)
        img[r0: r0 + 5, c0: c0 + 6] = 0
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--writers", type=int, default=8)
    ap.add_argument("--chars", type=int, default=len(CHARS))
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--online", action="store_true", help="also emit online samples")
    ap.add_argument("--no-lexicon", action="store_true",
                    help="skip the lexicon so the T2 agreement statistic is absent")
    a = ap.parse_args()

    lexicon = None
    if not a.no_lexicon:
        lex_path = PROJ / "data" / "lexicon" / "graphics.txt"
        if lex_path.exists():
            from CENT.common.lexicon import load_lexicon
            lexicon = load_lexicon(lex_path)

    tmp = Path(tempfile.mkdtemp(prefix="cent_fixture_src_"))
    off, on = tmp / "gnt", tmp / "pot"
    off.mkdir()
    chars = CHARS[: a.chars]
    for w in range(1, a.writers + 1):
        write_gnt(off / f"{w:03d}.gnt", [(c, blob(1000 * w + i)) for i, c in enumerate(chars)])
    if a.online:
        on.mkdir()
        for w in range(1, a.writers + 1):
            samples = []
            for i, c in enumerate(chars):
                strokes = [np.array([[i * 5, j * 4], [i * 5 + 8, j * 4 + 6]], np.int16) for j in range(2)]
                samples.append((c, strokes))
            write_pot(on / f"{w:03d}.pot", samples)

    meta = build(off, on if a.online else None, a.out, charset=chars, img_size=a.img_size,
                 lexicon=lexicon)
    print(f"fixture written to {Path(a.out).resolve()}")
    print(f"  classes={meta['num_classes']} counts={meta['counts']} img={meta['img_size']}")
    lex = meta.get("lexicon", {})
    print(f"  lexicon: supplied={lex.get('supplied')} agreement={lex.get('agreement')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
