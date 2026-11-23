"""Run a COMPOSED line baseline: per-character generations assembled into a line.

Which baselines need this: any character- or word-level generator. The paper reports them
under a composed label, because they have no layout and no inter-character connectivity of
their own. Two layouts are available (see `compose_line`): `uniform` is the mechanically
stitched floor the line-level papers criticise; `style` uses advance/baseline/jitter
estimated from the same k reference lines that CENT conditions on, which is the fair
comparison. Report which one was used or the number means nothing.

HONESTY RULE, and the reason this file has no `--pred-text` option:
a composed baseline is TOLD which characters to write, so scoring its output against that
same string would return ~100% accuracy by construction and measure nothing. Content
accuracy for a composed baseline therefore requires a recogniser run on the assembled
IMAGE. This runner emits images and a manifest only; it refuses to emit a predicted
transcript derived from the intent.

A `--stub` run renders characters from a glyph table instead of a model. It exists to test
the assembly path. Its manifest and its output directory both say STUB, and a stub line is
not a baseline result.

Usage
  python -m benchmark.run_composed --gt data/canonical_lines/v1 --split test \
      --stub --layout style --out runs/composed_stub
  python -m benchmark.run_composed --gt ... --chars runs/ganwriting/chars --layout style \
      --out runs/composed_ganwriting
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.canonical_lines import LineDataset, _save_png, _read_png
from common.content_render import load_glyphs, render_text_varwidth
from benchmark.compose_line import compose, estimate_layout


def _char_image(char: str, writer: int, cache: Path | None,
                glyphs: dict[int, np.ndarray], cell: int) -> tuple[np.ndarray, str]:
    """Prefer a generated character image; fall back to the glyph, and say which."""
    if cache is not None:
        for cand in (cache / f"{writer}" / f"{ord(char):05x}.png",
                     cache / f"{ord(char):05x}.png"):
            if cand.exists():
                return _read_png(cand), "generated"
    return render_text_varwidth(char, glyphs, cell=cell), "glyph"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chars", default=None, help="directory of per-character images")
    ap.add_argument("--stub", action="store_true", help="render characters from glyphs")
    ap.add_argument("--glyphs", default="baselines/diffbrush/files/unifont.pickle")
    ap.add_argument("--layout", choices=["uniform", "style"], default="style")
    ap.add_argument("--k-refs", type=int, default=5)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = LineDataset(a.gt, a.split)
    ds_traj = ds.traj
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cache = Path(a.chars) if a.chars else None
    glyphs = load_glyphs(a.glyphs) if (a.stub or cache is None) else {}
    rng = np.random.default_rng(a.seed)

    records = ds.records[: a.limit] if a.limit else ds.records
    by_writer = ds.by_writer()
    sources: dict[str, int] = {}
    modes: dict[str, int] = {}
    rows = []

    for r in records:
        text = r["text"]
        writer = int(r["writer"])
        # style references: k other lines from the same writer, as images when available
        refs = []
        for j in by_writer.get(writer, []):
            if j >= ds.__len__() or ds.records[j]["id"] == r["id"]:
                continue
            img = ds.image_of(j)
            if img is not None:
                refs.append(img)
            if len(refs) >= a.k_refs:
                break
        layout = estimate_layout(refs, len(text), a.height) if refs else None

        chars, src = [], []
        for ch in text:
            img, s = _char_image(ch, writer, cache, glyphs, a.height)
            chars.append(img)
            src.append(s)
        line, meta = compose(chars, layout, width=a.width, height=a.height,
                             rng=rng, mode=a.layout)
        _save_png(out / "images" / f"{r['id']}.png", line)
        for s in src:
            sources[s] = sources.get(s, 0) + 1
        modes[meta["mode"]] = modes.get(meta["mode"], 0) + 1
        rows.append({"id": r["id"], "writer": writer, "text": text,
                     "layout": meta, "char_sources": sorted(set(src)),
                     "image": f"images/{r['id']}.png"})

    manifest = {
        "kind": "composed-line",
        "STUB": bool(a.stub),
        "stub_warning": ("Characters were rendered from a glyph table, not by a model. "
                         "This exercises the assembly path and is NOT a baseline result."
                         if a.stub else None),
        "canonical": a.gt, "split": a.split,
        "char_source": a.chars,
        "layout_requested": a.layout,
        "layouts_used": modes,
        "char_image_sources": sources,
        "k_refs": a.k_refs, "width": a.width, "height": a.height,
        "n_lines": len(rows),
        "content_scoring": ("NOT SELF-SCORED: a composed baseline is given the target string, "
                            "so AR/CR must come from a recogniser on the assembled image"),
        "has_trajectories": ds_traj is not None,
        "lines": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    tag = "STUB " if a.stub else ""
    print(f"[{tag}composed] {len(rows)} lines -> {out}")
    print(f"  layout used: {modes}  char sources: {sources}")
    if a.stub:
        print("  NOTE: stub run, not a baseline result. Provide --chars from a real generator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
