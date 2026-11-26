import sys, pathlib
from pypdf import PdfReader
src = pathlib.Path(r"papers")
out = pathlib.Path(r"survey/_pdftext")
for p in sorted(src.glob("*.pdf")):
    try:
        r = PdfReader(str(p))
        parts = []
        for i, pg in enumerate(r.pages, 1):
            t = pg.extract_text() or ""
            parts.append(f"\n\n===== PAGE {i} =====\n{t}")
        txt = "".join(parts)
        dst = out / (p.stem + ".txt")
        dst.write_text(txt, encoding="utf-8")
        print(f"{p.name}: pages={len(r.pages)} chars={len(txt)} -> {dst.name}")
    except Exception as e:
        print(f"{p.name}: FAILED {e}")
