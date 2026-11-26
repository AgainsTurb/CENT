import pathlib
from pypdf import PdfReader
p = pathlib.Path(r"repos/FontRNN/PG_FontRNN_publish.pdf")
r = PdfReader(str(p))
parts=[]
for i,pg in enumerate(r.pages,1):
    parts.append(f"\n\n===== PAGE {i} =====\n"+(pg.extract_text() or ""))
out = pathlib.Path(r"survey/_pdftext/FontRNN_PG_publish.txt")
out.write_text("".join(parts), encoding="utf-8")
print("pages", len(r.pages), "chars", len("".join(parts)))
