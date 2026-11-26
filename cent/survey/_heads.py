import pathlib, re
for p in sorted(pathlib.Path(r"survey/_pdftext").glob("*.txt")):
    txt = p.read_text(encoding="utf-8")
    # take first 12 non-empty lines after PAGE 1
    body = txt.split("===== PAGE 1 =====",1)[-1]
    lines = [l.strip() for l in body.splitlines() if l.strip()][:8]
    print("="*70)
    print("FILE:", p.stem)
    for l in lines[:6]:
        print("   ", l[:110])
