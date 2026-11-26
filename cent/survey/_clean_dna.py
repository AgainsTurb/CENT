import pathlib, re
p = pathlib.Path(r"survey/_pdftext/DNA_WACV2026.txt")
raw = p.read_bytes()
print("bytes:", len(raw), "nul:", raw.count(b"\x00"))
txt = raw.decode("utf-8", errors="ignore")
txt = "".join(ch for ch in txt if ch == "\n" or ch == "\t" or ord(ch) >= 32)
p.with_name("DNA_WACV2026_clean.txt").write_text(txt, encoding="utf-8")
print("clean chars:", len(txt))
print(txt[:1200])
