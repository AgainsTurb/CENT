import ssl, urllib.request, pathlib
url = "https://iccvm.org/2026/files/papers/109.pdf"
dst = pathlib.Path(r"papers/WriteFreely_ICCV2026.pdf")
req = urllib.request.Request(url, headers={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept":"application/pdf,*/*",
    "Accept-Language":"en-US,en;q=0.9",
})
ctx = ssl.create_default_context()
ctx.set_ciphers("DEFAULT@SECLEVEL=1")
try:
    with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
        data = r.read()
    dst.write_bytes(data)
    print("OK bytes=", len(data), "magic=", data[:4])
except Exception as e:
    print("FAILED:", type(e).__name__, e)
