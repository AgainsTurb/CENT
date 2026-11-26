"""Which culprit? Text truncation, or style-prompt dominance.

Three measurements, ~5 minutes, no diffusion training:

 1. COLLISION: how many lines share their first 8 characters with another line? If the
    8-char prefix is not unique, the model provably cannot know which line to write.
 2. TEXT -> z0: train a small MLP from text ids (8 and 32 chars) to the line latent.
    Relative error against the trivial predictor (predict the mean latent).
 3. ZREF -> z0: same MLP from the STYLE PROMPT (another line by the same writer).
    If this beats the text predictor by a wide margin, the style prompt carries the
    information and the model is right to ignore the text (DEADiff's failure mode).
"""
import sys
sys.path.insert(0, "A:/Papers/CENT")
import json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, ZD = 384, 96
NL = 2000

from cent.data_strokes import StrokeLineDataset


def _refs(self, w, e):
    return torch.zeros(int(self.k_refs), int(self.img_h), 64), torch.ones(int(self.k_refs))


StrokeLineDataset._refs = _refs
base = StrokeLineDataset("A:/Papers/CENT/data/canonical_lines/v1", "train", k_refs=1,
                         img_h=64, canvas_w=1024, paired_only=False, limit=0)
X, T8, T32, W, LEN = [], [], [], [], []
for rid in base.usable[:NL]:
    pts = []
    for s in base.strokes_of(rid):
        a = np.asarray(s, np.float32)
        if len(a) >= 2:
            pts.append(a)
    p = np.concatenate(pts, 0)
    lo, hi = p.min(0), p.max(0)
    p = (p - lo) / np.maximum(hi - lo, 1e-6)
    idx = np.linspace(0, len(p) - 1, P)
    X.append(np.stack([np.interp(idx, np.arange(len(p)), p[:, 0]),
                       np.interp(idx, np.arange(len(p)), p[:, 1])], 1))
    text = base.records[rid].get("text", "") or ""
    a8 = np.zeros(8, np.int64)
    a32 = np.zeros(32, np.int64)
    for k in range(min(len(text), 32)):
        a32[k] = base.char2idx.get(text[k], base.pad_id)
        if k < 8:
            a8[k] = a32[k]
    T8.append(a8)
    T32.append(a32)
    W.append(int(base.records[rid]["writer"]))
    LEN.append(len(text))
X = torch.tensor(np.stack(X)).float().to(DEV)
T8 = torch.tensor(np.stack(T8)).long().to(DEV)
T32 = torch.tensor(np.stack(T32)).long().to(DEV)
W = torch.tensor(W).long().to(DEV)
N = len(X)
print(f"[data] {N} lines, median text length {int(np.median(LEN))}, max {max(LEN)}", flush=True)

print("== 1. TRUNCATION COLLISION ==", flush=True)
pref8 = [tuple(t.cpu().numpy().tolist()) for t in T8]
pref32 = [tuple(t.cpu().numpy().tolist()) for t in T32]
u8 = len(set(pref8))
u32 = len(set(pref32))
print(f"unique 8-char prefixes:  {u8}/{N}  ({u8/N:.1%})", flush=True)
print(f"unique 32-char prefixes: {u32}/{N}  ({u32/N:.1%})", flush=True)


class Enc(nn.Module):
    def __init__(s, h=256):
        super().__init__()
        s.n = nn.Sequential(nn.Conv1d(2, h, 5, 2, 2), nn.GELU(), nn.Conv1d(h, h, 5, 2, 2), nn.GELU())
        s.mu = nn.Conv1d(h, ZD, 1)
        s.lv = nn.Conv1d(h, ZD, 1)

    def forward(s, x):
        f = s.n(x)
        return s.mu(f), s.lv(f)


enc = Enc(h=192).to(DEV)
ck = torch.load("runs/v5/vae.pt", map_location=DEV)
enc.load_state_dict(ck["enc"])
enc.eval()
with torch.no_grad():
    zs = [enc(X[j:j + 512].transpose(1, 2))[0] for j in range(0, N, 512)]
    Z = torch.cat(zs)
Z = (Z - Z.mean()) / Z.std()
print(f"[latent] {tuple(Z.shape)}", flush=True)

ref = torch.zeros(N, dtype=torch.long, device=DEV)
wn = W.cpu().numpy()
for w in np.unique(wn):
    idx = np.where(wn == w)[0]
    ref[idx] = torch.tensor(np.roll(idx, 1) if len(idx) > 1 else idx, device=DEV)
ZREF = Z[ref]


def rel_err(pred, tgt):
    return (torch.norm(pred - tgt, dim=(1, 2)) / torch.norm(tgt, dim=(1, 2))).mean().item()


mean_pred = Z.mean(0, keepdim=True).expand_as(Z)
print(f"[baselines] predict mean latent: rel.err {rel_err(mean_pred, Z):.3f}", flush=True)
print(f"[baselines] copy style prompt  : rel.err {rel_err(ZREF, Z):.3f}", flush=True)


def fit_text(ids, tag, steps=1500):
    emb = nn.Embedding(2680, 128).to(DEV)
    head = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, ZD * 96)).to(DEV)
    opt = torch.optim.AdamW(list(emb.parameters()) + list(head.parameters()), lr=2e-3)
    for s in range(steps):
        i = torch.randint(0, N, (64,), device=DEV)
        e = emb(ids[i]).mean(1)
        pred = head(e).view(-1, ZD, 96)
        loss = F.mse_loss(pred, Z[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(emb(ids).mean(1)).view(-1, ZD, 96)
        r = rel_err(pred, Z)
    print(f"[text->z0 {tag}] rel.err {r:.3f}", flush=True)
    return r


def fit_z(zt, tag, steps=1500):
    m = nn.Sequential(nn.Conv1d(ZD, 256, 3, padding=1), nn.GELU(), nn.Conv1d(256, ZD, 3, padding=1)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        i = torch.randint(0, N, (64,), device=DEV)
        loss = F.mse_loss(m(zt[i]), Z[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        r = rel_err(m(zt), Z)
    print(f"[{tag}->z0] rel.err {r:.3f}", flush=True)
    return r


r8 = fit_text(T8, "8 chars")
r32 = fit_text(T32, "32 chars")
rz = fit_z(ZREF, "style prompt")

print(json.dumps({"unique_prefix_8": u8, "unique_prefix_32": u32, "lines": N,
                  "rel_err_mean": round(rel_err(mean_pred, Z), 3),
                  "rel_err_copy_style": round(rel_err(ZREF, Z), 3),
                  "rel_err_text8": round(r8, 3), "rel_err_text32": round(r32, 3),
                  "rel_err_style->z0": round(rz, 3)}), flush=True)
