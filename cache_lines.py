"""Pre-tokenise a split once, so training never pays the per-sample Python cost again.

After vectorising the verifier grouping, a step takes 51 ms at batch 16 and 161 ms at batch 64 --
400+ lines/s. `LineTokenDataset.__getitem__` cannot feed that: it runs ~350 `fit_stroke` calls per
line (one least-squares per stroke) plus a Python loop over characters, all on CPU, per sample.

So the tokenisation is done ONCE, offline, into flat arrays; training then does memmap reads,
`np.take` and `torch.from_numpy`, which is C-speed. This is the same trade the big diffusion
pipelines make when they pre-extract latents.

Cost: ~40k lines x ~415 tokens x 8 floats = ~530 MB in fp16, ~1.05 GB in fp32. Store fp16 and
cast per batch: the payloads are normalised control points, so fp16 precision (about 3 decimal
digits) is finer than the rasteriser's pixel grid.

Usage
    python -m cent.cache_lines --canonical data/canonical_lines/v1 --split train \
        --lexicon data/lexicon/graphics.txt --out data/cache/train --paired-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_lines import LineTokenDataset, collate_lines
from .models.line import PAYLOAD


def build_cache(canonical_root: str, split: str, out_dir: str, lexicon: dict,
                 k_refs: int = 5, img_h: int = 64, canvas_w: int = 1024,
                 max_chars: int = 32, max_tokens: int = 1024,
                 paired_only: bool = False, limit: int = 0, dtype=np.float16) -> dict:
    """Tokenise every usable line once. Returns a summary of what was written."""
    ds = LineTokenDataset(canonical_root, split, lexicon=lexicon, k_refs=k_refs, img_h=img_h,
                          canvas_w=canvas_w, max_chars=max_chars, max_tokens=max_tokens,
                          paired_only=paired_only, limit=limit)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokens, kinds, cpos, slots, masks = [], [], [], [], []
    joins, n_conns, writers, char_ids, refs, rm = [], [], [], [], [], []
    targets, texts = [], []
    n_tok_max = 0
    for i in range(len(ds)):
        s = ds[i]
        n_tok_max = max(n_tok_max, s["tokens"].shape[0])
        tokens.append(s["tokens"].numpy().astype(dtype))
        kinds.append(s["kind"].numpy().astype(np.int16))
        cpos.append(s["char_pos"].numpy().astype(np.int32))
        slots.append(s["char_slot"].numpy().astype(np.int16))
        masks.append(s["mask"].numpy().astype(np.uint8))
        joins.append(s["join"].numpy().astype(np.uint8))
        n_conns.append(int(s["n_conn"]))
        writers.append(int(s["writer_idx"]))
        char_ids.append(s["char_ids"].numpy().astype(np.int32))
        refs.append((s["refs"].numpy() * 255).astype(np.uint8))   # 0..1 -> u8 (paper=255)
        rm.append(s["ref_mask"].numpy().astype(np.uint8))
        targets.append((s["target_img"].numpy() * 255).astype(np.uint8))
        texts.append(s["text"])

    # pad the variable-length axes into rectangular arrays
    T = n_tok_max
    C = max(c.shape[0] for c in char_ids)
    K = k_refs
    tok = np.zeros((len(ds), T, PAYLOAD), dtype)
    kind = np.zeros((len(ds), T), np.int16)
    cpos_a = np.zeros((len(ds), T), np.int32)
    slot_a = np.zeros((len(ds), T), np.int16)
    mask_a = np.zeros((len(ds), T), np.uint8)
    for i in range(len(ds)):
        n = tokens[i].shape[0]
        tok[i, :n] = tokens[i]
        kind[i, :n] = kinds[i]
        cpos_a[i, :n] = cpos[i]
        slot_a[i, :n] = slots[i]
        mask_a[i, :n] = masks[i]
    cid = np.full((len(ds), C), -1, np.int32)
    for i, c in enumerate(char_ids):
        cid[i, : c.shape[0]] = c

    np.savez(out / "tokens.npz",
             tokens=tok, kind=kind, char_pos=cpos_a, char_slot=slot_a, mask=mask_a,
             join=np.stack(joins), n_conn=np.array(n_conns, np.int32),
             writer=np.array(writers, np.int32), char_ids=cid,
             refs=np.stack(refs), ref_mask=np.stack(rm),
             target_img=np.stack(targets),
             lengths=np.array([t.shape[0] for t in tokens], np.int32))
    (out / "meta.json").write_text(json.dumps({
        "split": split, "n": len(ds), "max_tokens": T, "max_chars": C, "k_refs": K,
        "img_h": img_h, "canvas_w": canvas_w, "dtype": str(dtype),
        "paired_only": paired_only, "n_too_long": ds.n_too_long, "n_unpaired": ds.n_unpaired,
        "texts": texts,
    }, ensure_ascii=False), encoding="utf-8")
    return {"n": len(ds), "max_tokens": T, "bytes_on_disk": sum(
        f.stat().st_size for f in out.iterdir() if f.is_file()), "path": str(out)}


class CachedLineDataset(Dataset):
    """Reads the pre-tokenised arrays. No per-sample Python, no image decoding."""

    def __init__(self, cache_dir: str, length_trim: bool = True):
        p = Path(cache_dir)
        self.z = dict(np.load(p / "tokens.npz"))
        self.meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        self.n = int(self.meta["n"])
        # trimming to the observed maximum halves the padding work at no cost to correctness
        self.T = int(self.z["lengths"].max()) if length_trim else self.z["tokens"].shape[1]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        z = self.z
        n = int(z["lengths"][i])
        return {
            "tokens": torch.from_numpy(z["tokens"][i, :n].astype(np.float32)),
            "kind": torch.from_numpy(z["kind"][i, :n].astype(np.int64)),
            "char_pos": torch.from_numpy(z["char_pos"][i, :n].astype(np.int64)),
            "char_slot": torch.from_numpy(z["char_slot"][i, :n].astype(np.int64)),
            "mask": torch.from_numpy(z["mask"][i, :n].astype(np.float32)),
            "join": torch.from_numpy(z["join"][i].astype(np.float32)),
            "n_conn": torch.tensor(int(z["n_conn"][i]), dtype=torch.long),
            "refs": torch.from_numpy(z["refs"][i].astype(np.float32) / 255.0),
            "ref_mask": torch.from_numpy(z["ref_mask"][i].astype(np.float32)),
            "target_img": torch.from_numpy(z["target_img"][i].astype(np.float32) / 255.0),
            "writer_idx": torch.tensor(int(z["writer"][i]), dtype=torch.long),
            "char_ids": torch.from_numpy(z["char_ids"][i].astype(np.int64)),
            "text": self.meta["texts"][i],
        }


def main() -> int:
    from common.lexicon import load_lexicon

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--lexicon", default="data/lexicon/graphics.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paired-only", action="store_true")
    ap.add_argument("--max-chars", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rep = build_cache(a.canonical, a.split, a.out, load_lexicon(a.lexicon),
                      max_chars=a.max_chars, paired_only=a.paired_only, limit=a.limit)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
