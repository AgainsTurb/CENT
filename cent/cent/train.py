"""Full config-driven training loop.

NOT executed in this session (the brief forbids training). It exists so the
architecture is complete and reproducible: `python -m cent.train --config ...`
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import CouchDataset, build_vocab_from_samples, collate
from .data.format_scut import load_scutcouch_textline, load_scutcouch_word_dir
from .model import CENTModel


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_samples(cfg: dict):
    samples = []
    d = cfg["data"]
    if d.get("unipen_dir") and Path(d["unipen_dir"]).exists():
        samples.extend(load_scutcouch_textline(d["unipen_dir"]))
    if d.get("word_dir") and Path(d["word_dir"]).exists():
        samples.extend(load_scutcouch_word_dir(d["word_dir"]))
    if not samples:
        raise SystemExit(
            "No data found. SCUT-COUCH archives are password-protected until the "
            "lab approves the application; see datasets/README.md."
        )
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cent_tiny.yaml")
    ap.add_argument("--out", default="runs/cent")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["train"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    samples = collect_samples(cfg)
    vocab = build_vocab_from_samples(samples)
    writers = sorted({s.writer for s in samples})
    writer_table = {w: i for i, w in enumerate(writers)}

    ds = CouchDataset(vocab, samples=samples, max_points=cfg["data"]["max_points"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    collate_fn=collate, drop_last=True)

    m_cfg = cfg["model"]
    model = CENTModel(
        n_chars=len(vocab),
        n_writers=max(32, len(writers)),
        d=m_cfg["d"], z_dim=m_cfg["z_dim"], vae_d=m_cfg["vae_d"],
        stride=m_cfg["stride"], dit_depth=m_cfg["dit_depth"],
        dit_heads=m_cfg["dit_heads"], n_mix=m_cfg["n_mix"],
        n_steps=m_cfg["n_steps"], device=device,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab.chars, ensure_ascii=False))

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            losses = model(batch, writer_table=writer_table)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"ep {epoch} step {step} "
                      f"total={losses['total'].item():.4f} "
                      f"vae_gmm={losses['vae_gmm'].item():.4f} "
                      f"vae_ctc={losses['vae_ctc'].item():.4f} "
                      f"dit_diff={losses['dit_diff'].item():.4f} "
                      f"dit_bnd={losses['dit_bnd'].item():.4f}")
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                   out / f"ckpt_{epoch:04d}.pt")


if __name__ == "__main__":
    main()
