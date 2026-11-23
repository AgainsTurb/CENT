"""CENT configuration (dataclass + JSON/YAML loading)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CENTConfig:
    # data
    canonical_root: str = "data/canonical/v1"
    lexicon_path: str = "data/lexicon/graphics.txt"
    split_train: str = "train"
    split_val: str = "val"
    img_size: int = 64
    canvas_width: int = 1024    # line canvas width (64 x 1024); the renderer is non-square
    k_refs: int = 5
    max_chars: int = 32         # measured cap: 40 chars of level-2 glyphs need 547 tokens
    max_tokens: int = 1024      # denoiser allocates 1024 positions
    paired_only: bool = False   # first campaign: paired lines only
    n_classes: int = 3755
    n_writers: int = 1020
    # model
    dim: int = 256
    depth: int = 4
    n_heads: int = 4
    n_types: int = 5
    # diffusion
    T: int = 1000
    guidance: float = 2.0
    drop_char_p: float = 0.1
    # losses
    lambda_legibility: float = 0.1
    lambda_verifier_real: float = 1.0
    lambda_image: float = 1.0
    lambda_writer: float = 0.1
    # optimisation
    epochs: int = 6
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 0
    device: str = "cuda"
    amp: str = "none"     # none | bf16 | fp16: autocast for the learned forwards
    # bookkeeping (filled by the runner, not by the config file)
    run_name: str = "cent_r1"
    out_dir: str = "runs"
    limit: int = 0          # >0: cap dataset size, for smoke runs
    dry_run: bool = False   # one batch, no checkpoint, prints shapes

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CENTConfig":
        p = Path(path)
        # utf-8-sig: Windows editors and PowerShell's Set-Content add a BOM,
        # which json.loads rejects with a confusing "line 1 column 1" error.
        text = p.read_text(encoding="utf-8-sig")
        data = json.loads(text) if p.suffix == ".json" else _mini_yaml(text)
        data = {k: v for k, v in data.items() if not k.startswith("_")}   # allow _comment etc.
        known = {f for f in cls().__dict__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


def _mini_yaml(text: str) -> dict:
    """Flat `key: value` YAML subset, so we do not require PyYAML here."""
    out: dict = {}
    for ln in text.splitlines():
        ln = ln.split("#", 1)[0].rstrip()
        if not ln.strip():
            continue
        if ":" not in ln:
            raise ValueError(f"not a flat key: value line: {ln!r}")
        k, v = (s.strip() for s in ln.split(":", 1))
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v.strip("'\"")
    return out
