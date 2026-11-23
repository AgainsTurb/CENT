"""Configuration for CENT v3 (autoregressive stroke model).

Kept separate from `cent/config.py` so the v2 diffusion scripts stay runnable for the
record. Every field is explicit and the loader rejects unknown keys, because three runs
were lost to a config field that existed on only one machine.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class V3Config:
    # ---- paths ----
    canonical_root: str = "data/canonical_lines/v1"
    run_name: str = "v3"
    out_dir: str = "runs"

    # ---- data ----
    img_height: int = 64
    canvas_width: int = 1024
    max_chars: int = 64
    max_strokes_char: int = 48
    max_strokes_line: int = 320
    k_refs: int = 5
    pair_prior_path: str = "data/pair_prior.json"
    prior_smoothing: float = 1.0
    # boundary label threshold, in canvas-width units
    join_tau: float = 0.05
    paired_only: bool = False
    limit: int | None = None

    # ---- model ----
    dim: int = 384
    depth: int = 8
    n_heads: int = 8
    n_classes: int = 2680
    style_dim: int = 256
    style_depth: int = 4
    n_style_types: int = 5
    mdn_components: int = 6
    gauss_sigma: float = 1.0
    dropout: float = 0.0

    # ---- training ----
    epochs: int = 60
    batch_size: int = 24
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 8
    amp: bool = True               # bf16 autocast; measured harmless and cheaper
    lambda_stroke: float = 1.0
    lambda_pen: float = 0.5
    lambda_stop: float = 0.5
    lambda_layout: float = 1.0
    # soft-anchor continuity: penalises anchor jerk, not glyph shape
    lambda_cont: float = 0.5
    # CURVE loss: L1 between sampled points along the predicted and target cubics.
    # The MDN likelihood cannot see a wrong bulge (position dims dominate it); this can.
    lambda_curve: float = 0.0
    # per-dimension standardisation of the 8 token dims. Measured means: Delta ~0.004,
    # control points ~0.2. One sigma over mixed scales crushed the Delta dims, so the
    # generated horizontal advance was 6x too small and the line collapsed left.
    stroke_mean: list | None = None
    stroke_std: list | None = None
    ss_probe_every: int = 4          # refresh the self-prefix probe every N steps
    lambda_join: float = 1.0
    lambda_bridge: float = 0.5
    lambda_writer: float = 0.1
    # stage gates: "s1" strokes only, "s2" + layout, "s3" + join planning
    stage: str = "s3"
    scheduled_sampling: float = 0.0
    lambda_abs: float = 1.0
    lambda_count: float = 1.0   # prob. of feeding the model its own prediction
    seed: int = 0

    # ---- inference ----
    temperature: float = 0.9
    top_k: int = 0
    n_chars_override: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "V3Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise KeyError(f"unknown config keys in {path}: {unknown}")
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
