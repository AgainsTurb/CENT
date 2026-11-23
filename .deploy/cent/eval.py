"""Evaluation for CENT, written to refuse rather than fake.

Every metric returns a `status` field. A metric whose instrument is absent is
reported as NOT_RUN with a reason, never as a number:

  stroke_count_acc  enforced by construction (the lexicon fixes the count). This
                    is reported as ENFORCED, and the *measured* count agreement
                    is computed only if a generated graph is re-lexiconised,
                    which is a no-op for a hard constraint. The informative
                    comparison is a baseline's count accuracy, not ours.
  image_l1 / ssim   measured against the paired target image (soft objective).
  writer_auc        measured from style-encoder embeddings (needs a checkpoint).
  fid               NOT_RUN unless torchvision's Inception is importable AND
                    >=512 samples exist -- FID on 200 samples is noise.
  cer               NOT_RUN without --htr-ckpt. CER is the paper's primary
                    legibility metric and the field's convention; we do not own
                    a frozen recogniser, so we say so instead of substituting.

Usage
  python -m cent.eval --config runs/cent_r1/config.json --ckpt runs/cent_r1/ckpt_last.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import CENTConfig
from .data import PairedStrokeDataset, collate
from .models.diffusion import Diffusion, StrokeVerifier
from .models.stroke_graph import BezierRenderer
from .models.style import StrokeDenoiser, StyleEncoder
from .train import style_tokens_from


def _not_run(reason: str) -> dict:
    return {"status": "NOT_RUN", "reason": reason}


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Global SSIM per item, 7x7 box window (a small, dependency-free version)."""
    w = 7
    mu_a = F.avg_pool2d(a, w, 1)
    mu_b = F.avg_pool2d(b, w, 1)
    va = F.avg_pool2d(a * a, w, 1) - mu_a ** 2
    vb = F.avg_pool2d(b * b, w, 1) - mu_b ** 2
    cov = F.avg_pool2d(a * b, w, 1) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean(dim=(1, 2, 3))


def writer_auc(z: torch.Tensor, w: torch.Tensor) -> float:
    """Pairwise AUC for 'do two embeddings share a writer?'."""
    if len(z) < 4:
        return float("nan")
    sim = F.normalize(z, dim=1) @ F.normalize(z, dim=1).t()
    same = (w.unsqueeze(0) == w.unsqueeze(1))
    eye = torch.eye(len(z), dtype=torch.bool)
    pos, neg = sim[same & ~eye], sim[~same]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos.unsqueeze(0) > neg.unsqueeze(1)).float().mean()
    eq = (pos.unsqueeze(0) == neg.unsqueeze(1)).float().mean()
    return float(gt + 0.5 * eq)


@torch.no_grad()
def evaluate(cfg: CENTConfig, ckpt_path: str, split: str = "test", max_items: int = 128,
             htr_ckpt: str | None = None) -> dict:
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    style = StyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types, n_heads=cfg.n_heads).to(device)
    denoiser = StrokeDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads,
                              n_classes=cfg.n_classes, n_types=cfg.n_types).to(device)
    style.load_state_dict(ck["style"]); denoiser.load_state_dict(ck["denoiser"])
    style.eval(); denoiser.eval()
    renderer = BezierRenderer(size=cfg.img_size).to(device)
    diff = Diffusion(T=cfg.T).to(device)

    ds = PairedStrokeDataset(cfg.canonical_root, split, k_refs=cfg.k_refs, img_size=cfg.img_size)
    ds.keys = ds.keys[:max_items]
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)

    rows, imgs, tgts = [], [], []
    for batch in loader:
        tok = batch["tokens"].to(device); mask = batch["stroke_mask"].to(device)
        char = batch["char_idx"].to(device)
        refs = batch["refs"].to(device); rmask = batch["ref_mask"].to(device)
        z_g, z_l, _ = style(refs, rmask)
        st = style_tokens_from(z_g, z_l)
        gen = diff.sample(denoiser, (tok.shape[0], tok.shape[1], tok.shape[2]), mask,
                          char, st, guidance=cfg.guidance, device=device)
        img = renderer(gen, mask)
        rows.append({"writer": batch["writer"], "z": z_g, "img": img,
                     "target": batch["target_img"].to(device), "has": batch["has_target"].to(device)})
        imgs.append(img); tgts.append(batch["target_img"].to(device))

    img = torch.cat(imgs); tgt = torch.cat(tgts)
    has = torch.cat([r["has"] for r in rows]).bool()
    report: dict = {"split": split, "n": int(img.shape[0]), "ckpt": str(ckpt_path)}

    if has.any():
        report["image_l1"] = {"status": "OK", "value": float(F.l1_loss(img[has], tgt[has]))}
        report["ssim"] = {"status": "OK", "value": float(ssim_simple(img[has], tgt[has]).mean())}
    else:
        report["image_l1"] = _not_run("no paired offline target in this split")
        report["ssim"] = _not_run("no paired offline target in this split")

    report["stroke_count_acc"] = {
        "status": "ENFORCED",
        "note": "the lexicon fixes the stroke count per character, so this is 1.0 by "
                "construction and is not a measurement of the model. Compare a baseline's "
                "measured count accuracy against it.",
    }

    z = torch.cat([r["z"] for r in rows]); w = torch.cat([r["writer"] for r in rows])
    auc = writer_auc(z, w)
    report["writer_auc"] = ({"status": "OK", "value": auc} if auc == auc
                            else _not_run("not enough writers/pairs in this split"))

    try:
        from torchvision.models import inception_v3  # noqa: F401
        have_inception = True
    except Exception:
        have_inception = False
    if not have_inception:
        report["fid"] = _not_run("torchvision Inception unavailable in this environment")
    elif img.shape[0] < 512:
        report["fid"] = _not_run(f"only {img.shape[0]} samples; FID is not meaningful below 512")
    else:
        report["fid"] = _not_run("Inception available but the FID implementation is not wired in "
                                 "this revision; wire it before reporting")

    if htr_ckpt is None:
        report["cer"] = _not_run("no frozen HTR checkpoint supplied (--htr-ckpt). CER is the "
                                 "paper's primary legibility metric; it is not substituted.")
    else:
        report["cer"] = _not_run("HTR checkpoint loading is not implemented in this revision")

    report["soec"] = _not_run("stroke-order equivalence needs stroke-type labels, which the "
                              "lexicon does not provide; the merge-rule implementation is pending")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-items", type=int, default=128)
    ap.add_argument("--htr-ckpt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = CENTConfig.load(a.config)
    rep = evaluate(cfg, a.ckpt, a.split, a.max_items, a.htr_ckpt)
    text = json.dumps(rep, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
