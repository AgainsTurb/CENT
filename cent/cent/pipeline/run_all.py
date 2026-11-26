"""Run the whole CENT pipeline in order, with a visual check after each stage.

    python -m cent.pipeline.run_all --smoke        # tiny, verifies plumbing
    python -m cent.pipeline.run_all               # real run (GPU box)
    python -m cent.pipeline.run_all --stop-after stage3

Each stage writes ``metrics.json`` **and** PNGs under its run directory. The
driver prints the paths; metrics alone are not treated as success.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import stage_boundaries, stage_dit, stage_eval, stage_vae


def _parse(stage_mod, argv):
    return stage_mod.build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CENT full pipeline")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny limits + a couple of steps, to verify plumbing only")
    ap.add_argument("--stages", default="1,2,3,4,6")
    ap.add_argument("--stop-after", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        # a smoke run is a plumbing check: a gate failing after 2 steps is expected
        a.force = True

    runs = Path(a.runs)
    vae_ckpt = runs / "stage2_vae" / "vae.pt"
    dit_ckpt = runs / "stage4_dit" / "dit.pt"
    wanted = [s.strip() for s in a.stages.split(",") if s.strip()]
    if a.smoke and "5" not in wanted:
        pass

    common = ["--root", a.root, "--device", a.device]
    if a.no_download:
        common.append("--no-download")
    if a.force:
        common.append("--force")
    elif a.strict:
        common.append("--strict")

    smoke_vae = ["--epochs", "1", "--steps", "2", "--limit-pages", "3",
                 "--limit-lines", "6", "--batch-size", "2", "--log-every", "1",
                 "--d", "32", "--z-dim", "16", "--stride", "8", "--n-mix", "4"]
    smoke_dit = ["--epochs", "1", "--steps", "2", "--limit-pages", "3",
                 "--limit-lines", "6", "--batch-size", "2", "--log-every", "1",
                 "--d", "32", "--z-dim", "16", "--stride", "8", "--n-mix", "4",
                 "--depth", "1", "--sample-steps", "2"]
    smoke_bnd = ["--limit-pages", "3", "--limit-lines", "6", "--d", "32",
                 "--z-dim", "16", "--n-mix", "4"]

    def maybe_stop(tag: str) -> bool:
        return a.stop_after == tag

    t0 = time.time()
    rc = 0

    if "1" in wanted and not a.skip_stage1:
        print("\n>>> STAGE 1 (character pretraining; skipped automatically if absent)")
        args = _parse(stage_vae, ["--stage", "1", "--out", str(runs / "stage1_chars"),
                                  "--limit-chars", "50" if a.smoke else "20000",
                                  *common] + (smoke_vae if a.smoke else []))
        rc |= stage_vae.run(args)
        if maybe_stop("stage1"):
            return rc

    if "2" in wanted:
        print("\n>>> STAGE 2 (InkVAE on lines)")
        args = _parse(stage_vae, ["--stage", "2", "--out", str(runs / "stage2_vae"),
                                  *common] + (smoke_vae if a.smoke else []))
        rc |= stage_vae.run(args)
        if maybe_stop("stage2"):
            return rc
        if not vae_ckpt.exists():
            print("stage 2 produced no checkpoint — stopping")
            return 1

    if "3" in wanted:
        print("\n>>> STAGE 3 (boundaries + join labels)")
        args = _parse(stage_boundaries, ["--root", a.root, "--device", a.device,
                                         "--vae", str(vae_ckpt),
                                         "--out", str(runs / "stage3_boundaries"),
                                         *common[2:]] + (smoke_bnd if a.smoke else []))
        rc |= stage_boundaries.run(args)
        if maybe_stop("stage3"):
            return rc

    if "4" in wanted:
        print("\n>>> STAGE 4 (BoundaryDiT)")
        args = _parse(stage_dit, ["--root", a.root, "--device", a.device,
                                  "--vae", str(vae_ckpt),
                                  "--out", str(runs / "stage4_dit"),
                                  *common[2:]] + (smoke_dit if a.smoke else []))
        rc |= stage_dit.run(args)
        if maybe_stop("stage4"):
            return rc

    if "6" in wanted and dit_ckpt.exists():
        print("\n>>> STAGE 6 (evaluation)")
        args = _parse(stage_eval, ["--root", a.root, "--device", a.device,
                                   "--dit", str(dit_ckpt),
                                   "--out", str(runs / "stage6_eval"),
                                   "--n", "3" if a.smoke else "40",
                                   "--steps", "2" if a.smoke else "20",
                                   "--d", "32" if a.smoke else "128",
                                   "--z-dim", "16" if a.smoke else "64",
                                   "--depth", "1" if a.smoke else "3",
                                   "--n-mix", "4" if a.smoke else "20"])
        rc |= stage_eval.run(args)

    print(f"\n[done] {time.time()-t0:.1f}s, exit={rc}")
    print(f"[visuals] {runs}/stage*/visuals/  and  {runs}/stage*/metrics.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
