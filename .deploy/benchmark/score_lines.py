"""Score a line-generation run against the canonical line ground truth.

None of the four baselines ships a metric: DiffBrush, TGC-Diff, DiffInk and OLHWG all
emit images or trajectories and stop. So the benchmark's scoring has to live here, be
identical for every system, and be honest about what it cannot measure.

Two input modes:

  --pred-text pred.jsonl     one {"id","text"} per generated line. AR/CR are computed
                             directly. This is the only mode that yields content numbers
                             today.
  --pred-images DIR          generated line images named <id>.png. AR/CR need a
                             recogniser; without --htr the result is NOT_RUN, and with
                             --htr this revision still refuses (no frozen HTR is bundled).

Metrics
  AR = (N - D - S - I) / N,  CR = (N - D - S) / N   over the whole test set in aggregate,
  computed from a DP edit distance on the transcript strings (the convention the Chinese
  line literature uses). FID is NOT_RUN below 512 samples or without Inception.

Usage
  python -m benchmark.score_lines --gt data/canonical_lines/v1 --split test \
      --pred-text runs/tgcdiff/pred.jsonl --out runs/tgcdiff/score.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def edit_ops(a: list[str], b: list[str]) -> tuple[int, int, int, int]:
    """Return (N, deletions, substitutions, insertions) between hypothesis a and ref b.

    N is the reference length. A standard Levenshtein DP with operation counts, so AR
    and CR can be derived without a separate alignment library.
    """
    n, m = len(a), len(b)
    # dp[i][j] = (cost, D, S, I)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (j, j, 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            best = None
            c, d, s, ins = dp[i - 1][j - 1]
            cand = (c + 1, d, s + 1, ins)          # substitution
            best = cand
            c, d, s, ins = dp[i - 1][j]
            cand = (c + 1, d + 1, s, ins)          # deletion (hyp extra char)
            if cand[0] < best[0]:
                best = cand
            c, d, s, ins = dp[i][j - 1]
            cand = (c + 1, d, s, ins + 1)          # insertion (missing char)
            if cand[0] < best[0]:
                best = cand
            dp[i][j] = best
    return (m, dp[n][m][1], dp[n][m][2], dp[n][m][3])


def _not_run(reason: str) -> dict:
    return {"status": "NOT_RUN", "reason": reason}


def load_gt(root: str | Path, split: str) -> dict[str, str]:
    d = Path(root) / split
    gt = {}
    for ln in (d / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            gt[r["id"]] = r["text"]
    return gt


def score_text(pred: dict[str, str], gt: dict[str, str]) -> dict:
    ids = [i for i in gt if i in pred]
    missing = [i for i in gt if i not in pred]
    N = D = S = I = 0
    per_line = []
    for i in ids:
        n, d, s, ins = edit_ops(list(pred[i]), list(gt[i]))
        N += n; D += d; S += s; I += ins
        per_line.append({"id": i, "ref": gt[i], "hyp": pred[i],
                         "N": n, "D": d, "S": s, "I": ins})
    if N == 0:
        return _not_run("no overlapping line ids between predictions and ground truth")
    ar = max(0.0, (N - D - S - I) / N)
    cr = max(0.0, (N - D - S) / N)
    return {"status": "OK", "lines_scored": len(ids), "lines_missing": len(missing),
            "N": N, "D": D, "S": S, "I": I,
            "AR": round(ar, 4), "CR": round(cr, 4), "per_line": per_line}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, help="canonical line root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-text", default=None)
    ap.add_argument("--pred-images", default=None)
    ap.add_argument("--htr", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt = load_gt(a.gt, a.split)
    rep: dict = {"split": a.split, "gt_lines": len(gt)}

    if a.pred_text:
        pred = {}
        for ln in Path(a.pred_text).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                pred[str(r["id"])] = r["text"]
        rep["content"] = score_text(pred, gt)
    elif a.pred_images:
        imgs = list(Path(a.pred_images).glob("*.png"))
        rep["content"] = _not_run(
            f"{len(imgs)} images found, but AR/CR need a recogniser. No frozen HTR is bundled "
            "in this revision, and substituting one would change what is being measured. "
            "Provide predicted transcripts via --pred-text, or wire a fixed HTR and record it "
            "in the run ledger."
        )
    else:
        rep["content"] = _not_run("no predictions supplied")

    rep["FID"] = _not_run("FID needs Inception features and >=512 lines; not enabled here")
    rep["connectivity"] = _not_run(
        "connectivity is scored by the system that produced trajectories (cent.eval_lines "
        "for CENT; a shim per online baseline). Image-only baselines cannot report it."
    )
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
