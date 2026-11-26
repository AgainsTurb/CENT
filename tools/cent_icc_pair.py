"""Pair-conditioned ICC: is junction geometry a WRITER property once the character pair is fixed?

The unconditioned test gave mean ICC 0.08, but junction geometry is dominated by WHICH two
characters meet. Here every character pair (c_i, c_{i+1}) is its own group, and the ICC is
computed inside a group across writers, then pooled. Pairs must be seen by >= 5 writers.

    ICC ~ 1  -> writer style is recoverable once the pair is controlled -> JGT is viable
    ICC ~ 0  -> junctions are a function of the pair only
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/DiffInk")
from utils.utils import load_config_from_yaml
from dataset import build_test_datasets_and_loaders

NBATCH = int(os.environ.get("NBATCH", 60))
MIN_WRITERS = int(os.environ.get("MIN_WRITERS", 5))
MIN_JUNC = int(os.environ.get("MIN_JUNC", 15))
NAMES = ["alpha_x", "alpha_y", "beta_x", "beta_y", "cos_t1", "sin_t1", "cos_t2", "sin_t2", "L"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def parse_writer(w):
    if hasattr(w, "item"):
        w = w.item()
    s = str(w)
    t = s.replace("writer_", "")
    return int(t) if t.isdigit() else hash(s) % 100000


def main():
    cfg = load_config_from_yaml("./configs/dit_val_config.yaml")
    loader, cfg = build_test_datasets_and_loaders(cfg)
    feats, writers, pairs = [], [], []
    for p, batch in enumerate(loader):
        if p >= NBATCH:
            break
        data, mask, text_idx, cpi, wid = batch
        for b in range(data.size(0)):
            L = int(mask[b].sum().item())
            gt = data[b, :L].numpy()
            e = np.where(gt[:, 4] > 0.5)[0]
            if len(e) < 3:
                continue
            starts = np.r_[0, e[:-1] + 1]
            n = min(len(e), len(starts))
            boxes = []
            for j in range(n):
                s0, s1 = int(starts[j]), min(int(e[j]) + 1, L)
                seg = gt[s0:s1, :2]
                if len(seg) < 2:
                    boxes.append(None)
                    continue
                lo, hi = seg.min(0), seg.max(0)
                boxes.append((lo, np.maximum(hi - lo, 1e-3)))
            w = parse_writer(wid[b])
            tx = text_idx[b].cpu().numpy()
            for j in range(n - 1):
                if boxes[j] is None or boxes[j + 1] is None:
                    continue
                k, nxt = int(e[j]), int(starts[j + 1])
                if k + 1 >= L or nxt + 1 >= L:
                    continue
                p_out, p_in = gt[k, :2], gt[nxt, :2]
                lo1, wh1 = boxes[j]
                lo2, wh2 = boxes[j + 1]
                a = (p_out - lo1) / wh1
                bta = (p_in - lo2) / wh2
                t1 = unit(gt[k, :2] - gt[max(k - 2, 0), :2])
                t2 = unit(gt[min(nxt + 2, L - 1), :2] - gt[nxt, :2])
                Ln = float(np.linalg.norm(p_in - p_out) / max(wh1[1], 1e-6))
                if Ln > 5.0:
                    continue
                feats.append([a[0], a[1], bta[0], bta[1], t1[0], t1[1], t2[0], t2[1], Ln])
                writers.append(w)
                pairs.append((int(tx[j]), int(tx[j + 1])))
    F = np.asarray(feats, float)
    W = np.asarray(writers)
    P = np.asarray(pairs)
    print(f"[data] {len(F)} junctions, {len(set(W.tolist()))} writers, "
          f"{len(set(map(tuple, P.tolist())))} distinct pairs", flush=True)

    def icc_block(idx):
        """between/within writer variance for the junctions selected by idx."""
        x_w, x_p = W[idx], P[idx]
        rows = []
        for d in range(F.shape[1]):
            x = F[idx, d]
            means, vars_, ns = [], [], []
            for w in np.unique(x_w):
                xw = x[x_w == w]
                if len(xw) < 2:
                    continue
                means.append(xw.mean()); vars_.append(xw.var(ddof=1)); ns.append(len(xw))
            if len(means) < 2:
                rows.append(np.nan); continue
            means, vars_, ns = map(np.asarray, (means, vars_, ns))
            grand = float((means * ns).sum() / ns.sum())
            between = float(((means - grand) ** 2 * ns).sum() / ns.sum())
            within = float((vars_ * ns).sum() / ns.sum())
            rows.append(between / max(between + within, 1e-12))
        return np.asarray(rows)

    # unconditioned reference
    unc = icc_block(np.arange(len(F)))
    print("[unconditioned] mean ICC %.3f" % np.nanmean(unc), flush=True)

    # pair-conditioned: average ICC inside each pair group, weighted by group size
    uniq_pairs = {}
    for pid in set(map(tuple, P.tolist())):
        sel = np.where((P[:, 0] == pid[0]) & (P[:, 1] == pid[1]))[0]
        if len(sel) < MIN_JUNC:
            continue
        if len(np.unique(W[sel])) < MIN_WRITERS:
            continue
        uniq_pairs[pid] = sel
    if not uniq_pairs:
        print("[pair] no pair met the thresholds; lower MIN_WRITERS/MIN_JUNC", flush=True)
        return
    acc = np.zeros(F.shape[1]); wsum = 0.0
    for pid, sel in uniq_pairs.items():
        r = icc_block(sel)
        acc += np.nan_to_num(r) * len(sel); wsum += len(sel)
    cond = acc / max(wsum, 1e-9)
    print(f"[pair] {len(uniq_pairs)} qualifying pairs covering {int(sum(len(s) for s in uniq_pairs.values()))} "
          f"junctions", flush=True)
    for nm, u, c in zip(NAMES, unc, cond):
        print(f"  {nm:9s} uncond {u:.3f}   pair-cond {c:.3f}", flush=True)
    print("=== SUMMARY ===")
    print(json.dumps({"junctions": len(F), "qualifying_pairs": len(uniq_pairs),
                      "mean_ICC_uncond": round(float(np.nanmean(unc)), 3),
                      "mean_ICC_pair_cond": round(float(np.mean(cond)), 3),
                      "per_dim": {nm: [round(float(u), 3), round(float(c), 3)]
                                  for nm, u, c in zip(NAMES, unc, cond)}}, indent=1))


main()
