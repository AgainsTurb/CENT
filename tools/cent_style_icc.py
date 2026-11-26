"""Is cursive junction geometry a WRITER property? (the test that licenses JGT)

For every character junction in the held-out split we extract

    alpha  : exit point, normalised inside character i's box
    beta   : entry point, normalised inside character i+1's box
    theta1 : exit tangent (as cos/sin)
    theta2 : entry tangent (as cos/sin)
    L      : |entry - exit| / character height

then compute the intraclass correlation for each dimension:

    ICC = between-writer variance / (between + within)

ICC near 1  -> junctions are a style signature, few-shot transfer is well posed.
ICC near 0  -> junctions vary per instance, the JGT claim fails.
"""
import json, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/DiffInk")
from utils.utils import load_config_from_yaml
from dataset import build_test_datasets_and_loaders

NBATCH = int(__import__("os").environ.get("NBATCH", 40))
CH = {"pen_up": 3, "eoc": 4}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def main():
    cfg = load_config_from_yaml("./configs/dit_val_config.yaml")
    loader, cfg = build_test_datasets_and_loaders(cfg)
    feats, writers = [], []
    for p, batch in enumerate(loader):
        if p >= NBATCH:
            break
        data, mask, text_idx, cpi, wid = batch
        for b in range(data.size(0)):
            L = int(mask[b].sum().item())
            gt = data[b, :L].numpy()
            e = np.where(gt[:, CH["eoc"]] > 0.5)[0]
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
            w = int(wid[b].item())
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
    F = np.asarray(feats, dtype=np.float64)
    W = np.asarray(writers)
    names = ["alpha_x", "alpha_y", "beta_x", "beta_y", "cos_t1", "sin_t1", "cos_t2", "sin_t2", "L"]
    print(f"[data] {len(F)} junctions from {len(set(W.tolist()))} writers", flush=True)

    uniq = np.unique(W)
    counts = np.array([(W == w).sum() for w in uniq])
    keep = uniq[counts >= 20]
    print(f"[data] writers with >=20 junctions: {len(keep)}", flush=True)

    rows = {}
    for d, nm in enumerate(names):
        x = F[:, d]
        means, vars_, ns = [], [], []
        for w in keep:
            xw = x[W == w]
            means.append(xw.mean()); vars_.append(xw.var(ddof=1) if len(xw) > 1 else 0.0); ns.append(len(xw))
        means = np.array(means); vars_ = np.array(vars_); ns = np.array(ns)
        grand = float((means * ns).sum() / ns.sum())
        between = float(((means - grand) ** 2 * ns).sum() / ns.sum())
        within = float((vars_ * ns).sum() / ns.sum())
        icc = between / max(between + within, 1e-12)
        rows[nm] = {"between_var": round(between, 5), "within_var": round(within, 5),
                    "ICC": round(icc, 3)}
    for nm, r in rows.items():
        print(f"  {nm:9s} between {r['between_var']:.5f}  within {r['within_var']:.5f}  ICC {r['ICC']:.3f}",
              flush=True)
    iccs = [r["ICC"] for r in rows.values()]
    print("=== SUMMARY ===")
    print(json.dumps({"junctions": len(F), "writers": int(len(keep)),
                      "mean_ICC": round(float(np.mean(iccs)), 3),
                      "per_dim": rows}, indent=1))


main()
