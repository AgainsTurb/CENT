"""CENT v3 dataset: layout boxes, boundary joins, relative-offset stroke sequences.

Built on top of `data_strokes.StrokeLineDataset` (same canonical corpus, same reference-line
handling, same normalisation) so that v2 and v3 see identical geometry. What v3 adds:

  * per-character grouping, from the corpus's own `char_strokes` field -- never inferred;
  * per-stroke **relative** payloads (offset from the running pen position), which is what
    makes pen-path continuity a property of the representation;
  * per-character boxes (h, w, y_centre, x-advance) for M3;
  * per-boundary join labels and anchor gaps for M4, in canvas units;
  * per-pair join priors for M4's input, when a prior table is supplied.

Fail-closed: a line whose `char_strokes` disagrees with its stroke count is dropped rather
than silently mis-grouped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_strokes import StrokeLineDataset, stroke_scale
from .models.stroke_graph import fit_stroke

IGNORE = -100


def _load_prior(path: str | None, smoothing: float) -> tuple[dict[int, float], float]:
    """Codepoint-pair join priors.

    Keys are stored as `"<ord(a)>,<ord(b)>"` so the table is independent of the charset
    ordering used by any particular build of the canonical corpus.
    """
    if path and Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        table = {int(k.split(",")[0]) * 0x110000 + int(k.split(",")[1]): float(v)
                 for k, v in raw.items()}
        glob = float(np.mean(list(table.values()))) if table else 0.2
        return table, glob
    return {}, 0.2


class StrokeDatasetV3(Dataset):
    def __init__(self, root: str, split: str, k_refs: int = 5, img_h: int = 64,
                 canvas_w: int = 1024, max_chars: int = 64, max_strokes_char: int = 48,
                 max_strokes_line: int = 320, join_tau: float = 0.05,
                 paired_only: bool = False, limit: int | None = None,
                 n_classes: int = 2680, pair_prior_path: str | None = None,
                 prior_smoothing: float = 1.0):
        self.base = StrokeLineDataset(root, split, k_refs=k_refs, img_h=img_h,
                                      canvas_w=canvas_w, max_strokes=max_strokes_line,
                                      paired_only=paired_only, limit=None)
        self.img_h, self.canvas_w = img_h, canvas_w
        self.max_chars, self.max_sc, self.max_sl = max_chars, max_strokes_char, max_strokes_line
        self.join_tau = join_tau
        self.n_classes = n_classes
        self.prior, self.prior_glob = _load_prior(pair_prior_path, prior_smoothing)
        self.prior_smoothing = prior_smoothing

        self.ids: list[int] = []
        dropped = {"no_char_split": 0, "bad_ranges": 0, "short_text": 0,
                   "too_long": 0, "too_short": 0}
        for rec_i in self.base.usable:
            r = self.base.records[rec_i]
            cs = r.get("char_strokes")
            n = int(r.get("n_strokes") or 0)
            # Two formats exist in the wild:
            #   * the real corpus stores [start, end) STROKE INDEX RANGES, one per inked
            #     character, with gaps where a connector stroke sits (measured: 14 ranges for a
            #     22-character line, last end == n_strokes);
            #   * the synthetic fixture stores per-character stroke COUNTS.
            if not cs:
                dropped["no_char_split"] += 1
                continue
            if isinstance(cs[0], (list, tuple)):
                spans = [(int(a), int(b)) for a, b in cs]
                # measured on the real corpus: monotone, zero-width entries exist (a character
                # with no stroke of its own), and the last end is n-1, not n
                if (any(a < 0 or b > n or a > b for a, b in spans)
                        or any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1))):
                    dropped["bad_ranges"] += 1
                    continue
                n_chars_rec = len(spans)
            else:
                counts = [int(c) for c in cs]
                if sum(counts) != n or any(c <= 0 for c in counts):
                    dropped["no_char_split"] += 1
                    continue
                n_chars_rec = len(counts)
            if len(r.get("text", "")) < n_chars_rec:
                dropped["short_text"] += 1
                continue
            if n < 2:
                dropped["too_short"] += 1
                continue
            if len(cs) > max_chars or n > max_strokes_line:
                dropped["too_long"] += 1
                continue
            self.ids.append(rec_i)
            if limit is not None and len(self.ids) >= limit:
                break
        self.dropped = dropped

    def __len__(self) -> int:
        return len(self.ids)

    # ---- helpers --------------------------------------------------------------------
    def _normalise(self, rec_i: int) -> tuple[list[np.ndarray], list[int]]:
        """Line strokes -> canvas coordinates, normalised to [0,1]x[0,1]."""
        st = self.base.strokes_of(rec_i)
        allp = np.concatenate(st)
        lo, hi = allp.min(0), allp.max(0)
        span = np.maximum(hi - lo, 1e-6)
        # float() both sides: numpy 0-d arrays make builtin min() raise an ambiguity error
        s = float(min(self.canvas_w / float(span[0]), self.img_h / float(span[1])))
        out, keep = [], []
        for j, x in enumerate(st):
            if len(x) <= 1:                      # degenerate stroke: cannot fit a cubic
                continue
            out.append((np.asarray(x, np.float32) - lo) * s
                       / np.array([self.canvas_w, self.img_h], np.float32))
            keep.append(j)
        return out, keep

    def __getitem__(self, i: int) -> dict:
        rec_i = self.ids[i]
        r = self.base.records[rec_i]
        strokes, orig = self._normalise(rec_i)
        pos_of = {o: i for i, o in enumerate(orig)}
        raw = r["char_strokes"]
        # LABEL EVERY STROKE, INCLUDING THE CONNECTORS.
        # char_strokes gives [start,end) ranges for characters only; the strokes in the gaps
        # between ranges are the inter-character connectors. An earlier revision built the
        # sequence from the ranges alone and silently dropped them (measured: 64 of 78 strokes
        # on one line), i.e. it deleted exactly the strokes this paper is about. Now the
        # sequence keeps every stroke in trajectory order and char_strokes only LABELS them:
        # a connector inherits the preceding character's index.
        n_all_kept = len(strokes)
        if isinstance(raw[0], (list, tuple)):
            lab = [-1] * n_all_kept
            for k in range(len(raw)):
                a, b = int(raw[k][0]), int(raw[k][1])
                for j in range(a, b):
                    if j in pos_of:
                        lab[pos_of[j]] = k
            last = 0
            for j in range(n_all_kept):          # forward-fill connectors
                if lab[j] < 0:
                    lab[j] = last
                else:
                    last = lab[j]
        else:
            lab = []
            for k, c in enumerate(raw):
                lab += [k] * int(c)
            lab = lab[:n_all_kept]
        n_lab = (max(lab) + 1) if lab else 0
        idxg = [[j for j in range(n_all_kept) if lab[j] == k] for k in range(n_lab)]
        # Drop characters with no strokes, then truncate the text to the surviving count.
        # NOTE: the gap against index.jsonl (7800 file chars vs 4385 over 300 lines) is NOT a
        # bug here: char_strokes annotates only ~56% of each line, and text+strokes are
        # consistently truncated to that annotated prefix. The remaining characters of a line
        # have no stroke labels at all, so they cannot be trained on.
        keep = [k for k, g in enumerate(idxg) if g]
        idxg = [idxg[k] for k in keep]

        groups = [[strokes[j] for j in g] for g in idxg]
        counts = [len(g) for g in groups]
        # characters with no strokes are kept for ALIGNMENT; their anchors fall back to the
        # nearest non-empty neighbour so boundary lookups cannot go out of range
        def _frst(k):
            for g in idxg[k:]:
                if g: return g[0]
            for g in reversed(idxg[:k]):
                if g: return g[-1]
            return 0
        def _last(k):
            for g in reversed(idxg[:k + 1]):
                if g: return g[-1]
            for g in idxg[k + 1:]:
                if g: return g[0]
            return 0

        text = r["text"][: self.max_chars]
        ids = [int(self.base.char2idx.get(ch, self.base.pad_id)) for ch in text]
        n_chars = min(len(ids), len(counts))
        ids, counts = ids[:n_chars], counts[:n_chars]

        # ---- per stroke: relative payload + pen state --------------------------------
        cont, pen = [], []
        pen_pos = np.zeros(2, np.float32)
        char_of_stroke = []
        boxes = []
        for ci, grp in enumerate(groups[:n_chars]):
            if not grp:
                continue
            for k, s in enumerate(grp):
                p = fit_stroke(s)                            # [4,2] absolute, canvas units
                d = p[0] - pen_pos
                rel = np.concatenate([d, (p[1:] - p[0]).reshape(-1)]).astype(np.float32)
                cont.append(rel)
                if k == 0 and ci > 0:
                    pen.append(1 if float(np.linalg.norm(d)) < self.join_tau else 0)
                else:
                    pen.append(1 if float(np.linalg.norm(d)) < 1e-3 else 0)
                pen_pos = p[3]
                char_of_stroke.append(ci)
            allp = np.concatenate(grp)
            lo, hi = allp.min(0), allp.max(0)
            boxes.append([float(hi[1] - lo[1]), float(hi[0] - lo[0]),
                          float((hi[1] + lo[1]) / 2), float(lo[0])])

        L = min(len(cont), self.max_sl)
        cont_t = torch.zeros(self.max_sl, 8)
        cont_t[:L] = torch.from_numpy(np.stack(cont[:L]).astype(np.float32))
        pen_t = torch.zeros(self.max_sl, dtype=torch.long)
        pen_t[:L] = torch.from_numpy(np.asarray(pen[:L], np.int64))
        ch_t = torch.zeros(self.max_sl, dtype=torch.long)
        ch_t[:L] = torch.from_numpy(np.asarray(char_of_stroke[:L], np.int64))
        mask = torch.zeros(self.max_sl)
        mask[:L] = 1.0

        # stop flag: last stroke of each character
        stop = torch.zeros(self.max_sl)
        last_of_char = {}
        for pos, ci in enumerate(char_of_stroke[:L]):
            last_of_char[ci] = pos
        for ci, pos in last_of_char.items():
            stop[pos] = 1.0

        # ---- boxes ------------------------------------------------------------------
        nb = min(len(boxes), self.max_chars)
        box_t = torch.zeros(self.max_chars, 4)
        box_t[:nb] = torch.from_numpy(np.asarray(boxes[:nb], np.float32))
        for k in range(1, nb):                                # advance instead of position
            box_t[k, 3] = box_t[k, 3] - box_t[k - 1, 3]

        # ---- boundary joins and anchor vectors ---------------------------------------
        # fixed width: one slot per possible boundary, so batches stack against the transcript
        J = max(nb - 1, 0)
        Jfix = max(self.max_chars - 1, 1)
        join_t = torch.zeros(Jfix, dtype=torch.float32)
        gap_t = torch.zeros(Jfix, 2)
        prior_t = torch.zeros(Jfix, dtype=torch.float32)
        for k in range(J):
            e = fit_stroke(strokes[_last(k)])[3]
            s2 = fit_stroke(strokes[_frst(k + 1)])[0]
            v = s2 - e
            gap_t[k] = torch.from_numpy(v.astype(np.float32))
            join_t[k] = 1.0 if float(np.linalg.norm(v)) < self.join_tau else 0.0
            key = ord(text[k]) * 0x110000 + ord(text[k + 1])
            prior_t[k] = float(self.prior.get(key, self.prior_glob))

        # per-character stroke count: the target for M3b (count head), which replaces the
        # degenerate per-stroke `stop` classifier (BCE plateaued at the 15% class prior and
        # fired on the first stroke of every character -> 13 strokes for a 64-stroke line).
        cnt_t = torch.zeros(self.max_chars, dtype=torch.long)
        cnt_t[:min(len(counts), self.max_chars)] = torch.as_tensor(
            counts[:self.max_chars], dtype=torch.long)
        ids_t = torch.full((self.max_chars,), int(self.base.pad_id), dtype=torch.long)
        ids_t[:n_chars] = torch.from_numpy(np.asarray(ids[:n_chars], np.int64))
        tmask = torch.zeros(self.max_chars)
        tmask[:n_chars] = 1.0

        refs, rmask = self.base._refs(int(r["writer"]), rec_i)

        return {"cont": cont_t, "pen": pen_t, "char_idx": ch_t, "mask": mask,
                "stop": stop, "counts": cnt_t, "boxes": box_t, "box_mask": tmask.clone(),
                "join": join_t, "gap": gap_t, "prior": prior_t,
                "text": ids_t, "text_mask": tmask, "n_chars": n_chars,
                "n_strokes": L, "writer": int(r["writer"]),
                "refs": refs, "ref_mask": rmask}


def collate_v3(batch: list[dict]) -> dict:
    out = {}
    for k in ("cont", "pen", "char_idx", "mask", "stop", "counts", "boxes", "box_mask",
              "join", "gap", "prior", "text", "text_mask", "refs", "ref_mask"):
        # refs/ref_mask come back from the base dataset as numpy; accept either type
        out[k] = torch.stack([torch.as_tensor(b[k]) for b in batch])
    out["n_chars"] = torch.tensor([b["n_chars"] for b in batch], dtype=torch.long)
    out["n_strokes"] = torch.tensor([b["n_strokes"] for b in batch], dtype=torch.long)
    out["writer"] = torch.tensor([b["writer"] for b in batch], dtype=torch.long)
    return out
