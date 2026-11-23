"""CENT v3 decoder, written once with unambiguous anchoring.

Token semantics (fixed by the training data):
    cont[:, 0:2] = start_k - end_{k-1}      (offset from the PREVIOUS STROKE'S END)
    cont[:, 2:8] = (P1,P2,P3) relative to start_k

Therefore reconstruction is a recurrence, never a cumsum:
    start_k = end_{k-1} + cont_k[0:2]
    end_k   = start_k + cont_k[6:8]

At generation there are two ways to place a stroke:
    free : start_k = abs_head_k                (absolute, bounded, no drift)
    tf   : start_k = true_end_{k-1} + pred[0:2]  (the model's own one-step prediction)
In BOTH cases the emitted token is the same quantity the model was trained on
(start - previous end), so the token stream stays in distribution.
"""
from __future__ import annotations

import torch


def draw_payload(pay, w=1600, h=240, thick=1, out_tone=0.55):
    """Draw every stroke and report how many left the canvas.

    Silent clipping is forbidden: a stroke whose geometry lies outside [0,1] is drawn in
    `out_tone` instead of ink black, and counted. A picture that hides out-of-frame
    geometry is worse than no picture (a generation with error 27.8 once rendered as a
    clean-looking line because everything was clipped onto the border).
    """
    import numpy as np
    a = torch.as_tensor(pay).detach().cpu().numpy()
    while a.ndim > 3:      # [1,N,4,2] -> [N,4,2]; never collapse below that
        a = a[0]
    a = a.reshape(-1, 4, 2)
    img = np.ones((h, w), np.float32)
    ts = np.linspace(0, 1, 16)[:, None]
    drawn = out_of_frame = 0
    for st in a:
        if not np.isfinite(st).all():
            continue
        p0, c1, c2, p3 = st[0], st[1], st[2], st[3]
        pts = ((1 - ts) ** 3 * p0 + 3 * (1 - ts) ** 2 * ts * c1
               + 3 * (1 - ts) * ts ** 2 * c2 + ts ** 3 * p3)
        off = bool(((pts < -0.02) | (pts > 1.02)).any())
        tone = out_tone if off else 0.0
        if off:
            out_of_frame += 1
        px = np.clip(pts, 0, 1) * np.array([w - 1, h - 1], np.float32)
        for A, B in zip(px[:-1], px[1:]):
            n = int(np.ceil(float(np.hypot(*(B - A))))) + 1
            for u in np.linspace(0, 1, n):
                x, y = (A + (B - A) * u).astype(int)
                img[max(y - thick, 0):y + thick + 1, max(x - thick, 0):x + thick + 1] = tone
        drawn += 1
    assert drawn == a.shape[0], f"drew {drawn} of {a.shape[0]} strokes"
    return img, drawn, out_of_frame


def reconstruct(cont: torch.Tensor) -> torch.Tensor:
    """[B,N,8] relative tokens -> [B,N,4,2] absolute Bezier payloads."""
    B, N = cont.shape[:2]
    pos = cont.new_zeros(B, 2)
    starts = []
    for k in range(N):
        st = pos + cont[:, k, 0:2]
        starts.append(st)
        pos = st + cont[:, k, 6:8]
    start = torch.stack(starts, 1)
    ctrl = cont[..., 2:8].view(B, N, 3, 2) + start.unsqueeze(-2)
    return torch.cat([start.unsqueeze(-2), ctrl], dim=-2)


def mixture_mean(out: dict, t: int) -> torch.Tensor:
    """Deterministic 8-vector: the DOMINANT COMPONENT's mean.

    The plain mixture mean is degenerate for stroke geometry: a bimodal distribution over
    direction (up-stroke / down-stroke) averages to a near-zero vertical extent, which renders
    as the flat dashes we kept seeing. Taking the highest-weight component preserves the stroke
    while staying deterministic.
    """
    k = out["pi"][:, t].argmax(-1)
    D = out["mu"].shape[-1]
    idx = k.view(-1, 1, 1).expand(-1, 1, D)
    return out["mu"][:, t].gather(1, idx).squeeze(1)


def mixture_sample(out: dict, t: int, temperature: float, generator=None) -> torch.Tensor:
    pi = out["pi"][:, t] / max(temperature, 1e-3)
    comp = torch.multinomial(pi.exp().clamp_min(1e-9), 1, generator=generator).squeeze(-1)
    idx = comp.view(pi.shape[0], 1, 1).expand(-1, 1, out["mu"].shape[-1])
    m = out["mu"][:, t].gather(1, idx).squeeze(1)
    s = torch.exp(out["ls"][:, t].gather(1, idx).squeeze(1)) * max(temperature, 1e-3) ** 0.5
    v = m + s * torch.randn(m.shape, dtype=m.dtype, device=m.device, generator=generator)
    v = v.clamp(-0.6, 1.6)
    v[..., 2:8] = v[..., 2:8].clamp(-0.6, 0.6)
    return v


@torch.no_grad()
def decode(model, cfg, text, tmask, refs, refmask, n_chars,
           mode: str = "free", true: dict | None = None, temperature: float = 0.0,
           generator=None, count_mode: str = "exact", clamp: str | None = None,
           anchor_place: bool = False) -> dict:
    """mode='free' uses the absolute head; mode='tf' follows the true history."""
    device = text.device
    H, zg, z_l, z_k = model.encode(text, tmask, refs, refmask)
    anchors = model.plan_layout(H, zg, tmask)      # [B,C,5], supervised
    _sm, _ss = getattr(cfg, "stroke_mean", None), getattr(cfg, "stroke_std", None)
    _mu_t = _sd_t = None
    if _sm is not None:
        _mu_t = torch.tensor(_sm, device=device)
        _sd_t = torch.tensor(_ss, device=device)
    counts = model.plan_counts(H, zg, tmask).argmax(-1) + 1          # [B,C]
    # normalise anchors by their own span over the valid characters (see anchor_place)
    _ax = anchors[:, :n_chars, 0]; _ay = anchors[:, :n_chars, 1]
    _axn = 0.02 + 0.93 * ((_ax - _ax.min(1, True).values)
                          / (_ax.max(1, True).values - _ax.min(1, True).values).clamp_min(1e-6))
    _ayn = 0.15 + 0.55 * ((_ay - _ay.min(1, True).values)
                          / (_ay.max(1, True).values - _ay.min(1, True).values).clamp_min(1e-6))
    B = text.shape[0]
    if mode == "tf":
        assert true is not None
        tc, tp, tci = true["cont"], true["pen"], true["char"]
        max_steps = int(true["n_strokes"].max())

    toks = [[] for _ in range(B)]
    taken = [0] * B                      # strokes owned by each sample
    pay = [[] for _ in range(B)]
    prev_end = [torch.zeros(2, device=device) for _ in range(B)]
    errs = []
    step = 0
    for ci in range(n_chars):
        if step >= cfg.max_strokes_line:
            break
        n_this = int(counts[:, ci].max().clamp(1, cfg.max_strokes_char))
        for _ in range(n_this):
            if step >= cfg.max_strokes_line:
                break
            # DRIFT FIX: at the FIRST stroke of each character, reset the pen to that
            # character's supervised anchor. Inside a character the recurrence is kept
            # (that is the connectivity); across characters the error can no longer
            # accumulate, so the pen cannot walk off the page. Anchors are canvas units,
            # strokes are normalised, hence the division.
            if anchor_place and _ == 0 and mode != "tf":
                # SCALE-INVARIANT: the corpus dx unit is ~9x smaller than the canvas
                # (absolute division gave x span 0.11 and stacked every glyph). Anchors are
                # mapped by their OWN span onto the page, so relative spacing is kept and
                # no unit is assumed.
                for i in range(B):
                    prev_end[i] = torch.stack([_axn[i, ci], _ayn[i, ci]]).to(prev_end[i].dtype)
            if mode == "tf":
                if step >= max_steps:
                    break
                pc = tc[:, :step]; pp = tp[:, :step]; pci = tci[:, :step]
            else:
                # toks[i] is a list of (token, pen, char) tuples; stack per sample then batch
                pc = (torch.stack([torch.stack([e[0] for e in t]) for t in toks], 0)
                      if step else torch.zeros(B, 0, 8, device=device))
                pp = (torch.stack([torch.stack([e[1] for e in t]) for t in toks], 0)
                      if step else torch.zeros(B, 0, dtype=torch.long, device=device))
                pci = (torch.stack([torch.stack([e[2] for e in t]) for t in toks], 0)
                       if step else torch.zeros(B, 0, dtype=torch.long, device=device))
            full_c = torch.cat([torch.zeros(B, 1, 8, device=device), pc], 1)
            full_p = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), pp], 1)
            full_ci = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), pci], 1)
            out = model.decoder(full_c, full_p, full_ci,
                                torch.ones(B, full_c.shape[1], device=device),
                                H, tmask, sigma=cfg.gauss_sigma)
            t_last = full_c.shape[1] - 1
            pred = (mixture_mean(out, t_last) if temperature <= 0
                    else mixture_sample(out, t_last, temperature, generator))
            if _mu_t is not None:
                pred = pred * _sd_t + _mu_t      # back to canvas units

            # Teacher-forced works with the relative path (err 0.159); the absolute head is
            # the weak link. Use the same relative composition for free-running, and keep the
            # absolute head only as an optional anchor.
            if mode == "abs":
                start = [out["abs"][i, t_last] for i in range(B)]
            else:
                start = [e + pred[i, 0:2] for i, e in enumerate(prev_end)]

            for i in range(B):
                if mode != "tf":
                    # DRIFT FIX. Free-running placement was chained from the previous stroke's
                    # end, so one bad stroke shifted everything after it: measured 51 of 108
                    # strokes off-canvas. The recurrence is kept INSIDE a character (that is
                    # the connectivity), but a stroke may not wander more than one character
                    # width/height from its own anchor, so drift cannot accumulate.
                    _ax, _ay, _aw, _ah = (anchors[i, ci, 0], anchors[i, ci, 1],
                                          anchors[i, ci, 2], anchors[i, ci, 3])
                    # Bound to the PAGE, not to the anchor. Anchors are canvas units
                    # (centre + size) while strokes are normalised [0,1]; clamping one by the
                    # other collapsed glyphs into clusters (measured: legible line before the
                    # anchor clamp, fragments after). Anchors condition the decoder through the
                    # memory; they must not be used as a placement rectangle.
                    start[i] = torch.stack([start[i][0].clamp(0.0, 1.0),
                                            start[i][1].clamp(0.0, 1.0)])
                # per-sample count. The loop length is the BATCH maximum, so without this
                # guard every sample in the batch inherits the longest line's stroke count
                # (measured: 108 strokes generated for a 78-stroke line while the predicted
                # counts summed to 78, i.e. the extra 30 strokes were a decoding artefact and
                # they accounted for most of the 51 out-of-frame strokes as well).
                # every sample shares one token stream (required for stacking), so ownership
                # is tracked instead of skipped: a stroke belongs to sample i only while it has
                # not used its predicted character count. Without this a 78-stroke line got
                # 108 strokes by inheriting the longer line's counts.
                if _ < int(counts[i, ci]):
                    taken[i] += 1
                c_new = pred[i].clone()
                c_new[0:2] = start[i] - prev_end[i]      # emit the trained quantity
                toks[i].append((c_new, torch.zeros((), dtype=torch.long, device=device),
                                torch.tensor(ci, device=device)))
                end_i = start[i] + c_new[6:8]
                p0 = start[i]
                ctrl = c_new[2:8].view(3, 2) + p0
                pay[i].append(torch.cat([p0.unsqueeze(0), ctrl], 0))
                prev_end[i] = end_i
            if mode == "tf" and step < true["start"].shape[1]:
                gtstart = true["start"][:, step]
                errs.append(float((torch.stack(start) - gtstart).abs().mean()))
            step += 1

    arrs = []
    for i in range(B):
        if not pay[i]:
            arrs.append(torch.zeros(0, 4, 2, device=device)); continue
        a_i = torch.stack([s.reshape(-1) for s in pay[i]], 0).view(-1, 4, 2)
        lim = taken[i] if count_mode == "exact" else a_i.shape[0]
        arrs.append(a_i[:max(0, min(lim, a_i.shape[0]))])
    mx = max([a.shape[0] for a in arrs] + [0])
    payload = (torch.stack([torch.cat([a, a.new_zeros(mx - a.shape[0], 4, 2)]) for a in arrs], 0)
               if mx else torch.zeros(B, 0, 4, 2, device=device))
    return {"payload": payload, "counts": counts, "steps": step,
            "errs": errs, "n_strokes": taken}
