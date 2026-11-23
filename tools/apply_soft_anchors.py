"""Apply the soft-anchor (M3') upgrade.

Design rule, stated once: an anchor is a CONDITIONING SIGNAL, never a constraint.
 - strokes are never clipped to an anchor rect;
 - the boundary contract (start(next) = exit_i + gap_i*(1-zeta_i)) is unchanged;
 - anchors reach the decoder through the transcript memory, so each stroke sees
   a_{i-1}, a_i, a_{i+1} (neighbourhood) and can still exceed all three.
"""
import ast, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def patch(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in s, f"{path}: anchor not found: {old[:70]!r}"
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    ast.parse(s)
    print(f"patched {path}")


# ---------------------------------------------------------------- model -----
patch("cent/models/v3.py", [
    (
        "        self.layout_head = nn.Linear(dim, 4)          # h, w, y_center, dx advance",
        "        # M3' soft anchors: (x_center, y_center, w, h, dx_advance). Five numbers, no clip\n"
        "        # region: strokes may leave every anchor, which is what keeps cursive overshoot\n"
        "        # and inter-character connection expressible.\n"
        "        self.layout_head = nn.Linear(dim, 5)\n"
        "        self.anchor_ctx = nn.Linear(5 * 3, dim)       # a_{i-1}, a_i, a_{i+1}",
    ),
    (
        "    def plan_layout(self, H: torch.Tensor, zg: torch.Tensor, text_mask: torch.Tensor):",
        "    def memory_and_mask(self, H, tmask, zg, stage):\n"
        "        \"\"\"Condition the transcript memory on anchor neighbourhoods, then hand the\n"
        "        decoder (memory, mask). Anchors only ADD information here; nothing is clipped.\n"
        "        \"\"\"\n"
        "        if stage in (\"s2\", \"s3\"):\n"
        "            a = self.plan_layout(H, zg, tmask)\n"
        "            nb = torch.cat([torch.nn.functional.pad(a[:, :-1], (0, 0, 1, 0)),\n"
        "                            a,\n"
        "                            torch.nn.functional.pad(a[:, 1:], (0, 0, 0, 1))], dim=-1)\n"
        "            H = H + self.anchor_ctx(nb)\n"
        "        return H, tmask\n"
        "\n"
        "    def plan_layout(self, H: torch.Tensor, zg: torch.Tensor, text_mask: torch.Tensor):",
    ),
])

# --------------------------------------------------------------- trainer ----
patch("cent/train_v3.py", [
    (
        "model.decoder(inp_c, inp_p, inp_ci, inp_m, H, tmask, sigma=cfg.gauss_sigma)",
        "model.decoder(inp_c, inp_p, inp_ci, inp_m,\n"
        "                                *model.memory_and_mask(H, tmask, zg, stage),\n"
        "                                sigma=cfg.gauss_sigma)",
    ),
    (
        "        pred = model.plan_layout(H, zg, tmask)",
        "        # ---- M3' soft anchors ---------------------------------------------------\n"
        "        pred = model.plan_layout(H, zg, tmask)                 # [B,C,5]\n"
        "        bx = torch.cumsum(box_t[..., 3], dim=1)                # x from dx advances\n"
        "        anchor_t = torch.stack([bx, box_t[..., 2], box_t[..., 1],\n"
        "                                box_t[..., 0], box_t[..., 3]], dim=-1)\n"
        "        l_cont = pred.new_zeros(())",
    ),
])

# the box loss line stays; add continuity right after it
p = ROOT / "cent/train_v3.py"
s = p.read_text(encoding="utf-8")
old = '        logs["layout"] = float(l_box.detach())'
new = ('        logs["layout"] = float(l_box.detach())\n'
       '        # continuity: handwriting advances smoothly, so penalise jerk in the anchors\n'
       '        # rather than only error against the corpus boxes\n'
       '        if pred.shape[1] > 2:\n'
       '            adv = pred[..., 0].diff(dim=1)\n'
       '            jerk = (adv[:, 1:] - adv[:, :-1]).abs()\n'
       '            mc = bmask[:, 1:] if bmask.shape[1] == jerk.shape[1] else bmask[:, 2:]\n'
       '            l_cont = (jerk * mc).sum() / mc.sum().clamp_min(1.0)\n'
       '            ys = pred[..., 1]\n'
       '            smooth = (ys[:, 2:] - 2 * ys[:, 1:-1] + ys[:, :-2]).abs()\n'
       '            ms = mc[:, :smooth.shape[1]]\n'
       '            l_cont = l_cont + (smooth * ms).sum() / ms.sum().clamp_min(1.0)\n'
       '        loss = loss + float(getattr(cfg, "lambda_cont", 0.5)) * l_cont\n'
       '        logs["cont"] = float(l_cont.detach())')
assert old in s
p.write_text(s.replace(old, new, 1), encoding="utf-8")
ast.parse(p.read_text(encoding="utf-8"))
print("patched cent/train_v3.py (continuity term)")

# ---------------------------------------------------------------- config ----
patch("cent/config_v3.py", [
    ("    lambda_layout: float = 1.0",
     "    lambda_layout: float = 1.0\n"
     "    # soft-anchor continuity: penalises anchor jerk, not glyph shape\n"
     "    lambda_cont: float = 0.5"),
])
print("done")

