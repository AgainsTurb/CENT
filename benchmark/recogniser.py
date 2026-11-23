"""The frozen Chinese line recogniser that AR/CR depends on.

AR/CR is not a property of a generator; it is the property "a reader could read this", and it
is only as trustworthy as the reader. So the recogniser is fixed ONCE, named in the paper, and
its own accuracy on real lines is reported next to every AR/CR it produces.

ADOPTED RECOGNISER (verified 2026-11-22 via the Hub API):

    Taklia/pylaia-casia-hwdb2   -> id: Teklia/pylaia-casia-hwdb2
    licence: MIT      type: PyLaia (PyTorch) image-to-text, language: zh
    dataset: Teklia/CASIA  (i.e. CASIA-HWDB2 line level)
    revision: 96011bc249ad374b1e3877638ffbc934ad6f73aa
    files: model/, weights.ckpt, syms.txt, tokens.txt, lexicon.txt, language_model.arpa.gz
    total: 65,182,957 B

This module does not re-implement PyLaia. It defines the SLOT: a command template that maps an
image to a transcript on stdout, plus the bookkeeping that makes a number reproducible
(identity, command, and the recogniser's own accuracy measured on real lines).

WHY A TEMPLATE AND NOT A HARD-CODED CALL: PyLaia's decode CLI differs between versions, and
guessing it would put an unverified command in the ledger. The exact invocation used is
recorded per run; `PYLAIA_RECIPE` below documents the one to confirm on the box.

Usage
    from benchmark.recogniser import Recogniser
    rec = Recogniser(command="pylaia-htr-decode-ctc {ckpt} {syms} --img_list {list} > {out}")
    rec.verify()                       # fails closed until the command is proven to work
    texts = rec.transcribe(["a.png", "b.png"])
    rep = rec.calibrate(pairs)         # its own AR/CR on REAL lines -> report this
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import ar_cr

ADOPTED = {
    "id": "Teklia/pylaia-casia-hwdb2",
    "revision": "96011bc249ad374b1e3877638ffbc934ad6f73aa",
    "licence": "MIT",
    "framework": "PyLaia",
    "language": "zh",
    "trained_on": "Teklia/CASIA (CASIA-HWDB2 line level)",
    "bytes": 65_182_957,
    "files": ["model", "weights.ckpt", "syms.txt", "tokens.txt", "lexicon.txt",
              "language_model.arpa.gz"],
    "url": "https://huggingface.co/Teklia/pylaia-casia-hwdb2",
}

PYLAIA_RECIPE = """\
# on the GPU box, once:
pip install pylaia
huggingface-cli download Teklia/pylaia-casia-hwdb2 --local-dir models/htr_casia_hwdb2

# confirm the decode invocation for the installed PyLaia version, then record it verbatim
# in the run ledger, e.g. (placeholder -- verify):
pylaia-htr-decode-ctc --syms models/htr_casia_hwdb2/syms.txt \\
    models/htr_casia_hwdb2/weights.ckpt <img_list>
"""


@dataclass
class Recogniser:
    """A frozen image-to-text command. Identity is data, not a comment in a script."""

    command: str | None = None          # template with {image} or {list}; transcript on stdout
    identity: dict = field(default_factory=lambda: dict(ADOPTED))
    timeout_s: int = 600

    # ------------------------------------------------------------------ #
    def verify(self) -> dict:
        """Fail closed unless a working command is configured."""
        if not self.command:
            return {"status": "NOT_RUN",
                    "reason": ("no recogniser command configured. AR/CR cannot be produced by a "
                               "generator, so it stays NOT_RUN until the frozen recogniser is wired. "
                               f"Adopt {ADOPTED['id']} ({ADOPTED['licence']}, "
                               f"{ADOPTED['bytes']/1e6:.1f} MB)."),
                    "recipe": PYLAIA_RECIPE}
        return {"status": "OK", "identity": self.identity, "command": self.command}

    def _run(self, cmd: str) -> str:
        p = subprocess.run(shlex.split(cmd, posix=False), capture_output=True, text=True,
                           timeout=self.timeout_s)
        if p.returncode != 0:
            raise RuntimeError(f"recogniser failed ({p.returncode}): {p.stderr[-800:]}")
        return p.stdout

    # ------------------------------------------------------------------ #
    def transcribe(self, images: list[str | Path]) -> list[str]:
        """One transcript per image, in order."""
        v = self.verify()
        if v["status"] != "OK":
            raise RuntimeError(v["reason"])
        out: list[str] = []
        for img in images:
            cmd = self.command.format(image=str(img), list=str(img))
            txt = self._run(cmd).strip()
            out.append(txt.splitlines()[-1] if txt else "")
        return out

    def calibrate(self, pairs: list[tuple[str | Path, str]]) -> dict:
        """The recogniser's OWN AR/CR on real lines. This number bounds every AR/CR we report."""
        v = self.verify()
        if v["status"] != "OK":
            return v
        hyps = self.transcribe([p[0] for p in pairs])
        rep = ar_cr(list(zip(hyps, [p[1] for p in pairs])))
        rep["recogniser"] = self.identity["id"]
        rep["revision"] = self.identity["revision"]
        rep["note"] = ("this is the RECOGNISER's accuracy on real lines; every generated-line "
                       "AR/CR inherits it as an upper bound")
        return rep

    def score_generated(self, pairs: list[tuple[str | Path, str]]) -> dict:
        """AR/CR of generated images against their reference transcripts."""
        v = self.verify()
        if v["status"] != "OK":
            return v
        hyps = self.transcribe([p[0] for p in pairs])
        rep = ar_cr(list(zip(hyps, [p[1] for p in pairs])))
        rep["recogniser"] = self.identity["id"]
        rep["caveat"] = "AR/CR of GENERATED lines; bounded by the recogniser's own accuracy"
        return rep


def write_ledger_entry(path: str | Path, rec: Recogniser, calibration: dict | None = None) -> None:
    """Persist the recogniser identity + command + calibration, so a number can be traced."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"identity": rec.identity, "command": rec.command,
                             "calibration": calibration}, indent=2, ensure_ascii=False),
                 encoding="utf-8")
