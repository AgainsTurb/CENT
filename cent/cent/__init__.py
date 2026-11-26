"""CENT — Chinese handwriting generation of text LINES with cursive continuity.

Package layout
--------------
cent.data     : SCUT-COUCH / CASIA readers, tokenisation, boundary labels
cent.models   : InkVAE, BoundaryDiT (main), AR baseline (ablation)
cent.losses   : diffusion / boundary / VAE objectives
cent.metrics  : continuity + layout + trajectory metrics
cent.sampling : DDIM with boundary (cursive) control
cent.train    : full training loop (config-driven, NOT run in this session)
"""

__version__ = "0.1.0"
