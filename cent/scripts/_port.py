import pathlib

p = pathlib.Path("cent/models/inkvae.py")
src = p.read_text(encoding="utf-8")

RESID = '''
class Residual(nn.Module):
    """DiffInk's residual block: GN -> GELU -> conv3 -> GN -> GELU -> conv1 + skip."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [Residual(in_channels, hidden_channels) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


'''

# ---- 1. insert the blocks right before the InkVAE class
anchor = "class InkVAE(nn.Module):"
assert anchor in src
src = src.replace(anchor, RESID + anchor, 1)

# ---- 2. replace the encoder/decoder construction
i = src.find("        enc: list[nn.Module] = [")
j = src.find("        self.post_proj = nn.Linear(", i)
j = src.find("\n", j) + 1
assert i > 0 and j > i
new_init = '''        self.enc_in = nn.Conv1d(in_dim, 128, kernel_size=3, stride=2, padding=1)
        self.enc_rs1 = ResidualStack(128, 64, 4)
        self.enc_down2 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_rs2 = ResidualStack(256, 128, 4)
        self.enc_down3 = nn.Conv1d(256, 384, kernel_size=3, stride=2, padding=1)
        self.enc_rs3 = ResidualStack(384, 192, 4)
        self.enc_out = nn.Conv1d(384, 2 * z_dim, kernel_size=3, padding=1)

        dd = 128
        self.dd = dd
        self.dec_in = nn.Conv1d(z_dim, 384, kernel_size=3, padding=1)
        self.dec_rs1 = ResidualStack(384, 192, 4)
        self.dec_up1 = nn.ConvTranspose1d(384, 256, kernel_size=4, stride=2, padding=1)
        self.dec_rs2 = ResidualStack(256, 128, 4)
        self.dec_up2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_rs3 = ResidualStack(128, 64, 4)
        self.dec_up3 = nn.ConvTranspose1d(128, dd, kernel_size=4, stride=2, padding=1)

        head_dim = 6 * n_mix + n_pen
        self.head_dim = head_dim

        # positional conditioning at the latent resolution
        self.pos_dim = 32
        self.pos_mlp = nn.Sequential(nn.Linear(self.pos_dim, dd), nn.GELU(),
                                     nn.Linear(dd, z_dim))
        self.skip_drop = 0.0

        # global context before the head (DiffInk: 4 layers, d=256, FFN 1024)
        self.post = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dd, nhead=4, dim_feedforward=4 * dd,
                                       batch_first=True, norm_first=True, dropout=0.0),
            num_layers=4)
        self.post_proj = nn.Linear(dd, head_dim)
'''
src = src[:i] + new_init + src[j:]

# ---- 3. replace encode()
i = src.find("    def encode(self, x: torch.Tensor")
j = src.find("    def reparameterise(", i)
assert i > 0 and j > i
new_encode = '''    def encode(self, x: torch.Tensor, mask: torch.Tensor, *, return_skips: bool = False):
        """x: (B, T, 6) -> mu, logvar, latent_mask[, skips]."""
        B, T, C = x.shape
        pad = (-T) % self.stride
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        h = x.transpose(1, 2)
        h = self.enc_rs1(self.enc_in(h))
        h = self.enc_rs2(self.enc_down2(h))
        h = self.enc_rs3(self.enc_down3(h))
        h = self.enc_out(h)                        # (B, 2z, T_lat)
        mu, logvar = h.chunk(2, dim=1)
        mu = mu.transpose(1, 2)
        logvar = logvar.transpose(1, 2).clamp(-8.0, 8.0)
        T_lat = mu.shape[1]
        idx = torch.arange(T_lat, device=mask.device).unsqueeze(0)
        lm = ((idx * self.stride) < mask.sum(dim=1, keepdim=True)).float()
        if return_skips:
            return mu, logvar, lm, []
        return mu, logvar, lm

'''
src = src[:i] + new_encode + src[j:]

# ---- 4. replace decode()
i = src.find("    def decode(self, z: torch.Tensor")
j = src.find("    def forward(self, x: torch.Tensor", i)
assert i > 0 and j > i
new_decode = '''    def decode(self, z: torch.Tensor, skips: list | None = None):
        """z: (B, T_lat, z_dim) -> gmm (B, T, M, 6), pen logits (B, T, 4).

        Structure follows DiffInk: ResidualStack -> deconv, three times, then a
        Transformer over the whole sequence before the GMM head. Conv-only
        decoders reconstruct handwriting as a smooth average.
        """
        h = z.transpose(1, 2)
        h = h + self._pos(h.shape[-1], z.device, h.dtype)
        h = self.dec_in(h)
        h = self.dec_up1(self.dec_rs1(h))
        h = self.dec_up2(self.dec_rs2(h))
        h = self.dec_up3(self.dec_rs3(h))
        h = h.transpose(1, 2)
        h = h + self._sin_pe(h.shape[1], h.shape[2], h.device, h.dtype)
        h = self.post(h)
        h = self.post_proj(h)
        M = self.n_mix
        gmm = h[..., : 6 * M].view(*h.shape[:2], M, 6)
        pen = h[..., 6 * M:]
        return gmm, pen

'''
src = src[:i] + new_decode + src[j:]

p.write_text(src, encoding="utf-8")
print("ported ResidualStack + wide channels + 4-layer transformer")
