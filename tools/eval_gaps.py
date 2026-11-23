import torch, sys
from torch.utils.data import DataLoader
from cent.data_strokes import StrokeLineDataset, collate_strokes
from cent.models.line_stroke import StrokeDenoiser
from cent.models.line import LineStyleEncoder
from cent.models.diffusion import Diffusion
from cent.config import CENTConfig

RUN = sys.argv[1] if len(sys.argv) > 1 else "configs_stroke_r5.json"
CK  = sys.argv[2] if len(sys.argv) > 2 else "runs/stroke_r5/ckpt_last.pt"
cfg = CENTConfig.load(RUN)
ck = torch.load(CK, map_location="cuda", weights_only=False)
den = StrokeDenoiser(dim=cfg.dim, depth=cfg.depth, n_heads=cfg.n_heads, n_classes=cfg.n_classes,
                     k_refs=cfg.k_refs, max_strokes=cfg.max_tokens).cuda().eval()
sty = LineStyleEncoder(dim=cfg.dim, k_max=cfg.k_refs, n_types=cfg.n_types, n_heads=cfg.n_heads).cuda().eval()
den.load_state_dict(ck["denoiser"]); sty.load_state_dict(ck["style"])
diff = Diffusion(T=1000).to("cuda")
ds = StrokeLineDataset(cfg.canonical_root, "val", k_refs=cfg.k_refs, img_h=cfg.img_size,
                       canvas_w=cfg.canvas_width, max_strokes=cfg.max_tokens,
                       tau=cfg.join_tau, paired_only=True, limit=300)
dl = DataLoader(ds, batch_size=16, collate_fn=collate_strokes)
gj, gl = [], []
with torch.no_grad():
    for b in dl:
        z = sty(b["refs"].cuda(), b["ref_mask"].cuda())
        st = torch.cat([z[0].unsqueeze(1), z[1], z[2].unsqueeze(1)], 1)
        mask = b["mask"].cuda()
        t = torch.full((mask.shape[0],), 50, device="cuda")
        eps = den(b["tokens"].cuda(), t, mask, st, b["text"].cuda(), b["text_mask"].cuda())
        abar = diff.alpha_bar.to("cuda")[t].view(-1, 1, 1)
        x0 = ((b["tokens"].cuda() - (1 - abar).sqrt() * eps) / abar.sqrt().clamp_min(1e-5)).clamp(-2, 2)
        p0, p3 = x0[..., 0:2], x0[..., 6:8]
        gap = (p3[:, :-1] - p0[:, 1:]).norm(dim=-1)
        j = b["join"][:, :-1].cuda()
        v = (mask[:, :-1] * mask[:, 1:]) > 0.5
        gj.append(gap[v & (j == 1)]); gl.append(gap[v & (j == 0)])
GJ = float(torch.cat(gj).mean()); GL = float(torch.cat(gl).mean())
print(f"  t=50 val: joined mean gap {GJ:.4f} | lifted {GL:.4f} | ratio {GL/max(GJ,1e-6):.2f}x")
print(f"  (real data: joined ~0, lifted p50 0.11)")

# --- the paper's metric: join-rate error and F1 from the predicted geometry ---
allg, allt = [], []
with torch.no_grad():
    for b in dl:
        z = sty(b["refs"].cuda(), b["ref_mask"].cuda())
        st = torch.cat([z[0].unsqueeze(1), z[1], z[2].unsqueeze(1)], 1)
        mask = b["mask"].cuda()
        t = torch.full((mask.shape[0],), 50, device="cuda")
        eps = den(b["tokens"].cuda(), t, mask, st, b["text"].cuda(), b["text_mask"].cuda())
        abar = diff.alpha_bar.to("cuda")[t].view(-1, 1, 1)
        x0 = ((b["tokens"].cuda() - (1 - abar).sqrt() * eps) / abar.sqrt().clamp_min(1e-5)).clamp(-2, 2)
        p0, p3 = x0[..., 0:2], x0[..., 6:8]
        gap = (p3[:, :-1] - p0[:, 1:]).norm(dim=-1)
        j = b["join"][:, :-1].cuda(); v = (mask[:, :-1] * mask[:, 1:]) > 0.5
        allg.append(gap[v]); allt.append(j[v])
g = torch.cat(allg); y = torch.cat(allt)
for tau in (0.05, 0.08, 0.10):
    p = (g < tau).float()
    tp = float(((p==1)&(y==1)).sum()); fp = float(((p==1)&(y==0)).sum()); fn = float(((p==0)&(y==1)).sum())
    pr = tp/max(tp+fp,1); rc = tp/max(tp+fn,1)
    print(f"  tau={tau}: pred_rate {float(p.mean()):.3f} gt {float(y.mean()):.3f} "
          f"rate_err {abs(float(p.mean())-float(y.mean())):.3f} P {pr:.3f} R {rc:.3f} F1 {2*pr*rc/max(pr+rc,1e-9):.3f}")
