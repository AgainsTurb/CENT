import json, time, torch, numpy as np
from PIL import Image, ImageDraw
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3, collate_v3
from cent.train_v3 import build, step_loss
from benchmark.decode_v3 import decode, reconstruct, draw_payload as dp
cfg=V3Config.load("cent/configs/v3_overfit1.json"); cfg.stage="s2"
cfg.use_anchor_memory=False; cfg.lambda_layout=1.0; cfg.stroke_mean=None; cfg.stroke_std=None
dev="cuda"; torch.manual_seed(cfg.seed)
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=1)
b=collate_v3([ds[0]]); gt=reconstruct(b["cont"].to(dev))[:, :int(b["n_strokes"][0])]
m=build(cfg,dev); opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg.lr,total_steps=14000,pct_start=0.1)
for s in range(14000):
    m.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16): loss,logs=step_loss(m,b,cfg,dev,"s2")
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step()
    if s<13999: sch.step()
m.eval()
with torch.no_grad():
    r=decode(m,cfg,b["text"].to(dev),b["text_mask"].to(dev),b["refs"].to(dev),
             b["ref_mask"].to(dev),int(b["n_chars"][0]),mode="free",temperature=0.0,anchor_place=True)
p=r["payload"][:,:r["n_strokes"][0]]
def ext(x):  # stroke extent = |P3 - P0|
    return (x[...,3,:]-x[...,0,:]).norm(dim=-1)
# per-character slices
ci=b["char_idx"][0,:int(b["n_strokes"][0])].cpu()
idx=b["char_idx"].new_tensor([0])
per_char=[]; per_ext_g=[]; per_ext_p=[]
for c in range(int(b["n_chars"][0])):
    sel=(ci==c)
    if sel.sum()==0: continue
    M=min(p.shape[1],gt.shape[1]); Mm=M
    sel=sel[:Mm]
    d=(p[0,:Mm]-gt[0,:Mm]).abs().mean(-1).mean(-1)
    per_char.append(round(float(d[sel].mean()),3))
    per_ext_g.append(round(float(ext(gt[0])[sel].mean()),4))
    per_ext_p.append(round(float(ext(p[0])[sel].mean()),4))
gi,_,_=dp(gt,2048,128); im,_,ooc=dp(p,2048,128)
M=min(gt.shape[1],p.shape[1]); err=float((p[0,:M]-gt[0,:M]).abs().mean())
ink=float((im<0.5).mean()); ink_gt=float((gi<0.5).mean())
print(json.dumps({"err":round(err,4),"ooc":ooc,"n":int(p.shape[1]),
 "per_char_err":per_char,"ext_gt":per_ext_g,"ext_gen":per_ext_p,
 "ext_ratio":round(float(np.mean(np.array(per_ext_p)/np.maximum(np.array(per_ext_g),1e-6))),3),
 "ink_gen":round(ink,5),"ink_gt":round(ink_gt,5),"sec":0}))
H,W,band=128,2048,22
rows=[("GT",gi),(f"MODEL err={err:.3f} ooc={ooc}",im)]
im2=Image.new("L",(W,len(rows)*(H+band)+band),255); d2=ImageDraw.Draw(im2)
for i,(lab,arr) in enumerate(rows):
    y=band+i*(H+band); d2.text((8,y-band+4),lab,fill=0)
    im2.paste(Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8)),(0,y))
im2.save("results/local_diag.png"); print("[img]")
