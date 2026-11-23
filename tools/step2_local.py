import json, time, torch, numpy as np
from PIL import Image, ImageDraw
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3, collate_v3
from cent.train_v3 import build, step_loss
from benchmark.decode_v3 import decode, reconstruct, draw_payload as dp
cfg=V3Config.load("cent/configs/v3_overfit1.json")
cfg.stage="s2"; cfg.use_anchor_memory=False; cfg.lambda_layout=1.0
cfg.stroke_mean=None; cfg.stroke_std=None
dev="cuda"; torch.manual_seed(cfg.seed)
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=1)
b=collate_v3([ds[0]]); gt=reconstruct(b["cont"].to(dev))[:, :int(b["n_strokes"][0])]
print("[data] strokes",int(b["n_strokes"][0]),"chars",int(b["n_chars"][0]),flush=True)
m=build(cfg,dev); opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg.lr,total_steps=14000,pct_start=0.1)
t0=time.time()
for s in range(14000):
    m.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16): loss,logs=step_loss(m,b,cfg,dev,"s2")
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step()
    if s<13999: sch.step()
m.eval()
with torch.no_grad():
    r=decode(m,cfg,b["text"].to(dev),b["text_mask"].to(dev),b["refs"].to(dev),
             b["ref_mask"].to(dev),int(b["n_chars"][0]),mode="free",temperature=0.0, anchor_place=True)
p=r["payload"][:,:r["n_strokes"][0]]; im,_,ooc=dp(p,2048,128); gi,_,_=dp(gt,2048,128)
M=min(gt.shape[1],p.shape[1]); err=float((p[0,:M]-gt[0,:M]).abs().mean())
print(json.dumps({"err":round(err,4),"ooc":ooc,"n":int(p.shape[1]),"n_gt":int(gt.shape[1]),
 "loss":round(float(loss),2),"count":round(float(logs.get("count",0)),4),
 "x":[round(float(p[0,:,0,0].min()),2),round(float(p[0,:,0,0].max()),2)],"sec":round(time.time()-t0,1)}),flush=True)
H,W,band=128,2048,22
rows=[("GT",gi),(f"MODEL err={err:.3f} ooc={ooc} n={p.shape[1]}/78",im)]
out=Image.new("L",(W,len(rows)*(H+band)+band),255); d=ImageDraw.Draw(out)
for i,(lab,a) in enumerate(rows):
    y=band+i*(H+band); d.text((8,y-band+4),lab,fill=0)
    out.paste(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)),(0,y))
out.save("results/local_step2.png"); print("[img] results/local_step2.png",flush=True)


