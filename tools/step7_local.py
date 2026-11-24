import json, time, torch, numpy as np
from PIL import Image, ImageDraw
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3, collate_v3
from cent.train_v3 import build, step_loss
from benchmark.decode_v3 import decode, reconstruct, draw_payload as dp
torch.backends.cudnn.benchmark = True
cfg=V3Config.load("cent/configs/v3_overfit1.json"); cfg.stage="s2"
cfg.use_anchor_memory=False; cfg.lambda_layout=1.0
cfg.lambda_curve=10.0; cfg.lambda_render=5.0; cfg.use_char_dict=True
cfg.stroke_mean=None; cfg.stroke_std=None
dev="cuda"; torch.manual_seed(cfg.seed); N=8
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=N)
b=collate_v3([ds[i] for i in range(len(ds))])
m=build(cfg,dev); m.use_slot_pos=True
opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
STEPS=14000
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg.lr,total_steps=STEPS,pct_start=0.1)
t0=time.time()
for s in range(STEPS):
    m.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16): loss,logs=step_loss(m,b,cfg,dev,"s2")
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step()
    if s<STEPS-1: sch.step()
    if s%3500==0: print("[step]",s,round(float(loss),2),"render",round(float(logs.get("render",0)),4),flush=True)
ms=round((time.time()-t0)/STEPS*1000,1)
m.eval(); errs=[]; oocs=[]; imgs={}
with torch.no_grad():
    for k in range(min(N,8)):
        gtk=reconstruct(b["cont"].to(dev))[k:k+1,:int(b["n_strokes"][k])]
        r=decode(m,cfg,b["text"][k:k+1].to(dev),b["text_mask"][k:k+1].to(dev),
                 b["refs"][k:k+1].to(dev),b["ref_mask"][k:k+1].to(dev),
                 int(b["n_chars"][k]),mode="free",temperature=0.0,anchor_place=True)
        p=r["payload"][:,:r["n_strokes"][0]]
        im,_,ooc=dp(p,2048,128); gj,_,_=dp(gtk,2048,128)
        M=min(gtk.shape[1],p.shape[1]); errs.append(float((p[0,:M]-gtk[0,:M]).abs().mean())); oocs.append(ooc)
        if k==0: imgs["gt"]=gj; imgs["gen"]=im
print(json.dumps({"err_mean":round(float(np.mean(errs)),4),"err_each":[round(e,3) for e in errs],
 "ooc_mean":round(float(np.mean(oocs)),1),"loss":round(float(loss),2),
 "render":round(float(logs.get("render",0)),4),"ms_per_step":ms,"sec":round(time.time()-t0,1)}),flush=True)
H,W,band=128,2048,22
rows=[("GT line0",imgs["gt"]),(f"MODEL line0 err={errs[0]:.3f} ooc={oocs[0]}",imgs["gen"])]
im2=Image.new("L",(W,len(rows)*(H+band)+band),255); d2=ImageDraw.Draw(im2)
for i,(lab,arr) in enumerate(rows):
    y=band+i*(H+band); d2.text((8,y-band+4),lab,fill=0)
    im2.paste(Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8)),(0,y))
im2.save("results/local_step7.png"); print("[img]")

