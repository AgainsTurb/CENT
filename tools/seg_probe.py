import numpy as np, torch
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3
from benchmark.decode_v3 import reconstruct
cfg=V3Config.load("cent/configs/v3_overfit1.json")
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=200)
per=[]
for k in range(len(ds)):
    it=ds[k]; N=int(it["n_strokes"])
    if N<4: continue
    ci=it["char_idx"][:N]
    pay=reconstruct(it["cont"][:N].unsqueeze(0))[0]
    starts=pay[:,0,0].numpy(); ends=pay[:,3,0].numpy()
    true=set(i for i in range(1,N) if int(ci[i])!=int(ci[i-1]))
    if len(true)<2: continue
    widths=[]
    b=0
    idxs=sorted(true)+[N]
    for e in idxs:
        seg=pay[b:e]; widths.append(float(seg[:,:,0].max()-seg[:,:,0].min())); b=e
    cw=float(np.median(widths))
    if cw<=1e-6: continue
    per.append((N, starts, ends, cw, true))
print("[lines]",len(per),"strokes",sum(p[0] for p in per),flush=True)
def score(f):
    tp=fp=fn=0
    for N,starts,ends,cw,true in per:
        gaps=starts[1:]-ends[:-1]
        pred=set(int(k+1) for k in range(len(gaps)) if gaps[k]>f*cw)
        tp+=len(pred&true); fp+=len(pred-true); fn+=len(true-pred)
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return round(p,3),round(r,3),round(2*p*r/max(p+r,1e-9),3)
for f in (0.6,0.9,1.2,1.6,2.0):
    print(f"[tau {f}] P,R,F1 =",score(f),flush=True)
