import json, torch, torch.nn as nn, numpy as np, pathlib
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3
from benchmark.decode_v3 import reconstruct
cfg=V3Config.load("cent/configs/v3_overfit1.json")
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=800)
K=12; S=[]
for i in range(len(ds)):
    it=ds[i]; N=int(it["n_strokes"]); n=int(it["n_chars"])
    w=int(ds.base.records[ds.base.usable[i]]["writer"])
    cont=it["cont"][:N]; ci=it["char_idx"][:N]; pay=reconstruct(cont.unsqueeze(0))[0]
    for c in range(n):
        s=(ci==c)
        if int(s.sum())==0: continue
        p=pay[s]; pts=p.reshape(-1,2)
        lo=pts.min(0).values; wh=(pts.max(0).values-lo).clamp_min(1e-3)
        loc=(p-lo)/wh; st=loc[:,0]; en=loc[:,3]
        d=torch.cat([st[0:1], st[1:]-en[:-1]],0)
        ctrl=(loc[:,1:]-st[:,None,:]).reshape(-1,6)
        tok=torch.cat([d,ctrl],-1)[:K]
        S.append((int(it["text"][c]), tok, tok.shape[0], w))
print("[data] chars",len(S),"distinct",len({s[0] for s in S}),"writers",len({s[3] for s in S}),flush=True)
ws=sorted({s[3] for s in S}); te_w=set(ws[-max(1,len(ws)//5):])
tr=[s for s in S if s[3] not in te_w]; te=[s for s in S if s[3] in te_w]
def batch(T,dev):
    ids=torch.tensor([s[0] for s in T],device=dev)
    tgt=torch.zeros(len(T),K,8,device=dev); m=torch.zeros(len(T),K,device=dev)
    cnt=torch.zeros(len(T),dtype=torch.long,device=dev)
    for j,(_,t,n,_) in enumerate(T):
        tgt[j,:n]=t.to(dev); m[j,:n]=1; cnt[j]=min(n,12)
    return ids,tgt,m,cnt
dev="cuda"; torch.manual_seed(0)
emb=nn.Embedding(cfg.n_classes,64).to(dev); hd=nn.Linear(64,K*8).to(dev); ch=nn.Linear(64,13).to(dev)
opt=torch.optim.AdamW(list(emb.parameters())+list(hd.parameters())+list(ch.parameters()),lr=3e-3)
ids,tgt,m,cnt=batch(tr,dev)
for ep in range(4000):
    e=emb(ids); pred=hd(e).view(-1,K,8); lg=ch(e)
    l1=((pred-tgt).abs().mean(-1)*m).sum()/m.sum().clamp_min(1)
    lc=nn.functional.cross_entropy(lg,cnt)
    opt.zero_grad(set_to_none=True); (l1+5.0*lc).backward(); opt.step()
with torch.no_grad():
    ids,tgt,m,cnt=batch(te,dev); e=emb(ids); pred=hd(e).view(-1,K,8); lg=ch(e)
    l1t=float(((pred-tgt).abs().mean(-1)*m).sum()/m.sum().clamp_min(1))
    acc=float((lg.argmax(-1)==cnt).float().mean())
    ids2,tgt2,m2,cnt2=batch(tr,dev); e2=emb(ids2); p2=hd(e2).view(-1,K,8)
    l1tr=float(((p2-tgt2).abs().mean(-1)*m2).sum()/m2.sum().clamp_min(1))
    cnt2acc=float((ch(e2).argmax(-1)==cnt2).float().mean())
print(json.dumps({"train":len(tr),"test":len(te),"test_writers":len(te_w),
 "train_l1":round(l1tr,4),"test_l1":round(l1t,4),
 "train_cnt_acc":round(cnt2acc,3),"test_cnt_acc":round(acc,3)}))
