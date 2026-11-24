import json, torch, torch.nn as nn
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3
from benchmark.decode_v3 import reconstruct
cfg=V3Config.load("cent/configs/v3_overfit1.json")
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=400)
K=12; samples=[]
for i in range(len(ds)):
    it=ds[i]; N=int(it["n_strokes"]); n=int(it["n_chars"])
    cont=it["cont"][:N]; ci=it["char_idx"][:N]; pay=reconstruct(cont.unsqueeze(0))[0]
    for c in range(n):
        s=(ci==c)
        if int(s.sum())==0: continue
        p=pay[s]; pts=p.reshape(-1,2)
        lo=pts.min(0).values; wh=(pts.max(0).values-lo).clamp_min(1e-3)
        loc=(p-lo)/wh                                  # LOCAL frame, unit box
        st=loc[:,0]; en=loc[:,3]
        d=torch.cat([st[0:1], st[1:]-en[:-1]],0)
        ctrl=(loc[:,1:]-st[:,None,:]).reshape(-1,6)
        tok=torch.cat([d,ctrl],-1)[:K]
        samples.append((int(it["text"][c]), tok, tok.shape[0]))
print("[data] chars",len(samples),"distinct",len({s[0] for s in samples}),flush=True)
dev="cuda"; torch.manual_seed(0)
lo_i=int(len(samples)*0.8); tr,te=samples[:lo_i],samples[lo_i:]
emb=nn.Embedding(cfg.n_classes,64).to(dev); head=nn.Linear(64,K*8).to(dev); ch=nn.Linear(64,13).to(dev)
opt=torch.optim.AdamW(list(emb.parameters())+list(head.parameters())+list(ch.parameters()),lr=3e-3)
def batch(S):
    ids=torch.tensor([s[0] for s in S],device=dev)
    tgt=torch.zeros(len(S),K,8,device=dev); m=torch.zeros(len(S),K,device=dev)
    for j,(_,t,n) in enumerate(S):
        tgt[j,:n]=t.to(dev); m[j,:n]=1
    cnt=torch.tensor([s[2] for s in S],device=dev).clamp(0,12)
    return ids,tgt,m,cnt
for ep in range(300):
    ids,tgt,m,cnt=batch(tr)
    e=emb(ids); pred=head(e).view(-1,K,8); lg=ch(e)
    l1=((pred-tgt).abs().mean(-1)*m).sum()/m.sum().clamp_min(1)
    lc=nn.functional.cross_entropy(lg,cnt)
    opt.zero_grad(set_to_none=True); (l1+lc).backward(); opt.step()
with torch.no_grad():
    ids,tgt,m,cnt=batch(te); e=emb(ids); pred=head(e).view(-1,K,8); lg=ch(e)
    l1=float(((pred-tgt).abs().mean(-1)*m).sum()/m.sum().clamp_min(1))
    acc=float((lg.argmax(-1)==cnt).float().mean())
print(json.dumps({"train_chars":len(tr),"test_chars":len(te),"local_l1":round(l1,4),
 "count_acc":round(acc,3),"final_train_loss":round(float(l1+lc),3)}))
