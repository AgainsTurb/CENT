import json, torch, torch.nn as nn, pathlib
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3
from benchmark.decode_v3 import reconstruct
cfg=V3Config.load("cent/configs/v3_overfit1.json")
meta=json.loads((pathlib.Path(cfg.canonical_root)/"meta.json").read_text(encoding="utf-8"))
CID=meta["charset"].index("\u7684")
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=3000)
K=12; S=[]
for i in range(len(ds)):
    it=ds[i]
    ids=it["text"][:int(it["n_chars"])].tolist()
    if CID not in ids: continue
    N=int(it["n_strokes"]); cont=it["cont"][:N]; ci=it["char_idx"][:N]
    pay=reconstruct(cont.unsqueeze(0))[0]
    for c in range(int(it["n_chars"])):
        if int(it["text"][c])!=CID: continue
        s=(ci==c)
        if int(s.sum())==0: continue
        p=pay[s]; pts=p.reshape(-1,2)
        lo=pts.min(0).values; wh=(pts.max(0).values-lo).clamp_min(1e-3)
        loc=(p-lo)/wh; st=loc[:,0]; en=loc[:,3]
        d=torch.cat([st[0:1], st[1:]-en[:-1]],0)
        ctrl=(loc[:,1:]-st[:,None,:]).reshape(-1,6)
        tok=torch.cat([d,ctrl],-1)[:K]
        S.append((tok, tok.shape[0]))
print("[data] samples:",len(S),flush=True)
torch.manual_seed(0); dev="cuda"
idx=torch.randperm(len(S)); ntr=int(len(S)*0.8)
tr=[S[j] for j in idx[:ntr]]; te=[S[j] for j in idx[ntr:]]
def batch(T):
    tgt=torch.zeros(len(T),K,8,device=dev); m=torch.zeros(len(T),K,device=dev)
    for j,(t,n) in enumerate(T):
        tgt[j,:n]=t.to(dev); m[j,:n]=1
    return tgt,m
tgt,m=batch(te); tgt_tr,m_tr=batch(tr)
mean_proto=(tgt_tr*m_tr.unsqueeze(-1)).sum(0)/m_tr.sum(0,keepdim=True).clamp_min(1)
base_l1=float((((mean_proto.unsqueeze(0)-tgt).abs().mean(-1))*m).sum()/m.sum().clamp_min(1))
proto=nn.Parameter(torch.zeros(K,8,device=dev))
opt=torch.optim.Adam([proto],lr=1e-2)
for ep in range(2000):
    l1=((((proto.unsqueeze(0)-tgt_tr).abs().mean(-1))*m_tr).sum()/m_tr.sum().clamp_min(1))
    opt.zero_grad(set_to_none=True); l1.backward(); opt.step()
with torch.no_grad():
    l1=float((((proto.unsqueeze(0)-tgt).abs().mean(-1))*m).sum()/m.sum().clamp_min(1))
print(json.dumps({"samples":len(S),"train":len(tr),"test":len(te),
 "mean_proto_l1":round(base_l1,4),"learned_l1":round(l1,4)}))
