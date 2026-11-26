import torch, numpy as np, json, pathlib, sys
sys.argv=["x"]
exec(open("tools/m9_test1.py",encoding="utf-8").read().split("dev=\"cuda\"")[0])
dev="cuda"
def batch(T):
    ids=torch.tensor([s[0] for s in T],device=dev)
    tgt=torch.zeros(len(T),K,8,device=dev); m=torch.zeros(len(T),K,device=dev); cnt=torch.zeros(len(T),dtype=torch.long,device=dev)
    for j,(_,t,n,_) in enumerate(T):
        tgt[j,:n]=t.to(dev); m[j,:n]=1; cnt[j]=min(n,12)
    return ids,tgt,m,cnt
ids,tgt,m,cnt=batch(tr)
# per-character MEAN prototype (the best possible constant)
acc=torch.zeros(cfg.n_classes,K,8,device=dev); num=torch.zeros(cfg.n_classes,1,device=dev)
acc.index_add_(0,ids,tgt*m.unsqueeze(-1)); num.index_add_(0,ids,m.sum(1,keepdim=True))
mean_proto=acc/num.clamp_min(1).unsqueeze(-1)
ids_e,tgt_e,m_e,cnt_e=batch(te)
mp=mean_proto[ids_e]
l1_mean=float((((mp-tgt_e).abs().mean(-1))*m_e).sum()/m_e.sum().clamp_min(1))
# per-character MODE count
cacc=torch.zeros(cfg.n_classes,13,device=dev); cacc.index_add_(0,ids,torch.nn.functional.one_hot(cnt,13).float())
mode_cnt=cacc.argmax(-1)
cacc_e=float((mode_cnt[ids_e]==cnt_e).float().mean())
print(json.dumps({"mean_proto_l1":round(l1_mean,4),"mean_proto_cnt_acc":round(float(cacc_e),3),
 "test_chars":len(te),"distinct_train_chars":int((num.squeeze(-1)>0).sum())}))
