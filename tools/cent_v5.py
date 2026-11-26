import json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3
from benchmark.decode_v3 import reconstruct

ZD, P, TDIFF = 64, 384, 100

def bez(p,n):
    t=torch.linspace(0,1,n,device=p.device).view(1,n,1)
    return ((1-t)**3*p[:,0:1]+3*(1-t)**2*t*p[:,1:2]+3*(1-t)*t**2*p[:,2:3]+t**3*p[:,3:4])[0]

def to_fixed(pts, cs, n=P):
    """Resample a variable-length polyline to n points; remap character starts."""
    a=pts.numpy(); k=len(a)
    idx=np.linspace(0,k-1,n)
    out=np.stack([np.interp(idx,np.arange(k),a[:,0]),np.interp(idx,np.arange(k),a[:,1])],1)
    cs2=np.round(np.array(cs,dtype=float)*(n-1)/max(k-1,1)).astype(int)
    return torch.tensor(out,dtype=torch.float32), np.clip(cs2,0,n-1)

class Enc(nn.Module):
    def __init__(s,h=128):
        super().__init__(); s.n=nn.Sequential(nn.Conv1d(2,h,5,2,2),nn.GELU(),nn.Conv1d(h,h,5,2,2),nn.GELU(),nn.Conv1d(h,h,5,2,2),nn.GELU())
        s.mu=nn.Conv1d(h,ZD,1); s.lv=nn.Conv1d(h,ZD,1)
    def forward(s,x):
        f=s.n(x); return s.mu(f),s.lv(f)

class Dec(nn.Module):
    def __init__(s,k=8,h=128):
        super().__init__(); s.k=k
        s.up=nn.Sequential(nn.ConvTranspose1d(ZD,h,4,2,1),nn.GELU(),nn.ConvTranspose1d(h,h,4,2,1),nn.GELU(),nn.ConvTranspose1d(h,h,4,2,1),nn.GELU(),nn.Conv1d(h,6*k,1))
    def forward(s,z): return F.interpolate(s.up(z),size=P,mode="linear",align_corners=False)

def nll(p,xy,k=8):
    B,_,T=p.shape; q=p.view(B,k,6,T).permute(0,3,1,2)
    pi=torch.softmax(q[...,0],-1).clamp_min(1e-8)
    s1=torch.exp(q[...,3].clamp(-6,3)).clamp_min(1e-2); s2=torch.exp(q[...,4].clamp(-6,3)).clamp_min(1e-2)
    r=torch.tanh(q[...,5]).clamp(-.98,.98); one=(1-r*r).clamp_min(1e-4)
    z1=(xy[...,0:1]-q[...,1])/s1; z2=(xy[...,1:2]-q[...,2])/s2
    quad=(z1*z1+z2*z2-2*r*z1*z2)/one
    logN=-.5*quad-math.log(2*math.pi)-torch.log(s1)-torch.log(s2)-.5*torch.log(one)
    return -torch.logsumexp(torch.log(pi)+logN,-1).mean()

class CTC(nn.Module):
    def __init__(s,ncls):
        super().__init__(); s.proj=nn.Conv1d(ZD,128,1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,256,.1,batch_first=True,norm_first=True),2)
        s.out=nn.Linear(128,ncls+1)
    def forward(s,z):
        h=F.interpolate(z,size=P,mode="linear",align_corners=False)
        h=s.tf(s.proj(h).transpose(1,2))
        return torch.log_softmax(s.out(h),-1).transpose(0,1)

class Heads(nn.Module):
    def __init__(s):
        super().__init__()
        s.anc=nn.Sequential(nn.Linear(ZD,128),nn.GELU(),nn.Linear(128,4))
        s.bnd=nn.Sequential(nn.Linear(2*ZD,128),nn.GELU(),nn.Linear(128,3))
    @staticmethod
    def probe(z,idx):
        B,_,L=z.shape; zt=z.transpose(1,2)
        li=(idx.float()/ (P-1) *(L-1)).round().long().clamp(0,L-1)
        return torch.stack([zt[b,li[b]] for b in range(B)])

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    cfg=V3Config.load("cent/configs/v3_overfit1.json")
    ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
      canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
      max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=24)
    LINES=[]
    for i in range(len(ds)):
        it=ds[i]; N=int(it["n_strokes"]); ci=it["char_idx"][:N].numpy()
        pay=reconstruct(it["cont"][:N].unsqueeze(0).to(dev))[0]
        per=40; pts=[]; cs=[]; ks=[]
        for k in range(N):
            if k==0 or ci[k]!=ci[k-1]: cs.append(len(pts)); ks.append(k)
            pts.append(bez(pay[k:k+1],per).cpu())
        pts=torch.cat(pts,0); lo=pts.min(0).values; hi=pts.max(0).values
        pts=(pts-lo)/(hi-lo).clamp_min(1e-6)
        f,cs2=to_fixed(pts,cs)
        # JOIN LABEL, correct version: measured on the ORIGINAL strokes in canvas units.
        # The previous version measured the gap between resampled points, where resampling
        # itself creates gaps, so every boundary was labelled "not joined" (zeta_mean 0.0).
        span=(hi-lo).clamp_min(1e-6)
        bnd=[]
        for j in range(1,len(ks)):
            prev_end=pay[ks[j]-1,3,:].cpu()    # P3 of the previous character's last stroke
            cur_start=pay[ks[j],0,:].cpu()     # P0 of this character's first stroke
            gap=(cur_start-prev_end)/span      # normalised gap, same frame as the trajectory
            zt=1.0 if float((cur_start-prev_end).norm())<float(cfg.join_tau) else 0.0
            bnd.append(torch.cat([torch.tensor([zt]),gap.to('cpu')]))
        nc=int(it["n_chars"]); t=it["text"][:min(nc,8)].numpy()
        ids=np.zeros(8,np.int64); ids[:len(t)]=t
        LINES.append((f,cs2,ids,len(t),bnd))
    C=min(max(len(l[1])-1 for l in LINES),16)
    X=torch.stack([l[0] for l in LINES]).to(dev)
    S=torch.zeros(len(LINES),C+1,dtype=torch.long,device=dev)
    A=torch.zeros(len(LINES),C,4,device=dev); Mc=torch.zeros(len(LINES),C,device=dev)
    Bt=torch.zeros(len(LINES),max(C-1,1),3,device=dev); Mb=torch.zeros(len(LINES),max(C-1,1),device=dev)
    for b,(f,cs2,_,_,bnd_l) in enumerate(LINES):
        n=min(len(cs2)-1,C)
        S[b,:n+1]=torch.tensor(cs2[:n+1],device=dev); S[b,n+1:]=cs2[n]
        for j in range(n):
            g=(f[cs2[j]:cs2[j+1]] if cs2[j+1]>cs2[j] else f[cs2[j]:cs2[j]+2]).to(dev)
            lo=g.min(0).values; hi=g.max(0).values
            A[b,j]=torch.cat([lo,(hi-lo).clamp_min(1e-6)]); Mc[b,j]=1
        for j in range(1,n):
            if j-1 < len(bnd_l):
                Bt[b,j-1]=bnd_l[j-1].to(dev); Mb[b,j-1]=1
    T=torch.tensor(np.stack([l[2] for l in LINES])).to(dev); TL=torch.tensor([l[3] for l in LINES]).to(dev)
    print(f"[data] B={X.shape[0]} P={P} C={C} chars/line median={int(np.median([len(l[1])-1 for l in LINES]))} join_frac={round(float(Mb.sum() and (Bt[...,0]*Mb).sum()/Mb.sum()),3)}",flush=True)
    enc,dec,ctc,heads=Enc().to(dev),Dec().to(dev),CTC(cfg.n_classes).to(dev),Heads().to(dev)
    prm=[*enc.parameters(),*dec.parameters(),*ctc.parameters(),*heads.parameters()]
    opt=torch.optim.AdamW(prm,lr=2e-3); cl=nn.CTCLoss(blank=0,zero_infinity=True)
    t0=time.time()
    for s in range(900):
        mu,lv=enc(X.transpose(1,2)); z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
        rec=dec(z); lnll=nll(rec,X)
        f=heads.probe(mu,S[:,:C]); pa=heads.anc(f)
        la=((pa-A).abs().sum(-1)*Mc).sum()/Mc.sum().clamp_min(1)
        if C>2:
            _j=(pa[:,2:,0]-2*pa[:,1:-1,0]+pa[:,:-2,0]).abs()
            _m=Mc[:,1:C-1][:, :_j.shape[1]]
            la=la+0.5*(_j*_m).sum()/_m.sum().clamp_min(1)
        f2=torch.cat([f[:,:-1],f[:,1:]],-1) if C>1 else torch.cat([f,f],-1)
        pb=heads.bnd(f2); zt=torch.sigmoid(pb[...,0]); g=pb[...,1:]
        lb=((F.binary_cross_entropy(zt,Bt[...,0],reduction="none")*Mb).sum()/Mb.sum().clamp_min(1)
            + ((g-Bt[...,1:]).abs().sum(-1)*Mb).sum()/Mb.sum().clamp_min(1))
        contract=((g-Bt[...,1:]*(1-zt).unsqueeze(-1)).abs().sum(-1)*Mb).sum()/Mb.sum().clamp_min(1)
        lc=cl(ctc(mu),T,torch.full((X.shape[0],),P,dtype=torch.long,device=dev),TL)
        loss=lnll+la+lb+0.5*contract+lc
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(prm,1.0); opt.step()
        if s%150==0: print(f"[{s}] nll {lnll.item():.2f} anc {la.item():.3f} bnd {lb.item():.3f} "
                           f"contract {contract.item():.4f} ctc {lc.item():.2f}",flush=True)
    with torch.no_grad():
        mu,_=enc(X.transpose(1,2)); rec=dec(mu)
        q=rec.view(-1,8,6,P).permute(0,3,1,2); pi=torch.softmax(q[...,0],-1)
        rc=torch.stack([(q[...,1]*pi).sum(-1)[0],(q[...,2]*pi).sum(-1)[0]],1).cpu().numpy()
        f=heads.probe(mu,S[:,:C]); f2=torch.cat([f[:,:-1],f[:,1:]],-1)
        zt=torch.sigmoid(heads.bnd(f2)[...,0])[0].cpu().numpy()
        for j,t in enumerate(zt):
            if t>0.5 and j+1<=C:
                si=int(S[0,j+1].item()); si=min(max(si,2),len(rc)-3); rc[si]=(rc[si-1]+rc[si+1])/2
    def render(xy,w=1024,h=96):
        img=np.ones((h,w),np.float32); pp=(np.clip(xy,0,1)*np.array([w-1,h-1])).astype(int)
        for a,b in zip(pp[:-1],pp[1:]):
            n=int(np.hypot(*(b-a)))+1
            for u in np.linspace(0,1,n):
                x,y=(a+(b-a)*u).astype(int); img[max(y-1,0):y+2,max(x-1,0):x+2]=0
        return img
    rows=[("GT",render(X[0].cpu().numpy())),("CENT full line (anchors+contract+C1)",render(rc))]
    out=Image.new("L",(1024,2*116+20),255); d=ImageDraw.Draw(out)
    for i,(lab,arr) in enumerate(rows):
        y=20+i*116; d.text((6,y-14),lab,fill=0); out.paste(Image.fromarray((arr*255).astype(np.uint8)),(0,y))
    Path("results").mkdir(exist_ok=True); out.save("results/cent_v5.png")
    print(json.dumps({"out":"results/cent_v5.png","sec":round(time.time()-t0,1),
                      "C":C,"zeta_mean":round(float(zt.mean()),3),"boundaries":int(Mb.sum())}))
main()
