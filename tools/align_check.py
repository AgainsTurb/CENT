import json, pathlib
from cent.config_v3 import V3Config
from cent.data_v3 import StrokeDatasetV3
cfg=V3Config.load("cent/configs/v3_overfit1.json")
root=pathlib.Path(cfg.canonical_root)
recs=[json.loads(l) for l in (root/"train"/"index.jsonl").open(encoding="utf-8") if l.strip()][:300]
ds=StrokeDatasetV3(cfg.canonical_root,"train",k_refs=cfg.k_refs,img_h=cfg.img_height,
 canvas_w=cfg.canvas_width,max_chars=cfg.max_chars,max_strokes_char=cfg.max_strokes_char,
 max_strokes_line=cfg.max_strokes_line,join_tau=cfg.join_tau,n_classes=cfg.n_classes,limit=300)
n=min(len(ds),len(recs))
sum_rec=sum(len(r.get("text","")) for r in recs[:n])
sum_ds=sum(int(ds[i]["n_chars"]) for i in range(n))
de_rec=sum(r.get("text","").count("\u7684") for r in recs[:n])
meta=json.loads((root/"meta.json").read_text(encoding="utf-8")); cid=meta["charset"].index("\u7684")
de_ds=sum((ds[i]["text"][:int(ds[i]["n_chars"])]==cid).sum().item() for i in range(n))
first=[{"file_chars":len(recs[i].get("text","")),"ds_chars":int(ds[i]["n_chars"]),
        "file_head":recs[i].get("text","")[:12]} for i in range(min(5,n))]
print(json.dumps({"compared":n,"file_total_chars":sum_rec,"ds_total_chars":sum_ds,
 "de_in_file":de_rec,"de_in_ds":de_ds,"first5":first},ensure_ascii=False))
