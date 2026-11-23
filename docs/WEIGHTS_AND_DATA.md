# Weights and data: what is here, what the GPU box must fetch

Policy applied 2026-11-22: **any single weight file larger than 300 MB is recorded, not
downloaded** — the GPU box fetches it once at first run. Everything partially fetched during
this audit was deleted and the pip cache purged.

Only two weight files were small enough to fetch, and both are installed and load-verified.

## 1. Weights already downloaded and verified

| Baseline | File | Size | Destination | Verification |
|---|---|---|---|---|
| OLHWG | `unetddpm450000.pth` | 164,622,792 B | `baselines/olhwg/models/` | `torch.load` → 229 tensors, 41.13M params, keys `denoiser.*`, `char_embeddings.weight` |
| OLHWG | `style_encoder450000.pth` | 31,760,219 B | `baselines/olhwg/models/` | `torch.load` → 46 tensors, 7.94M params |

Source (public, no gating):
`https://huggingface.co/datasets/Immortalman12/OLHWD/resolve/main/<file>`
Byte counts match the Hub API exactly, so the downloads are complete rather than truncated.

## 2. Weights to fetch on the GPU box (all > 300 MB)

| Baseline | File | Size | Link | Destination |
|---|---|---|---|---|
| DiffBrush | `DiffBrush-ckpt.pt` | 1,165,546,469 B (1.09 GB) | https://drive.google.com/file/d/1EWzBmLtnQ42cTf3k_CYQ-nF3RXCb35I6/view | `baselines/diffbrush/model_zoo/` |
| DiffBrush | SD-1.5 VAE (`vae/diffusion_pytorch_model.safetensors`) | 334,643,276 B (319 MB) | https://huggingface.co/sd-legacy/stable-diffusion-v1-5 | `baselines/diffbrush/model_zoo/` (diffusers layout) |
| TGC-Diff | `temp_weight.pt` (Template Generation) | 931,372,922 B (888 MB) | https://drive.google.com/file/d/1rfH6ZFty-RQFtFf6_XwgXG8ZEx4njcvI/view | `baselines/tgc_diff/template/` |
| TGC-Diff | `weight.pt` (Finetuned Imitating-Diff) | 2,033,807,758 B (1.89 GB) | https://drive.google.com/file/d/1fjGTLNjXt7tVza1ORllxcuakFBWo5bZO/view | `baselines/tgc_diff/weight/` |
| DiffInk | `dit_epoch_1.pt` (InkDiT) | 2,914,621,632 B (2.78 GB) | https://drive.google.com/file/d/13sApjo9rqFHdfNnWmiWaRjFVWewJqECY/view | `baselines/diffink/outputs/ink_dit_tune_03_21_13_37/ckpt/` |

All four Drive files answered HTTP 200 with the byte counts above and the expected
`Content-Disposition` filenames, so they are open files, not permission-walled.

## 3. Weights under 300 MB that I did **not** fetch (download stopped on request)

| Baseline | File | Size | Link | Destination |
|---|---|---|---|---|
| TGC-Diff | `.cache.zip` (VAE + ResNet18) | 234,623,734 B (224 MB) | https://drive.google.com/file/d/1MA0ndX0kdMik11klXBfbx3MFb0wXgyP7/view | `baselines/tgc_diff/.cache/` (unzip) |
| DiffInk | `vae_epoch_100.pt` (InkVAE) | 178,524,954 B (170 MB) | https://drive.google.com/file/d/11fprScAKJnML2Dv_BFZ5JDSQ1cQm341t/view | `baselines/diffink/outputs/ink_vae_03_14_02_12/ckpt/` |
| SDT | `checkpoint-iter199999.pth` (Chinese) | 273,587,031 B (261 MB) | https://drive.google.com/file/d/1OAjW463bVgreCuDKT_aT8a94Cx-53ZKu/view | `baselines/sdt/model_zoo/` |
| DiffInk | `All_zi.json` — **the release vocabulary**, required for a valid score | 44,043 B | https://drive.google.com/file/d/1yQpL0oxC5dv8yXTdWuHsdkaZpAI4XYeQ/view | `baselines/diffink/datas/meta/` |

`All_zi.json` is tiny but load-bearing: DiffInk's text embedding is derived from it, so
without the release vocabulary its text conditioning cannot be reproduced (see
`docs/BASELINE_AUDIT.md`). It should be fetched even though the model files are deferred.

DiffInk and SDT publish **folders**, not individual files. Enumerate or fetch them with:

```bash
pip install gdown          # on the GPU box
gdown --folder https://drive.google.com/drive/folders/1h_uLmn-55WmbSBGh1ES8-rftAbDs8riB  # DiffInk: ckpts + datas
gdown --folder https://drive.google.com/drive/folders/1LendizOwcNXlyY946ThS8HQ4wJX--YL7  # SDT: Chinese/English/Japanese
gdown --folder https://drive.google.com/drive/folders/1N-MGRnXEZmxAW-98Hz2f-o80oHrNaN_a  # SDT content encoder
```
The DiffInk folder also contains `datas/*.h5` (its preprocessed dataset, several GB) — fetch
the `outputs/` subtree only unless the GPU box needs that data.

## 4. Not downloadable here at all

| Baseline | Artefact | Link | Note |
|---|---|---|---|
| MetaScript | `checkpoint/generator.pth` | https://pan.baidu.com/s/1UGHPKFVSvRj2QY_PbSjJGQ?pwd=1024 | **Baidu Netdisk only.** No Google Drive, no Hugging Face release (checked the GitHub releases API and the HF model API: zero hits). Needs a Baidu account and client, or a full retrain from the released recipe. |

## 5. Datasets: reachable, not downloaded (per instruction)

| Dataset | Source | Size | Note |
|---|---|---|---|
| CASIA-HWDB2.0–2.2 lines (`.dgrl`) | https://nlpr.ia.ac.cn/databases/handwriting/Download.html | 2,203 MB total (train 1,764 / test 439) | Free for academic use **after a signed application**; commercial use paid. Not a click-through download. |
| CASIA-OLHWDB2.0–2.2 lines (`.wptt`) | same page | 193 MB total (train 156 / test 37) | Same agreement. |
| ICDAR-2013 offline text | same page (competition test data) | 140 MB | 60 writers, 3,432 lines. |
| ICDAR-2013 online text | same page | 12 MB | |
| OLHWG release `all_datas.npy` | HuggingFace `Immortalman12/OLHWD` | 1,676,792,545 B (1.60 GB) | **HTTP 200**, public. OLHWG-format line data derived from CASIA-OLHWDB 2.0–2.2; card declares MIT, but the underlying CASIA terms still apply — a third party's label cannot relicense the corpus. |
| Teklia/CASIA-HWDB2-line mirror | HuggingFace `Teklia/CASIA-HWDB2-line` | 1.36 GB (train 835 MB / val 209 / test 255) | **HTTP 200**. 52,160 rows, height 128, RGB JPEG. 70 lines fewer than the official 52,230 and recompressed: fine for plumbing, not for headline numbers without saying so. |
| TGC-Diff Chinese line data | https://drive.google.com/file/d/1IAyfVImGdAPMAufQJHpBSdr-Afa2AAO_/view | unverified size | Their own preprocessed Chinese line set; check size before use. |
| TGC-Diff template labels | https://drive.google.com/file/d/14uHggU_ztBvxHGevld9jF1MCKQcnIYkd/view | unverified size | Required input to `test_ch.py`. |
| DiffInk dataset | `datas/train_val_400_100/{train,val}.h5` inside the DiffInk folder above | several GB | Their preprocessed 400/100-writer split. |
| SDT datasets | https://drive.google.com/drive/folders/17Ju2chVwlNvoX7HCKrhJOqySK-Y-hU8K | unverified | Chinese/Japanese/English character sets. |

**CORRECTION (2026-11-22, verified on the supplier's own conditions page).** This document
previously said the CASIA corpora require a signed application. That is wrong for this data:
the NLPR conditions-of-use page states "From 2020, application form is not required for
downloading the datasets fro acedemic research", and every line-data URL answers HTTP 200
directly. The conditions that DO apply: academic research only (no commercial use), CASIA
retains copyright, and cite the ICDAR 2011 databases paper. Exact URLs, byte counts and the
download script are in `docs/DATASETS.md`.

The Hugging Face mirrors remain convenient but are third-party redistributions; the official
NLPR files are the primary source, and they are now the plan of record.

## 6. What was cleaned up

- partial `DiffBrush-ckpt.pt` (31,408,020 B) — deleted
- partial `tgc_diff/weight/weight.pt` (83,030,016 B) — deleted
- partial `tgc_diff/template/temp_weight.pt` (77,099,008 B) — deleted
- `.fixture/gdown_env` (7.8 MB, isolated pip target) — removed
- pip cache — purged (3.84 GB reclaimed)

No background download is still running (checked: no `curl` process).
