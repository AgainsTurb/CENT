# Datasets: availability, exact sizes, and how to get them

Audited 2026-11-22. Every link below was checked with an HTTP HEAD request from this machine
and returned **200 OK with `Content-Type: application/zip`** and the byte count shown — so these
are open downloads, not forms, not mirrors, not login walls.

## 0. The headline

**No application form is required.** The NLPR conditions-of-use page states it directly:

> "From 2020, application form is not required for downloading the datasets fro acedemic
> research. All data can be downloaded at the page Data Download."

The same page keeps three conditions, which we must honour: academic research only (**no
commercial use**), CASIA retains copyright, and publications must cite
Liu, Yin, Wang, Wang, *CASIA online and offline Chinese handwriting databases*, ICDAR 2011.

Earlier notes in this repo said a signed agreement was required. That is **wrong for this
data since 2020**, and the correction is recorded in `docs/WEIGHTS_AND_DATA.md`.

## 1. Essential: the line corpora (2.62 GB total)

### Offline lines, `.dgrl` — CASIA-HWDB2.0–2.2

| File | Bytes | URL |
|---|---|---|
| HWDB2.0Train | 726,855,826 | `https://nlpr.ia.ac.cn/databases/Download/Offline/TextlineData/HWDB2.0Train.zip` |
| HWDB2.0Test | 179,749,084 | `.../Offline/TextlineData/HWDB2.0Test.zip` |
| HWDB2.1Train | 553,925,763 | `.../Offline/TextlineData/HWDB2.1Train.zip` |
| HWDB2.1Test | 139,912,874 | `.../Offline/TextlineData/HWDB2.1Test.zip` |
| HWDB2.2Train | 527,868,892 | `.../Offline/TextlineData/HWDB2.2Train.zip` |
| HWDB2.2Test | 131,544,385 | `.../Offline/TextlineData/HWDB2.2Test.zip` |
| **subtotal** | **2,259,856,824 (2.26 GB)** | 5,091 pages · 52,230 lines · 1,019 writers |

### Online lines, `.wptt` — CASIA-OLHWDB2.0–2.2

| File | Bytes | URL |
|---|---|---|
| WPTT2.0-Train | 63,653,107 | `https://nlpr.ia.ac.cn/databases/Download/Online/WPTTData/WPTT2.0-Train.zip` |
| WPTT2.0-Test | 15,998,990 | `.../Online/WPTTData/WPTT2.0-Test.zip` |
| WPTT2.1-Train | 51,470,049 | `.../Online/WPTTData/WPTT2.1-Train.zip` |
| WPTT2.1-Test | 12,947,463 | `.../Online/WPTTData/WPTT2.1-Test.zip` |
| WPTT2.2-Train | 45,522,785 | `.../Online/WPTTData/WPTT2.2-Train.zip` |
| WPTT2.2-Test | 11,237,051 | `.../Online/WPTTData/WPTT2.2-Test.zip` |
| **subtotal** | **200,829,445 (0.20 GB)** | 5,092 pages · 52,220 lines |

### External test set — ICDAR-2013 text

| File | Bytes | URL |
|---|---|---|
| Offline text (DGRL) | 143,515,530 | `https://nlpr.ia.ac.cn/databases/Download/Offline/TextlineData/Competition13Line.zip` |
| Online text (WPTT) | 13,297,449 | `https://nlpr.ia.ac.cn/databases/Download/Online/WPTTData/Competition13wptt.zip` |
| **subtotal** | **156,812,979 (0.16 GB)** | 60 writers · 3,432 lines each modality |

**Essential total: 2,617,499,248 B = 2.62 GB** (2.44 GiB). Unpacked, expect roughly 3 GB —
the archives hold bitmaps and coordinate tables that barely compress. That unpacked figure is an
estimate, not a measurement.

### Plus the recogniser (65 MB)

```
https://huggingface.co/Teklia/pylaia-casia-hwdb2        MIT, PyLaia, zh
files: model/, weights.ckpt, syms.txt, tokens.txt, lexicon.txt, language_model.arpa.gz
```
Needed for AR/CR, D_CER/D_WER and every baseline's content column. See
`benchmark/recogniser.py` for the wiring and the calibration protocol.

## 2. Download script

```bash
#!/usr/bin/env bash
set -euo pipefail
B=https://nlpr.ia.ac.cn/databases/Download
mkdir -p data/raw/dgrl data/raw/wptt data/raw/icdar13
cd data/raw

# offline lines  (693 + 171 + 528 + 133 + 503 + 125 MiB)
for f in HWDB2.0Train HWDB2.0Test HWDB2.1Train HWDB2.1Test HWDB2.2Train HWDB2.2Test; do
  curl -fL -C - -o "$f.zip" "$B/Offline/TextlineData/$f.zip"
done
# online lines
for f in WPTT2.0-Train WPTT2.0-Test WPTT2.1-Train WPTT2.1-Test WPTT2.2-Train WPTT2.2-Test; do
  curl -fL -C - -o "$f.zip" "$B/Online/WPTTData/$f.zip"
done
# ICDAR-2013 text
curl -fL -C - -o Competition13Line.zip  "$B/Offline/TextlineData/Competition13Line.zip"
curl -fL -C - -o Competition13wptt.zip  "$B/Online/WPTTData/Competition13wptt.zip"

# the recogniser
huggingface-cli download Teklia/pylaia-casia-hwdb2 --local-dir htr_casia_hwdb2
```

`-C -` resumes, so an interrupted transfer is not a restart. Then:

```bash
unzip -q 'HWDB2.*.zip' -d dgrl/ ; unzip -q Competition13Line.zip -d icdar13/
unzip -q 'WPTT2.*.zip' -d wptt/ ; unzip -q Competition13wptt.zip -d icdar13/
```

## 3. Verify before building

Sizes first (they catch a truncated download far faster than a parse failure):

| check | expected |
|---|---|
| `dgrl/*.dgrl` pages | 5,091 |
| `wptt/*.wptt` pages | 5,092 |
| lines parsed offline | 52,230 |
| lines parsed online | 52,220 |
| ICDAR-2013 lines | 3,432 per modality |
| writers | 1,019 (1,020 numbered; one writer's pages are missing) |

Then the format probe, which is the only real proof the bytes are what we think:

```bash
python -m common.probe_lines dir data/raw/dgrl --kind dgrl --limit 10
python -m common.probe_lines dir data/raw/wptt --kind wptt
```

The parser now **trusts the declared header size** and repositions past undocumented bytes
(P1 in the training plan), so a header that is larger than the documented fields no longer
rejects the file. If a probe still fails, the failure names the file and the offset — record
it, do not hand-patch.

## 4. Optional alternatives (download only if a reason appears)

| Source | Bytes | When it is worth having |
|---|---|---|
| `Teklia/CASIA-HWDB2-line` (HF) | 1.36 GB (train 876 MB) | No agreement needed, but 52,160 rows (70 fewer than official), height **128** RGB JPEG — recompressed. Good for a plumbing run; not for headline numbers without saying so. |
| `Immortalman12/OLHWD` `all_datas.npy` (HF) | 1.60 GB | OLHWG-format preprocessed lines. Useful to sanity-check our `.wptt` parse against an independent conversion. Their MIT card cannot relicense CASIA data. |
| TGC-Diff `temp_data.zip` | 23,656,446 | Their real template labels. Cheap, and better than our synthesised templates for TGC-Diff fidelity. Recommended. |
| TGC-Diff `chinese_textline.zip` | 5,995,618,827 (6.0 GB) | Their preprocessed pixel data. **Not needed**: our adapter builds `data_ch`/`data_temp_ch` from canonical lines. |
| DiffInk `train.h5` + `val.h5` | 1,570,401,156 + 80,310,344 (1.65 GB) | Their exact 400/100 split. **Not needed**: our adapter writes their HDF5 contract from canonical lines. |

## 5. Not needed at all (saves 7.7 GB)

| Artefact | Why we can skip it |
|---|---|
| SDT Chinese/Japanese/English **datasets** | SDT is a composed baseline: it needs only its **weights** plus style references and content images taken from *our* data. |
| MetaScript **dataset** | Same: composed baseline, weights only. If its Baidu checkpoint cannot be obtained and we retrain, the recipe uses CASIA-HWDB**1.1** characters (Gnt1.1Train 897+943 MB), which is a different download decision. |
| One-DM data | Chinese-impossible baseline, cite-only. |
| DiffBrush training data | It trains on the same CASIA-HWDB2 lines we are already downloading. |

## 6. Terms (must appear in the paper)

- **Academic research only**; commercial use is separately licensed and paid.
- CASIA (Institute of Automation, CAS) **retains copyright**; we do not redistribute the corpus.
- Cite: C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, *CASIA online and offline Chinese handwriting
  databases*, ICDAR 2011, pp. 37–41. The paper's data statement already carries this.
- ICDAR-2013 test data: competition data, released for research; same acknowledgement practice.

## 7. What was verified, and what was not

**Verified here:** every URL above answered 200 with the stated byte count and a zip content
type; the conditions-of-use page text quoted in §0; the Teklia model's identity, licence and
size via the Hub API; the HF mirrors via the Hub API.

**Not verified:** that the archives unpack to the advertised internal structure, and that the
page counts match after extraction — that needs the actual download. The verification table in
§3 is what the GPU box must satisfy before the canonical build runs.
