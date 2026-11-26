# DOLPHIN's three subsets — do they suit CENT?

**Question.** `SCUT-DLVCLab/DOLPHIN` distributes CASIA-OLHWDB2, DCOH-E and SCUT-COUCH2009 as direct downloads. Can CENT use them instead of the password-locked SCUT-COUCH release?

**Short answer.** **No — not for CENT's core claims.** The pipeline is built for *writer retrieval*, and in converting the corpora it discards exactly the two things CENT needs: **where the character boundaries are** and **what characters they are**. They are still useful for one secondary purpose: pretraining the trajectory VAE and the writer-style encoder on real Chinese lines, unencrypted and with 1,731 writer labels.

Evidence below is from the repository's own code (fetched and read, not assumed).

---

## 1. What the repo actually ships

From `README.md` and the Google Drive / Baidu folders: three zips, `OLHWDB2.zip`, `DCOH-E.zip`, `COUCH09.zip`, which unzip into per-writer folders:

```
data-raw
├── COUCH09/   001/ ...          # writer folders, one text file per sample
├── DCOH-E/    dcoh-e313/ ...
└── OLHWDB2/   001/ ...
```

Code is GPLv3. Note the redistributor is **SCUT-DLVCLab** — the same lab as SCUT-COUCH's authors (Jin Lianwen), so this is plausibly an authorised repackaging rather than a stray mirror. The underlying corpus terms (CASIA research agreement, DCOH CC BY-NC-ND, COUCH research letter) still apply.

## 2. What the preprocessing keeps — and throws away

`preprocess.py`, for all three subsets, does the same four things per sample file:

```python
lines = f.readlines()[1:]                          # drop the header line
info  = [list(map(float, l.split()[:3])) for l in lines]   # keep 3 columns per point
info  = centernorm_size(info)                      # per-sample centre + scale normalise (x,y)
info  = interpolate_torch(info, interp_ratio=4)    # bicubic resample ALL 3 channels x4
```

Then `divide.py` merges by writer key (dropping any writer with ≤20 samples) and `dataset.py` consumes only:

```python
return feature, len(feature), self.user_labels[idx]     # (points, length, writer id)
```

**Consequences, in order of importance to CENT:**

| CENT needs | Kept? | Why |
|---|---|---|
| Pen trajectory (x, y) | **yes** | columns 0–1 |
| Stroke segmentation (pen lifts) | **no** | points are flattened to one sequence per sample; no separators survive |
| **Character boundaries** | **no** | never in the 3 columns; `.SDX`/`.SGM`-style annotation is not used anywhere in the repo |
| **Transcript / text content** | **no** | never read; the header line is explicitly discarded (`readlines()[1:]`) |
| Writer identity | **yes** | folder name → dict key; this is the whole point of the repo |
| Join / lift labels | **no** | needs the two rows above |
| Timestamps (DCOH) | **no** | DCOH's 4th column is truncated by `[:3]` |
| Absolute position / size (style) | **partly** | `centernorm_size` centres and rescales each sample, so a writer's size and placement habits are normalised away |

The bicubic interpolation is also actively hostile to any discrete column: if the third column were a pen-state flag, `F.interpolate(..., mode='bicubic')` would smear it into fractions.

## 3. Subset-by-subset verdict

| Subset | Contents (per the paper + code) | Usable for CENT's **core** work? | Usable for pretraining? |
|---|---|---|---|
| **CASIA-OLHWDB2** | 52,220 line trajectories, 1,019 writers, x,y only | **No** — no char boundaries, no transcript | **Yes** — best of the three for line geometry + many writers |
| **DCOH-E** | 87,759 lines, 567 writers | **No** — and its one distinguishing asset (timestamps) is truncated by `[:3]` | Yes, similar to above |
| **COUCH09** | 145 writers × 1,000 words (2–5 characters each) | **No**, but it is the *closest*: every sample spans 2–5 boundaries, so the boundary structure is dense — just unlabelled | Yes, and the word length makes it a useful shape prior |

Ranking *for CENT*: **COUCH09 > CASIA-OLHWDB2 > DCOH-E**. That ordering is about sample geometry (words vs lines), not about supervision — none of them supervises the join.

## 4. The one thing that could change this — and how to check it in 60 seconds

Two facts are **not** determined by the code, and they decide whether these files are merely "pretraining-only" or actually usable:

1. **What is the third column?** `extract_features` treats it as *pressure* (`pressure = handwriting[:, 2]`), and the paper says CASIA and COUCH get `p = 1` added manually. But `interpolate_torch` is hard-coded to 3 channels, which implies the raw files do carry a third value. If that value is a **pen state (0/1)**, stroke segmentation is recoverable and the data becomes far more useful (though still without character boundaries).
2. **What is in the header line** (the one that `readlines()[1:]` throws away), and **do the filenames carry the transcription**? For COUCH09 the word identity must live *somewhere* — if it is the filename or the header, text conditioning is recoverable.

Decisive check, once you are willing to download one zip:

```bash
head -3 data-raw/COUCH09/001/<some_file>.txt      # header line?
awk 'NR>1{print NF}' data-raw/COUCH09/001/<file>.txt | sort -u   # column count: 2? 3? 4?
awk 'NR>1{print $3}' data-raw/COUCH09/001/<file>.txt | sort -u   # binary => pen state; continuous => pressure
ls data-raw/COUCH09/001/ | head                  # do filenames look like words/GB codes?
```

If column 3 is binary **and** the transcript is in the filename or header, then COUCH09 becomes usable for join supervision (stroke segmentation + character count from the word, with boundaries recoverable by proportional alignment). If not, it does not.

## 5. Recommendation

**Do not base CENT on DOLPHIN's copies.** For the boundary contribution you need per-character annotation, and these files do not have it.

Better routes, in order:

1. **Official CASIA-OLHWDB 2.x (WPTT)** — the NLPR links are live and unencrypted (`https://nlpr.ia.ac.cn/Download/Online/WPTTData/WPTT2.0-Train.zip`, 62 MB; `-Test.zip`, 15 MB; I confirmed HTTP 200). This format carries, per page, **the line transcript and the per-line stroke index list plus character codes** — i.e. exactly the character boundaries CENT needs, with no password. Same corpus content as DOLPHIN's `OLHWDB2.zip`, but *with* annotation. This is strictly better and it is the dataset every baseline is measured on anyway.
2. **Official SCUT-COUCH WORD8888** once the application is approved — the `.SDX/.SGM` segmentation is the only source where the per-character stroke assignment is *authoritative* (labelled by the authors).
3. **DOLPHIN's zips only if** you want a fast, unencrypted corpus for **VAE / style-encoder pretraining**, or if the 60-second check in §4 shows the third column is a pen state. In that case take **COUCH09** — it is the most boundary-dense.

## 6. What this changes in the plan

Nothing architectural. The model already expects per-character `char_ids`; CASIA WPTT supplies them directly, so the loader should be written for WPTT next (the format is documented in `cent/data/format_scut.py`'s docstring, which already describes the per-line stroke/char tables). The DOLPHIN route would only ever have fed the VAE a reconstruction objective.
