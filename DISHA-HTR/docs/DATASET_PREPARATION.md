# DISHA-HTR Dataset Preparation (Phase 1 — IAM)

Automated collection/preparation of the **IAM Handwriting Database** for future English HTR experiments.

**This phase does not train or fine-tune.** Baseline inference modules are unchanged.

## Environment notes (this machine)

| Check | Value |
| --- | --- |
| Project | `D:\project Disha\hack-disha\DISHA-HTR` |
| Python venv | `.venv` (do **not** recreate) |
| GPU | NVIDIA GeForce RTX 3050 6GB (available; **not** used in Phase 1) |
| Extra deps needed? | No — uses stdlib + existing `requests` |

## Licensing (must read)

1. Register at the official IAM page:  
   https://fki.tic.heia-fr.ch/databases/iam-handwriting-database  
2. Terms: free for **non-commercial research**; cite Marti & Bunke (IJDAR 2002) if publishing.  
3. Scripts **will not** bypass login. Provide credentials via env vars.  
4. Do **not** commit `datasets/iam/raw/` or extracted images.

Aachen form-id split lists (OpenSLR SLR56) are public metadata only:  
https://www.openslr.org/56/

## Directory layout

```
datasets/iam/
  raw/                 # archives (gitignored content)
  extracted/           # derived extract trees
  processed/           # reserved
  manifests/           # train.csv val.csv test.csv stats.json
training/              # placeholder (no training in Phase 1)
docs/DATASET_PREPARATION.md
scripts/download_iam.py
scripts/prepare_iam_dataset.py
src/disha_htr/iam_dataset.py
tests/test_iam_dataset_prep.py
```

## Manifest CSV columns

```
image_path,text,writer_id,line_id,form_id,split
```

- `image_path` — relative to extracted lines root (e.g. `a01/a01-000u/a01-000u-00.png`)
- `text` — verified IAM ascii transcription (`|` → spaces); **never invented**
- `writer_id` — from `forms.txt`
- Writer-independent schemes: `official` (FKI task) or `aachen` (OpenSLR)

## PowerShell commands (existing `.venv`)

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"

# 0) CPU unit tests (no network / no GPU)
.\.venv\Scripts\python.exe -m unittest tests.test_iam_dataset_prep -v

# 1) Public Aachen split lists only (no IAM password)
.\.venv\Scripts\python.exe scripts\download_iam.py --aachen-only

# 2) Official IAM archives (after FKI registration)
$env:IAM_USERNAME = "your_fki_username"
$env:IAM_PASSWORD = "your_fki_password"
.\.venv\Scripts\python.exe scripts\download_iam.py

# 3) Extract + build manifests (Aachen scheme default)
.\.venv\Scripts\python.exe scripts\prepare_iam_dataset.py --split-scheme aachen

# Official FKI large writer-independent task lists:
.\.venv\Scripts\python.exe scripts\prepare_iam_dataset.py --split-scheme official

# Ascii-only dry run if lines.tgz not yet available:
.\.venv\Scripts\python.exe scripts\prepare_iam_dataset.py --allow-missing-images
```

## Idempotence

- Downloads skip existing archives when size matches (`.partial` used for resume).  
- Extracts skip when a `.extract_ok_*` marker exists.  
- Manifest rebuild is deterministic (sorted line IDs).  
- Raw archives are never modified in place.

## What Phase 1 does **not** do

- No training / fine-tuning / weight downloads  
- No changes to `recognize.py` / `model_loader.py` / `inference.py`  
- No edits to `evaluation/` samples or ground-truth  
- No commits/pushes from this phase task  
- No PHI / clinical handwriting in IAM path
