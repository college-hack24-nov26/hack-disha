# DISHA-HTR Evaluation

Automated CER / WER / exact-match evaluation over category image folders.

## Layout

```
evaluation/
  images/
    non_cursive/
    cursive/
    medical_vocabulary/
    prescription_style/
  ground_truth/
    transcriptions.csv    # image_filename, category, ground_truth
  results/                # generated: predictions.csv, metrics.json, evaluation_report.txt
```

**Do not place real patient PHI in committed samples.** Prefer synthetic or de-identified lines.

## PowerShell — use the existing `.venv` (do not recreate)

From `DISHA-HTR/` (`D:\project Disha\hack-disha\DISHA-HTR`):

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"

# 1) Discover images + create/extend GT template only (no model load)
.\.venv\Scripts\python.exe scripts\run_evaluation.py --template-only

# 2) Edit evaluation\ground_truth\transcriptions.csv
#    Fill the ground_truth column for each image_filename.
#    Leave blank to exclude that image from accuracy metrics.
#    Existing non-blank transcriptions are never overwritten by the template step.

# 3) Full evaluation (loads TrOCR once, reuses recognize_image)
.\.venv\Scripts\python.exe scripts\run_evaluation.py

# Optional device / FP16 flags
.\.venv\Scripts\python.exe scripts\run_evaluation.py --device cuda
.\.venv\Scripts\python.exe scripts\run_evaluation.py --no-fp16
.\.venv\Scripts\python.exe scripts\run_evaluation.py --device cpu
```

Outputs (overwritten each full run):

- `evaluation\results\predictions.csv`
- `evaluation\results\metrics.json`
- `evaluation\results\evaluation_report.txt`

## Lightweight unit tests (no GPU / no model load)

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"
.\.venv\Scripts\python.exe -m unittest tests.test_metrics_and_csv -v
```

## Metric rules

- Missing or blank `ground_truth` → prediction may still be stored; **excluded** from CER/WER/exact-match.
- Missing GT is **never** treated as an empty-string reference.
- Failed images are recorded with `status=failed` and excluded from accuracy metrics.
- Category-wise metrics use the same exclusion rules.
