# DISHA-HTR Evaluation

Automated CER / WER / exact-match evaluation. TrOCR is **single-line**, so full handwritten pages are cropped into lines first.

## Layout

```
evaluation/
  images/                 # ORIGINAL uploaded pages (never modified by tools)
    non_cursive|cursive|medical_vocabulary|prescription_style/
  pages/originals/        # one-time archive copies (created if missing; never overwritten)
  lines/                  # generated single-line crops (*_L01.png …) + crops_manifest.csv
  ground_truth/
    transcriptions.csv         # optional page-level GT
    line_transcriptions.csv    # per-line GT (verified by human; never auto-filled)
  results/                # page-mode outputs
  results/lines/          # line-mode outputs
```

**Do not place real patient PHI in committed samples.** Prefer synthetic or de-identified lines.

## PowerShell — line-level handwriting eval (recommended)

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"

# 1) Crop lines from originals (does not alter evaluation\images\)
.\.venv\Scripts\python.exe scripts\prepare_line_crops.py

# 2) Fill verified per-line text in:
#    evaluation\ground_truth\line_transcriptions.csv
#    Leave blank to exclude that crop from accuracy metrics.
#    Tools never invent ground_truth.

# 3) Run line evaluation (loads TrOCR once; reuses recognize_image + metrics)
.\.venv\Scripts\python.exe scripts\run_evaluation.py --mode lines
```

Outputs:

- `evaluation\results\lines\predictions.csv` (includes `source_page`)
- `evaluation\results\lines\metrics.json`
- `evaluation\results\lines\evaluation_report.txt`

## Page-mode (legacy / whole-image)

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py --template-only
.\.venv\Scripts\python.exe scripts\run_evaluation.py --mode pages
```

## Lightweight unit tests (no GPU / no model load)

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"
.\.venv\Scripts\python.exe -m unittest tests.test_metrics_and_csv tests.test_line_crop_metadata -v
```

## Metric rules

- Missing or blank `ground_truth` → prediction may still be stored; **excluded** from CER/WER/exact-match.
- Missing GT is **never** treated as an empty-string reference.
- Failed images are recorded with `status=failed` and excluded from accuracy metrics.
- Category-wise metrics use the same exclusion rules.
- Line crops always record `source_page` pointing at the original page filename.
