# DISHA-HTR Judge Demo

Offline handwriting demo using a **local** TrOCR checkpoint (prefers `checkpoints/trocr-demo-finetuned` when present).

## Launch (UI)

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"
.\.venv\Scripts\python.exe scripts\run_demo.py
```

- Gallery locked to `demo/approved/` (**13** single-line images)
- Shows image + transcription + inference time
- **No ground-truth / scores in the UI**
- Offline after setup; uses `.venv` + local weights

## Private metrics (not shown in UI)

```powershell
# Held-out generalization (images the fine-tune did NOT train on)
.\.venv\Scripts\python.exe scripts\run_demo_eval.py --label heldout

# Adaptation check on gallery lines (seen during tiny demo fine-tune — NOT a held-out claim)
.\.venv\Scripts\python.exe scripts\run_demo_eval.py --label adaptation --approved-dir demo\approved --ground-truth demo\eval_private\adaptation_check_gt.csv
```

## Honest accuracy notes

| Set | Meaning | 80% char accuracy? |
| --- | --- | --- |
| `adaptation` | Gallery lines used in short demo fine-tune | Measured **Yes (~94.5%)** — demo quality only |
| `heldout` | Separate verified lines never trained on | Measured **No (~7.6%)** |

Do **not** present held-out as 80%. Original pages under `evaluation/images/` are never modified.
