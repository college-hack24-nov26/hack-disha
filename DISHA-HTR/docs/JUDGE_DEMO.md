# DISHA-HTR testing workspace (standalone)

**Audience: clinicians / developers.** This workspace is for **testing and improving**
handwriting recognition and digital prescription capture. It is not the patient-facing
DISHA application.

## Launch

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"
.\.venv\Scripts\python.exe scripts\run_judge_demo_server.py --port 8766 --preload-model
```

`--preload-model` loads TrOCR at startup so the first recognize call is fast.

- Laptop: http://127.0.0.1:8766/
- Phone (same Wi‑Fi): http://&lt;laptop-LAN-IP&gt;:8766/
- Digital pad alone: http://127.0.0.1:8766/pad/

## Checkpoint policy

Uses **`checkpoints/trocr-small-handwritten`** (base) — best measured held-out
character accuracy on the identical 8-image set (**14.01%**).

Does **not** overwrite original checkpoints. Treat AI text as a draft to improve against.

## Local storage

`judge_demo_data/` — local testing only, not the production DISHA database.

## Tabs

| Tab | Purpose |
|---|---|
| Overview | Testing goals and current held-out accuracy |
| Handwriting recognition | Upload image → draft transcription |
| Prescription pad | Mouse / finger writing; PNG + stroke JSON |
| Paper archive | Save original image unchanged |
| Saved records | List + open local testing records |

Doctor-specific personalization is not enabled in this testing build until that module is started separately.

### Digital pad notes

- Header (hospital/clinician) and patient footer are independent of the canvas.
- Optional clinical draft fields never rewrite stroke coordinates or the handwriting PNG.
- Save is append-only. Reopen restores strokes from stroke JSON when present.
