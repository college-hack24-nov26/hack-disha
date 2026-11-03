# DISHA — Digital Handwriting Prescription Pad (Testing)

Standalone testing pad: write a prescription with **mouse, touchpad, or finger**, and preserve the **original handwriting** as stroke JSON + PNG.

Not integrated into the patient-facing DISHA application. Does not modify TrOCR checkpoints or evaluation data.

## Run (recommended via testing server)

```powershell
.\.venv\Scripts\python.exe scripts\run_judge_demo_server.py --port 8766 --preload-model
```

Then open `http://127.0.0.1:8766/pad/`

Static-only (no recognition API):

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --directory prescription_pad
```

## Features

- Pointer Events (mouse / touch / future stylus)
- Pen colour + thickness
- Undo last stroke / clear with confirmation
- Page scroll locked while drawing
- Header / writing canvas / patient footer (synthetic testing data)
- Save → PNG + stroke JSON downloads + browser localStorage
- Local testing storage only — not the production DISHA database

## Stroke schema (stylus-ready later)

Each stroke: `id`, `color`, `width`, `pointerType`, timestamps, `points[{x,y,t,...}]`.  
`pressure` / tilt stored **only** when hardware reports them. Mouse does **not** claim stylus pressure.

## Legacy vs digital

| Kind | Preserved |
|---|---|
| Paper archive | Original uploaded image unchanged |
| Digital prescription | Stroke JSON + rendered PNG |

## Files

- `index.html`, `styles.css`, `app.js`
- `test_pad_smoke.py` — structural smoke checks
