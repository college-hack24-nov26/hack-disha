# Samples

Synthetic demo images only. Do not place real patient handwriting or PHI here.

## Existing baseline sample

`synthetic_line.png` — synthetic Segoe Print glyphs (not real handwriting).

Create/refresh it:

```powershell
..\.venv\Scripts\python.exe ..\scripts\make_sample_image.py
```

## Adding a real handwriting sample (for quality evaluation)

Place **one clear single-line** scan as e.g. `samples/handwritten_line.png`:

- Dark ink on light paper, decent contrast
- Cropped tightly around the line
- No PHI in anything that will be committed

Then:

```powershell
..\.venv\Scripts\python.exe ..\scripts\test_recognize.py --image samples\handwritten_line.png
```

If no image exists under `samples/`, the runner prints these instructions and exits without inventing text.
