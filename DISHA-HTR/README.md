# DISHA HTR (standalone)

Independent handwriting text recognition (HTR) module for DISHA healthcare experiments.

This is a **standalone Python project**. It does **not** modify the DISHA MERN application.

## Scope (v0.1)

- Inference only (no training / fine-tuning)
- Initial model: `microsoft/trocr-small-handwritten`
- Single-line recognition baseline
- GPU when CUDA is available; automatic CPU fallback
- FP16 inference on CUDA when supported
- Modular layout ready for a later FastAPI service

## Verified environment (this workspace)

Captured during setup — do not assume these values elsewhere without re-checking:

| Check | Result |
| --- | --- |
| Python | 3.14.8 (`py -3.14`) |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| VRAM | 6144 MiB |
| Driver | 581.86 |
| nvidia-smi CUDA Version | 13.0 (driver capability) |
| PyTorch | 2.11.0+cu128 |
| `torch.cuda.is_available()` | **True** |
| transformers | 5.22.1 |
| Pillow | 13.0.0 |
| opencv-python-headless | 5.0.0 |

## Project layout

```
DISHA-HTR/
  config/default.yaml          # preserved defaults
  checkpoints/                 # local model snapshots (gitignored weights)
  samples/                     # synthetic demo images (no real PHI)
  scripts/
    make_sample_image.py       # create a synthetic line image
    test_recognize.py          # end-to-end baseline with metrics
  src/disha_htr/
    hardware.py                # GPU/CUDA detection + device selection
    model_loader.py            # TrOCR load / local checkpoint / FP16
    preprocess.py              # image load + light prep
    recognize.py               # single-line inference + timings
    inference.py               # reusable high-level baseline API
    logging_config.py
  requirements.txt
  .venv/                       # local virtualenv (not committed)
```

## Setup

Do **not** recreate the venv or reinstall PyTorch if they already work.

```powershell
cd "DISHA-HTR"
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If CUDA install fails, install the CPU wheel and re-run; `hardware.resolve_device` will fall back automatically.

### Tokenizer deps

TrOCR needs `sentencepiece` **and** `protobuf` (`import google.protobuf`). Without them, transformers 5.x falls back to a tiktoken path that fails on `sentencepiece.bpe.model`.

## How to run baseline inference

From the project root (`DISHA-HTR/`):

```powershell
# Optional: refresh synthetic sample (not real handwriting)
.\.venv\Scripts\python.exe scripts\make_sample_image.py

# Auto-discover samples/synthetic_line.png (or first image under samples/)
.\.venv\Scripts\python.exe scripts\test_recognize.py

# Explicit image + force/disable FP16
.\.venv\Scripts\python.exe scripts\test_recognize.py --image samples\synthetic_line.png
.\.venv\Scripts\python.exe scripts\test_recognize.py --no-fp16
.\.venv\Scripts\python.exe scripts\test_recognize.py --fp16
```

### Reusable Python API

```python
from disha_htr import run_baseline_inference

payload = run_baseline_inference()  # CUDA+FP16 when available
print(payload["recognition"]["text"])
print(payload["recognition"]["device"])
print(payload["recognition"]["model_load_time_s"])
print(payload["recognition"]["inference_time_s"])
print(payload["recognition"]["gpu_memory_peak_mb"])
```

Lower-level:

```python
from disha_htr import detect_hardware, load_trocr, recognize_image

hw = detect_hardware()
loaded = load_trocr(device=hw.selected_device)  # FP16 auto on CUDA
result = recognize_image("samples/synthetic_line.png", loaded)
```

### Sample image requirements

If `samples/` has no image, the runner exits with instructions (it does **not** invent text). For quality evaluation later, use a real single-line handwriting scan (no PHI in committed files). See `samples/README.md`.

## Smoke / baseline — measured results

See the latest run output from `scripts/test_recognize.py`. Synthetic image text is demo glyphs only — TrOCR-handwritten on printed/segoe text is expected to be unreliable. **No accuracy claim.**

## Category evaluation (CER / WER)

See `evaluation/README.md`. Uses the same `load_trocr` + `recognize_image` APIs (model loaded once). Fill `evaluation/ground_truth/transcriptions.csv` before treating metrics as meaningful.

## Notes

- No real patient information — samples are synthetic unless you add your own.
- Do not invent accuracy metrics; report only measured outputs.
- Checkpoints and `config/default.yaml` are meant to be preserved across iterations.
- FastAPI wrapping will sit on top of `run_baseline_inference` / `recognize_image` later.
- Never commit `.venv/`, model weight blobs, datasets, secrets, or temp caches.
