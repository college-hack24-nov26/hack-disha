# DISHA HTR (standalone)

Independent handwriting text recognition (HTR) module for DISHA healthcare experiments.

This is a **standalone Python project**. It does **not** modify the DISHA MERN application.

## Scope (v0.1)

- Inference only (no training)
- Initial model: `microsoft/trocr-small-handwritten`
- Single-line recognition smoke test
- GPU when CUDA is available; automatic CPU fallback
- Modular layout ready for a later FastAPI service

## Verified environment (this workspace)

Captured during initial setup — do not assume these values elsewhere without re-checking:

| Check | Result |
| --- | --- |
| Python | 3.14.8 (`py -3.14`) |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| VRAM | 6144 MiB |
| Driver | 581.86 |
| nvidia-smi CUDA Version | 13.0 (driver capability) |
| PyTorch | 2.11.0+cu128 |
| `torch.cuda.is_available()` | **True** |

## Project layout

```
DISHA-HTR/
  config/default.yaml          # preserved defaults
  checkpoints/                 # local model snapshots (gitignored weights)
  samples/                     # synthetic demo images (no real PHI)
  scripts/
    make_sample_image.py       # create a synthetic line image
    test_recognize.py          # end-to-end smoke test
  src/disha_htr/
    hardware.py                # GPU/CUDA detection + device selection
    model_loader.py            # TrOCR load / local checkpoint reuse
    preprocess.py              # image load + light prep
    recognize.py               # single-line inference
    logging_config.py
  requirements.txt
  .venv/                       # local virtualenv (not committed)
```

## Setup

```powershell
cd "DISHA-HTR"
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If CUDA install fails, install the CPU wheel and re-run; `hardware.resolve_device` will fall back automatically.

## Smoke test

```powershell
cd "DISHA-HTR"
.\.venv\Scripts\python.exe scripts\make_sample_image.py
.\.venv\Scripts\python.exe scripts\test_recognize.py
```

First Hub download of `microsoft/trocr-small-handwritten` is cached under `.cache/huggingface` and also saved into `checkpoints/trocr-small-handwritten` when possible.

## Smoke test — actual result (2026-11-03)

Synthetic image text: `Patient feels better today` (Segoe Print / demo glyphs — not real handwriting, not PHI).

```json
{
  "text": "retirement federal intervention .",
  "device": "cuda",
  "model_name": "microsoft/trocr-small-handwritten"
}
```

No accuracy claim: TrOCR-handwritten on synthetic font text is expected to be unreliable. Use real handwriting scans for quality evaluation later.

### Tokenizer deps

TrOCR needs `sentencepiece` **and** `protobuf`. Without them, transformers 5.x falls back to a tiktoken path that fails on `sentencepiece.bpe.model`.

## Notes

- No real patient information — samples are synthetic.
- Do not invent accuracy metrics; report only measured outputs.
- Checkpoints and `config/default.yaml` are meant to be preserved across iterations.
- FastAPI wrapping will sit on top of `recognize_image` / `load_trocr` later.
