# Cursor AI Integration Audit — DISHA-HTR

**Scope:** Read-only inspection of `D:\project Disha\hack-disha\DISHA-HTR` (Cursor-built standalone TrOCR/HTR).  
**Date:** 2026-11-03  
**Rules followed:** No code/config/deps/env/Git changes; did not modify `DISHA_AI_INTEGRATION_HANDOVER.md`; no installs, Hub downloads, or model reloads.  
**Evidence basis:** Source under `src/`, `scripts/`, `config/`, `requirements*.txt`, `README.md`, existing logs under `logs/`, on-disk samples/checkpoints listing.  
**Legend:** **implemented** | **partially implemented** | **missing** | **unverified**

---

## 1. Architecture and dependencies

| Finding | Path / symbol | Status | Evidence vs assumption |
| --- | --- | --- | --- |
| Standalone Python package `disha_htr` (v0.1.1), inference-only | `src/disha_htr/__init__.py` | **implemented** | Code present; exports `detect_hardware`, `load_trocr`, `recognize_image`, `run_baseline_inference`, `find_sample_image` |
| Module layout: hardware → load → preprocess → recognize → baseline API | `hardware.py`, `model_loader.py`, `preprocess.py`, `recognize.py`, `inference.py`, `logging_config.py` | **implemented** | All six modules exist |
| Config file preserved | `config/default.yaml` | **partially implemented** | File defines model/hardware/recognition/logging; **no Python reader** loads it (grep of `src/` shows no YAML load) — CLI/API use hardcoded defaults |
| Declared deps | `requirements.txt`, `requirements-lock.txt` | **implemented** | torch, transformers, Pillow, opencv-python-headless, numpy, PyYAML, sentencepiece, protobuf, tiktoken |
| FastAPI / HTTP / Node deps | `requirements*.txt` | **missing** | No fastapi/uvicorn/flask; no `package.json` in this tree |
| Training / adaptation stack | `requirements*.txt` | **missing** | No datasets/accelerate/peft/etc. |
| Local venv + CUDA torch | `.venv/`, lock: `torch==2.11.0+cu128` | **implemented** (on this machine) | Lockfile + prior log evidence; **not re-probed** this audit |
| Local checkpoint snapshot | `checkpoints/trocr-small-handwritten/` (`model.safetensors`, configs, tokenizer) | **implemented** | Directory listing confirms weights present |
| Synthetic sample | `samples/synthetic_line.png` | **implemented** | Present (~11 KB); README states Segoe Print demo glyphs, not real handwriting |
| Isolation from MERN app | `README.md` | **implemented** (by design) | Docs state no modification of DISHA MERN app |

**Architecture summary:** Library + CLI smoke runner only. Ready-for-service shape is documented intent, not shipped.

---

## 2. TrOCR load / inference

| Finding | Path / symbol | Status | Evidence vs assumption |
| --- | --- | --- | --- |
| Hub/local load of `microsoft/trocr-small-handwritten` | `model_loader.py` → `load_trocr`, `_has_local_weights`, `LoadedModel` | **implemented** | Prefers populated `local_dir`; else Hub + optional save back to checkpoint |
| FP16 on CUDA (capability ≥ 7.0) with float32 fallback | `model_loader.py` → `_fp16_supported`, `load_trocr`; `recognize.py` → `torch.autocast` | **implemented** | Code path present; baseline log shows `dtype=float16`, `use_fp16=true` |
| Device selection | `hardware.py` → `resolve_device`, `detect_hardware`, `HardwareInfo` | **implemented** | auto/cuda/cpu; nvidia-smi best-effort |
| Generate + decode | `recognize.py` → `recognize_image` | **implemented** | `model.generate(..., max_new_tokens, num_beams)` + `batch_decode` |
| Timings / GPU memory metrics | `RecognitionResult`, `_gpu_memory_mb` | **implemented** | load_time, inference_time, allocated/peak MB |
| YAML knobs (`max_new_tokens`, `num_beams`, `use_fp16`) wired | `config/default.yaml` vs callers | **partially implemented** | Defaults duplicated in `run_baseline_inference` / CLI; config file unused |
| Encoder pooler MISSING keys on Hub load | `logs/smoke_test_stdout.txt` | **unverified** impact | Transformers load report noted missing `encoder.pooler.dense.*`; inference still completed — accuracy impact not measured |

**Test evidence (not re-run):** `logs/baseline_inference_stdout.txt` (2026-11-03 16:25) — local checkpoint, CUDA FP16, load ~4.27 s, infer ~3.21 s, peak ~140 MB, text `'retirement federal intervention .'`.

**Historical failures (resolved later same day):** `logs/disha_htr.log` — tokenizer path failed without/with broken tiktoken/sentencepiece handling before successful Hub load (~16:17).

---

## 3. Image preprocessing and HTR

| Finding | Path / symbol | Status | Evidence vs assumption |
| --- | --- | --- | --- |
| Image load (OpenCV → RGB PIL, Pillow fallback) | `preprocess.py` → `load_image` | **implemented** | Code |
| Mild CLAHE if very low contrast (`std < 15`) | `preprocess.py` → `prepare_for_trocr` | **implemented** | Code; synthetic sample unlikely to trigger (assumption) |
| Deskew / binarize / denoise / crop heuristics | `preprocess.py` | **missing** | Docstring: “No aggressive deskew/line-seg yet” |
| Single-line HTR end-to-end | `recognize.py` → `recognize_image`; `inference.py` → `run_baseline_inference` | **implemented** | Smoke/baseline logs |
| Real handwriting HTR quality | samples + metrics | **unverified** | Only synthetic printed glyphs exercised |

---

## 4. Full-page / line segmentation / reading order

| Finding | Path / symbol | Status | Evidence vs assumption |
| --- | --- | --- | --- |
| Full-page OCR pipeline | — | **missing** | No modules |
| Line / word segmentation | `preprocess.py` | **missing** | Explicitly deferred |
| Reading-order / layout | — | **missing** | No code |
| Multi-line prescription handling | — | **missing** | Sample contract is single-line only (`inference.py` `SAMPLE_INSTRUCTIONS`, `samples/README.md`) |

---

## 5. FastAPI / API and Node.js integration

| Finding | Path / symbol | Status | Evidence vs assumption |
| --- | --- | --- | --- |
| FastAPI (or any HTTP) service | repo tree / deps | **missing** | README: “later FastAPI”; no app module |
| REST contract / OpenAPI | — | **missing** | — |
| CLI / library API | `scripts/test_recognize.py`; `disha_htr` exports | **implemented** | Runnable baseline runner + importable API |
| Node/Express bridge, IPC, or client SDK | under `DISHA-HTR/` | **missing** | No JS/TS sources |
| Integration into `dishanew` MERN | sibling app | **unverified** this pass | Prior handover claimed no HTR refs; path `D:\project Disha\dishanew` **not found** in this environment — not re-audited |

---

## 6. Doctor correction / approval / adaptation

| Finding | Path / symbol | Status | Evidence vs assumption |
| --- | --- | --- | --- |
| Correction UI / API | — | **missing** | — |
| Approval / human-in-the-loop before record write | — | **missing** | — |
| Feedback store / audit of edits | — | **missing** | — |
| Doctor-specific fine-tuning / personalization | — | **missing** | Out of stated v0.1 scope (`README.md`) |
| Spec categories for doctor training | external screenshot cited in handover | **unverified** | Not re-opened this audit; no code implements them |

---

## 7. Evaluation metrics and actual test evidence

### What is measured in code

| Metric | Where | Status |
| --- | --- | --- |
| Hardware snapshot | `hardware.detect_hardware` → `HardwareInfo.to_dict` | **implemented** |
| Model load time (s) | `LoadedModel.load_time_s` → `RecognitionResult.model_load_time_s` | **implemented** |
| Inference time (s) | `recognize_image` | **implemented** |
| GPU allocated / peak MB | `_gpu_memory_mb` | **implemented** |
| Raw decoded text | `RecognitionResult.text` | **implemented** |
| CER / WER / edit distance / accuracy vs GT | — | **missing** |
| Golden/real handwriting eval set | `samples/` | **missing** (only synthetic demo) |
| Automated unit/integration test suite (pytest etc.) | — | **missing** |

### Logged test evidence (on disk; not re-executed)

| Run | File | Result |
| --- | --- | --- |
| Smoke (Hub download + first recognize) | `logs/smoke_test_stdout.txt` | Success; text `retirement federal intervention .`; CUDA |
| Baseline metrics (local ckpt + FP16) | `logs/baseline_inference_stdout.txt` | Success; load 4.2662 s; infer 3.213 s; peak 139.83 MB; same text |
| Application log | `logs/disha_htr.log` | Earlier tokenizer failures; later success matching above |

**Interpretation (evidence-backed):** Pipeline runs on CUDA with FP16. Output ≠ intended synthetic phrase `"Patient feels better today"` (`scripts/make_sample_image.py`). That mismatch is **not** CER/WER proof and does **not** establish clinical HTR quality — only that inference returned a string.

---

## 8. Missing / incomplete / broken

### Missing (product gaps)

1. FastAPI/HTTP service and auth  
2. Node/Express ↔ Python bridge  
3. Page/line segmentation and reading order  
4. Doctor correction, approval, adaptation/fine-tune  
5. CER/WER (or any GT-based) evaluation + real handwriting samples (no PHI)  
6. Config loader for `config/default.yaml`  
7. Packaging (`pyproject.toml` / installable entry points) — **missing** (path injection via `sys.path` in `test_recognize.py` only)  
8. Docker/Compose / Render service definitions for HTR — **missing** in this folder  

### Incomplete / brittle

| Item | Path / note | Status |
| --- | --- | --- |
| YAML config unused | `config/default.yaml` | **partially implemented** |
| Logging “rotating” claimed vs plain `FileHandler` | `logging_config.setup_logging` docstring vs implementation | **partially implemented** |
| Windows HF cache symlinks warning | smoke log | Environment limitation; not a code bug |
| Tokenizer dependency sensitivity (sentencepiece/protobuf/tiktoken) | early `disha_htr.log` failures | **partially implemented** mitigation via deps in `requirements.txt`; fragile on incomplete installs |

### Broken / failed (historical, later recovered)

- Pre-fix Hub loads failed on tiktoken/sentencepiece path (`logs/disha_htr.log` ~16:03–16:05).  
- Subsequent Hub load + local checkpoint reuse **succeeded** (`smoke_test_stdout`, `baseline_inference_stdout`). Current tree appears runnable from logs; **runtime not re-verified** this audit (**unverified** live).

### Non-goals of current Cursor baseline (documented)

- Training/fine-tuning  
- PHI samples  
- MERN app edits  

---

## Inventory of inspected artifacts

```
DISHA-HTR/
  README.md
  DISHA_AI_INTEGRATION_HANDOVER.md   # prior audit; not modified
  requirements.txt, requirements-lock.txt
  config/default.yaml
  src/disha_htr/{__init__,hardware,model_loader,preprocess,recognize,inference,logging_config}.py
  scripts/{test_recognize,make_sample_image}.py
  samples/{README.md,synthetic_line.png}
  checkpoints/trocr-small-handwritten/   # weights present on disk
  logs/{baseline_inference_stdout,smoke_test_stdout,disha_htr,pptx_extract}.txt
```

---

## Verdict

Cursor work in this folder delivers a **working single-line TrOCR baseline library + CLI** with CUDA/FP16 and timing metrics (**implemented**, with log evidence). Product integration layers — HTTP API, Node bridge, segmentation, doctor HITL/adaptation, and real accuracy evaluation — are **missing**. Config YAML and packaging remain **partially implemented**. Stop here pending next authorized milestone.
