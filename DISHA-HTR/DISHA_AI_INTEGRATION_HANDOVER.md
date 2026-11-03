# DISHA AI Integration Handover

**Document type:** Documentation / audit only (no code changes in this pass beyond creating this file).  
**Primary project:** `D:\project Disha\hack-disha\DISHA-HTR`  
**Audit date:** 2026-11-03  
**Auditor note:** Full text of any separate “audit brief (sections 1–14)” file was **not found** on disk in searchable project/AgentStore paths. This document uses **Parts 1–14** covering every topic that brief call-out required (code inspection, Track A MERN / Node↔Python, Master Spec milestone compare, baseline metrics, stop-and-wait). If a canonical brief with different Part titles exists offline, rename headings to match without changing factual content.

---

## Part 1 — Executive status (read this first)

| Item | Status |
| --- | --- |
| Standalone HTR module | **Present** under `DISHA-HTR/` |
| Base model | `microsoft/trocr-small-handwritten` (local checkpoint reuse) |
| GPU baseline inference | **Done** — CUDA + FP16 measured (see Part 8) |
| Fine-tuning / doctor-specific training | **Not started** (explicitly out of current baseline scope) |
| FastAPI / HTTP service wrapping HTR | **Not present** |
| Track A MERN ↔ Python bridge | **Not present** (no HTR/OCR references in reachable MERN app) |
| Master Spec file | **NOT INSPECTABLE** as a complete document (see Part 2) |
| Safe next step | Wire a thin inference HTTP/CLI contract from Node → existing `run_baseline_inference` / `recognize_image`; do **not** reinstall torch or retrain yet |

**Bottom line:** Track B (standalone TrOCR baseline) is established on this machine. Track A product integration and doctor-specific adaptation remain open. Do not invent recognition quality claims from the synthetic sample.

---

## Part 2 — Sources inspected

| Source | Path / note | Result |
| --- | --- | --- |
| DISHA-HTR code | `D:\project Disha\hack-disha\DISHA-HTR\src\disha_htr\*.py`, `scripts\`, `config\`, `README.md` | Inspected |
| DISHA-HTR requirements | `requirements.txt`, `requirements-lock.txt` | Inspected |
| Baseline inference report | `...\bc-6f1fae26-...\files\internal\htr\baseline-inference-report.md` | Inspected (preferred metrics source; **no model reload** for this audit) |
| Setup report | `...\internal\htr\setup-report.md` | Inspected |
| Parent git repo | `D:\project Disha\hack-disha` | Inspected (read-only git) |
| Track A MERN (reachable sibling) | `D:\project Disha\dishanew\` (`client/`, `server/`) | Inspected; **outside** `hack-disha` git root |
| `hack-disha` root README | Template only (`# TEMPLATE`) | No MERN code inside this repo |
| “DISHA Master Project Handover & Technical Specification” | Searched under `D:\project Disha`, AgentStores, common user docs | **NOT INSPECTABLE** as a full file |
| Spec fragment (screenshot) | `D:\project Disha\Screenshot 2026-11-03 150726.png` | Partial: doctor-specific training categories + model decision table |
| Hackathon deck | `D:\project Disha\DISHA_Hackathon_2k26.pptx` | Inspected (OCR timeline / TRL narrative) |
| Full numbered audit-brief file | Not located on disk | Structure inferred from assignment topics; labeled Parts 1–14 |

**Not done (per audit rules):** feature work, refactors, commits, pushes, dep reinstalls, Hub downloads, weight loads, training starts.

---

## Part 3 — Repository layout (as-built)

### Inside `hack-disha` (git)

```
hack-disha/
  README.md                 # template stub
  DISHA-HTR/                # sole real product work in this repo
    DISHA_AI_INTEGRATION_HANDOVER.md   # this file
    README.md
    requirements.txt
    requirements-lock.txt
    config/default.yaml
    src/disha_htr/
      __init__.py
      hardware.py
      model_loader.py
      preprocess.py
      recognize.py
      inference.py
      logging_config.py
    scripts/
      make_sample_image.py
      test_recognize.py
    samples/                # synthetic_line.png (+ README)
    checkpoints/            # local TrOCR snapshot (weights gitignored)
    .venv/                  # local only
    .cache/                 # HF cache (gitignored)
    logs/                   # gitignored
```

### Reachable Track A app (not in this git root)

```
D:\project Disha\dishanew\
  client/   # Vite + React
  server/   # Express + MongoDB (Mongoose)
  docs/     # empty
  README.md
```

---

## Part 4 — Git state (read-only)

| Field | Value |
| --- | --- |
| Repo root | `D:\project Disha\hack-disha` |
| Current branch | `cursor/htr-baseline-inference-53f4` |
| Related base branch | `cursor/disha-htr-standalone-c50d` |
| `main` tip | `7efd059` Initial commit |
| HTR commits observed | `effe342` standalone module; `fa0d3a3` FP16 baseline + metrics |
| Working tree at audit | Clean after `fa0d3a3` (handover file is **new** and uncommitted until user commits) |
| Agent commit policy | **Permanent:** agents must **not** `git commit`; user commits |

Suggested user commit for **this file only** (when ready):

```powershell
cd "D:\project Disha\hack-disha"
git add DISHA-HTR/DISHA_AI_INTEGRATION_HANDOVER.md
git commit -m "Add DISHA AI integration handover audit document."
```

**Do not commit:** `.venv/`, `checkpoints/*.safetensors`, `.cache/`, `logs/`, secrets, datasets, PHI samples.

---

## Part 5 — Environment & hardware (verified earlier; not re-probed with model load)

From prior setup + baseline reports (trust lightly; values below are **recorded**, not re-measured in this audit pass):

| Check | Recorded value |
| --- | --- |
| OS | Windows 11 (AMD64) |
| Python | 3.14.8 (`py -3.14`) |
| Venv | `DISHA-HTR\.venv` |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| VRAM (torch) | ~6143.5 MiB |
| Driver | 581.86 |
| nvidia-smi CUDA Version field | 13.0 |
| PyTorch | 2.11.0+cu128 |
| `torch.cuda.is_available()` | True |
| Selected inference device (baseline) | `cuda` |
| FP16 on CUDA | Enabled (`use_fp16: true`, dtype `float16`) |

---

## Part 6 — Dependency inventory

Declared in `requirements.txt` (install torch CUDA wheel **first**, then this file):

- `torch>=2.11.0`
- `transformers>=4.40.0`
- `Pillow>=10.0.0`
- `opencv-python-headless>=4.8.0`
- `numpy>=1.26.0`
- `PyYAML>=6.0`
- `sentencepiece>=0.2.0`
- `protobuf>=4.25.0`
- `tiktoken>=0.7.0`

Locked / previously verified versions (from lock + baseline report):

| Package | Version |
| --- | --- |
| torch | 2.11.0+cu128 |
| transformers | 5.22.1 |
| Pillow | 13.0.0 |
| opencv-python-headless | 5.0.0.93 |
| sentencepiece | 0.2.2 |
| protobuf | 7.36.2 |
| numpy | 2.5.4 |

**Known tokenizer pitfall:** without `sentencepiece` **and** `protobuf` (`google.protobuf`), transformers 5.x falls back to a tiktoken path that fails on TrOCR’s `sentencepiece.bpe.model`.

**Not present:** FastAPI/uvicorn, training stack (datasets/accelerate/peft), Node bindings, Docker/Compose for HTR.

---

## Part 7 — Python architecture & public API

Package: `disha_htr` (`src/disha_htr/`), version `__version__ = "0.1.1"`.

| Module | Responsibility | Key symbols |
| --- | --- | --- |
| `hardware.py` | CUDA detect, device resolve, HardwareInfo JSON | `detect_hardware`, `resolve_device` |
| `model_loader.py` | Hub/local load, FP16 auto, load timing | `load_trocr`, `LoadedModel` |
| `preprocess.py` | OpenCV/Pillow load, light CLAHE | `load_image`, `prepare_for_trocr` |
| `recognize.py` | Single-line generate + timings/VRAM | `recognize_image`, `RecognitionResult` |
| `inference.py` | High-level baseline orchestration | `run_baseline_inference`, `find_sample_image` |
| `logging_config.py` | Console + file logger | `setup_logging` |

**Exported package API** (`__init__.py`):  
`detect_hardware`, `resolve_device`, `load_trocr`, `recognize_image`, `find_sample_image`, `run_baseline_inference`.

**Scripts:**

- `scripts/make_sample_image.py` — synthetic Segoe Print line (not real handwriting).
- `scripts/test_recognize.py` — CLI wrapper around `run_baseline_inference`.

**Config:** `config/default.yaml` — model name/paths, device preference, `use_fp16: null` (auto), recognition `max_new_tokens` / `num_beams`.

**Design intent (documented):** FastAPI can later wrap `run_baseline_inference` / `recognize_image`. No HTTP layer exists yet.

---

## Part 8 — Baseline evidence (verbatim; from prior run — model not reloaded)

**Command (already executed earlier):**

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"
.\.venv\Scripts\python.exe scripts\test_recognize.py
```

**Sample:** `samples\synthetic_line.png` (synthetic text `Patient feels better today` — not PHI, not real handwriting).

**Recognition JSON (actual):**

```json
{
  "text": "retirement federal intervention .",
  "image_path": "samples\\synthetic_line.png",
  "device": "cuda",
  "model_name": "microsoft/trocr-small-handwritten",
  "model_source": "local:D:\\project Disha\\hack-disha\\DISHA-HTR\\checkpoints\\trocr-small-handwritten",
  "dtype": "float16",
  "use_fp16": true,
  "model_load_time_s": 4.2662,
  "inference_time_s": 3.213,
  "gpu_memory_allocated_mb": 129.82,
  "gpu_memory_peak_mb": 139.83
}
```

| Metric | Verbatim value |
| --- | --- |
| Extracted text | `retirement federal intervention .` |
| Device | `cuda` |
| Model load time | `4.2662` s |
| Inference time | `3.213` s |
| GPU allocated | `129.82` MiB |
| GPU peak | `139.83` MiB |

**Interpretation constraint:** Mismatch vs synthetic glyphs is **expected**; do **not** treat as CER/WER proof. Real single-line handwriting samples are still required for quality evaluation.

---

## Part 9 — Track A MERN surface (reachable at `dishanew`)

**App summary** (`dishanew/README.md`): Phase 1 accounts with phone OTP, Disha ID, documents, doctor OTP access, permanent Patient History.

**Server:** Express (`disha-server`), MongoDB/Mongoose, modules:

- `/auth`, `/documents`, `/access`, `/history`, `/admin`, `/users`, `/emergency`, `/meta/specialities`

**Client:** React + Vite feature folders (`auth`, `documents`, `history`, `emergency`, `admin`, `access`, …).

**Document upload:** Multer present; uploads under `server/uploads/`. Natural future hook for OCR/HTR **after** upload — **not wired**.

**Cheap test present:** `server/tests/helpers.test.js` (profile/age/regex helpers only). **No** OCR/HTR tests. Not re-run for this audit (would not inform AI integration).

**Search for HTR/OCR/TrOCR/Python/FastAPI in `dishanew` `src`:** **no matches**.

---

## Part 10 — Node ↔ Python integration status

| Capability | Status |
| --- | --- |
| Shared repo packaging of HTR inside Track A | Absent (`dishanew` ≠ `hack-disha`) |
| HTTP microservice for TrOCR | Absent |
| `child_process` / queue calling Python | Absent |
| Authn/z for inference endpoint | N/A (no endpoint) |
| Async job model for long GPU inference | Absent |
| Contract (JSON schema) for recognize request/response | Only implied by `RecognitionResult.to_dict()` / `run_baseline_inference` payload |
| PHI-safe upload → temp → infer → purge flow | Absent |

**Implication:** Integration is a **greenfield** adapter task. Recommended shape (guidance only, not implemented): Node documents service posts image bytes/path → Python process or local FastAPI → returns `{ text, device, timings, model_name }` using existing `recognize_image`.

---

## Part 11 — Checkpoints, samples, logging, gitignore

| Artifact | Location | Commit? |
| --- | --- | --- |
| Local TrOCR snapshot | `checkpoints/trocr-small-handwritten/` (~235 MB weights) | **No** (weights gitignored; README kept) |
| HF cache | `.cache/huggingface/` | **No** |
| Synthetic sample | `samples/synthetic_line.png` | Present locally; treat as demo only |
| Sample instructions | `samples/README.md` | Yes (docs) |
| Runtime logs | `logs/` | **No** |
| Venv | `.venv/` | **No** |

Missing-sample behavior: `find_sample_image` / `run_baseline_inference` raises `FileNotFoundError` with operator instructions — **does not invent text**.

---

## Part 12 — Spec / milestone gap analysis

### Master handover document

**Status: NOT INSPECTABLE** as a complete “DISHA Master Project Handover & Technical Specification” file.

**Partial substitute inspected:** screenshot `Screenshot 2026-11-03 150726.png` (doctor-specific training + model decisions).

### Model decisions (from screenshot fragment)

| Component | Spec decision | Current code |
| --- | --- | --- |
| Language | English only | English TrOCR baseline only (no multilingual stack) |
| Handwriting | Cursive + non-cursive; primary focus non-cursive | General `trocr-small-handwritten`; no style split |
| Base model | Pretrained general HTR | **Matches** pretrained TrOCR small handwritten |
| Personalization | Doctor-specific adaptation | **Not started** |
| Training interface | Separate doctor-specific section | **Not started** (no UI/API) |
| Training samples | Seven categories (A–Z, a–z, 0–9, medical vocab, meds/dosage, prescription sentences, real handwriting + verified transcripts) | **None collected / no training pipeline** |

### Hackathon deck milestones (`DISHA_Hackathon_2k26.pptx`)

| Deck claim | Observed now |
| --- | --- |
| Months 6–7: OCR module | Standalone inference module exists; not product-integrated |
| Prototype “not yet started” (deck narrative) | **Stale relative to workspace** — MERN Phase 1 + HTR baseline both exist locally |
| TRL 2–3 (deck) | HTR experimental PoC on GPU is further along than deck text; still not clinical-grade |

### Project task board (Agent Store `notes.md`)

| ID | Title | Board state at audit |
| --- | --- | --- |
| DI-1 | Standalone TrOCR module | Done |
| DI-2 | Reliable baseline inference | For review |
| DI-3 | AI integration audit (this doc) | In progress → complete when this file accepted |

### Gaps vs a minimal “AI integrated into DISHA” milestone

1. No Node↔Python bridge  
2. No HTR API in Express routes  
3. No UI for handwriting upload → transcript  
4. No real handwriting evaluation set / metrics (CER/WER)  
5. No doctor-specific training / personalization  
6. No page/line segmentation for multi-line prescriptions  
7. No ABDM/FHIR structuring of OCR output (deck ambition; not in HTR code)

---

## Part 13 — Risks, constraints, unknowns

| Risk / constraint | Detail |
| --- | --- |
| Synthetic-sample bias | Baseline text output is **not** an accuracy claim |
| PHI | Never commit real patient handwriting; samples README forbids PHI |
| GPU VRAM | 6 GB laptop GPU; FP16 helps (~140 MiB peak on small TrOCR — headroom exists, but larger models / batching need care) |
| Ephemeral caches | `.cache/` / local weights must be preserved across machines; not in git |
| Dual trees | `hack-disha` vs `dishanew` split increases integration friction |
| Python 3.14 | Unusual pin; package compatibility must stay watched |
| No service auth | Any future HTTP wrapper needs auth aligned with doctor/patient RBAC |
| Master Spec incomplete | Decisions may exist only in screenshots/chat — confirm with owners before training work |
| Permanent git rule | Agents must not commit; user owns commits |

---

## Part 14 — Handover actions for the next engineer (ordered)

**Do next (recommended order):**

1. **Confirm Master Spec** — obtain the full “DISHA Master Project Handover & Technical Specification” (or export from the source of the screenshot) and reconcile Part 12 against the authoritative milestones.  
2. **User-commit this handover** (and any already-approved HTR code) per Part 4 commands — agents will not commit.  
3. **Add one real single-line handwriting sample** (no PHI) under `samples/` and re-run `scripts/test_recognize.py --image ...` to get a meaningful qualitative baseline.  
4. **Define a frozen JSON contract** matching `RecognitionResult` / `run_baseline_inference` payload for Node consumers.  
5. **Prototype Node→Python call** (subprocess *or* small FastAPI on `0.0.0.0:$PORT`) from `dishanew` documents flow — **without** changing DB schemas until contract is agreed.  
6. **Only then** plan doctor-specific data collection (seven categories) and fine-tuning — not before evaluation samples exist.

**Do not:**

- Recreate `.venv` or reinstall PyTorch if CUDA already works  
- Fine-tune or download alternate large models “just to explore” without an evaluation set  
- Commit weights, caches, logs, secrets, or PHI  
- Claim clinical accuracy from synthetic Segoe Print results  
- Modify Track A APIs/DB in the same change set as the first bridge spike unless explicitly requested  

**Stop condition for this audit agent:** document delivered; **no further implementation** until a new explicit task.

---

## Appendix A — How to re-run baseline (operator)

```powershell
cd "D:\project Disha\hack-disha\DISHA-HTR"
.\.venv\Scripts\python.exe scripts\test_recognize.py
# or
.\.venv\Scripts\python.exe scripts\test_recognize.py --image samples\synthetic_line.png
```

## Appendix B — Related absolute paths

- HTR root: `D:\project Disha\hack-disha\DISHA-HTR`  
- This handover: `D:\project Disha\hack-disha\DISHA-HTR\DISHA_AI_INTEGRATION_HANDOVER.md`  
- Baseline report: `C:\Users\karna-gr\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-6f1fae26-0879-485d-ba4c-4f18326b7a55\files\internal\htr\baseline-inference-report.md`  
- Setup report: `C:\Users\karna-gr\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-6f1fae26-0879-485d-ba4c-4f18326b7a55\files\internal\htr\setup-report.md`  
- Track A app: `D:\project Disha\dishanew`  
- Spec screenshot fragment: `D:\project Disha\Screenshot 2026-11-03 150726.png`  
- Hackathon PPT: `D:\project Disha\DISHA_Hackathon_2k26.pptx`
