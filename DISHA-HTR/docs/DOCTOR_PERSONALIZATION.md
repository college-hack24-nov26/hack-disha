# Doctor-specific handwriting personalization

Architecture for personalized TrOCR recognition **without modifying** the shared
general checkpoint or the judge demo.

## Status

**Architecture ready.** Personalization storage, isolation, selection, fallback,
versioning/rollback, verified-sample registry, correction workflow, and a gated
decoder-only trainer scaffold are implemented.

**No personalized accuracy claims.** Large-scale doctor training is gated and
must not run without explicit approval (`allow_training=True` plus sufficient
verified train/val samples).

## Layout

```
personalization/
  doctors/
    <doctor_id>/
      profile.json
      samples/
        train/
        val/
        verified_transcriptions.csv
      adapters/
        active.json          # optional pointer
        v0001/
          meta.json
          model.safetensors  # when trained
        v0002/
      corrections/
        pending/ | approved/ | rejected/
      audit/
        audit.jsonl
```

General base (never overwritten): `checkpoints/trocr-small-handwritten`

## Recognition flow

1. Host app authenticates the clinician → `DoctorAuthContext(doctor_id=...)`.
2. `select_checkpoint_for_doctor` picks:
   - active personalized adapter **if** weights exist **and** `improves_over_general`
     is not `False`;
   - otherwise **general** TrOCR.
3. `recognize_for_doctor` runs recognition; optional `compare_general=True`.
4. Corrections go through `CorrectionWorkflow` (pending → approved). Only approved
   promotions enter verified training data.

## Training policy

- Method: **decoder-only fine-tune** from the general checkpoint into a new
  per-doctor version directory (parameter-efficient relative to full retrain;
  no extra LoRA dependency required).
- Shared general weights are **read-only**.
- Activation refuses versions that do not beat general val CER (unless `force`
  for admin tests).
- CLI `plan` reports readiness blockers; default refuses training.

## CLI

```text
.\.venv\Scripts\python.exe scripts\doctor_personalization_cli.py create-profile --doctor-id doc_a
.\.venv\Scripts\python.exe scripts\doctor_personalization_cli.py plan --doctor-id doc_a
.\.venv\Scripts\python.exe scripts\doctor_personalization_cli.py select --doctor-id doc_a
.\.venv\Scripts\python.exe scripts\doctor_personalization_cli.py rollback --doctor-id doc_a
```

## Privacy

- Strict per-doctor directories; path escape checks.
- No patient ID fields in personalization payloads.
- Audit log redacts common PII patterns; does not store full transcriptions on
  recognition events (length only).
- Unverified / model-prediction sources cannot become ground truth.

## Tests

```text
.\.venv\Scripts\python.exe -m unittest tests.test_doctor_personalization -v
```

Covers cross-doctor isolation, model selection, general fallback, worse-model
fallback, versioning/rollback, correction gating, training gate, demo untouched.
