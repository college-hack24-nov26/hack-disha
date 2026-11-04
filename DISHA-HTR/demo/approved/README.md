# Verified IAM sample showcase

These showcase PNGs were downloaded from Hugging Face
[`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line) (test split).

Each row in `samples_manifest.json` binds a stable `sample_id` to:

1. the handwriting image file
2. the official IAM transcription (`expected_text` / verified reference)

## Demo

1. Click a sample card — the matching image and verified reference load together.
2. Press **Recognize** — TrOCR produces a **model prediction** (real inference).
3. The UI shows prediction and verified reference separately. A match is reported
   only when the prediction equals the reference; never by substituting GT.

IAM terms: non-commercial research; cite Marti & Bunke / FKI as required by the dataset.
