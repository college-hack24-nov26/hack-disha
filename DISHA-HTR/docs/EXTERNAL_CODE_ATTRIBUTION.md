# External code attribution (DISHA-HTR generalization work)

External repositories were cloned outside this project under
`D:\project Disha\_external_htr_references\` for study. They are **not**
Git submodules and must not be committed into DISHA-HTR.

## Repositories inspected

| Repository | Path | License |
|---|---|---|
| https://github.com/rsommerfeld/trocr | `_external_htr_references/rsommerfeld-trocr` | **MIT** (Copyright 2022 rsommerfeld) — verified `LICENSE` |
| https://github.com/ertugd/trocr-handwriting | `_external_htr_references/ertugd-trocr-handwriting` | **No LICENSE file** — reference only; source not copied |
| https://github.com/Tripathijii147/handwriting-recognition-trocr-crnn | `_external_htr_references/Tripathijii147-handwriting-recognition-trocr-crnn` | **No LICENSE file** — reference only; source not copied |
| https://github.com/ThanhTamK4/handwriting-recognition | `_external_htr_references/ThanhTamK4-handwriting-recognition` | **No LICENSE file** — reference only; source not copied |
| https://huggingface.co/datasets/Teklia/IAM-line | `_external_htr_references/Teklia-IAM-line-README.md` | Dataset card README only; **IAM image rights remain with FKI** — do not ship IAM images without registration |

## Code adapted (MIT)

From **rsommerfeld/trocr** (MIT), adapted ideas into
`scripts/finetune_improved_trocr.py` (not a verbatim file copy):

- Pattern of configuring TrOCR `decoder_start_token_id` / `pad_token_id` /
  `vocab_size` before training (`src/util.py` `init_model_for_training`).
  **Change:** we keep the pretrained handwritten start id (`sep` / 2) instead
  of forcing `cls`, because diagnosis showed `cls` collapsed held-out generation.
- Linear learning-rate schedule via `transformers.get_scheduler("linear", ...)`
  (`src/scripts.py` `train`).

Required MIT notice for adapted portions:

```
Copyright (c) 2022 rsommerfeld
MIT License — see the original LICENSE in the rsommerfeld/trocr repository.
```

## Techniques used as reference only (not copied)

From **ertugd/trocr-handwriting** (no license): decoder_start aligned to
`sep`, warmup ratio, mild handwriting-safe augmentation, validation CER,
early stopping / best checkpoint — reimplemented independently.

From **Tripathijii147/handwriting-recognition-trocr-crnn** (no license):
high-level train/eval layout observed only; no source files copied. Their
bundled IAM-processed images were **not** used (IAM requires FKI registration;
this project only has `splits.zip` locally).

From **ThanhTamK4/handwriting-recognition** (no license): observed Flask/TrOCR
demo patterns only; not copied into DISHA-HTR (this project uses its own
stdlib HTTP demo server).

From **Teklia/IAM-line** (dataset card): line-level IAM packaging on Hugging Face.
Demo training may download a **local research subset** via
`scripts/download_teklia_iam_line.py` into `datasets/iam/hf_teklia/` (gitignored).
Original IAM rights remain with FKI / University of Bern — do not redistribute
image binaries; keep exports out of git commits.
