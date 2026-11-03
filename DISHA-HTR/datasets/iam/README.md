# IAM Handwriting Database (Phase 1)

English offline handwriting lines for future TrOCR fine-tuning experiments.

## Licensing / access

- **Official source:** [IAM Handwriting Database (FKI / HEIA-FR)](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- **Terms:** Free for **non-commercial research**; **registration required** before download.
- **Citation (required if you publish):** Marti & Bunke, IJDAR 2002 — see FKI terms page.
- This project does **not** redistribute IAM images. Archives stay local under `raw/` (gitignored).

## Layout

```
datasets/iam/
  raw/              # downloaded archives (unchanged once saved)
  extracted/        # extracted lines/ascii/splits (derived)
  processed/        # reserved for later phases
  manifests/        # train.csv / val.csv / test.csv + stats.json
```

## Manifest columns

`image_path,text,writer_id,line_id,form_id,split`

See `docs/DATASET_PREPARATION.md` for PowerShell commands.
