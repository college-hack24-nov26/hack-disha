"""
Download Teklia/IAM-line from Hugging Face into local folders for training.

Does NOT place clones inside DISHA-HTR git history as submodules.
Writes under datasets/iam/hf_teklia/ (gitignored via datasets/iam/processed patterns
and explicit ignore below if needed).

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\download_teklia_iam_line.py --max-train 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "iam" / "hf_teklia"


def _import_hf_datasets():
    # Local folder named datasets/ shadows the Hugging Face package on sys.path.
    cleaned = []
    for p in sys.path:
        if not p:
            continue
        norm = p.replace("\\", "/").rstrip("/")
        if norm.endswith("/DISHA-HTR") or norm.endswith("/hack-disha/DISHA-HTR"):
            continue
        if norm.endswith("/DISHA-HTR/datasets") or norm.endswith("/datasets") and "site-packages" not in norm:
            # Keep site-packages datasets; drop project-local datasets dir entries.
            if "site-packages" not in norm:
                continue
        cleaned.append(p)
    sys.path[:] = cleaned
    import datasets  # noqa: WPS433

    return datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=200)
    args = parser.parse_args()

    datasets = _import_hf_datasets()
    print("Loading Teklia/IAM-line (streaming slices)…")

    for split, limit in (
        ("train", args.max_train),
        ("validation", args.max_val),
        ("test", args.max_test),
    ):
        split_dir = OUT / split / "images"
        split_dir.mkdir(parents=True, exist_ok=True)
        # Clear partial exports from a previous interrupted run.
        for old in split_dir.glob("*.png"):
            old.unlink(missing_ok=True)
        rows = []
        print(f"{split}: streaming up to {limit}")
        stream = datasets.load_dataset(
            "Teklia/IAM-line", split=split, streaming=True
        )
        for i, ex in enumerate(stream):
            if i >= limit:
                break
            text = (ex.get("text") or "").strip()
            img = ex.get("image")
            if not text or img is None:
                continue
            name = f"{split}_{len(rows):05d}.png"
            path = split_dir / name
            img.convert("RGB").save(path)
            rows.append(
                {
                    "image_filename": name,
                    "ground_truth": text,
                    "source_page": f"teklia_{split}",
                    "split": split,
                }
            )
            if (len(rows) % 50) == 0:
                print(f"  {split}: {len(rows)} saved")
        csv_path = OUT / split / "labels.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["image_filename", "ground_truth", "source_page", "split"]
            )
            w.writeheader()
            w.writerows(rows)
        (OUT / split / "meta.json").write_text(
            json.dumps({"count": len(rows), "source": "Teklia/IAM-line"}, indent=2),
            encoding="utf-8",
        )
        print(f"  wrote {len(rows)} -> {csv_path}")

    (OUT / "SOURCE.txt").write_text(
        "Source: https://huggingface.co/datasets/Teklia/IAM-line\n"
        "Original IAM rights: FKI / University of Bern — use for research/demo.\n"
        "Do not commit image binaries.\n",
        encoding="utf-8",
    )
    print("Done:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
