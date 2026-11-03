"""One-shot rebuild of demo approved gallery + held-out eval images."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def clean_line(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    horiz = cv2.morphologyEx(
        thr,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, bgr.shape[1] // 15), 1)),
    )
    cleaned = gray.copy()
    cleaned[horiz > 0] = 255
    h, w = cleaned.shape
    scale = max(1.0, 128 / max(h, 1))
    cleaned = cv2.resize(
        cleaned, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    p2, p98 = np.percentile(cleaned, (2, 98))
    cleaned = np.clip((cleaned - p2) * 255.0 / max(p98 - p2, 1), 0, 255).astype(
        np.uint8
    )
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)


def main() -> None:
    approved = ROOT / "demo" / "approved"
    approved.mkdir(parents=True, exist_ok=True)
    for p in approved.glob("*.png"):
        p.unlink()

    src = ROOT / "demo" / "train_private"
    rows = list(
        csv.DictReader(
            (src / "train_ground_truth.csv").open(encoding="utf-8-sig", newline="")
        )
    )
    manifest = []
    for i, r in enumerate(rows, start=1):
        stem = Path(r["image_filename"]).stem.replace("train_", "")
        dest_name = f"demo_{i:02d}_{stem}.png"
        Image.open(src / r["image_filename"]).convert("RGB").save(approved / dest_name)
        manifest.append(
            {
                "id": dest_name.replace(".png", ""),
                "filename": dest_name,
                "source_page": r["source_page"],
                "label": r["ground_truth"][:42],
                "train_source": r["image_filename"],
            }
        )

    (ROOT / "demo" / "approved_manifest.json").write_text(
        json.dumps(
            {
                "images": manifest,
                "note": "Gallery for UI only; GT not stored here. Held-out metrics use eval_private/.",
                "checkpoint_preference": "trocr-demo-finetuned if present else base",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (ROOT / "demo" / "eval_private" / "adaptation_check_gt.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_filename", "source_page", "ground_truth", "note"]
        )
        writer.writeheader()
        for m, r in zip(manifest, rows, strict=True):
            writer.writerow(
                {
                    "image_filename": m["filename"],
                    "source_page": r["source_page"],
                    "ground_truth": r["ground_truth"],
                    "note": "NOT held-out; seen during demo fine-tune",
                }
            )

    held = ROOT / "demo" / "eval_private" / "heldout_images"
    held.mkdir(parents=True, exist_ok=True)
    held_specs = [
        (
            "line01_ghatboral.png",
            "evaluation/lines/non_cursive/NC02_L03.png",
            "I am Ghatboral Iq",
            "non_cursive/NC02.jpeg",
        ),
        (
            "line02_karnataka.png",
            "evaluation/lines/non_cursive/NC02_L05.png",
            "Karnataka 585418",
            "non_cursive/NC02.jpeg",
        ),
        (
            "line03_dsa.png",
            "evaluation/lines/non_cursive/NC02_L08.png",
            "DSA Questions I feel",
            "non_cursive/NC02.jpeg",
        ),
        (
            "line04_happy.png",
            "evaluation/lines/non_cursive/NC02_L09.png",
            "happy when I am about",
            "non_cursive/NC02.jpeg",
        ),
        (
            "line05_aba.png",
            "evaluation/lines/non_cursive/NC05_L08.png",
            "A B a",
            "non_cursive/NC05.jpeg",
        ),
        (
            "line06_practicing.png",
            "evaluation/lines/non_cursive/NC02_L07.png",
            "I Have started practicing",
            "non_cursive/NC02.jpeg",
        ),
    ]
    page = cv2.imread(str(ROOT / "evaluation/images/non_cursive/NC01.jpeg"))
    held_manual = [
        ("line07_hemanth.png", page[115:185, 40:480], "HEMANTH", "non_cursive/NC01.jpeg"),
        (
            "line08_howareyou.png",
            page[530:585, 40:480],
            "how are you",
            "non_cursive/NC01.jpeg",
        ),
    ]
    gt_rows = []
    for fname, src_path, gt, sp in held_specs:
        bgr = cv2.imread(str(ROOT / src_path))
        Image.fromarray(clean_line(bgr)).save(held / fname)
        gt_rows.append(
            {"image_filename": fname, "source_page": sp, "ground_truth": gt}
        )
    for fname, crop, gt, sp in held_manual:
        Image.fromarray(clean_line(crop)).save(held / fname)
        gt_rows.append(
            {"image_filename": fname, "source_page": sp, "ground_truth": gt}
        )

    with (ROOT / "demo" / "eval_private" / "ground_truth.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_filename", "source_page", "ground_truth"]
        )
        writer.writeheader()
        writer.writerows(gt_rows)

    print("approved", len(manifest))
    print("heldout", len(gt_rows))


if __name__ == "__main__":
    main()
