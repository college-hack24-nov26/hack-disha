"""Evaluation image discovery and ground-truth CSV helpers (no model I/O)."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("disha_htr.eval_data")

CATEGORIES = (
    "non_cursive",
    "cursive",
    "medical_vocabulary",
    "prescription_style",
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

GT_FIELDS = ("image_filename", "category", "ground_truth")


@dataclass(frozen=True)
class EvalImage:
    """One evaluation image on disk."""

    path: Path
    image_filename: str  # relative to images root, POSIX-style
    category: str


@dataclass(frozen=True)
class GroundTruthRow:
    image_filename: str
    category: str
    ground_truth: str | None  # None = missing / blank (excluded from metrics)


def discover_eval_images(images_root: str | Path) -> list[EvalImage]:
    """
    Recursively discover supported images under
    images_root/{non_cursive,cursive,medical_vocabulary,prescription_style}/.
    """
    root = Path(images_root)
    found: list[EvalImage] = []

    for category in CATEGORIES:
        cat_dir = root / category
        if not cat_dir.is_dir():
            logger.warning("Category folder missing: %s", cat_dir)
            continue
        for path in sorted(cat_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            found.append(EvalImage(path=path, image_filename=rel, category=category))

    return found


def _is_blank_gt(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip() == ""


def load_ground_truth_csv(csv_path: str | Path) -> dict[str, GroundTruthRow]:
    """
    Load GT keyed by image_filename.

    Blank / whitespace-only ground_truth → treated as missing (None).
    Missing file → empty dict.
    """
    path = Path(csv_path)
    if not path.is_file() or path.stat().st_size == 0:
        return {}

    rows: dict[str, GroundTruthRow] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return {}
        # Tolerate minor header variants
        field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
        required = ("image_filename", "category", "ground_truth")
        for req in required:
            if req not in field_map:
                raise ValueError(
                    f"Ground-truth CSV missing required column '{req}': {path}"
                )

        for raw in reader:
            filename = (raw.get(field_map["image_filename"]) or "").strip()
            if not filename:
                continue
            category = (raw.get(field_map["category"]) or "").strip()
            gt_raw = raw.get(field_map["ground_truth"])
            gt = None if _is_blank_gt(gt_raw) else str(gt_raw).strip()
            rows[filename] = GroundTruthRow(
                image_filename=filename,
                category=category,
                ground_truth=gt,
            )
    return rows


def ensure_ground_truth_template(
    csv_path: str | Path,
    images: Iterable[EvalImage],
) -> tuple[Path, int, int]:
    """
    Create or extend a GT CSV template.

    Never overwrites an existing non-blank ground_truth cell.
    Adds rows for newly discovered images with empty ground_truth.
    Preserves existing rows (including orphan filenames not on disk).

    Returns (path, rows_written_total, newly_added_count).
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_ground_truth_csv(path)
    images_list = list(images)
    newly_added = 0

    # Start from existing order, then append new image keys
    ordered_keys: list[str] = list(existing.keys())
    for img in images_list:
        if img.image_filename not in existing:
            existing[img.image_filename] = GroundTruthRow(
                image_filename=img.image_filename,
                category=img.category,
                ground_truth=None,
            )
            ordered_keys.append(img.image_filename)
            newly_added += 1
        else:
            # Refresh category from disk folder if blank in CSV; never touch GT text
            prev = existing[img.image_filename]
            if not prev.category:
                existing[img.image_filename] = GroundTruthRow(
                    image_filename=prev.image_filename,
                    category=img.category,
                    ground_truth=prev.ground_truth,
                )

    # If file was missing/empty and no images, still write header-only template
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(GT_FIELDS))
        writer.writeheader()
        for key in ordered_keys:
            row = existing[key]
            writer.writerow(
                {
                    "image_filename": row.image_filename,
                    "category": row.category,
                    "ground_truth": "" if row.ground_truth is None else row.ground_truth,
                }
            )

    logger.info(
        "Ground-truth CSV ready at %s (%d rows, %d newly added)",
        path,
        len(ordered_keys),
        newly_added,
    )
    return path, len(ordered_keys), newly_added


def has_usable_ground_truth(row: GroundTruthRow | None) -> bool:
    """True only when GT is present and non-blank — never treat missing as ''."""
    return row is not None and row.ground_truth is not None and not _is_blank_gt(
        row.ground_truth
    )
