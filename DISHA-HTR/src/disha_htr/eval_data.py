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
GT_FIELDS_WITH_SOURCE = ("image_filename", "category", "source_page", "ground_truth")


@dataclass(frozen=True)
class EvalImage:
    """One evaluation image on disk."""

    path: Path
    image_filename: str  # relative to images root, POSIX-style
    category: str
    source_page: str | None = None  # original page for line crops


@dataclass(frozen=True)
class GroundTruthRow:
    image_filename: str
    category: str
    ground_truth: str | None  # None = missing / blank (excluded from metrics)
    source_page: str | None = None


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
            found.append(
                EvalImage(
                    path=path,
                    image_filename=rel,
                    category=category,
                    source_page=None,
                )
            )

    return found


def discover_line_crop_images(
    lines_root: str | Path,
    manifest_path: str | Path | None = None,
) -> list[EvalImage]:
    """
    Discover line-crop images under lines_root/{category}/.

    If a crops manifest is provided, attach source_page from it; otherwise
    infer source_page as best-effort from filename stem before `_Lxx`.
    """
    root = Path(lines_root)
    source_by_crop: dict[str, str] = {}
    if manifest_path is not None:
        mpath = Path(manifest_path)
        if mpath.is_file() and mpath.stat().st_size > 0:
            with mpath.open("r", encoding="utf-8-sig", newline="") as fh:
                for raw in csv.DictReader(fh):
                    crop = (raw.get("crop_filename") or "").strip()
                    src = (raw.get("source_page") or "").strip()
                    if crop:
                        source_by_crop[crop] = src

    found: list[EvalImage] = []
    for category in CATEGORIES:
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue
        for path in sorted(cat_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            source_page = source_by_crop.get(rel)
            if not source_page:
                # Best-effort: NC01_L02.png → category/NC01.jpeg (unknown ext)
                stem = path.stem
                if "_L" in stem:
                    page_stem = stem.rsplit("_L", 1)[0]
                    source_page = f"{category}/{page_stem}"
                else:
                    source_page = None
            found.append(
                EvalImage(
                    path=path,
                    image_filename=rel,
                    category=category,
                    source_page=source_page,
                )
            )
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
            source_page = None
            if "source_page" in field_map:
                source_page = (raw.get(field_map["source_page"]) or "").strip() or None
            rows[filename] = GroundTruthRow(
                image_filename=filename,
                category=category,
                ground_truth=gt,
                source_page=source_page,
            )
    return rows


def ensure_ground_truth_template(
    csv_path: str | Path,
    images: Iterable[EvalImage],
    *,
    include_source_page: bool | None = None,
) -> tuple[Path, int, int]:
    """
    Create or extend a GT CSV template.

    Never overwrites an existing non-blank ground_truth cell.
    Never invents ground_truth text (new rows get blank GT).
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
                source_page=img.source_page,
            )
            ordered_keys.append(img.image_filename)
            newly_added += 1
        else:
            # Refresh category / source_page metadata if blank; never touch GT text
            prev = existing[img.image_filename]
            new_category = prev.category or img.category
            new_source = prev.source_page or img.source_page
            if new_category != prev.category or new_source != prev.source_page:
                existing[img.image_filename] = GroundTruthRow(
                    image_filename=prev.image_filename,
                    category=new_category,
                    ground_truth=prev.ground_truth,
                    source_page=new_source,
                )

    use_source = include_source_page
    if use_source is None:
        use_source = any(
            (existing[k].source_page for k in ordered_keys)
        ) or any(img.source_page for img in images_list)

    fields = list(GT_FIELDS_WITH_SOURCE if use_source else GT_FIELDS)

    # If file was missing/empty and no images, still write header-only template
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in ordered_keys:
            row = existing[key]
            payload = {
                "image_filename": row.image_filename,
                "category": row.category,
                "ground_truth": "" if row.ground_truth is None else row.ground_truth,
            }
            if use_source:
                payload["source_page"] = row.source_page or ""
            writer.writerow(payload)

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
