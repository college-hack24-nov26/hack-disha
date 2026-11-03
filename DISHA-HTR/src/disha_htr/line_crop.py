"""
Line cropping for full-page handwriting images.

Originals are never modified. Crops are written to a separate directory and
always reference the source page filename in metadata / output names.
Does not invent ground-truth text.
"""

from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .eval_data import IMAGE_EXTENSIONS

logger = logging.getLogger("disha_htr.line_crop")

MANIFEST_FIELDS = (
    "crop_filename",
    "source_page",
    "category",
    "line_index",
    "x0",
    "y0",
    "x1",
    "y1",
    "crop_width",
    "crop_height",
)


@dataclass(frozen=True)
class LineCropSpec:
    """One line crop derived from a page image."""

    crop_filename: str  # relative to lines root, POSIX
    source_page: str  # relative to pages/images root, POSIX
    category: str
    line_index: int  # 1-based
    x0: int
    y0: int
    x1: int
    y1: int
    crop_path: Path

    @property
    def crop_width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def crop_height(self) -> int:
        return max(0, self.y1 - self.y0)

    def to_manifest_row(self) -> dict[str, str | int]:
        return {
            "crop_filename": self.crop_filename,
            "source_page": self.source_page,
            "category": self.category,
            "line_index": self.line_index,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
        }


def crop_filename_for(source_page: str, line_index: int) -> str:
    """
    Build a crop relative path that embeds the original page stem.

    Example: non_cursive/NC01.jpeg + line 2 → non_cursive/NC01_L02.png
    """
    page = Path(source_page)
    stem = page.stem
    category = page.parent.as_posix()
    name = f"{stem}_L{line_index:02d}.png"
    if category in ("", "."):
        return name
    return f"{category}/{name}"


def preserve_original(
    source_path: Path,
    originals_root: Path,
    source_page: str,
) -> Path:
    """
    Copy page into originals_root/{source_page} if absent.

    Never overwrites an existing archived original.
    """
    dest = originals_root / source_page
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        logger.debug("Original already archived (left unchanged): %s", dest)
        return dest
    shutil.copy2(source_path, dest)
    logger.info("Archived original page copy: %s", dest)
    return dest


def _spans_from_projection(
    projection: np.ndarray,
    *,
    width: int,
    min_ink_ratio: float,
    thresh_frac: float,
    min_h: int,
    max_h: int,
    merge_gap: int,
) -> list[tuple[int, int]]:
    if projection.size == 0 or float(projection.max()) <= 0:
        return []
    thresh = max(float(projection.max()) * thresh_frac, width * min_ink_ratio)
    active = projection >= thresh
    h = int(projection.shape[0])
    raw_spans: list[tuple[int, int]] = []
    i = 0
    while i < h:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < h and active[j]:
            j += 1
        raw_spans.append((i, j))
        i = j
    if not raw_spans:
        return []
    merged: list[tuple[int, int]] = [raw_spans[0]]
    for start, end in raw_spans[1:]:
        prev_s, prev_e = merged[-1]
        if start - prev_e <= merge_gap:
            merged[-1] = (prev_s, end)
        else:
            merged.append((start, end))

    # Split oversized merges on local minima (common on dense notebook pages).
    spans: list[tuple[int, int]] = []
    for start, end in merged:
        height = end - start
        if height <= max_h:
            if height >= min_h:
                spans.append((start, end))
            continue
        seg = projection[start:end]
        mid = float(seg.mean()) * 0.7
        cuts: list[int] = []
        for k in range(4, len(seg) - 4):
            val = float(seg[k])
            if val < mid and val <= float(seg[k - 1]) and val <= float(seg[k + 1]):
                if not cuts or (k - cuts[-1]) > max(6, min_h // 2):
                    cuts.append(k)
        if not cuts:
            # Force equal-ish chunks rather than dropping the band entirely.
            chunk = max(min_h, max_h)
            t = start
            while t < end:
                spans.append((t, min(end, t + chunk)))
                t += chunk
            continue
        prev = start
        for c in cuts:
            spans.append((prev, start + c))
            prev = start + c
        spans.append((prev, end))
    return [(s, e) for s, e in spans if min_h <= (e - s) <= max_h * 2]


def _boxes_from_spans(
    ink: np.ndarray,
    spans: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    pad_y_ratio: float,
    min_width_ratio: float = 0.08,
    min_density: float = 0.008,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for start, end in spans:
        band = ink[start:end, :]
        ink_cols = np.where(band.sum(axis=0) > 0)[0]
        if ink_cols.size == 0:
            continue
        x0 = int(max(0, ink_cols[0] - 4))
        x1 = int(min(width, ink_cols[-1] + 5))
        line_h = end - start
        if (x1 - x0) < max(12, int(width * min_width_ratio)):
            continue
        dens = float(band.sum()) / 255.0 / max(1, line_h * (x1 - x0))
        if dens < min_density:
            continue
        pad = max(1, int(line_h * pad_y_ratio))
        y0 = max(0, start - pad)
        y1 = min(height, end + pad)
        boxes.append((x0, y0, x1, y1))
    return boxes


def _nms_y_overlap(
    boxes: list[tuple[int, int, int, int]],
    *,
    iou_thresh: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    """Drop heavily overlapping vertical bands (keep taller / earlier)."""
    if len(boxes) <= 1:
        return boxes
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[1], -(b[3] - b[1]))):
        _x0, y0, _x1, y1 = box
        h = max(1, y1 - y0)
        drop = False
        for _kx0, ky0, _kx1, ky1 in kept:
            inter = max(0, min(y1, ky1) - max(y0, ky0))
            union = h + max(1, ky1 - ky0) - inter
            if union > 0 and (inter / union) >= iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(box)
    return sorted(kept, key=lambda b: b[1])


def detect_line_boxes(
    image_bgr: np.ndarray,
    *,
    min_line_height_ratio: float = 0.012,
    max_line_height_ratio: float = 0.18,
    min_ink_ratio: float = 0.004,
    pad_y_ratio: float = 0.18,
    merge_gap_ratio: float = 0.006,
) -> list[tuple[int, int, int, int]]:
    """
    Detect approximate text-line bounding boxes.

    Uses CLAHE + adaptive threshold, suppresses long ruled notebook lines,
    then RLSA-style horizontal dilation + projection. Falls back to the
    simpler projection path if the enhanced detector finds nothing.

    Returns list of (x0, y0, x1, y1) in image coordinates, top-to-bottom.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12,
    )
    # Suppress long horizontal ruled lines common on notebook Rx pages.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 6), 1))
    ruled = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    ink = cv2.subtract(binary, ruled)
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    # Connect characters within a line (Run-Length Smoothing style).
    rlsa = cv2.dilate(
        ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 40), 1))
    )

    win = max(3, h // 100)
    kernel = np.ones(win, dtype=np.float64) / float(win)
    projection = np.convolve(rlsa.sum(axis=1).astype(np.float64) / 255.0, kernel, mode="same")

    # Notebook pages need a taller minimum so ruled-line crumbs are ignored.
    min_h = max(16, int(h * max(min_line_height_ratio, 0.022)))
    max_h = max(min_h + 1, int(h * max_line_height_ratio))
    merge_gap = max(2, int(h * merge_gap_ratio))

    spans = _spans_from_projection(
        projection,
        width=w,
        min_ink_ratio=min_ink_ratio,
        thresh_frac=0.12,
        min_h=min_h,
        max_h=max_h,
        merge_gap=merge_gap,
    )
    boxes = _boxes_from_spans(
        ink,
        spans,
        width=w,
        height=h,
        pad_y_ratio=pad_y_ratio,
        min_density=0.012,
    )
    boxes = _nms_y_overlap(boxes, iou_thresh=0.35)
    # Drop remaining ultra-short bands (often ruled paper ghosts).
    boxes = [b for b in boxes if (b[3] - b[1]) >= min_h]

    if boxes:
        return boxes

    # Fallback: original simple adaptive projection (no RLSA).
    simple_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    simple = cv2.adaptiveThreshold(
        simple_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    proj2 = simple.sum(axis=1).astype(np.float64) / 255.0
    spans2 = _spans_from_projection(
        proj2,
        width=w,
        min_ink_ratio=min_ink_ratio,
        thresh_frac=0.08,
        min_h=max(3, int(h * min_line_height_ratio)),
        max_h=max(min_h + 1, int(h * 0.25)),
        merge_gap=max(1, int(h * merge_gap_ratio)),
    )
    return _boxes_from_spans(
        simple,
        spans2,
        width=w,
        height=h,
        pad_y_ratio=pad_y_ratio,
        min_density=0.004,
    )


def crop_page_lines(
    page_path: Path,
    *,
    source_page: str,
    category: str,
    lines_root: Path,
    originals_root: Path | None = None,
) -> list[LineCropSpec]:
    """
    Detect lines on a page, write crop PNGs under lines_root, archive original.

    Never modifies page_path. Returns crop specs (may be empty if none found).
    """
    page_path = Path(page_path)
    if not page_path.is_file():
        raise FileNotFoundError(f"Page image not found: {page_path}")

    if originals_root is not None:
        preserve_original(page_path, Path(originals_root), source_page)

    image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode page image: {page_path}")

    boxes = detect_line_boxes(image)
    lines_root = Path(lines_root)
    crops: list[LineCropSpec] = []

    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        rel = crop_filename_for(source_page, idx)
        out_path = lines_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop_img = image[y0:y1, x0:x1]
        if crop_img.size == 0:
            continue
        if not cv2.imwrite(str(out_path), crop_img):
            raise OSError(f"Failed to write crop: {out_path}")
        crops.append(
            LineCropSpec(
                crop_filename=rel,
                source_page=source_page,
                category=category,
                line_index=idx,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                crop_path=out_path,
            )
        )

    logger.info(
        "Cropped %d line(s) from %s → %s",
        len(crops),
        source_page,
        lines_root,
    )
    return crops


def write_crops_manifest(path: Path, crops: list[LineCropSpec]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for crop in crops:
            writer.writerow(crop.to_manifest_row())
    return path


def load_crops_manifest(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def prepare_line_crops_from_pages(
    pages_root: Path,
    lines_root: Path,
    *,
    originals_root: Path | None,
    categories: tuple[str, ...],
) -> list[LineCropSpec]:
    """
    Walk category folders under pages_root, crop lines, archive originals.

    Does not invent GT. Caller should build a blank GT template for crops.
    """
    pages_root = Path(pages_root)
    lines_root = Path(lines_root)
    all_crops: list[LineCropSpec] = []

    for category in categories:
        cat_dir = pages_root / category
        if not cat_dir.is_dir():
            logger.warning("Category folder missing: %s", cat_dir)
            continue
        for path in sorted(cat_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            source_page = path.relative_to(pages_root).as_posix()
            crops = crop_page_lines(
                path,
                source_page=source_page,
                category=category,
                lines_root=lines_root,
                originals_root=originals_root,
            )
            all_crops.extend(crops)

    return all_crops
