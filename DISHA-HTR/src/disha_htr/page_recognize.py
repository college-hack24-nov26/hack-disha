"""Full-page recognition via line segmentation + single-line TrOCR.

Original page bytes are never modified. Segmentation can fail on dense or
noisy pages — callers should surface that limitation to users.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .line_crop import detect_line_boxes
from .model_loader import LoadedModel
from .recognize import recognize_image

logger = logging.getLogger("disha_htr.page_recognize")


def _ink_ratio(bgr: np.ndarray, threshold: int = 140) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float((gray < threshold).mean())


def _looks_like_collapse(text: str) -> bool:
    """Detect common TrOCR collapse modes (zeros / alphabet dumps)."""
    t = (text or "").strip().lower()
    if not t:
        return True
    compact = t.replace(" ", "")
    if compact.count("0") >= max(8, int(0.6 * len(compact))):
        return True
    if compact in {"abcdefghijklmnopqrstuvwxyz", "abcdefghij", "abcdefghijklm"}:
        return True
    if "000000" in compact:
        return True
    return False


@dataclass
class LinePrediction:
    line_index: int
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    inference_time_s: float
    status: str
    error: str = ""


@dataclass
class PageRecognitionResult:
    mode: str  # single_line | page_lines | page_fallback_single
    combined_text: str
    lines: list[LinePrediction]
    total_inference_time_s: float
    segmentation_note: str
    n_lines_detected: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "combined_text": self.combined_text,
            "lines": [asdict(x) for x in self.lines],
            "total_inference_time_s": self.total_inference_time_s,
            "segmentation_note": self.segmentation_note,
            "n_lines_detected": self.n_lines_detected,
        }


def recognize_page_or_line(
    image_path: str | Path,
    loaded: LoadedModel,
    *,
    force_single_line: bool = False,
    max_lines: int = 40,
) -> PageRecognitionResult:
    """
    Recognize handwriting from a file path.

    If the image looks like a tall page and line boxes are found, recognize
    each crop in reading order. Otherwise run single-line recognition on the
    whole image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")

    h, w = bgr.shape[:2]
    aspect = h / max(w, 1)

    if force_single_line or aspect < 1.15:
        t0 = time.perf_counter()
        result = recognize_image(path, loaded)
        elapsed = time.perf_counter() - t0
        text = result.text or ""
        status = "ok"
        note = "Treated as a single-line/crop image (no page split)."
        combined = text
        if _looks_like_collapse(text):
            status = "low_confidence"
            note += " Model returned low-confidence/collapse text; draft left empty for review."
            combined = ""
        line = LinePrediction(
            line_index=1,
            x0=0,
            y0=0,
            x1=w,
            y1=h,
            text=text,
            inference_time_s=round(result.inference_time_s, 4),
            status=status,
            error="collapse_pattern" if status == "low_confidence" else "",
        )
        return PageRecognitionResult(
            mode="single_line",
            combined_text=combined,
            lines=[line],
            total_inference_time_s=round(elapsed, 4),
            segmentation_note=note,
            n_lines_detected=1,
        )

    boxes = detect_line_boxes(bgr)
    if not boxes:
        t0 = time.perf_counter()
        result = recognize_image(path, loaded)
        elapsed = time.perf_counter() - t0
        line = LinePrediction(
            line_index=1,
            x0=0,
            y0=0,
            x1=w,
            y1=h,
            text=result.text,
            inference_time_s=round(result.inference_time_s, 4),
            status="ok",
        )
        return PageRecognitionResult(
            mode="page_fallback_single",
            combined_text=result.text,
            lines=[line],
            total_inference_time_s=round(elapsed, 4),
            segmentation_note=(
                "Full-page line segmentation found no reliable lines; "
                "fell back to whole-image recognition (may be unreliable)."
            ),
            n_lines_detected=0,
        )

    boxes = boxes[:max_lines]
    preds: list[LinePrediction] = []
    texts: list[str] = []
    total = 0.0
    with tempfile.TemporaryDirectory(prefix="disha_page_lines_") as tmp:
        tmp_path = Path(tmp)
        for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
            crop = bgr[y0:y1, x0:x1]
            if crop.size == 0:
                preds.append(
                    LinePrediction(
                        line_index=idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text="",
                        inference_time_s=0.0,
                        status="error",
                        error="empty_crop",
                    )
                )
                continue
            ink = _ink_ratio(crop)
            if ink < 0.035:
                preds.append(
                    LinePrediction(
                        line_index=idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text="",
                        inference_time_s=0.0,
                        status="skipped_low_ink",
                        error=f"ink_ratio={ink:.4f}",
                    )
                )
                continue
            # White pad helps TrOCR on tight notebook crops.
            pad = max(4, int(0.08 * max(crop.shape[0], 1)))
            crop = cv2.copyMakeBorder(
                crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            crop_file = tmp_path / f"line_{idx:02d}.png"
            if not cv2.imwrite(str(crop_file), crop):
                preds.append(
                    LinePrediction(
                        line_index=idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text="",
                        inference_time_s=0.0,
                        status="error",
                        error="write_failed",
                    )
                )
                continue
            try:
                result = recognize_image(crop_file, loaded)
                total += result.inference_time_s
                text = result.text or ""
                status = "ok"
                err = ""
                if _looks_like_collapse(text):
                    status = "low_confidence"
                    err = "collapse_pattern"
                    # Keep text visible for debugging, but do not promote into combined draft.
                    text_for_combine = ""
                else:
                    text_for_combine = text
                preds.append(
                    LinePrediction(
                        line_index=idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text=text,
                        inference_time_s=round(result.inference_time_s, 4),
                        status=status,
                        error=err,
                    )
                )
                if text_for_combine:
                    texts.append(text_for_combine)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Line %s recognition failed", idx)
                preds.append(
                    LinePrediction(
                        line_index=idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text="",
                        inference_time_s=0.0,
                        status="error",
                        error=str(exc)[:200],
                    )
                )

    n_ok = sum(1 for p in preds if p.status == "ok")
    n_low = sum(1 for p in preds if p.status == "low_confidence")
    n_skip = sum(1 for p in preds if p.status.startswith("skipped"))
    return PageRecognitionResult(
        mode="page_lines",
        combined_text="\n".join(texts),
        lines=preds,
        total_inference_time_s=round(total, 4),
        segmentation_note=(
            f"Segmented into {len(boxes)} line crop(s); kept {n_ok} readable, "
            f"{n_low} low-confidence, {n_skip} skipped (low ink). "
            "Draft only — doctor review required. Notebook pages remain hard for the base model."
        ),
        n_lines_detected=len(boxes),
    )
