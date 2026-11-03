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
from .print_detect import looks_like_printed_text
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


_COMMON_ENGLISH = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "and",
        "in",
        "on",
        "for",
        "with",
        "this",
        "that",
        "idea",
        "week",
        "polish",
        "vocabulary",
        "medical",
        "handwriting",
        "rejection",
        "newly",
        "blood",
        "infection",
        "feet",
        "well",
    }
)


def _looks_like_letter_pairs(text: str) -> bool:
    """True for alphabet-drill style tokens like 'Aa Bb Cc' / 'Gg Hh Ii'."""
    words = [w for w in (text or "").replace(",", " ").split() if w]
    if len(words) < 3:
        return False
    pairish = 0
    for w in words:
        if w.lower() in _COMMON_ENGLISH:
            continue
        if 1 <= len(w) <= 2 and w.isalpha():
            pairish += 1
    return pairish >= max(3, int(0.6 * len(words)))


def _looks_like_fluent_hallucination(text: str) -> bool:
    """Heuristic: fluent English words that are unlikely for a handwriting crop."""
    t = (text or "").strip()
    if not t:
        return False
    if _looks_like_letter_pairs(t):
        return False
    words = [w for w in t.replace(",", " ").replace(".", " ").replace("'", " ").split() if w]
    if not words:
        return False
    lower = [w.lower() for w in words]
    # Alphabet drills: many non-English 1–2 letter tokens (Aa, Bb…) — keep those.
    letterish_short = sum(
        1 for w in words if 1 <= len(w) <= 2 and w.isalpha() and w.lower() not in _COMMON_ENGLISH
    )
    if letterish_short >= max(3, len(words) // 2):
        return False
    common_hits = sum(1 for w in lower if w in _COMMON_ENGLISH)
    alpha = sum(ch.isalpha() or ch.isspace() for ch in t) / max(len(t), 1)
    # Classic TrOCR collapse: full English sentence on letter/line crops.
    if len(t) >= 18 and len(words) >= 4 and alpha > 0.85:
        return True
    # Short but clearly English phrases ("The week is", "medical vocabulary").
    if len(words) >= 2 and common_hits >= 2 and alpha > 0.8:
        return True
    if len(words) >= 3 and common_hits >= 2 and alpha > 0.85:
        return True
    return False


def _sanitize_line_text(
    text: str,
    *,
    allow_fluent_english: bool = False,
) -> tuple[str, str, str]:
    """Return (display_text, combine_text, error_code)."""
    raw = text or ""
    if _looks_like_collapse(raw):
        return raw, "", "collapse_pattern"
    # Printed / screenshot English is expected to be fluent — do not wipe it.
    if not allow_fluent_english and _looks_like_fluent_hallucination(raw):
        return raw, "", "fluent_hallucination"
    return raw, raw, ""


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
    content_type: str = "handwriting"  # handwriting | printed

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "combined_text": self.combined_text,
            "lines": [asdict(x) for x in self.lines],
            "total_inference_time_s": self.total_inference_time_s,
            "segmentation_note": self.segmentation_note,
            "n_lines_detected": self.n_lines_detected,
            "content_type": self.content_type,
        }


def recognize_page_or_line(
    image_path: str | Path,
    loaded: LoadedModel,
    *,
    force_single_line: bool = False,
    force_printed: bool = False,
    auto_printed: bool = True,
    max_lines: int = 40,
    max_new_tokens: int | None = None,
) -> PageRecognitionResult:
    """
    Recognize handwriting or clear printed/screenshot English from a file path.

    If the image looks like a tall page and line boxes are found, recognize
    each crop in reading order. Otherwise run single-line recognition on the
    whole image.

    Printed / screenshot content uses fluent English without wiping it as a
    hallucination (handwriting path still guards against invented sentences).
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")

    h, w = bgr.shape[:2]
    printed = bool(force_printed) or (auto_printed and looks_like_printed_text(bgr))
    allow_fluent = printed
    tokens = max_new_tokens if max_new_tokens is not None else (128 if printed else 64)
    beams = 5 if printed else 4
    content_type = "printed" if printed else "handwriting"

    # Only skip page split when the caller explicitly asks for single-line mode.
    # Wide phone photos of notebooks are often landscape (aspect < 1) but still
    # contain multiple handwriting lines — do NOT force single-line from aspect.
    if force_single_line:
        t0 = time.perf_counter()
        result = recognize_image(path, loaded, max_new_tokens=tokens, num_beams=beams)
        elapsed = time.perf_counter() - t0
        text, combined, err = _sanitize_line_text(
            result.text or "", allow_fluent_english=allow_fluent
        )
        status = "ok" if not err else "low_confidence"
        note = (
            "Printed/screenshot single-crop recognition."
            if printed
            else "Single-line mode: recognized the whole image as one crop (no page split)."
        )
        if err:
            note += (
                " Model returned unlikely/hallucinated text for this crop; "
                "draft cleared for clinician review. Try page mode (uncheck single-line) "
                "or a tighter line crop."
            )
        line = LinePrediction(
            line_index=1,
            x0=0,
            y0=0,
            x1=w,
            y1=h,
            text=text,
            inference_time_s=round(result.inference_time_s, 4),
            status=status,
            error=err,
        )
        return PageRecognitionResult(
            mode="single_line",
            combined_text=combined,
            lines=[line],
            total_inference_time_s=round(elapsed, 4),
            segmentation_note=note,
            n_lines_detected=1,
            content_type=content_type,
        )

    boxes = detect_line_boxes(bgr)
    # Keep only reasonably tall bands (filters ruled-paper crumbs on phone photos).
    # Wide short strips already use a high min height inside detect_line_boxes;
    # keep this floor mild so real 2-line crops are not wiped out.
    aspect = h / max(w, 1)
    min_box_h = max(14, int((0.12 if aspect < 0.35 else 0.035) * h))
    if printed:
        # Screenshots often have shorter glyph bands than notebook lines.
        min_box_h = max(10, int(0.018 * h))
    boxes = [b for b in boxes if (b[3] - b[1]) >= min_box_h]
    if not boxes:
        t0 = time.perf_counter()
        result = recognize_image(path, loaded, max_new_tokens=tokens, num_beams=beams)
        elapsed = time.perf_counter() - t0
        text, combined, err = _sanitize_line_text(
            result.text or "", allow_fluent_english=allow_fluent
        )
        status = "ok" if not err else "low_confidence"
        note = (
            "Full-page line segmentation found no reliable lines; "
            "fell back to whole-image recognition (may be unreliable)."
        )
        if printed:
            note = "Printed/screenshot fallback: whole-image recognition."
        if err:
            note += (
                " Fallback text looked hallucinated and was cleared from the draft."
            )
        line = LinePrediction(
            line_index=1,
            x0=0,
            y0=0,
            x1=w,
            y1=h,
            text=text,
            inference_time_s=round(result.inference_time_s, 4),
            status=status,
            error=err,
        )
        return PageRecognitionResult(
            mode="page_fallback_single",
            combined_text=combined,
            lines=[line],
            total_inference_time_s=round(elapsed, 4),
            segmentation_note=note,
            n_lines_detected=0,
            content_type=content_type,
        )

    line_cap = max(max_lines, 40 if printed else max_lines)
    boxes = boxes[:line_cap]
    preds: list[LinePrediction] = []
    texts: list[str] = []
    total = 0.0
    ink_floor = 0.004 if printed else 0.012
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
            # Letter practice / light pencil can be sparse — keep threshold low.
            if ink < ink_floor:
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
                result = recognize_image(
                    crop_file, loaded, max_new_tokens=tokens, num_beams=beams
                )
                total += result.inference_time_s
                text, text_for_combine, err = _sanitize_line_text(
                    result.text or "", allow_fluent_english=allow_fluent
                )
                status = "ok" if not err else "low_confidence"
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
    combined = "\n".join(texts)
    kind = "printed/screenshot" if printed else "handwriting"
    note = (
        f"{kind.capitalize()} mode · segmented into {len(boxes)} line crop(s); "
        f"kept {n_ok} readable, {n_low} low-confidence, {n_skip} skipped (low ink). "
        "Draft only — clinician review required."
    )
    # If most lines look hallucinated, do not promote a single stray "ok" line.
    # Skip this wipe for printed screenshots — fluent English is expected.
    if not printed and (n_ok == 0 or (n_low > 0 and n_low >= n_ok)):
        combined = ""
        note += (
            " Most line reads looked unreliable (model hallucination). "
            "Alphabet drills / letter grids are especially hard for the handwriting "
            "model — for clear Google/PDF text, enable Printed/screenshot mode. "
            "Or open Technical details for per-line attempts."
        )
    return PageRecognitionResult(
        mode="page_lines",
        combined_text=combined,
        lines=preds,
        total_inference_time_s=round(total, 4),
        segmentation_note=note,
        n_lines_detected=len(boxes),
        content_type=content_type,
    )
