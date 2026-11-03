"""Heuristics to tell phone handwriting crops from clear printed / screenshot text."""

from __future__ import annotations

import cv2
import numpy as np


def looks_like_printed_text(image_bgr: np.ndarray) -> bool:
    """
    Return True when the image looks like sharp digital/print text
    (Google screenshot, PDF snippet, employment table) rather than pen strokes.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 20 or w < 40:
        return False

    # Digital screenshots are usually very sharp; phone handwriting is softer.
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float((bw > 0).mean())
    if ink_ratio < 0.008 or ink_ratio > 0.45:
        return False

    # Component height regularity — printed glyphs share similar heights.
    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    heights = []
    for i in range(1, n):
        _x, _y, bw_, bh, area = stats[i]
        if area < 12 or bh < 6:
            continue
        if bw_ > 0.85 * w and bh < 8:
            continue  # ruled line
        heights.append(bh)
    if len(heights) < 8:
        # Sparse but sharp digital text (short snippet) still counts as printed.
        return lap_var >= 180.0 and 0.01 <= ink_ratio <= 0.35

    heights_arr = np.array(heights, dtype=np.float32)
    med = float(np.median(heights_arr))
    if med <= 0:
        return False
    rel_std = float(heights_arr.std() / med)
    # Handwriting stroke heights vary a lot; print is tighter.
    regular = rel_std < 0.55
    sharp = lap_var >= 120.0
    return bool(sharp and regular and 0.01 <= ink_ratio <= 0.38)
