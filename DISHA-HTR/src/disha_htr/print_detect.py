"""Heuristics to tell phone handwriting crops from clear printed / screenshot text."""

from __future__ import annotations

import cv2
import numpy as np


def looks_like_printed_text(image_bgr: np.ndarray) -> bool:
    """
    Return True when the image looks like sharp digital/print text
    (Google screenshot, PDF snippet, employment table) rather than pen strokes.

    Conservative by design: WhatsApp / phone photos of handwriting must stay on
    the handwriting checkpoint — misrouting them to printed TrOCR destroys CER.
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
    if ink_ratio < 0.008 or ink_ratio > 0.42:
        return False

    # Phone photos of lined notebooks: large canvas + mild blur + color cast.
    # Screenshots are flat RGB with little channel disagreement on paper areas.
    b, g, r = cv2.split(image_bgr)
    channel_spread = float(
        np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
        + np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
    ) / 2.0
    phone_photo_like = (
        max(h, w) >= 900
        and lap_var < 900.0
        and channel_spread >= 2.5
    )
    if phone_photo_like:
        return False

    # Component height regularity — printed glyphs share similar heights.
    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    heights: list[float] = []
    elongations: list[float] = []
    for i in range(1, n):
        _x, _y, bw_, bh, area = stats[i]
        if area < 12 or bh < 6:
            continue
        if bw_ > 0.85 * w and bh < 8:
            continue  # ruled line
        heights.append(float(bh))
        elongations.append(float(max(bw_, bh) / max(min(bw_, bh), 1)))

    if len(heights) < 8:
        # Sparse but sharp digital text (short snippet) still counts as printed.
        # Raise the bar so soft phone crops do not slip through.
        return lap_var >= 420.0 and 0.012 <= ink_ratio <= 0.32 and channel_spread < 6.0

    heights_arr = np.array(heights, dtype=np.float32)
    med = float(np.median(heights_arr))
    if med <= 0:
        return False
    rel_std = float(heights_arr.std() / med)
    # Handwriting stroke heights vary a lot; print is tighter.
    regular = rel_std < 0.42
    sharp = lap_var >= 220.0
    # Pen strokes are often more elongated / irregular than screenshot glyphs.
    mean_elon = float(np.mean(elongations)) if elongations else 1.0
    stroke_like = mean_elon >= 3.8 and rel_std >= 0.28
    if stroke_like and lap_var < 1500.0:
        return False
    return bool(sharp and regular and 0.012 <= ink_ratio <= 0.36)
