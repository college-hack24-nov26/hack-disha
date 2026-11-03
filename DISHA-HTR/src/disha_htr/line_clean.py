"""Paper-aware ruled-line cleanup for phone-notebook line crops.

Replaces long horizontal rulings with local paper colour (not pure white),
which avoids bright streak artifacts that confuse TrOCR.
"""

from __future__ import annotations

import cv2
import numpy as np


def paper_clean_line(bgr: np.ndarray, target_h: int = 128) -> np.ndarray:
    """Return RGB uint8 line image with rulings softened to paper tone."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Estimate paper tone from bright pixels (ignore ink).
    paper = float(np.percentile(gray, 70))
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    k = max(25, bgr.shape[1] // 15)
    horiz = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    )
    # Keep ink where both ink and "horiz" fire (thick stroke overlapping a rule).
    ink = thr > 0
    rule = horiz > 0
    rule_only = rule & ~cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
    cleaned = gray.copy()
    cleaned[rule_only] = np.uint8(np.clip(paper, 0, 255))
    # Mild contrast stretch.
    p2, p98 = np.percentile(cleaned, (2, 98))
    cleaned = np.clip((cleaned - p2) * 255.0 / max(p98 - p2, 1), 0, 255).astype(np.uint8)
    h, w = cleaned.shape
    scale = max(1.0, target_h / max(h, 1))
    cleaned = cv2.resize(
        cleaned, (max(32, int(w * scale)), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)


def _elastic_distort(rgb: np.ndarray, rng: np.random.Generator, alpha: float = 6.0, sigma: float = 4.0) -> np.ndarray:
    """Mild elastic warp (helps HTR generalization; keep alpha/sigma low)."""
    h, w = rgb.shape[:2]
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs + dx).astype(np.float32)
    map_y = (ys + dy).astype(np.float32)
    fill = int(np.percentile(rgb, 70))
    return cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(fill, fill, fill),
    )


def strong_phone_aug(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic phone-notebook capture noise for domain fine-tuning."""
    img = rgb.copy()
    h, w = img.shape[:2]
    # Slight rotation
    if rng.random() < 0.7:
        angle = float(rng.uniform(-3.5, 3.5))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(int(np.percentile(img, 70)),) * 3)
    # Elastic stroke jitter (research: strong for TrOCR HTR)
    if rng.random() < 0.45 and min(h, w) >= 32:
        img = _elastic_distort(
            img,
            rng,
            alpha=float(rng.uniform(3.0, 7.0)),
            sigma=float(rng.uniform(3.0, 5.5)),
        )
    # Ink thicken / thin
    if rng.random() < 0.25:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv = cv2.bitwise_not(gray)
        k = np.ones((2, 2), np.uint8)
        if rng.random() < 0.5:
            inv = cv2.dilate(inv, k, iterations=1)
        else:
            inv = cv2.erode(inv, k, iterations=1)
        img = cv2.cvtColor(cv2.bitwise_not(inv), cv2.COLOR_GRAY2RGB)
    # Brightness / contrast
    if rng.random() < 0.8:
        alpha = float(rng.uniform(0.75, 1.3))
        beta = float(rng.uniform(-25, 25))
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # Shadow gradient
    if rng.random() < 0.55:
        xs = np.linspace(0, 1, w, dtype=np.float32)
        if rng.random() < 0.5:
            grad = 0.65 + 0.35 * xs
        else:
            grad = 0.65 + 0.35 * (1 - xs)
        img = np.clip(img.astype(np.float32) * grad[None, :, None], 0, 255).astype(np.uint8)
    # White glare streaks (common in held-out phone photos)
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 4))):
            y = int(rng.integers(0, h))
            thickness = int(rng.integers(1, 3))
            cv2.line(img, (0, y), (w - 1, y), (255, 255, 255), thickness)
    # Ruled notebook lines
    if rng.random() < 0.45:
        gap = int(rng.integers(max(14, h // 4), max(15, h // 2)))
        y0 = int(rng.integers(2, max(3, gap // 2)))
        tone = int(rng.integers(150, 195))
        for y in range(y0, h, gap):
            cv2.line(img, (0, y), (w - 1, y), (tone, tone, tone), 1)
    # Mild blur / jpeg
    if rng.random() < 0.4:
        k = int(rng.choice([3, 3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.35:
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(35, 85))])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Horizontal crop jitter / pad
    if rng.random() < 0.4 and w > 64:
        cut = int(rng.integers(0, max(1, w // 30)))
        if rng.random() < 0.5:
            img = img[:, cut:]
        else:
            img = img[:, : max(32, w - cut)]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img
