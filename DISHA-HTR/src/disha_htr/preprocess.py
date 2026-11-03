"""Image preprocessing for single-line handwriting recognition."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("disha_htr.preprocess")


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB PIL Image. Raises FileNotFoundError / ValueError."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # Prefer OpenCV for robust decode, then convert to RGB PIL
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        # Fallback to Pillow
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not decode image: {path}") from exc
        logger.debug("Loaded via Pillow: %s size=%s", path, img.size)
        return img

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    logger.debug("Loaded via OpenCV: %s size=%s", path, img.size)
    return img


def prepare_for_trocr(image: Image.Image) -> Image.Image:
    """
    Light single-line prep: ensure RGB, optional mild contrast stretch.

    No aggressive deskew/line-seg yet — kept simple for baseline TrOCR.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.asarray(image)
    # Mild contrast normalization if nearly washed out
    if arr.std() < 15:
        logger.info("Low-contrast image detected; applying CLAHE on L channel")
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        arr = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(arr)

    return image
