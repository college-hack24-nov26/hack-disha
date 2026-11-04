"""Print/screenshot detector must not steal phone handwriting (CER killer)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.print_detect import looks_like_printed_text  # noqa: E402


class TestPrintDetect(unittest.TestCase):
    def test_flat_screenshot_like_is_printed(self) -> None:
        # Sharp high-contrast glyphs, regular heights, low color cast.
        img = np.full((120, 400, 3), 245, dtype=np.uint8)
        for x0 in range(20, 380, 28):
            img[40:70, x0 : x0 + 12] = 20
        self.assertTrue(looks_like_printed_text(img))

    def test_soft_phone_photo_like_is_not_printed(self) -> None:
        rng = np.random.default_rng(0)
        # Large soft canvas with mild RGB cast — WhatsApp notebook photo style.
        base = np.full((1280, 960, 3), 210, dtype=np.uint8)
        base[:, :, 0] = np.clip(base[:, :, 0].astype(np.int16) - 6, 0, 255).astype(np.uint8)
        noise = rng.integers(-8, 9, size=base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # A few irregular ink blobs
        img[200:260, 120:480] = 40
        img[400:430, 100:700] = 35
        self.assertFalse(looks_like_printed_text(img))


if __name__ == "__main__":
    unittest.main()
