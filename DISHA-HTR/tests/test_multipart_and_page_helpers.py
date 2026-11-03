"""CPU tests for multipart parser and page-recognize helpers (no model load)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.line_crop import detect_line_boxes  # noqa: E402
from disha_htr.multipart_util import FormFile, parse_multipart  # noqa: E402
from disha_htr.page_recognize import (  # noqa: E402
    _looks_like_fluent_hallucination,
    _looks_like_letter_pairs,
)


class TestMultipart(unittest.TestCase):
    def test_parse_file_and_field(self) -> None:
        boundary = "----dishaBoundary7"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="doctor_id"\r\n\r\n'
            f"doc_demo\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="a.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + b"\x89PNG\r\n" + f"\r\n--{boundary}--\r\n".encode("utf-8")
        ctype = f"multipart/form-data; boundary={boundary}"
        parsed = parse_multipart(ctype, body)
        self.assertEqual(parsed["doctor_id"], "doc_demo")
        self.assertIsInstance(parsed["image"], FormFile)
        self.assertEqual(parsed["image"].filename, "a.png")
        self.assertTrue(parsed["image"].data.startswith(b"\x89PNG"))


class TestLineDetect(unittest.TestCase):
    def test_blank_image_no_boxes(self) -> None:
        img = np.full((200, 400, 3), 255, dtype=np.uint8)
        self.assertEqual(detect_line_boxes(img), [])

    def test_wide_strip_two_ink_bands(self) -> None:
        """Phone-style landscape crop with two handwriting rows."""
        import cv2

        img = np.full((160, 960, 3), 245, dtype=np.uint8)
        # Ruled lines (light blue-ish gray)
        for y in (30, 80, 130):
            img[y : y + 1, :] = (200, 180, 160)
        # Two dark handwriting bands
        img[35:70, 60:900] = (25, 25, 25)
        img[95:130, 60:900] = (20, 20, 20)
        boxes = detect_line_boxes(img)
        self.assertGreaterEqual(len(boxes), 2)
        self.assertLessEqual(len(boxes), 3)
        # Boxes should be vertically separated, not one full-page blob.
        heights = [b[3] - b[1] for b in boxes]
        self.assertTrue(all(h < 140 for h in heights))


class TestHallucinationGuards(unittest.TestCase):
    def test_polish_vocabulary_flagged(self) -> None:
        self.assertTrue(
            _looks_like_fluent_hallucination("The idea is a Polish vocabulary")
        )
        self.assertTrue(_looks_like_fluent_hallucination("The week is"))
        self.assertTrue(_looks_like_fluent_hallucination("medical vocabulary"))
        self.assertTrue(_looks_like_fluent_hallucination("I Had to come to"))

    def test_letter_pairs_kept(self) -> None:
        self.assertTrue(_looks_like_letter_pairs("Aa Bb Cc Dd Ee Ff"))
        self.assertFalse(
            _looks_like_fluent_hallucination("Aa Bb Cc Dd Ee Ff")
        )
        self.assertFalse(
            _looks_like_fluent_hallucination("Gg Hh Ii Jj Kk Ll")
        )
        self.assertFalse(
            _looks_like_fluent_hallucination("Htq So Ce Dy")
        )


if __name__ == "__main__":
    unittest.main()
