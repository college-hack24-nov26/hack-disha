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


if __name__ == "__main__":
    unittest.main()
