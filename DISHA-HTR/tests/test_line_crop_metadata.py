"""
CPU-only tests for line-crop naming/metadata and eval_data handling.

Does NOT load TrOCR / GPU models.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.eval_data import (  # noqa: E402
    EvalImage,
    discover_line_crop_images,
    ensure_ground_truth_template,
    has_usable_ground_truth,
    load_ground_truth_csv,
)
from disha_htr.line_crop import (  # noqa: E402
    crop_filename_for,
    crop_page_lines,
    load_crops_manifest,
    preserve_original,
    write_crops_manifest,
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class TestCropNaming(unittest.TestCase):
    def test_crop_filename_references_source_page_stem(self) -> None:
        name = crop_filename_for("non_cursive/NC01.jpeg", 2)
        self.assertEqual(name, "non_cursive/NC01_L02.png")
        self.assertIn("NC01", name)
        self.assertTrue(name.endswith("_L02.png"))


class TestPreserveOriginal(unittest.TestCase):
    def test_original_bytes_unchanged_and_archive_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            page = tmp_path / "page.jpeg"
            # Minimal valid JPEG via OpenCV
            import cv2

            img = np.full((40, 120, 3), 240, dtype=np.uint8)
            cv2.imwrite(str(page), img)
            before = _file_sha256(page)

            originals = tmp_path / "originals"
            archived = preserve_original(page, originals, "non_cursive/NC01.jpeg")
            self.assertTrue(archived.is_file())
            self.assertEqual(_file_sha256(page), before)

            # Second archive must not overwrite existing archive content
            archived.write_bytes(b"sentinel-should-remain")
            preserve_original(page, originals, "non_cursive/NC01.jpeg")
            self.assertEqual(archived.read_bytes(), b"sentinel-should-remain")
            self.assertEqual(_file_sha256(page), before)


class TestCropAndManifest(unittest.TestCase):
    def test_crops_reference_source_and_blank_gt_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            import cv2

            # Synthetic page: two dark horizontal ink bands on light paper
            page = np.full((200, 300, 3), 245, dtype=np.uint8)
            page[40:55, 20:280] = (20, 20, 20)
            page[110:130, 30:270] = (15, 15, 15)
            page_path = tmp_path / "NC99.jpeg"
            cv2.imwrite(str(page_path), page)

            lines_root = tmp_path / "lines"
            originals = tmp_path / "originals"
            crops = crop_page_lines(
                page_path,
                source_page="non_cursive/NC99.jpeg",
                category="non_cursive",
                lines_root=lines_root,
                originals_root=originals,
            )
            self.assertGreaterEqual(len(crops), 1)
            for crop in crops:
                self.assertEqual(crop.source_page, "non_cursive/NC99.jpeg")
                self.assertIn("NC99_L", crop.crop_filename)
                self.assertTrue(crop.crop_path.is_file())
                self.assertTrue(
                    (originals / "non_cursive" / "NC99.jpeg").is_file()
                )

            manifest = tmp_path / "crops_manifest.csv"
            write_crops_manifest(manifest, crops)
            rows = load_crops_manifest(manifest)
            self.assertEqual(len(rows), len(crops))
            self.assertEqual(rows[0]["source_page"], "non_cursive/NC99.jpeg")

            eval_images = [
                EvalImage(
                    path=c.crop_path,
                    image_filename=c.crop_filename,
                    category=c.category,
                    source_page=c.source_page,
                )
                for c in crops
            ]
            gt_path = tmp_path / "line_transcriptions.csv"
            ensure_ground_truth_template(
                gt_path, eval_images, include_source_page=True
            )
            loaded = load_ground_truth_csv(gt_path)
            for c in crops:
                self.assertIn(c.crop_filename, loaded)
                self.assertFalse(has_usable_ground_truth(loaded[c.crop_filename]))
                self.assertEqual(
                    loaded[c.crop_filename].source_page, "non_cursive/NC99.jpeg"
                )

            # Discover with manifest attaches source_page
            discovered = discover_line_crop_images(lines_root, manifest_path=manifest)
            self.assertEqual(len(discovered), len(crops))
            self.assertEqual(discovered[0].source_page, "non_cursive/NC99.jpeg")


class TestGtNeverGuessed(unittest.TestCase):
    def test_template_does_not_fill_gt_from_page_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path = tmp_path / "line_transcriptions.csv"
            images = [
                EvalImage(
                    path=tmp_path / "non_cursive" / "NC01_L01.png",
                    image_filename="non_cursive/NC01_L01.png",
                    category="non_cursive",
                    source_page="non_cursive/NC01.jpeg",
                )
            ]
            ensure_ground_truth_template(
                gt_path, images, include_source_page=True
            )
            loaded = load_ground_truth_csv(gt_path)
            self.assertIsNone(loaded["non_cursive/NC01_L01.png"].ground_truth)


if __name__ == "__main__":
    unittest.main()
