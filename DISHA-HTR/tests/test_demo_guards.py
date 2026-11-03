"""CPU tests for demo gallery restrictions and GT isolation from UI module."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestDemoGuards(unittest.TestCase):
    def test_approved_manifest_only_lists_files_in_approved_dir(self) -> None:
        manifest = ROOT / "demo" / "approved_manifest.json"
        approved = ROOT / "demo" / "approved"
        self.assertTrue(manifest.is_file())
        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data["images"]), 1)
        for row in data["images"]:
            path = approved / row["filename"]
            self.assertTrue(path.is_file(), msg=row["filename"])
            # no path traversal
            self.assertEqual(path.resolve().parent, approved.resolve())

    def test_private_gt_exists_and_is_outside_approved(self) -> None:
        gt = ROOT / "demo" / "eval_private" / "ground_truth.csv"
        self.assertTrue(gt.is_file())
        text = gt.read_text(encoding="utf-8")
        self.assertIn("ground_truth", text)
        # approved manifest must not embed GT strings column
        manifest = (ROOT / "demo" / "approved_manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("ground_truth", manifest)

    def test_run_demo_source_does_not_read_eval_private(self) -> None:
        src = (ROOT / "scripts" / "run_demo.py").read_text(encoding="utf-8")
        # Static checks: UI script must not open private GT
        self.assertNotIn("ground_truth.csv", src)
        self.assertIn('ROOT / "demo" / "approved"', src)
        tree = ast.parse(src)
        dumped = ast.dump(tree)
        self.assertNotIn("ground_truth.csv", dumped)
        # FORBIDDEN_GT_DIRS may mention eval_private, but must not open GT files
        self.assertIn("FORBIDDEN_GT_DIRS", src)

    def test_invalid_image_outside_approved_rejected_by_relative_check(self) -> None:
        approved = (ROOT / "demo" / "approved").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            outsider = Path(tmp) / "evil.png"
            outsider.write_bytes(b"x")
            with self.assertRaises(ValueError):
                outsider.resolve().relative_to(approved)


if __name__ == "__main__":
    unittest.main()
