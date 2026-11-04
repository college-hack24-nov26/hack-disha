"""
Lightweight tests for metrics + GT CSV handling.

Does NOT load TrOCR / GPU / torch models.
Run:
  .\\.venv\\Scripts\\python.exe -m unittest tests.test_metrics_and_csv -v
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.eval_data import (  # noqa: E402
    EvalImage,
    ensure_ground_truth_template,
    has_usable_ground_truth,
    load_ground_truth_csv,
)
from disha_htr.metrics import (  # noqa: E402
    aggregate_scores,
    character_error_rate,
    exact_match,
    levenshtein_distance,
    pair_error_rates,
    word_error_rate,
)


class TestMetrics(unittest.TestCase):
    def test_levenshtein_basic(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)

    def test_cer_exact_and_errors(self) -> None:
        self.assertEqual(character_error_rate("hello", "hello"), 0.0)
        self.assertGreater(character_error_rate("hello", "hallo"), 0.0)
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "x"), 1.0)

    def test_wer_and_exact_match(self) -> None:
        self.assertEqual(word_error_rate("a b c", "a b c"), 0.0)
        self.assertEqual(word_error_rate("a b c", "a x c"), 1 / 3)
        self.assertTrue(exact_match("  Foo  Bar ", "Foo Bar"))
        self.assertFalse(exact_match("Foo", "Bar"))

    def test_aggregate_empty_is_none_not_zero(self) -> None:
        empty = aggregate_scores([])
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["cer_mean"])
        self.assertIsNone(empty["wer_mean"])
        self.assertIsNone(empty["char_accuracy_mean"])
        self.assertIsNone(empty["exact_match_accuracy"])

    def test_aggregate_nonzero(self) -> None:
        scored = aggregate_scores([("abc", "abc"), ("abc", "abd")])
        self.assertEqual(scored["count"], 2)
        self.assertIsNotNone(scored["cer_mean"])
        self.assertIsNotNone(scored["char_accuracy_mean"])
        self.assertLess(scored["exact_match_accuracy"], 1.0)

    def test_pair_error_rates_cer_primary(self) -> None:
        rates = pair_error_rates("my Name is hemanth", "MAY NONE 4 REWARDMENTS")
        self.assertGreater(rates["cer"], rates["wer"])
        self.assertTrue(rates["cer_worse_than_wer"])
        self.assertIn("char_accuracy", rates)


class TestGroundTruthCsv(unittest.TestCase):
    def test_missing_gt_never_usable_as_empty(self) -> None:
        rows = load_ground_truth_csv(Path("definitely_missing_gt_file.csv"))
        self.assertEqual(rows, {})
        self.assertFalse(has_usable_ground_truth(None))

    def test_blank_gt_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcriptions.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["image_filename", "category", "ground_truth"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_filename": "non_cursive/a.png",
                        "category": "non_cursive",
                        "ground_truth": "",
                    }
                )
                writer.writerow(
                    {
                        "image_filename": "cursive/b.png",
                        "category": "cursive",
                        "ground_truth": "   ",
                    }
                )
                writer.writerow(
                    {
                        "image_filename": "cursive/c.png",
                        "category": "cursive",
                        "ground_truth": "hello",
                    }
                )
            loaded = load_ground_truth_csv(path)
            self.assertFalse(has_usable_ground_truth(loaded["non_cursive/a.png"]))
            self.assertFalse(has_usable_ground_truth(loaded["cursive/b.png"]))
            self.assertTrue(has_usable_ground_truth(loaded["cursive/c.png"]))
            self.assertEqual(loaded["cursive/c.png"].ground_truth, "hello")

    def test_template_does_not_overwrite_existing_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "transcriptions.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["image_filename", "category", "ground_truth"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_filename": "non_cursive/NC01.jpeg",
                        "category": "non_cursive",
                        "ground_truth": "KEEP ME",
                    }
                )

            # Fake on-disk images (paths need not exist for template merge)
            images = [
                EvalImage(
                    path=tmp_path / "non_cursive" / "NC01.jpeg",
                    image_filename="non_cursive/NC01.jpeg",
                    category="non_cursive",
                ),
                EvalImage(
                    path=tmp_path / "cursive" / "CU01.jpeg",
                    image_filename="cursive/CU01.jpeg",
                    category="cursive",
                ),
            ]
            ensure_ground_truth_template(csv_path, images)
            loaded = load_ground_truth_csv(csv_path)
            self.assertEqual(loaded["non_cursive/NC01.jpeg"].ground_truth, "KEEP ME")
            self.assertIsNone(loaded["cursive/CU01.jpeg"].ground_truth)
            self.assertFalse(has_usable_ground_truth(loaded["cursive/CU01.jpeg"]))


if __name__ == "__main__":
    unittest.main()
