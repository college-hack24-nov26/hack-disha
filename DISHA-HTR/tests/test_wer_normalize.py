"""WER-oriented decode cleanup."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.metrics import word_error_rate  # noqa: E402
from disha_htr.wer_normalize import join_spaced_letter_runs, wer_normalize_hypothesis  # noqa: E402


class TestWerNormalize(unittest.TestCase):
    def test_joins_spaced_name(self) -> None:
        self.assertEqual(join_spaced_letter_runs("H E M A N T H"), "HEMANTH")
        self.assertEqual(wer_normalize_hypothesis("H E M A N T H"), "HEMANTH")

    def test_leaves_normal_phrase(self) -> None:
        phrase = "how are you"
        self.assertEqual(wer_normalize_hypothesis(phrase), phrase)

    def test_leaves_alphabet_drill(self) -> None:
        self.assertEqual(wer_normalize_hypothesis("A B a"), "A B a")

    def test_wer_improves_on_spaced_name(self) -> None:
        gt = "HEMANTH"
        raw = "H E M A N T H"
        fixed = wer_normalize_hypothesis(raw)
        self.assertLess(word_error_rate(gt, fixed), word_error_rate(gt, raw))
        self.assertEqual(word_error_rate(gt, fixed), 0.0)


if __name__ == "__main__":
    unittest.main()
