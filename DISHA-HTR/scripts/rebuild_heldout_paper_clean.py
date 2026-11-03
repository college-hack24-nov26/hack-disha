"""Rebuild held-out eval crops with paper-aware cleaning (no GT changes)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.line_clean import paper_clean_line  # noqa: E402

HELDOUT = ROOT / "demo" / "eval_private" / "heldout_images"
GT = ROOT / "demo" / "eval_private" / "ground_truth.csv"

SPECS = [
    ("line01_ghatboral.png", "evaluation/lines/non_cursive/NC02_L03.png"),
    ("line02_karnataka.png", "evaluation/lines/non_cursive/NC02_L05.png"),
    ("line03_dsa.png", "evaluation/lines/non_cursive/NC02_L08.png"),
    ("line04_happy.png", "evaluation/lines/non_cursive/NC02_L09.png"),
    ("line05_aba.png", "evaluation/lines/non_cursive/NC05_L08.png"),
    ("line06_practicing.png", "evaluation/lines/non_cursive/NC02_L07.png"),
    ("line07_hemanth.png", "evaluation/lines/non_cursive/NC01_L04.png"),
    ("line08_howareyou.png", "evaluation/lines/non_cursive/NC01_L14.png"),
]


def main() -> int:
    HELDOUT.mkdir(parents=True, exist_ok=True)
    # Keep GT file as-is; only rewrite images.
    assert GT.is_file(), GT
    for fname, src in SPECS:
        bgr = cv2.imread(str(ROOT / src))
        if bgr is None:
            print("missing", src)
            return 2
        rgb = paper_clean_line(bgr)
        Image.fromarray(rgb).save(HELDOUT / fname)
        print("wrote", fname, rgb.shape)
    print("held-out images rebuilt with paper_clean_line; GT unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
