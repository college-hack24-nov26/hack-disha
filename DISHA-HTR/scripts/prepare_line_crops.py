"""
Prepare single-line crops from uploaded full-page handwriting images.

- Never modifies files under evaluation/images/ (originals stay intact).
- Archives a one-time copy under evaluation/pages/originals/ (no overwrite).
- Writes crops under evaluation/lines/{category}/ referencing the page name.
- Writes crops_manifest.csv + blank line_transcriptions.csv (GT never guessed).

Then fill line_transcriptions.csv and run:
  .\\.venv\\Scripts\\python.exe scripts\\run_evaluation.py --mode lines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.eval_data import (  # noqa: E402
    CATEGORIES,
    EvalImage,
    ensure_ground_truth_template,
)
from disha_htr.line_crop import (  # noqa: E402
    prepare_line_crops_from_pages,
    write_crops_manifest,
)
from disha_htr.logging_config import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop handwriting pages into single-line images for TrOCR eval"
    )
    parser.add_argument(
        "--pages-root",
        default=str(ROOT / "evaluation" / "images"),
        help="Folder of original page images (read-only; never overwritten)",
    )
    parser.add_argument(
        "--lines-root",
        default=str(ROOT / "evaluation" / "lines"),
        help="Output folder for line crops",
    )
    parser.add_argument(
        "--originals-root",
        default=str(ROOT / "evaluation" / "pages" / "originals"),
        help="Archive copies of originals (created only if missing)",
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "evaluation" / "lines" / "crops_manifest.csv"),
        help="Crop metadata CSV path",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(ROOT / "evaluation" / "ground_truth" / "line_transcriptions.csv"),
        help="Line-level GT template (blank GT; never invents text)",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Do not copy pages into originals archive",
    )
    args = parser.parse_args()

    import os

    os.chdir(ROOT)
    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR prepare line crops ===")

    pages_root = Path(args.pages_root)
    lines_root = Path(args.lines_root)
    originals_root = None if args.skip_archive else Path(args.originals_root)

    crops = prepare_line_crops_from_pages(
        pages_root,
        lines_root,
        originals_root=originals_root,
        categories=CATEGORIES,
    )
    write_crops_manifest(Path(args.manifest), crops)

    eval_images = [
        EvalImage(
            path=c.crop_path,
            image_filename=c.crop_filename,
            category=c.category,
            source_page=c.source_page,
        )
        for c in crops
    ]
    ensure_ground_truth_template(
        Path(args.ground_truth),
        eval_images,
        include_source_page=True,
    )

    print(f"Pages root (unchanged): {pages_root}")
    if originals_root is not None:
        print(f"Originals archive:     {originals_root}")
    print(f"Line crops:            {lines_root} ({len(crops)} crops)")
    print(f"Manifest:              {args.manifest}")
    print(f"Line GT template:      {args.ground_truth}")
    print(
        "Fill verified per-line ground_truth values, then run:\n"
        "  .\\.venv\\Scripts\\python.exe scripts\\run_evaluation.py --mode lines"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
