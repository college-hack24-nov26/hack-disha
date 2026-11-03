"""Create a synthetic single-line sample image (no real patient data)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic HTR sample image")
    parser.add_argument(
        "--out",
        default="samples/synthetic_line.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--text",
        default="Patient feels better today",
        help="Synthetic phrase (demo only — not real PHI)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Off-white paper-like background
    width, height = 900, 128
    img = Image.new("RGB", (width, height), color=(245, 242, 235))
    draw = ImageDraw.Draw(img)

    # Prefer a non-default TTF if present; fall back to PIL default
    font = None
    for candidate in (
        r"C:\Windows\Fonts\segoepr.ttf",  # Segoe Print — closer to handwriting
        r"C:\Windows\Fonts\comic.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if Path(candidate).is_file():
            try:
                font = ImageFont.truetype(candidate, 42)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()

    # Slight vertical offset to mimic a written line
    draw.text((40, 36), args.text, fill=(25, 35, 55), font=font)
    # Faint baseline
    draw.line((30, 100, width - 30, 102), fill=(200, 200, 210), width=1)

    img.save(out)
    print(f"Wrote {out.resolve()}")
    print(f"Synthetic text: {args.text!r}")


if __name__ == "__main__":
    main()
