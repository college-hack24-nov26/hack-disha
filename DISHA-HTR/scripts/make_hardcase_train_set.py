"""Build synthetic hard-case notebook lines for held-out failure modes.

Does NOT copy held-out image pixels. Renders phrase types that currently fail
(short greetings, ALL-CAPS names, place+PIN) onto lined paper using local fonts,
then applies phone-style noise. Safe to mix into domain fine-tuning.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.line_clean import strong_phone_aug  # noqa: E402

OUT = ROOT / "training" / "hardcase"
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\inkfree.ttf"),
    Path(r"C:\Windows\Fonts\segoesc.ttf"),
    Path(r"C:\Windows\Fonts\segoepr.ttf"),
    Path(r"C:\Windows\Fonts\comic.ttf"),
    Path(r"C:\Windows\Fonts\LHANDW.TTF"),
    Path(r"C:\Windows\Fonts\FRSCRIPT.TTF"),
    Path(r"C:\Windows\Fonts\MISTRAL.TTF"),
    Path(r"C:\Windows\Fonts\cour.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]

# Phrase bank aimed at held-out + unknown-doc CER failure modes
# (character substitutions inside words; style match, not image leak).
PHRASES = [
    "how are you",
    "How are you",
    "how are You",
    "How are You?",
    "how r you",
    "HEMANTH",
    "Hemanth",
    "SANKALP",
    "ASHWIN",
    "PRIYA",
    "KARAN",
    "ANANYA",
    "A B a",
    "A B C",
    "H I",
    "I Have started practicing",
    "I have started practicing",
    "started practicing",
    "happy when I am about",
    "DSA Questions I feel",
    "Karnataka 585418",
    "Bidar 585401",
    "Humnabad 585330",
    "Bangalore 560001",
    "Mysore 570001",
    "Gulbarga 585102",
    "I am Ghatboral Iq",
    "I am Humnabad Iq",
    "I am Bidar town",
    "my Name is hemanth Kumar",
    "my name IS HEMANTH kumar",
    "my Name is hemanth",
    "Can I know",
    "Hello we are pleased to",
    "meet you.",
    # Unknown-doc / WhatsApp notebook CER focus (GT == rendered text).
    "we are participating in hackathon competition",
    "we are participating",
    "hackathon competition",
    "Ghatboral",
    "Ghatboral Iq",
    "boral",
    "HEMANTH KUMAR",
    "H E M A N T H",
    "started",
    "practicing",
    "attended",
    "receipt",
    "offer",
]


def pick_fonts() -> list[Path]:
    found = [p for p in FONT_CANDIDATES if p.is_file()]
    if not found:
        raise SystemExit("No usable fonts found under C:\\Windows\\Fonts")
    return found


def render_line(text: str, font_path: Path, rng: random.Random) -> np.ndarray:
    font_size = rng.randint(36, 56)
    font = ImageFont.truetype(str(font_path), font_size)
    # Measure text
    tmp = Image.new("RGB", (8, 8), "white")
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 40, 28
    w = max(320, tw + pad_x * 2)
    h = max(96, th + pad_y * 2)
    paper = rng.randint(232, 248)
    img = Image.new("RGB", (w, h), (paper, paper, paper - 2))
    draw = ImageDraw.Draw(img)
    # Ruled lines
    gap = rng.randint(max(22, h // 3), max(24, h // 2))
    y0 = rng.randint(8, max(9, gap // 2))
    rule = rng.randint(160, 195)
    for y in range(y0, h, gap):
        draw.line((0, y, w, y), fill=(rule, rule + 8, rule + 15), width=1)
    # Margin
    if rng.random() < 0.7:
        draw.line((28, 0, 28, h), fill=(210, 120, 120), width=2)
    ink = rng.choice([(20, 20, 25), (15, 25, 40), (35, 20, 15), (10, 10, 10)])
    x = pad_x + rng.randint(-6, 10)
    y = (h - th) // 2 + rng.randint(-4, 4)
    draw.text((x, y), text, font=font, fill=ink)
    return np.array(img)


def main() -> int:
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    fonts = pick_fonts()
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    n_per = 4  # variants per phrase
    idx = 0
    for phrase in PHRASES:
        for _ in range(n_per):
            font = rng.choice(fonts)
            rgb = render_line(phrase, font, rng)
            rgb = strong_phone_aug(rgb, np_rng)
            name = f"hard_{idx:03d}.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_dir / name), bgr)
            split = "val" if idx % 9 == 0 else "train"
            rows.append(
                {
                    "image_filename": name,
                    "ground_truth": phrase,
                    "split": split,
                    "source": "synthetic_hardcase",
                }
            )
            idx += 1
    csv_path = OUT / "labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_filename", "ground_truth", "split", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} hard-case lines -> {OUT}")
    print(
        f"fonts_used={len(fonts)} "
        f"train={sum(1 for r in rows if r['split']=='train')} "
        f"val={sum(1 for r in rows if r['split']=='val')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
