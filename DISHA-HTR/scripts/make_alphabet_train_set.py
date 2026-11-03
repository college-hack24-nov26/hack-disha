"""Create synthetic alphabet-pair notebook lines for domain adaptation."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "alphabet_pairs"
IMG = OUT / "images"
CSV_PATH = OUT / "labels.csv"


def ruled_bg(w: int, h: int, rng: random.Random) -> Image.Image:
    paper = np.full((h, w, 3), rng.randint(235, 248), dtype=np.uint8)
    gap = max(22, h // 2)
    for y in range(rng.randint(8, 16), h, gap):
        color = (rng.randint(160, 195),) * 3
        cv2.line(paper, (0, y), (w - 1, y), color, 1)
    if rng.random() < 0.8:
        x = rng.randint(12, 36)
        cv2.line(paper, (x, 0), (x, h - 1), (80, 80, 200), 1)
    return Image.fromarray(paper)


def pick_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\segoepr.ttf",
        r"C:\Windows\Fonts\segoeprb.ttf",
        r"C:\Windows\Fonts\comic.ttf",
        r"C:\Windows\Fonts\comicbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def line_text(rng: random.Random) -> str:
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    start = rng.randint(0, 20)
    n = rng.randint(4, 8)
    chunk = letters[start : start + n]
    if len(chunk) < 4:
        chunk = letters[:6]
    return " ".join(f"{ch}{ch.lower()}" for ch in chunk)


def main() -> int:
    rng = random.Random(42)
    IMG.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(120):
        text = line_text(rng)
        w = rng.randint(900, 1400)
        h = rng.randint(96, 140)
        img = ruled_bg(w, h, rng)
        draw = ImageDraw.Draw(img)
        font = pick_font(rng.randint(34, 54))
        # Slight baseline jitter per glyph cluster
        x = rng.randint(40, 70)
        y = rng.randint(18, max(20, h // 3))
        for token in text.split(" "):
            dy = rng.randint(-3, 3)
            draw.text((x, y + dy), token, fill=(20, 20, 20), font=font)
            bbox = draw.textbbox((x, y + dy), token, font=font)
            x = bbox[2] + rng.randint(18, 34)
        # Mild phone noise
        arr = np.array(img)
        if rng.random() < 0.4:
            arr = cv2.GaussianBlur(arr, (3, 3), 0)
        if rng.random() < 0.5:
            alpha = rng.uniform(0.85, 1.15)
            arr = np.clip(arr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        name = f"alpha_{i:03d}.png"
        Image.fromarray(arr).save(IMG / name)
        rows.append({"image_filename": name, "ground_truth": text, "split": "train"})

    # Small val set
    for i in range(12):
        text = line_text(rng)
        w, h = 1100, 120
        img = ruled_bg(w, h, rng)
        draw = ImageDraw.Draw(img)
        font = pick_font(42)
        draw.text((50, 30), text, fill=(15, 15, 15), font=font)
        name = f"alpha_val_{i:02d}.png"
        img.save(IMG / name)
        rows.append({"image_filename": name, "ground_truth": text, "split": "val"})

    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_filename", "ground_truth", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
