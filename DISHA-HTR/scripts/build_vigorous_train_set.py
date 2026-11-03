"""
Build a large domain-matched training set for held-out phone-notebook lines.

Mixes:
  - Teklia/IAM-line export (clean handwriting foundation)
  - Local non-held-out notebook crops (domain)
  - Synthetic notebook-style composites of IAM onto ruled paper

Never includes held-out eval images or their exact source crop paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "vigorous"
OUT_IMG = OUT / "images"
CSV_PATH = OUT / "labels.csv"
META_PATH = OUT / "meta.json"

HELDOUT_SOURCES = {
    "evaluation/lines/non_cursive/NC02_L03.png",
    "evaluation/lines/non_cursive/NC02_L05.png",
    "evaluation/lines/non_cursive/NC02_L07.png",
    "evaluation/lines/non_cursive/NC02_L08.png",
    "evaluation/lines/non_cursive/NC02_L09.png",
    "evaluation/lines/non_cursive/NC05_L08.png",
    "evaluation/lines/non_cursive/NC01_L04.png",
    "evaluation/lines/non_cursive/NC01_L14.png",
}

# Verified local domain labels only (copied from prepare_improved_train_set; no invented GT).
LOCAL_LABELED: list[tuple[str, str, str]] = [
    ("demo/train_private/train_cu01_hello.png", "Hello we are pleased to", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_meet.png", "meet you.", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_topic.png", "Our innovation topic was", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_phase.png", "development phase of", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_hack.png", "our hackathon.", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_today.png", "today attended the", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_progress.png", "The overall progress of", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_project.png", "the project is going", "cursive/CU01.jpeg"),
    ("demo/train_private/train_cu01_well.png", "well.", "cursive/CU01.jpeg"),
    ("demo/train_private/train_nc01_hi.png", "H I", "non_cursive/NC01.jpeg"),
    ("evaluation/lines/non_cursive/NC01_L06.png", "my Name is hemanth Kumar", "non_cursive/NC01.jpeg"),
    ("evaluation/lines/non_cursive/NC01_L07.png", "my name IS HEMANTH kumar", "non_cursive/NC01.jpeg"),
    ("evaluation/lines/non_cursive/NC01_L10.png", "A B a", "non_cursive/NC01.jpeg"),
    ("evaluation/lines/non_cursive/NC01_L13.png", "what is your name", "non_cursive/NC01.jpeg"),
    ("evaluation/lines/non_cursive/NC01_L15.png", "Can I know", "non_cursive/NC01.jpeg"),
    ("evaluation/lines/non_cursive/NC02_L04.png", "Humnabad bidar dist", "non_cursive/NC02.jpeg"),
    ("evaluation/lines/non_cursive/NC02_L10.png", "Think about the solution.", "non_cursive/NC02.jpeg"),
    ("evaluation/lines/non_cursive/NC05_L03.png", "my Name is hemanth Kumar", "non_cursive/NC05.jpeg"),
    ("evaluation/lines/non_cursive/NC05_L04.png", "my Name IS HEMANTH Kumar", "non_cursive/NC05.jpeg"),
    ("evaluation/lines/non_cursive/NC05_L09.png", "b", "non_cursive/NC05.jpeg"),
    ("evaluation/lines/non_cursive/NC05_L11.png", "what is your name", "non_cursive/NC05.jpeg"),
    ("evaluation/lines/non_cursive/NC05_L14.png", "Can I know", "non_cursive/NC05.jpeg"),
    ("evaluation/lines/medical_vocabulary/MV01_L04.png", "Blood 20", "medical_vocabulary/MV01.jpeg"),
    ("evaluation/lines/medical_vocabulary/MV01_L06.png", "data mg pg", "medical_vocabulary/MV01.jpeg"),
    ("evaluation/lines/medical_vocabulary/MV01_L08.png", "kg", "medical_vocabulary/MV01.jpeg"),
]


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def make_ruled_bg(w: int, h: int, rng: random.Random) -> np.ndarray:
    paper = np.full((h, w, 3), rng.randint(230, 250), dtype=np.uint8)
    # slight paper texture
    noise = rng.randint(0, 8)
    if noise:
        jitter = np.random.default_rng(rng.randint(0, 10_000_000)).integers(
            -noise, noise + 1, size=paper.shape, dtype=np.int16
        )
        paper = np.clip(paper.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    gap = max(18, h // 3)
    y0 = rng.randint(4, max(5, gap // 2))
    for y in range(y0, h, gap):
        color = (rng.randint(150, 190),) * 3
        cv2.line(paper, (0, y), (w - 1, y), color, 1)
    # red margin
    if rng.random() < 0.7:
        x = rng.randint(8, max(9, w // 12))
        cv2.line(paper, (x, 0), (x, h - 1), (80, 80, 200), 1)
    return paper


def paste_ink_on_paper(ink_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Composite handwriting onto ruled notebook paper (held-out-like)."""
    h, w = ink_rgb.shape[:2]
    target_h = 128
    scale = target_h / max(h, 1)
    nw, nh = max(32, int(w * scale)), target_h
    ink = cv2.resize(ink_rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    pad_w = int(nw * rng.uniform(0.05, 0.25))
    canvas_w = nw + pad_w * 2
    bg = make_ruled_bg(canvas_w, target_h, rng)
    # ink mask: darker pixels
    gray = cv2.cvtColor(ink, cv2.COLOR_RGB2GRAY)
    mask = gray < rng.randint(150, 190)
    x0 = pad_w
    region = bg[:, x0 : x0 + nw]
    region[mask] = ink[mask]
    bg[:, x0 : x0 + nw] = region
    # white streak artifacts like phone capture
    if rng.random() < 0.55:
        for _ in range(rng.randint(1, 3)):
            yy = rng.randint(0, target_h - 1)
            cv2.line(bg, (0, yy), (canvas_w - 1, yy), (255, 255, 255), 1)
    # mild blur / jpeg
    if rng.random() < 0.35:
        bg = cv2.GaussianBlur(bg, (3, 3), 0)
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(bg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(55, 85)])
        if ok:
            bg = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # brightness uneven
    if rng.random() < 0.5:
        grad = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.7, 0.95), canvas_w, dtype=np.float32)
        bg = np.clip(bg.astype(np.float32) * grad[None, :, None], 0, 255).astype(np.uint8)
    return bg


def mild_geom(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    img = Image.fromarray(rgb)
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-3.0, 3.0), resample=Image.BICUBIC, fillcolor=(245, 245, 245))
    arr = np.asarray(img)
    if rng.random() < 0.3:
        alpha = rng.uniform(0.85, 1.2)
        arr = np.clip(arr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return arr


def save_sample(name: str, rgb: np.ndarray, text: str, source: str, split: str, rows: list) -> None:
    path = OUT_IMG / name
    Image.fromarray(rgb).save(path)
    rows.append(
        {
            "image_filename": name,
            "ground_truth": text,
            "source": source,
            "split": split,
            "sha16": sha16(path),
        }
    )


def load_iam_rows(limit: int) -> list[tuple[Path, str]]:
    csv_path = ROOT / "datasets" / "iam" / "hf_teklia" / "train" / "labels.csv"
    img_dir = csv_path.parent / "images"
    if not csv_path.is_file():
        return []
    out = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            p = img_dir / name
            if name and text and p.is_file():
                out.append((p, text))
            if len(out) >= limit:
                break
    return out


def load_iam_val(limit: int) -> list[tuple[Path, str]]:
    csv_path = ROOT / "datasets" / "iam" / "hf_teklia" / "validation" / "labels.csv"
    img_dir = csv_path.parent / "images"
    if not csv_path.is_file():
        return []
    out = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            p = img_dir / name
            if name and text and p.is_file():
                out.append((p, text))
            if len(out) >= limit:
                break
    return out


def main() -> int:
    rng = random.Random(42)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # 1) Local domain crops — heavy oversample (phone-notebook style is the gap)
    local_ok = 0
    for i, (rel, text, page) in enumerate(LOCAL_LABELED):
        key = rel.replace("\\", "/")
        if key in HELDOUT_SOURCES:
            raise RuntimeError(f"held-out leak: {rel}")
        path = ROOT / rel
        if not path.is_file():
            print("missing local", rel)
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # original-ish
        save_sample(f"local_{i:03d}_raw.png", mild_geom(rgb, rng), text, page, "train", rows)
        # notebook composite variants
        for v in range(8):
            comp = paste_ink_on_paper(rgb, rng)
            save_sample(
                f"local_{i:03d}_nb{v}.png",
                mild_geom(comp, rng),
                text,
                page,
                "train",
                rows,
            )
        local_ok += 1
    print("local sources used", local_ok)

    # 2) IAM foundation + notebook composites (use whatever local export exists)
    iam = load_iam_rows(100000)
    print("iam train available", len(iam))
    for i, (path, text) in enumerate(iam):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        # keep some clean IAM
        if i % 2 == 0:
            save_sample(f"iam_clean_{i:05d}.png", mild_geom(rgb, rng), text, "teklia_iam", "train", rows)
        # domain transfer — denser for available IAM size
        for v in range(2):
            comp = paste_ink_on_paper(rgb, rng)
            save_sample(
                f"iam_nb_{i:05d}_{v}.png",
                mild_geom(comp, rng),
                text,
                "teklia_iam_notebook",
                "train",
                rows,
            )

    # 3) Validation: IAM val notebook composites + a few local held-aside
    iam_val = load_iam_val(200)
    for i, (path, text) in enumerate(iam_val):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        comp = paste_ink_on_paper(rgb, rng)
        save_sample(
            f"val_iam_nb_{i:04d}.png",
            mild_geom(comp, rng),
            text,
            "teklia_iam_val",
            "val",
            rows,
        )

    # local val: last 4 local sources only once (not in heavy train oversample index conflict — use copies)
    # Already all locals in train; val uses IAM-notebook only to avoid tiny local val noise.

    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["image_filename", "ground_truth", "source", "split", "sha16"]
        )
        w.writeheader()
        w.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    meta = {
        "n_total": len(rows),
        "n_train": n_train,
        "n_val": n_val,
        "local_sources": local_ok,
        "iam_train_used": len(iam),
        "heldout_excluded": sorted(HELDOUT_SOURCES),
        "note": "Held-out eval images never included.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
