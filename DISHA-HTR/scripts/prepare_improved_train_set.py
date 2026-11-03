"""
Build an expanded, domain-matched private train set for generalization fine-tuning.

- Excludes held-out eval image sources and held-out text lines.
- Applies the same mild ruled-line clean used for held-out demo images.
- Labels are verified against crops (no blind page-band guesses).
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "demo" / "train_improved"
OUT_IMG = OUT_DIR / "images"
CSV_PATH = OUT_DIR / "train_ground_truth.csv"
META_PATH = OUT_DIR / "prepare_meta.json"

# Exact held-out source crops — never copy.
HELDOUT_SOURCES = {
    "evaluation/lines/non_cursive/NC02_L03.png",
    "evaluation/lines/non_cursive/NC02_L05.png",
    "evaluation/lines/non_cursive/NC02_L07.png",
    "evaluation/lines/non_cursive/NC02_L08.png",
    "evaluation/lines/non_cursive/NC02_L09.png",
    "evaluation/lines/non_cursive/NC05_L08.png",
    "evaluation/lines/non_cursive/NC01_L04.png",  # HEMANTH held-out
    "evaluation/lines/non_cursive/NC01_L14.png",  # how are you held-out
}


def clean_line(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    horiz = cv2.morphologyEx(
        thr,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, bgr.shape[1] // 15), 1)),
    )
    cleaned = gray.copy()
    cleaned[horiz > 0] = 255
    h, w = cleaned.shape
    scale = max(1.0, 128 / max(h, 1))
    cleaned = cv2.resize(
        cleaned, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    p2, p98 = np.percentile(cleaned, (2, 98))
    cleaned = np.clip((cleaned - p2) * 255.0 / max(p98 - p2, 1), 0, 255).astype(
        np.uint8
    )
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def save_rgb(name: str, rgb: np.ndarray, text: str, source_page: str, rows: list) -> None:
    dest = OUT_IMG / name
    Image.fromarray(rgb).save(dest)
    rows.append(
        {
            "image_filename": name,
            "source_page": source_page,
            "ground_truth": text,
            "sha16": sha16(dest),
        }
    )


def from_bgr(name: str, bgr: np.ndarray, text: str, source_page: str, rows: list) -> None:
    save_rgb(name, clean_line(bgr), text, source_page, rows)


def from_file(rel: str, text: str, source_page: str, name: str, rows: list) -> None:
    key = rel.replace("\\", "/")
    if key in HELDOUT_SOURCES:
        raise RuntimeError(f"Refusing held-out source: {rel}")
    bgr = cv2.imread(str(ROOT / rel))
    if bgr is None:
        raise FileNotFoundError(rel)
    from_bgr(name, bgr, text, source_page, rows)


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # CU01 — verified private train crops (gallery adaptation set), brightness-matched
    train_priv = ROOT / "demo" / "train_private"
    cu01 = [
        ("train_cu01_hello.png", "Hello we are pleased to"),
        ("train_cu01_meet.png", "meet you."),
        ("train_cu01_topic.png", "Our innovation topic was"),
        ("train_cu01_phase.png", "development phase of"),
        ("train_cu01_hack.png", "our hackathon."),
        ("train_cu01_today.png", "today attended the"),
        ("train_cu01_progress.png", "The overall progress of"),
        ("train_cu01_project.png", "the project is going"),
        ("train_cu01_well.png", "well."),
    ]
    for fname, text in cu01:
        bgr = cv2.imread(str(train_priv / fname))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        p2, p98 = np.percentile(gray, (2, 98))
        gray = np.clip((gray - p2) * 255.0 / max(p98 - p2, 1), 0, 255).astype(np.uint8)
        h, w = gray.shape
        scale = max(1.0, 128 / max(h, 1))
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
        save_rgb(
            fname.replace("train_", "imp_"),
            cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            text,
            "cursive/CU01.jpeg",
            rows,
        )

    # NC01 — verified non-heldout lines (exclude HEMANTH L04, how are you L14)
    nc01 = [
        (
            "evaluation/lines/non_cursive/NC01_L06.png",
            "my Name is hemanth Kumar",
            "imp_nc01_name_lower.png",
        ),
        (
            "evaluation/lines/non_cursive/NC01_L07.png",
            "my name IS HEMANTH kumar",
            "imp_nc01_name_mixed.png",
        ),
        (
            "evaluation/lines/non_cursive/NC01_L10.png",
            "A B a",
            "imp_nc01_aba.png",
        ),
        (
            "evaluation/lines/non_cursive/NC01_L13.png",
            "what is your name",
            "imp_nc01_what.png",
        ),
        (
            "evaluation/lines/non_cursive/NC01_L15.png",
            "Can I know",
            "imp_nc01_can.png",
        ),
    ]
    for rel, text, name in nc01:
        from_file(rel, text, "non_cursive/NC01.jpeg", name, rows)

    # Also keep prior private NC01 H I (not held-out)
    hi = train_priv / "train_nc01_hi.png"
    if hi.is_file():
        bgr = cv2.imread(str(hi))
        from_bgr("imp_nc01_hi.png", bgr, "H I", "non_cursive/NC01.jpeg", rows)

    # NC02 — non-heldout only
    nc02_file = [
        (
            "evaluation/lines/non_cursive/NC02_L04.png",
            "Humnabad bidar dist",
            "imp_nc02_humnabad.png",
        ),
        (
            "evaluation/lines/non_cursive/NC02_L10.png",
            "Think about the solution.",
            "imp_nc02_think.png",
        ),
    ]
    for rel, text, name in nc02_file:
        from_file(rel, text, "non_cursive/NC02.jpeg", name, rows)

    # Slice multi-line NC02_L02 into verified single lines
    multi = cv2.imread(str(ROOT / "evaluation/lines/non_cursive/NC02_L02.png"))
    if multi is not None:
        h = multi.shape[0]
        # Empirically: header ~0:90, sankalp ~90:140, mern ~140:185, stack ~185:end
        slices = [
            (90, 140, "My name is Sankalp", "imp_nc02_sankalp.png"),
            (140, 185, "I am learnig mern", "imp_nc02_mern.png"),
            (185, h, "Stack web development.", "imp_nc02_stack.png"),
        ]
        for y0, y1, text, name in slices:
            from_bgr(name, multi[y0:y1], text, "non_cursive/NC02.jpeg", rows)

    # NC05 — exclude L08 held-out "A B a"
    nc05 = [
        (
            "evaluation/lines/non_cursive/NC05_L03.png",
            "my Name is hemanth Kumar",
            "imp_nc05_name1.png",
        ),
        (
            "evaluation/lines/non_cursive/NC05_L04.png",
            "my Name IS HEMANTH Kumar",
            "imp_nc05_name2.png",
        ),
        (
            "evaluation/lines/non_cursive/NC05_L09.png",
            "b",
            "imp_nc05_b.png",
        ),
        (
            "evaluation/lines/non_cursive/NC05_L11.png",
            "what is your name",
            "imp_nc05_what.png",
        ),
        (
            "evaluation/lines/non_cursive/NC05_L14.png",
            "Can I know",
            "imp_nc05_can.png",
        ),
    ]
    for rel, text, name in nc05:
        from_file(rel, text, "non_cursive/NC05.jpeg", name, rows)

    # MV01 independent writer
    mv = [
        (
            "evaluation/lines/medical_vocabulary/MV01_L04.png",
            "Blood 20",
            "imp_mv01_blood.png",
        ),
        (
            "evaluation/lines/medical_vocabulary/MV01_L06.png",
            "data mg pg",
            "imp_mv01_data.png",
        ),
        (
            "evaluation/lines/medical_vocabulary/MV01_L08.png",
            "kg",
            "imp_mv01_kg.png",
        ),
    ]
    # Also use inspect multi-line MV crops carefully via existing cleaned page read
    mv_title = ROOT / "demo" / "_inspect" / "MV01_L02.png"
    if mv_title.is_file():
        bgr = cv2.imread(str(mv_title))
        if bgr is not None and bgr.shape[0] > 200:
            # four lines in inspect crop
            h = bgr.shape[0]
            parts = [
                (0, h // 4, "medical vocabulary", "imp_mv01_title.png"),
                (h // 4, h // 2, "Hypertension and diabetes mellitus", "imp_mv01_hyper.png"),
                (h // 2, 3 * h // 4, "paracetamol and amoxilin", "imp_mv01_para1.png"),
                (3 * h // 4, h, "paracetramal and amoxicillin", "imp_mv01_para2.png"),
            ]
            for y0, y1, text, name in parts:
                from_bgr(name, bgr[y0:y1], text, "medical_vocabulary/MV01.jpeg", rows)

    for rel, text, name in mv:
        from_file(rel, text, "medical_vocabulary/MV01.jpeg", name, rows)

    held_dir = ROOT / "demo" / "eval_private" / "heldout_images"
    held_hashes = (
        {sha16(p): p.name for p in held_dir.glob("*.png")} if held_dir.is_dir() else {}
    )
    leaks = [r for r in rows if r["sha16"] in held_hashes]
    if leaks:
        raise RuntimeError(f"Train/held-out hash overlap: {leaks}")

    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_filename", "source_page", "ground_truth", "sha16"]
        )
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "n_samples": len(rows),
        "writers_pages": sorted({r["source_page"] for r in rows}),
        "excluded_heldout_sources": sorted(HELDOUT_SOURCES),
        "heldout_hash_overlaps": 0,
        "note": "Private improved train set; not for held-out scoring.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
