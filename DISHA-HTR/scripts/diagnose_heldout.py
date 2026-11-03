"""Diagnose held-out failures: labels, leakage, base vs fine-tuned, CER sanity."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.metrics import character_error_rate, exact_match, word_error_rate


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    import os

    os.chdir(ROOT)
    held_dir = ROOT / "demo" / "eval_private" / "heldout_images"
    gt_path = ROOT / "demo" / "eval_private" / "ground_truth.csv"
    train_csv = ROOT / "demo" / "train_private" / "train_ground_truth.csv"
    train_dir = ROOT / "demo" / "train_private"
    approved = ROOT / "demo" / "approved"

    report: dict = {"checks": [], "pairs": [], "leakage": {}, "base_vs_ft": []}

    gt_rows = list(csv.DictReader(gt_path.open(encoding="utf-8-sig", newline="")))
    for row in gt_rows:
        img = held_dir / row["image_filename"]
        ok = img.is_file()
        report["pairs"].append(
            {
                "image": row["image_filename"],
                "exists": ok,
                "gt": row["ground_truth"],
                "gt_len": len(row["ground_truth"]),
                "sha": sha(img) if ok else None,
                "size": Image_size(img) if ok else None,
            }
        )
        report["checks"].append(
            f"{'OK' if ok else 'MISSING'} {row['image_filename']} gt={row['ground_truth']!r}"
        )

    # CER sanity
    assert abs(character_error_rate("abc", "abc")) < 1e-9
    assert character_error_rate("abc", "ab") > 0
    report["checks"].append("CER implementation sanity: OK")

    # Leakage: hash overlap train/approved vs heldout
    held_hashes = {p["sha"]: p["image"] for p in report["pairs"] if p["sha"]}
    leak_hits = []
    for folder in (train_dir, approved):
        if not folder.is_dir():
            continue
        for p in folder.glob("*.png"):
            h = sha(p)
            if h in held_hashes:
                leak_hits.append({"file": str(p), "matches_heldout": held_hashes[h]})
    report["leakage"]["hash_overlaps"] = leak_hits
    report["checks"].append(
        f"Image hash leakage train/approved vs heldout: {len(leak_hits)} overlaps"
    )

    # Writer/page overlap
    train_pages = set()
    if train_csv.is_file():
        for row in csv.DictReader(train_csv.open(encoding="utf-8-sig", newline="")):
            train_pages.add(row.get("source_page") or "")
    held_pages = {r["source_page"] for r in gt_rows}
    report["leakage"]["shared_source_pages"] = sorted(train_pages & held_pages)
    report["checks"].append(
        f"Shared source pages train∩heldout: {report['leakage']['shared_source_pages']}"
    )

    # Base vs fine-tuned predictions
    from disha_htr.hardware import resolve_device
    from disha_htr.model_loader import load_trocr
    from disha_htr.recognize import recognize_image

    device = resolve_device("auto")
    base = load_trocr(
        local_dir=ROOT / "checkpoints" / "trocr-small-handwritten",
        cache_dir=ROOT / ".cache" / "huggingface",
        device=device,
    )
    ft_dir = ROOT / "checkpoints" / "trocr-demo-finetuned"
    ft = None
    if (ft_dir / "model.safetensors").is_file():
        ft = load_trocr(local_dir=ft_dir, cache_dir=ROOT / ".cache" / "huggingface", device=device)

    # Token config check (VisionEncoderDecoderConfig may nest decoder_start)
    def _tok_cfg(loaded):
        cfg = loaded.model.config
        dec = getattr(cfg, "decoder", None)
        gen = getattr(loaded.model, "generation_config", None)
        return {
            "config_decoder_start": getattr(cfg, "decoder_start_token_id", None),
            "nested_decoder_start": getattr(dec, "decoder_start_token_id", None) if dec else None,
            "gen_decoder_start": getattr(gen, "decoder_start_token_id", None) if gen else None,
            "config_eos": getattr(cfg, "eos_token_id", None),
            "nested_eos": getattr(dec, "eos_token_id", None) if dec else None,
            "config_pad": getattr(cfg, "pad_token_id", None),
            "tok_cls": loaded.processor.tokenizer.cls_token_id,
            "tok_sep": loaded.processor.tokenizer.sep_token_id,
            "tok_pad": loaded.processor.tokenizer.pad_token_id,
        }

    report["token_config"] = {"base": _tok_cfg(base)}
    if ft is not None:
        report["token_config"]["ft"] = _tok_cfg(ft)

    for row in gt_rows:
        img = held_dir / row["image_filename"]
        gt = row["ground_truth"]
        rb = recognize_image(img, base)
        entry = {
            "image": row["image_filename"],
            "gt": gt,
            "base_pred": rb.text,
            "base_cer": round(character_error_rate(gt, rb.text), 4),
            "base_exact": exact_match(gt, rb.text),
        }
        if ft is not None:
            rf = recognize_image(img, ft)
            entry.update(
                {
                    "ft_pred": rf.text,
                    "ft_cer": round(character_error_rate(gt, rf.text), 4),
                    "ft_exact": exact_match(gt, rf.text),
                }
            )
        report["base_vs_ft"].append(entry)

    out = ROOT / "demo" / "results" / "heldout_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("checks", "token_config", "leakage")}, indent=2))
    print(f"Wrote {out}")
    return 0


def Image_size(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return list(im.size)


if __name__ == "__main__":
    raise SystemExit(main())
