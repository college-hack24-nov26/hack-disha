"""
Private demo evaluation (NOT used by the UI).

Reads demo/approved images + demo/eval_private/ground_truth.csv,
writes CER / char accuracy / WER / word accuracy / exact-match.
Never invents GT. Uses local checkpoint only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.logging_config import setup_logging  # noqa: E402
from disha_htr.metrics import (  # noqa: E402
    aggregate_scores,
    character_error_rate,
    exact_match,
    word_error_rate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Private DISHA-HTR demo evaluation")
    parser.add_argument(
        "--approved-dir",
        default=str(ROOT / "demo" / "eval_private" / "heldout_images"),
        help="Image folder to score (default: held-out set, not UI gallery)",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(ROOT / "demo" / "eval_private" / "ground_truth.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "demo" / "results"),
    )
    parser.add_argument(
        "--label",
        default="heldout",
        help="Tag for output filenames (heldout|adaptation)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint dir (default: fine-tuned if present else base)",
    )
    args = parser.parse_args()

    import os

    os.chdir(ROOT)
    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR private demo eval ===")

    approved = Path(args.approved_dir)
    gt_path = Path(args.ground_truth)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_path.is_file():
        print(f"Missing private GT: {gt_path}", file=sys.stderr)
        return 2

    gt_map: dict[str, str] = {}
    with gt_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            if name and text:
                gt_map[name] = text

    from disha_htr.hardware import detect_hardware, resolve_device
    from disha_htr.model_loader import load_trocr
    from disha_htr.recognize import recognize_image

    hw = detect_hardware(preference=args.device)
    finetuned = ROOT / "checkpoints" / "trocr-demo-finetuned"
    base = ROOT / "checkpoints" / "trocr-small-handwritten"
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif (finetuned / "model.safetensors").is_file() or (
        finetuned / "pytorch_model.bin"
    ).is_file():
        ckpt = finetuned
    else:
        ckpt = base

    loaded = load_trocr(
        local_dir=ckpt,
        cache_dir=ROOT / ".cache" / "huggingface",
        device=resolve_device(preference=args.device),
    )

    rows = []
    pairs = []
    for name, gt in sorted(gt_map.items()):
        img = approved / name
        row = {
            "image_filename": name,
            "ground_truth": gt,
            "prediction": "",
            "cer": "",
            "char_accuracy_pct": "",
            "wer": "",
            "word_accuracy_pct": "",
            "exact_match": "",
            "inference_time_s": "",
            "status": "",
            "error": "",
        }
        if not img.is_file():
            row["status"] = "missing_image"
            row["error"] = str(img)
            rows.append(row)
            continue
        try:
            result = recognize_image(img, loaded)
            pred = result.text
            cer = character_error_rate(gt, pred)
            wer = word_error_rate(gt, pred)
            row.update(
                {
                    "prediction": pred,
                    "cer": round(cer, 6),
                    "char_accuracy_pct": round(max(0.0, (1.0 - cer) * 100.0), 2),
                    "wer": round(wer, 6),
                    "word_accuracy_pct": round(max(0.0, (1.0 - wer) * 100.0), 2),
                    "exact_match": exact_match(gt, pred),
                    "inference_time_s": result.inference_time_s,
                    "status": "ok",
                }
            )
            pairs.append((gt, pred))
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        logger.info("%s status=%s pred=%r", name, row["status"], row["prediction"])

    overall = aggregate_scores(pairs)
    mean_cer = overall["cer_mean"]
    char_acc = None if mean_cer is None else round(max(0.0, (1.0 - mean_cer) * 100.0), 2)
    mean_wer = overall["wer_mean"]
    word_acc = None if mean_wer is None else round(max(0.0, (1.0 - mean_wer) * 100.0), 2)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt),
        "device": hw.to_dict(),
        "n_scored": overall["count"],
        "cer_mean": mean_cer,
        "char_accuracy_pct_mean": char_acc,
        "wer_mean": mean_wer,
        "word_accuracy_pct_mean": word_acc,
        "exact_match_accuracy": overall["exact_match_accuracy"],
        "target_char_accuracy_pct": 80.0,
        "meets_80_percent_char_accuracy": bool(char_acc is not None and char_acc >= 80.0),
        "diagnosis_if_below_target": (
            None
            if char_acc is not None and char_acc >= 80.0
                    else [
                "Held-out phone-photo notebook lines differ from the small demo adaptation set.",
                "Demo fine-tune (encoder frozen, 14 private lines) improves adapted gallery lines but does not generalize to held-out lines.",
                "Safe fixes tried: single-line crops (originals preserved), ruled-line whitening, upscale, contrast; short local fine-tune.",
                "Reaching >=80% on held-out real photos likely needs larger licensed data (e.g. IAM after registration) + fuller training.",
            ]
        ),
        "eval_label": args.label,
        "per_image": rows,
    }

    preds_csv = out_dir / f"demo_eval_{args.label}_predictions.csv"
    with preds_csv.open("w", encoding="utf-8", newline="") as fh:
        fields = list(rows[0].keys()) if rows else [
            "image_filename",
            "ground_truth",
            "prediction",
            "status",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = out_dir / f"demo_eval_{args.label}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        f"DISHA-HTR Demo Private Evaluation ({args.label})",
        "=================================",
        f"Checkpoint: {ckpt}",
        f"Scored images: {overall['count']}",
        f"CER (mean): {mean_cer}",
        f"Char accuracy % (mean): {char_acc}",
        f"WER (mean): {mean_wer}",
        f"Word accuracy % (mean): {word_acc}",
        f"Exact-match accuracy: {overall['exact_match_accuracy']}",
        f"Meets 80% char accuracy: {summary['meets_80_percent_char_accuracy']}",
        "",
    ]
    if summary["diagnosis_if_below_target"]:
        report.append("Diagnosis:")
        for line in summary["diagnosis_if_below_target"]:
            report.append(f"  - {line}")
    report_path = out_dir / f"demo_eval_{args.label}_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"Wrote {preds_csv}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
