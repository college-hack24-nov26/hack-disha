"""
Run DISHA-HTR automated evaluation over evaluation/images/* categories.

Reuses load_trocr + recognize_image (model loaded once). Does not train or
modify weights. Does not alter source images or baseline inference modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.eval_data import (  # noqa: E402
    CATEGORIES,
    discover_eval_images,
    discover_line_crop_images,
    ensure_ground_truth_template,
    has_usable_ground_truth,
    load_ground_truth_csv,
)
from disha_htr.logging_config import setup_logging  # noqa: E402
from disha_htr.metrics import (  # noqa: E402
    aggregate_scores,
    character_error_rate,
    exact_match,
    word_error_rate,
)
# Torch / TrOCR imports are deferred until a full eval run (not --template-only).


PREDICTION_FIELDS = [
    "image_filename",
    "category",
    "source_page",
    "ground_truth",
    "prediction",
    "inference_time_s",
    "status",
    "error",
    "included_in_metrics",
    "cer",
    "wer",
    "exact_match",
]


def _write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PREDICTION_FIELDS})


def _build_metrics_payload(
    prediction_rows: list[dict[str, Any]],
    hardware: dict[str, Any],
    model_meta: dict[str, Any],
) -> dict[str, Any]:
    scored: list[tuple[str, str]] = []
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)

    n_total = len(prediction_rows)
    n_ok = 0
    n_failed = 0
    n_missing_gt = 0
    n_scored = 0

    for row in prediction_rows:
        status = row["status"]
        if status == "ok":
            n_ok += 1
        elif status == "failed":
            n_failed += 1

        if row["included_in_metrics"]:
            n_scored += 1
            pair = (str(row["ground_truth"]), str(row["prediction"]))
            scored.append(pair)
            by_category[str(row["category"])].append(pair)
        elif status == "ok" and not row["included_in_metrics"]:
            n_missing_gt += 1

    overall = aggregate_scores(scored)
    category_metrics: dict[str, Any] = {}
    for cat in CATEGORIES:
        category_metrics[cat] = aggregate_scores(by_category.get(cat, []))

    # Also include any unexpected category keys
    for cat, pairs in by_category.items():
        if cat not in category_metrics:
            category_metrics[cat] = aggregate_scores(pairs)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "images_total": n_total,
            "inference_ok": n_ok,
            "inference_failed": n_failed,
            "missing_or_blank_ground_truth": n_missing_gt,
            "included_in_accuracy_metrics": n_scored,
        },
        "overall": overall,
        "by_category": category_metrics,
        "hardware": hardware,
        "model": model_meta,
        "notes": [
            "Missing/blank ground_truth rows are excluded from CER/WER/exact-match.",
            "Missing GT is never treated as an empty-string reference.",
            "Failed inferences are recorded but excluded from accuracy metrics.",
        ],
    }


def _format_report(metrics: dict[str, Any], predictions_path: Path) -> str:
    counts = metrics["counts"]
    overall = metrics["overall"]
    lines = [
        "DISHA-HTR Evaluation Report",
        "===========================",
        f"Generated (UTC): {metrics['generated_at_utc']}",
        f"Predictions CSV: {predictions_path}",
        "",
        "Counts",
        "------",
        f"  Images total:              {counts['images_total']}",
        f"  Inference OK:              {counts['inference_ok']}",
        f"  Inference failed:          {counts['inference_failed']}",
        f"  Missing/blank GT:          {counts['missing_or_blank_ground_truth']}",
        f"  Included in metrics:       {counts['included_in_accuracy_metrics']}",
        "",
        "Overall (scored images only)",
        "----------------------------",
    ]
    if overall["count"] == 0:
        lines.append("  No scored images - accuracy metrics not computed.")
    else:
        lines.extend(
            [
                f"  N:                       {overall['count']}",
                f"  CER (mean):              {overall['cer_mean']}",
                f"  WER (mean):              {overall['wer_mean']}",
                f"  Exact-match accuracy:    {overall['exact_match_accuracy']}",
            ]
        )

    lines.extend(["", "By category", "-----------"])
    for cat, block in metrics["by_category"].items():
        if block["count"] == 0:
            lines.append(f"  {cat}: no scored images")
        else:
            lines.append(
                f"  {cat}: n={block['count']}  CER={block['cer_mean']}  "
                f"WER={block['wer_mean']}  exact={block['exact_match_accuracy']}"
            )

    model = metrics.get("model") or {}
    hw = metrics.get("hardware") or {}
    lines.extend(
        [
            "",
            "Runtime",
            "-------",
            f"  Device: {hw.get('selected_device')}  GPU: {hw.get('gpu_name')}",
            f"  Model:  {model.get('name')}  source={model.get('source')}",
            f"  FP16:   {model.get('use_fp16')}  load_time_s={model.get('load_time_s')}",
            "",
            "Notes",
            "-----",
        ]
    )
    for note in metrics.get("notes") or []:
        lines.append(f"  - {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="DISHA-HTR automated evaluation")
    parser.add_argument(
        "--mode",
        choices=["pages", "lines"],
        default="pages",
        help="pages=full images under evaluation/images; lines=crops under evaluation/lines",
    )
    parser.add_argument(
        "--images-root",
        default=None,
        help="Root containing category subfolders (default depends on --mode)",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Ground-truth CSV path (created/extended if missing; never overwrites GT text)",
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "evaluation" / "lines" / "crops_manifest.csv"),
        help="Line-crop manifest (used when --mode lines)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output directory for predictions.csv, metrics.json, evaluation_report.txt",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--local-dir",
        default=str(ROOT / "checkpoints" / "trocr-small-handwritten"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / ".cache" / "huggingface"),
    )
    parser.add_argument(
        "--model",
        default="microsoft/trocr-small-handwritten",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="Force FP16 (CUDA only)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 even on CUDA",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Discover images + write/extend GT template only (no model load)",
    )
    args = parser.parse_args()

    import os

    os.chdir(ROOT)
    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR evaluation (mode=%s) ===", args.mode)

    if args.mode == "lines":
        images_root = Path(
            args.images_root or (ROOT / "evaluation" / "lines")
        )
        gt_path = Path(
            args.ground_truth
            or (ROOT / "evaluation" / "ground_truth" / "line_transcriptions.csv")
        )
        results_dir = Path(
            args.results_dir or (ROOT / "evaluation" / "results" / "lines")
        )
        images = discover_line_crop_images(
            images_root, manifest_path=Path(args.manifest)
        )
        ensure_ground_truth_template(gt_path, images, include_source_page=True)
    else:
        images_root = Path(
            args.images_root or (ROOT / "evaluation" / "images")
        )
        gt_path = Path(
            args.ground_truth
            or (ROOT / "evaluation" / "ground_truth" / "transcriptions.csv")
        )
        results_dir = Path(args.results_dir or (ROOT / "evaluation" / "results"))
        images = discover_eval_images(images_root)
        ensure_ground_truth_template(gt_path, images)

    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Discovered %d evaluation image(s) under %s", len(images), images_root)

    gt_map = load_ground_truth_csv(gt_path)

    if args.template_only:
        print(f"Ground-truth template ready: {gt_path}")
        print(f"Images discovered: {len(images)}")
        print("Fill the ground_truth column, then re-run without --template-only.")
        return 0

    if not images:
        print(
            f"No images found under {images_root}. "
            "Place files in non_cursive/, cursive/, medical_vocabulary/, prescription_style/.",
            file=sys.stderr,
        )
        # Still write empty artifacts for a clear no-op run
        empty_rows: list[dict[str, Any]] = []
        preds_path = results_dir / "predictions.csv"
        metrics_path = results_dir / "metrics.json"
        report_path = results_dir / "evaluation_report.txt"
        _write_predictions_csv(preds_path, empty_rows)
        metrics = _build_metrics_payload(empty_rows, {}, {})
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(_format_report(metrics, preds_path), encoding="utf-8")
        print(_format_report(metrics, preds_path))
        return 2

    from disha_htr.hardware import detect_hardware, resolve_device
    from disha_htr.model_loader import load_trocr
    from disha_htr.recognize import recognize_image

    hw = detect_hardware(preference=args.device, require_cuda=False)
    device = resolve_device(preference=args.device, require_cuda=False)
    loaded = load_trocr(
        model_name=args.model,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        device=device,
        use_fp16=args.fp16,
    )
    model_meta = {
        "name": loaded.name,
        "source": loaded.source,
        "device": str(loaded.device),
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "use_fp16": loaded.use_fp16,
        "load_time_s": round(loaded.load_time_s, 4),
    }

    prediction_rows: list[dict[str, Any]] = []
    for img in images:
        gt_row = gt_map.get(img.image_filename)
        usable_gt = has_usable_ground_truth(gt_row)
        gt_text = gt_row.ground_truth if usable_gt else None
        source_page = ""
        if gt_row is not None and gt_row.source_page:
            source_page = gt_row.source_page
        elif img.source_page:
            source_page = img.source_page

        row: dict[str, Any] = {
            "image_filename": img.image_filename,
            "category": img.category,
            "source_page": source_page,
            "ground_truth": gt_text if gt_text is not None else "",
            "prediction": "",
            "inference_time_s": "",
            "status": "",
            "error": "",
            "included_in_metrics": False,
            "cer": "",
            "wer": "",
            "exact_match": "",
        }

        try:
            result = recognize_image(img.path, loaded)
            row["prediction"] = result.text
            row["inference_time_s"] = result.inference_time_s
            row["status"] = "ok"
            if usable_gt and gt_text is not None:
                row["included_in_metrics"] = True
                row["cer"] = round(character_error_rate(gt_text, result.text), 6)
                row["wer"] = round(word_error_rate(gt_text, result.text), 6)
                row["exact_match"] = exact_match(gt_text, result.text)
            else:
                # Explicit: missing GT never scores as empty-string reference
                row["included_in_metrics"] = False
        except Exception as exc:  # noqa: BLE001 — record and continue
            logger.exception("Recognition failed for %s", img.image_filename)
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["included_in_metrics"] = False

        prediction_rows.append(row)
        logger.info(
            "Eval %s status=%s included=%s pred=%r",
            img.image_filename,
            row["status"],
            row["included_in_metrics"],
            row["prediction"],
        )

    preds_path = results_dir / "predictions.csv"
    metrics_path = results_dir / "metrics.json"
    report_path = results_dir / "evaluation_report.txt"

    _write_predictions_csv(preds_path, prediction_rows)
    metrics = _build_metrics_payload(prediction_rows, hw.to_dict(), model_meta)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = _format_report(metrics, preds_path)
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote: {preds_path}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {report_path}")
    logger.info("Evaluation completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)

