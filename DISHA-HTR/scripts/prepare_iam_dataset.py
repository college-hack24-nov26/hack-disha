"""
Extract + validate + build writer-independent IAM manifests (Phase 1).

- Leaves datasets/iam/raw archives unchanged (extracts into extracted/).
- Does not invent transcriptions.
- Does not touch evaluation/ samples or GT.
- Idempotent: re-running refreshes manifests deterministically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.iam_dataset import (  # noqa: E402
    build_split_line_ids,
    compute_stats,
    find_ascii_dir,
    find_lines_root,
    pair_lines,
    parse_forms_txt,
    parse_lines_txt,
    safe_extract_tar,
    safe_extract_zip,
    summarize_writers_for_report,
    write_manifest_csv,
    write_stats_json,
)
from disha_htr.logging_config import setup_logging  # noqa: E402


def _extract_if_needed(archive: Path, dest: Path, marker_name: str) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / marker_name
    if marker.is_file():
        return "skipped"
    if archive.suffix == ".zip" or archive.name.endswith(".zip"):
        safe_extract_zip(archive, dest)
    else:
        safe_extract_tar(archive, dest)
    marker.write_text(f"extracted_from={archive.name}\n", encoding="utf-8")
    return "extracted"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare IAM manifests for DISHA-HTR")
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "datasets" / "iam" / "raw"),
    )
    parser.add_argument(
        "--extracted-dir",
        default=str(ROOT / "datasets" / "iam" / "extracted"),
    )
    parser.add_argument(
        "--manifests-dir",
        default=str(ROOT / "datasets" / "iam" / "manifests"),
    )
    parser.add_argument(
        "--split-scheme",
        choices=["official", "aachen"],
        default="aachen",
        help="Writer-independent split source (default: aachen / OpenSLR)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Assume archives already extracted",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Build manifests even when PNG files are absent (ascii-only dry run)",
    )
    args = parser.parse_args()

    import os

    os.chdir(ROOT)
    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR IAM prepare (Phase 1) ===")

    raw_dir = Path(args.raw_dir)
    extracted_dir = Path(args.extracted_dir)
    manifests_dir = Path(args.manifests_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "split_scheme": args.split_scheme,
        "extract": {},
        "status": "ok",
        "blocked_reasons": [],
        "failures": [],
    }

    if not args.skip_extract:
        mapping = {
            "lines.tgz": ("lines", ".extract_ok_lines"),
            "ascii.tgz": ("ascii", ".extract_ok_ascii"),
            "largeWriterIndependentTextLineRecognitionTask.zip": (
                "official_splits",
                ".extract_ok_official_splits",
            ),
            "splits.zip": ("aachen_splits", ".extract_ok_aachen_splits"),
        }
        for fname, (subdir, marker) in mapping.items():
            archive = raw_dir / fname
            dest = extracted_dir / subdir
            if not archive.is_file():
                report["extract"][fname] = "missing_archive"
                continue
            try:
                report["extract"][fname] = _extract_if_needed(archive, dest, marker)
            except Exception as exc:  # noqa: BLE001
                report["extract"][fname] = f"error:{exc}"
                report["failures"].append(f"extract:{fname}:{exc}")

    ascii_dir = find_ascii_dir(extracted_dir / "ascii") or find_ascii_dir(extracted_dir)
    lines_root = find_lines_root(extracted_dir / "lines") or find_lines_root(extracted_dir)

    if ascii_dir is None:
        report["status"] = "blocked"
        report["blocked_reasons"].append(
            "ascii metadata not found — download ascii.tgz after FKI registration"
        )
        (manifests_dir / "prepare_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        print(
            "Structure is ready under datasets/iam/. "
            "Register at FKI, download with scripts/download_iam.py, then re-run."
        )
        return 2

    lines_meta = parse_lines_txt(ascii_dir / "lines.txt")
    form_to_writer = parse_forms_txt(ascii_dir / "forms.txt")
    logger.info(
        "Parsed ascii: %d lines, %d forms/writers",
        len(lines_meta),
        len(form_to_writer),
    )

    if args.split_scheme == "official":
        splits_dir = extracted_dir / "official_splits"
        if not any(splits_dir.rglob("trainset.txt")):
            report["status"] = "blocked"
            report["blocked_reasons"].append(
                "official split zip not extracted — needs FKI download"
            )
            (manifests_dir / "prepare_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, indent=2))
            return 2
    else:
        splits_dir = extracted_dir / "aachen_splits"
        if not any(Path(splits_dir).rglob("train.uttlist")):
            # try raw extract location
            alt = extracted_dir / "aachen_splits"
            if not any(alt.rglob("train.uttlist")):
                report["status"] = "blocked"
                report["blocked_reasons"].append(
                    "Aachen splits missing — run: scripts/download_iam.py --aachen-only"
                )
                (manifests_dir / "prepare_report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(json.dumps(report, indent=2))
                return 2
            splits_dir = alt

    try:
        split_ids = build_split_line_ids(
            split_scheme=args.split_scheme,
            splits_dir=splits_dir,
            lines_meta=lines_meta,
        )
    except FileNotFoundError as exc:
        report["status"] = "blocked"
        report["blocked_reasons"].append(str(exc))
        (manifests_dir / "prepare_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 2

    require_image = lines_root is not None and not args.allow_missing_images
    if lines_root is None:
        report["failures"].append("lines_root_missing")
        if not args.allow_missing_images:
            report["status"] = "blocked"
            report["blocked_reasons"].append(
                "lines images not found — download lines.tgz after FKI registration "
                "(or pass --allow-missing-images for ascii-only dry manifests)"
            )
            (manifests_dir / "prepare_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, indent=2))
            return 2
        lines_root = extracted_dir / "lines"  # placeholder path for relative names

    split_records = {}
    problems = {}
    for split_name, line_ids in split_ids.items():
        records, probs = pair_lines(
            lines_meta,
            form_to_writer,
            lines_root,
            line_ids,
            require_image=require_image,
            ok_status_only=True,
        )
        # Drop rows without writer_id for strict writer-independent reporting
        missing_writer = [r for r in records if not r.writer_id]
        if missing_writer:
            probs.extend(f"dropped_missing_writer:{r.line_id}" for r in missing_writer)
            records = [r for r in records if r.writer_id]
        split_records[split_name] = records
        problems[split_name] = probs
        out_name = {
            "train": "train.csv",
            "val": "val.csv",
            "val2": "val2.csv",
            "test": "test.csv",
        }.get(split_name, f"{split_name}.csv")
        write_manifest_csv(manifests_dir / out_name, records, split=split_name)
        logger.info(
            "Wrote %s (%d lines, %d problems)",
            out_name,
            len(records),
            len(probs),
        )

    stats = compute_stats(
        split_records, problems, split_scheme=args.split_scheme
    )
    stats["images_required"] = require_image
    stats["lines_root"] = str(lines_root)
    stats["ascii_dir"] = str(ascii_dir)
    write_stats_json(manifests_dir / "stats.json", stats)

    overlap = summarize_writers_for_report(split_records)
    report["counts"] = {k: len(v) for k, v in split_records.items()}
    report["writer_overlap"] = overlap
    report["stats_path"] = str(manifests_dir / "stats.json")
    if overlap["has_any_overlap"] and args.split_scheme == "official":
        report["failures"].append("unexpected_writer_overlap_in_official_split")
    # Aachen is form-based; writers should still be independent across splits
    if overlap["has_any_overlap"]:
        report["failures"].append("writer_overlap_detected")

    (manifests_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Manifests: {manifests_dir}")
    return 0 if report["status"] == "ok" and not report["blocked_reasons"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
