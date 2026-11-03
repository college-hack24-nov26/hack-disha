"""
IAM Handwriting Database preparation helpers (Phase 1).

CPU-only parsing / pairing / validation / writer-independent manifests.
Does not download credentials, train models, or modify evaluation samples.
Raw archives under datasets/iam/raw must remain unchanged after extract.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("disha_htr.iam_dataset")

MANIFEST_FIELDS = (
    "image_path",
    "text",
    "writer_id",
    "line_id",
    "form_id",
    "split",
)

# Official Large Writer Independent Text Line Recognition Task files
OFFICIAL_SPLIT_FILES = {
    "train": "trainset.txt",
    "val": "validationset1.txt",  # primary val; val2 tracked separately in stats
    "val2": "validationset2.txt",
    "test": "testset.txt",
}

AACHEN_SPLIT_FILES = {
    "train": "train.uttlist",
    "val": "validation.uttlist",
    "test": "test.uttlist",
}


@dataclass(frozen=True)
class IamLineRecord:
    line_id: str
    form_id: str
    writer_id: str
    text: str
    status: str
    image_path: str  # relative path under extracted lines root


def form_id_from_line_id(line_id: str) -> str:
    """a01-000u-00 → a01-000u ; a01-003-01 → a01-003."""
    parts = line_id.split("-")
    if len(parts) < 3:
        return line_id
    return "-".join(parts[:-1])


def transcription_from_ascii_token(token: str) -> str:
    """IAM ascii uses '|' as space separator in the final field."""
    return token.replace("|", " ").strip()


def parse_lines_txt(path: Path) -> dict[str, tuple[str, str]]:
    """
    Parse ascii/lines.txt → {line_id: (status, text)}.

    Format (whitespace-separated; transcription may contain spaces after first tokens):
      line_id status graylevel ... transcription_with_|_for_spaces
    Comment lines start with '#'.
    """
    path = Path(path)
    records: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                # Minimal: id status ... text
                if len(parts) < 2:
                    continue
            line_id = parts[0]
            status = parts[1] if len(parts) > 1 else "err"
            # Transcription is the last field in IAM lines.txt (may include |).
            # Official format places transcription as the final token(s) after geometry.
            # Kaldi/community parsers take everything after field index 8 joined.
            if len(parts) >= 9:
                text = transcription_from_ascii_token(" ".join(parts[8:]))
            else:
                text = ""
            records[line_id] = (status, text)
    return records


def parse_forms_txt(path: Path) -> dict[str, str]:
    """
    Parse ascii/forms.txt → {form_id: writer_id}.

    Format starts with: form_id writer_id ...
    """
    path = Path(path)
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def load_id_list(path: Path) -> list[str]:
    """Load a split file of line IDs or form IDs (one per line)."""
    path = Path(path)
    ids: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            item = raw.strip()
            if not item or item.startswith("#"):
                continue
            ids.append(item)
    return ids


def expected_line_image_relpath(line_id: str) -> str:
    """
    IAM lines layout: lines/<form_prefix>/<form_id>/<line_id>.png
    Example: a01-000u-00 → a01/a01-000u/a01-000u-00.png
    """
    form_id = form_id_from_line_id(line_id)
    prefix = form_id.split("-")[0]
    return f"{prefix}/{form_id}/{line_id}.png"


def pair_lines(
    lines_meta: dict[str, tuple[str, str]],
    form_to_writer: dict[str, str],
    lines_root: Path,
    line_ids: Iterable[str],
    *,
    require_image: bool = True,
    ok_status_only: bool = True,
) -> tuple[list[IamLineRecord], list[str]]:
    """
    Pair split line IDs with ascii transcriptions + on-disk PNG paths.

    Returns (records, problems). Does not invent text.
    """
    lines_root = Path(lines_root)
    records: list[IamLineRecord] = []
    problems: list[str] = []

    for line_id in line_ids:
        meta = lines_meta.get(line_id)
        if meta is None:
            problems.append(f"missing_ascii:{line_id}")
            continue
        status, text = meta
        if ok_status_only and status != "ok":
            problems.append(f"bad_status:{line_id}:{status}")
            continue
        if not text:
            problems.append(f"empty_text:{line_id}")
            continue

        form_id = form_id_from_line_id(line_id)
        writer_id = form_to_writer.get(form_id, "")
        if not writer_id:
            problems.append(f"missing_writer:{line_id}")
            # Still allow record with blank writer for visibility; caller can drop

        rel = expected_line_image_relpath(line_id)
        img_path = lines_root / rel
        if require_image and not img_path.is_file():
            problems.append(f"missing_image:{line_id}:{rel}")
            continue

        records.append(
            IamLineRecord(
                line_id=line_id,
                form_id=form_id,
                writer_id=writer_id,
                text=text,
                status=status,
                image_path=rel.replace("\\", "/"),
            )
        )

    records.sort(key=lambda r: r.line_id)
    return records, problems


def expand_aachen_form_ids_to_line_ids(
    form_ids: Iterable[str],
    lines_meta: dict[str, tuple[str, str]],
) -> list[str]:
    """Aachen uttlists are form IDs; expand to all matching line IDs in ascii."""
    wanted = set(form_ids)
    line_ids = [
        line_id
        for line_id in lines_meta
        if form_id_from_line_id(line_id) in wanted
    ]
    line_ids.sort()
    return line_ids


def writer_overlap(
    split_records: dict[str, list[IamLineRecord]],
) -> dict[str, list[str]]:
    """Return overlapping writer_ids between each pair of splits."""
    writers = {
        name: {r.writer_id for r in rows if r.writer_id}
        for name, rows in split_records.items()
    }
    names = sorted(writers)
    overlaps: dict[str, list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = sorted(writers[a] & writers[b])
            overlaps[f"{a}|{b}"] = both
    return overlaps


def write_manifest_csv(path: Path, records: list[IamLineRecord], split: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "image_path": rec.image_path,
                    "text": rec.text,
                    "writer_id": rec.writer_id,
                    "line_id": rec.line_id,
                    "form_id": rec.form_id,
                    "split": split,
                }
            )
    return path


def compute_stats(
    split_records: dict[str, list[IamLineRecord]],
    problems: dict[str, list[str]],
    *,
    split_scheme: str,
) -> dict:
    overlaps = writer_overlap(split_records)
    stats = {
        "split_scheme": split_scheme,
        "splits": {},
        "writer_overlap": {k: len(v) for k, v in overlaps.items()},
        "writer_overlap_ids_sample": {k: v[:20] for k, v in overlaps.items()},
        "problems_count": {k: len(v) for k, v in problems.items()},
        "total_lines": sum(len(v) for v in split_records.values()),
        "total_writers": len(
            {r.writer_id for rows in split_records.values() for r in rows if r.writer_id}
        ),
    }
    for name, rows in split_records.items():
        writers = {r.writer_id for r in rows if r.writer_id}
        text_lens = [len(r.text) for r in rows]
        stats["splits"][name] = {
            "lines": len(rows),
            "writers": len(writers),
            "empty_text": sum(1 for r in rows if not r.text),
            "mean_text_len": round(sum(text_lens) / len(text_lens), 2) if text_lens else 0,
        }
    return stats


def write_stats_json(path: Path, stats: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return path


def safe_extract_tar(archive: Path, dest: Path) -> None:
    """Extract tar/tgz into dest (idempotent if marker present handled by caller)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        # Basic path-traversal guard
        for member in tf.getmembers():
            member_path = dest / member.name
            if not _is_within_directory(dest, member_path):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        # Python 3.12+ supports filter= to block unsafe members
        tf.extractall(dest, filter="data")


def safe_extract_zip(archive: Path, dest: Path) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        for name in zf.namelist():
            member_path = dest / name
            if not _is_within_directory(dest, member_path):
                raise RuntimeError(f"Unsafe path in archive: {name}")
        zf.extractall(dest)


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
        return directory == target or directory in target.parents
    except OSError:
        return False


def find_lines_root(extracted_root: Path) -> Path | None:
    """Locate directory that contains form-prefix folders (a01, a02, ...)."""
    extracted_root = Path(extracted_root)
    candidates = [
        extracted_root / "lines",
        extracted_root / "data" / "lines",
        extracted_root,
    ]
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("a01")):
            return cand
        # also accept any two-letter prefix dirs
        if cand.is_dir() and any(p.is_dir() and re.fullmatch(r"[a-z]\d{2}", p.name) for p in cand.iterdir()):
            return cand
    # Deep search one level
    if extracted_root.is_dir():
        for child in extracted_root.iterdir():
            if child.is_dir() and any(child.glob("a01")):
                return child
    return None


def find_ascii_dir(extracted_root: Path) -> Path | None:
    extracted_root = Path(extracted_root)
    for cand in (
        extracted_root / "ascii",
        extracted_root / "data" / "ascii",
        extracted_root,
    ):
        if (cand / "lines.txt").is_file() and (cand / "forms.txt").is_file():
            return cand
    if extracted_root.is_dir():
        for path in extracted_root.rglob("lines.txt"):
            if (path.parent / "forms.txt").is_file():
                return path.parent
    return None


def build_split_line_ids(
    *,
    split_scheme: str,
    splits_dir: Path,
    lines_meta: dict[str, tuple[str, str]],
) -> dict[str, list[str]]:
    """
    Return {split_name: [line_id, ...]} for official or aachen schemes.

    official: uses train/val(=val1)/val2/test line-id lists
    aachen: expands form-id lists to line ids
    """
    splits_dir = Path(splits_dir)
    out: dict[str, list[str]] = {}

    if split_scheme == "official":
        mapping = {
            "train": OFFICIAL_SPLIT_FILES["train"],
            "val": OFFICIAL_SPLIT_FILES["val"],
            "val2": OFFICIAL_SPLIT_FILES["val2"],
            "test": OFFICIAL_SPLIT_FILES["test"],
        }
        for name, fname in mapping.items():
            path = splits_dir / fname
            if not path.is_file():
                # Sometimes nested
                matches = list(splits_dir.rglob(fname))
                if not matches:
                    raise FileNotFoundError(f"Missing official split file: {fname}")
                path = matches[0]
            out[name] = load_id_list(path)
    elif split_scheme == "aachen":
        for name, fname in AACHEN_SPLIT_FILES.items():
            path = splits_dir / fname
            if not path.is_file():
                matches = list(splits_dir.rglob(fname))
                if not matches:
                    raise FileNotFoundError(f"Missing Aachen split file: {fname}")
                path = matches[0]
            form_ids = load_id_list(path)
            out[name] = expand_aachen_form_ids_to_line_ids(form_ids, lines_meta)
    else:
        raise ValueError(f"Unknown split_scheme: {split_scheme}")

    return out


def summarize_writers_for_report(split_records: dict[str, list[IamLineRecord]]) -> dict:
    overlaps = writer_overlap(split_records)
    return {
        "overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "has_any_overlap": any(len(v) > 0 for v in overlaps.values()),
    }
