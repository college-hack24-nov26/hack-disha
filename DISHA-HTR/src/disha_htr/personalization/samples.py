"""Verified handwriting sample registry — never trusts model predictions as GT."""

from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .auth import DoctorAuthContext
from .doctor_id import normalize_doctor_id
from .privacy import safe_path_name
from .storage import PersonalizationStore

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class VerifiedSample:
    image_filename: str
    split: str
    ground_truth: str
    verified_by: str
    verified_at_utc: str
    source: str
    sha256: str | None = None


class SampleRegistry:
    """
    Manages per-doctor verified image/transcription pairs.

    Rules:
    - Ground truth must be human-verified (verified_by required).
    - Model predictions cannot be registered as ground truth.
    - Train/val kept separate via split column + folders.
    - Files stay inside the doctor's samples directory only.
    """

    def __init__(self, store: PersonalizationStore) -> None:
        self.store = store

    def list_samples(self, doctor_id: str, split: str | None = None) -> list[VerifiedSample]:
        did = normalize_doctor_id(doctor_id)
        rows = self._read_csv(did)
        out: list[VerifiedSample] = []
        for row in rows:
            if split and row.split != split:
                continue
            out.append(row)
        return out

    def add_verified_sample(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        image_path: str | Path,
        ground_truth: str,
        split: str,
        *,
        verified_by: str,
        source: str = "clinician_verified",
        allow_overwrite: bool = False,
    ) -> VerifiedSample:
        """
        Copy an image into the doctor's split folder and append a verified row.

        Raises if ground_truth empty, verified_by missing, source looks like
        auto-prediction, or image escapes isolation after copy.
        """
        auth.require_doctor(doctor_id)
        did = normalize_doctor_id(doctor_id)
        if split not in ("train", "val"):
            raise ValueError("split must be train|val")
        gt = (ground_truth or "").strip()
        if not gt:
            raise ValueError("ground_truth is required and must be non-empty")
        if not (verified_by or "").strip():
            raise ValueError("verified_by is required — unverified GT rejected")
        src_flag = (source or "").strip().lower()
        if src_flag in {
            "model_prediction",
            "auto",
            "unverified",
            "trocr_output",
            "predicted",
        }:
            raise ValueError(
                "refusing to store model predictions as ground truth"
            )

        src = Path(image_path)
        if not src.is_file():
            raise FileNotFoundError(f"image not found: {src}")
        if src.suffix.lower() not in ALLOWED_IMAGE_EXTS:
            raise ValueError(f"unsupported image type: {src.suffix}")

        dest_name = safe_path_name(src.name)
        dest_dir = self.store.samples_dir(did, split)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / dest_name
        if dest.exists() and not allow_overwrite:
            raise FileExistsError(f"sample already exists: {dest_name}")
        shutil.copy2(src, dest)
        self.store.assert_path_owned_by(did, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()

        sample = VerifiedSample(
            image_filename=dest_name,
            split=split,
            ground_truth=gt,
            verified_by=verified_by.strip(),
            verified_at_utc=datetime.now(timezone.utc).isoformat(),
            source=source.strip() or "clinician_verified",
            sha256=digest,
        )
        self._upsert_csv(did, sample, allow_overwrite=allow_overwrite)
        self.store.audit_log(did).append(
            "sample_verified_added",
            {
                "doctor_id": did,
                "image_filename": dest_name,
                "split": split,
                "verified_by": verified_by.strip(),
                "source": sample.source,
                "sha256": digest,
                # Do not log full transcription; length only
                "gt_len": len(gt),
            },
        )
        return sample

    def assert_no_cross_doctor_leak(
        self, doctor_a: str, doctor_b: str
    ) -> dict[str, int]:
        """
        Verify no shared image filenames or identical sha256 between doctors.
        Returns counts; raises AssertionError on overlap.
        """
        a = normalize_doctor_id(doctor_a)
        b = normalize_doctor_id(doctor_b)
        if a == b:
            raise ValueError("doctor_a and doctor_b must differ")

        rows_a = self.list_samples(a)
        rows_b = self.list_samples(b)
        names_a = {r.image_filename for r in rows_a}
        names_b = {r.image_filename for r in rows_b}
        # Filenames may coincidentally match; check content hashes under each root
        hashes_a = self._collect_image_hashes(a)
        hashes_b = self._collect_image_hashes(b)
        overlap = set(hashes_a.values()) & set(hashes_b.values())
        if overlap:
            raise AssertionError(
                f"cross-doctor image content overlap between {a} and {b}: "
                f"{len(overlap)} shared hashes"
            )
        return {
            "doctor_a_samples": len(rows_a),
            "doctor_b_samples": len(rows_b),
            "shared_filenames": len(names_a & names_b),
            "shared_content_hashes": 0,
        }

    def _collect_image_hashes(self, doctor_id: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for split in ("train", "val"):
            d = self.store.samples_dir(doctor_id, split)
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS:
                    out[f"{split}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    def _read_csv(self, doctor_id: str) -> list[VerifiedSample]:
        path = self.store.transcriptions_csv(doctor_id)
        if not path.is_file():
            return []
        rows: list[VerifiedSample] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("image_filename") or "").strip()
                split = (row.get("split") or "").strip()
                gt = (row.get("ground_truth") or "").strip()
                if not name or not gt or split not in ("train", "val"):
                    continue
                rows.append(
                    VerifiedSample(
                        image_filename=name,
                        split=split,
                        ground_truth=gt,
                        verified_by=(row.get("verified_by") or "").strip(),
                        verified_at_utc=(row.get("verified_at_utc") or "").strip(),
                        source=(row.get("source") or "").strip(),
                        sha256=(row.get("sha256") or None) or None,
                    )
                )
        return rows

    def _upsert_csv(
        self, doctor_id: str, sample: VerifiedSample, *, allow_overwrite: bool
    ) -> None:
        path = self.store.transcriptions_csv(doctor_id)
        existing = {
            (r.image_filename, r.split): r for r in self._read_csv(doctor_id)
        }
        key = (sample.image_filename, sample.split)
        if key in existing and not allow_overwrite:
            raise FileExistsError(
                f"transcription already registered for {sample.image_filename} ({sample.split})"
            )
        existing[key] = sample
        fieldnames = [
            "image_filename",
            "split",
            "ground_truth",
            "verified_by",
            "verified_at_utc",
            "source",
            "sha256",
        ]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(existing.values(), key=lambda x: (x.split, x.image_filename)):
                writer.writerow(
                    {
                        "image_filename": r.image_filename,
                        "split": r.split,
                        "ground_truth": r.ground_truth,
                        "verified_by": r.verified_by,
                        "verified_at_utc": r.verified_at_utc,
                        "source": r.source,
                        "sha256": r.sha256 or "",
                    }
                )
