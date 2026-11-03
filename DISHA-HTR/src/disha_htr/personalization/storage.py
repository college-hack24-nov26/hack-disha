"""Per-doctor isolated filesystem layout for personalization data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .doctor_id import normalize_doctor_id

DEFAULT_ROOT_NAME = "personalization"
GENERAL_CHECKPOINT_REL = "checkpoints/trocr-small-handwritten"


@dataclass
class DoctorProfile:
    doctor_id: str
    display_name: str
    created_at_utc: str
    general_base_checkpoint: str = GENERAL_CHECKPOINT_REL
    active_adapter_version: str | None = None
    status: str = "active"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoctorProfile":
        return cls(
            doctor_id=normalize_doctor_id(data["doctor_id"]),
            display_name=str(data.get("display_name") or data["doctor_id"]),
            created_at_utc=str(data["created_at_utc"]),
            general_base_checkpoint=str(
                data.get("general_base_checkpoint") or GENERAL_CHECKPOINT_REL
            ),
            active_adapter_version=data.get("active_adapter_version"),
            status=str(data.get("status") or "active"),
            notes=str(data.get("notes") or ""),
        )


class PersonalizationStore:
    """
    Root store: personalization/doctors/<doctor_id>/...

    Isolation rule: all paths for a doctor resolve under that doctor's
    directory. Cross-doctor relative paths are rejected.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.doctors_root = self.root / "doctors"
        self.doctors_root.mkdir(parents=True, exist_ok=True)

    def doctor_dir(self, doctor_id: str) -> Path:
        did = normalize_doctor_id(doctor_id)
        path = (self.doctors_root / did).resolve()
        # Ensure confinement
        path.relative_to(self.doctors_root.resolve())
        return path

    def profile_path(self, doctor_id: str) -> Path:
        return self.doctor_dir(doctor_id) / "profile.json"

    def samples_dir(self, doctor_id: str, split: str) -> Path:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        return self.doctor_dir(doctor_id) / "samples" / split

    def transcriptions_csv(self, doctor_id: str) -> Path:
        return self.doctor_dir(doctor_id) / "samples" / "verified_transcriptions.csv"

    def adapters_dir(self, doctor_id: str) -> Path:
        return self.doctor_dir(doctor_id) / "adapters"

    def adapter_version_dir(self, doctor_id: str, version: str) -> Path:
        ver = _safe_version(version)
        path = (self.adapters_dir(doctor_id) / ver).resolve()
        path.relative_to(self.adapters_dir(doctor_id).resolve())
        return path

    def active_pointer_path(self, doctor_id: str) -> Path:
        return self.adapters_dir(doctor_id) / "active.json"

    def corrections_dir(self, doctor_id: str, bucket: str) -> Path:
        if bucket not in ("pending", "approved", "rejected"):
            raise ValueError("bucket must be pending|approved|rejected")
        return self.doctor_dir(doctor_id) / "corrections" / bucket

    def audit_log(self, doctor_id: str) -> AuditLog:
        return AuditLog(self.doctor_dir(doctor_id) / "audit" / "audit.jsonl")

    def list_doctor_ids(self) -> list[str]:
        if not self.doctors_root.is_dir():
            return []
        ids = []
        for p in sorted(self.doctors_root.iterdir()):
            if p.is_dir() and (p / "profile.json").is_file():
                ids.append(p.name)
        return ids

    def load_profile(self, doctor_id: str) -> DoctorProfile:
        path = self.profile_path(doctor_id)
        if not path.is_file():
            raise FileNotFoundError(f"no profile for doctor_id={doctor_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return DoctorProfile.from_dict(data)

    def save_profile(self, profile: DoctorProfile) -> None:
        path = self.profile_path(profile.doctor_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    def assert_path_owned_by(self, doctor_id: str, path: Path) -> Path:
        """Raise if path escapes the doctor's directory."""
        did = normalize_doctor_id(doctor_id)
        owned = self.doctor_dir(did)
        resolved = path.resolve()
        try:
            resolved.relative_to(owned)
        except ValueError as exc:
            raise PermissionError(
                f"path escapes doctor isolation boundary for {did}"
            ) from exc
        return resolved


def create_doctor_profile(
    store: PersonalizationStore,
    doctor_id: str,
    display_name: str | None = None,
    general_base_checkpoint: str = GENERAL_CHECKPOINT_REL,
) -> DoctorProfile:
    """Create empty isolated directories for a doctor. Idempotent if exists."""
    did = normalize_doctor_id(doctor_id)
    existing = store.profile_path(did)
    if existing.is_file():
        return store.load_profile(did)

    profile = DoctorProfile(
        doctor_id=did,
        display_name=(display_name or did),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        general_base_checkpoint=general_base_checkpoint,
    )
    # Directory skeleton
    for split in ("train", "val"):
        store.samples_dir(did, split).mkdir(parents=True, exist_ok=True)
    store.adapters_dir(did).mkdir(parents=True, exist_ok=True)
    for bucket in ("pending", "approved", "rejected"):
        store.corrections_dir(did, bucket).mkdir(parents=True, exist_ok=True)
    (store.doctor_dir(did) / "audit").mkdir(parents=True, exist_ok=True)

    csv_path = store.transcriptions_csv(did)
    if not csv_path.is_file():
        csv_path.write_text(
            "image_filename,split,ground_truth,verified_by,verified_at_utc,source\n",
            encoding="utf-8",
        )
    store.save_profile(profile)
    store.audit_log(did).append(
        "profile_created",
        {"doctor_id": did, "display_name": profile.display_name},
    )
    return profile


def _safe_version(version: str) -> str:
    ver = str(version).strip().lower()
    if not ver or any(c in ver for c in ("/", "\\", "..")):
        raise ValueError(f"invalid adapter version: {version!r}")
    if not ver.startswith("v"):
        raise ValueError("adapter version must start with 'v' (e.g. v0001)")
    return ver
