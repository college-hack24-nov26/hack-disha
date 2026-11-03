"""Adapter/checkpoint versioning, activation, and rollback per doctor."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .auth import DoctorAuthContext
from .doctor_id import normalize_doctor_id
from .storage import PersonalizationStore

GENERAL_FALLBACK = "general"


@dataclass
class AdapterMeta:
    doctor_id: str
    version: str
    method: str  # decoder_finetune | placeholder_adapter
    base_checkpoint: str
    created_at_utc: str
    val_cer: float | None = None
    general_val_cer: float | None = None
    improves_over_general: bool | None = None
    status: str = "candidate"  # candidate | active | rolled_back | rejected
    notes: str = ""
    artifact_kind: str = "vision_encoder_decoder_dir"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterMeta":
        return cls(
            doctor_id=normalize_doctor_id(data["doctor_id"]),
            version=str(data["version"]),
            method=str(data.get("method") or "decoder_finetune"),
            base_checkpoint=str(data["base_checkpoint"]),
            created_at_utc=str(data["created_at_utc"]),
            val_cer=data.get("val_cer"),
            general_val_cer=data.get("general_val_cer"),
            improves_over_general=data.get("improves_over_general"),
            status=str(data.get("status") or "candidate"),
            notes=str(data.get("notes") or ""),
            artifact_kind=str(
                data.get("artifact_kind") or "vision_encoder_decoder_dir"
            ),
        )


class ModelRegistry:
    """
    Tracks per-doctor personalized artifact versions.

    Activation requires measured val CER improvement over the general model
    (unless force=True for admin rollback tests). Shared general checkpoint
    is never modified.
    """

    def __init__(self, store: PersonalizationStore) -> None:
        self.store = store

    def list_versions(self, doctor_id: str) -> list[AdapterMeta]:
        did = normalize_doctor_id(doctor_id)
        root = self.store.adapters_dir(did)
        if not root.is_dir():
            return []
        metas: list[AdapterMeta] = []
        for child in sorted(root.iterdir()):
            meta_path = child / "meta.json"
            if child.is_dir() and meta_path.is_file():
                metas.append(
                    AdapterMeta.from_dict(
                        json.loads(meta_path.read_text(encoding="utf-8"))
                    )
                )
        return metas

    def get_active_version(self, doctor_id: str) -> str | None:
        did = normalize_doctor_id(doctor_id)
        pointer = self.store.active_pointer_path(did)
        if not pointer.is_file():
            profile = None
            try:
                profile = self.store.load_profile(did)
            except FileNotFoundError:
                return None
            return profile.active_adapter_version
        data = json.loads(pointer.read_text(encoding="utf-8"))
        ver = data.get("version")
        return str(ver) if ver else None

    def resolve_adapter_dir(self, doctor_id: str, version: str | None = None) -> Path | None:
        did = normalize_doctor_id(doctor_id)
        ver = version or self.get_active_version(did)
        if not ver:
            return None
        path = self.store.adapter_version_dir(did, ver)
        if not path.is_dir():
            return None
        # Require weights or explicit placeholder marker
        has_weights = any(
            (path / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin", "adapter_placeholder.json")
        )
        if not has_weights:
            return None
        return path

    def register_candidate(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        version: str,
        *,
        method: str,
        base_checkpoint: str,
        artifact_dir: str | Path | None = None,
        val_cer: float | None = None,
        general_val_cer: float | None = None,
        notes: str = "",
    ) -> AdapterMeta:
        """
        Register a new version directory. Copies artifact_dir contents if given.
        Does not activate. Does not modify the general checkpoint.
        """
        auth.require_doctor(doctor_id)
        did = normalize_doctor_id(doctor_id)
        dest = self.store.adapter_version_dir(did, version)
        if dest.exists():
            raise FileExistsError(f"adapter version already exists: {version}")
        dest.mkdir(parents=True, exist_ok=False)

        if artifact_dir is not None:
            src = Path(artifact_dir)
            if not src.is_dir():
                raise FileNotFoundError(f"artifact_dir not found: {src}")
            for item in src.iterdir():
                target = dest / item.name
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        else:
            # Architecture placeholder until approved training produces weights
            (dest / "adapter_placeholder.json").write_text(
                json.dumps(
                    {
                        "doctor_id": did,
                        "version": version,
                        "note": "No weights yet — training gated pending approval.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        improves = None
        if val_cer is not None and general_val_cer is not None:
            improves = float(val_cer) < float(general_val_cer)

        meta = AdapterMeta(
            doctor_id=did,
            version=version,
            method=method,
            base_checkpoint=str(base_checkpoint),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            val_cer=val_cer,
            general_val_cer=general_val_cer,
            improves_over_general=improves,
            status="candidate",
            notes=notes,
        )
        (dest / "meta.json").write_text(
            json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
        )
        self.store.audit_log(did).append(
            "adapter_candidate_registered",
            {
                "doctor_id": did,
                "version": version,
                "method": method,
                "val_cer": val_cer,
                "general_val_cer": general_val_cer,
                "improves_over_general": improves,
            },
        )
        return meta

    def activate(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        version: str,
        *,
        force: bool = False,
    ) -> AdapterMeta:
        """
        Activate a version if it improves on general val CER (unless force).
        """
        auth.require_doctor(doctor_id)
        did = normalize_doctor_id(doctor_id)
        dest = self.store.adapter_version_dir(did, version)
        meta_path = dest / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing meta for {version}")
        meta = AdapterMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))

        if meta.improves_over_general is False and not force:
            raise ValueError(
                "refusing to activate personalized model that does not improve "
                "validation CER over general; pass force=True only for admin tests"
            )
        if meta.improves_over_general is None and not force:
            raise ValueError(
                "val CER comparison missing; evaluate before activation "
                "or pass force=True for placeholder testing"
            )

        # Mark previous active as rolled_back historically
        prev = self.get_active_version(did)
        if prev and prev != version:
            prev_meta_path = self.store.adapter_version_dir(did, prev) / "meta.json"
            if prev_meta_path.is_file():
                prev_meta = AdapterMeta.from_dict(
                    json.loads(prev_meta_path.read_text(encoding="utf-8"))
                )
                prev_meta.status = "rolled_back"
                prev_meta_path.write_text(
                    json.dumps(prev_meta.to_dict(), indent=2), encoding="utf-8"
                )

        meta.status = "active"
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        pointer = {
            "version": version,
            "activated_at_utc": datetime.now(timezone.utc).isoformat(),
            "activated_by": auth.doctor_id,
        }
        self.store.active_pointer_path(did).write_text(
            json.dumps(pointer, indent=2), encoding="utf-8"
        )
        profile = self.store.load_profile(did)
        profile.active_adapter_version = version
        self.store.save_profile(profile)
        self.store.audit_log(did).append(
            "adapter_activated",
            {"doctor_id": did, "version": version, "force": force, "previous": prev},
        )
        return meta

    def rollback(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        to_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Roll back to a previous version, or clear personalization (general fallback).
        """
        auth.require_doctor(doctor_id)
        did = normalize_doctor_id(doctor_id)
        current = self.get_active_version(did)
        if to_version is None:
            # Clear active → general fallback
            pointer = self.store.active_pointer_path(did)
            if pointer.is_file():
                pointer.unlink()
            if current:
                cur_meta = self.store.adapter_version_dir(did, current) / "meta.json"
                if cur_meta.is_file():
                    meta = AdapterMeta.from_dict(
                        json.loads(cur_meta.read_text(encoding="utf-8"))
                    )
                    meta.status = "rolled_back"
                    cur_meta.write_text(
                        json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
                    )
            profile = self.store.load_profile(did)
            profile.active_adapter_version = None
            self.store.save_profile(profile)
            self.store.audit_log(did).append(
                "adapter_rollback_to_general",
                {"doctor_id": did, "previous": current},
            )
            return {"active": None, "fallback": GENERAL_FALLBACK, "previous": current}

        meta = self.activate(auth, did, to_version, force=True)
        self.store.audit_log(did).append(
            "adapter_rollback_to_version",
            {"doctor_id": did, "version": to_version, "previous": current},
        )
        return {"active": meta.version, "previous": current}

    def next_version(self, doctor_id: str) -> str:
        versions = self.list_versions(doctor_id)
        n = len(versions) + 1
        return f"v{n:04d}"
