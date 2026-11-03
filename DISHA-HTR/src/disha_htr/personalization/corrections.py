"""Authorized correction workflow — no auto-learning from unapproved text."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .auth import DoctorAuthContext
from .doctor_id import normalize_doctor_id
from .privacy import assert_no_patient_fields, redact_text, safe_path_name
from .samples import SampleRegistry
from .storage import PersonalizationStore


@dataclass
class CorrectionTicket:
    ticket_id: str
    doctor_id: str
    image_filename: str
    proposed_transcription: str
    model_prediction: str
    status: str  # pending | approved | rejected
    created_at_utc: str
    reviewed_by: str | None = None
    reviewed_at_utc: str | None = None
    promote_to_split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorrectionWorkflow:
    """
    Doctors may propose corrections; promotion into training data requires
    explicit approval (verified_by). Unapproved tickets never enter SampleRegistry.
    """

    def __init__(self, store: PersonalizationStore) -> None:
        self.store = store
        self.samples = SampleRegistry(store)

    def submit(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        *,
        image_filename: str,
        proposed_transcription: str,
        model_prediction: str = "",
    ) -> CorrectionTicket:
        auth.require_doctor(doctor_id)
        did = normalize_doctor_id(doctor_id)
        text = (proposed_transcription or "").strip()
        if not text:
            raise ValueError("proposed_transcription required")
        name = safe_path_name(image_filename)
        ticket = CorrectionTicket(
            ticket_id=uuid4().hex,
            doctor_id=did,
            image_filename=name,
            proposed_transcription=text,
            model_prediction=model_prediction or "",
            status="pending",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        path = self.store.corrections_dir(did, "pending") / f"{ticket.ticket_id}.json"
        payload = ticket.to_dict()
        assert_no_patient_fields(payload)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.store.audit_log(did).append(
            "correction_submitted",
            {
                "doctor_id": did,
                "ticket_id": ticket.ticket_id,
                "image_filename": name,
                "proposed_len": len(text),
                "pred_preview": redact_text(model_prediction)[:40],
            },
        )
        return ticket

    def approve(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        ticket_id: str,
        *,
        reviewed_by: str,
        promote_to_split: str | None = None,
        image_src: str | Path | None = None,
    ) -> CorrectionTicket:
        """
        Approve a correction. Optionally promote into verified train/val samples
        when image_src is provided and promote_to_split is set.
        """
        auth.require_doctor(doctor_id)
        did = normalize_doctor_id(doctor_id)
        pending = self.store.corrections_dir(did, "pending") / f"{ticket_id}.json"
        if not pending.is_file():
            raise FileNotFoundError(f"pending ticket not found: {ticket_id}")
        ticket = CorrectionTicket(**json.loads(pending.read_text(encoding="utf-8")))
        ticket.status = "approved"
        ticket.reviewed_by = reviewed_by
        ticket.reviewed_at_utc = datetime.now(timezone.utc).isoformat()
        ticket.promote_to_split = promote_to_split

        approved = self.store.corrections_dir(did, "approved") / f"{ticket_id}.json"
        approved.write_text(json.dumps(ticket.to_dict(), indent=2), encoding="utf-8")
        pending.unlink()

        if promote_to_split in ("train", "val"):
            if image_src is None:
                raise ValueError("image_src required to promote approved correction")
            self.samples.add_verified_sample(
                auth,
                did,
                image_path=image_src,
                ground_truth=ticket.proposed_transcription,
                split=promote_to_split,
                verified_by=reviewed_by,
                source="approved_correction",
            )

        self.store.audit_log(did).append(
            "correction_approved",
            {
                "doctor_id": did,
                "ticket_id": ticket_id,
                "reviewed_by": reviewed_by,
                "promote_to_split": promote_to_split,
            },
        )
        return ticket

    def reject(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        ticket_id: str,
        *,
        reviewed_by: str,
    ) -> CorrectionTicket:
        auth.require_doctor(doctor_id)
        did = normalize_doctor_id(doctor_id)
        pending = self.store.corrections_dir(did, "pending") / f"{ticket_id}.json"
        if not pending.is_file():
            raise FileNotFoundError(f"pending ticket not found: {ticket_id}")
        ticket = CorrectionTicket(**json.loads(pending.read_text(encoding="utf-8")))
        ticket.status = "rejected"
        ticket.reviewed_by = reviewed_by
        ticket.reviewed_at_utc = datetime.now(timezone.utc).isoformat()
        rejected = self.store.corrections_dir(did, "rejected") / f"{ticket_id}.json"
        rejected.write_text(json.dumps(ticket.to_dict(), indent=2), encoding="utf-8")
        pending.unlink()
        self.store.audit_log(did).append(
            "correction_rejected",
            {"doctor_id": did, "ticket_id": ticket_id, "reviewed_by": reviewed_by},
        )
        return ticket
