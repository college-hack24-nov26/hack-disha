"""Append-only audit history for personalization updates (no patient data)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .privacy import assert_no_patient_fields, redact_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    """JSONL audit trail scoped to one doctor's personalization directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self.path.write_text("", encoding="utf-8")

    def append(self, event_type: str, payload: dict[str, Any] | None = None) -> dict:
        body = dict(payload or {})
        assert_no_patient_fields(body)
        # Redact free-text fields commonly logged
        for key in ("note", "message", "transcription_preview", "error"):
            if key in body and isinstance(body[key], str):
                body[key] = redact_text(body[key])
        record = {
            "ts_utc": _utc_now(),
            "event_type": event_type,
            "payload": body,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_all(self) -> list[dict]:
        rows: list[dict] = []
        if not self.path.is_file():
            return rows
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
