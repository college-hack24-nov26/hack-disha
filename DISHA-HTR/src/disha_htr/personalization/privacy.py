"""Privacy helpers — keep patient identifiers out of logs and exports."""

from __future__ import annotations

import re
from pathlib import Path

# Conservative patterns for accidental PHI-ish strings in free text logs.
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s()]{7,}\d)\b")
_MRN_RE = re.compile(r"\b(?:MRN|UHID|PATIENT[_\s-]?ID)[:\s#]*\w+\b", re.I)


def redact_text(text: str | None) -> str:
    """Redact common PII/PHI patterns from loggable strings."""
    if not text:
        return ""
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
    out = _MRN_RE.sub("[REDACTED_ID]", out)
    return out


def safe_path_name(name: str) -> str:
    """Basename-only filename for storage (no directories)."""
    base = Path(name).name
    if not base or base in (".", ".."):
        raise ValueError("invalid filename")
    if "/" in base or "\\" in base:
        raise ValueError("filename must not contain separators")
    return base


def assert_no_patient_fields(payload: dict) -> None:
    """Reject payloads that appear to carry patient identifiers."""
    forbidden = {
        "patient_id",
        "patient_name",
        "mrn",
        "uhid",
        "dob",
        "date_of_birth",
        "ssn",
        "aadhaar",
    }
    keys = {str(k).lower() for k in payload.keys()}
    hit = keys & forbidden
    if hit:
        raise ValueError(f"patient-identifiable fields not allowed: {sorted(hit)}")
