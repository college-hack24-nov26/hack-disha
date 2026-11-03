"""Doctor ID normalization and path-safety checks."""

from __future__ import annotations

import re

_DOCTOR_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")


def normalize_doctor_id(doctor_id: str) -> str:
    """
    Normalize and validate a doctor ID for filesystem isolation.

    Accepts letters/digits/underscore/hyphen only (lowercased). Rejects
    path separators and traversal. Raises ValueError on invalid input.
    """
    if doctor_id is None:
        raise ValueError("doctor_id is required")
    raw = str(doctor_id).strip().lower()
    if not raw:
        raise ValueError("doctor_id is empty")
    if any(sep in raw for sep in ("/", "\\", "..")):
        raise ValueError("doctor_id must not contain path separators")
    if not _DOCTOR_ID_RE.match(raw):
        raise ValueError(
            "doctor_id must match [a-z0-9][a-z0-9_-]{1,63} after lowercasing"
        )
    return raw
