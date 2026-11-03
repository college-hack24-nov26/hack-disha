"""Authenticated doctor context (no built-in login — caller supplies proof)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .doctor_id import normalize_doctor_id


@dataclass(frozen=True)
class DoctorAuthContext:
    """
    Proof that a caller is acting as a specific doctor.

    DISHA-HTR does not implement identity providers here. The host app must
    authenticate the clinician and pass a validated context. Personalization
    APIs refuse empty / mismatched IDs.
    """

    doctor_id: str
    session_id: str | None = None
    roles: tuple[str, ...] = ("doctor",)
    claims: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "doctor_id", normalize_doctor_id(self.doctor_id))

    def require_doctor(self, expected: str) -> None:
        exp = normalize_doctor_id(expected)
        if self.doctor_id != exp:
            raise PermissionError(
                "authenticated doctor_id does not match requested profile"
            )

    def to_audit_dict(self) -> dict[str, Any]:
        """Safe audit fields — never include patient identifiers."""
        return {
            "doctor_id": self.doctor_id,
            "session_id": self.session_id,
            "roles": list(self.roles),
        }
