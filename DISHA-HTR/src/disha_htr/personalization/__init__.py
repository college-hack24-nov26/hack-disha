"""Doctor-specific handwriting personalization (architecture layer).

Preserves the shared general TrOCR checkpoint. Personalized artifacts are
stored per doctor ID and selected at recognition time with general-model
fallback. Large-scale training is gated and not started automatically.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DoctorAuthContext",
    "DoctorProfile",
    "PersonalizationStore",
    "SampleRegistry",
    "ModelRegistry",
    "RecognitionDecision",
    "normalize_doctor_id",
    "create_doctor_profile",
    "select_checkpoint_for_doctor",
    "recognize_for_doctor",
]


def __getattr__(name: str) -> Any:
    if name == "normalize_doctor_id":
        from .doctor_id import normalize_doctor_id

        return normalize_doctor_id
    if name in ("DoctorAuthContext",):
        from .auth import DoctorAuthContext

        return DoctorAuthContext
    if name in ("DoctorProfile", "PersonalizationStore", "create_doctor_profile"):
        from .storage import DoctorProfile, PersonalizationStore, create_doctor_profile

        return {
            "DoctorProfile": DoctorProfile,
            "PersonalizationStore": PersonalizationStore,
            "create_doctor_profile": create_doctor_profile,
        }[name]
    if name == "SampleRegistry":
        from .samples import SampleRegistry

        return SampleRegistry
    if name == "ModelRegistry":
        from .registry import ModelRegistry

        return ModelRegistry
    if name in ("RecognitionDecision", "select_checkpoint_for_doctor", "recognize_for_doctor"):
        from .selector import (
            RecognitionDecision,
            recognize_for_doctor,
            select_checkpoint_for_doctor,
        )

        return {
            "RecognitionDecision": RecognitionDecision,
            "select_checkpoint_for_doctor": select_checkpoint_for_doctor,
            "recognize_for_doctor": recognize_for_doctor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
