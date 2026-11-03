"""Select personalized vs general checkpoint; recognize with fallback."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .auth import DoctorAuthContext
from .doctor_id import normalize_doctor_id
from .privacy import redact_text
from .registry import ModelRegistry
from .storage import GENERAL_CHECKPOINT_REL, PersonalizationStore

logger = logging.getLogger("disha_htr.personalization.selector")


@dataclass
class RecognitionDecision:
    doctor_id: str
    source: str  # personalized | general | general_fallback
    checkpoint_dir: str
    adapter_version: str | None
    reason: str
    compared_general: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_checkpoint_for_doctor(
    store: PersonalizationStore,
    doctor_id: str,
    *,
    project_root: str | Path,
    require_improves_over_general: bool = True,
) -> RecognitionDecision:
    """
    Choose personalized adapter dir if active + valid; else general checkpoint.

    Never mutates the general checkpoint. If active meta says personalization
    does not improve CER, fall back to general (unless require_improves=False).
    """
    did = normalize_doctor_id(doctor_id)
    root = Path(project_root)
    general = (root / GENERAL_CHECKPOINT_REL).resolve()
    # Allow absolute override stored in profile
    try:
        profile = store.load_profile(did)
        g_rel = profile.general_base_checkpoint
        candidate_general = Path(g_rel)
        general = (
            candidate_general.resolve()
            if candidate_general.is_absolute()
            else (root / g_rel).resolve()
        )
    except FileNotFoundError:
        return RecognitionDecision(
            doctor_id=did,
            source="general",
            checkpoint_dir=str(general),
            adapter_version=None,
            reason="no_doctor_profile",
        )

    registry = ModelRegistry(store)
    version = registry.get_active_version(did)
    if not version:
        return RecognitionDecision(
            doctor_id=did,
            source="general",
            checkpoint_dir=str(general),
            adapter_version=None,
            reason="no_active_personalization",
        )

    adapter_dir = registry.resolve_adapter_dir(did, version)
    if adapter_dir is None:
        return RecognitionDecision(
            doctor_id=did,
            source="general_fallback",
            checkpoint_dir=str(general),
            adapter_version=version,
            reason="active_version_missing_weights",
        )

    meta_path = adapter_dir / "meta.json"
    if meta_path.is_file():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        improves = meta.get("improves_over_general")
        # Placeholder-only artifacts are never used for recognition
        if (adapter_dir / "adapter_placeholder.json").is_file() and not any(
            (adapter_dir / w).is_file()
            for w in ("model.safetensors", "pytorch_model.bin")
        ):
            return RecognitionDecision(
                doctor_id=did,
                source="general_fallback",
                checkpoint_dir=str(general),
                adapter_version=version,
                reason="placeholder_without_weights",
            )
        if require_improves_over_general and improves is False:
            return RecognitionDecision(
                doctor_id=did,
                source="general_fallback",
                checkpoint_dir=str(general),
                adapter_version=version,
                reason="personalized_worse_than_general_on_val_cer",
            )

    return RecognitionDecision(
        doctor_id=did,
        source="personalized",
        checkpoint_dir=str(adapter_dir.resolve()),
        adapter_version=version,
        reason="active_personalization",
    )


def recognize_for_doctor(
    auth: DoctorAuthContext,
    store: PersonalizationStore,
    image_path: str | Path,
    *,
    project_root: str | Path,
    doctor_id: str | None = None,
    compare_general: bool = False,
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Authenticated recognition using doctor personalization selection.

    Loads personalized weights when eligible; otherwise general TrOCR.
    Optional compare_general runs the shared checkpoint for side-by-side text
    (does not log patient data).
    """
    did = normalize_doctor_id(doctor_id or auth.doctor_id)
    auth.require_doctor(did)
    root = Path(project_root)
    decision = select_checkpoint_for_doctor(store, did, project_root=root)

    # Lazy import torch stack only when recognizing
    from disha_htr.hardware import resolve_device
    from disha_htr.model_loader import load_trocr
    from disha_htr.recognize import recognize_image

    cache = cache_dir if cache_dir is not None else root / ".cache" / "huggingface"
    device_obj = resolve_device(preference=device)

    loaded = load_trocr(
        local_dir=decision.checkpoint_dir,
        cache_dir=cache,
        device=device_obj,
    )
    primary = recognize_image(image_path, loaded)
    payload: dict[str, Any] = {
        "doctor_id": did,
        "decision": decision.to_dict(),
        "recognition": primary.to_dict(),
        "text": primary.text,
    }

    if compare_general and decision.source == "personalized":
        general_decision = RecognitionDecision(
            doctor_id=did,
            source="general",
            checkpoint_dir=str((root / GENERAL_CHECKPOINT_REL).resolve()),
            adapter_version=None,
            reason="comparison",
            compared_general=True,
        )
        gen_loaded = load_trocr(
            local_dir=general_decision.checkpoint_dir,
            cache_dir=cache,
            device=device_obj,
        )
        gen_res = recognize_image(image_path, gen_loaded)
        payload["general_comparison"] = {
            "decision": general_decision.to_dict(),
            "text": gen_res.text,
            "recognition": gen_res.to_dict(),
        }
        decision.compared_general = True
        payload["decision"] = decision.to_dict()

    logger.info(
        "doctor=%s source=%s version=%s text=%r",
        did,
        decision.source,
        decision.adapter_version,
        redact_text(primary.text)[:80],
    )
    store.audit_log(did).append(
        "recognition",
        {
            "doctor_id": did,
            "source": decision.source,
            "adapter_version": decision.adapter_version,
            "image_name": Path(image_path).name,
            "text_len": len(primary.text or ""),
        },
    )
    return payload
