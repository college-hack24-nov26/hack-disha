"""TrOCR model loading with local checkpoint preference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger("disha_htr.model_loader")

DEFAULT_MODEL_NAME = "microsoft/trocr-small-handwritten"


@dataclass
class LoadedModel:
    """Loaded processor + model pair bound to a device."""

    name: str
    source: str
    processor: Any
    model: VisionEncoderDecoderModel
    device: torch.device


def _has_local_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = (
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "preprocessor_config.json",
    )
    present = {p.name for p in path.iterdir() if p.is_file()}
    return "config.json" in present and bool(
        present.intersection({"pytorch_model.bin", "model.safetensors"})
    )


def load_trocr(
    model_name: str = DEFAULT_MODEL_NAME,
    local_dir: str | Path | None = "checkpoints/trocr-small-handwritten",
    cache_dir: str | Path | None = ".cache/huggingface",
    device: torch.device | str | None = None,
) -> LoadedModel:
    """
    Load TrOCR processor and VisionEncoderDecoderModel.

    Prefers a populated local_dir (preserves checkpoints). Otherwise downloads
    from Hugging Face Hub into cache_dir.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    cache = str(cache_dir) if cache_dir else None
    source = model_name
    load_id = model_name

    if local_dir is not None:
        local_path = Path(local_dir)
        if _has_local_weights(local_path):
            load_id = str(local_path)
            source = f"local:{local_path}"
            logger.info("Loading TrOCR from local checkpoint: %s", local_path)
        else:
            logger.info(
                "Local checkpoint missing or incomplete at %s — loading '%s' from Hub",
                local_path,
                model_name,
            )

    try:
        processor = TrOCRProcessor.from_pretrained(load_id, cache_dir=cache)
        model = VisionEncoderDecoderModel.from_pretrained(load_id, cache_dir=cache)
    except Exception:
        logger.exception("Failed to load TrOCR model from %s", load_id)
        raise

    model.to(device)
    model.eval()
    logger.info("TrOCR ready on %s (source=%s)", device, source)

    # Persist Hub download into local_dir when we loaded from Hub and dir is empty-ish
    if source == model_name and local_dir is not None:
        local_path = Path(local_dir)
        try:
            local_path.mkdir(parents=True, exist_ok=True)
            if not _has_local_weights(local_path):
                logger.info("Saving model snapshot to %s for reuse", local_path)
                processor.save_pretrained(local_path)
                model.save_pretrained(local_path)
        except OSError as exc:
            logger.warning("Could not persist checkpoint to %s: %s", local_path, exc)

    return LoadedModel(
        name=model_name,
        source=source,
        processor=processor,
        model=model,
        device=device,
    )
