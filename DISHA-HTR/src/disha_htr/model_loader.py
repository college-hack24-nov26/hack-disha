"""TrOCR model loading with local checkpoint preference and FP16 on CUDA."""

from __future__ import annotations

import logging
import time
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
    dtype: torch.dtype
    use_fp16: bool
    load_time_s: float


def _has_local_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    present = {p.name for p in path.iterdir() if p.is_file()}
    has_config = "config.json" in present
    has_weights = bool(
        present.intersection({"pytorch_model.bin", "model.safetensors"})
    )
    # transformers 5.x may write processor_config.json; older used preprocessor_config.json
    has_processor = bool(
        present.intersection({"processor_config.json", "preprocessor_config.json"})
    )
    return has_config and has_weights and has_processor


def _fp16_supported(device: torch.device) -> bool:
    """True when CUDA device can run FP16 inference."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        # Capability >= 7.0 generally has solid FP16; RTX 3050 is 8.6
        major, _minor = torch.cuda.get_device_capability(device)
        return major >= 7
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not query CUDA capability for FP16: %s", exc)
        return False


def load_trocr(
    model_name: str = DEFAULT_MODEL_NAME,
    local_dir: str | Path | None = "checkpoints/trocr-small-handwritten",
    cache_dir: str | Path | None = ".cache/huggingface",
    device: torch.device | str | None = None,
    use_fp16: bool | None = None,
) -> LoadedModel:
    """
    Load TrOCR processor and VisionEncoderDecoderModel.

    Prefers a populated local_dir (preserves checkpoints). Otherwise downloads
    from Hugging Face Hub into cache_dir.

    On CUDA, enables FP16 when supported unless use_fp16=False.
    Records wall-clock load_time_s (processor + model + device move).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    if use_fp16 is None:
        use_fp16 = _fp16_supported(device)
    elif use_fp16 and device.type != "cuda":
        logger.warning("FP16 requested on non-CUDA device; using float32")
        use_fp16 = False
    elif use_fp16 and not _fp16_supported(device):
        logger.warning("FP16 not supported on this CUDA device; using float32")
        use_fp16 = False

    dtype = torch.float16 if use_fp16 else torch.float32
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

    t0 = time.perf_counter()
    try:
        processor = TrOCRProcessor.from_pretrained(load_id, cache_dir=cache)
        model = VisionEncoderDecoderModel.from_pretrained(load_id, cache_dir=cache)
    except Exception:
        logger.exception("Failed to load TrOCR model from %s", load_id)
        raise

    try:
        model.to(device=device, dtype=dtype)
    except Exception:
        logger.exception(
            "Failed to move model to %s with dtype=%s — retrying float32",
            device,
            dtype,
        )
        if use_fp16:
            use_fp16 = False
            dtype = torch.float32
            model.to(device=device, dtype=dtype)
        else:
            raise

    model.eval()
    load_time_s = time.perf_counter() - t0
    logger.info(
        "TrOCR ready on %s dtype=%s fp16=%s source=%s load_time_s=%.3f",
        device,
        dtype,
        use_fp16,
        source,
        load_time_s,
    )

    # Persist Hub download into local_dir when we loaded from Hub and dir is incomplete
    if source == model_name and local_dir is not None:
        local_path = Path(local_dir)
        try:
            local_path.mkdir(parents=True, exist_ok=True)
            if not _has_local_weights(local_path):
                logger.info("Saving model snapshot to %s for reuse", local_path)
                # Save in full precision for checkpoint reuse
                processor.save_pretrained(local_path)
                model.float().save_pretrained(local_path)
                model.to(device=device, dtype=dtype)
        except OSError as exc:
            logger.warning("Could not persist checkpoint to %s: %s", local_path, exc)

    return LoadedModel(
        name=model_name,
        source=source,
        processor=processor,
        model=model,
        device=device,
        dtype=dtype,
        use_fp16=use_fp16,
        load_time_s=load_time_s,
    )
