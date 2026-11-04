"""Single-line handwriting recognition using TrOCR."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .model_loader import LoadedModel
from .preprocess import load_image, prepare_for_trocr
from .wer_normalize import wer_normalize_hypothesis

logger = logging.getLogger("disha_htr.recognize")


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    image_path: str
    device: str
    model_name: str
    model_source: str
    dtype: str
    use_fp16: bool
    model_load_time_s: float
    inference_time_s: float
    gpu_memory_allocated_mb: float | None
    gpu_memory_peak_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gpu_memory_mb(device: torch.device) -> tuple[float | None, float | None]:
    """Return (allocated_mb, peak_allocated_mb) for CUDA, else (None, None)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return round(allocated, 2), round(peak, 2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read CUDA memory stats: %s", exc)
        return None, None


def recognize_image(
    image_path: str | Path,
    loaded: LoadedModel,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> RecognitionResult:
    """Run TrOCR on a single image path and return the decoded text + metrics."""
    path = Path(image_path)
    logger.info(
        "Recognizing: %s on %s dtype=%s fp16=%s",
        path,
        loaded.device,
        loaded.dtype,
        loaded.use_fp16,
    )

    if loaded.device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(loaded.device)
            torch.cuda.synchronize(loaded.device)
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA sync/reset skipped: %s", exc)

    try:
        image = prepare_for_trocr(load_image(path))
        pixel_values = loaded.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=loaded.device, dtype=loaded.dtype)

        if loaded.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(loaded.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            if loaded.use_fp16 and loaded.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    generated_ids = loaded.model.generate(
                        pixel_values,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
            else:
                generated_ids = loaded.model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )

        if loaded.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(loaded.device)
        inference_time_s = time.perf_counter() - t0

        text = loaded.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = wer_normalize_hypothesis((text or "").strip())
    except FileNotFoundError:
        logger.exception("Image not found: %s", path)
        raise
    except Exception:
        logger.exception("Recognition failed for %s", path)
        raise

    allocated_mb, peak_mb = _gpu_memory_mb(loaded.device)
    logger.info(
        "Raw output: %r | inference_time_s=%.3f | gpu_peak_mb=%s",
        text,
        inference_time_s,
        peak_mb,
    )
    return RecognitionResult(
        text=text,
        image_path=str(path),
        device=str(loaded.device),
        model_name=loaded.name,
        model_source=loaded.source,
        dtype=str(loaded.dtype).replace("torch.", ""),
        use_fp16=loaded.use_fp16,
        model_load_time_s=round(loaded.load_time_s, 4),
        inference_time_s=round(inference_time_s, 4),
        gpu_memory_allocated_mb=allocated_mb,
        gpu_memory_peak_mb=peak_mb,
    )
