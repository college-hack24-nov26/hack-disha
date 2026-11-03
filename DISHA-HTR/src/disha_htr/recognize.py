"""Single-line handwriting recognition using TrOCR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from .model_loader import LoadedModel
from .preprocess import load_image, prepare_for_trocr

logger = logging.getLogger("disha_htr.recognize")


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    image_path: str
    device: str
    model_name: str
    model_source: str


def recognize_image(
    image_path: str | Path,
    loaded: LoadedModel,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> RecognitionResult:
    """Run TrOCR on a single image path and return the decoded text."""
    path = Path(image_path)
    logger.info("Recognizing: %s on %s", path, loaded.device)

    try:
        image = prepare_for_trocr(load_image(path))
        pixel_values = loaded.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(loaded.device)

        with torch.inference_mode():
            generated_ids = loaded.model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        text = loaded.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = (text or "").strip()
    except Exception:
        logger.exception("Recognition failed for %s", path)
        raise

    logger.info("Raw output: %r", text)
    return RecognitionResult(
        text=text,
        image_path=str(path),
        device=str(loaded.device),
        model_name=loaded.name,
        model_source=loaded.source,
    )
