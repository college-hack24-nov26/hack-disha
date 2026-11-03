"""Reusable high-level baseline inference API for DISHA HTR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .hardware import detect_hardware, resolve_device
from .model_loader import LoadedModel, load_trocr
from .recognize import RecognitionResult, recognize_image

logger = logging.getLogger("disha_htr.inference")

DEFAULT_SAMPLE = Path("samples/synthetic_line.png")
SAMPLE_INSTRUCTIONS = """
No usable sample image was found.

Place a clear single-line handwriting scan here:
  samples/<your_image>.png  (or .jpg / .jpeg / .webp)

Requirements for a useful baseline sample:
  - One handwritten line only (not a full page)
  - Dark ink on light paper, decent contrast
  - Cropped tightly around the line (little empty margin)
  - No real patient PHI in committed samples

Then run:
  .\\.venv\\Scripts\\python.exe scripts\\test_recognize.py --image samples\\<your_image>.png

Or generate a synthetic (non-handwriting) demo image:
  .\\.venv\\Scripts\\python.exe scripts\\make_sample_image.py
""".strip()


def find_sample_image(
    preferred: str | Path | None = None,
    samples_dir: str | Path = "samples",
) -> Path | None:
    """
    Resolve a sample image path.

    Order: preferred path (if exists) → samples/synthetic_line.png → first image in samples/.
    Returns None if nothing suitable is found.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    if preferred is not None:
        path = Path(preferred)
        if path.is_file():
            return path

    root = Path(samples_dir)
    synthetic = root / "synthetic_line.png"
    if synthetic.is_file():
        return synthetic

    if not root.is_dir():
        return None

    candidates = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    return candidates[0] if candidates else None


def run_baseline_inference(
    image_path: str | Path | None = None,
    model_name: str = "microsoft/trocr-small-handwritten",
    local_dir: str | Path | None = "checkpoints/trocr-small-handwritten",
    cache_dir: str | Path | None = ".cache/huggingface",
    device_preference: str = "auto",
    use_fp16: bool | None = None,
    max_new_tokens: int = 64,
    num_beams: int = 4,
    loaded: LoadedModel | None = None,
) -> dict[str, Any]:
    """
    End-to-end single-line baseline recognition with measured metrics.

    Auto-selects CUDA when available (else CPU). Uses FP16 on CUDA when supported.
    Does not invent recognition text — returns measured model output only.

    Returns a dict with hardware, recognition result, and sample path used.
    Raises FileNotFoundError with SAMPLE_INSTRUCTIONS when no image is available.
    """
    sample = find_sample_image(preferred=image_path)
    if sample is None:
        raise FileNotFoundError(SAMPLE_INSTRUCTIONS)

    hw = detect_hardware(preference=device_preference, require_cuda=False)
    device = resolve_device(preference=device_preference, require_cuda=False)

    if loaded is None:
        loaded = load_trocr(
            model_name=model_name,
            local_dir=local_dir,
            cache_dir=cache_dir,
            device=device,
            use_fp16=use_fp16,
        )

    result: RecognitionResult = recognize_image(
        sample,
        loaded,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    payload = {
        "hardware": hw.to_dict(),
        "recognition": result.to_dict(),
        "sample_image": str(sample.resolve()),
        "sample_note": (
            "Synthetic demo glyphs are not real handwriting; "
            "do not treat output as accuracy evidence."
            if sample.name == "synthetic_line.png"
            else "User-provided sample."
        ),
    }
    logger.info(
        "Baseline done: device=%s text=%r load_s=%.3f infer_s=%.3f",
        result.device,
        result.text,
        result.model_load_time_s,
        result.inference_time_s,
    )
    return payload
