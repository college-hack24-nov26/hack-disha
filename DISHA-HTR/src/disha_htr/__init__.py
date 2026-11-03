"""DISHA Handwriting Text Recognition (HTR) — standalone TrOCR module."""

__version__ = "0.1.1"

from .hardware import detect_hardware, resolve_device
from .inference import find_sample_image, run_baseline_inference
from .model_loader import load_trocr
from .recognize import recognize_image

__all__ = [
    "__version__",
    "detect_hardware",
    "resolve_device",
    "load_trocr",
    "recognize_image",
    "find_sample_image",
    "run_baseline_inference",
]
