"""DISHA Handwriting Text Recognition (HTR) — standalone TrOCR module."""

__version__ = "0.1.0"

from .hardware import detect_hardware, resolve_device
from .model_loader import load_trocr
from .recognize import recognize_image

__all__ = [
    "__version__",
    "detect_hardware",
    "resolve_device",
    "load_trocr",
    "recognize_image",
]
