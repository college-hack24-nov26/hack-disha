"""DISHA Handwriting Text Recognition (HTR) — standalone TrOCR module."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "detect_hardware",
    "resolve_device",
    "load_trocr",
    "recognize_image",
    "find_sample_image",
    "run_baseline_inference",
]


def __getattr__(name: str) -> Any:
    """Lazy exports so metrics/eval_data imports do not load Torch."""
    if name in ("detect_hardware", "resolve_device"):
        from .hardware import detect_hardware, resolve_device

        return {
            "detect_hardware": detect_hardware,
            "resolve_device": resolve_device,
        }[name]
    if name == "load_trocr":
        from .model_loader import load_trocr

        return load_trocr
    if name == "recognize_image":
        from .recognize import recognize_image

        return recognize_image
    if name in ("find_sample_image", "run_baseline_inference"):
        from .inference import find_sample_image, run_baseline_inference

        return {
            "find_sample_image": find_sample_image,
            "run_baseline_inference": run_baseline_inference,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
