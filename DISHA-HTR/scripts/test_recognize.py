"""Basic single-line handwriting recognition smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.hardware import detect_hardware  # noqa: E402
from disha_htr.logging_config import setup_logging  # noqa: E402
from disha_htr.model_loader import load_trocr  # noqa: E402
from disha_htr.recognize import recognize_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="DISHA HTR single-line recognition test")
    parser.add_argument(
        "--image",
        default=str(ROOT / "samples" / "synthetic_line.png"),
        help="Path to input image",
    )
    parser.add_argument(
        "--model",
        default="microsoft/trocr-small-handwritten",
        help="HF model id",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device preference",
    )
    parser.add_argument(
        "--local-dir",
        default=str(ROOT / "checkpoints" / "trocr-small-handwritten"),
        help="Local checkpoint directory",
    )
    args = parser.parse_args()

    # Run relative to project root so config paths resolve
    import os

    os.chdir(ROOT)

    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR smoke test ===")

    hw = detect_hardware(preference=args.device, require_cuda=False)
    print("\n=== Hardware (verified) ===")
    print(json.dumps(hw.to_dict(), indent=2))

    image_path = Path(args.image)
    if not image_path.is_file():
        logger.error("Sample image missing: %s — run scripts/make_sample_image.py first", image_path)
        print(f"ERROR: sample image not found: {image_path}", file=sys.stderr)
        return 2

    loaded = load_trocr(
        model_name=args.model,
        local_dir=args.local_dir,
        cache_dir=ROOT / ".cache" / "huggingface",
        device=hw.selected_device,
    )

    result = recognize_image(image_path, loaded)
    print("\n=== Recognition result (actual) ===")
    print(json.dumps(
        {
            "image": result.image_path,
            "text": result.text,
            "device": result.device,
            "model_name": result.model_name,
            "model_source": result.model_source,
        },
        indent=2,
    ))
    logger.info("Smoke test completed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 — surface honest failure
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
