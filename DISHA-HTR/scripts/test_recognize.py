"""Basic single-line handwriting recognition smoke test with metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.inference import SAMPLE_INSTRUCTIONS, run_baseline_inference  # noqa: E402
from disha_htr.logging_config import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="DISHA HTR single-line recognition test")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image (default: auto-discover under samples/)",
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
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="Force FP16 (CUDA only)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 even on CUDA",
    )
    args = parser.parse_args()

    import os

    os.chdir(ROOT)

    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR baseline inference ===")

    try:
        payload = run_baseline_inference(
            image_path=args.image,
            model_name=args.model,
            local_dir=args.local_dir,
            cache_dir=ROOT / ".cache" / "huggingface",
            device_preference=args.device,
            use_fp16=args.fp16,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(str(exc), file=sys.stderr)
        if "No usable sample" in str(exc) or "Place a clear" in str(exc):
            print(SAMPLE_INSTRUCTIONS, file=sys.stderr)
        return 2

    print("\n=== Hardware (verified) ===")
    print(json.dumps(payload["hardware"], indent=2))
    print("\n=== Recognition result (actual) ===")
    print(json.dumps(payload["recognition"], indent=2))
    print("\n=== Sample ===")
    print(json.dumps(
        {"sample_image": payload["sample_image"], "sample_note": payload["sample_note"]},
        indent=2,
    ))
    logger.info("Baseline inference completed successfully")
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
