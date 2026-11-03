"""
Short demo-only fine-tune of local TrOCR-small on verified private train lines.

- Does NOT use eval_private images (held out).
- Does NOT download a new base model; starts from checkpoints/trocr-small-handwritten.
- RTX 3050 6GB friendly: freeze encoder, batch_size=1, FP16 autocast.
- Saves to checkpoints/trocr-demo-finetuned (weights gitignored).

This is optional adaptation for demo quality — not a full IAM train.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo fine-tune TrOCR on private lines")
    parser.add_argument(
        "--train-csv",
        default=str(ROOT / "demo" / "train_private" / "train_ground_truth.csv"),
    )
    parser.add_argument(
        "--train-dir",
        default=str(ROOT / "demo" / "train_private"),
    )
    parser.add_argument(
        "--base-dir",
        default=str(ROOT / "checkpoints" / "trocr-small-handwritten"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "checkpoints" / "trocr-demo-finetuned"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-target-length", type=int, default=64)
    args = parser.parse_args()

    import os

    os.chdir(ROOT)

    import torch
    from PIL import Image
    from torch.optim import AdamW
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    from disha_htr.logging_config import setup_logging

    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR demo fine-tune ===")

    train_csv = Path(args.train_csv)
    train_dir = Path(args.train_dir)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    if not base_dir.is_dir():
        print(f"Missing base checkpoint: {base_dir}", file=sys.stderr)
        return 2
    if not train_csv.is_file():
        print(f"Missing train CSV: {train_csv}", file=sys.stderr)
        return 2

    rows = []
    with train_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            if name and text and (train_dir / name).is_file():
                rows.append((train_dir / name, text))
    if len(rows) < 3:
        print(f"Need at least 3 train samples, found {len(rows)}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    logger.info("device=%s fp16=%s n_train=%d", device, use_fp16, len(rows))

    processor = TrOCRProcessor.from_pretrained(str(base_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(base_dir))
    model.to(device)
    # Freeze encoder to fit 6GB and train faster
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.train()

    # Decoder / LM head only
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr)

    # Keep pretrained handwritten start token (sep). Forcing cls collapsed
    # held-out generation in diagnosis (nested/gen start=2, cls=0).
    nested = getattr(model.config.decoder, "decoder_start_token_id", None)
    start_id = nested if nested is not None else processor.tokenizer.sep_token_id
    model.config.decoder_start_token_id = start_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    gen = model.generation_config
    gen.decoder_start_token_id = start_id
    gen.pad_token_id = processor.tokenizer.pad_token_id
    gen.eos_token_id = processor.tokenizer.sep_token_id

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for img_path, text in rows:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            labels = processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=args.max_target_length,
                truncation=True,
            ).input_ids
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(pixel_values=pixel_values, labels=labels)
                    loss = out.loss
                loss.backward()
            else:
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss
                loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        avg = total_loss / len(rows)
        history.append({"epoch": epoch, "avg_loss": avg})
        logger.info("epoch %d/%d avg_loss=%.4f", epoch, args.epochs, avg)
        print(f"epoch {epoch}/{args.epochs} avg_loss={avg:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Save full precision for reload compatibility
    model.float()
    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    meta = {
        "base": str(base_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "n_train": len(rows),
        "train_seconds": round(time.perf_counter() - t0, 2),
        "device": str(device),
        "encoder_frozen": True,
        "history": history,
        "note": "Demo adaptation on private verified lines; eval images held out.",
    }
    (out_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned demo checkpoint to {out_dir}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
