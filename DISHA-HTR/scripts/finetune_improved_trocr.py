"""
Improved TrOCR fine-tune aimed at held-out generalization.

Techniques (licenses):
- MIT-adapted ideas from rsommerfeld/trocr (token init pattern, linear LR
  schedule via transformers get_scheduler): see docs/EXTERNAL_CODE_ATTRIBUTION.md
- ertugd/trocr-handwriting and Tripathijii147 used as technical references only
  (no LICENSE) — not copied: sep decoder_start matching pretrained, warmup,
  mild PIL augment, best-val CER checkpoint, early stopping.

Constraints:
- Starts from checkpoints/trocr-small-handwritten (does not overwrite it).
- Never reads held-out images for training.
- Saves to checkpoints/trocr-improved-gen/
- RTX 3050 6GB: freeze encoder, batch=1, grad accum, FP16 autocast.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def mild_augment(image):
    """Label-preserving PIL augment (no torchvision required)."""
    from PIL import ImageEnhance, ImageFilter

    if random.random() < 0.5:
        angle = random.uniform(-2.5, 2.5)
        image = image.rotate(angle, resample=3, fillcolor=(255, 255, 255))  # BICUBIC
    if random.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.25:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))
    return image


def configure_decoder_tokens(model, processor) -> dict:
    """Keep pretrained handwritten start token (sep=2), sync generation_config."""
    tok = processor.tokenizer
    nested = getattr(model.config.decoder, "decoder_start_token_id", None)
    start_id = nested if nested is not None else tok.sep_token_id
    model.config.decoder_start_token_id = start_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    gen = model.generation_config
    gen.decoder_start_token_id = start_id
    gen.pad_token_id = tok.pad_token_id
    gen.eos_token_id = tok.sep_token_id
    gen.max_new_tokens = 64
    return {
        "decoder_start_token_id": start_id,
        "eos_token_id": tok.sep_token_id,
        "pad_token_id": tok.pad_token_id,
        "cls_token_id": tok.cls_token_id,
        "sep_token_id": tok.sep_token_id,
    }


def load_rows(csv_path: Path, img_dir: Path) -> list[tuple[Path, str, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            page = (row.get("source_page") or "").strip()
            path = img_dir / name
            if name and text and path.is_file():
                rows.append((path, text, page))
    return rows


def writer_split(rows, val_ratio=0.2, seed=42):
    """Hold out whole source pages for validation when possible."""
    rng = random.Random(seed)
    pages = sorted({p for _, _, p in rows})
    rng.shuffle(pages)
    n_val = max(1, int(round(len(pages) * val_ratio)))
    val_pages = set(pages[:n_val])
    train = [r for r in rows if r[2] not in val_pages]
    val = [r for r in rows if r[2] in val_pages]
    if len(train) < 3 or len(val) < 1:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * val_ratio))
        return shuffled[cut:], shuffled[:cut], {"mode": "random"}
    return train, val, {"mode": "page", "val_pages": sorted(val_pages)}


def eval_cer(model, processor, pairs, device, max_target_length=64) -> float:
    from disha_htr.metrics import character_error_rate
    from PIL import Image
    import torch

    model.eval()
    total = 0.0
    with torch.inference_mode():
        for path, text, _ in pairs:
            image = Image.open(path).convert("RGB")
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            out = model.generate(pv, max_new_tokens=max_target_length, num_beams=3)
            pred = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            total += character_error_rate(text, pred)
    model.train()
    return total / max(len(pairs), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Improved generalization fine-tune")
    parser.add_argument(
        "--train-csv",
        default=str(ROOT / "demo" / "train_improved" / "train_ground_truth.csv"),
    )
    parser.add_argument(
        "--train-dir",
        default=str(ROOT / "demo" / "train_improved" / "images"),
    )
    parser.add_argument(
        "--base-dir",
        default=str(ROOT / "checkpoints" / "trocr-small-handwritten"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "checkpoints" / "trocr-improved-gen"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    import os

    os.chdir(ROOT)
    random.seed(args.seed)

    import torch
    from PIL import Image
    from torch.optim import AdamW
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

    from disha_htr.logging_config import setup_logging

    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR improved fine-tune ===")

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    train_csv = Path(args.train_csv)
    train_dir = Path(args.train_dir)

    if not base_dir.is_dir():
        print(f"Missing base checkpoint: {base_dir}", file=sys.stderr)
        return 2
    rows = load_rows(train_csv, train_dir)
    if len(rows) < 4:
        print(f"Need >=4 train samples, found {len(rows)}", file=sys.stderr)
        return 2

    train_rows, val_rows, split_meta = writer_split(rows, seed=args.seed)
    logger.info(
        "n_all=%d n_train=%d n_val=%d split=%s",
        len(rows),
        len(train_rows),
        len(val_rows),
        split_meta,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    processor = TrOCRProcessor.from_pretrained(str(base_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(base_dir))
    token_meta = configure_decoder_tokens(model, processor)
    logger.info("token_config=%s", token_meta)

    for p in model.encoder.parameters():
        p.requires_grad = False
    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_rows) / args.grad_accum)
    num_training_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(args.warmup_ratio * num_training_steps))
    # MIT-adapted linear schedule pattern from rsommerfeld/trocr/src/scripts.py
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=num_training_steps,
    )

    history = []
    best_val = float("inf")
    best_epoch = 0
    bad = 0
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, (img_path, text, _) in enumerate(train_rows, start=1):
            image = Image.open(img_path).convert("RGB")
            if not args.no_augment:
                image = mild_augment(image)
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

            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(pixel_values=pixel_values, labels=labels)
                    loss = out.loss / args.grad_accum
                loss.backward()
            else:
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss / args.grad_accum
                loss.backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum

            if i % args.grad_accum == 0 or i == len(train_rows):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / len(train_rows)
        val_cer = eval_cer(
            model, processor, val_rows, device, args.max_target_length
        )
        entry = {
            "epoch": epoch,
            "avg_train_loss": round(avg_loss, 6),
            "val_cer": round(val_cer, 6),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        history.append(entry)
        logger.info(
            "epoch %d/%d loss=%.4f val_cer=%.4f",
            epoch,
            args.epochs,
            avg_loss,
            val_cer,
        )
        print(
            f"epoch {epoch}/{args.epochs} loss={avg_loss:.4f} val_cer={val_cer:.4f}"
        )

        if val_cer < best_val - 1e-4:
            best_val = val_cer
            best_epoch = epoch
            bad = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            model.float()
            processor.save_pretrained(out_dir)
            model.save_pretrained(out_dir)
            model.to(device)
            (out_dir / "train_history.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "best_val_cer": best_val,
                        "history": history,
                        "token_config": token_meta,
                        "split": split_meta,
                        "n_train": len(train_rows),
                        "n_val": len(val_rows),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("saved best checkpoint epoch=%d val_cer=%.4f", epoch, val_cer)
        else:
            bad += 1
            if bad >= args.patience:
                logger.info(
                    "early stop at epoch %d (patience=%d)", epoch, args.patience
                )
                print(f"early stop at epoch {epoch}")
                break

    elapsed = time.perf_counter() - t0
    peak_mb = None
    if device.type == "cuda":
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    meta = {
        "base": str(base_dir),
        "out": str(out_dir),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_cer": best_val,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "grad_accum": args.grad_accum,
        "augment": not args.no_augment,
        "freeze_encoder": True,
        "token_config": token_meta,
        "split": split_meta,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_seconds": round(elapsed, 1),
        "gpu_peak_mb": peak_mb,
        "history": history,
        "attribution": "See docs/EXTERNAL_CODE_ATTRIBUTION.md",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: meta[k]
                for k in (
                    "best_epoch",
                    "best_val_cer",
                    "train_seconds",
                    "gpu_peak_mb",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
