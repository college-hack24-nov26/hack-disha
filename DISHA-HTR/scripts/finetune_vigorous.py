"""
Vigorous domain fine-tune aiming for >=50% held-out character accuracy.

- Starts from checkpoints/trocr-small-handwritten (never overwrites it).
- Correct decoder_start = sep (2).
- Unfreezes encoder+decoder with differential LRs (domain shift needs encoder).
- Trains on training/vigorous (IAM + notebook composites + local domain).
- Saves best-val CER to checkpoints/trocr-vigorous-50/.
- Does NOT read held-out images during training.
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


def configure_decoder_tokens(model, processor) -> dict:
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
    }


def load_split(csv_path: Path, img_dir: Path, split: str) -> list[tuple[Path, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("split") or "").strip() != split:
                continue
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            path = img_dir / name
            if name and text and path.is_file():
                rows.append((path, text))
    return rows


def mild_aug(image, rng: random.Random):
    from PIL import ImageEnhance, ImageFilter

    if rng.random() < 0.5:
        image = image.rotate(rng.uniform(-2.5, 2.5), resample=3, fillcolor=(245, 245, 245))
    if rng.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.2))
    if rng.random() < 0.25:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.6)))
    return image


def eval_cer(model, processor, pairs, device) -> float:
    from disha_htr.metrics import character_error_rate
    import torch
    from PIL import Image

    model.eval()
    total = 0.0
    with torch.inference_mode():
        for path, text in pairs:
            image = Image.open(path).convert("RGB")
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            out = model.generate(pv, max_new_tokens=64, num_beams=4)
            pred = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            total += character_error_rate(text, pred)
    model.train()
    return total / max(len(pairs), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all")
    parser.add_argument("--max-val", type=int, default=120)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.optim import AdamW
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_path = ROOT / "training" / "vigorous" / "labels.csv"
    img_dir = ROOT / "training" / "vigorous" / "images"
    if not csv_path.is_file():
        print("Missing vigorous set. Run scripts/build_vigorous_train_set.py")
        return 2

    train_rows = load_split(csv_path, img_dir, "train")
    val_rows = load_split(csv_path, img_dir, "val")
    if args.max_train > 0:
        train_rows = train_rows[: args.max_train]
    val_rows = val_rows[: args.max_val]
    print(f"train={len(train_rows)} val={len(val_rows)}")

    base = ROOT / "checkpoints" / "trocr-small-handwritten"
    out_dir = ROOT / "checkpoints" / "trocr-vigorous-50"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(str(base))
    model = VisionEncoderDecoderModel.from_pretrained(str(base))
    token_meta = configure_decoder_tokens(model, processor)
    print("tokens", token_meta)

    # Full fine-tune in FP32 (safe on GradScaler); freeze nothing.
    model.to(device=device, dtype=torch.float32)
    model.train()

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    optim = AdamW(
        [
            {"params": enc_params, "lr": args.encoder_lr},
            {"params": dec_params, "lr": args.decoder_lr},
        ],
        weight_decay=0.01,
    )
    steps = math.ceil(len(train_rows) / args.grad_accum) * args.epochs
    sched = get_scheduler(
        "cosine",
        optimizer=optim,
        num_warmup_steps=max(1, steps // 10),
        num_training_steps=max(1, steps),
    )

    print("baseline val CER…")
    best = eval_cer(model, processor, val_rows, device)
    print(f"baseline_val_cer={best:.4f}")
    bad = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        running = 0.0
        n_loss = 0
        optim.zero_grad(set_to_none=True)
        for i, (path, text) in enumerate(train_rows, 1):
            image = Image.open(path).convert("RGB")
            image = mild_aug(image, rng)
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            labels = processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            ).input_ids
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels = labels.to(device)
            loss = model(pixel_values=pv, labels=labels).loss / args.grad_accum
            loss.backward()
            running += float(loss.detach().item()) * args.grad_accum
            n_loss += 1
            if i % args.grad_accum == 0 or i == len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            if i % 200 == 0:
                print(f"  epoch {epoch} item {i}/{len(train_rows)} loss={running/max(1,n_loss):.4f}")

        val = eval_cer(model, processor, val_rows, device)
        entry = {
            "epoch": epoch,
            "train_loss": round(running / max(1, n_loss), 6),
            "val_cer": round(val, 6),
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(entry)
        print(f"epoch {epoch} val_cer={val:.4f} best={best:.4f} loss={entry['train_loss']}")
        if val < best - 1e-4:
            best = val
            bad = 0
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            (out_dir / "train_meta.json").write_text(
                json.dumps(
                    {
                        "best_val_cer": best,
                        "epoch": epoch,
                        "token_meta": token_meta,
                        "n_train": len(train_rows),
                        "n_val": len(val_rows),
                        "history": history,
                        "base": str(base),
                        "note": "Do not overwrite trocr-small-handwritten",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("saved", out_dir)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    print("done best_val_cer", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
