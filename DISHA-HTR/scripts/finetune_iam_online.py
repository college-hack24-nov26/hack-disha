"""
Fine-tune TrOCR-small on Teklia/IAM-line export.

- Starts from checkpoints/trocr-small-handwritten (never overwrites it).
- Saves to checkpoints/trocr-iam-online/ only if val CER improves vs start.
- Correct decoder_start = sep (2). Encoder frozen. FP16. Small batch.

  .\\.venv\\Scripts\\python.exe scripts\\download_teklia_iam_line.py
  .\\.venv\\Scripts\\python.exe scripts\\finetune_iam_online.py --epochs 2
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
    from PIL import ImageEnhance, ImageFilter

    if random.random() < 0.5:
        image = image.rotate(random.uniform(-2.5, 2.5), resample=3, fillcolor=(255, 255, 255))
    if random.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.25:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))
    return image


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


def load_csv(csv_path: Path, img_dir: Path) -> list[tuple[Path, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            path = img_dir / name
            if name and text and path.is_file():
                rows.append((path, text))
    return rows


def cer(ref: str, hyp: str) -> float:
    # local levenshtein
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev = dp[0]
        dp[0] = i
        for j, hc in enumerate(h, 1):
            cur = dp[j]
            if rc == hc:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(r)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train", type=int, default=1500)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import get_scheduler

    from disha_htr.hardware import resolve_device
    from disha_htr.model_loader import load_trocr
    from disha_htr.preprocess import prepare_for_trocr

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_csv = ROOT / "datasets" / "iam" / "hf_teklia" / "train" / "labels.csv"
    val_csv = ROOT / "datasets" / "iam" / "hf_teklia" / "validation" / "labels.csv"
    train_img = train_csv.parent / "images"
    val_img = val_csv.parent / "images"
    if not train_csv.is_file():
        print("Missing IAM export. Run scripts/download_teklia_iam_line.py first.")
        return 2

    train_rows = load_csv(train_csv, train_img)[: args.max_train]
    val_rows = load_csv(val_csv, val_img)[: args.max_val]
    print(f"train={len(train_rows)} val={len(val_rows)}")

    base = ROOT / "checkpoints" / "trocr-small-handwritten"
    out_dir = ROOT / "checkpoints" / "trocr-iam-online"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    # Train in FP32 (GradScaler cannot unscale FP16 parameters).
    loaded = load_trocr(
        local_dir=base,
        cache_dir=ROOT / ".cache" / "huggingface",
        device=device,
        use_fp16=False,
    )
    model = loaded.model
    processor = loaded.processor
    token_info = configure_decoder_tokens(model, processor)
    print("tokens", token_info)

    for p in model.encoder.parameters():
        p.requires_grad = False
    model.train()
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01
    )
    steps = math.ceil(len(train_rows) / args.grad_accum) * args.epochs
    sched = get_scheduler(
        "linear", optim, num_warmup_steps=max(1, steps // 10), num_training_steps=max(1, steps)
    )

    def eval_cer(rows: list[tuple[Path, str]]) -> float:
        model.eval()
        scores = []
        with torch.no_grad():
            for path, text in rows:
                image = prepare_for_trocr(Image.open(path).convert("RGB"))
                pv = processor(images=image, return_tensors="pt").pixel_values
                pv = pv.to(device=device, dtype=loaded.dtype)
                out = model.generate(pv, max_new_tokens=64, num_beams=3)
                hyp = processor.batch_decode(out, skip_special_tokens=True)[0]
                scores.append(cer(text, hyp))
        model.train()
        return float(sum(scores) / max(1, len(scores)))

    print("baseline val CER…")
    best_cer = eval_cer(val_rows)
    print(f"baseline_val_cer={best_cer:.4f}")
    (out_dir / "baseline_metrics.json").write_text(
        json.dumps({"val_cer": best_cer, "token_info": token_info}, indent=2), encoding="utf-8"
    )

    t0 = time.time()
    global_step = 0
    optim.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        running = 0.0
        n_loss = 0
        for i, (path, text) in enumerate(train_rows, 1):
            image = Image.open(path).convert("RGB")
            image = mild_augment(image)
            image = prepare_for_trocr(image)
            pv = processor(images=image, return_tensors="pt").pixel_values
            labels = processor.tokenizer(
                text, return_tensors="pt", padding="max_length", max_length=64, truncation=True
            ).input_ids
            labels[labels == processor.tokenizer.pad_token_id] = -100
            pv = pv.to(device=device, dtype=loaded.dtype)
            labels = labels.to(device)

            loss = model(pixel_values=pv, labels=labels).loss / args.grad_accum
            loss.backward()
            running += float(loss.detach().item()) * args.grad_accum
            n_loss += 1

            if i % args.grad_accum == 0 or i == len(train_rows):
                optim.step()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1
                if global_step % 20 == 0:
                    print(
                        f"epoch {epoch} step {global_step} loss={running/max(1,n_loss):.4f}"
                    )

        val = eval_cer(val_rows)
        print(f"epoch {epoch} val_cer={val:.4f} best={best_cer:.4f}")
        if val < best_cer - 1e-4:
            best_cer = val
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            (out_dir / "train_meta.json").write_text(
                json.dumps(
                    {
                        "base": str(base),
                        "val_cer": best_cer,
                        "epochs_done": epoch,
                        "train_n": len(train_rows),
                        "val_n": len(val_rows),
                        "token_info": token_info,
                        "elapsed_s": round(time.time() - t0, 1),
                        "note": "Do not overwrite trocr-small-handwritten",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("saved improved checkpoint ->", out_dir)
        else:
            print("no improvement; base remains preferred if this CER is worse than held-out policy")

    print("done best_val_cer", best_cer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
