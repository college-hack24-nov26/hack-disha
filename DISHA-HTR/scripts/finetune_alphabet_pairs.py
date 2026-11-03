"""Short adaptation on alphabet-pair lines (Aa Bb Cc …)."""

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


def configure_decoder_tokens(model, processor) -> None:
    tok = processor.tokenizer
    start_id = getattr(model.config.decoder, "decoder_start_token_id", None) or tok.sep_token_id
    model.config.decoder_start_token_id = start_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id
    model.generation_config.decoder_start_token_id = start_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.max_new_tokens = 64


def load_split(csv_path: Path, img_dir: Path, split: str):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--init",
        default=str(ROOT / "checkpoints" / "trocr-domain-70-best-heldout"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "checkpoints" / "trocr-domain-70-best-heldout"),
    )
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.optim import AdamW
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

    csv_path = ROOT / "training" / "alphabet_pairs" / "labels.csv"
    img_dir = ROOT / "training" / "alphabet_pairs" / "images"
    if not csv_path.is_file():
        print("Run scripts/make_alphabet_train_set.py first")
        return 2

    train_rows = load_split(csv_path, img_dir, "train")
    val_rows = load_split(csv_path, img_dir, "val")
    # Mix a few improved domain lines so we don't forget English words entirely.
    improved_csv = ROOT / "demo" / "train_improved" / "train_ground_truth.csv"
    improved_img = ROOT / "demo" / "train_improved" / "images"
    if improved_csv.is_file():
        with improved_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("image_filename") or "").strip()
                text = (row.get("ground_truth") or "").strip()
                path = improved_img / name
                if name and text and path.is_file():
                    train_rows.append((path, text))
    print(f"train={len(train_rows)} val={len(val_rows)}")

    init = Path(args.init)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(str(init))
    model = VisionEncoderDecoderModel.from_pretrained(str(init))
    configure_decoder_tokens(model, processor)
    model.to(device=device, dtype=torch.float32)
    model.train()

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train_rows) / args.grad_accum) * args.epochs
    sched = get_scheduler(
        "cosine",
        optimizer=optim,
        num_warmup_steps=max(1, steps // 10),
        num_training_steps=max(1, steps),
    )

    def eval_acc() -> float:
        from disha_htr.metrics import character_error_rate

        model.eval()
        total = 0.0
        with torch.inference_mode():
            for path, text in val_rows:
                image = Image.open(path).convert("RGB")
                pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
                out_ids = model.generate(pv, max_new_tokens=64, num_beams=4)
                pred = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                total += character_error_rate(text, pred)
        model.train()
        return 1.0 - (total / max(len(val_rows), 1))

    best = eval_acc()
    print(f"baseline_val_char_acc={best*100:.2f}%")
    t0 = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        running = 0.0
        n = 0
        optim.zero_grad(set_to_none=True)
        for i, (path, text) in enumerate(train_rows, 1):
            image = Image.open(path).convert("RGB")
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
            n += 1
            if i % args.grad_accum == 0 or i == len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
        acc = eval_acc()
        entry = {
            "epoch": epoch,
            "loss": round(running / max(n, 1), 4),
            "val_char_acc_pct": round(acc * 100, 2),
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(entry)
        print(entry)
        if acc >= best - 1e-6:
            best = acc
            model.save_pretrained(out)
            processor.save_pretrained(out)
            snap = {
                "heldout_char_acc_pct": None,
                "alphabet_val_char_acc_pct": round(best * 100, 2),
                "note": "Alphabet-pair adaptation; re-check held-out after save.",
                "history": history,
            }
            # Preserve prior held-out score if present.
            old = out / "heldout_snapshot.json"
            if old.is_file():
                try:
                    prev = json.loads(old.read_text(encoding="utf-8"))
                    if prev.get("heldout_char_acc_pct") is not None:
                        snap["heldout_char_acc_pct"] = prev["heldout_char_acc_pct"]
                        snap["heldout_detail"] = prev.get("heldout_detail")
                except Exception:  # noqa: BLE001
                    pass
            (out / "alphabet_adapt_meta.json").write_text(
                json.dumps(snap, indent=2), encoding="utf-8"
            )
            print("saved", out)

    print("done best_val_char_acc", round(best * 100, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
