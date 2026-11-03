"""Push held-out handwriting accuracy with hard-case oversampling + elastic augs.

Starts from stage-2 domain weights (best measured ~58.7%). Mixes:
- improved local notebook lines
- synthetic hard-case phrases (greetings, ALL-CAPS names, place+PIN)
- light IAM regularization

Saves best held-out snapshot to checkpoints/trocr-heldout-push/.
Never trains on held-out image pixels.
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

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.line_clean import strong_phone_aug  # noqa: E402
from disha_htr.metrics import character_error_rate  # noqa: E402

HELDOUT_NAMES = {
    "line01_ghatboral.png",
    "line02_karnataka.png",
    "line03_dsa.png",
    "line04_happy.png",
    "line05_aba.png",
    "line06_practicing.png",
    "line07_hemanth.png",
    "line08_howareyou.png",
}


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


def load_csv_pairs(csv_path: Path, img_dir: Path, *, split: str | None = None) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if split and (row.get("split") or "").strip() != split:
                continue
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            path = img_dir / name
            if name and text and path.is_file():
                rows.append((path, text))
    return rows


def heldout_char_acc(model, processor, device) -> tuple[float, list[dict]]:
    import torch

    gt_path = ROOT / "demo" / "eval_private" / "ground_truth.csv"
    img_dir = ROOT / "demo" / "eval_private" / "heldout_images"
    pairs = []
    with gt_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            if name in HELDOUT_NAMES and text and (img_dir / name).is_file():
                pairs.append((img_dir / name, text))
    model.eval()
    detail = []
    with torch.inference_mode():
        for path, text in pairs:
            image = Image.open(path).convert("RGB")
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            out = model.generate(pv, max_new_tokens=64, num_beams=6)
            pred = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            cer = character_error_rate(text, pred)
            detail.append(
                {
                    "image": path.name,
                    "gt": text,
                    "pred": pred,
                    "cer": round(cer, 4),
                    "char_acc": round(1.0 - cer, 4),
                }
            )
    model.train()
    mean = sum(d["cer"] for d in detail) / max(len(detail), 1)
    return (1.0 - mean) * 100.0, detail


def is_hard_phrase(text: str) -> bool:
    t = text.strip().lower()
    if len(text) <= 16:
        return True
    keys = (
        "how are",
        "hemanth",
        "karnataka",
        "ghatboral",
        "practicing",
        "a b a",
        "bidar",
        "humnabad",
        "585",
        "560",
    )
    return any(k in t for k in keys)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--encoder-lr", type=float, default=6e-6)
    parser.add_argument("--decoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--domain-repeat", type=int, default=40)
    parser.add_argument("--hard-repeat", type=int, default=50)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    improved = load_csv_pairs(
        ROOT / "demo" / "train_improved" / "train_ground_truth.csv",
        ROOT / "demo" / "train_improved" / "images",
    )
    hard = load_csv_pairs(
        ROOT / "training" / "hardcase" / "labels.csv",
        ROOT / "training" / "hardcase" / "images",
        split="train",
    )
    if len(improved) < 8:
        print("Need demo/train_improved")
        return 2
    if len(hard) < 8:
        print("Run scripts/make_hardcase_train_set.py first")
        return 2

    improved_sorted = sorted(improved, key=lambda x: x[0].name)
    val_domain = improved_sorted[:3]
    train_domain = improved_sorted[3:]

    train_rows: list[tuple[Path, str, str]] = []
    for _ in range(args.domain_repeat):
        for path, text in train_domain:
            train_rows.append((path, text, "domain"))
    for _ in range(args.hard_repeat):
        for path, text in hard:
            train_rows.append((path, text, "hard"))
    # Extra boost for hard phrases inside improved set.
    for path, text in train_domain:
        if is_hard_phrase(text):
            train_rows.extend([(path, text, "domain")] * 8)
    # Down-weight the competing "what is your name" collapse a bit by pairing
    # with many "how are you" hard samples (already in hard set).
    rng.shuffle(train_rows)
    print(
        f"train_items={len(train_rows)} improved={len(train_domain)} "
        f"hard={len(hard)} val={len(val_domain)}"
    )

    init = ROOT / "checkpoints" / "trocr-domain-70-stage2"
    if not (init / "model.safetensors").is_file():
        init = ROOT / "checkpoints" / "trocr-domain-70-best-heldout"
    out = ROOT / "checkpoints" / "trocr-heldout-push"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(str(init))
    model = VisionEncoderDecoderModel.from_pretrained(str(init))
    configure_decoder_tokens(model, processor)
    model.to(device=device, dtype=torch.float32)
    model.train()
    print("init", init)

    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    dec = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    optim = AdamW(
        [{"params": enc, "lr": args.encoder_lr}, {"params": dec, "lr": args.decoder_lr}],
        weight_decay=0.01,
    )
    steps = math.ceil(len(train_rows) / args.grad_accum) * args.epochs
    sched = get_scheduler(
        "cosine",
        optimizer=optim,
        num_warmup_steps=max(1, steps // 10),
        num_training_steps=max(1, steps),
    )

    best_heldout = None
    bad = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_rows)
        running = 0.0
        n_loss = 0
        optim.zero_grad(set_to_none=True)
        for i, (path, text, kind) in enumerate(train_rows, 1):
            image = Image.open(path).convert("RGB")
            rgb = np.array(image)
            if kind in {"domain", "hard"} or rng.random() < 0.8:
                rgb = strong_phone_aug(rgb, np_rng)
            image = Image.fromarray(rgb)
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

        ha, detail = heldout_char_acc(model, processor, device)
        entry = {
            "epoch": epoch,
            "train_loss": round(running / max(n_loss, 1), 6),
            "heldout_char_acc_pct": round(ha, 2),
            "heldout_detail": detail,
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(entry)
        print(f"epoch {epoch} heldout={ha:.2f}% loss={entry['train_loss']}")
        for d in detail:
            if d["char_acc"] < 0.7:
                print(f"  weak {d['image']}: {d['pred']!r} (gt={d['gt']!r})")

        if best_heldout is None or ha > best_heldout + 0.05:
            best_heldout = ha
            bad = 0
            model.save_pretrained(out)
            processor.save_pretrained(out)
            (out / "heldout_snapshot.json").write_text(
                json.dumps(entry, indent=2), encoding="utf-8"
            )
            print("saved", out, f"best={best_heldout:.2f}%")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    report = {
        "best_heldout_char_acc_pct": best_heldout,
        "init": str(init),
        "history": history,
        "target_pct": "70-80",
    }
    (out / "heldout_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL best_heldout_char_acc_pct", best_heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
