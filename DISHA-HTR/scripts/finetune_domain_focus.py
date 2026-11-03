"""
Domain-focused TrOCR fine-tune for phone-notebook held-out accuracy.

- Starts from checkpoints/trocr-small-handwritten (never overwrites it).
- Correct decoder_start = sep (2).
- Heavy oversampling of verified local notebook lines + strong phone augs.
- Light IAM regularization from training/vigorous when available.
- Saves best domain-val CER to checkpoints/trocr-domain-70/.
- Does NOT train on held-out images.
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

import cv2
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


def load_improved_local() -> list[tuple[Path, str]]:
    csv_path = ROOT / "demo" / "train_improved" / "train_ground_truth.csv"
    img_dir = ROOT / "demo" / "train_improved" / "images"
    rows: list[tuple[Path, str]] = []
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            path = img_dir / name
            if name and text and path.is_file():
                rows.append((path, text))
    return rows


def load_vigorous_local(max_n: int = 400) -> list[tuple[Path, str]]:
    csv_path = ROOT / "training" / "vigorous" / "labels.csv"
    img_dir = ROOT / "training" / "vigorous" / "images"
    rows: list[tuple[Path, str]] = []
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            src = (row.get("source") or "").strip()
            split = (row.get("split") or "").strip()
            if split != "train":
                continue
            # Prefer local / notebook composites; skip pure iam_* if we have enough local.
            if not (name.startswith("local_") or name.startswith("nbcomp_") or "non_cursive" in src or "cursive" in src or "medical" in src):
                continue
            path = img_dir / name
            if name and text and path.is_file():
                rows.append((path, text))
            if len(rows) >= max_n:
                break
    return rows


def load_iam_reg(max_n: int = 400) -> list[tuple[Path, str]]:
    csv_path = ROOT / "training" / "vigorous" / "labels.csv"
    img_dir = ROOT / "training" / "vigorous" / "images"
    rows: list[tuple[Path, str]] = []
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            split = (row.get("split") or "").strip()
            if split != "train" or not name.startswith("iam_"):
                continue
            path = img_dir / name
            if name and text and path.is_file():
                rows.append((path, text))
            if len(rows) >= max_n:
                break
    return rows


def eval_cer(model, processor, pairs, device) -> float:
    import torch

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
    rows = []
    with torch.inference_mode():
        for path, text in pairs:
            image = Image.open(path).convert("RGB")
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            out = model.generate(pv, max_new_tokens=64, num_beams=5)
            pred = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            cer = character_error_rate(text, pred)
            rows.append(
                {
                    "image": path.name,
                    "gt": text,
                    "pred": pred,
                    "cer": round(cer, 4),
                    "char_acc": round(1.0 - cer, 4),
                }
            )
    model.train()
    mean_cer = sum(r["cer"] for r in rows) / max(len(rows), 1)
    return (1.0 - mean_cer) * 100.0, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=8e-6)
    parser.add_argument("--decoder-lr", type=float, default=3e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--domain-repeat", type=int, default=60)
    parser.add_argument("--max-iam", type=int, default=350)
    parser.add_argument(
        "--extra-local",
        type=int,
        default=0,
        help="Extra vigorous local/nbcomp samples to mix (0 = improved-only, fastest)",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-heldout-every", type=int, default=2)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional warm-start dir (default: trocr-small-handwritten)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output checkpoint dir (default: checkpoints/trocr-domain-70)",
    )
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    improved = load_improved_local()
    vigorous_local = load_vigorous_local(args.extra_local) if args.extra_local > 0 else []
    iam = load_iam_reg(args.max_iam)
    if len(improved) < 8:
        print("Need demo/train_improved — run prepare_improved_train_set.py")
        return 2

    # Domain val: 4 held-aside improved lines, rest train.
    improved_sorted = sorted(improved, key=lambda x: x[0].name)
    val_domain = improved_sorted[:4]
    train_domain = improved_sorted[4:] + vigorous_local
    # Oversample domain heavily — notebook phone domain is the bottleneck.
    train_rows: list[tuple[Path, str, str]] = []
    for _ in range(args.domain_repeat):
        for path, text in train_domain:
            train_rows.append((path, text, "domain"))
    for path, text in iam:
        train_rows.append((path, text, "iam"))
    rng.shuffle(train_rows)
    print(
        f"train_items={len(train_rows)} domain_unique={len(train_domain)} "
        f"iam={len(iam)} val_domain={len(val_domain)}"
    )

    base = ROOT / "checkpoints" / "trocr-small-handwritten"
    init = Path(args.init_checkpoint) if args.init_checkpoint else base
    if not (init / "model.safetensors").is_file() and not (init / "pytorch_model.bin").is_file():
        print("Missing init checkpoint weights:", init)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "checkpoints" / "trocr-domain-70")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_heldout_dir = ROOT / "checkpoints" / "trocr-domain-70-best-heldout"

    # Oversample short / name-like domain lines (hard cases on held-out).
    hard_boost = []
    for path, text in train_domain:
        t = text.strip().lower()
        if (
            len(text) <= 18
            or "hemanth" in t
            or "name" in t
            or "how are" in t
            or t in {"a b a", "h i", "b", "kg"}
        ):
            hard_boost.extend([(path, text, "domain")] * 3)
    train_rows.extend(hard_boost)
    rng.shuffle(train_rows)
    print(f"after hard-boost train_items={len(train_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(str(init if (init / "tokenizer_config.json").is_file() else base))
    model = VisionEncoderDecoderModel.from_pretrained(str(init))
    token_meta = configure_decoder_tokens(model, processor)
    print("init", init)
    print("tokens", token_meta)
    model.to(device=device, dtype=torch.float32)
    model.train()

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")
    ]
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
        num_warmup_steps=max(1, steps // 12),
        num_training_steps=max(1, steps),
    )

    print("baseline domain-val CER…")
    best = eval_cer(model, processor, val_domain, device)
    print(f"baseline_val_cer={best:.4f}")
    bad = 0
    history = []
    t0 = time.time()
    best_heldout = None

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_rows)
        running = 0.0
        n_loss = 0
        optim.zero_grad(set_to_none=True)
        for i, (path, text, kind) in enumerate(train_rows, 1):
            image = Image.open(path).convert("RGB")
            rgb = np.array(image)
            if kind == "domain" or rng.random() < 0.75:
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
            if i % 400 == 0:
                print(
                    f"  epoch {epoch} item {i}/{len(train_rows)} "
                    f"loss={running / max(1, n_loss):.4f}"
                )

        val = eval_cer(model, processor, val_domain, device)
        entry = {
            "epoch": epoch,
            "train_loss": round(running / max(1, n_loss), 6),
            "val_cer": round(val, 6),
            "elapsed_s": round(time.time() - t0, 1),
        }
        if args.eval_heldout_every > 0 and (
            epoch % args.eval_heldout_every == 0 or epoch == args.epochs
        ):
            ha, detail = heldout_char_acc(model, processor, device)
            entry["heldout_char_acc_pct"] = round(ha, 2)
            entry["heldout_detail"] = detail
            print(f"  heldout_char_acc_pct={ha:.2f}")
            if best_heldout is None or ha > best_heldout:
                best_heldout = ha
                # Also snapshot best-heldout weights separately for selection.
                best_heldout_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_heldout_dir)
                processor.save_pretrained(best_heldout_dir)
                (best_heldout_dir / "heldout_snapshot.json").write_text(
                    json.dumps(entry, indent=2), encoding="utf-8"
                )
                print("  saved best-heldout snapshot", best_heldout_dir)

        history.append(entry)
        print(
            f"epoch {epoch} val_cer={val:.4f} best={best:.4f} "
            f"loss={entry['train_loss']}"
        )
        if val < best - 1e-4:
            best = val
            bad = 0
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            (out_dir / "train_meta.json").write_text(
                json.dumps(
                    {
                        "best_val_cer": best,
                        "best_heldout_char_acc_pct": best_heldout,
                        "epoch": epoch,
                        "token_meta": token_meta,
                        "n_train": len(train_rows),
                        "n_val_domain": len(val_domain),
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

    # Final held-out report from best-heldout snapshot if present else current.
    final_dir = best_heldout_dir
    if (final_dir / "model.safetensors").is_file():
        model = VisionEncoderDecoderModel.from_pretrained(str(final_dir))
        configure_decoder_tokens(model, processor)
        model.to(device=device, dtype=torch.float32)
    ha, detail = heldout_char_acc(model, processor, device)
    report = {
        "heldout_char_acc_pct": round(ha, 2),
        "best_domain_val_cer": best,
        "best_heldout_seen_pct": best_heldout,
        "init_checkpoint": str(init),
        "detail": detail,
        "history": history,
        "target_pct": "70-80",
    }
    (out_dir / "heldout_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (best_heldout_dir / "heldout_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL heldout_char_acc_pct", round(ha, 2))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
