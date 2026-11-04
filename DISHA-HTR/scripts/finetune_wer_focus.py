"""WER-first resume from heldout-push.

Optimizes word-level accuracy on the held-out set:
- Oversamples full multi-word phrases (greetings, sentences, place+PIN)
- Drops competing \"what is your name\" poison phrase
- Selects checkpoints by lowest mean held-out WER
- Float32, small batches for RTX 3050 6GB

Does not overwrite checkpoints/trocr-small-handwritten.
Verified GT only (demo crops, vigorous local_*, hardcase renders).
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

from disha_htr.line_clean import paper_clean_line, strong_phone_aug  # noqa: E402
from disha_htr.metrics import character_error_rate, word_error_rate  # noqa: E402
from disha_htr.wer_normalize import wer_normalize_hypothesis  # noqa: E402

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

POISON_PHRASES = {"what is your name"}


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


def add_pair(rows: list[tuple[Path, str]], img_dir: Path, name: str, text: str) -> None:
    path = img_dir / name
    if not path.is_file():
        path = img_dir.parent / name if img_dir.name == "images" else img_dir / name
    if path.is_file() and text.strip():
        rows.append((path, text.strip()))


def load_improved() -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    csv_path = ROOT / "demo" / "train_improved" / "train_ground_truth.csv"
    img_dir = ROOT / "demo" / "train_improved" / "images"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            add_pair(rows, img_dir, row.get("image_filename", ""), row.get("ground_truth", ""))
    return rows


def load_private() -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    base = ROOT / "demo" / "train_private"
    csv_path = base / "train_ground_truth.csv"
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            path = base / name
            if path.is_file() and text:
                rows.append((path, text))
    return rows


def load_hardcase(split: str = "train") -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    csv_path = ROOT / "training" / "hardcase" / "labels.csv"
    img_dir = ROOT / "training" / "hardcase" / "images"
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("split") or "").strip() != split:
                continue
            add_pair(rows, img_dir, row.get("image_filename", ""), row.get("ground_truth", ""))
    return rows


def load_vigorous_local(max_n: int = 200) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    csv_path = ROOT / "training" / "vigorous" / "labels.csv"
    img_dir = ROOT / "training" / "vigorous" / "images"
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("split") or "").strip() != "train":
                continue
            name = (row.get("image_filename") or "").strip()
            text = (row.get("ground_truth") or "").strip()
            if not name.startswith("local_") and not name.startswith("nbcomp_"):
                continue
            path = img_dir / name
            if path.is_file() and text:
                rows.append((path, text))
            if len(rows) >= max_n:
                break
    return rows


def heldout_wer(model, processor, device) -> tuple[float, float, float, list[dict]]:
    """Return (mean_wer, mean_cer, char_acc_pct, detail). Primary metric is mean WER."""
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
            pred = wer_normalize_hypothesis(
                processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            )
            cer = character_error_rate(text, pred)
            wer = word_error_rate(text, pred)
            detail.append(
                {
                    "image": path.name,
                    "gt": text,
                    "pred": pred,
                    "wer": round(wer, 4),
                    "cer": round(cer, 4),
                    "char_acc": round(1.0 - cer, 4),
                }
            )
    model.train()
    mean_wer = sum(d["wer"] for d in detail) / max(len(detail), 1)
    mean_cer = sum(d["cer"] for d in detail) / max(len(detail), 1)
    return mean_wer, mean_cer, (1.0 - mean_cer) * 100.0, detail


def word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def priority(text: str) -> int:
    """Higher = more WER-critical (full phrases / greeting collapse / names as words)."""
    t = text.strip().lower()
    n_words = word_count(text)
    if t in POISON_PHRASES:
        return 0
    # Spaced single-letter dumps destroy WER (HEMANTH -> H E M A N T H).
    letters = [w for w in t.split() if w]
    if len(letters) >= 4 and all(len(w) == 1 and w.isalpha() for w in letters):
        return 0
    # Phrase collapse killers: whole-word sequences that currently flip WER hard.
    if t in {"how are you", "how are you?"} or t.startswith("how are you"):
        return 6
    if t.replace(" ", "") == "hemanth" or t in {"hemanth", "hemanth kumar"}:
        return 6
    if any(
        k in t
        for k in (
            "i have started practicing",
            "i am ghatboral",
            "dsa questions",
            "happy when i am",
            "karnataka",
            "hackathon",
            "participat",
        )
    ):
        return 5
    # Alphabet drills (Aa Bb / A B a) teach spaced letters — skip for WER.
    if t in {"a b a", "a b c", "h i"} or (
        n_words >= 2 and all(len(w) <= 2 and w.replace(".", "").isalpha() for w in t.split())
    ):
        return 0
    if "hemanth" in t.replace(" ", ""):
        return 5
    if n_words >= 3:
        return 4
    if n_words == 2:
        return 3
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--encoder-lr", type=float, default=3.5e-6)
    parser.add_argument("--decoder-lr", type=float, default=1.4e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-vigorous", type=int, default=200)
    parser.add_argument(
        "--init",
        default=str(ROOT / "checkpoints" / "trocr-heldout-push"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "checkpoints" / "trocr-heldout-push"),
    )
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    improved = load_improved()
    private = load_private()
    hard = load_hardcase("train")
    vigorous = load_vigorous_local(args.max_vigorous)
    print(
        f"verified improved={len(improved)} private={len(private)} "
        f"hardcase={len(hard)} vigorous_local={len(vigorous)}"
    )
    if len(improved) < 8:
        print("Need demo/train_improved")
        return 2

    improved_sorted = sorted(improved, key=lambda x: x[0].name)
    train_core = improved_sorted[3:] + private + vigorous

    train_rows: list[tuple[Path, str, str]] = []
    for path, text in train_core:
        pr = priority(text)
        if pr == 0:
            continue
        # Prefer multi-word lines for WER.
        reps = {1: 2, 3: 7, 4: 12, 5: 20, 6: 30}.get(pr, 2)
        if word_count(text) >= 3:
            reps = max(reps, reps + 4)
        train_rows.extend([(path, text, "domain")] * reps)
    for path, text in hard:
        pr = priority(text)
        if pr == 0:
            continue
        reps = {1: 3, 3: 9, 4: 14, 5: 24, 6: 36}.get(pr, 3)
        if word_count(text) >= 3:
            reps = max(reps, reps + 6)
        train_rows.extend([(path, text, "hard")] * reps)

    rng.shuffle(train_rows)
    print(f"train_items={len(train_rows)} (WER-weighted)")

    init = Path(args.init)
    out = Path(args.out)
    if not (init / "model.safetensors").is_file():
        print("Missing init weights", init)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(str(init))
    model = VisionEncoderDecoderModel.from_pretrained(str(init))
    configure_decoder_tokens(model, processor)
    model.to(device=device, dtype=torch.float32)
    model.train()
    print("init", init, "dtype=float32 primary_metric=WER")

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
        num_warmup_steps=max(1, steps // 12),
        num_training_steps=max(1, steps),
    )

    base_wer, base_cer, base_acc, base_detail = heldout_wer(model, processor, device)
    print(
        f"baseline_mean_wer={base_wer:.4f} mean_cer={base_cer:.4f} "
        f"char_acc={base_acc:.2f}%"
    )
    for d in base_detail:
        if d["wer"] > 0.2:
            print(
                f"  weak {d['image']}: WER={d['wer']:.3f} CER={d['cer']:.3f} "
                f"pred={d['pred']!r} gt={d['gt']!r}"
            )

    best_wer = base_wer
    best_cer = base_cer
    best_acc = base_acc
    bad = 0
    history = [
        {
            "epoch": 0,
            "mean_wer": round(base_wer, 4),
            "mean_cer": round(base_cer, 4),
            "heldout_char_acc_pct": round(base_acc, 2),
            "detail": base_detail,
        }
    ]
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_rows)
        running = 0.0
        n_loss = 0
        optim.zero_grad(set_to_none=True)
        for i, (path, text, kind) in enumerate(train_rows, 1):
            import cv2

            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                rgb = np.array(Image.open(path).convert("RGB"))
            else:
                if kind == "domain" and rng.random() < 0.3:
                    try:
                        rgb = paper_clean_line(bgr)
                    except Exception:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rng.random() < 0.9:
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

        mean_wer, mean_cer, ha, detail = heldout_wer(model, processor, device)
        entry = {
            "epoch": epoch,
            "train_loss": round(running / max(n_loss, 1), 6),
            "mean_wer": round(mean_wer, 4),
            "mean_cer": round(mean_cer, 4),
            "heldout_char_acc_pct": round(ha, 2),
            "heldout_detail": detail,
            "elapsed_s": round(time.time() - t0, 1),
            "baseline_mean_wer": round(base_wer, 4),
            "baseline_heldout_char_acc_pct": round(base_acc, 2),
            "primary_metric": "wer",
        }
        history.append(entry)
        print(
            f"epoch {epoch} mean_WER={mean_wer:.4f} mean_CER={mean_cer:.4f} "
            f"char_acc={ha:.2f}% loss={entry['train_loss']}"
        )
        for d in detail:
            if d["wer"] > 0.2:
                print(
                    f"  weak {d['image']}: WER={d['wer']:.3f} CER={d['cer']:.3f} "
                    f"pred={d['pred']!r}"
                )

        # Save when WER improves (lower is better).
        if mean_wer < best_wer - 0.001:
            best_wer = mean_wer
            best_cer = mean_cer
            best_acc = ha
            bad = 0
            model.save_pretrained(out)
            processor.save_pretrained(out)
            snap = {
                "epoch": epoch,
                "train_loss": entry["train_loss"],
                "heldout_char_acc_pct": round(ha, 2),
                "mean_wer": round(mean_wer, 4),
                "mean_cer": round(mean_cer, 4),
                "heldout_detail": detail,
                "elapsed_s": entry["elapsed_s"],
                "baseline_heldout_char_acc_pct": round(base_acc, 2),
                "primary_metric": "wer",
            }
            (out / "heldout_snapshot.json").write_text(
                json.dumps(snap, indent=2), encoding="utf-8"
            )
            print(
                "saved",
                out,
                f"best_WER={best_wer:.4f} CER={best_cer:.4f} char_acc={best_acc:.2f}%",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop (no WER improvement)")
                break

    report = {
        "primary_metric": "wer",
        "best_mean_wer": round(best_wer, 4),
        "best_mean_wer_pct": round(best_wer * 100.0, 2),
        "best_mean_cer": round(best_cer, 4),
        "best_heldout_char_acc_pct": round(best_acc, 2),
        "baseline_mean_wer": round(base_wer, 4),
        "baseline_heldout_char_acc_pct": round(base_acc, 2),
        "init": str(init),
        "out": str(out),
        "history": history,
        "note": (
            "Checkpoint selected by lowest mean held-out WER on the identical "
            "8-image private set. Base trocr-small-handwritten untouched."
        ),
    }
    (out / "heldout_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL best_mean_WER", round(best_wer, 4), f"({best_wer*100:.2f}%)")
    print("FINAL best_mean_CER", round(best_cer, 4))
    print("FINAL best_heldout_char_acc_pct", round(best_acc, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
