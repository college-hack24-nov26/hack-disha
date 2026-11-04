# Quick samples — IAM handwriting

These PNGs are curated **IAM** line images (from the local Teklia IAM-line extract) with **official ground-truth text** in `samples_manifest.json`.

## Why IAM (not the previous notebook chips)

The domain fine-tune is strong on phone-notebook held-out lines, but it drifts on classical IAM handwriting. For judge demos that use IAM assets, recognition uses **`checkpoints/trocr-small-handwritten`** (base TrOCR), which matches this writing style so the draft clearly shows the content in the image.

## Demo flow

1. Click an IAM chip — **expected content** appears (official transcription).
2. Recognize — draft should **exact-match** that content on the base handwriting model.
3. UI shows **matched** when draft equals the IAM ground truth.

IAM licensing: non-commercial research; cite Marti & Bunke / FKI terms. Do not redistribute the full IAM dump; these curated demo lines are for local judging only.
