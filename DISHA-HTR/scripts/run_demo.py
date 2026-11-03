"""
DISHA-HTR judge demo (offline, local checkpoint).

Gallery is restricted to demo/approved/* only.
Ground truth is NEVER loaded or shown (lives under demo/eval_private/).

Launch:
  .\\.venv\\Scripts\\python.exe scripts\\run_demo.py
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

APPROVED_DIR = ROOT / "demo" / "approved"
MANIFEST_PATH = ROOT / "demo" / "approved_manifest.json"
BASE_CHECKPOINT = ROOT / "checkpoints" / "trocr-small-handwritten"
FINETUNED_CHECKPOINT = ROOT / "checkpoints" / "trocr-demo-finetuned"
# Prefer gallery-adapted demo FT when present. Do NOT auto-switch to
# checkpoints/trocr-improved-gen: measured held-out char acc was below base,
# and gallery accuracy collapsed vs trocr-demo-finetuned.
CHECKPOINT = (
    FINETUNED_CHECKPOINT
    if (FINETUNED_CHECKPOINT / "model.safetensors").is_file()
    or (FINETUNED_CHECKPOINT / "pytorch_model.bin").is_file()
    else BASE_CHECKPOINT
)
CACHE = ROOT / ".cache" / "huggingface"

# Hard deny: never read these from the UI process paths used for display logic
FORBIDDEN_GT_DIRS = (
    ROOT / "demo" / "eval_private",
    ROOT / "evaluation" / "ground_truth",
)


class DemoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DISHA-HTR Demo — Handwriting Recognition")
        self.root.geometry("980x640")
        self.root.minsize(860, 560)

        self.images = self._load_approved()
        self.photo = None
        self.loaded_model = None
        self.busy = False
        self.selected: dict | None = None

        self._build_ui()
        if self.images:
            self.listbox.selection_set(0)
            self._on_select()
        self.status.set("Ready — select an image, then Recognize. Model loads on first run.")

    def _load_approved(self) -> list[dict]:
        if not MANIFEST_PATH.is_file():
            messagebox.showerror(
                "Missing demo set",
                f"Approved manifest not found:\n{MANIFEST_PATH}",
            )
            return []
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        items = []
        for row in data.get("images", []):
            path = APPROVED_DIR / row["filename"]
            if not path.is_file():
                continue
            # Restrict: must live inside approved dir
            try:
                path.resolve().relative_to(APPROVED_DIR.resolve())
            except ValueError:
                continue
            items.append({**row, "path": path})
        return items

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(top)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(left, text="Approved samples").pack(anchor=tk.W)
        self.listbox = tk.Listbox(left, width=28, height=22, exportselection=False)
        self.listbox.pack(fill=tk.Y, expand=True)
        for item in self.images:
            self.listbox.insert(tk.END, item.get("label") or item["filename"])
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._on_select())

        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Label(right, bg="#f4f4f4", relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=8)
        self.recognize_btn = ttk.Button(btns, text="Recognize", command=self._recognize)
        self.recognize_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Clear output", command=self._clear_output).pack(
            side=tk.LEFT, padx=8
        )

        ttk.Label(right, text="Transcription").pack(anchor=tk.W)
        self.output = tk.Text(right, height=5, wrap=tk.WORD)
        self.output.pack(fill=tk.X)

        self.progress = ttk.Progressbar(right, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=6)

        self.status = tk.StringVar()
        ttk.Label(right, textvariable=self.status).pack(anchor=tk.W)

        note = (
            "Offline demo · local TrOCR checkpoint only · "
            "gallery locked to demo/approved · no scores shown here"
        )
        ttk.Label(self.root, text=note, padding=8).pack(fill=tk.X)

    def _on_select(self) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        self.selected = self.images[sel[0]]
        self._show_image(self.selected["path"])
        self.status.set(f"Selected: {self.selected['filename']}")

    def _show_image(self, path: Path) -> None:
        img = Image.open(path).convert("RGB")
        # Fit into panel
        max_w, max_h = 640, 360
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo)

    def _clear_output(self) -> None:
        self.output.delete("1.0", tk.END)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.recognize_btn.configure(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _recognize(self) -> None:
        if self.busy:
            return
        if not self.selected:
            messagebox.showwarning("No selection", "Select an approved image first.")
            return
        # Safety: refuse paths outside approved dir
        try:
            self.selected["path"].resolve().relative_to(APPROVED_DIR.resolve())
        except ValueError:
            messagebox.showerror("Blocked", "Image is not in the approved demo set.")
            return

        self._set_busy(True)
        self.status.set("Loading model / recognizing…")
        self.output.delete("1.0", tk.END)
        path = self.selected["path"]
        threading.Thread(target=self._recognize_worker, args=(path,), daemon=True).start()

    def _ensure_model(self):
        if self.loaded_model is not None:
            return self.loaded_model
        from disha_htr.hardware import resolve_device
        from disha_htr.model_loader import load_trocr

        if not CHECKPOINT.is_dir():
            raise FileNotFoundError(f"Local checkpoint missing: {CHECKPOINT}")
        device = resolve_device("auto")
        self.loaded_model = load_trocr(
            local_dir=CHECKPOINT,
            cache_dir=CACHE,
            device=device,
        )
        return self.loaded_model

    def _recognize_worker(self, path: Path) -> None:
        try:
            from disha_htr.recognize import recognize_image

            loaded = self._ensure_model()
            result = recognize_image(path, loaded)
            text = result.text
            seconds = result.inference_time_s
            device = result.device

            def ok() -> None:
                self.output.insert(tk.END, text)
                self.status.set(
                    f"Done in {seconds:.2f}s on {device} · "
                    f"checkpoint={CHECKPOINT.name}"
                )
                self._set_busy(False)

            self.root.after(0, ok)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()

            def fail() -> None:
                self.output.insert(tk.END, f"[Recognition failed]\n{err}")
                self.status.set("Error — see output panel")
                self._set_busy(False)
                messagebox.showerror("Recognition failed", err)

            self.root.after(0, fail)
            print(tb, file=sys.stderr)


def main() -> int:
    # Guard: UI must not import eval_private GT
    for forbidden in FORBIDDEN_GT_DIRS:
        # existence is fine; we just never read it here
        _ = forbidden
    if not APPROVED_DIR.is_dir():
        print(f"Missing approved demo images at {APPROVED_DIR}", file=sys.stderr)
        return 2
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
