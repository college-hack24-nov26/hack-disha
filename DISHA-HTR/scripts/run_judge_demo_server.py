"""
Standalone testing server for DISHA-HTR handwriting recognition.

Serves the testing UI and local APIs for:
- New image upload + recognition (best verified general checkpoint = base)
- Optional doctor personalization selection / general fallback (if configured)
- Paper original preservation
- Local record listing

Does NOT touch checkpoints. Does NOT claim production DB writes.
Does NOT start large-scale training.

Launch:
  .\.venv\Scripts\python.exe scripts\run_judge_demo_server.py --port 8766 --preload-model
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.multipart_util import FormFile, parse_multipart  # noqa: E402

BASE_CHECKPOINT = ROOT / "checkpoints" / "trocr-small-handwritten"
DOMAIN_BEST_CHECKPOINT = ROOT / "checkpoints" / "trocr-domain-70-best-heldout"
DOMAIN_VAL_CHECKPOINT = ROOT / "checkpoints" / "trocr-domain-70"
DEMO_DATA = ROOT / "judge_demo_data"
STATIC_DIR = ROOT / "judge_demo"
PAD_DIR = ROOT / "prescription_pad"
SAMPLES_DIR = ROOT / "demo" / "approved"
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

_model_lock = threading.Lock()
_loaded = None
_loaded_source = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()


def _heldout_pct_from_snapshot(path: Path) -> float | None:
    snap = path / "heldout_snapshot.json"
    if not snap.is_file():
        return None
    try:
        data = json.loads(snap.read_text(encoding="utf-8"))
        val = data.get("heldout_char_acc_pct")
        return float(val) if val is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def resolve_active_checkpoint() -> tuple[Path, float, str]:
    """Prefer domain fine-tune when it beats the base held-out score."""
    base_pct = 14.01
    domain_pct = _heldout_pct_from_snapshot(DOMAIN_BEST_CHECKPOINT)
    if domain_pct is not None and _has_weights(DOMAIN_BEST_CHECKPOINT) and domain_pct >= base_pct:
        return (
            DOMAIN_BEST_CHECKPOINT,
            domain_pct,
            "Domain fine-tune best-heldout snapshot (measured on identical 8-image set).",
        )
    if _has_weights(DOMAIN_VAL_CHECKPOINT):
        # Fall back to domain-val weights only if snapshot exists elsewhere
        meta = DOMAIN_VAL_CHECKPOINT / "train_meta.json"
        if meta.is_file():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                hp = m.get("best_heldout_char_acc_pct")
                if hp is not None and float(hp) >= base_pct:
                    return (
                        DOMAIN_VAL_CHECKPOINT,
                        float(hp),
                        "Domain fine-tune checkpoint (held-out score from train meta).",
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
    return (
        BASE_CHECKPOINT,
        base_pct,
        "Base trocr-small-handwritten (held-out 14.01%).",
    )


def ensure_dirs() -> None:
    for p in (
        DEMO_DATA,
        DEMO_DATA / "tmp",
        DEMO_DATA / "legacy" / "originals",
        DEMO_DATA / "records",
        DEMO_DATA / "uploads",
        DEMO_DATA / "corrections",
    ):
        p.mkdir(parents=True, exist_ok=True)


def load_model(force_dir: Path | None = None):
    global _loaded, _loaded_source
    with _model_lock:
        target = Path(force_dir) if force_dir else resolve_active_checkpoint()[0]
        if _loaded is not None and _loaded_source == str(target):
            return _loaded
        from disha_htr.hardware import resolve_device
        from disha_htr.model_loader import load_trocr

        _loaded = load_trocr(
            local_dir=target,
            cache_dir=ROOT / ".cache" / "huggingface",
            device=resolve_device("auto"),
        )
        _loaded_source = str(target)
        return _loaded


def json_bytes(payload: dict, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, indent=2).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


class Handler(BaseHTTPRequestHandler):
    server_version = "DISHAHTRTesting/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Avoid logging path contents that might include patient-ish names
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._serve_file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            return self._serve_file(STATIC_DIR / rel)
        if path.startswith("/pad/"):
            rel = path[len("/pad/") :] or "index.html"
            return self._serve_file(PAD_DIR / rel)

        if path == "/api/health":
            ckpt, pct, reason = resolve_active_checkpoint()
            status, body, ctype = json_bytes(
                {
                    "ok": True,
                    "service": "disha-htr-testing",
                    "storage": "Local testing workspace — not the production DISHA database",
                    "active_checkpoint": str(ckpt),
                    "heldout_char_accuracy_pct": pct,
                    "heldout_char_accuracy_pct_base": 14.01,
                    "target_char_accuracy_pct": "70-80% held-out character accuracy",
                    "checkpoint_reason": reason,
                    "model_loaded": _loaded is not None,
                    "ts": utc_now(),
                }
            )
            return self._send(status, body, ctype)

        if path == "/api/checkpoint":
            ckpt, pct, reason = resolve_active_checkpoint()
            domain_pct = _heldout_pct_from_snapshot(DOMAIN_BEST_CHECKPOINT)
            status, body, ctype = json_bytes(
                {
                    "selected": str(ckpt),
                    "active_heldout_char_accuracy_pct": pct,
                    "reason": reason,
                    "comparisons": {
                        "base": 14.01,
                        "domain_best_heldout": domain_pct,
                        "improved_gen": 12.45,
                        "demo_finetuned": 7.62,
                    },
                    "decoder_start_note": (
                        "Pretrained handwritten start token is sep/2; do not force cls/0."
                    ),
                }
            )
            return self._send(status, body, ctype)

        if path == "/api/samples":
            return self._list_samples()

        if path.startswith("/api/samples/") and path.count("/") == 3:
            name = path[len("/api/samples/") :].strip("/")
            return self._sample_file(name)

        if path.startswith("/api/doctor/") and path.endswith("/selection"):
            doctor_id = path[len("/api/doctor/") : -len("/selection")].strip("/")
            return self._doctor_selection(doctor_id)

        if path == "/api/records":
            return self._list_records()

        if path.startswith("/api/records/") and path.endswith("/file"):
            rid = path[len("/api/records/") : -len("/file")]
            kind = parse_qs(parsed.query).get("kind", ["png"])[0]
            return self._record_file(rid, kind)

        if path.startswith("/api/records/") and path.count("/") == 3:
            rid = path[len("/api/records/") :].strip("/")
            return self._get_record(rid)

        self._send(404, b'{"error":"not_found"}', "application/json")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/recognize":
                return self._recognize()
            if path == "/api/legacy/save":
                return self._legacy_save()
            if path == "/api/records/digital":
                return self._save_digital_record()
            if path == "/api/corrections":
                return self._save_correction()
            self._send(404, b'{"error":"not_found"}', "application/json")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            status, body, ctype = json_bytes(
                {"error": "server_error", "detail": str(exc)[:300]}, status=500
            )
            self._send(status, body, ctype)

    def _serve_file(self, file_path: Path) -> None:
        file_path = file_path.resolve()
        # Confine to judge_demo or prescription_pad
        allowed_roots = (STATIC_DIR.resolve(), PAD_DIR.resolve())
        ok = False
        for root in allowed_roots:
            try:
                file_path.relative_to(root)
                ok = True
                break
            except ValueError:
                continue
        if not ok or not file_path.is_file():
            self._send(404, b"Not found", "text/plain")
            return
        data = file_path.read_bytes()
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self._send(200, data, ctype)

    def _read_multipart(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("file too large (max 12MB)")
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")
        return parse_multipart(ctype, body)

    def _recognize(self) -> None:
        form = self._read_multipart()
        item = form.get("image")
        if not isinstance(item, FormFile):
            status, body, ctype = json_bytes({"error": "missing_image"}, 400)
            return self._send(status, body, ctype)
        filename = Path(item.filename or "upload.png").name
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            status, body, ctype = json_bytes(
                {"error": "unsupported_type", "allowed": sorted(ALLOWED_EXT)}, 400
            )
            return self._send(status, body, ctype)
        raw = item.data
        if not raw:
            status, body, ctype = json_bytes({"error": "empty_file"}, 400)
            return self._send(status, body, ctype)

        doctor_id = str(form.get("doctor_id") or "").strip()
        force_single = str(form.get("force_single_line") or "").lower() in {
            "1",
            "true",
            "yes",
        }

        # Temporary upload — not training data
        tmp_dir = DEMO_DATA / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"upload_{uuid.uuid4().hex}{ext}"
        tmp_path.write_bytes(raw)

        checkpoint, heldout_pct, ck_reason = resolve_active_checkpoint()
        selection = {
            "source": "active_general",
            "reason": ck_reason,
            "heldout_char_accuracy_pct": heldout_pct,
            "adapter_version": None,
        }
        if doctor_id:
            selection = self._selection_dict(doctor_id)
            # Only use personalized weights if selector says personalized and dir has weights
            if selection.get("source") == "personalized":
                cand = Path(selection["checkpoint_dir"])
                if (cand / "model.safetensors").is_file() or (
                    cand / "pytorch_model.bin"
                ).is_file():
                    checkpoint = cand
                else:
                    selection["source"] = "general_fallback"
                    selection["reason"] = "personalized_weights_missing"
                    checkpoint, heldout_pct, ck_reason = resolve_active_checkpoint()
                    selection["heldout_char_accuracy_pct"] = heldout_pct

        try:
            loaded = load_model(checkpoint)
            from disha_htr.page_recognize import recognize_page_or_line

            result = recognize_page_or_line(
                tmp_path,
                loaded,
                force_single_line=force_single,
                max_lines=12,
            )
            payload = {
                "ok": True,
                "filename": filename,
                "checkpoint": str(checkpoint),
                "selection": selection,
                "recognition": result.to_dict(),
                "text": result.combined_text,
                "inference_time_s": result.total_inference_time_s,
                "storage_note": (
                    "Upload kept temporary under local testing storage unless explicitly saved. "
                    "Not added to training data."
                ),
            }
            status, body, ctype = json_bytes(payload)
            self._send(status, body, ctype)
        finally:
            # Keep temp briefly for debugging? User asked temporary unless saved — delete
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _selection_dict(self, doctor_id: str) -> dict:
        from disha_htr.personalization.doctor_id import normalize_doctor_id
        from disha_htr.personalization.selector import select_checkpoint_for_doctor
        from disha_htr.personalization.storage import PersonalizationStore

        store = PersonalizationStore(ROOT / "personalization")
        try:
            did = normalize_doctor_id(doctor_id)
        except ValueError as exc:
            return {
                "source": "general",
                "reason": f"invalid_doctor_id: {exc}",
                "doctor_id": doctor_id,
                "checkpoint_dir": str(resolve_active_checkpoint()[0]),
                "adapter_version": None,
            }
        try:
            decision = select_checkpoint_for_doctor(
                store, did, project_root=ROOT
            )
            return decision.to_dict()
        except Exception as exc:  # noqa: BLE001
            return {
                "source": "general_fallback",
                "reason": f"selection_error: {exc}"[:200],
                "doctor_id": did,
                "checkpoint_dir": str(resolve_active_checkpoint()[0]),
                "adapter_version": None,
            }

    def _doctor_selection(self, doctor_id: str) -> None:
        status, body, ctype = json_bytes(self._selection_dict(doctor_id))
        self._send(status, body, ctype)

    def _legacy_save(self) -> None:
        form = self._read_multipart()
        item = form.get("image")
        if not isinstance(item, FormFile):
            status, body, ctype = json_bytes({"error": "missing_image"}, 400)
            return self._send(status, body, ctype)
        filename = Path(item.filename or "legacy.png").name
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            status, body, ctype = json_bytes({"error": "unsupported_type"}, 400)
            return self._send(status, body, ctype)
        raw = item.data
        patient_id = str(form.get("patient_id") or "TEST-P-LEGACY").strip()
        doctor_id = str(form.get("doctor_id") or "clinician_test").strip()
        patient_name = str(form.get("patient_name") or "").strip()[:120]
        doctor_name = str(form.get("doctor_name") or "").strip()[:120]
        note = str(form.get("note") or "").strip()

        rid = f"legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        originals = DEMO_DATA / "legacy" / "originals"
        originals.mkdir(parents=True, exist_ok=True)
        # Never overwrite: unique filename
        dest_name = f"{rid}__{filename}"
        dest = originals / dest_name
        dest.write_bytes(raw)

        record = {
            "id": rid,
            "type": "legacy_paper",
            "createdAt": utc_now(),
            "doctorId": doctor_id,
            "doctorName": doctor_name,
            "patientId": patient_id,
            "patientName": patient_name,
            "originalFilename": filename,
            "storedAs": dest_name,
            "originalPath": str(dest.relative_to(ROOT)),
            "bytes": len(raw),
            "note": note[:200],
            "ocrRequired": False,
            "authoritativeVisualSource": "original_uploaded_image",
            "storageNote": "Local testing only — original image preserved unchanged.",
        }
        (DEMO_DATA / "records" / f"{rid}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        status, body, ctype = json_bytes({"ok": True, "record": record})
        self._send(status, body, ctype)

    def _save_digital_record(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            status, body, ctype = json_bytes({"error": "too_large"}, 400)
            return self._send(status, body, ctype)
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        rid = f"digital_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = DEMO_DATA / "records" / rid
        out_dir.mkdir(parents=True, exist_ok=True)

        stroke_doc = data.get("strokeDocument") or {}
        png_b64 = data.get("pngDataUrl") or ""
        clinical = data.get("clinicalDraft") or {}
        meta = {
            "id": rid,
            "type": "digital_handwriting_prescription",
            "createdAt": utc_now(),
            "doctorId": data.get("doctorId"),
            "doctorName": data.get("doctorName"),
            "patientName": data.get("patientName"),
            "patientId": data.get("patientId"),
            "patientAge": data.get("patientAge"),
            "visitDate": data.get("visitDate"),
            "clinicalDraft": {
                "diagnosis": clinical.get("diagnosis") or "",
                "medicines": clinical.get("medicines") or "",
                "dosage": clinical.get("dosage") or "",
                "frequency": clinical.get("frequency") or "",
                "duration": clinical.get("duration") or "",
                "instructions": clinical.get("instructions") or "",
                "transcription": clinical.get("transcription")
                or data.get("transcriptionDraft")
                or "",
                "note": "Draft layer only — does not alter original strokes or PNG.",
            },
            "transcriptionDraft": (clinical.get("transcription") or data.get("transcriptionDraft") or ""),
            "approvalStatus": data.get("approvalStatus") or "draft_unapproved",
            "versionOf": data.get("versionOf") or None,
            "storageNote": "Local testing only — not the production DISHA database.",
            "files": {
                "strokes": "strokes.json",
                "png": "handwriting.png" if png_b64.startswith("data:image") else None,
            },
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out_dir / "strokes.json").write_text(
            json.dumps(stroke_doc, indent=2), encoding="utf-8"
        )
        if png_b64.startswith("data:image"):
            import base64

            header, b64 = png_b64.split(",", 1)
            (out_dir / "handwriting.png").write_bytes(base64.b64decode(b64))
        # Index pointer
        (DEMO_DATA / "records" / f"{rid}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        status, body, ctype = json_bytes({"ok": True, "record": meta})
        self._send(status, body, ctype)

    def _list_records(self) -> None:
        rows = []
        for p in sorted((DEMO_DATA / "records").glob("*.json"), reverse=True):
            try:
                rows.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
        status, body, ctype = json_bytes(
            {
                "storageNote": "Local testing only",
                "count": len(rows),
                "records": rows[:50],
            }
        )
        self._send(status, body, ctype)

    def _get_record(self, rid: str) -> None:
        if ".." in rid or "/" in rid or "\\" in rid:
            status, body, ctype = json_bytes({"error": "bad_id"}, 400)
            return self._send(status, body, ctype)
        meta_path = DEMO_DATA / "records" / f"{rid}.json"
        if not meta_path.is_file():
            status, body, ctype = json_bytes({"error": "not_found"}, 404)
            return self._send(status, body, ctype)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stroke_path = DEMO_DATA / "records" / rid / "strokes.json"
        if stroke_path.is_file():
            meta = dict(meta)
            meta["strokeDocument"] = json.loads(stroke_path.read_text(encoding="utf-8"))
        status, body, ctype = json_bytes({"ok": True, "record": meta})
        self._send(status, body, ctype)

    def _record_file(self, rid: str, kind: str) -> None:
        # Confine
        if ".." in rid or "/" in rid or "\\" in rid:
            self._send(400, b"bad id", "text/plain")
            return
        if kind == "png":
            path = DEMO_DATA / "records" / rid / "handwriting.png"
            if not path.is_file():
                # legacy originals referenced by storedAs in meta
                meta_path = DEMO_DATA / "records" / f"{rid}.json"
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    stored = meta.get("storedAs")
                    if stored:
                        path = DEMO_DATA / "legacy" / "originals" / stored
            if path.is_file():
                return self._send(200, path.read_bytes(), "image/png")
        if kind == "strokes":
            path = DEMO_DATA / "records" / rid / "strokes.json"
            if path.is_file():
                return self._send(
                    200, path.read_bytes(), "application/json; charset=utf-8"
                )
        self._send(404, b"missing", "text/plain")

    def _list_samples(self) -> None:
        items = []
        if SAMPLES_DIR.is_dir():
            for p in sorted(SAMPLES_DIR.glob("*.png"))[:24]:
                items.append(
                    {
                        "id": p.name,
                        "label": p.stem.replace("_", " "),
                        "url": "/api/samples/" + p.name,
                    }
                )
        status, body, ctype = json_bytes({"count": len(items), "samples": items})
        self._send(status, body, ctype)

    def _sample_file(self, name: str) -> None:
        if ".." in name or "/" in name or "\\" in name:
            self._send(400, b"bad name", "text/plain")
            return
        path = (SAMPLES_DIR / name).resolve()
        try:
            path.relative_to(SAMPLES_DIR.resolve())
        except ValueError:
            self._send(404, b"missing", "text/plain")
            return
        if not path.is_file():
            self._send(404, b"missing", "text/plain")
            return
        self._send(200, path.read_bytes(), "image/png")

    def _save_correction(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            status, body, ctype = json_bytes({"error": "too_large"}, 400)
            return self._send(status, body, ctype)
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        draft = str(data.get("correctedText") or data.get("text") or "").strip()
        if not draft:
            status, body, ctype = json_bytes({"error": "empty_correction"}, 400)
            return self._send(status, body, ctype)
        rid = f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        record = {
            "id": rid,
            "type": "transcription_correction",
            "createdAt": utc_now(),
            "originalText": str(data.get("originalText") or "")[:2000],
            "correctedText": draft[:4000],
            "filename": str(data.get("filename") or "")[:200],
            "checkpoint": str(data.get("checkpoint") or ""),
            "note": "Clinician correction for testing/improvement — local only.",
            "storageNote": "Local testing only — not the production DISHA database.",
        }
        out = DEMO_DATA / "corrections" / f"{rid}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        (DEMO_DATA / "records" / f"{rid}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        status, body, ctype = json_bytes({"ok": True, "record": record})
        self._send(status, body, ctype)


def main() -> int:
    parser = argparse.ArgumentParser(description="DISHA HTR testing server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--preload-model",
        action="store_true",
        help="Load TrOCR at startup (slower start, faster first recognize)",
    )
    args = parser.parse_args()

    ensure_dirs()
    if not STATIC_DIR.is_dir():
        print(f"Missing testing UI at {STATIC_DIR}", file=sys.stderr)
        return 2
    if not BASE_CHECKPOINT.is_dir():
        print(f"Missing checkpoint: {BASE_CHECKPOINT}", file=sys.stderr)
        return 2

    active, pct, reason = resolve_active_checkpoint()
    if args.preload_model:
        print(f"Preloading active checkpoint ({pct}% held-out)...")
        print(reason)
        load_model(active)
        print("Model ready.")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DISHA HTR testing: http://127.0.0.1:{args.port}/")
    print(f"Phone (same Wi-Fi): http://<laptop-LAN-IP>:{args.port}/")
    print(f"Active checkpoint: {active}")
    print(f"Held-out accuracy: {pct}%")
    print("Storage: local testing only under judge_demo_data/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
