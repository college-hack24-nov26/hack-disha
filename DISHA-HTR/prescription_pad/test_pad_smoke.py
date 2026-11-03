"""Smoke checks for the prescription pad testing UI (no browser automation required)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class TestPrescriptionPadSmoke(unittest.TestCase):
    def test_files_exist(self) -> None:
        for name in ("index.html", "styles.css", "app.js", "README.md"):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_html_has_sections_and_canvas(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="rxCanvas"', html)
        self.assertIn("HEADER", html.upper() or "rx-header")
        self.assertIn("rx-header", html)
        self.assertIn("patientName", html)
        self.assertIn("btnUndo", html)
        self.assertIn("btnClear", html)
        self.assertIn("btnSave", html)
        self.assertIn("btnRecognizePad", html)
        self.assertIn("Recognize handwriting", html)
        self.assertIn("draftDiagnosis", html)
        self.assertIn("draftTranscription", html)
        self.assertIn("does not alter handwriting", html)
        self.assertIn("testing", html.lower())
        self.assertIn("patient-facing DISHA", html)
        self.assertNotIn("trocr", html.lower())

    def test_js_pointer_events_and_exports(self) -> None:
        js = (ROOT / "app.js").read_text(encoding="utf-8")
        for token in (
            "pointerdown",
            "pointermove",
            "pointerup",
            "setPointerCapture",
            "drawing-lock",
            "toBlob",
            "application/json",
            "localStorage",
            "does not provide real stylus pressure",
            "digital_handwriting_prescription",
            "legacyPaper",
            "strokes",
            "reopenRecord",
            "restoreStrokesFromDocument",
            "/api/records/digital",
            "clinicalDraft",
            "draft_unapproved",
        ):
            self.assertIn(token, js)

        # Mouse must not blindly store pressure=evt.pressure
        self.assertIn('pointerType === "pen"', js)
        self.assertIn("Mouse: intentionally omit pressure", js)

    def test_css_responsive_and_touch_action(self) -> None:
        css = (ROOT / "styles.css").read_text(encoding="utf-8")
        self.assertIn("touch-action: none", css)
        self.assertIn("@media", css)
        self.assertIn("drawing-lock", css)

    def test_no_trocr_pipeline_imports(self) -> None:
        product = ("index.html", "styles.css", "app.js", "README.md")
        for name in product:
            text = (ROOT / name).read_text(encoding="utf-8", errors="ignore").lower()
            self.assertNotIn("checkpoints/trocr", text, msg=name)
            self.assertNotIn("model_loader", text, msg=name)
            self.assertNotIn("from disha_htr", text, msg=name)


if __name__ == "__main__":
    unittest.main()
