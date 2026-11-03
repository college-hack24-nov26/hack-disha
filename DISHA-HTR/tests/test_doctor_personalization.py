"""CPU tests for doctor personalization isolation, selection, versioning."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.personalization.auth import DoctorAuthContext  # noqa: E402
from disha_htr.personalization.corrections import CorrectionWorkflow  # noqa: E402
from disha_htr.personalization.doctor_id import normalize_doctor_id  # noqa: E402
from disha_htr.personalization.registry import ModelRegistry  # noqa: E402
from disha_htr.personalization.samples import SampleRegistry  # noqa: E402
from disha_htr.personalization.selector import select_checkpoint_for_doctor  # noqa: E402
from disha_htr.personalization.storage import (  # noqa: E402
    PersonalizationStore,
    create_doctor_profile,
)
from disha_htr.personalization.trainer import PersonalizationTrainer  # noqa: E402


def _png(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> Path:
    img = Image.new("RGB", (64, 32), color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


class TestDoctorId(unittest.TestCase):
    def test_normalize_ok(self) -> None:
        self.assertEqual(normalize_doctor_id("Doc_A1"), "doc_a1")

    def test_reject_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            normalize_doctor_id("../etc")
        with self.assertRaises(ValueError):
            normalize_doctor_id("a/b")


class TestIsolationAndSamples(unittest.TestCase):
    def test_doctor_a_samples_cannot_enter_doctor_b_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonalizationStore(Path(tmp) / "personalization")
            create_doctor_profile(store, "doctor_a", display_name="A")
            create_doctor_profile(store, "doctor_b", display_name="B")
            auth_a = DoctorAuthContext(doctor_id="doctor_a")
            auth_b = DoctorAuthContext(doctor_id="doctor_b")
            reg = SampleRegistry(store)

            src_a = _png(Path(tmp) / "a.png", (10, 10, 10))
            src_b = _png(Path(tmp) / "b.png", (200, 200, 200))
            reg.add_verified_sample(
                auth_a,
                "doctor_a",
                src_a,
                ground_truth="aspirin 75 mg",
                split="train",
                verified_by="clinician_a",
            )
            reg.add_verified_sample(
                auth_b,
                "doctor_b",
                src_b,
                ground_truth="metformin 500 mg",
                split="train",
                verified_by="clinician_b",
            )

            # Auth mismatch: A cannot write into B
            with self.assertRaises(PermissionError):
                reg.add_verified_sample(
                    auth_a,
                    "doctor_b",
                    src_a,
                    ground_truth="leak",
                    split="train",
                    verified_by="evil",
                )

            # Content isolation
            stats = reg.assert_no_cross_doctor_leak("doctor_a", "doctor_b")
            self.assertEqual(stats["shared_content_hashes"], 0)

            # Paths confined
            a_file = store.samples_dir("doctor_a", "train") / "a.png"
            with self.assertRaises(PermissionError):
                store.assert_path_owned_by("doctor_b", a_file)

    def test_rejects_model_prediction_as_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonalizationStore(Path(tmp) / "personalization")
            create_doctor_profile(store, "doctor_a")
            auth = DoctorAuthContext(doctor_id="doctor_a")
            reg = SampleRegistry(store)
            src = _png(Path(tmp) / "x.png")
            with self.assertRaises(ValueError):
                reg.add_verified_sample(
                    auth,
                    "doctor_a",
                    src,
                    ground_truth="hello",
                    split="train",
                    verified_by="bot",
                    source="model_prediction",
                )


class TestSelectionFallbackVersioning(unittest.TestCase):
    def test_general_fallback_without_personalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Fake general checkpoint dir marker
            general = root / "checkpoints" / "trocr-small-handwritten"
            general.mkdir(parents=True)
            store = PersonalizationStore(root / "personalization")
            create_doctor_profile(store, "doctor_a")
            decision = select_checkpoint_for_doctor(
                store, "doctor_a", project_root=root
            )
            self.assertEqual(decision.source, "general")
            self.assertIn("trocr-small-handwritten", decision.checkpoint_dir)

    def test_placeholder_falls_back_to_general(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints" / "trocr-small-handwritten").mkdir(parents=True)
            store = PersonalizationStore(root / "personalization")
            create_doctor_profile(store, "doctor_a")
            auth = DoctorAuthContext(doctor_id="doctor_a")
            trainer = PersonalizationTrainer(store, project_root=root)
            out = trainer.prepare_candidate_placeholder(auth, "doctor_a")
            version = out["meta"]["version"]
            reg = ModelRegistry(store)
            # Force-activate placeholder for selection test
            reg.activate(auth, "doctor_a", version, force=True)
            decision = select_checkpoint_for_doctor(
                store, "doctor_a", project_root=root
            )
            self.assertEqual(decision.source, "general_fallback")
            self.assertEqual(decision.reason, "placeholder_without_weights")

    def test_worse_personalization_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints" / "trocr-small-handwritten").mkdir(parents=True)
            store = PersonalizationStore(root / "personalization")
            create_doctor_profile(store, "doctor_a")
            auth = DoctorAuthContext(doctor_id="doctor_a")
            reg = ModelRegistry(store)
            # Create version dir with fake weights + worse CER meta
            ver = "v0001"
            dest = store.adapter_version_dir("doctor_a", ver)
            dest.mkdir(parents=True)
            (dest / "model.safetensors").write_bytes(b"fake")
            meta = {
                "doctor_id": "doctor_a",
                "version": ver,
                "method": "decoder_finetune",
                "base_checkpoint": str(root / "checkpoints" / "trocr-small-handwritten"),
                "created_at_utc": "2026-11-03T00:00:00+00:00",
                "val_cer": 0.9,
                "general_val_cer": 0.2,
                "improves_over_general": False,
                "status": "candidate",
                "notes": "",
                "artifact_kind": "vision_encoder_decoder_dir",
            }
            (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                reg.activate(auth, "doctor_a", ver, force=False)
            # Force activate then selector must still fall back when require_improves
            reg.activate(auth, "doctor_a", ver, force=True)
            decision = select_checkpoint_for_doctor(
                store, "doctor_a", project_root=root, require_improves_over_general=True
            )
            self.assertEqual(decision.source, "general_fallback")
            self.assertEqual(
                decision.reason, "personalized_worse_than_general_on_val_cer"
            )

    def test_versioning_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints" / "trocr-small-handwritten").mkdir(parents=True)
            store = PersonalizationStore(root / "personalization")
            create_doctor_profile(store, "doctor_a")
            auth = DoctorAuthContext(doctor_id="doctor_a")
            reg = ModelRegistry(store)
            for ver, improves in (("v0001", True), ("v0002", True)):
                dest = store.adapter_version_dir("doctor_a", ver)
                dest.mkdir(parents=True)
                (dest / "model.safetensors").write_bytes(b"fake-" + ver.encode())
                meta = {
                    "doctor_id": "doctor_a",
                    "version": ver,
                    "method": "decoder_finetune",
                    "base_checkpoint": "checkpoints/trocr-small-handwritten",
                    "created_at_utc": "2026-11-03T00:00:00+00:00",
                    "val_cer": 0.1,
                    "general_val_cer": 0.5,
                    "improves_over_general": improves,
                    "status": "candidate",
                    "notes": "",
                    "artifact_kind": "vision_encoder_decoder_dir",
                }
                (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            reg.activate(auth, "doctor_a", "v0001")
            self.assertEqual(reg.get_active_version("doctor_a"), "v0001")
            reg.activate(auth, "doctor_a", "v0002")
            self.assertEqual(reg.get_active_version("doctor_a"), "v0002")
            # rollback to v0001
            out = reg.rollback(auth, "doctor_a", to_version="v0001")
            self.assertEqual(out["active"], "v0001")
            # rollback to general
            out2 = reg.rollback(auth, "doctor_a", to_version=None)
            self.assertIsNone(out2["active"])
            self.assertEqual(out2["fallback"], "general")
            decision = select_checkpoint_for_doctor(
                store, "doctor_a", project_root=root
            )
            self.assertEqual(decision.source, "general")


class TestCorrectionsAndTrainingGate(unittest.TestCase):
    def test_unapproved_correction_not_in_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonalizationStore(Path(tmp) / "personalization")
            create_doctor_profile(store, "doctor_a")
            auth = DoctorAuthContext(doctor_id="doctor_a")
            wf = CorrectionWorkflow(store)
            ticket = wf.submit(
                auth,
                "doctor_a",
                image_filename="line.png",
                proposed_transcription="corrected text",
                model_prediction="wrong",
            )
            self.assertEqual(ticket.status, "pending")
            samples = SampleRegistry(store).list_samples("doctor_a")
            self.assertEqual(samples, [])

    def test_training_gated_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints" / "trocr-small-handwritten").mkdir(parents=True)
            store = PersonalizationStore(root / "personalization")
            create_doctor_profile(store, "doctor_a")
            trainer = PersonalizationTrainer(store, project_root=root)
            plan = trainer.build_plan("doctor_a", allow_training=False)
            self.assertFalse(plan.ready)
            self.assertTrue(any("gated" in b for b in plan.blockers))


class TestDemoUntouched(unittest.TestCase):
    def test_run_demo_does_not_import_personalization(self) -> None:
        src = (ROOT / "scripts" / "run_demo.py").read_text(encoding="utf-8")
        self.assertNotIn("personalization", src)
        self.assertIn("trocr-demo-finetuned", src)


if __name__ == "__main__":
    unittest.main()
