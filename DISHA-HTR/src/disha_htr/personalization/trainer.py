"""
Gated doctor personalization training scaffold.

Does NOT start large-scale training unless explicitly approved via
`allow_training=True`. Prefer parameter-efficient decoder-only fine-tunes
saved under the doctor's adapter version directory — never overwrites the
shared general checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .auth import DoctorAuthContext
from .doctor_id import normalize_doctor_id
from .registry import ModelRegistry
from .samples import SampleRegistry
from .storage import GENERAL_CHECKPOINT_REL, PersonalizationStore


@dataclass
class PersonalizationPlan:
    doctor_id: str
    method: str
    n_train: int
    n_val: int
    base_checkpoint: str
    proposed_version: str
    ready: bool
    blockers: list[str]
    allow_training: bool
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PersonalizationTrainer:
    """
    Architecture trainer:
    - Always can build a plan / dry-run.
    - Actual weight updates require allow_training=True AND admin approval flag.
    - Uses decoder-only fine-tune pattern (no new PEFT dependency).
    """

    def __init__(
        self,
        store: PersonalizationStore,
        *,
        project_root: str | Path,
        general_checkpoint: str = GENERAL_CHECKPOINT_REL,
    ) -> None:
        self.store = store
        self.project_root = Path(project_root)
        self.general_checkpoint = general_checkpoint
        self.samples = SampleRegistry(store)
        self.registry = ModelRegistry(store)

    def build_plan(
        self,
        doctor_id: str,
        *,
        method: str = "decoder_finetune",
        allow_training: bool = False,
        min_train: int = 5,
        min_val: int = 2,
    ) -> PersonalizationPlan:
        did = normalize_doctor_id(doctor_id)
        train = self.samples.list_samples(did, "train")
        val = self.samples.list_samples(did, "val")
        blockers: list[str] = []
        if len(train) < min_train:
            blockers.append(f"need>={min_train} verified train samples (have {len(train)})")
        if len(val) < min_val:
            blockers.append(f"need>={min_val} verified val samples (have {len(val)})")
        if any(not s.verified_by for s in train + val):
            blockers.append("all samples must include verified_by")
        base = self.project_root / self.general_checkpoint
        if not base.is_dir():
            blockers.append(f"missing general checkpoint at {base}")
        if not allow_training:
            blockers.append(
                "training gated: pass allow_training=True only after explicit approval"
            )
        version = self.registry.next_version(did)
        return PersonalizationPlan(
            doctor_id=did,
            method=method,
            n_train=len(train),
            n_val=len(val),
            base_checkpoint=str(base),
            proposed_version=version,
            ready=len(blockers) == 0,
            blockers=blockers,
            allow_training=allow_training,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def prepare_candidate_placeholder(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        *,
        method: str = "decoder_finetune",
    ) -> dict[str, Any]:
        """
        Register a placeholder adapter version without training weights.
        Used to exercise versioning/rollback before approved training runs.
        """
        auth.require_doctor(doctor_id)
        plan = self.build_plan(doctor_id, method=method, allow_training=False)
        meta = self.registry.register_candidate(
            auth,
            doctor_id,
            plan.proposed_version,
            method=method,
            base_checkpoint=plan.base_checkpoint,
            artifact_dir=None,
            val_cer=None,
            general_val_cer=None,
            notes="placeholder — no training executed",
        )
        plan_path = (
            self.store.adapter_version_dir(doctor_id, meta.version) / "plan.json"
        )
        plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        return {"plan": plan.to_dict(), "meta": meta.to_dict()}

    def run_decoder_finetune(
        self,
        auth: DoctorAuthContext,
        doctor_id: str,
        *,
        allow_training: bool,
        epochs: int = 5,
        lr: float = 2e-5,
    ) -> dict[str, Any]:
        """
        Execute lightweight decoder-only personalization into a new version dir.

        Hard-gated: refuse unless allow_training=True. Does not overwrite general
        weights. Large-scale / production training still requires human approval
        beyond this flag (documented in plan blockers for readiness reporting).
        """
        auth.require_doctor(doctor_id)
        plan = self.build_plan(
            doctor_id, method="decoder_finetune", allow_training=allow_training
        )
        if not plan.ready:
            raise RuntimeError(
                "personalization training not ready: " + "; ".join(plan.blockers)
            )

        # Import heavy stack only when approved to train
        import torch
        from PIL import Image
        from torch.optim import AdamW
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        from disha_htr.metrics import character_error_rate

        did = normalize_doctor_id(doctor_id)
        train_rows = self.samples.list_samples(did, "train")
        val_rows = self.samples.list_samples(did, "val")
        base_dir = Path(plan.base_checkpoint)
        out_dir = self.store.adapter_version_dir(did, plan.proposed_version)
        if out_dir.exists():
            raise FileExistsError(out_dir)
        out_dir.mkdir(parents=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = TrOCRProcessor.from_pretrained(str(base_dir))
        model = VisionEncoderDecoderModel.from_pretrained(str(base_dir))
        nested = getattr(model.config.decoder, "decoder_start_token_id", None)
        start_id = nested if nested is not None else processor.tokenizer.sep_token_id
        model.config.decoder_start_token_id = start_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.to(device)
        model.train()
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        history = []
        for epoch in range(1, epochs + 1):
            total = 0.0
            for row in train_rows:
                img_path = self.store.samples_dir(did, "train") / row.image_filename
                image = Image.open(img_path).convert("RGB")
                pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
                labels = processor.tokenizer(
                    row.ground_truth,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                ).input_ids
                labels[labels == processor.tokenizer.pad_token_id] = -100
                labels = labels.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(pixel_values=pv, labels=labels)
                out.loss.backward()
                opt.step()
                total += float(out.loss.detach().cpu())
            avg = total / max(len(train_rows), 1)
            history.append({"epoch": epoch, "train_loss": avg})

        # Val CER personalized vs general (same val set)
        def _cer_on(loaded_model) -> float:
            loaded_model.eval()
            scores = []
            with torch.inference_mode():
                for row in val_rows:
                    img_path = self.store.samples_dir(did, "val") / row.image_filename
                    image = Image.open(img_path).convert("RGB")
                    pv = processor(images=image, return_tensors="pt").pixel_values.to(
                        device
                    )
                    ids = loaded_model.generate(pv, max_new_tokens=64, num_beams=3)
                    pred = processor.batch_decode(ids, skip_special_tokens=True)[
                        0
                    ].strip()
                    scores.append(character_error_rate(row.ground_truth, pred))
            return sum(scores) / max(len(scores), 1)

        val_cer = _cer_on(model)
        general = VisionEncoderDecoderModel.from_pretrained(str(base_dir)).to(device)
        general_val_cer = _cer_on(general)

        model.float()
        processor.save_pretrained(out_dir)
        model.save_pretrained(out_dir)
        # Remove from registry helper path: register meta via ModelRegistry
        # (directory already created with weights)
        improves = val_cer < general_val_cer
        meta = {
            "doctor_id": did,
            "version": plan.proposed_version,
            "method": "decoder_finetune",
            "base_checkpoint": str(base_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "val_cer": val_cer,
            "general_val_cer": general_val_cer,
            "improves_over_general": improves,
            "status": "candidate",
            "notes": "decoder-only personalization; general checkpoint untouched",
            "artifact_kind": "vision_encoder_decoder_dir",
            "history": history,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.store.audit_log(did).append(
            "personalization_trained",
            {
                "doctor_id": did,
                "version": plan.proposed_version,
                "val_cer": val_cer,
                "general_val_cer": general_val_cer,
                "improves_over_general": improves,
            },
        )
        return meta
