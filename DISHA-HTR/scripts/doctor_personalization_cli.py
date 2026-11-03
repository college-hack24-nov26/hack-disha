"""
CLI for doctor personalization architecture (no large-scale training by default).

Examples:
  create-profile --doctor-id doc_a --display-name "Dr A"
  plan --doctor-id doc_a
  prepare-placeholder --doctor-id doc_a
  rollback --doctor-id doc_a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.personalization.auth import DoctorAuthContext  # noqa: E402
from disha_htr.personalization.registry import ModelRegistry  # noqa: E402
from disha_htr.personalization.selector import select_checkpoint_for_doctor  # noqa: E402
from disha_htr.personalization.storage import (  # noqa: E402
    PersonalizationStore,
    create_doctor_profile,
)
from disha_htr.personalization.trainer import PersonalizationTrainer  # noqa: E402


def _store(args: argparse.Namespace) -> PersonalizationStore:
    root = Path(args.store_root)
    if not root.is_absolute():
        root = ROOT / root
    return PersonalizationStore(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Doctor personalization CLI")
    parser.add_argument(
        "--store-root",
        default="personalization",
        help="Root directory for per-doctor data (default: ./personalization)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create-profile", help="Create isolated doctor profile")
    p_create.add_argument("--doctor-id", required=True)
    p_create.add_argument("--display-name", default=None)

    p_plan = sub.add_parser("plan", help="Show gated training readiness plan")
    p_plan.add_argument("--doctor-id", required=True)
    p_plan.add_argument(
        "--allow-training",
        action="store_true",
        help="Only set after explicit approval (still needs enough samples)",
    )

    p_ph = sub.add_parser(
        "prepare-placeholder",
        help="Register placeholder adapter version (no weight training)",
    )
    p_ph.add_argument("--doctor-id", required=True)

    p_sel = sub.add_parser("select", help="Show which checkpoint would be used")
    p_sel.add_argument("--doctor-id", required=True)

    p_rb = sub.add_parser("rollback", help="Rollback to general or a version")
    p_rb.add_argument("--doctor-id", required=True)
    p_rb.add_argument("--to-version", default=None)

    p_list = sub.add_parser("list-doctors", help="List doctor profiles")

    args = parser.parse_args()
    store = _store(args)

    if args.cmd == "create-profile":
        profile = create_doctor_profile(
            store, args.doctor_id, display_name=args.display_name
        )
        print(json.dumps(profile.to_dict(), indent=2))
        return 0

    if args.cmd == "plan":
        trainer = PersonalizationTrainer(store, project_root=ROOT)
        plan = trainer.build_plan(
            args.doctor_id, allow_training=bool(args.allow_training)
        )
        print(json.dumps(plan.to_dict(), indent=2))
        return 0

    if args.cmd == "prepare-placeholder":
        auth = DoctorAuthContext(doctor_id=args.doctor_id)
        trainer = PersonalizationTrainer(store, project_root=ROOT)
        out = trainer.prepare_candidate_placeholder(auth, args.doctor_id)
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "select":
        decision = select_checkpoint_for_doctor(
            store, args.doctor_id, project_root=ROOT
        )
        print(json.dumps(decision.to_dict(), indent=2))
        return 0

    if args.cmd == "rollback":
        auth = DoctorAuthContext(doctor_id=args.doctor_id)
        reg = ModelRegistry(store)
        out = reg.rollback(auth, args.doctor_id, to_version=args.to_version)
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "list-doctors":
        print(json.dumps({"doctors": store.list_doctor_ids()}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
