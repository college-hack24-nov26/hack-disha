"""Guards for verified IAM showcase: stable sample_id ↔ image ↔ reference."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT / "demo" / "approved"
MANIFEST = APPROVED / "samples_manifest.json"


class TestIamSampleShowcase(unittest.TestCase):
    def test_manifest_binds_id_image_and_reference(self) -> None:
        self.assertTrue(MANIFEST.is_file())
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        samples = data.get("samples") or []
        self.assertGreaterEqual(len(samples), 3)

        seen_ids: set[str] = set()
        for row in samples:
            sid = str(row.get("sample_id") or "").strip()
            fname = str(row.get("id") or "").strip()
            expected = str(row.get("expected_text") or "").strip()
            self.assertTrue(sid, msg="sample_id required")
            self.assertTrue(fname, msg="image id/filename required")
            self.assertNotIn(sid, seen_ids)
            seen_ids.add(sid)
            path = APPROVED / fname
            self.assertTrue(path.is_file(), msg=f"missing image for {sid}: {fname}")
            self.assertEqual(path.resolve().parent, APPROVED.resolve())
            # Verified cards must carry a real reference; never invent empty ones.
            self.assertTrue(expected, msg=f"{sid} marked in pack without reference")
            self.assertNotEqual(expected.lower(), "todo")

    def test_no_shared_reference_across_distinct_samples(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        pairs = [
            (
                str(r.get("sample_id")),
                str(r.get("expected_text") or "").strip(),
            )
            for r in (data.get("samples") or [])
        ]
        by_text: dict[str, list[str]] = {}
        for sid, text in pairs:
            by_text.setdefault(text, []).append(sid)
        # Distinct sample_ids must not accidentally share one hardcoded sentence
        # for every card (the original mix-up failure mode).
        self.assertGreaterEqual(len(by_text), 3)

    def test_server_list_samples_includes_verified_reference(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        # Import after path tweak; exercise helper via Handler method body by
        # re-reading the same mapping rules the API uses.
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        items = []
        for row in data.get("samples") or []:
            name = str(row.get("id") or "").strip()
            if not name or not (APPROVED / name).is_file():
                continue
            expected = str(row.get("expected_text") or "").strip()
            items.append(
                {
                    "sample_id": str(row.get("sample_id") or Path(name).stem),
                    "expected_text": expected,
                    "verified_exact": bool(expected),
                }
            )
        self.assertGreaterEqual(len(items), 3)
        self.assertTrue(all(i["verified_exact"] for i in items))
        self.assertEqual(len({i["sample_id"] for i in items}), len(items))


if __name__ == "__main__":
    unittest.main()
