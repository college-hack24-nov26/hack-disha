"""
CPU-only tests for IAM dataset parsing / pairing / manifests.

No network, no GPU, no model load.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.iam_dataset import (  # noqa: E402
    build_split_line_ids,
    compute_stats,
    expand_aachen_form_ids_to_line_ids,
    form_id_from_line_id,
    pair_lines,
    parse_forms_txt,
    parse_lines_txt,
    writer_overlap,
    write_manifest_csv,
)


def _write_fixture(tmp: Path) -> tuple[Path, Path, Path]:
    ascii_dir = tmp / "ascii"
    ascii_dir.mkdir()
    # Official IAM lines.txt: id status graylevel components x y w h transcription
    (ascii_dir / "lines.txt").write_text(
        "\n".join(
            [
                "# comment",
                "a01-000u-00 ok 154 1 1 1 10 10 A|MOVE|to|stop",
                "a01-000u-01 ok 154 1 1 1 10 10 Mr.|Gaitskell",
                "a01-000x-00 ok 154 1 1 1 10 10 Hello|world",
                "a01-003-00 err 154 1 1 1 10 10 should|skip",
                "b01-000-00 ok 154 1 1 1 10 10 Other|writer|line",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ascii_dir / "forms.txt").write_text(
        "\n".join(
            [
                "# form_id writer_id ...",
                "a01-000u 000 x",
                "a01-000x 000 x",
                "a01-003 001 x",
                "b01-000 050 x",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    lines_root = tmp / "lines"
    for line_id in ("a01-000u-00", "a01-000u-01", "a01-000x-00", "b01-000-00"):
        form_id = form_id_from_line_id(line_id)
        prefix = form_id.split("-")[0]
        out = lines_root / prefix / form_id
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{line_id}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    splits = tmp / "splits"
    splits.mkdir()
    (splits / "train.uttlist").write_text("a01-000u\na01-000x\n", encoding="utf-8")
    (splits / "validation.uttlist").write_text("b01-000\n", encoding="utf-8")
    (splits / "test.uttlist").write_text("a01-003\n", encoding="utf-8")
    return ascii_dir, lines_root, splits


class TestIamParsing(unittest.TestCase):
    def test_form_id_and_transcription(self) -> None:
        self.assertEqual(form_id_from_line_id("a01-000u-00"), "a01-000u")
        with tempfile.TemporaryDirectory() as tmp:
            ascii_dir, _, _ = _write_fixture(Path(tmp))
            meta = parse_lines_txt(ascii_dir / "lines.txt")
            self.assertEqual(meta["a01-000u-00"][1], "A MOVE to stop")
            forms = parse_forms_txt(ascii_dir / "forms.txt")
            self.assertEqual(forms["a01-000u"], "000")


class TestPairingAndSplits(unittest.TestCase):
    def test_pair_skips_err_and_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ascii_dir, lines_root, _ = _write_fixture(Path(tmp))
            meta = parse_lines_txt(ascii_dir / "lines.txt")
            forms = parse_forms_txt(ascii_dir / "forms.txt")
            records, problems = pair_lines(
                meta,
                forms,
                lines_root,
                ["a01-000u-00", "a01-003-00", "missing-line-00"],
                require_image=True,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].text, "A MOVE to stop")
            self.assertTrue(any(p.startswith("bad_status:") for p in problems))
            self.assertTrue(any(p.startswith("missing_ascii:") for p in problems))

    def test_aachen_expand_and_no_writer_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ascii_dir, lines_root, splits = _write_fixture(Path(tmp))
            meta = parse_lines_txt(ascii_dir / "lines.txt")
            forms = parse_forms_txt(ascii_dir / "forms.txt")
            split_ids = build_split_line_ids(
                split_scheme="aachen", splits_dir=splits, lines_meta=meta
            )
            self.assertIn("a01-000u-00", split_ids["train"])
            self.assertIn("b01-000-00", split_ids["val"])

            split_records = {}
            for name, ids in split_ids.items():
                rows, _ = pair_lines(meta, forms, lines_root, ids, require_image=True)
                # drop err-only forms (test list has err line → empty)
                split_records[name] = [r for r in rows if r.writer_id]
            overlaps = writer_overlap(
                {
                    "train": split_records["train"],
                    "val": split_records["val"],
                }
            )
            self.assertEqual(overlaps["train|val"], [])

            out = Path(tmp) / "train.csv"
            write_manifest_csv(out, split_records["train"], split="train")
            text = out.read_text(encoding="utf-8")
            self.assertIn("image_path,text,writer_id,line_id,form_id,split", text)

            stats = compute_stats(split_records, {"train": []}, split_scheme="aachen")
            self.assertGreaterEqual(stats["splits"]["train"]["lines"], 1)


class TestAachenExpandHelper(unittest.TestCase):
    def test_expand_form_ids(self) -> None:
        meta = {
            "a01-000u-00": ("ok", "x"),
            "a01-000u-01": ("ok", "y"),
            "b01-000-00": ("ok", "z"),
        }
        ids = expand_aachen_form_ids_to_line_ids(["a01-000u"], meta)
        self.assertEqual(ids, ["a01-000u-00", "a01-000u-01"])


if __name__ == "__main__":
    unittest.main()
