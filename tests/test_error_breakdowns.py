from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path
from unittest import mock

from scripts import build_error_breakdowns as breakdowns
from scripts.error_analysis import EVENT_COLUMNS, build_error_analysis


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_error_breakdowns.py"


def prediction_row(**updates: str) -> dict[str, str]:
    row = {
        "utt_id": "utt-001",
        "dataset": "fleurs",
        "model": "phowhisper",
        "model_size": "base",
        "train_type": "ordinary_lora",
        "lambda": "0",
        "seed": "42",
        "snr": "clean",
        "noise_type": "clean",
        "ref": "xin chào",
        "hyp": "xin chào",
    }
    row.update(updates)
    return row


def make_events(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    events, _ = build_error_analysis(rows)
    return [{column: str(event[column]) for column in EVENT_COLUMNS} for event in events]


def write_events(
    path: Path,
    rows: list[dict[str, str]],
    *,
    columns: list[str] | None = None,
) -> None:
    selected = columns or EVENT_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=selected,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def full_fixture() -> list[dict[str, str]]:
    ordinary_pairs = [
        ("tone-loss", "má", "ma"),
        ("tone-addition", "ma", "má"),
        ("tone-substitution", "má", "mà"),
        ("tone-word-deletion", "má", ""),
        ("quality-loss-breve", "băn", "ban"),
        ("quality-loss-circ-a", "bân", "ban"),
        ("quality-loss-circ-e", "bên", "ben"),
        ("quality-loss-circ-o", "bôn", "bon"),
        ("quality-loss-horn-o", "bơn", "bon"),
        ("quality-loss-horn-u", "bưng", "bung"),
        ("quality-addition", "ban", "băn"),
        ("quality-marked-marked", "băn", "bân"),
        ("d-stroke", "đi", "di"),
        ("mixed", "đăm", "dam"),
        ("combined-loss", "bắc", "bac"),
        ("wer-substitution", "tôi đi", "tôi về"),
        ("wer-deletion", "tôi đi", "tôi"),
        ("wer-insertion", "tôi", "tôi đi"),
    ]
    predictions = [
        prediction_row(utt_id=utt_id, ref=ref, hyp=hyp)
        for utt_id, ref, hyp in ordinary_pairs
    ]
    predictions.extend(
        [
            prediction_row(
                utt_id="candidate-exact",
                train_type="tone_aware_lora",
                **{"lambda": "0.05"},
            ),
            prediction_row(
                utt_id="candidate-tone-loss",
                train_type="tone_aware_lora",
                ref="má",
                hyp="ma",
                **{"lambda": "0.1"},
            ),
        ]
    )
    return make_events(predictions)


class ErrorBreakdownTest(unittest.TestCase):
    def test_schemas_and_primary_categories_are_fixed(self) -> None:
        self.assertEqual(
            breakdowns.TER_CATEGORY_ORDER,
            ("word_deletion", "tone_loss", "tone_addition", "tone_substitution"),
        )
        self.assertEqual(
            breakdowns.DER_CATEGORY_ORDER,
            (
                "vowel_quality_loss",
                "vowel_quality_addition",
                "vowel_quality_substitution",
                "d_stroke_loss_or_change",
                "mixed_or_other",
            ),
        )
        self.assertEqual(
            breakdowns.ORTHOGRAPHIC_CATEGORY_ORDER,
            ("missing_diacritic", "wrong_tone_mark", "wrong_vowel_mark"),
        )
        self.assertEqual(len(breakdowns.OUTPUT_NAMES), 5)

    def test_breakdowns_reconcile_and_keep_three_runs_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "error_events.csv"
            write_events(event_path, full_fixture())

            result = breakdowns.build_breakdowns(event_path)

        self.assertEqual(len(result.run_keys), 3)
        self.assertEqual(
            {(row["train_type"], row["lambda"]) for row in result.wer_rows},
            {
                ("ordinary_lora", "0"),
                ("tone_aware_lora", "0.05"),
                ("tone_aware_lora", "0.1"),
            },
        )
        for row in result.wer_rows:
            components = sum(
                int(row[name]) for name in ("substitutions", "deletions", "insertions")
            )
            self.assertEqual(components, int(row["word_errors"]))
            component_rate = sum(
                float(row[name])
                for name in ("substitution_rate", "deletion_rate", "insertion_rate")
            )
            self.assertAlmostEqual(component_rate, float(row["wer"]), places=11)

        for run in result.run_keys:
            metadata = dict(zip(breakdowns.RUN_COLUMNS, run))
            ter = [
                row
                for row in result.ter_rows
                if all(row[column] == metadata[column] for column in breakdowns.RUN_COLUMNS)
            ]
            der = [
                row
                for row in result.der_rows
                if all(row[column] == metadata[column] for column in breakdowns.RUN_COLUMNS)
            ]
            self.assertEqual(sum(int(row["count"]) for row in ter), int(ter[0]["tone_errors"]))
            self.assertEqual(
                sum(int(row["count"]) for row in der), int(der[0]["diacritic_errors"])
            )

        ordinary_ter = {
            row["category"]: int(row["count"])
            for row in result.ter_rows
            if row["train_type"] == "ordinary_lora"
        }
        self.assertEqual(
            ordinary_ter,
            {
                "word_deletion": 2,
                "tone_loss": 1,
                "tone_addition": 1,
                "tone_substitution": 1,
            },
        )

    def test_quality_pairs_d_stroke_combined_loss_and_overlap_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "error_events.csv"
            write_events(event_path, full_fixture())
            result = breakdowns.build_breakdowns(event_path)

        ordinary_der = {
            row["category"]: int(row["count"])
            for row in result.der_rows
            if row["train_type"] == "ordinary_lora"
        }
        self.assertEqual(ordinary_der["vowel_quality_loss"], 7)
        self.assertEqual(ordinary_der["vowel_quality_addition"], 1)
        self.assertEqual(ordinary_der["vowel_quality_substitution"], 1)
        self.assertEqual(ordinary_der["d_stroke_loss_or_change"], 1)
        self.assertEqual(ordinary_der["mixed_or_other"], 1)

        ordinary_orthographic = {
            row["category"]: int(row["count"])
            for row in result.orthographic_rows
            if row["train_type"] == "ordinary_lora"
        }
        self.assertEqual(ordinary_orthographic["missing_diacritic"], 10)
        self.assertEqual(ordinary_orthographic["wrong_tone_mark"], 2)
        self.assertEqual(ordinary_orthographic["wrong_vowel_mark"], 2)

        transitions = {
            row["quality_transitions"]
            for row in result.event_rows
            if row["der_primary_category"] == "vowel_quality_loss"
        }
        self.assertTrue({"ă→a", "â→a", "ê→e", "ô→o", "ơ→o", "ư→u"} <= transitions)

        combined = next(row for row in result.event_rows if row["utt_id"] == "combined-loss")
        self.assertEqual(combined["der_primary_category"], "vowel_quality_loss")
        self.assertEqual(combined["tone_primary_category"], "")
        self.assertEqual(combined["has_tone_loss"], "true")
        self.assertEqual(combined["has_vowel_quality_loss"], "true")
        self.assertEqual(combined["missing_diacritic"], "true")

        mixed = next(row for row in result.event_rows if row["utt_id"] == "mixed")
        self.assertEqual(mixed["der_primary_category"], "mixed_or_other")
        self.assertEqual(mixed["quality_transitions"], "đ→d;ă→a")
        self.assertEqual(mixed["has_d_stroke_loss"], "true")
        self.assertEqual(mixed["has_vowel_quality_loss"], "true")

    def test_reverse_and_marked_marked_pairs_are_classified(self) -> None:
        cases = {
            ("a", "ă"): "vowel_quality_addition",
            ("a", "â"): "vowel_quality_addition",
            ("e", "ê"): "vowel_quality_addition",
            ("o", "ô"): "vowel_quality_addition",
            ("o", "ơ"): "vowel_quality_addition",
            ("u", "ư"): "vowel_quality_addition",
            ("ă", "â"): "vowel_quality_substitution",
            ("ô", "ơ"): "vowel_quality_substitution",
            ("đ", "d"): "d_stroke_loss_or_change",
            ("d", "đ"): "d_stroke_loss_or_change",
        }
        for transition, category in cases.items():
            with self.subTest(transition=transition):
                self.assertEqual(breakdowns._der_category((transition,)), category)

    def test_nfd_fields_are_normalized_to_nfc_without_changing_classification(self) -> None:
        rows = make_events([prediction_row(ref="bắc", hyp="bac")])
        for row in rows:
            for field in (
                "ref_token",
                "hyp_token",
                "ref_tone_base",
                "hyp_tone_base",
                "ref_plain_base",
                "hyp_plain_base",
                "ref",
                "hyp",
            ):
                row[field] = unicodedata.normalize("NFD", row[field])
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "error_events.csv"
            write_events(event_path, rows)
            result = breakdowns.build_breakdowns(event_path)

        self.assertEqual(result.der_rows[0]["category"], "vowel_quality_loss")
        event = result.event_rows[0]
        self.assertEqual(event["ref_token"], "bắc")
        self.assertTrue(unicodedata.is_normalized("NFC", str(event["ref"])))

    def test_orthographic_diagnostics_keep_combined_tone_and_vowel_error(self) -> None:
        rows = make_events([prediction_row(ref="mọi", hyp="mỗi")])
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "error_events.csv"
            write_events(event_path, rows)
            result = breakdowns.build_breakdowns(event_path)

        counts = {
            row["category"]: int(row["count"])
            for row in result.orthographic_rows
        }
        self.assertEqual(counts["missing_diacritic"], 0)
        self.assertEqual(counts["wrong_tone_mark"], 1)
        self.assertEqual(counts["wrong_vowel_mark"], 1)

    def test_cli_is_deterministic_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "breakdowns"
            write_events(event_path, full_fixture())
            source_before = event_path.read_bytes()
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(SCRIPT),
                "--events",
                str(event_path),
                "--out-dir",
                str(output_dir),
            ]

            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            outputs = [output_dir / name for name in breakdowns.OUTPUT_NAMES]
            first_bytes = [path.read_bytes() for path in outputs]
            self.assertTrue((output_dir / "error_breakdowns.provenance.json").is_file())
            self.assertTrue((output_dir / "error_breakdowns.bundle.commit.json").is_file())
            resumed = subprocess.run(
                [*command, "--resume"], capture_output=True, text=True, check=False
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual([path.read_bytes() for path in outputs], first_bytes)
            self.assertEqual(event_path.read_bytes(), source_before)
            expected_columns = (
                breakdowns.WER_COLUMNS,
                breakdowns.TER_COLUMNS,
                breakdowns.DER_COLUMNS,
                breakdowns.ORTHOGRAPHIC_COLUMNS,
                breakdowns.DIACRITIC_EVENT_COLUMNS,
            )
            for path, columns in zip(outputs, expected_columns):
                header, _ = read_csv(path)
                self.assertEqual(header, columns)

            refused = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("output already exists", refused.stderr)
            self.assertEqual([path.read_bytes() for path in outputs], first_bytes)

            rerun = subprocess.run(
                [*command, "--overwrite"], capture_output=True, text=True, check=False
            )
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual([path.read_bytes() for path in outputs], first_bytes)
            self.assertFalse(any(output_dir.glob(".*.tmp")))
            self.assertFalse(any(output_dir.glob(".*.bak")))
            self.assertFalse((output_dir / breakdowns.OUTPUT_LOCK_NAME).exists())

    def test_exact_header_boolean_and_duplicate_event_are_validated(self) -> None:
        rows = make_events([prediction_row(ref="má", hyp="ma")])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_header = root / "bad_header.csv"
            write_events(bad_header, rows, columns=EVENT_COLUMNS[:-1])
            with self.assertRaisesRegex(ValueError, "expected exact"):
                breakdowns.build_breakdowns(bad_header)

            bad_bool = root / "bad_bool.csv"
            invalid = [dict(rows[0], tone_error="yes")]
            write_events(bad_bool, invalid)
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                breakdowns.build_breakdowns(bad_bool)

            duplicate = root / "duplicate.csv"
            write_events(duplicate, [rows[0], rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate aligned event"):
                breakdowns.build_breakdowns(duplicate)

    def test_writer_failure_removes_temporary_files_and_leaves_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "events.csv"
            output_dir = root / "output"
            write_events(event_path, full_fixture())
            result = breakdowns.build_breakdowns(event_path)
            original = breakdowns._write_csv_temp
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected writer failure")
                return original(*args, **kwargs)

            with mock.patch.object(breakdowns, "_write_csv_temp", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected writer failure"):
                    breakdowns.write_breakdown_outputs(event_path, output_dir, result)

            self.assertFalse(any((output_dir / name).exists() for name in breakdowns.OUTPUT_NAMES))
            self.assertFalse(any(output_dir.glob(".*.tmp")))
            self.assertFalse((output_dir / breakdowns.OUTPUT_LOCK_NAME).exists())


if __name__ == "__main__":
    unittest.main()
