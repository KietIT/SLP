from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

from scripts.error_analysis import (
    CANONICAL_COLUMNS,
    EVENT_COLUMNS,
    METRIC_VERSION,
    SUMMARY_COLUMNS,
    build_error_analysis,
    load_prediction_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "error_analysis.py"
EXPECTED_EVENT_COLUMNS = [
    "metric_version",
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "utt_id",
    "snr",
    "noise_type",
    "operation",
    "ref_token",
    "hyp_token",
    "ref_index",
    "hyp_index",
    "ref",
    "hyp",
    "ref_tone_base",
    "hyp_tone_base",
    "ref_plain_base",
    "hyp_plain_base",
    "ref_tone",
    "hyp_tone",
    "ref_coda",
    "hyp_coda",
    "tone_eligible",
    "tone_error",
    "diacritic_eligible",
    "diacritic_error",
    "final_consonant_eligible",
    "final_consonant_error",
    "short_word_deletion",
]


def prediction_row(**updates: str) -> dict[str, str]:
    row = {
        "utt_id": "utt-001",
        "dataset": "vivos",
        "model": "phowhisper",
        "model_size": "base",
        "train_type": "zero_shot",
        "lambda": "",
        "seed": "42",
        "snr": "clean",
        "noise_type": "clean",
        "ref": "tôi đã đi",
        "hyp": "tôi đã đi",
    }
    row.update(updates)
    return row


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def assert_summary_reconciles(test: unittest.TestCase, row: dict[str, object]) -> None:
    matches = int(row["matches"])
    substitutions = int(row["substitutions"])
    deletions = int(row["deletions"])
    insertions = int(row["insertions"])
    test.assertEqual(int(row["n_events"]), matches + substitutions + deletions + insertions)
    test.assertEqual(int(row["n_ref_tokens"]), matches + substitutions + deletions)
    test.assertEqual(int(row["n_hyp_tokens"]), matches + substitutions + insertions)
    test.assertEqual(int(row["word_errors"]), substitutions + deletions + insertions)
    test.assertAlmostEqual(
        float(row["word_error_rate"]),
        (substitutions + deletions + insertions) / max(matches + substitutions + deletions, 1),
        places=11,
    )
    for prefix in ("tone", "diacritic", "final_consonant"):
        eligible = int(row[f"{prefix}_eligible"])
        errors = int(row[f"{prefix}_errors"])
        test.assertLessEqual(errors, eligible)
        test.assertAlmostEqual(
            float(row[f"{prefix}_error_rate"]), errors / max(eligible, 1), places=11
        )


class ErrorAnalysisTest(unittest.TestCase):
    def test_event_schema_is_the_approved_gate_c_contract(self) -> None:
        self.assertEqual(EVENT_COLUMNS, EXPECTED_EVENT_COLUMNS)

    def test_empty_hypothesis_produces_only_deletions(self) -> None:
        events, summaries = build_error_analysis(
            [prediction_row(ref="đã có một", hyp="")]
        )

        self.assertEqual([event["operation"] for event in events], ["deletion"] * 3)
        self.assertEqual([event["ref_token"] for event in events], ["đã", "có", "một"])
        self.assertTrue(all(event["hyp_token"] == "" for event in events))
        self.assertTrue(all(event["hyp_index"] == "" for event in events))
        self.assertTrue(all(event["metric_version"] == METRIC_VERSION for event in events))
        self.assertTrue(all(event["ref"] == "đã có một" for event in events))
        self.assertTrue(all(event["hyp"] == "" for event in events))
        self.assertEqual(
            [event["ref_tone_base"] for event in events], ["đa", "co", "môt"]
        )
        self.assertEqual(
            [event["ref_plain_base"] for event in events], ["da", "co", "mot"]
        )
        self.assertTrue(all(event["tone_eligible"] == "true" for event in events))
        self.assertTrue(all(event["tone_error"] == "true" for event in events))
        self.assertTrue(all(event["diacritic_eligible"] == "false" for event in events))
        self.assertEqual(
            [event["final_consonant_eligible"] for event in events],
            ["false", "false", "true"],
        )
        self.assertTrue(all(event["short_word_deletion"] == "true" for event in events))
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["deletions"], 3)
        self.assertEqual(summaries[0]["n_hyp_tokens"], 0)
        assert_summary_reconciles(self, summaries[0])

    def test_repeated_tokens_have_deterministic_indexed_alignment(self) -> None:
        events, summaries = build_error_analysis(
            [prediction_row(ref="tôi đã tôi", hyp="tôi tôi")]
        )

        self.assertEqual(
            [(event["operation"], event["ref_token"], event["hyp_token"]) for event in events],
            [("match", "tôi", "tôi"), ("deletion", "đã", ""), ("match", "tôi", "tôi")],
        )
        self.assertEqual([event["ref_index"] for event in events], [0, 1, 2])
        self.assertEqual([event["hyp_index"] for event in events], [0, "", 1])
        assert_summary_reconciles(self, summaries[0])

    def test_event_text_and_bases_are_normalized_to_nfc(self) -> None:
        decomposed_ref = unicodedata.normalize("NFD", "má")
        decomposed_hyp = unicodedata.normalize("NFD", "ma")

        events, summaries = build_error_analysis(
            [prediction_row(ref=decomposed_ref, hyp=decomposed_hyp)]
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["ref"], "má")
        self.assertEqual(event["hyp"], "ma")
        self.assertEqual(event["ref_tone_base"], "ma")
        self.assertEqual(event["hyp_tone_base"], "ma")
        self.assertEqual(event["ref_plain_base"], "ma")
        self.assertEqual(event["hyp_plain_base"], "ma")
        self.assertEqual(event["tone_eligible"], "true")
        self.assertEqual(event["tone_error"], "true")
        self.assertEqual(event["diacritic_eligible"], "true")
        self.assertEqual(event["diacritic_error"], "false")
        self.assertEqual(event["short_word_deletion"], "false")
        assert_summary_reconciles(self, summaries[0])

    def test_run_metadata_is_never_mixed(self) -> None:
        ordinary = prediction_row(
            train_type="ordinary_lora",
            hyp="tôi đi",
            **{"lambda": "0"},
        )
        tone_005 = prediction_row(
            train_type="tone_aware_lora",
            hyp="tôi đã đi",
            **{"lambda": "0.05"},
        )
        tone_01 = prediction_row(
            train_type="tone_aware_lora",
            hyp="tôi đi",
            **{"lambda": "0.1"},
        )

        events, summaries = build_error_analysis([tone_01, ordinary, tone_005])

        expected_runs = {
            ("ordinary_lora", "0"),
            ("tone_aware_lora", "0.05"),
            ("tone_aware_lora", "0.1"),
        }
        self.assertEqual(len(summaries), 3)
        self.assertEqual(
            {(row["train_type"], row["lambda"]) for row in summaries},
            expected_runs,
        )
        for event in events:
            self.assertIn((event["train_type"], event["lambda"]), expected_runs)
        for summary in summaries:
            self.assertEqual(summary["n_utterances"], 1)
            assert_summary_reconciles(self, summary)

    def test_duplicate_key_across_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "pred_first.csv"
            second = root / "pred_second.csv"
            write_predictions(first, [prediction_row()])
            write_predictions(second, [prediction_row()])

            with self.assertRaisesRegex(ValueError, "duplicate prediction key"):
                load_prediction_rows([first, second])

    def test_cross_run_reference_and_condition_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "pred_first.csv"
            second = root / "pred_second.csv"
            write_predictions(first, [prediction_row()])
            write_predictions(
                second,
                [
                    prediction_row(
                        train_type="ordinary_lora",
                        hyp="tôi đi",
                        ref="reference khác",
                        **{"lambda": "0"},
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "inconsistent ref/snr/noise_type"):
                load_prediction_rows([first, second])

    def test_missing_csv_cell_is_reported_as_analysis_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pred_short.csv"
            source.write_text(",".join(CANONICAL_COLUMNS) + "\nutt-001,vivos\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing or extra CSV cells"):
                load_prediction_rows([source])

    def test_cli_is_deterministic_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pred_dir = root / "predictions" / "fixture"
            source = pred_dir / "pred_fixture.csv"
            rows = [
                prediction_row(utt_id="utt-002", ref="tôi đi", hyp="tôi sẽ đi"),
                prediction_row(utt_id="utt-001", ref="đã có", hyp="đã"),
            ]
            write_predictions(source, rows)
            source_before = source.read_bytes()
            output_dir = root / "analysis"
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(SCRIPT),
                "--pred-glob",
                str(root / "predictions" / "*" / "pred_*.csv"),
                "--out-dir",
                str(output_dir),
            ]

            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(source.read_bytes(), source_before)
            event_path = output_dir / "error_events.csv"
            summary_path = output_dir / "error_summary.csv"
            event_before = event_path.read_bytes()
            summary_before = summary_path.read_bytes()
            self.assertTrue((output_dir / "error_analysis.provenance.json").is_file())
            self.assertTrue((output_dir / "error_analysis.bundle.commit.json").is_file())

            resumed = subprocess.run(
                [*command, "--resume"], capture_output=True, text=True, check=False
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(event_path.read_bytes(), event_before)
            self.assertEqual(summary_path.read_bytes(), summary_before)

            event_header, event_rows = read_csv(event_path)
            summary_header, summary_rows = read_csv(summary_path)
            self.assertEqual(event_header, EXPECTED_EVENT_COLUMNS)
            self.assertEqual(summary_header, SUMMARY_COLUMNS)
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual([row["utt_id"] for row in event_rows[:2]], ["utt-001", "utt-001"])
            self.assertTrue(
                all(row["metric_version"] == METRIC_VERSION for row in event_rows)
            )
            self.assertTrue(all(row["ref"] and "hyp" in row for row in event_rows))
            self.assertTrue(
                all(
                    row[name] in {"true", "false"}
                    for row in event_rows
                    for name in (
                        "tone_eligible",
                        "diacritic_eligible",
                        "final_consonant_eligible",
                        "short_word_deletion",
                    )
                )
            )
            assert_summary_reconciles(self, summary_rows[0])

            refused = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("output already exists", refused.stderr)
            self.assertEqual(event_path.read_bytes(), event_before)
            self.assertEqual(summary_path.read_bytes(), summary_before)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertFalse((output_dir / ".error_events.csv.tmp").exists())
            self.assertFalse((output_dir / ".error_summary.csv.tmp").exists())

            rerun = subprocess.run(
                [*command, "--overwrite"], capture_output=True, text=True, check=False
            )
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual(event_path.read_bytes(), event_before)
            self.assertEqual(summary_path.read_bytes(), summary_before)
            self.assertEqual(source.read_bytes(), source_before)


if __name__ == "__main__":
    unittest.main()
