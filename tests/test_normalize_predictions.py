from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.normalize_predictions import (
    CANONICAL_COLUMNS,
    LEGACY_MIDTERM_COLUMNS,
    LEGACY_ZERO_SHOT_COLUMNS,
    MetadataOverrides,
    NormalizationError,
    load_benchmark_index,
    prepare_prediction_file,
    validate_against_benchmark,
    write_and_verify,
)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class NormalizePredictionsTest(unittest.TestCase):
    def test_zero_shot_conversion_preserves_all_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pred.csv"
            raw = {
                "utt_id": "vivos_001_clean",
                "dataset": "vivos",
                "model": "whisper",
                "model_size": "tiny",
                "snr": "clean",
                "noise_type": "clean",
                "ref": "Tôi là Trung.",
                "hyp": "tôi là trung",
            }
            write_csv(source, LEGACY_ZERO_SHOT_COLUMNS, [raw])

            prepared = prepare_prediction_file(
                source,
                MetadataOverrides(train_type="zero_shot", seed=42),
            )

            self.assertEqual(prepared.source_schema, "legacy_zero_shot_8col")
            self.assertTrue(prepared.legacy_fields_preserved)
            self.assertEqual(prepared.rows[0]["train_type"], "zero_shot")
            self.assertEqual(prepared.rows[0]["lambda"], "")
            self.assertEqual(prepared.rows[0]["seed"], "42")
            for column in LEGACY_ZERO_SHOT_COLUMNS:
                self.assertEqual(prepared.rows[0][column], raw[column])

            output = root / "normalized.csv"
            write_and_verify(prepared, output)
            with output.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle).fieldnames or []), CANONICAL_COLUMNS)

    def test_midterm_mapping_requires_explicit_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "midterm.csv"
            write_csv(
                source,
                LEGACY_MIDTERM_COLUMNS,
                [
                    {
                        "utt_id": "vivos_001",
                        "audio": "audio.wav",
                        "text": "xin chào",
                        "prediction": "xin chào",
                        "snr": "clean",
                        "noise_type": "",
                        "dataset": "vivos",
                    }
                ],
            )

            with self.assertRaisesRegex(NormalizationError, "empty required fields"):
                prepare_prediction_file(
                    source,
                    MetadataOverrides(train_type="tone_lora", lambda_value="0.1", seed=42),
                )

            prepared = prepare_prediction_file(
                source,
                MetadataOverrides(
                    model="phowhisper",
                    model_size="base",
                    train_type="tone_lora",
                    lambda_value="0.10",
                    seed=42,
                ),
            )
            self.assertEqual(prepared.rows[0]["ref"], "xin chào")
            self.assertEqual(prepared.rows[0]["hyp"], "xin chào")
            self.assertEqual(prepared.rows[0]["noise_type"], "clean")
            self.assertEqual(prepared.rows[0]["lambda"], "0.1")

    def test_duplicate_utt_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "duplicates.csv"
            row = {
                "utt_id": "duplicate",
                "dataset": "vivos",
                "model": "whisper",
                "model_size": "tiny",
                "snr": "clean",
                "noise_type": "clean",
                "ref": "một",
                "hyp": "",
            }
            write_csv(source, LEGACY_ZERO_SHOT_COLUMNS, [row, row])
            with self.assertRaisesRegex(NormalizationError, "duplicate utt_id"):
                prepare_prediction_file(
                    source,
                    MetadataOverrides(train_type="zero_shot", seed=42),
                )

    def test_zero_shot_rejects_nonempty_lambda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad_lambda.csv"
            write_csv(
                source,
                LEGACY_ZERO_SHOT_COLUMNS,
                [
                    {
                        "utt_id": "sample",
                        "dataset": "vivos",
                        "model": "whisper",
                        "model_size": "base",
                        "snr": "0",
                        "noise_type": "noise",
                        "ref": "một",
                        "hyp": "một",
                    }
                ],
            )
            with self.assertRaisesRegex(NormalizationError, "zero_shot must use an empty lambda"):
                prepare_prediction_file(
                    source,
                    MetadataOverrides(train_type="zero_shot", lambda_value="0.1", seed=42),
                )

    def test_hypothesis_may_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "empty_hyp.csv"
            write_csv(
                source,
                LEGACY_ZERO_SHOT_COLUMNS,
                [
                    {
                        "utt_id": "sample",
                        "dataset": "vivos",
                        "model": "whisper",
                        "model_size": "small",
                        "snr": "0",
                        "noise_type": "music",
                        "ref": "đã có",
                        "hyp": "",
                    }
                ],
            )
            prepared = prepare_prediction_file(
                source,
                MetadataOverrides(train_type="zero_shot", seed=42),
            )
            self.assertEqual(prepared.blank_hypotheses, 1)

    def test_benchmark_validation_checks_reference_and_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pred.csv"
            benchmark_path = root / "benchmark.csv"
            prediction_row = {
                "utt_id": "sample_snr5",
                "dataset": "vivos",
                "model": "phowhisper",
                "model_size": "base",
                "snr": "5",
                "noise_type": "music",
                "ref": "giữ nguyên văn bản",
                "hyp": " giữ nguyên khoảng trắng ",
            }
            write_csv(source, LEGACY_ZERO_SHOT_COLUMNS, [prediction_row])
            write_csv(
                benchmark_path,
                ["utt_id", "dataset", "snr", "noise_type", "transcript"],
                [
                    {
                        "utt_id": "sample_snr5",
                        "dataset": "vivos",
                        "snr": "5",
                        "noise_type": "music",
                        "transcript": "giữ nguyên văn bản",
                    }
                ],
            )
            prepared = prepare_prediction_file(
                source,
                MetadataOverrides(train_type="zero_shot", seed=42),
            )
            validate_against_benchmark(prepared, load_benchmark_index(benchmark_path))
            self.assertEqual(prepared.rows[0]["hyp"], " giữ nguyên khoảng trắng ")

    def test_canonical_metadata_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "canonical.csv"
            write_csv(
                source,
                CANONICAL_COLUMNS,
                [
                    {
                        "utt_id": "sample",
                        "dataset": "vivos",
                        "model": "phowhisper",
                        "model_size": "base",
                        "train_type": "tone_lora",
                        "lambda": "0.1",
                        "seed": "7",
                        "snr": "clean",
                        "noise_type": "clean",
                        "ref": "một",
                        "hyp": "một",
                    }
                ],
            )
            with self.assertRaisesRegex(NormalizationError, "conflicts with input seed"):
                prepare_prediction_file(source, MetadataOverrides(seed=42))

    def test_cli_refuses_report_path_that_is_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.csv"
            raw = {
                "utt_id": "sample",
                "dataset": "vivos",
                "model": "whisper",
                "model_size": "tiny",
                "snr": "clean",
                "noise_type": "clean",
                "ref": "một",
                "hyp": "một",
            }
            write_csv(source, LEGACY_ZERO_SHOT_COLUMNS, [raw])
            before = source.read_bytes()
            script = Path(__file__).resolve().parents[1] / "scripts" / "normalize_predictions.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(source),
                    "--output_dir",
                    str(root / "normalized"),
                    "--train_type",
                    "zero_shot",
                    "--seed",
                    "42",
                    "--report",
                    str(source),
                    "--overwrite",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to overwrite an input or benchmark", result.stderr)
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
