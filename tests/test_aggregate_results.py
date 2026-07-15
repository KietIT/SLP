from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from scripts.aggregate_results import (
    AggregationError,
    EXPECTED_METRIC_VERSION,
    RESULTS_BY_NOISE_TYPE_COLUMNS,
    RESULTS_BY_SNR_COLUMNS,
    aggregate_runs,
    build_parser,
    discover_inputs,
    load_benchmark_index,
    load_prediction_runs,
    run_aggregation,
)
from src.vitonesr.analysis import CANONICAL_PREDICTION_COLUMNS


BENCHMARK_COLUMNS = ["utt_id", "dataset", "snr", "noise_type", "transcript"]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def benchmark_rows(*, include_zero_db: bool = True) -> list[dict[str, str]]:
    rows = [
        {
            "utt_id": "u_clean",
            "dataset": "vivos",
            "snr": "clean",
            "noise_type": "clean",
            "transcript": "tôi đi",
        },
        {
            "utt_id": "u_20",
            "dataset": "vivos",
            "snr": "20",
            "noise_type": "music",
            "transcript": "xin chào",
        },
        {
            "utt_id": "u_10",
            "dataset": "vivos",
            "snr": "10",
            "noise_type": "noise",
            "transcript": "đã có",
        },
        {
            "utt_id": "u_5",
            "dataset": "vivos",
            "snr": "5",
            "noise_type": "speech",
            "transcript": "một người",
        },
    ]
    if include_zero_db:
        rows.append(
            {
                "utt_id": "u_0",
                "dataset": "vivos",
                "snr": "0",
                "noise_type": "music",
                "transcript": "và tôi",
            }
        )
    return rows


def prediction_rows(
    benchmark: list[dict[str, str]],
    *,
    model: str,
    model_size: str,
    train_type: str,
    lambda_value: str,
    seed: str = "42",
) -> list[dict[str, str]]:
    return [
        {
            "utt_id": row["utt_id"],
            "dataset": row["dataset"],
            "model": model,
            "model_size": model_size,
            "train_type": train_type,
            "lambda": lambda_value,
            "seed": seed,
            "snr": row["snr"],
            "noise_type": row["noise_type"],
            "ref": row["transcript"],
            "hyp": row["transcript"],
        }
        for row in benchmark
    ]


class AggregateResultsTest(unittest.TestCase):
    def test_discovery_uses_only_requested_prediction_pattern_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "predictions" / "a" / "pred_a.csv"
            second = root / "predictions" / "b" / "pred_b.csv"
            report = root / "predictions" / "a" / "normalization_report.csv"
            for path in (first, second, report):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n", encoding="utf-8")

            pattern = str(root / "predictions" / "*" / "pred_*.csv")
            discovered = discover_inputs([str(first)], [pattern])

            self.assertEqual([path.resolve() for path in discovered], [first.resolve(), second.resolve()])
            self.assertNotIn(report.resolve(), [path.resolve() for path in discovered])
            with self.assertRaisesRegex(AggregationError, "matched no files"):
                discover_inputs([], [str(root / "missing" / "pred_*.csv")])

    def test_parser_accepts_repeated_inputs_and_globs(self) -> None:
        args = build_parser().parse_args(
            [
                "--input",
                "a.csv",
                "--input",
                "b.csv",
                "--input-glob",
                "x/pred_*.csv",
                "--input_glob",
                "y/pred_*.csv",
                "--metric-version",
                "aligned_v1",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.input, ["a.csv", "b.csv"])
        self.assertEqual(args.input_glob, ["x/pred_*.csv", "y/pred_*.csv"])
        self.assertEqual(args.metric_version, EXPECTED_METRIC_VERSION)
        self.assertEqual(args.output_dir, "out")

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["--metric-version", "simple_v1", "--output-dir", "out"]
            )

    def test_two_runs_generate_exact_schemas_groups_counts_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = benchmark_rows()
            benchmark_path = root / "benchmark.csv"
            write_csv(benchmark_path, BENCHMARK_COLUMNS, benchmark)

            zero_path = root / "predictions" / "zero_shot" / "pred_zero.csv"
            tone_path = root / "predictions" / "phat" / "pred_tone.csv"
            write_csv(
                zero_path,
                list(CANONICAL_PREDICTION_COLUMNS),
                prediction_rows(
                    benchmark,
                    model="whisper",
                    model_size="tiny",
                    train_type="zero_shot",
                    lambda_value="",
                ),
            )
            write_csv(
                tone_path,
                list(CANONICAL_PREDICTION_COLUMNS),
                prediction_rows(
                    benchmark,
                    model="phowhisper",
                    model_size="base",
                    train_type="tone_aware_lora",
                    lambda_value="0.05",
                ),
            )

            output_dir = root / "analysis"
            result = run_aggregation(
                [zero_path, tone_path],
                benchmark_path,
                output_dir,
            )

            self.assertEqual(result["input_files"], 2)
            self.assertEqual(result["prediction_rows"], 10)
            self.assertEqual(result["runs"], 2)
            self.assertEqual(result["results_by_snr_rows"], 14)
            self.assertEqual(result["results_by_noise_type_rows"], 8)
            self.assertEqual(result["metric_version"], EXPECTED_METRIC_VERSION)

            snr_columns, snr_rows = read_csv(output_dir / "results_by_snr.csv")
            noise_columns, noise_rows = read_csv(output_dir / "results_by_noise_type.csv")
            self.assertEqual(snr_columns, RESULTS_BY_SNR_COLUMNS)
            self.assertEqual(noise_columns, RESULTS_BY_NOISE_TYPE_COLUMNS)
            self.assertEqual(len(snr_rows), 14)
            self.assertEqual(len(noise_rows), 8)

            for row in [*snr_rows, *noise_rows]:
                self.assertEqual(row["dataset"], "vivos")
                self.assertEqual(row["seed"], "42")
                self.assertEqual(row["metric_version"], EXPECTED_METRIC_VERSION)
                for metric in ("wer", "cer", "ter", "der", "fcer", "swdr"):
                    self.assertEqual(float(row[metric]), 0.0)

            zero_snr = [row for row in snr_rows if row["train_type"] == "zero_shot"]
            self.assertEqual([row["snr"] for row in zero_snr], [
                "clean",
                "20",
                "10",
                "5",
                "0",
                "noisy_all",
                "all",
            ])
            self.assertEqual(
                {row["snr"]: int(row["n"]) for row in zero_snr},
                {"clean": 1, "20": 1, "10": 1, "5": 1, "0": 1, "noisy_all": 4, "all": 5},
            )
            zero_noise = [row for row in noise_rows if row["train_type"] == "zero_shot"]
            self.assertEqual(
                [row["noise_type"] for row in zero_noise],
                ["clean", "music", "noise", "speech"],
            )
            self.assertEqual(
                {row["noise_type"]: int(row["n"]) for row in zero_noise},
                {"clean": 1, "music": 2, "noise": 1, "speech": 1},
            )

            tone_all = next(
                row
                for row in snr_rows
                if row["train_type"] == "tone_aware_lora" and row["snr"] == "all"
            )
            self.assertEqual(tone_all["model"], "phowhisper")
            self.assertEqual(tone_all["model_size"], "base")
            self.assertEqual(tone_all["lambda"], "0.05")

    def test_benchmark_mismatch_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = benchmark_rows()
            benchmark_path = root / "benchmark.csv"
            write_csv(benchmark_path, BENCHMARK_COLUMNS, benchmark)
            rows = prediction_rows(
                benchmark,
                model="whisper",
                model_size="base",
                train_type="zero_shot",
                lambda_value="",
            )
            rows[0]["ref"] = "sai tham chiếu"
            prediction_path = root / "pred_bad.csv"
            write_csv(prediction_path, list(CANONICAL_PREDICTION_COLUMNS), rows)
            output_dir = root / "analysis"

            with self.assertRaisesRegex(AggregationError, "ref does not match benchmark"):
                run_aggregation([prediction_path], benchmark_path, output_dir)
            self.assertFalse(output_dir.exists())

    def test_duplicate_run_across_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = benchmark_rows()
            benchmark_path = root / "benchmark.csv"
            write_csv(benchmark_path, BENCHMARK_COLUMNS, benchmark)
            rows = prediction_rows(
                benchmark,
                model="whisper",
                model_size="small",
                train_type="zero_shot",
                lambda_value="",
            )
            first = root / "pred_first.csv"
            second = root / "pred_second.csv"
            write_csv(first, list(CANONICAL_PREDICTION_COLUMNS), rows)
            write_csv(second, list(CANONICAL_PREDICTION_COLUMNS), rows)

            with self.assertRaisesRegex(AggregationError, "duplicate prediction run"):
                load_prediction_runs(
                    [first, second],
                    load_benchmark_index(benchmark_path),
                )

    def test_missing_required_snr_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = benchmark_rows(include_zero_db=False)
            benchmark_path = root / "benchmark.csv"
            prediction_path = root / "pred.csv"
            write_csv(benchmark_path, BENCHMARK_COLUMNS, benchmark)
            write_csv(
                prediction_path,
                list(CANONICAL_PREDICTION_COLUMNS),
                prediction_rows(
                    benchmark,
                    model="whisper",
                    model_size="tiny",
                    train_type="zero_shot",
                    lambda_value="",
                ),
            )
            runs = load_prediction_runs(
                [prediction_path],
                load_benchmark_index(benchmark_path),
            )
            with self.assertRaisesRegex(AggregationError, "required SNR groups"):
                aggregate_runs(runs)

    def test_no_overwrite_preserves_both_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = benchmark_rows()
            benchmark_path = root / "benchmark.csv"
            prediction_path = root / "pred.csv"
            output_dir = root / "analysis"
            write_csv(benchmark_path, BENCHMARK_COLUMNS, benchmark)
            write_csv(
                prediction_path,
                list(CANONICAL_PREDICTION_COLUMNS),
                prediction_rows(
                    benchmark,
                    model="phowhisper",
                    model_size="base",
                    train_type="ordinary_lora",
                    lambda_value="0",
                ),
            )

            run_aggregation([prediction_path], benchmark_path, output_dir)
            snr_path = output_dir / "results_by_snr.csv"
            noise_path = output_dir / "results_by_noise_type.csv"
            before = (snr_path.read_bytes(), noise_path.read_bytes())

            with self.assertRaisesRegex(AggregationError, "output already exists"):
                run_aggregation([prediction_path], benchmark_path, output_dir)
            self.assertEqual(before, (snr_path.read_bytes(), noise_path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
