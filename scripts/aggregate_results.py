from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.analysis import (  # noqa: E402
    CANONICAL_PREDICTION_COLUMNS,
    METRIC_VERSION,
    PredictionValidationError,
    compute_aligned_metrics,
    load_prediction_csv,
    validate_prediction_rows,
)


DEFAULT_INPUT_GLOB = "outputs/predictions/*/pred_*.csv"
DEFAULT_BENCHMARK_MANIFEST = "outputs/benchmark/benchmark_manifest.csv"
EXPECTED_METRIC_VERSION = "aligned_v1"

RUN_COLUMNS = ["dataset", "model", "model_size", "train_type", "lambda", "seed"]
METRIC_COLUMNS = ["wer", "cer", "ter", "der", "fcer", "swdr"]
SNR_LEAF_ORDER = ["clean", "20", "10", "5", "0"]
SNR_OUTPUT_ORDER = [*SNR_LEAF_ORDER, "noisy_all", "all"]
KNOWN_NOISE_ORDER = ["clean", "music", "noise", "speech", "babble"]

RESULTS_BY_SNR_COLUMNS = [
    *RUN_COLUMNS,
    "snr",
    "n",
    *METRIC_COLUMNS,
    "metric_version",
]
RESULTS_BY_NOISE_TYPE_COLUMNS = [
    *RUN_COLUMNS,
    "noise_type",
    "n",
    *METRIC_COLUMNS,
    "metric_version",
]


class AggregationError(ValueError):
    """Raised when prediction inputs cannot be aggregated safely."""


@dataclass(frozen=True)
class BenchmarkIndex:
    path: Path
    rows_by_id: dict[str, dict[str, str]]


@dataclass
class PredictionRun:
    source_path: Path
    metadata: dict[str, str]
    rows: list[dict[str, str]]

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(self.metadata[column] for column in RUN_COLUMNS)


def discover_inputs(
    explicit: Sequence[str] | None,
    patterns: Sequence[str] | None,
    *,
    default_pattern: str = DEFAULT_INPUT_GLOB,
) -> list[Path]:
    explicit_values = list(explicit or [])
    pattern_values = list(patterns or [])
    if not explicit_values and not pattern_values:
        pattern_values = [default_pattern]

    candidates = [Path(value) for value in explicit_values]
    for pattern in pattern_values:
        matches = [Path(value) for value in glob.glob(pattern, recursive=True)]
        if not matches:
            raise AggregationError(f"input glob matched no files: {pattern}")
        candidates.extend(matches)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(candidates, key=lambda value: value.as_posix().lower()):
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise AggregationError(f"prediction input is not a file: {path}")
        if path.suffix.lower() != ".csv":
            raise AggregationError(f"prediction input must be CSV: {path}")
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise AggregationError("no prediction CSV files were discovered")
    return unique


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise AggregationError(f"CSV file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AggregationError(f"{path}: CSV has no header")
            columns = list(reader.fieldnames)
            if len(columns) != len(set(columns)):
                raise AggregationError(f"{path}: duplicate CSV header columns")
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise AggregationError(
                        f"{path}: row {row_number} has more cells than the header"
                    )
                missing = [name for name, value in row.items() if value is None]
                if missing:
                    raise AggregationError(
                        f"{path}: row {row_number} has missing cells for {missing}"
                    )
                rows.append(dict(row))
    except UnicodeDecodeError as exc:
        raise AggregationError(f"{path}: CSV must be UTF-8") from exc
    if not rows:
        raise AggregationError(f"{path}: CSV is empty")
    return columns, rows


def _normalize_snr(value: object) -> str:
    text = str(value).strip()
    if text.lower() == "clean":
        return "clean"
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise AggregationError(f"SNR must be 'clean' or numeric, found {text!r}") from exc
    if not number.is_finite():
        raise AggregationError(f"SNR must be finite, found {text!r}")
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"", "-0"} else normalized


def load_benchmark_index(path: str | Path) -> BenchmarkIndex:
    benchmark_path = Path(path)
    columns, rows = _read_csv(benchmark_path)
    required = {"utt_id", "dataset", "snr", "noise_type", "transcript"}
    missing = sorted(required.difference(columns))
    if missing:
        raise AggregationError(f"{benchmark_path}: benchmark is missing columns {missing}")

    rows_by_id: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        utt_id = row["utt_id"]
        if not utt_id:
            raise AggregationError(f"{benchmark_path}: row {row_number} has blank utt_id")
        if utt_id in rows_by_id:
            raise AggregationError(
                f"{benchmark_path}: duplicate benchmark utt_id {utt_id!r} at row {row_number}"
            )
        rows_by_id[utt_id] = row
    return BenchmarkIndex(path=benchmark_path, rows_by_id=rows_by_id)


def validate_against_benchmark(
    source_path: Path,
    rows: Sequence[dict[str, str]],
    benchmark: BenchmarkIndex,
) -> None:
    prediction_ids = {row["utt_id"] for row in rows}
    benchmark_ids = set(benchmark.rows_by_id)
    missing = sorted(benchmark_ids.difference(prediction_ids))
    unexpected = sorted(prediction_ids.difference(benchmark_ids))
    if missing or unexpected:
        raise AggregationError(
            f"{source_path}: utt_id set does not match {benchmark.path}; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    comparisons = {
        "dataset": "dataset",
        "noise_type": "noise_type",
        "ref": "transcript",
    }
    for row_number, prediction in enumerate(rows, start=2):
        benchmark_row = benchmark.rows_by_id[prediction["utt_id"]]
        for prediction_column, benchmark_column in comparisons.items():
            if prediction[prediction_column] != benchmark_row[benchmark_column]:
                raise AggregationError(
                    f"{source_path}: row {row_number} {prediction_column} does not match "
                    f"benchmark for utt_id={prediction['utt_id']!r}"
                )
        if _normalize_snr(prediction["snr"]) != _normalize_snr(benchmark_row["snr"]):
            raise AggregationError(
                f"{source_path}: row {row_number} snr does not match benchmark "
                f"for utt_id={prediction['utt_id']!r}"
            )


def _run_metadata(source_path: Path, rows: Sequence[dict[str, str]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for column in RUN_COLUMNS:
        values = {row[column] for row in rows}
        if len(values) != 1:
            raise AggregationError(
                f"{source_path}: run metadata column {column!r} must be constant; "
                f"found {sorted(values)[:5]}"
            )
        metadata[column] = next(iter(values))
    return metadata


def load_prediction_runs(
    input_paths: Sequence[str | Path],
    benchmark: BenchmarkIndex,
) -> list[PredictionRun]:
    runs: list[PredictionRun] = []
    seen_sample_keys: dict[tuple[str, ...], Path] = {}
    seen_run_keys: dict[tuple[str, ...], Path] = {}

    for raw_path in input_paths:
        path = Path(raw_path)
        try:
            loaded_rows = load_prediction_csv(path)
            rows = validate_prediction_rows(loaded_rows, source=path)
        except PredictionValidationError as exc:
            raise AggregationError(f"{path}: {exc}") from exc
        if not rows:
            raise AggregationError(f"{path}: prediction CSV is empty")

        metadata = _run_metadata(path, rows)
        run_key = tuple(metadata[column] for column in RUN_COLUMNS)
        if run_key in seen_run_keys:
            raise AggregationError(
                f"duplicate prediction run across {seen_run_keys[run_key]} and {path}: {run_key}"
            )
        seen_run_keys[run_key] = path

        validate_against_benchmark(path, rows, benchmark)
        for row in rows:
            sample_key = (
                row["dataset"],
                row["utt_id"],
                row["model"],
                row["model_size"],
                row["train_type"],
                row["lambda"],
                row["seed"],
            )
            if sample_key in seen_sample_keys:
                raise AggregationError(
                    f"duplicate combined prediction key across "
                    f"{seen_sample_keys[sample_key]} and {path}: {sample_key}"
                )
            seen_sample_keys[sample_key] = path
        runs.append(PredictionRun(source_path=path, metadata=metadata, rows=rows))

    return sorted(runs, key=lambda run: run.key)


def _metric_row(
    run: PredictionRun,
    group_column: str,
    group_value: str,
    rows: Sequence[dict[str, str]],
) -> dict[str, object]:
    if not rows:
        raise AggregationError(
            f"{run.source_path}: group {group_column}={group_value!r} is empty"
        )
    metrics = compute_aligned_metrics(
        [row["ref"] for row in rows],
        [row["hyp"] for row in rows],
    )
    metric_version = str(metrics.get("metric_version", ""))
    if METRIC_VERSION != EXPECTED_METRIC_VERSION or metric_version != EXPECTED_METRIC_VERSION:
        raise AggregationError(
            f"expected metric_version={EXPECTED_METRIC_VERSION!r}, "
            f"analysis module returned {metric_version!r}"
        )

    metric_values: dict[str, float] = {}
    for name in METRIC_COLUMNS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise AggregationError(f"aligned metric {name!r} is unavailable") from exc
        if not math.isfinite(value) or value < 0:
            raise AggregationError(f"aligned metric {name!r} must be finite and non-negative")
        metric_values[name] = value

    return {
        **run.metadata,
        group_column: group_value,
        "n": len(rows),
        **metric_values,
        "metric_version": metric_version,
    }


def _noise_sort_key(value: str) -> tuple[int, int | str]:
    if value in KNOWN_NOISE_ORDER:
        return (0, KNOWN_NOISE_ORDER.index(value))
    return (1, value)


def aggregate_runs(
    runs: Sequence[PredictionRun],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not runs:
        raise AggregationError("no prediction runs to aggregate")

    observed_noise_types = sorted(
        {row["noise_type"] for run in runs for row in run.rows},
        key=_noise_sort_key,
    )
    if not observed_noise_types:
        raise AggregationError("prediction inputs contain no noise_type values")

    snr_results: list[dict[str, object]] = []
    noise_results: list[dict[str, object]] = []
    required_snrs = set(SNR_LEAF_ORDER)

    for run in runs:
        by_snr: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_noise: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in run.rows:
            by_snr[_normalize_snr(row["snr"])].append(row)
            by_noise[row["noise_type"]].append(row)

        observed_snrs = set(by_snr)
        if observed_snrs != required_snrs:
            raise AggregationError(
                f"{run.source_path}: required SNR groups are {SNR_LEAF_ORDER}; "
                f"found {sorted(observed_snrs)}"
            )

        for snr in SNR_LEAF_ORDER:
            snr_results.append(_metric_row(run, "snr", snr, by_snr[snr]))
        noisy_rows = [row for row in run.rows if _normalize_snr(row["snr"]) != "clean"]
        snr_results.append(_metric_row(run, "snr", "noisy_all", noisy_rows))
        snr_results.append(_metric_row(run, "snr", "all", run.rows))

        run_noise_types = set(by_noise)
        if run_noise_types != set(observed_noise_types):
            raise AggregationError(
                f"{run.source_path}: noise_type groups differ across runs; "
                f"expected {observed_noise_types}, found {sorted(run_noise_types)}"
            )
        for noise_type in observed_noise_types:
            noise_results.append(
                _metric_row(run, "noise_type", noise_type, by_noise[noise_type])
            )

    return snr_results, noise_results


def _write_csv_temp(
    destination: Path,
    rows: Iterable[dict[str, object]],
    columns: Sequence[str],
) -> Path:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return temporary


def _result_destinations(output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    return directory / "results_by_snr.csv", directory / "results_by_noise_type.csv"


def ensure_outputs_available(output_dir: str | Path, *, overwrite: bool) -> None:
    destinations = _result_destinations(output_dir)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise AggregationError(
            "output already exists; choose another --output-dir or use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def write_results(
    output_dir: str | Path,
    snr_rows: Sequence[dict[str, object]],
    noise_rows: Sequence[dict[str, object]],
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    snr_path, noise_path = _result_destinations(directory)
    destinations = [snr_path, noise_path]
    ensure_outputs_available(directory, overwrite=overwrite)

    directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        temporary_paths.append(_write_csv_temp(snr_path, snr_rows, RESULTS_BY_SNR_COLUMNS))
        temporary_paths.append(
            _write_csv_temp(noise_path, noise_rows, RESULTS_BY_NOISE_TYPE_COLUMNS)
        )
        if not overwrite:
            raced = [path for path in destinations if path.exists()]
            if raced:
                raise AggregationError(
                    "output appeared while aggregation was running: "
                    + ", ".join(str(path) for path in raced)
                )
        temporary_paths[0].replace(snr_path)
        temporary_paths[1].replace(noise_path)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
    return snr_path, noise_path


def run_aggregation(
    input_paths: Sequence[str | Path],
    benchmark_manifest: str | Path,
    output_dir: str | Path,
    *,
    metric_version: str = EXPECTED_METRIC_VERSION,
    overwrite: bool = False,
) -> dict[str, object]:
    if metric_version != EXPECTED_METRIC_VERSION:
        raise AggregationError(
            f"unsupported metric_version={metric_version!r}; "
            f"expected {EXPECTED_METRIC_VERSION!r}"
        )
    ensure_outputs_available(output_dir, overwrite=overwrite)
    benchmark = load_benchmark_index(benchmark_manifest)
    runs = load_prediction_runs(input_paths, benchmark)
    snr_rows, noise_rows = aggregate_runs(runs)
    snr_path, noise_path = write_results(
        output_dir,
        snr_rows,
        noise_rows,
        overwrite=overwrite,
    )
    return {
        "input_files": len(input_paths),
        "prediction_rows": sum(len(run.rows) for run in runs),
        "runs": len(runs),
        "results_by_snr_rows": len(snr_rows),
        "results_by_noise_type_rows": len(noise_rows),
        "results_by_snr": str(snr_path),
        "results_by_noise_type": str(noise_path),
        "metric_version": EXPECTED_METRIC_VERSION,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate canonical ASR predictions by SNR and noise type."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Canonical prediction CSV; repeatable.",
    )
    parser.add_argument(
        "--input_glob",
        "--input-glob",
        action="append",
        default=[],
        help=f"Prediction glob; repeatable. Default: {DEFAULT_INPUT_GLOB}",
    )
    parser.add_argument(
        "--benchmark_manifest",
        "--benchmark-manifest",
        default=DEFAULT_BENCHMARK_MANIFEST,
        help="Benchmark CSV used for exact ID/reference/condition validation.",
    )
    parser.add_argument(
        "--metric_version",
        "--metric-version",
        choices=[EXPECTED_METRIC_VERSION],
        default=EXPECTED_METRIC_VERSION,
        help=f"Metric implementation version (only {EXPECTED_METRIC_VERSION} is supported).",
    )
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        inputs = discover_inputs(args.input, args.input_glob)
        result = run_aggregation(
            inputs,
            args.benchmark_manifest,
            args.output_dir,
            metric_version=args.metric_version,
            overwrite=args.overwrite,
        )
    except (AggregationError, PredictionValidationError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        f"PASS inputs={result['input_files']} runs={result['runs']} "
        f"predictions={result['prediction_rows']} metric_version={result['metric_version']}"
    )
    print(
        f"wrote {result['results_by_snr']} "
        f"({result['results_by_snr_rows']} rows)"
    )
    print(
        f"wrote {result['results_by_noise_type']} "
        f"({result['results_by_noise_type_rows']} rows)"
    )


if __name__ == "__main__":
    main()
