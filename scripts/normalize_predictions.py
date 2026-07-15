from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence


CANONICAL_COLUMNS = [
    "utt_id",
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "snr",
    "noise_type",
    "ref",
    "hyp",
]

LEGACY_ZERO_SHOT_COLUMNS = [
    "utt_id",
    "dataset",
    "model",
    "model_size",
    "snr",
    "noise_type",
    "ref",
    "hyp",
]

LEGACY_MIDTERM_COLUMNS = [
    "utt_id",
    "audio",
    "text",
    "prediction",
    "snr",
    "noise_type",
    "dataset",
]

REPORT_COLUMNS = [
    "source_path",
    "source_sha256",
    "output_path",
    "output_sha256",
    "source_schema",
    "schema_version",
    "rows",
    "blank_hypotheses",
    "legacy_fields_preserved",
    "train_type",
    "lambda",
    "seed",
    "metadata_origin",
    "benchmark_manifest",
    "benchmark_sha256",
    "status",
]

CONSISTENT_RUN_COLUMNS = ["dataset", "model", "model_size", "train_type", "lambda", "seed"]
TRAIN_TYPE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class NormalizationError(ValueError):
    """Raised when a prediction file cannot be normalized safely."""


@dataclass(frozen=True)
class MetadataOverrides:
    dataset: str | None = None
    model: str | None = None
    model_size: str | None = None
    train_type: str | None = None
    lambda_value: str | None = None
    seed: int | None = None


@dataclass
class PreparedPrediction:
    source_path: Path
    source_schema: str
    rows: list[dict[str, str]]
    blank_hypotheses: int
    legacy_fields_preserved: bool


@dataclass(frozen=True)
class BenchmarkIndex:
    path: Path
    sha256: str
    rows_by_id: dict[str, dict[str, str]]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise NormalizationError(f"{path}: CSV has no header")
            columns = list(reader.fieldnames)
            if len(columns) != len(set(columns)):
                raise NormalizationError(f"{path}: duplicate column names in header: {columns}")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise NormalizationError(
                        f"{path}: row {row_number} has more cells than the header"
                    )
                missing_cells = [name for name, value in row.items() if value is None]
                if missing_cells:
                    raise NormalizationError(
                        f"{path}: row {row_number} has missing cells for columns {missing_cells}"
                    )
                rows.append(dict(row))
    except UnicodeDecodeError as exc:
        raise NormalizationError(f"{path}: file must be UTF-8 CSV") from exc
    if not rows:
        raise NormalizationError(f"{path}: prediction CSV is empty")
    return columns, rows


def detect_schema(path: Path, columns: Sequence[str]) -> str:
    if list(columns) == CANONICAL_COLUMNS:
        return "canonical_v1"
    if list(columns) == LEGACY_ZERO_SHOT_COLUMNS:
        return "legacy_zero_shot_8col"
    if list(columns) == LEGACY_MIDTERM_COLUMNS:
        return "legacy_midterm_7col"
    raise NormalizationError(
        f"{path}: unsupported columns {list(columns)}. Expected one of: "
        f"{CANONICAL_COLUMNS}, {LEGACY_ZERO_SHOT_COLUMNS}, {LEGACY_MIDTERM_COLUMNS}"
    )


def _clean_override(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned != "" else None


def _canonical_decimal(value: object, *, field: str, allow_empty: bool) -> str:
    text = "" if value is None else str(value).strip()
    if text == "":
        if allow_empty:
            return ""
        raise NormalizationError(f"{field} must not be empty")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise NormalizationError(f"{field} must be numeric, found {text!r}") from exc
    if not number.is_finite() or number < 0:
        raise NormalizationError(f"{field} must be a finite non-negative number, found {text!r}")
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def _value(row: dict[str, str], column: str, override: str | None) -> str:
    existing = "" if row.get(column) is None else str(row.get(column, ""))
    if override is None:
        return existing
    if existing.strip() != "" and existing != override:
        raise NormalizationError(
            f"CLI value for {column}={override!r} conflicts with input value {existing!r}"
        )
    return override


def _normalize_row(
    row: dict[str, str],
    source_schema: str,
    overrides: MetadataOverrides,
) -> dict[str, str]:
    dataset_override = _clean_override(overrides.dataset)
    model_override = _clean_override(overrides.model)
    model_size_override = _clean_override(overrides.model_size)
    train_type_override = _clean_override(overrides.train_type)

    if source_schema == "legacy_zero_shot_8col":
        normalized = {
            "utt_id": row["utt_id"],
            "dataset": _value(row, "dataset", dataset_override),
            "model": _value(row, "model", model_override),
            "model_size": _value(row, "model_size", model_size_override),
            "train_type": train_type_override or "",
            "lambda": overrides.lambda_value or "",
            "seed": "" if overrides.seed is None else str(overrides.seed),
            "snr": row["snr"],
            "noise_type": row["noise_type"],
            "ref": row["ref"],
            "hyp": row["hyp"],
        }
    elif source_schema == "legacy_midterm_7col":
        snr = row["snr"]
        noise_type = row["noise_type"]
        if str(snr).strip().lower() == "clean" and str(noise_type).strip() == "":
            noise_type = "clean"
        normalized = {
            "utt_id": row["utt_id"],
            "dataset": _value(row, "dataset", dataset_override),
            "model": model_override or "",
            "model_size": model_size_override or "",
            "train_type": train_type_override or "",
            "lambda": overrides.lambda_value or "",
            "seed": "" if overrides.seed is None else str(overrides.seed),
            "snr": snr,
            "noise_type": noise_type,
            "ref": row["text"],
            "hyp": row["prediction"],
        }
    else:
        normalized = {column: row.get(column, "") for column in CANONICAL_COLUMNS}
        for column, override in (
            ("dataset", dataset_override),
            ("model", model_override),
            ("model_size", model_size_override),
            ("train_type", train_type_override),
        ):
            if override is not None:
                normalized[column] = _value(normalized, column, override)
        if overrides.lambda_value is not None:
            current_lambda = _canonical_decimal(
                normalized.get("lambda", ""), field="lambda", allow_empty=True
            )
            override_lambda = _canonical_decimal(
                overrides.lambda_value, field="lambda", allow_empty=True
            )
            if current_lambda != override_lambda:
                raise NormalizationError(
                    f"CLI lambda={override_lambda!r} conflicts with input lambda={current_lambda!r}"
                )
        if overrides.seed is not None:
            current_seed = str(normalized.get("seed", "")).strip()
            if current_seed != str(overrides.seed):
                raise NormalizationError(
                    f"CLI seed={overrides.seed!r} conflicts with input seed={current_seed!r}"
                )

    normalized["lambda"] = _canonical_decimal(
        normalized.get("lambda", ""), field="lambda", allow_empty=True
    )
    seed_text = str(normalized.get("seed", "")).strip()
    try:
        seed_value = int(seed_text)
    except ValueError as exc:
        raise NormalizationError(f"seed must be an integer, found {seed_text!r}") from exc
    if seed_value < 0:
        raise NormalizationError(f"seed must be non-negative, found {seed_value}")
    normalized["seed"] = str(seed_value)
    return normalized


def _validate_row(path: Path, row: dict[str, str], row_number: int) -> None:
    required_nonempty = [
        "utt_id",
        "dataset",
        "model",
        "model_size",
        "train_type",
        "seed",
        "snr",
        "noise_type",
        "ref",
    ]
    missing = [name for name in required_nonempty if str(row.get(name, "")).strip() == ""]
    if missing:
        raise NormalizationError(f"{path}: row {row_number} has empty required fields: {missing}")

    train_type = row["train_type"]
    if train_type != train_type.strip() or not TRAIN_TYPE_RE.fullmatch(train_type):
        raise NormalizationError(
            f"{path}: row {row_number} train_type must be lowercase snake_case, found {train_type!r}"
        )

    lambda_value = row["lambda"]
    if train_type == "zero_shot" and lambda_value != "":
        raise NormalizationError(
            f"{path}: row {row_number} zero_shot must use an empty lambda, found {lambda_value!r}"
        )
    if train_type == "ordinary_lora" and lambda_value != "0":
        raise NormalizationError(
            f"{path}: row {row_number} ordinary_lora must use lambda=0, found {lambda_value!r}"
        )
    if "tone" in train_type and lambda_value == "":
        raise NormalizationError(
            f"{path}: row {row_number} tone-supervised train_type requires lambda"
        )

    snr = str(row["snr"]).strip()
    if snr.lower() != "clean":
        try:
            snr_number = Decimal(snr)
        except InvalidOperation as exc:
            raise NormalizationError(
                f"{path}: row {row_number} snr must be 'clean' or numeric, found {snr!r}"
            ) from exc
        if not snr_number.is_finite():
            raise NormalizationError(f"{path}: row {row_number} snr must be finite")
    noise_type = str(row["noise_type"]).strip().lower()
    if (snr.lower() == "clean") != (noise_type == "clean"):
        raise NormalizationError(
            f"{path}: row {row_number} must pair snr=clean with noise_type=clean"
        )


def _legacy_fields_preserved(
    source_schema: str,
    raw_rows: Sequence[dict[str, str]],
    normalized_rows: Sequence[dict[str, str]],
) -> bool:
    if source_schema == "legacy_zero_shot_8col":
        return all(
            all(raw.get(column, "") == normalized.get(column, "") for column in LEGACY_ZERO_SHOT_COLUMNS)
            for raw, normalized in zip(raw_rows, normalized_rows)
        )
    if source_schema == "canonical_v1":
        return all(
            all(raw.get(column, "") == normalized.get(column, "") for column in CANONICAL_COLUMNS)
            for raw, normalized in zip(raw_rows, normalized_rows)
        )
    legacy_mapping = {
        "utt_id": "utt_id",
        "dataset": "dataset",
        "snr": "snr",
        "text": "ref",
        "prediction": "hyp",
    }
    return all(
        all(raw.get(old, "") == normalized.get(new, "") for old, new in legacy_mapping.items())
        for raw, normalized in zip(raw_rows, normalized_rows)
    )


def prepare_prediction_file(path: Path, overrides: MetadataOverrides) -> PreparedPrediction:
    columns, raw_rows = read_csv(path)
    source_schema = detect_schema(path, columns)
    normalized_rows: list[dict[str, str]] = []
    seen_ids: dict[str, int] = {}

    for row_number, raw_row in enumerate(raw_rows, start=2):
        try:
            normalized = _normalize_row(raw_row, source_schema, overrides)
        except NormalizationError as exc:
            raise NormalizationError(f"{path}: row {row_number}: {exc}") from exc
        _validate_row(path, normalized, row_number)
        utt_id = normalized["utt_id"]
        if utt_id in seen_ids:
            raise NormalizationError(
                f"{path}: duplicate utt_id {utt_id!r} at rows {seen_ids[utt_id]} and {row_number}"
            )
        seen_ids[utt_id] = row_number
        normalized_rows.append(normalized)

    for column in CONSISTENT_RUN_COLUMNS:
        values = {row[column] for row in normalized_rows}
        if len(values) != 1:
            preview = sorted(values)[:5]
            raise NormalizationError(
                f"{path}: column {column!r} must be constant within one prediction run; found {preview}"
            )

    preserved = _legacy_fields_preserved(source_schema, raw_rows, normalized_rows)
    if source_schema == "legacy_zero_shot_8col" and not preserved:
        raise NormalizationError(f"{path}: normalization changed one or more legacy prediction fields")

    return PreparedPrediction(
        source_path=path,
        source_schema=source_schema,
        rows=normalized_rows,
        blank_hypotheses=sum(1 for row in normalized_rows if row["hyp"] == ""),
        legacy_fields_preserved=preserved,
    )


def atomic_write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_index(path: Path) -> BenchmarkIndex:
    if not path.is_file():
        raise NormalizationError(f"benchmark manifest is not a file: {path}")
    columns, rows = read_csv(path)
    required = {"utt_id", "dataset", "snr", "noise_type", "transcript"}
    missing = sorted(required.difference(columns))
    if missing:
        raise NormalizationError(f"{path}: benchmark is missing columns {missing}")
    rows_by_id: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        utt_id = row["utt_id"]
        if utt_id in rows_by_id:
            raise NormalizationError(f"{path}: duplicate benchmark utt_id {utt_id!r} at row {row_number}")
        rows_by_id[utt_id] = row
    return BenchmarkIndex(path=path, sha256=sha256_file(path), rows_by_id=rows_by_id)


def validate_against_benchmark(prepared: PreparedPrediction, benchmark: BenchmarkIndex) -> None:
    prediction_ids = {row["utt_id"] for row in prepared.rows}
    benchmark_ids = set(benchmark.rows_by_id)
    missing = sorted(benchmark_ids.difference(prediction_ids))
    unexpected = sorted(prediction_ids.difference(benchmark_ids))
    if missing or unexpected:
        raise NormalizationError(
            f"{prepared.source_path}: utt_id set does not match {benchmark.path}; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    comparisons = {
        "dataset": "dataset",
        "snr": "snr",
        "noise_type": "noise_type",
        "ref": "transcript",
    }
    for row_number, prediction in enumerate(prepared.rows, start=2):
        benchmark_row = benchmark.rows_by_id[prediction["utt_id"]]
        for prediction_column, benchmark_column in comparisons.items():
            if prediction[prediction_column] != benchmark_row[benchmark_column]:
                raise NormalizationError(
                    f"{prepared.source_path}: row {row_number} {prediction_column} does not match "
                    f"benchmark for utt_id={prediction['utt_id']!r}"
                )


def write_and_verify(prepared: PreparedPrediction, output_path: Path) -> None:
    atomic_write_csv(output_path, prepared.rows, CANONICAL_COLUMNS)
    columns, written_rows = read_csv(output_path)
    if columns != CANONICAL_COLUMNS:
        raise NormalizationError(f"{output_path}: written header verification failed")
    if written_rows != prepared.rows:
        raise NormalizationError(f"{output_path}: written row verification failed")


def discover_inputs(explicit: Sequence[str] | None, patterns: Sequence[str] | None) -> list[Path]:
    candidates: list[Path] = []
    for raw in explicit or []:
        candidates.append(Path(raw))
    for pattern in patterns or []:
        candidates.extend(Path(match) for match in glob.glob(pattern, recursive=True))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(candidates, key=lambda item: item.as_posix().lower()):
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise NormalizationError(f"input is not a file: {path}")
        if path.suffix.lower() != ".csv":
            raise NormalizationError(f"input must be a CSV file: {path}")
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise NormalizationError("no input CSV files matched --input/--input_glob")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize legacy/current ASR prediction CSVs to the shared 11-column schema."
    )
    parser.add_argument("--input", action="append", default=[], help="Input CSV; repeat for multiple files.")
    parser.add_argument(
        "--input_glob",
        "--input-glob",
        action="append",
        default=[],
        help="Glob pattern such as outputs/zero_shot/pred_*.csv; repeatable.",
    )
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--dataset", default=None, help="Override/fill dataset for every input row.")
    parser.add_argument("--model", default=None, help="Override/fill model family.")
    parser.add_argument("--model_size", "--model-size", default=None, help="Fill/check model size.")
    parser.add_argument("--train_type", "--train-type", default=None, help="Fill/check train type.")
    parser.add_argument(
        "--lambda", "--lambda-value", dest="lambda_value", default=None, help="Fill/check tone-loss lambda."
    )
    parser.add_argument(
        "--seed", "--run_seed", "--run-seed", type=int, default=None, help="Fill/check declared run seed."
    )
    parser.add_argument(
        "--benchmark_manifest",
        "--benchmark-manifest",
        default=None,
        help="Optional benchmark CSV for exact ID/ref/condition validation.",
    )
    parser.add_argument(
        "--expected_rows", "--expected-rows", type=int, default=None, help="Expected rows in every input file."
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Report CSV path; defaults to OUTPUT_DIR/normalization_report.csv.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing normalized outputs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        inputs = discover_inputs(args.input, args.input_glob)
        output_dir = Path(args.output_dir)
        report_path = Path(args.report) if args.report else output_dir / "normalization_report.csv"
        overrides = MetadataOverrides(
            dataset=args.dataset,
            model=args.model,
            model_size=args.model_size,
            train_type=args.train_type,
            lambda_value=args.lambda_value,
            seed=args.seed,
        )

        destinations = [output_dir / source.name for source in inputs]
        if len({path.resolve() for path in destinations}) != len(destinations):
            raise NormalizationError("multiple inputs have the same basename and would overwrite each other")
        for source, destination in zip(inputs, destinations):
            if source.resolve() == destination.resolve():
                raise NormalizationError(
                    f"refusing in-place normalization for {source}; choose a different --output_dir"
                )

        read_paths = {source.resolve() for source in inputs}
        if args.benchmark_manifest:
            read_paths.add(Path(args.benchmark_manifest).resolve())
        write_paths = [*destinations, report_path]
        resolved_writes = [path.resolve() for path in write_paths]
        if len(set(resolved_writes)) != len(resolved_writes):
            raise NormalizationError("report/output paths collide with each other")
        for path, resolved in zip(write_paths, resolved_writes):
            if resolved in read_paths:
                raise NormalizationError(f"refusing to overwrite an input or benchmark with output: {path}")
            temporary = path.with_name(f".{path.name}.tmp").resolve()
            if temporary in read_paths or temporary in set(resolved_writes):
                raise NormalizationError(f"temporary output path collides with a protected file: {temporary}")

        existing = [path for path in [*destinations, report_path] if path.exists()]
        if existing and not args.overwrite:
            raise NormalizationError(
                "output already exists; use a new --output_dir or --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

        prepared_files = [prepare_prediction_file(path, overrides) for path in inputs]
        if args.expected_rows is not None:
            if args.expected_rows < 1:
                raise NormalizationError("--expected_rows must be at least 1")
            for prepared in prepared_files:
                if len(prepared.rows) != args.expected_rows:
                    raise NormalizationError(
                        f"{prepared.source_path}: expected {args.expected_rows} rows, found {len(prepared.rows)}"
                    )

        benchmark = None
        if args.benchmark_manifest:
            benchmark = load_benchmark_index(Path(args.benchmark_manifest))
            for prepared in prepared_files:
                validate_against_benchmark(prepared, benchmark)

        key_columns = ["dataset", "utt_id", "model", "model_size", "train_type", "lambda", "seed"]
        seen_keys: dict[tuple[str, ...], Path] = {}
        for prepared in prepared_files:
            for row in prepared.rows:
                key = tuple(row[column] for column in key_columns)
                if key in seen_keys:
                    raise NormalizationError(
                        f"duplicate combined prediction key across {seen_keys[key]} and {prepared.source_path}: {key}"
                    )
                seen_keys[key] = prepared.source_path

        for prepared, destination in zip(prepared_files, destinations):
            write_and_verify(prepared, destination)

        report_rows = []
        for prepared, destination in zip(prepared_files, destinations):
            report_rows.append(
                {
                    "source_path": str(prepared.source_path),
                    "source_sha256": sha256_file(prepared.source_path),
                    "output_path": str(destination),
                    "output_sha256": sha256_file(destination),
                    "source_schema": prepared.source_schema,
                    "schema_version": "1",
                    "rows": len(prepared.rows),
                    "blank_hypotheses": prepared.blank_hypotheses,
                    "legacy_fields_preserved": str(prepared.legacy_fields_preserved).lower(),
                    "train_type": prepared.rows[0]["train_type"],
                    "lambda": prepared.rows[0]["lambda"],
                    "seed": prepared.rows[0]["seed"],
                    "metadata_origin": "cli_backfill" if prepared.source_schema != "canonical_v1" else "input_checked",
                    "benchmark_manifest": str(benchmark.path) if benchmark else "",
                    "benchmark_sha256": benchmark.sha256 if benchmark else "",
                    "status": "PASS",
                }
            )
            print(
                f"PASS {prepared.source_path} -> {destination} "
                f"({len(prepared.rows)} rows, {prepared.source_schema})"
            )
        atomic_write_csv(report_path, report_rows, REPORT_COLUMNS)
        print(f"wrote {report_path}")
    except NormalizationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
