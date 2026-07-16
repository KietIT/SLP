from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import math
import os
import sys
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.analysis import (  # noqa: E402
    CANONICAL_PREDICTION_COLUMNS,
    METRIC_VERSION,
    PredictionValidationError,
    compute_aligned_metric_result,
    load_prediction_csv,
    validate_prediction_rows,
)
from src.vitonesr.prediction_evidence import (  # noqa: E402
    FormalPredictionSet,
    PredictionEvidenceError,
    formal_protocol_parameters,
    verify_formal_prediction_set,
)


DEFAULT_INPUT_GLOB = "outputs/predictions/*/pred_*.csv"
DEFAULT_BENCHMARK_MANIFEST = "outputs/benchmark/benchmark_manifest.csv"
EXPECTED_METRIC_VERSION = "aligned_v1"
RESULT_BUNDLE_VERSION = "aggregate_results_bundle_v2"
RESULT_BUNDLE_MARKER = "results.bundle.commit.json"
RESULT_BUNDLE_JOURNAL = ".results.bundle.transaction.json"
RESULT_BUNDLE_STAGE_PREFIX = ".results.bundle.stage."
RESULT_PROVENANCE_NAME = "aggregate_results.provenance.json"
RESULT_PROVENANCE_VERSION = "aggregate_results_provenance_v1"

RUN_COLUMNS = ["dataset", "model", "model_size", "train_type", "lambda", "seed"]
METRIC_COLUMNS = ["wer", "cer", "ter", "der", "fcer", "swdr"]
METRIC_COUNT_COLUMNS = [
    item
    for metric in METRIC_COLUMNS
    for item in (f"{metric}_numerator", f"{metric}_denominator")
]
METRIC_COVERAGE_COLUMNS = ["ter_coverage", "der_coverage", "fcer_coverage"]
METRIC_EVIDENCE_COLUMNS = [*METRIC_COUNT_COLUMNS, *METRIC_COVERAGE_COLUMNS]
SNR_LEAF_ORDER = ["clean", "20", "10", "5", "0"]
SNR_OUTPUT_ORDER = [*SNR_LEAF_ORDER, "noisy_all", "all"]
KNOWN_NOISE_ORDER = ["clean", "music", "noise", "speech", "babble"]

RESULTS_BY_SNR_COLUMNS = [
    *RUN_COLUMNS,
    "snr",
    "n",
    *METRIC_COLUMNS,
    "metric_version",
    "prediction_sha256",
    "benchmark_manifest_sha256",
    "benchmark_manifest_format",
    *METRIC_EVIDENCE_COLUMNS,
]
RESULTS_BY_NOISE_TYPE_COLUMNS = [
    *RUN_COLUMNS,
    "noise_type",
    "n",
    *METRIC_COLUMNS,
    "metric_version",
    "prediction_sha256",
    "benchmark_manifest_sha256",
    "benchmark_manifest_format",
    *METRIC_EVIDENCE_COLUMNS,
]


class AggregationError(ValueError):
    """Raised when prediction inputs cannot be aggregated safely."""


@dataclass(frozen=True)
class BenchmarkIndex:
    path: Path
    rows_by_id: dict[str, dict[str, str]]
    sha256: str
    manifest_format: str


@dataclass
class PredictionRun:
    source_path: Path
    source_sha256: str
    benchmark_sha256: str
    benchmark_format: str
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, object]]]:
    if not path.is_file():
        raise AggregationError(f"JSONL file does not exist: {path}")
    records: list[tuple[int, dict[str, object]]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise AggregationError(
                        f"{path}: JSONL line {line_number} is blank"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AggregationError(
                        f"{path}: invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise AggregationError(
                        f"{path}: JSONL line {line_number} must be an object"
                    )
                records.append((line_number, value))
    except UnicodeDecodeError as exc:
        raise AggregationError(f"{path}: JSONL must be UTF-8") from exc
    if not records:
        raise AggregationError(f"{path}: JSONL is empty")
    return records


def _reference_value(
    raw: dict[str, object], *, path: Path, row_number: int
) -> str:
    observed = [
        unicodedata.normalize("NFC", str(raw[name]))
        for name in ("transcript", "ref", "text")
        if raw.get(name) is not None and str(raw[name]).strip()
    ]
    if not observed:
        raise AggregationError(
            f"{path}: row {row_number} lacks transcript/ref/text"
        )
    if len(set(observed)) != 1:
        raise AggregationError(
            f"{path}: row {row_number} has conflicting transcript/ref/text values"
        )
    return observed[0]


def _normalize_benchmark_row(
    raw: dict[str, object], *, path: Path, row_number: int
) -> dict[str, str]:
    # A final benchmark has both condition-specific ``utt_id`` and clean-source
    # ``source_utt_id``.  Never replace the former when both are present; the
    # source ID is retained for downstream cluster-bootstrap provenance.
    utt_id = str(raw.get("utt_id") or raw.get("source_utt_id") or "").strip()
    source_utt_id = str(raw.get("source_utt_id") or utt_id).strip()
    dataset = str(raw.get("dataset") or "").strip()
    snr = _normalize_snr(raw.get("snr", ""))
    noise_type = str(raw.get("noise_type") or "").strip()
    transcript = _reference_value(raw, path=path, row_number=row_number)
    blank = [
        name
        for name, value in (
            ("utt_id/source_utt_id", utt_id),
            ("dataset", dataset),
            ("noise_type", noise_type),
            ("transcript/ref", transcript.strip()),
        )
        if not value
    ]
    if blank:
        raise AggregationError(
            f"{path}: row {row_number} has blank required fields {blank}"
        )
    if (snr == "clean") != (noise_type.casefold() == "clean"):
        raise AggregationError(
            f"{path}: row {row_number} has inconsistent snr/noise_type"
        )
    return {
        "utt_id": utt_id,
        "source_utt_id": source_utt_id,
        "dataset": dataset,
        "snr": snr,
        "noise_type": "clean" if snr == "clean" else noise_type,
        "transcript": transcript,
    }


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
    hash_before = _sha256_file(benchmark_path)
    suffix = benchmark_path.suffix.casefold()
    if suffix == ".jsonl":
        raw_rows = _read_jsonl(benchmark_path)
        manifest_format = "jsonl"
    elif suffix == ".csv":
        _columns, csv_rows = _read_csv(benchmark_path)
        raw_rows = [
            (row_number, dict(row))
            for row_number, row in enumerate(csv_rows, start=2)
        ]
        manifest_format = "csv"
    else:
        raise AggregationError(
            f"{benchmark_path}: benchmark manifest must end in .csv or .jsonl"
        )

    rows_by_id: dict[str, dict[str, str]] = {}
    for row_number, raw in raw_rows:
        row = _normalize_benchmark_row(
            raw, path=benchmark_path, row_number=row_number
        )
        utt_id = row["utt_id"]
        if utt_id in rows_by_id:
            raise AggregationError(
                f"{benchmark_path}: duplicate benchmark utt_id {utt_id!r} at row {row_number}"
            )
        rows_by_id[utt_id] = row
    hash_after = _sha256_file(benchmark_path)
    if hash_after != hash_before:
        raise AggregationError(
            f"{benchmark_path}: benchmark changed while it was being read"
        )
    return BenchmarkIndex(
        path=benchmark_path,
        rows_by_id=rows_by_id,
        sha256=hash_after,
        manifest_format=manifest_format,
    )


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
        source_hash_before = _sha256_file(path)
        try:
            loaded_rows = load_prediction_csv(path)
            rows = validate_prediction_rows(loaded_rows, source=path)
        except PredictionValidationError as exc:
            raise AggregationError(f"{path}: {exc}") from exc
        if not rows:
            raise AggregationError(f"{path}: prediction CSV is empty")
        source_hash_after = _sha256_file(path)
        if source_hash_after != source_hash_before:
            raise AggregationError(f"{path}: prediction changed while it was being read")

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
        runs.append(
            PredictionRun(
                source_path=path,
                source_sha256=source_hash_after,
                benchmark_sha256=benchmark.sha256,
                benchmark_format=benchmark.manifest_format,
                metadata=metadata,
                rows=rows,
            )
        )

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
    result = compute_aligned_metric_result(
        [row["ref"] for row in rows],
        [row["hyp"] for row in rows],
    )
    metrics = result.to_dict(include_counts=True)
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
        "prediction_sha256": run.source_sha256,
        "benchmark_manifest_sha256": run.benchmark_sha256,
        "benchmark_manifest_format": run.benchmark_format,
        **{column: metrics[column] for column in METRIC_EVIDENCE_COLUMNS},
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


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _csv_bytes(
    rows: Sequence[dict[str, object]], columns: Sequence[str]
) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _artifact_reference(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_metadata_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_bundle_metadata(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregationError(f"{label} is unreadable or corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"{label} must contain a JSON object: {path}")
    return value


def _bundle_descriptor(contents: Mapping[str, bytes]) -> dict[str, Any]:
    outputs = [
        {
            "name": name,
            "bytes": len(contents[name]),
            "sha256": _sha256_bytes(contents[name]),
        }
        for name in sorted(
            contents, key=lambda item: (item == RESULT_PROVENANCE_NAME, item)
        )
    ]
    identity = {
        "bundle_version": RESULT_BUNDLE_VERSION,
        "outputs": outputs,
    }
    return {
        **identity,
        "bundle_sha256": _sha256_bytes(_canonical_json_bytes(identity)),
    }


def _validate_descriptor(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key in ("bundle_version", "bundle_sha256", "outputs"):
        if value.get(key) != expected.get(key):
            raise AggregationError(
                f"{label} does not match the requested deterministic bundle ({key})"
            )


def _validate_journal_integrity(journal: Mapping[str, Any]) -> None:
    recorded = journal.get("journal_sha256")
    unsigned = {key: value for key, value in journal.items() if key != "journal_sha256"}
    if recorded != _sha256_bytes(_canonical_json_bytes(unsigned)):
        raise AggregationError("aggregate transaction journal integrity check failed")


def _validate_completed_marker(
    marker_path: Path,
    destinations: Mapping[str, Path],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    marker = _load_bundle_metadata(marker_path, label="aggregate commit marker")
    if marker.get("status") != "COMMITTED":
        raise AggregationError("aggregate commit marker is not COMMITTED")
    if marker.get("bundle_version") != RESULT_BUNDLE_VERSION:
        raise AggregationError("aggregate commit marker uses an unsupported version")
    outputs = marker.get("outputs")
    if not isinstance(outputs, list):
        raise AggregationError("aggregate commit marker has invalid outputs")
    identity = {
        "bundle_version": marker.get("bundle_version"),
        "outputs": outputs,
    }
    if marker.get("bundle_sha256") != _sha256_bytes(_canonical_json_bytes(identity)):
        raise AggregationError("aggregate commit marker identity is corrupt")
    if expected is not None:
        _validate_descriptor(marker, expected, label="aggregate commit marker")
    recorded = {
        str(item.get("name")): item
        for item in outputs
        if isinstance(item, dict) and item.get("name")
    }
    if set(recorded) != set(destinations):
        raise AggregationError("aggregate commit marker output set is invalid")
    for name, destination in destinations.items():
        item = recorded[name]
        if not destination.is_file():
            raise AggregationError(
                f"committed aggregate output is missing: {destination}"
            )
        if destination.stat().st_size != item.get("bytes") or _sha256_file(
            destination
        ) != item.get("sha256"):
            raise AggregationError(
                f"committed aggregate output was tampered with: {destination}"
            )
    return marker


def _stage_path(stage_dir: Path, name: str) -> Path:
    return stage_dir / name


def _validate_stage_inventory(stage_dir: Path, output_names: Sequence[str]) -> None:
    if not stage_dir.exists():
        return
    allowed = set(output_names)
    unexpected = sorted(
        path.name for path in stage_dir.iterdir() if path.name not in allowed
    )
    if unexpected:
        raise AggregationError(
            f"aggregate recovery stage contains unexpected entries: {unexpected}"
        )


def _write_stage_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _promote_staged_file(stage_path: Path, destination: Path) -> None:
    """Promotion seam kept small so crash recovery is directly testable."""

    os.replace(stage_path, destination)


def _cleanup_transaction(
    journal_path: Path, stage_dir: Path, output_names: Sequence[str]
) -> None:
    if journal_path.exists():
        journal_path.unlink()
    for name in output_names:
        candidate = _stage_path(stage_dir, name)
        if candidate.exists():
            candidate.unlink()
    try:
        stage_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        # Unknown files are deliberately not removed recursively.  A committed
        # marker still proves the canonical bundle; the residue remains visible
        # for manual inspection rather than being silently deleted.
        pass


def _commit_result_bundle(
    directory: Path,
    contents: Mapping[str, bytes],
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[Path, Path]:
    if overwrite and resume:
        raise AggregationError("--overwrite and --resume are mutually exclusive")
    descriptor = _bundle_descriptor(contents)
    output_names = [str(item["name"]) for item in descriptor["outputs"]]
    destinations = {name: directory / name for name in output_names}
    marker_path = directory / RESULT_BUNDLE_MARKER
    journal_path = directory / RESULT_BUNDLE_JOURNAL
    stage_dir = directory / (
        RESULT_BUNDLE_STAGE_PREFIX + str(descriptor["bundle_sha256"])
    )
    directory.mkdir(parents=True, exist_ok=True)

    if not resume and journal_path.exists():
        raise AggregationError(
            f"unfinished aggregate transaction exists; rerun with --resume: {journal_path}"
        )
    if not resume and stage_dir.exists():
        raise AggregationError(
            f"orphan aggregate stage exists; rerun the exact command with --resume: {stage_dir}"
        )

    if resume and not journal_path.exists() and marker_path.exists():
        _validate_completed_marker(
            marker_path, destinations, expected=descriptor
        )
        return destinations["results_by_snr.csv"], destinations[
            "results_by_noise_type.csv"
        ]

    occupied = [path for path in destinations.values() if path.exists()]
    if not resume and not overwrite and (occupied or marker_path.exists()):
        raise AggregationError(
            "output already exists; choose another --output-dir or use --overwrite: "
            + ", ".join(str(path) for path in occupied or [marker_path])
        )

    journal: dict[str, Any]
    if journal_path.exists():
        if not resume:
            raise AggregationError(
                f"unfinished aggregate transaction exists; rerun with --resume: {journal_path}"
            )
        journal = _load_bundle_metadata(
            journal_path, label="aggregate transaction journal"
        )
        _validate_journal_integrity(journal)
        if journal.get("status") != "PREPARED":
            raise AggregationError("aggregate transaction journal is not PREPARED")
        _validate_descriptor(
            journal, descriptor, label="aggregate transaction journal"
        )
        if journal.get("mode") not in {"create", "overwrite"}:
            raise AggregationError("aggregate transaction journal has invalid mode")
    else:
        prior_hashes: dict[str, str | None] = {}
        for name, destination in destinations.items():
            if not destination.exists():
                prior_hashes[name] = None
                continue
            current_hash = _sha256_file(destination)
            expected_hash = _sha256_bytes(contents[name])
            if resume:
                if current_hash != expected_hash:
                    raise AggregationError(
                        f"cannot recover aggregate bundle: unexpected canonical file {destination}"
                    )
                prior_hashes[name] = expected_hash
            else:
                prior_hashes[name] = current_hash

        prior_marker_sha256: str | None = None
        if marker_path.exists():
            if resume:
                raise AggregationError(
                    "cannot recover aggregate bundle: marker does not bind the requested outputs"
                )
            _validate_completed_marker(marker_path, destinations)
            prior_marker_sha256 = _sha256_file(marker_path)

        stage_dir.mkdir(parents=False, exist_ok=True)
        _validate_stage_inventory(stage_dir, output_names)
        for name, content in contents.items():
            destination = destinations[name]
            if destination.is_file() and _sha256_file(destination) == _sha256_bytes(
                content
            ):
                continue
            staged = _stage_path(stage_dir, name)
            if staged.exists():
                if _sha256_file(staged) != _sha256_bytes(content):
                    raise AggregationError(
                        f"aggregate recovery stage was tampered with: {staged}"
                    )
            else:
                _write_stage_file(staged, content)
        journal_unsigned = {
            **descriptor,
            "status": "PREPARED",
            "mode": "overwrite" if overwrite else "create",
            "prior_sha256": prior_hashes,
            "prior_marker_sha256": prior_marker_sha256,
        }
        journal = {
            **journal_unsigned,
            "journal_sha256": _sha256_bytes(
                _canonical_json_bytes(journal_unsigned)
            ),
        }
        _atomic_metadata_write(journal_path, _canonical_json_bytes(journal))

    _validate_stage_inventory(stage_dir, output_names)

    # Invalidate an earlier completed bundle only after the durable journal is
    # present.  A crash from this point is recoverable with --resume.
    if marker_path.exists():
        expected_marker_hash = journal.get("prior_marker_sha256")
        if expected_marker_hash and _sha256_file(marker_path) == expected_marker_hash:
            marker_path.unlink()
        elif journal.get("mode") == "create":
            _validate_completed_marker(
                marker_path, destinations, expected=descriptor
            )
            _cleanup_transaction(journal_path, stage_dir, output_names)
            return destinations["results_by_snr.csv"], destinations[
                "results_by_noise_type.csv"
            ]
        else:
            raise AggregationError(
                "aggregate commit marker changed during recovery; refusing to overwrite"
            )

    prior = journal.get("prior_sha256")
    if not isinstance(prior, dict) or set(prior) != set(destinations):
        raise AggregationError("aggregate transaction journal has invalid prior hashes")
    for name in output_names:
        destination = destinations[name]
        expected_hash = _sha256_bytes(contents[name])
        if destination.exists():
            current_hash = _sha256_file(destination)
            if current_hash == expected_hash:
                continue
            if journal.get("mode") != "overwrite" or current_hash != prior.get(name):
                raise AggregationError(
                    f"aggregate canonical output changed during transaction: {destination}"
                )
        staged = _stage_path(stage_dir, name)
        if not staged.is_file() or _sha256_file(staged) != expected_hash:
            raise AggregationError(
                f"aggregate staged output is missing or tampered: {staged}"
            )
        _promote_staged_file(staged, destination)

    marker = {**descriptor, "status": "COMMITTED"}
    _atomic_metadata_write(marker_path, _canonical_json_bytes(marker))
    _validate_completed_marker(marker_path, destinations, expected=descriptor)
    _cleanup_transaction(journal_path, stage_dir, output_names)
    return destinations["results_by_snr.csv"], destinations[
        "results_by_noise_type.csv"
    ]


def _result_destinations(output_dir: str | Path) -> tuple[Path, ...]:
    directory = Path(output_dir)
    return (
        directory / "results_by_snr.csv",
        directory / "results_by_noise_type.csv",
        directory / RESULT_PROVENANCE_NAME,
    )


def ensure_outputs_available(
    output_dir: str | Path, *, overwrite: bool, resume: bool = False
) -> None:
    if overwrite and resume:
        raise AggregationError("--overwrite and --resume are mutually exclusive")
    if resume:
        return
    directory = Path(output_dir)
    destinations = _result_destinations(output_dir)
    existing = [path for path in destinations if path.exists()]
    journal = directory / RESULT_BUNDLE_JOURNAL
    marker = directory / RESULT_BUNDLE_MARKER
    stages = (
        sorted(directory.glob(f"{RESULT_BUNDLE_STAGE_PREFIX}*"))
        if directory.is_dir()
        else []
    )
    if journal.exists():
        raise AggregationError(
            f"unfinished aggregate transaction exists; rerun with --resume: {journal}"
        )
    if stages:
        raise AggregationError(
            "orphan aggregate stage exists; rerun the exact command with --resume: "
            + ", ".join(str(path) for path in stages)
        )
    if (existing or marker.exists()) and not overwrite:
        raise AggregationError(
            "output already exists; choose another --output-dir or use --overwrite: "
            + ", ".join(str(path) for path in existing or [marker])
        )


def write_results(
    output_dir: str | Path,
    snr_rows: Sequence[dict[str, object]],
    noise_rows: Sequence[dict[str, object]],
    *,
    provenance_context: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    contents = {
        "results_by_snr.csv": _csv_bytes(snr_rows, RESULTS_BY_SNR_COLUMNS),
        "results_by_noise_type.csv": _csv_bytes(
            noise_rows, RESULTS_BY_NOISE_TYPE_COLUMNS
        ),
    }
    data_outputs = [
        {
            "name": name,
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
        }
        for name, content in sorted(contents.items())
    ]
    provenance = {
        "provenance_version": RESULT_PROVENANCE_VERSION,
        "bundle_version": RESULT_BUNDLE_VERSION,
        "metric_version": EXPECTED_METRIC_VERSION,
        **dict(provenance_context or {}),
        "data_outputs": data_outputs,
    }
    contents[RESULT_PROVENANCE_NAME] = (
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return _commit_result_bundle(
        directory, contents, overwrite=overwrite, resume=resume
    )


def run_aggregation(
    input_paths: Sequence[str | Path],
    benchmark_manifest: str | Path,
    output_dir: str | Path,
    *,
    metric_version: str = EXPECTED_METRIC_VERSION,
    formal_paper_v2: bool = False,
    split_lock: str | Path | None = None,
    decision_lock: str | Path | None = None,
    final_benchmark_lock: str | Path | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    if metric_version != EXPECTED_METRIC_VERSION:
        raise AggregationError(
            f"unsupported metric_version={metric_version!r}; "
            f"expected {EXPECTED_METRIC_VERSION!r}"
        )
    ensure_outputs_available(output_dir, overwrite=overwrite, resume=resume)
    formal_evidence: FormalPredictionSet | None = None
    if formal_paper_v2:
        missing = [
            name
            for name, value in (
                ("split_lock", split_lock),
                ("decision_lock", decision_lock),
                ("final_benchmark_lock", final_benchmark_lock),
            )
            if value is None
        ]
        if missing:
            raise AggregationError(
                "formal paper-v2 aggregation requires " + ", ".join(missing)
            )
        try:
            formal_evidence = verify_formal_prediction_set(
                input_paths,
                benchmark_path=benchmark_manifest,
                split_lock_path=split_lock,
                decision_path=decision_lock,
                final_benchmark_lock_path=final_benchmark_lock,
                root=ROOT,
            )
        except PredictionEvidenceError as exc:
            raise AggregationError(str(exc)) from exc
    benchmark = load_benchmark_index(benchmark_manifest)
    runs = load_prediction_runs(input_paths, benchmark)
    snr_rows, noise_rows = aggregate_runs(runs)
    if formal_evidence is not None:
        try:
            current_evidence = verify_formal_prediction_set(
                input_paths,
                benchmark_path=benchmark_manifest,
                split_lock_path=split_lock,
                decision_path=decision_lock,
                final_benchmark_lock_path=final_benchmark_lock,
                root=ROOT,
            )
        except PredictionEvidenceError as exc:
            raise AggregationError(str(exc)) from exc
        if formal_protocol_parameters(current_evidence) != formal_protocol_parameters(
            formal_evidence
        ):
            raise AggregationError(
                "formal prediction evidence changed during aggregation"
            )
        formal_evidence = current_evidence
    evidence_by_path = (
        {
            item.prediction_path.resolve(): item
            for item in formal_evidence.predictions
        }
        if formal_evidence is not None
        else {}
    )
    provenance_inputs: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: _artifact_reference(item.source_path).casefold()):
        binding: dict[str, Any] = {
            "path": _artifact_reference(run.source_path),
            "bytes": run.source_path.stat().st_size,
            "sha256": run.source_sha256,
        }
        evidence = evidence_by_path.get(run.source_path.resolve())
        if evidence is not None:
            binding.update(
                {
                    "provenance_path": _artifact_reference(evidence.provenance_path),
                    "provenance_sha256": evidence.provenance_sha256,
                    "provenance_version": evidence.provenance_version,
                }
            )
        provenance_inputs.append(binding)
    provenance_context: dict[str, Any] = {
        "inputs": provenance_inputs,
        "benchmark": {
            "path": _artifact_reference(benchmark.path),
            "bytes": benchmark.path.stat().st_size,
            "sha256": benchmark.sha256,
            "format": benchmark.manifest_format,
        },
        "formal_protocol": (
            formal_protocol_parameters(formal_evidence)
            if formal_evidence is not None
            else None
        ),
    }
    snr_path, noise_path = write_results(
        output_dir,
        snr_rows,
        noise_rows,
        provenance_context=provenance_context,
        overwrite=overwrite,
        resume=resume,
    )
    return {
        "input_files": len(input_paths),
        "prediction_rows": sum(len(run.rows) for run in runs),
        "runs": len(runs),
        "results_by_snr_rows": len(snr_rows),
        "results_by_noise_type_rows": len(noise_rows),
        "results_by_snr": str(snr_path),
        "results_by_noise_type": str(noise_path),
        "provenance": str(Path(output_dir) / RESULT_PROVENANCE_NAME),
        "metric_version": EXPECTED_METRIC_VERSION,
        "benchmark_manifest_sha256": benchmark.sha256,
        "benchmark_manifest_format": benchmark.manifest_format,
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
        help=(
            "Benchmark CSV or JSONL used for exact ID/reference/condition "
            "validation. JSONL accepts transcript/ref/text reference aliases."
        ),
    )
    parser.add_argument(
        "--metric_version",
        "--metric-version",
        choices=[EXPECTED_METRIC_VERSION],
        default=EXPECTED_METRIC_VERSION,
        help=f"Metric implementation version (only {EXPECTED_METRIC_VERSION} is supported).",
    )
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument(
        "--formal-paper-v2",
        action="store_true",
        help="Require decision/lock/sidecar verification before reading formal results.",
    )
    parser.add_argument("--split-lock")
    parser.add_argument("--decision-lock")
    parser.add_argument("--final-benchmark-lock")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Recover an interrupted bundle only when every existing canonical, "
            "staged, and journal hash matches this exact recomputation."
        ),
    )
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
            formal_paper_v2=args.formal_paper_v2,
            split_lock=args.split_lock,
            decision_lock=args.decision_lock,
            final_benchmark_lock=args.final_benchmark_lock,
            overwrite=args.overwrite,
            resume=args.resume,
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
    print(f"wrote {result['provenance']}")


if __name__ == "__main__":
    main()
