"""Auditable paired cluster bootstrap for locked ASR comparisons.

The final robustness benchmark contains several observations derived from the
same clean utterance (clean plus SNR/noise replicas).  Those observations are
not independent.  This module therefore samples ``source_utt_id`` clusters and
keeps every observation in a sampled cluster together.

Comparison roles are resolved from a locked method-decision artifact rather
than inferred from fixed lambda values.  This matters when the selected lambda
changes between protocol revisions.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import (
    METRIC_VERSION,
    RUN_METADATA_COLUMNS,
    PredictionValidationError,
    compute_aligned_metric_result,
    load_prediction_csv,
)
from .artifact_bundle import bind_input_files, commit_artifact_bundle
from .prediction_evidence import (
    FormalPredictionSet,
    PredictionEvidenceError,
    formal_protocol_parameters,
    verify_formal_prediction_set,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_METRIC_VERSION = "aligned_v1"
BOOTSTRAP_BUNDLE_VERSION = "cluster_bootstrap_bundle_v1"
COMPARISON_ROLES = (
    "ordinary_baseline",
    "selected_method",
    "locked_control",
)
CLUSTER_UNITS = (
    "source_utt_id",
    "utt_id_singleton_external",
)
PAIR_SPECS = (
    ("ordinary_baseline_vs_selected_method", "ordinary_baseline", "selected_method"),
    ("ordinary_baseline_vs_locked_control", "ordinary_baseline", "locked_control"),
    ("selected_method_vs_locked_control", "selected_method", "locked_control"),
)
METRIC_FIELDS = {
    "wer": ("word_errors", "word_reference_units", "word_reference_units"),
    "cer": (
        "character_errors",
        "character_reference_units",
        "character_reference_units",
    ),
    "ter": ("tone_errors", "tone_reference_units", "word_reference_units"),
    "der": (
        "diacritic_errors",
        "diacritic_reference_units",
        "word_reference_units",
    ),
}
OUTPUT_COLUMNS = (
    "metric_version",
    "decision_sha256",
    "benchmark_sha256",
    "comparison_set_sha256",
    "dataset",
    "split",
    "pair_id",
    "role_a",
    "configuration_id_a",
    "method_id_a",
    "train_type_a",
    "lambda_a",
    "seed_a",
    "prediction_sha256_a",
    "role_b",
    "configuration_id_b",
    "method_id_b",
    "train_type_b",
    "lambda_b",
    "seed_b",
    "prediction_sha256_b",
    "n_source_clusters",
    "n_paired_conditions",
    "metric",
    "numerator_a",
    "denominator_a",
    "coverage_numerator_a",
    "coverage_denominator_a",
    "coverage_a",
    "estimate_a",
    "numerator_b",
    "denominator_b",
    "coverage_numerator_b",
    "coverage_denominator_b",
    "coverage_b",
    "estimate_b",
    "delta_b_minus_a",
    "n_bootstrap",
    "n_valid_bootstrap",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "ci_excludes_zero",
    "bootstrap_seed",
    "bootstrap_unit",
    "ci_method",
)


class ClusterBootstrapError(ValueError):
    """Raised when a comparison is unpaired, ambiguous, or unauditable."""


@dataclass(frozen=True, slots=True)
class LockedConfiguration:
    configuration_id: str
    role: str
    method_id: str
    train_type: str
    lambda_value: str
    seed: str


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    utt_id: str
    source_utt_id: str
    dataset: str
    split: str
    snr: str
    noise_type: str
    ref: str


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    configuration: LockedConfiguration
    prediction_path: Path
    prediction_sha256: str
    metadata: Mapping[str, str]
    rows_by_utt_id: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class ClusterBootstrapInputs:
    decision_path: Path
    decision_sha256: str
    benchmark_path: Path
    benchmark_sha256: str
    comparison_set_sha256: str
    dataset: str
    split: str
    bootstrap_unit: str
    observations: tuple[BenchmarkObservation, ...]
    runs: Mapping[str, ComparisonRun]
    formal_evidence: FormalPredictionSet | None = None


@dataclass(frozen=True, slots=True)
class MetricContribution:
    numerator: np.ndarray
    denominator: np.ndarray
    coverage_numerator: np.ndarray
    coverage_denominator: np.ndarray


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_number(value: object, *, label: str) -> str:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ClusterBootstrapError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ClusterBootstrapError(f"{label} must be finite and non-negative")
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _canonical_snr(value: object, *, label: str) -> str:
    text = str(value).strip()
    if text.casefold() == "clean":
        return "clean"
    return _canonical_number(text, label=label)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClusterBootstrapError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClusterBootstrapError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_configurations(
    decision: Mapping[str, Any],
    comparison_set: Mapping[str, str] | None,
) -> dict[str, LockedConfiguration]:
    if decision.get("status") != "LOCKED" or decision.get("test_unlocked") is not True:
        raise ClusterBootstrapError(
            "method decision must have status=LOCKED and test_unlocked=true"
        )
    raw_configurations = decision.get("locked_configurations")
    if not isinstance(raw_configurations, list) or not raw_configurations:
        raise ClusterBootstrapError("decision has no locked_configurations")

    by_role: dict[str, list[LockedConfiguration]] = {
        role: [] for role in COMPARISON_ROLES
    }
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_configurations):
        if not isinstance(raw, Mapping):
            raise ClusterBootstrapError(
                f"locked_configurations[{index}] must be an object"
            )
        role = str(raw.get("role", "")).strip()
        if role not in by_role:
            raise ClusterBootstrapError(
                f"locked_configurations[{index}] has unsupported role {role!r}"
            )
        configuration_id = str(raw.get("configuration_id", "")).strip()
        method_id = str(raw.get("method_id", "")).strip()
        train_type = str(raw.get("train_type", "")).strip()
        try:
            seed = str(int(raw["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ClusterBootstrapError(
                f"configuration {configuration_id or index!r} has invalid seed"
            ) from exc
        if not configuration_id or not method_id or not train_type:
            raise ClusterBootstrapError(
                f"locked_configurations[{index}] lacks identity metadata"
            )
        if configuration_id in seen_ids:
            raise ClusterBootstrapError(
                f"duplicate locked configuration_id: {configuration_id}"
            )
        seen_ids.add(configuration_id)
        by_role[role].append(
            LockedConfiguration(
                configuration_id=configuration_id,
                role=role,
                method_id=method_id,
                train_type=train_type,
                lambda_value=_canonical_number(
                    raw.get("lambda"), label=f"{configuration_id}.lambda"
                ),
                seed=seed,
            )
        )

    if comparison_set is None:
        ambiguous = {
            role: len(items) for role, items in by_role.items() if len(items) != 1
        }
        if ambiguous:
            raise ClusterBootstrapError(
                "decision roles are missing or ambiguous; provide an explicit "
                f"comparison_set mapping: {ambiguous}"
            )
        resolved = {role: items[0] for role, items in by_role.items()}
    else:
        if set(comparison_set) != set(COMPARISON_ROLES):
            raise ClusterBootstrapError(
                "comparison_set must map exactly the roles "
                f"{list(COMPARISON_ROLES)}"
            )
        resolved = {}
        for role in COMPARISON_ROLES:
            configuration_id = str(comparison_set[role]).strip()
            matches = [
                item
                for item in by_role[role]
                if item.configuration_id == configuration_id
            ]
            if len(matches) != 1:
                raise ClusterBootstrapError(
                    f"comparison_set configuration {configuration_id!r} does not "
                    f"identify exactly one {role} run"
                )
            resolved[role] = matches[0]

    if len({item.configuration_id for item in resolved.values()}) != 3:
        raise ClusterBootstrapError("comparison roles must use three distinct configurations")
    selected = resolved["selected_method"]
    selected_method_id = str(decision.get("selected_method_id", "")).strip()
    if selected.method_id != selected_method_id:
        raise ClusterBootstrapError(
            "selected_method role does not match decision.selected_method_id"
        )
    selected_lambda = _canonical_number(
        decision.get("selected_lambda"), label="decision.selected_lambda"
    )
    if selected.lambda_value != selected_lambda:
        raise ClusterBootstrapError(
            "selected_method lambda does not match decision.selected_lambda"
        )
    ordinary = resolved["ordinary_baseline"]
    if ordinary.train_type != "ordinary_lora" or ordinary.lambda_value != "0":
        raise ClusterBootstrapError(
            "ordinary_baseline must be ordinary_lora with lambda=0"
        )
    return resolved


def _load_manifest_records(path: Path) -> list[tuple[int, Mapping[str, object]]]:
    suffix = path.suffix.casefold()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ClusterBootstrapError("benchmark CSV has no header")
                records: list[tuple[int, Mapping[str, object]]] = []
                for row_number, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise ClusterBootstrapError(
                            f"benchmark CSV row {row_number} has missing/extra cells"
                        )
                    records.append((row_number, row))
                return records
        if suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ClusterBootstrapError(
                            f"benchmark JSONL line {line_number} is blank"
                        )
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ClusterBootstrapError(
                            f"benchmark JSONL line {line_number} is invalid: {exc.msg}"
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise ClusterBootstrapError(
                            f"benchmark JSONL line {line_number} must be an object"
                        )
                    records.append((line_number, value))
            return records
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ClusterBootstrapError(
            f"Cannot read benchmark manifest {path}: {exc}"
        ) from exc
    raise ClusterBootstrapError(
        f"benchmark manifest must end in .csv or .jsonl: {path}"
    )


def _manifest_reference(
    row: Mapping[str, object], *, row_number: int
) -> str:
    values = [
        unicodedata.normalize("NFC", str(row[name]))
        for name in ("ref", "transcript", "text")
        if row.get(name) is not None and str(row[name]).strip()
    ]
    if not values:
        raise ClusterBootstrapError(
            f"benchmark row {row_number} lacks ref/transcript/text"
        )
    if len(set(values)) != 1:
        raise ClusterBootstrapError(
            f"benchmark row {row_number} has conflicting ref/transcript/text"
        )
    return values[0]


def _load_benchmark(
    path: Path, *, cluster_unit: str = "source_utt_id"
) -> tuple[BenchmarkObservation, ...]:
    if cluster_unit not in CLUSTER_UNITS:
        raise ClusterBootstrapError(
            f"cluster_unit must be one of {list(CLUSTER_UNITS)}"
        )
    records = _load_manifest_records(path)
    if not records:
        raise ClusterBootstrapError("benchmark manifest is empty")

    observations: list[BenchmarkObservation] = []
    seen_ids: set[str] = set()
    external_source_ids: list[str] = []
    for row_number, row in records:
        utt_id = str(row.get("utt_id") or row.get("source_utt_id") or "").strip()
        declared_source_id = str(row.get("source_utt_id") or "").strip()
        if cluster_unit == "source_utt_id":
            source_utt_id = declared_source_id
        else:
            source_utt_id = utt_id
            external_source_ids.append(declared_source_id or utt_id)
        dataset = str(row.get("dataset", "")).strip()
        split = str(row.get("split", "")).strip()
        noise_type = str(row.get("noise_type", "")).strip()
        ref = _manifest_reference(row, row_number=row_number)
        if not all((utt_id, source_utt_id, dataset, split, noise_type, ref.strip())):
            raise ClusterBootstrapError(
                f"benchmark row {row_number} has an empty paired field"
            )
        if utt_id in seen_ids:
            raise ClusterBootstrapError(f"benchmark has duplicate utt_id {utt_id!r}")
        seen_ids.add(utt_id)
        snr = _canonical_snr(row.get("snr", ""), label=f"row {row_number}.snr")
        if (snr == "clean") != (noise_type.casefold() == "clean"):
            raise ClusterBootstrapError(
                f"benchmark row {row_number} has inconsistent snr/noise_type"
            )
        observations.append(
            BenchmarkObservation(
                utt_id=utt_id,
                source_utt_id=source_utt_id,
                dataset=dataset,
                split=split,
                snr=snr,
                noise_type=noise_type,
                ref=ref,
            )
        )
    datasets = {row.dataset for row in observations}
    splits = {row.split for row in observations}
    if len(datasets) != 1 or len(splits) != 1:
        raise ClusterBootstrapError(
            "bootstrap manifest must contain exactly one dataset and one split"
        )
    dataset = next(iter(datasets))
    if cluster_unit == "utt_id_singleton_external":
        if dataset.casefold() == "vivos":
            raise ClusterBootstrapError(
                "VIVOS final evaluation requires source_utt_id cluster bootstrap"
            )
        if len(external_source_ids) != len(set(external_source_ids)):
            raise ClusterBootstrapError(
                "utt_id_singleton_external requires one condition per source utterance"
            )
    return tuple(sorted(observations, key=lambda row: row.utt_id.casefold()))


def _load_run(
    configuration: LockedConfiguration,
    prediction_path: Path,
    observations: Sequence[BenchmarkObservation],
) -> ComparisonRun:
    prediction_hash_before = sha256_file(prediction_path)
    try:
        rows = load_prediction_csv(prediction_path)
    except (FileNotFoundError, PredictionValidationError) as exc:
        raise ClusterBootstrapError(str(exc)) from exc
    expected_ids = {item.utt_id for item in observations}
    rows_by_id = {row["utt_id"]: row for row in rows}
    actual_ids = set(rows_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:5]
        extra = sorted(actual_ids - expected_ids)[:5]
        raise ClusterBootstrapError(
            f"{prediction_path}: paired utt_id set differs from benchmark; "
            f"missing={missing}, extra={extra}"
        )
    metadata = {column: rows[0][column] for column in RUN_METADATA_COLUMNS}
    expected_metadata = {
        "train_type": configuration.train_type,
        "lambda": configuration.lambda_value,
        "seed": configuration.seed,
    }
    for field, expected in expected_metadata.items():
        if metadata[field] != expected:
            raise ClusterBootstrapError(
                f"{prediction_path}: {field}={metadata[field]!r} does not match "
                f"decision configuration {configuration.configuration_id} ({expected!r})"
            )
    for observation in observations:
        row = rows_by_id[observation.utt_id]
        if row["dataset"] != observation.dataset:
            raise ClusterBootstrapError(
                f"{prediction_path}: dataset differs for utt_id={observation.utt_id!r}"
            )
        if row["snr"] != observation.snr or row["noise_type"] != observation.noise_type:
            raise ClusterBootstrapError(
                f"{prediction_path}: condition differs for utt_id={observation.utt_id!r}"
            )
        if row["ref"] != observation.ref:
            raise ClusterBootstrapError(
                f"{prediction_path}: ref differs for utt_id={observation.utt_id!r}"
            )
    prediction_hash_after = sha256_file(prediction_path)
    if prediction_hash_after != prediction_hash_before:
        raise ClusterBootstrapError(
            f"{prediction_path}: prediction changed while it was being read"
        )
    return ComparisonRun(
        configuration=configuration,
        prediction_path=prediction_path,
        prediction_sha256=prediction_hash_after,
        metadata=metadata,
        rows_by_utt_id=rows_by_id,
    )


def load_cluster_bootstrap_inputs(
    decision_path: str | Path,
    benchmark_path: str | Path,
    prediction_paths: Mapping[str, str | Path],
    *,
    comparison_set: Mapping[str, str] | None = None,
    cluster_unit: str = "source_utt_id",
    formal_paper_v2: bool = False,
    split_lock_path: str | Path | None = None,
    final_benchmark_lock_path: str | Path | None = None,
) -> ClusterBootstrapInputs:
    """Load and exactly pair three decision-locked prediction runs.

    ``prediction_paths`` is keyed by ``configuration_id``.  When a decision
    contains more than one configuration for any comparison role (for example,
    multi-seed results), ``comparison_set`` must explicitly select one
    configuration ID for each role.
    """

    if METRIC_VERSION != EXPECTED_METRIC_VERSION:
        raise ClusterBootstrapError(
            f"expected metric_version={EXPECTED_METRIC_VERSION!r}, found {METRIC_VERSION!r}"
        )
    decision_file = Path(decision_path)
    benchmark_file = Path(benchmark_path)
    formal_evidence: FormalPredictionSet | None = None
    if formal_paper_v2:
        if split_lock_path is None:
            raise ClusterBootstrapError(
                "formal paper-v2 bootstrap requires split_lock_path"
            )
        try:
            formal_evidence = verify_formal_prediction_set(
                list(prediction_paths.values()),
                benchmark_path=benchmark_file,
                split_lock_path=split_lock_path,
                decision_path=decision_file,
                final_benchmark_lock_path=final_benchmark_lock_path,
                required_configuration_ids={
                    path: configuration_id
                    for configuration_id, path in prediction_paths.items()
                },
            )
        except PredictionEvidenceError as exc:
            raise ClusterBootstrapError(str(exc)) from exc
        decision_hash_before = formal_evidence.decision.sha256
        decision_hash_after = decision_hash_before
        decision = dict(formal_evidence.decision.raw)
    else:
        decision_hash_before = sha256_file(decision_file)
        decision = _load_json_object(decision_file, label="method decision")
        decision_hash_after = sha256_file(decision_file)
        if decision_hash_after != decision_hash_before:
            raise ClusterBootstrapError("method decision changed while it was being read")
    configurations = _resolve_configurations(decision, comparison_set)
    expected_configuration_ids = {
        configuration.configuration_id for configuration in configurations.values()
    }
    if set(prediction_paths) != expected_configuration_ids:
        raise ClusterBootstrapError(
            "prediction_paths must map exactly the selected configuration IDs; "
            f"expected={sorted(expected_configuration_ids)}, "
            f"found={sorted(prediction_paths)}"
        )
    benchmark_hash_before = sha256_file(benchmark_file)
    observations = _load_benchmark(benchmark_file, cluster_unit=cluster_unit)
    benchmark_hash_after = sha256_file(benchmark_file)
    if benchmark_hash_after != benchmark_hash_before:
        raise ClusterBootstrapError(
            "benchmark manifest changed while it was being read"
        )
    resolved_paths = {
        role: Path(prediction_paths[configuration.configuration_id])
        for role, configuration in configurations.items()
    }
    if len({path.resolve() for path in resolved_paths.values()}) != 3:
        raise ClusterBootstrapError("comparison roles must use three distinct prediction files")
    runs = {
        role: _load_run(configuration, resolved_paths[role], observations)
        for role, configuration in configurations.items()
    }
    if formal_evidence is not None:
        expected_prediction_hashes = {
            item.prediction_path.resolve(): item.prediction_sha256
            for item in formal_evidence.predictions
        }
        for run in runs.values():
            if expected_prediction_hashes.get(run.prediction_path.resolve()) != run.prediction_sha256:
                raise ClusterBootstrapError(
                    f"{run.prediction_path}: prediction changed after provenance verification"
                )
        for item in formal_evidence.predictions:
            if sha256_file(item.provenance_path) != item.provenance_sha256:
                raise ClusterBootstrapError(
                    f"{item.provenance_path}: provenance changed during bootstrap loading"
                )
        if (
            sha256_file(decision_file) != formal_evidence.decision.sha256
            or benchmark_hash_after != formal_evidence.benchmark.sha256
        ):
            raise ClusterBootstrapError(
                "Formal decision/benchmark changed during bootstrap loading"
            )
        final_lock = formal_evidence.benchmark.final_benchmark_lock_path
        if final_lock is not None and sha256_file(final_lock) != (
            formal_evidence.benchmark.final_benchmark_lock_sha256
        ):
            raise ClusterBootstrapError(
                "Final benchmark lock changed during bootstrap loading"
            )
    dataset = observations[0].dataset
    split = observations[0].split
    comparison_identity = {
        role: configurations[role].configuration_id for role in COMPARISON_ROLES
    }
    return ClusterBootstrapInputs(
        decision_path=decision_file,
        decision_sha256=decision_hash_after,
        benchmark_path=benchmark_file,
        benchmark_sha256=benchmark_hash_after,
        comparison_set_sha256=_canonical_json_sha256(comparison_identity),
        dataset=dataset,
        split=split,
        bootstrap_unit=cluster_unit,
        observations=observations,
        runs=runs,
        formal_evidence=formal_evidence,
    )


def _cluster_contributions(
    inputs: ClusterBootstrapInputs,
) -> tuple[tuple[str, ...], dict[str, dict[str, MetricContribution]]]:
    cluster_ids = tuple(
        sorted({item.source_utt_id for item in inputs.observations}, key=str.casefold)
    )
    cluster_indices = {source_id: index for index, source_id in enumerate(cluster_ids)}
    values: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for role, run in inputs.runs.items():
        values[role] = {}
        for metric in METRIC_FIELDS:
            values[role][metric] = {
                field: np.zeros(len(cluster_ids), dtype=np.int64)
                for field in (
                    "numerator",
                    "denominator",
                    "coverage_numerator",
                    "coverage_denominator",
                )
            }
        for observation in inputs.observations:
            row = run.rows_by_utt_id[observation.utt_id]
            result = compute_aligned_metric_result([row["ref"]], [row["hyp"]])
            cluster_index = cluster_indices[observation.source_utt_id]
            for metric, (
                numerator_field,
                denominator_field,
                coverage_denominator_field,
            ) in METRIC_FIELDS.items():
                denominator = int(getattr(result, denominator_field))
                metric_values = values[role][metric]
                metric_values["numerator"][cluster_index] += int(
                    getattr(result, numerator_field)
                )
                metric_values["denominator"][cluster_index] += denominator
                metric_values["coverage_numerator"][cluster_index] += denominator
                metric_values["coverage_denominator"][cluster_index] += int(
                    getattr(result, coverage_denominator_field)
                )
    contributions = {
        role: {
            metric: MetricContribution(**metric_values)
            for metric, metric_values in role_values.items()
        }
        for role, role_values in values.items()
    }
    return cluster_ids, contributions


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ClusterBootstrapError("bootstrap produced a non-finite statistic")
    return f"{value:.12f}"


def _safe_scalar_ratio(numerator: int, denominator: int, *, label: str) -> float:
    if denominator <= 0:
        raise ClusterBootstrapError(f"{label} has zero metric denominator")
    return numerator / denominator


def build_cluster_bootstrap_rows(
    inputs: ClusterBootstrapInputs,
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: int = 42,
    chunk_size: int = 64,
) -> list[dict[str, object]]:
    """Return 3 pairs x 4 metrics using ratio-of-totals estimates.

    Sampling is paired across all roles and is performed in bounded chunks.
    A sampled cluster retains all clean/SNR/noise observations belonging to its
    source utterance.
    """

    if n_bootstrap < 1:
        raise ClusterBootstrapError("n_bootstrap must be at least 1")
    if not 0.0 < ci_level < 1.0:
        raise ClusterBootstrapError("ci_level must be strictly between 0 and 1")
    if bootstrap_seed < 0:
        raise ClusterBootstrapError("bootstrap_seed must be non-negative")
    if chunk_size < 1:
        raise ClusterBootstrapError("chunk_size must be at least 1")
    if set(inputs.runs) != set(COMPARISON_ROLES):
        raise ClusterBootstrapError("inputs must contain exactly the three comparison roles")

    cluster_ids, contributions = _cluster_contributions(inputs)
    if not cluster_ids:
        raise ClusterBootstrapError("paired cluster bootstrap requires at least one cluster")
    point: dict[str, dict[str, tuple[int, int, int, int, float, float]]] = {}
    for role in COMPARISON_ROLES:
        point[role] = {}
        for metric in METRIC_FIELDS:
            contribution = contributions[role][metric]
            numerator = int(contribution.numerator.sum())
            denominator = int(contribution.denominator.sum())
            coverage_numerator = int(contribution.coverage_numerator.sum())
            coverage_denominator = int(contribution.coverage_denominator.sum())
            estimate = _safe_scalar_ratio(
                numerator, denominator, label=f"{role}/{metric}"
            )
            coverage = _safe_scalar_ratio(
                coverage_numerator,
                coverage_denominator,
                label=f"{role}/{metric} coverage",
            )
            if not 0.0 <= coverage <= 1.0 + 1e-12:
                raise ClusterBootstrapError(
                    f"{role}/{metric} coverage is outside [0, 1]: {coverage}"
                )
            point[role][metric] = (
                numerator,
                denominator,
                coverage_numerator,
                coverage_denominator,
                estimate,
                min(coverage, 1.0),
            )

    replicate_rates = {
        role: {
            metric: np.full(n_bootstrap, np.nan, dtype=np.float64)
            for metric in METRIC_FIELDS
        }
        for role in COMPARISON_ROLES
    }
    rng = np.random.default_rng(bootstrap_seed)
    n_clusters = len(cluster_ids)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sample_indices = rng.integers(
            0,
            n_clusters,
            size=(stop - start, n_clusters),
            dtype=np.int64,
        )
        for role in COMPARISON_ROLES:
            for metric in METRIC_FIELDS:
                contribution = contributions[role][metric]
                numerators = contribution.numerator[sample_indices].sum(axis=1)
                denominators = contribution.denominator[sample_indices].sum(axis=1)
                np.divide(
                    numerators,
                    denominators,
                    out=replicate_rates[role][metric][start:stop],
                    where=denominators > 0,
                )

    alpha = (1.0 - ci_level) / 2.0
    rows: list[dict[str, object]] = []
    for pair_id, role_a, role_b in PAIR_SPECS:
        run_a = inputs.runs[role_a]
        run_b = inputs.runs[role_b]
        config_a = run_a.configuration
        config_b = run_b.configuration
        for metric in METRIC_FIELDS:
            a = point[role_a][metric]
            b = point[role_b][metric]
            deltas = replicate_rates[role_b][metric] - replicate_rates[role_a][metric]
            valid_deltas = deltas[np.isfinite(deltas)]
            if valid_deltas.size == 0:
                raise ClusterBootstrapError(
                    f"{pair_id}/{metric} has no valid bootstrap replicates"
                )
            ci_lower, ci_upper = np.quantile(
                valid_deltas,
                [alpha, 1.0 - alpha],
                method="linear",
            )
            delta = b[4] - a[4]
            rows.append(
                {
                    "metric_version": METRIC_VERSION,
                    "decision_sha256": inputs.decision_sha256,
                    "benchmark_sha256": inputs.benchmark_sha256,
                    "comparison_set_sha256": inputs.comparison_set_sha256,
                    "dataset": inputs.dataset,
                    "split": inputs.split,
                    "pair_id": pair_id,
                    "role_a": role_a,
                    "configuration_id_a": config_a.configuration_id,
                    "method_id_a": config_a.method_id,
                    "train_type_a": config_a.train_type,
                    "lambda_a": config_a.lambda_value,
                    "seed_a": config_a.seed,
                    "prediction_sha256_a": run_a.prediction_sha256,
                    "role_b": role_b,
                    "configuration_id_b": config_b.configuration_id,
                    "method_id_b": config_b.method_id,
                    "train_type_b": config_b.train_type,
                    "lambda_b": config_b.lambda_value,
                    "seed_b": config_b.seed,
                    "prediction_sha256_b": run_b.prediction_sha256,
                    "n_source_clusters": n_clusters,
                    "n_paired_conditions": len(inputs.observations),
                    "metric": metric,
                    "numerator_a": a[0],
                    "denominator_a": a[1],
                    "coverage_numerator_a": a[2],
                    "coverage_denominator_a": a[3],
                    "coverage_a": _format_float(a[5]),
                    "estimate_a": _format_float(a[4]),
                    "numerator_b": b[0],
                    "denominator_b": b[1],
                    "coverage_numerator_b": b[2],
                    "coverage_denominator_b": b[3],
                    "coverage_b": _format_float(b[5]),
                    "estimate_b": _format_float(b[4]),
                    "delta_b_minus_a": _format_float(delta),
                    "n_bootstrap": n_bootstrap,
                    "n_valid_bootstrap": int(valid_deltas.size),
                    "ci_level": _format_float(ci_level),
                    "ci_lower": _format_float(float(ci_lower)),
                    "ci_upper": _format_float(float(ci_upper)),
                    "ci_excludes_zero": str(
                        float(ci_upper) < 0.0 or float(ci_lower) > 0.0
                    ).lower(),
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_unit": inputs.bootstrap_unit,
                    "ci_method": "paired_cluster_percentile_ratio_of_totals",
                }
            )
    return rows


def write_cluster_bootstrap_csv(
    output_path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    overwrite: bool = False,
    protected_inputs: Sequence[str | Path] = (),
) -> Path:
    """Atomically write output, refusing overwrite unless explicitly allowed."""

    destination = Path(output_path)
    protected = {Path(path).resolve() for path in protected_inputs}
    if destination.resolve() in protected:
        raise ClusterBootstrapError("refusing to overwrite an input artifact")
    if destination.exists() and not overwrite:
        raise ClusterBootstrapError(
            f"output already exists: {destination}; use overwrite=True explicitly"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(OUTPUT_COLUMNS),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        if not overwrite and destination.exists():
            raise ClusterBootstrapError(f"output appeared while writing: {destination}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _cluster_bootstrap_csv_bytes(
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(OUTPUT_COLUMNS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_formal_cluster_bootstrap_bundle(
    output_path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    protected_inputs: Sequence[str | Path],
    formal_protocol: Mapping[str, Any],
    parameters: Mapping[str, Any],
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    """Commit a formal bootstrap CSV and its transitive provenance together."""

    destination = Path(output_path)
    provenance_path = destination.with_suffix(
        destination.suffix + ".provenance.json"
    )
    protected = {Path(path).resolve() for path in protected_inputs}
    collisions = [
        path
        for path in (destination, provenance_path)
        if path.resolve() in protected
    ]
    if collisions:
        raise ClusterBootstrapError(
            "refusing to overwrite an input artifact: "
            + ", ".join(str(path) for path in collisions)
        )
    try:
        bindings = bind_input_files(protected_inputs, root=ROOT)
    except (FileNotFoundError, OSError) as exc:
        raise ClusterBootstrapError(
            f"cannot bind formal bootstrap inputs: {exc}"
        ) from exc
    commit_artifact_bundle(
        bundle_name="cluster_bootstrap",
        bundle_version=BOOTSTRAP_BUNDLE_VERSION,
        data_destinations={"bootstrap_results": destination},
        data_contents={"bootstrap_results": _cluster_bootstrap_csv_bytes(rows)},
        provenance_path=provenance_path,
        input_bindings=bindings,
        parameters={
            **dict(parameters),
            "formal_protocol": dict(formal_protocol),
        },
        overwrite=overwrite,
        resume=resume,
        error_type=ClusterBootstrapError,
    )
    return destination


def run_cluster_bootstrap(
    decision_path: str | Path,
    benchmark_path: str | Path,
    prediction_paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    comparison_set: Mapping[str, str] | None = None,
    cluster_unit: str = "source_utt_id",
    formal_paper_v2: bool = False,
    split_lock_path: str | Path | None = None,
    final_benchmark_lock_path: str | Path | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: int = 42,
    chunk_size: int = 64,
    overwrite: bool = False,
    resume: bool = False,
) -> tuple[Path, list[dict[str, object]]]:
    destination = Path(output_path)
    if overwrite and resume:
        raise ClusterBootstrapError("overwrite and resume are mutually exclusive")
    if resume and not formal_paper_v2:
        raise ClusterBootstrapError("resume is available only in formal paper-v2 mode")
    if destination.exists() and not (overwrite or resume):
        raise ClusterBootstrapError(
            f"output already exists: {destination}; use overwrite=True explicitly"
        )
    inputs = load_cluster_bootstrap_inputs(
        decision_path,
        benchmark_path,
        prediction_paths,
        comparison_set=comparison_set,
        cluster_unit=cluster_unit,
        formal_paper_v2=formal_paper_v2,
        split_lock_path=split_lock_path,
        final_benchmark_lock_path=final_benchmark_lock_path,
    )
    rows = build_cluster_bootstrap_rows(
        inputs,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        bootstrap_seed=bootstrap_seed,
        chunk_size=chunk_size,
    )
    protected = [decision_path, benchmark_path, *prediction_paths.values()]
    if formal_paper_v2:
        if inputs.formal_evidence is None or split_lock_path is None:
            raise ClusterBootstrapError(
                "formal bootstrap lost its verified prediction evidence"
            )
        try:
            current_evidence = verify_formal_prediction_set(
                list(prediction_paths.values()),
                benchmark_path=benchmark_path,
                split_lock_path=split_lock_path,
                decision_path=decision_path,
                final_benchmark_lock_path=final_benchmark_lock_path,
                required_configuration_ids={
                    path: configuration_id
                    for configuration_id, path in prediction_paths.items()
                },
            )
        except PredictionEvidenceError as exc:
            raise ClusterBootstrapError(str(exc)) from exc
        initial_protocol = formal_protocol_parameters(inputs.formal_evidence)
        current_protocol = formal_protocol_parameters(current_evidence)
        if current_protocol != initial_protocol:
            raise ClusterBootstrapError(
                "formal prediction evidence changed while bootstrap was running"
            )
        protected.extend(
            [
                split_lock_path,
                *(item.provenance_path for item in current_evidence.predictions),
            ]
        )
        if final_benchmark_lock_path is not None:
            protected.append(final_benchmark_lock_path)
        return (
            write_formal_cluster_bootstrap_bundle(
                destination,
                rows,
                protected_inputs=protected,
                formal_protocol=current_protocol,
                parameters={
                    "metric_version": EXPECTED_METRIC_VERSION,
                    "comparison_set_sha256": inputs.comparison_set_sha256,
                    "cluster_unit": cluster_unit,
                    "n_bootstrap": n_bootstrap,
                    "ci_level": ci_level,
                    "bootstrap_seed": bootstrap_seed,
                    "chunk_size": chunk_size,
                    "output_rows": len(rows),
                },
                overwrite=overwrite,
                resume=resume,
            ),
            rows,
        )
    return (
        write_cluster_bootstrap_csv(
            destination,
            rows,
            overwrite=overwrite,
            protected_inputs=protected,
        ),
        rows,
    )


__all__ = [
    "BOOTSTRAP_BUNDLE_VERSION",
    "CLUSTER_UNITS",
    "COMPARISON_ROLES",
    "EXPECTED_METRIC_VERSION",
    "OUTPUT_COLUMNS",
    "PAIR_SPECS",
    "BenchmarkObservation",
    "ClusterBootstrapError",
    "ClusterBootstrapInputs",
    "ComparisonRun",
    "LockedConfiguration",
    "build_cluster_bootstrap_rows",
    "load_cluster_bootstrap_inputs",
    "run_cluster_bootstrap",
    "sha256_file",
    "write_cluster_bootstrap_csv",
    "write_formal_cluster_bootstrap_bundle",
]
