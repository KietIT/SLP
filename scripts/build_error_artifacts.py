from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import unicodedata
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from scripts.error_analysis import EVENT_COLUMNS  # noqa: E402
from src.vitonesr.analysis import (  # noqa: E402
    FINAL_CODAS,
    METRIC_VERSION,
    RUN_METADATA_COLUMNS,
    SHORT_WORDS,
    TONE_LABELS,
)
from src.vitonesr.artifact_bundle import (  # noqa: E402
    bind_input_files,
    commit_artifact_bundle,
    verify_input_bindings,
)
from src.vitonesr.prediction_evidence import (  # noqa: E402
    PredictionEvidenceError,
    verify_formal_error_events,
)
from src.vitonesr.text_norm import normalize_vi_text  # noqa: E402


RUN_COLUMNS = list(RUN_METADATA_COLUMNS)
TONE_ORDER = tuple(TONE_LABELS)
TONE_DISPLAY = {
    "ngang": "Ngang",
    "sac": "Sắc",
    "huyen": "Huyền",
    "hoi": "Hỏi",
    "nga": "Ngã",
    "nang": "Nặng",
}
SCOPE_ORDER = ("overall", "low_snr")
SCOPE_DISPLAY = {
    "overall": "Overall",
    "low_snr": "Low SNR",
}
VALID_OPERATIONS = {"match", "substitution", "deletion", "insertion"}
PAIRED_OPERATIONS = {"match", "substitution"}

TONE_MATRIX_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "group_type",
    "group_value",
    "ref_tone",
    "hyp_tone",
    "count",
    "ref_total",
    "row_rate",
]
TONE_CSV_NAME = "tone_confusion_matrix.csv"
TONE_PNG_NAME = "tone_confusion_matrix.png"
OUTPUT_LOCK_NAME = ".tone_confusion_matrix.lock"
CODA_NONE = "none"
CODA_ORDER = (CODA_NONE, "n", "ng", "nh", "t", "c", "ch", "m", "p")
CODA_DISPLAY = {label: ("∅" if label == CODA_NONE else label) for label in CODA_ORDER}
CODA_MATRIX_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "group_type",
    "group_value",
    "ref_coda",
    "hyp_coda",
    "count",
    "ref_total",
    "row_rate",
]
CODA_CSV_NAME = "final_coda_confusion_matrix.csv"
CODA_PNG_NAME = "final_coda_confusion_matrix.png"
CODA_OUTPUT_LOCK_NAME = ".final_coda_confusion_matrix.lock"
SHORT_WORD_ORDER = ("đã", "có", "là", "một", "và")
SHORT_WORD_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "utt_id",
    "snr",
    "noise_type",
    "low_snr_scope",
    "deleted_word",
    "ref_index",
    "context_window",
    "left_context",
    "right_context",
    "context",
    "ref",
    "hyp",
]
SHORT_WORD_CSV_NAME = "short_word_deletion_examples.csv"
SHORT_WORD_OUTPUT_LOCK_NAME = ".short_word_deletion_examples.lock"
ERROR_ARTIFACT_BUNDLE_VERSION = "error_artifacts_aligned_v1_bundle_v1"

RunKey = tuple[str, ...]
MatrixKey = tuple[RunKey, str]


class ErrorArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class ToneAggregation:
    run_keys: tuple[RunKey, ...]
    counts: Mapping[MatrixKey, Counter[tuple[str, str]]]
    eligible_deletions: Mapping[MatrixKey, int]
    utterance_ids: Mapping[MatrixKey, frozenset[str]]
    event_rows: int
    low_snrs: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    csv_path: Path
    png_path: Path
    matrix_rows: int
    aggregation: ToneAggregation
    candidate_runs: tuple[tuple[str, RunKey], ...]


@dataclass(frozen=True)
class CodaAggregation:
    run_keys: tuple[RunKey, ...]
    counts: Mapping[MatrixKey, Counter[tuple[str, str]]]
    word_deletions: Mapping[MatrixKey, int]
    utterance_ids: Mapping[MatrixKey, frozenset[str]]
    event_rows: int
    low_snrs: tuple[str, ...]


@dataclass(frozen=True)
class CodaBuildResult:
    csv_path: Path
    png_path: Path
    matrix_rows: int
    aggregation: CodaAggregation
    candidate_runs: tuple[tuple[str, RunKey], ...]


@dataclass(frozen=True)
class ShortWordAggregation:
    run_keys: tuple[RunKey, ...]
    examples: tuple[dict[str, object], ...]
    deletion_counts: Mapping[MatrixKey, int]
    reference_units: Mapping[MatrixKey, int]
    word_counts: Mapping[MatrixKey, Counter[str]]
    event_rows: int
    low_snrs: tuple[str, ...]


@dataclass(frozen=True)
class ShortWordBuildResult:
    csv_path: Path
    example_rows: int
    aggregation: ShortWordAggregation
    candidate_runs: tuple[tuple[str, RunKey], ...]


def _sort_text(value: object) -> str:
    return str(value).casefold()


def _run_sort_key(run_key: RunKey) -> tuple[str, ...]:
    return tuple(_sort_text(value) for value in run_key)


def _canonical_decimal(
    value: object,
    *,
    field: str,
    non_negative: bool = True,
) -> str:
    text = str(value).strip()
    if not text:
        raise ErrorArtifactError(f"{field} must not be empty")
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ErrorArtifactError(f"{field} must be a decimal number, found {value!r}") from exc
    if not parsed.is_finite():
        raise ErrorArtifactError(f"{field} must be finite, found {value!r}")
    if non_negative and parsed < 0:
        raise ErrorArtifactError(f"{field} must be non-negative, found {value!r}")
    if parsed == 0:
        return "0"
    return format(parsed.normalize(), "f")


def canonical_candidate_lambdas(values: Sequence[object]) -> tuple[str, ...]:
    canonical = tuple(
        _canonical_decimal(value, field="candidate lambda") for value in values
    )
    if len(canonical) != 2:
        raise ErrorArtifactError(
            "Gate D requires exactly two --candidate-lambda values"
        )
    if len(set(canonical)) != len(canonical):
        raise ErrorArtifactError("candidate lambda values must be unique")
    return canonical


def canonical_low_snrs(values: Sequence[object]) -> tuple[str, ...]:
    canonical = tuple(
        _canonical_decimal(value, field="low SNR", non_negative=False)
        for value in values
    )
    if not canonical:
        raise ErrorArtifactError("at least one --low-snr value is required")
    if len(set(canonical)) != len(canonical):
        raise ErrorArtifactError("low SNR values must be unique")
    return canonical


def canonical_focus_runs(values: Sequence[object]) -> tuple[tuple[str, str], ...]:
    selectors: list[tuple[str, str]] = []
    for value in values:
        text = str(value).strip()
        if ":" not in text:
            raise ErrorArtifactError(
                "focus run must use TRAIN_TYPE:LAMBDA, for example ordinary_lora:0"
            )
        train_type, lambda_value = (part.strip() for part in text.split(":", 1))
        if not train_type or not lambda_value:
            raise ErrorArtifactError(
                "focus run must include non-empty TRAIN_TYPE and LAMBDA"
            )
        selectors.append(
            (
                train_type,
                _canonical_decimal(lambda_value, field="focus run lambda"),
            )
        )
    if not selectors:
        raise ErrorArtifactError("at least one --focus-run value is required")
    if len(set(selectors)) != len(selectors):
        raise ErrorArtifactError("focus run selectors must be unique")
    return tuple(selectors)


def _canonical_scopes(
    low_snrs: Sequence[object],
    *,
    overall_only: bool,
) -> tuple[str, ...]:
    if overall_only:
        if low_snrs:
            raise ErrorArtifactError(
                "--overall-only cannot be combined with low-SNR values"
            )
        return ()
    return canonical_low_snrs(low_snrs)


def _scope_order(low_snrs: Sequence[str]) -> tuple[str, ...]:
    return SCOPE_ORDER if low_snrs else ("overall",)


def _event_snr(value: object, *, path: Path, row_number: int) -> str:
    text = str(value).strip()
    if text.casefold() == "clean":
        return "clean"
    try:
        return _canonical_decimal(
            text,
            field="event SNR",
            non_negative=False,
        )
    except ErrorArtifactError as exc:
        raise ErrorArtifactError(f"{path}: row {row_number}: {exc}") from exc


def _parse_bool(value: object, *, path: Path, row_number: int, field: str) -> bool:
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ErrorArtifactError(
        f"{path}: row {row_number}: {field} must be true or false, found {value!r}"
    )


def _scope_keys(snr: str, low_snrs: frozenset[str]) -> tuple[str, ...]:
    if snr in low_snrs:
        return SCOPE_ORDER
    return ("overall",)


def load_tone_aggregation(
    event_path: str | Path,
    *,
    low_snrs: Sequence[str],
    overall_only: bool = False,
) -> ToneAggregation:
    path = Path(event_path)
    if not path.is_file():
        raise ErrorArtifactError(f"event CSV does not exist: {path}")

    low_snr_values = _canonical_scopes(low_snrs, overall_only=overall_only)
    low_snr_set = frozenset(low_snr_values)
    counts: defaultdict[MatrixKey, Counter[tuple[str, str]]] = defaultdict(Counter)
    eligible_deletions: Counter[MatrixKey] = Counter()
    utterance_ids: defaultdict[MatrixKey, set[str]] = defaultdict(set)
    run_keys: set[RunKey] = set()
    observed_snrs_by_run: defaultdict[RunKey, set[str]] = defaultdict(set)
    event_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if columns != EVENT_COLUMNS:
            raise ErrorArtifactError(
                f"{path}: expected exact Gate C event columns {EVENT_COLUMNS}, found {columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            event_rows += 1
            if None in row or any(row.get(column) is None for column in EVENT_COLUMNS):
                raise ErrorArtifactError(
                    f"{path}: row {row_number} has missing or extra CSV cells"
                )
            if row["metric_version"] != METRIC_VERSION:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: expected metric_version={METRIC_VERSION!r}, "
                    f"found {row['metric_version']!r}"
                )

            run_key = tuple(row[column] for column in RUN_COLUMNS)
            for column, value in zip(RUN_COLUMNS, run_key):
                if column != "lambda" and not value:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: run metadata {column!r} is empty"
                    )
            run_keys.add(run_key)

            operation = row["operation"]
            if operation not in VALID_OPERATIONS:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unknown operation {operation!r}"
                )
            ref_tone = row["ref_tone"]
            hyp_tone = row["hyp_tone"]
            if ref_tone and ref_tone not in TONE_ORDER:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unknown ref_tone {ref_tone!r}"
                )
            if hyp_tone and hyp_tone not in TONE_ORDER:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unknown hyp_tone {hyp_tone!r}"
                )

            tone_eligible = _parse_bool(
                row["tone_eligible"],
                path=path,
                row_number=row_number,
                field="tone_eligible",
            )
            tone_error = _parse_bool(
                row["tone_error"],
                path=path,
                row_number=row_number,
                field="tone_error",
            )
            if tone_error and not tone_eligible:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: tone_error=true requires tone_eligible=true"
                )

            snr = _event_snr(row["snr"], path=path, row_number=row_number)
            observed_snrs_by_run[run_key].add(snr)
            scopes = _scope_keys(snr, low_snr_set)
            utt_id = row["utt_id"]
            if not utt_id:
                raise ErrorArtifactError(f"{path}: row {row_number}: utt_id is empty")
            for scope in scopes:
                utterance_ids[(run_key, scope)].add(utt_id)

            if not tone_eligible:
                continue
            if operation == "deletion":
                if ref_tone not in TONE_ORDER or hyp_tone:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: eligible tone deletion must have a "
                        "valid ref_tone and empty hyp_tone"
                    )
                if not tone_error:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: eligible tone deletion must be an error"
                    )
                for scope in scopes:
                    eligible_deletions[(run_key, scope)] += 1
                continue
            if operation not in PAIRED_OPERATIONS:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: tone-eligible operation must be paired or "
                    f"a deletion, found {operation!r}"
                )
            if ref_tone not in TONE_ORDER or hyp_tone not in TONE_ORDER:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: eligible paired event requires valid "
                    "ref_tone and hyp_tone"
                )
            if tone_error != (ref_tone != hyp_tone):
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: tone_error disagrees with the tone pair"
                )
            for scope in scopes:
                counts[(run_key, scope)][(ref_tone, hyp_tone)] += 1

    if event_rows == 0:
        raise ErrorArtifactError(f"event CSV is empty: {path}")
    if not run_keys:
        raise ErrorArtifactError(f"event CSV contains no run metadata: {path}")
    missing_low_snrs: list[str] = []
    for run_key in sorted(run_keys, key=_run_sort_key):
        missing = [
            value
            for value in low_snr_values
            if value not in observed_snrs_by_run[run_key]
        ]
        if missing:
            metadata = ", ".join(
                f"{column}={value!r}"
                for column, value in zip(RUN_COLUMNS, run_key)
            )
            missing_low_snrs.append(f"[{metadata}] missing {','.join(missing)}")
    if missing_low_snrs:
        raise ErrorArtifactError(
            "requested low SNR values are absent for one or more runs: "
            + "; ".join(missing_low_snrs)
        )

    ordered_runs = tuple(sorted(run_keys, key=_run_sort_key))
    frozen_utterances = {
        key: frozenset(value) for key, value in utterance_ids.items()
    }
    return ToneAggregation(
        run_keys=ordered_runs,
        counts=dict(counts),
        eligible_deletions=dict(eligible_deletions),
        utterance_ids=frozen_utterances,
        event_rows=event_rows,
        low_snrs=low_snr_values,
    )


def load_coda_aggregation(
    event_path: str | Path,
    *,
    low_snrs: Sequence[str],
    overall_only: bool = False,
) -> CodaAggregation:
    path = Path(event_path)
    if not path.is_file():
        raise ErrorArtifactError(f"event CSV does not exist: {path}")

    low_snr_values = _canonical_scopes(low_snrs, overall_only=overall_only)
    low_snr_set = frozenset(low_snr_values)
    valid_codas = frozenset(FINAL_CODAS)
    if valid_codas != frozenset(CODA_ORDER[1:]):
        raise ErrorArtifactError("Gate E coda classes disagree with the analysis contract")

    counts: defaultdict[MatrixKey, Counter[tuple[str, str]]] = defaultdict(Counter)
    word_deletions: Counter[MatrixKey] = Counter()
    utterance_ids: defaultdict[MatrixKey, set[str]] = defaultdict(set)
    run_keys: set[RunKey] = set()
    observed_snrs_by_run: defaultdict[RunKey, set[str]] = defaultdict(set)
    event_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if columns != EVENT_COLUMNS:
            raise ErrorArtifactError(
                f"{path}: expected exact Gate C event columns {EVENT_COLUMNS}, found {columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            event_rows += 1
            if None in row or any(row.get(column) is None for column in EVENT_COLUMNS):
                raise ErrorArtifactError(
                    f"{path}: row {row_number} has missing or extra CSV cells"
                )
            if row["metric_version"] != METRIC_VERSION:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: expected metric_version={METRIC_VERSION!r}, "
                    f"found {row['metric_version']!r}"
                )

            run_key = tuple(row[column] for column in RUN_COLUMNS)
            for column, value in zip(RUN_COLUMNS, run_key):
                if column != "lambda" and not value:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: run metadata {column!r} is empty"
                    )
            run_keys.add(run_key)

            operation = row["operation"]
            if operation not in VALID_OPERATIONS:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unknown operation {operation!r}"
                )
            ref_coda = row["ref_coda"]
            hyp_coda = row["hyp_coda"]
            if ref_coda and ref_coda not in valid_codas:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unknown ref_coda {ref_coda!r}"
                )
            if hyp_coda and hyp_coda not in valid_codas:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unknown hyp_coda {hyp_coda!r}"
                )

            final_eligible = _parse_bool(
                row["final_consonant_eligible"],
                path=path,
                row_number=row_number,
                field="final_consonant_eligible",
            )
            final_error = _parse_bool(
                row["final_consonant_error"],
                path=path,
                row_number=row_number,
                field="final_consonant_error",
            )
            if final_error and not final_eligible:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: final_consonant_error=true requires "
                    "final_consonant_eligible=true"
                )

            if operation == "insertion":
                if ref_coda:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: insertion must have empty ref_coda"
                    )
                if final_eligible:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: an insertion cannot be "
                        "final-coda eligible"
                    )
                expected_eligible = False
            elif operation == "deletion":
                if hyp_coda:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: deletion must have empty hyp_coda"
                    )
                expected_eligible = bool(ref_coda)
            else:
                expected_eligible = bool(ref_coda or hyp_coda)
                if final_eligible and not expected_eligible:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: eligible paired coda event requires "
                        "at least one coda"
                    )
            if final_eligible != expected_eligible:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: final_consonant_eligible disagrees "
                    "with the operation and coda pair"
                )

            ref_class = ref_coda or CODA_NONE
            hyp_class = hyp_coda or CODA_NONE
            expected_error = final_eligible and ref_class != hyp_class
            if final_error != expected_error:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: final_consonant_error disagrees with "
                    "the coda pair"
                )

            snr = _event_snr(row["snr"], path=path, row_number=row_number)
            observed_snrs_by_run[run_key].add(snr)
            scopes = _scope_keys(snr, low_snr_set)
            utt_id = row["utt_id"]
            if not utt_id:
                raise ErrorArtifactError(f"{path}: row {row_number}: utt_id is empty")
            for scope in scopes:
                utterance_ids[(run_key, scope)].add(utt_id)

            if not final_eligible:
                continue
            if operation == "deletion":
                if ref_coda not in valid_codas or hyp_coda:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: eligible coda deletion must have a "
                        "valid ref_coda and empty hyp_coda"
                    )
                if not final_error:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: eligible coda deletion must be an error"
                    )
            elif operation in PAIRED_OPERATIONS:
                if not ref_coda and not hyp_coda:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: eligible paired coda event requires "
                        "at least one coda"
                    )
                if operation == "match" and ref_coda != hyp_coda:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: a match cannot change final coda"
                    )
            else:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unsupported eligible coda operation "
                    f"{operation!r}"
                )

            for scope in scopes:
                counts[(run_key, scope)][(ref_class, hyp_class)] += 1
                if operation == "deletion":
                    word_deletions[(run_key, scope)] += 1

    if event_rows == 0:
        raise ErrorArtifactError(f"event CSV is empty: {path}")
    if not run_keys:
        raise ErrorArtifactError(f"event CSV contains no run metadata: {path}")

    missing_low_snrs: list[str] = []
    for run_key in sorted(run_keys, key=_run_sort_key):
        missing = [
            value
            for value in low_snr_values
            if value not in observed_snrs_by_run[run_key]
        ]
        if missing:
            metadata = ", ".join(
                f"{column}={value!r}" for column, value in zip(RUN_COLUMNS, run_key)
            )
            missing_low_snrs.append(f"[{metadata}] missing {','.join(missing)}")
    if missing_low_snrs:
        raise ErrorArtifactError(
            "requested low SNR values are absent for one or more runs: "
            + "; ".join(missing_low_snrs)
        )

    ordered_runs = tuple(sorted(run_keys, key=_run_sort_key))
    frozen_utterances = {
        key: frozenset(value) for key, value in utterance_ids.items()
    }
    return CodaAggregation(
        run_keys=ordered_runs,
        counts=dict(counts),
        word_deletions=dict(word_deletions),
        utterance_ids=frozen_utterances,
        event_rows=event_rows,
        low_snrs=low_snr_values,
    )


def _short_example_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    run_key = tuple(str(row[column]) for column in RUN_COLUMNS)
    return (
        *_run_sort_key(run_key),
        str(row["utt_id"]).casefold(),
        int(row["ref_index"]),
        str(row["deleted_word"]).casefold(),
        str(row["ref"]),
        str(row["hyp"]),
    )


def load_short_word_aggregation(
    event_path: str | Path,
    *,
    low_snrs: Sequence[str],
    context_window: int,
    overall_only: bool = False,
) -> ShortWordAggregation:
    path = Path(event_path)
    if not path.is_file():
        raise ErrorArtifactError(f"event CSV does not exist: {path}")
    if isinstance(context_window, bool) or not isinstance(context_window, int):
        raise ErrorArtifactError("context window must be an integer")
    if context_window < 0:
        raise ErrorArtifactError("context window must be non-negative")
    if frozenset(SHORT_WORD_ORDER) != SHORT_WORDS:
        raise ErrorArtifactError(
            "Gate F short-word display order disagrees with the analysis contract"
        )

    low_snr_values = _canonical_scopes(low_snrs, overall_only=overall_only)
    low_snr_set = frozenset(low_snr_values)
    deletion_counts: Counter[MatrixKey] = Counter()
    reference_units: Counter[MatrixKey] = Counter()
    word_counts: defaultdict[MatrixKey, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, object]] = []
    selected_keys: set[tuple[object, ...]] = set()
    run_keys: set[RunKey] = set()
    observed_snrs_by_run: defaultdict[RunKey, set[str]] = defaultdict(set)
    event_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if columns != EVENT_COLUMNS:
            raise ErrorArtifactError(
                f"{path}: expected exact Gate C event columns {EVENT_COLUMNS}, found {columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            event_rows += 1
            if None in row or any(row.get(column) is None for column in EVENT_COLUMNS):
                raise ErrorArtifactError(
                    f"{path}: row {row_number} has missing or extra CSV cells"
                )
            if row["metric_version"] != METRIC_VERSION:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: expected metric_version={METRIC_VERSION!r}, "
                    f"found {row['metric_version']!r}"
                )

            run_key = tuple(row[column] for column in RUN_COLUMNS)
            for column, value in zip(RUN_COLUMNS, run_key):
                if column != "lambda" and not value:
                    raise ErrorArtifactError(
                        f"{path}: row {row_number}: run metadata {column!r} is empty"
                    )
            run_keys.add(run_key)

            operation = row["operation"]
            if operation not in VALID_OPERATIONS:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: unknown operation {operation!r}"
                )
            snr = _event_snr(row["snr"], path=path, row_number=row_number)
            observed_snrs_by_run[run_key].add(snr)
            scopes = _scope_keys(snr, low_snr_set)
            utt_id = row["utt_id"]
            if not utt_id:
                raise ErrorArtifactError(f"{path}: row {row_number}: utt_id is empty")
            if not row["noise_type"]:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: noise_type is empty"
                )

            ref_token = row["ref_token"]
            if ref_token in SHORT_WORDS:
                for scope in scopes:
                    reference_units[(run_key, scope)] += 1

            short_word_deletion = _parse_bool(
                row["short_word_deletion"],
                path=path,
                row_number=row_number,
                field="short_word_deletion",
            )
            expected_deletion = operation == "deletion" and ref_token in SHORT_WORDS
            if short_word_deletion != expected_deletion:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: short_word_deletion disagrees with "
                    "the operation and fixed lexicon"
                )
            if not short_word_deletion:
                continue

            if row["hyp_token"] or row["hyp_index"]:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: short-word deletion requires empty "
                    "hyp_token and hyp_index"
                )
            ref_index_text = row["ref_index"].strip()
            if not ref_index_text.isdigit():
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: short-word deletion requires a "
                    "non-negative integer ref_index"
                )
            ref_index = int(ref_index_text)
            ref_text = unicodedata.normalize("NFC", row["ref"])
            hyp_text = unicodedata.normalize("NFC", row["hyp"])
            normalized_tokens = normalize_vi_text(ref_text).split()
            if ref_index >= len(normalized_tokens):
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: ref_index {ref_index} is outside the "
                    f"normalized reference with {len(normalized_tokens)} tokens"
                )
            if normalized_tokens[ref_index] != ref_token:
                raise ErrorArtifactError(
                    f"{path}: row {row_number}: ref_index resolves to "
                    f"{normalized_tokens[ref_index]!r}, not ref_token={ref_token!r}"
                )

            unique_key = (*run_key, utt_id, ref_index)
            if unique_key in selected_keys:
                raise ErrorArtifactError(
                    f"{path}: duplicate short-word deletion key {unique_key}"
                )
            selected_keys.add(unique_key)

            left_tokens = normalized_tokens[
                max(0, ref_index - context_window) : ref_index
            ]
            right_tokens = normalized_tokens[
                ref_index + 1 : ref_index + 1 + context_window
            ]
            left_context = " ".join(left_tokens)
            right_context = " ".join(right_tokens)
            context = " ".join([*left_tokens, f"⟦{ref_token}⟧", *right_tokens])
            is_low_snr = snr in low_snr_set
            examples.append(
                {
                    "metric_version": METRIC_VERSION,
                    **dict(zip(RUN_COLUMNS, run_key)),
                    "utt_id": utt_id,
                    "snr": snr,
                    "noise_type": row["noise_type"],
                    "low_snr_scope": "true" if is_low_snr else "false",
                    "deleted_word": ref_token,
                    "ref_index": ref_index,
                    "context_window": context_window,
                    "left_context": left_context,
                    "right_context": right_context,
                    "context": context,
                    "ref": ref_text,
                    "hyp": hyp_text,
                }
            )
            for scope in scopes:
                deletion_counts[(run_key, scope)] += 1
                word_counts[(run_key, scope)][ref_token] += 1

    if event_rows == 0:
        raise ErrorArtifactError(f"event CSV is empty: {path}")
    if not run_keys:
        raise ErrorArtifactError(f"event CSV contains no run metadata: {path}")

    missing_low_snrs: list[str] = []
    for run_key in sorted(run_keys, key=_run_sort_key):
        missing = [
            value
            for value in low_snr_values
            if value not in observed_snrs_by_run[run_key]
        ]
        if missing:
            metadata = ", ".join(
                f"{column}={value!r}" for column, value in zip(RUN_COLUMNS, run_key)
            )
            missing_low_snrs.append(f"[{metadata}] missing {','.join(missing)}")
    if missing_low_snrs:
        raise ErrorArtifactError(
            "requested low SNR values are absent for one or more runs: "
            + "; ".join(missing_low_snrs)
        )

    examples.sort(key=_short_example_sort_key)
    return ShortWordAggregation(
        run_keys=tuple(sorted(run_keys, key=_run_sort_key)),
        examples=tuple(examples),
        deletion_counts=dict(deletion_counts),
        reference_units=dict(reference_units),
        word_counts=dict(word_counts),
        event_rows=event_rows,
        low_snrs=low_snr_values,
    )


def build_tone_matrix_rows(
    aggregation: ToneAggregation,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_key in aggregation.run_keys:
        metadata = dict(zip(RUN_COLUMNS, run_key))
        for scope in _scope_order(aggregation.low_snrs):
            matrix = aggregation.counts.get((run_key, scope), Counter())
            for ref_tone in TONE_ORDER:
                ref_total = sum(matrix[(ref_tone, hyp_tone)] for hyp_tone in TONE_ORDER)
                for hyp_tone in TONE_ORDER:
                    count = matrix[(ref_tone, hyp_tone)]
                    row_rate = count / ref_total if ref_total else 0.0
                    rows.append(
                        {
                            "metric_version": METRIC_VERSION,
                            **metadata,
                            "group_type": "scope",
                            "group_value": scope,
                            "ref_tone": ref_tone,
                            "hyp_tone": hyp_tone,
                            "count": count,
                            "ref_total": ref_total,
                            "row_rate": f"{row_rate:.12f}",
                        }
                    )
    return rows


def build_coda_matrix_rows(
    aggregation: CodaAggregation,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_key in aggregation.run_keys:
        metadata = dict(zip(RUN_COLUMNS, run_key))
        for scope in _scope_order(aggregation.low_snrs):
            matrix = aggregation.counts.get((run_key, scope), Counter())
            for ref_coda in CODA_ORDER:
                ref_total = sum(matrix[(ref_coda, hyp_coda)] for hyp_coda in CODA_ORDER)
                for hyp_coda in CODA_ORDER:
                    count = matrix[(ref_coda, hyp_coda)]
                    row_rate = count / ref_total if ref_total else 0.0
                    rows.append(
                        {
                            "metric_version": METRIC_VERSION,
                            **metadata,
                            "group_type": "scope",
                            "group_value": scope,
                            "ref_coda": ref_coda,
                            "hyp_coda": hyp_coda,
                            "count": count,
                            "ref_total": ref_total,
                            "row_rate": f"{row_rate:.12f}",
                        }
                    )
    return rows


def resolve_candidate_runs(
    aggregation: ToneAggregation,
    candidate_lambdas: Sequence[str],
) -> tuple[tuple[str, RunKey], ...]:
    return _resolve_candidate_run_keys(aggregation.run_keys, candidate_lambdas)


def resolve_coda_candidate_runs(
    aggregation: CodaAggregation,
    candidate_lambdas: Sequence[str],
) -> tuple[tuple[str, RunKey], ...]:
    return _resolve_candidate_run_keys(aggregation.run_keys, candidate_lambdas)


def _resolve_candidate_run_keys(
    run_keys: Sequence[RunKey],
    candidate_lambdas: Sequence[str],
) -> tuple[tuple[str, RunKey], ...]:
    selected: list[tuple[str, RunKey]] = []
    for candidate in candidate_lambdas:
        matches: list[RunKey] = []
        for run_key in run_keys:
            metadata = dict(zip(RUN_COLUMNS, run_key))
            if metadata["train_type"] != "tone_aware_lora" or not metadata["lambda"]:
                continue
            try:
                observed = _canonical_decimal(
                    metadata["lambda"], field="event lambda"
                )
            except ErrorArtifactError as exc:
                raise ErrorArtifactError(
                    f"invalid lambda in run metadata {run_key}: {exc}"
                ) from exc
            if observed == candidate:
                matches.append(run_key)
        if len(matches) != 1:
            raise ErrorArtifactError(
                f"candidate lambda {candidate} must match exactly one tone_aware_lora run; "
                f"found {len(matches)}"
            )
        selected.append((candidate, matches[0]))
    return tuple(selected)


def _resolve_focus_run_keys(
    run_keys: Sequence[RunKey],
    focus_runs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, RunKey], ...]:
    selected: list[tuple[str, RunKey]] = []
    for train_type, lambda_value in focus_runs:
        matches: list[RunKey] = []
        for run_key in run_keys:
            metadata = dict(zip(RUN_COLUMNS, run_key))
            if metadata["train_type"] != train_type or not metadata["lambda"]:
                continue
            try:
                observed = _canonical_decimal(
                    metadata["lambda"], field="event lambda"
                )
            except ErrorArtifactError as exc:
                raise ErrorArtifactError(
                    f"invalid lambda in run metadata {run_key}: {exc}"
                ) from exc
            if observed == lambda_value:
                matches.append(run_key)
        label = f"{train_type}:{lambda_value}"
        if len(matches) != 1:
            raise ErrorArtifactError(
                f"focus run {label} must match exactly one run; found {len(matches)}"
            )
        selected.append((label, matches[0]))
    return tuple(selected)


def _resolve_report_runs(
    run_keys: Sequence[RunKey],
    *,
    candidate_lambdas: Sequence[object] | None,
    focus_runs: Sequence[object] | None,
) -> tuple[tuple[str, RunKey], ...]:
    if focus_runs:
        if candidate_lambdas:
            raise ErrorArtifactError(
                "use either --focus-run or --candidate-lambda, not both"
            )
        return _resolve_focus_run_keys(
            run_keys,
            canonical_focus_runs(focus_runs),
        )
    if not candidate_lambdas:
        raise ErrorArtifactError(
            "two --candidate-lambda values or explicit --focus-run values are required"
        )
    candidates = canonical_candidate_lambdas(candidate_lambdas)
    return _resolve_candidate_run_keys(run_keys, candidates)


def _focus_display(label: str) -> str:
    if ":" in label:
        train_type, lambda_value = label.split(":", 1)
        return f"{train_type} — λ={lambda_value}"
    return f"λ={label}"


def _focus_console(label: str) -> str:
    if ":" in label:
        return f"focus={label}"
    return f"candidate lambda={label}"


def _matrix_values(
    aggregation: ToneAggregation,
    run_key: RunKey,
    scope: str,
) -> tuple[list[list[int]], list[list[float]], int, int, int]:
    counter = aggregation.counts.get((run_key, scope), Counter())
    raw: list[list[int]] = []
    rates: list[list[float]] = []
    for ref_tone in TONE_ORDER:
        row = [counter[(ref_tone, hyp_tone)] for hyp_tone in TONE_ORDER]
        total = sum(row)
        raw.append(row)
        rates.append([value / total if total else 0.0 for value in row])
    matrix_total = sum(sum(row) for row in raw)
    off_diagonal = sum(
        raw[index][column]
        for index in range(len(TONE_ORDER))
        for column in range(len(TONE_ORDER))
        if index != column
    )
    deletions = int(aggregation.eligible_deletions.get((run_key, scope), 0))
    return raw, rates, matrix_total, off_diagonal, deletions


def render_tone_confusion_png(
    path: Path,
    aggregation: ToneAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
) -> None:
    labels = [TONE_DISPLAY[label] for label in TONE_ORDER]
    scopes = _scope_order(aggregation.low_snrs)
    figure, axes = plt.subplots(
        nrows=len(scopes),
        ncols=len(candidate_runs),
        squeeze=False,
        figsize=(max(7 * len(candidate_runs), 7), max(5 * len(scopes) + 1, 6)),
    )
    image = None
    low_label = "/".join(aggregation.low_snrs)

    for row_index, scope in enumerate(scopes):
        for column_index, (candidate, run_key) in enumerate(candidate_runs):
            axis = axes[row_index][column_index]
            raw, rates, matrix_total, off_diagonal, deletions = _matrix_values(
                aggregation, run_key, scope
            )
            image = axis.imshow(rates, cmap="Blues", vmin=0.0, vmax=1.0)
            axis.set_xticks(range(len(TONE_ORDER)), labels=labels, rotation=35, ha="right")
            axis.set_yticks(range(len(TONE_ORDER)), labels=labels)
            axis.set_xlabel("Thanh dự đoán")
            axis.set_ylabel("Thanh tham chiếu")
            scope_label = SCOPE_DISPLAY[scope]
            if scope == "low_snr":
                scope_label = f"{scope_label} ({low_label} dB)"
            axis.set_title(
                f"{_focus_display(candidate)} — {scope_label}\n"
                f"N={matrix_total:,}; nhầm={off_diagonal:,}; xóa loại trừ={deletions:,}",
                fontsize=11,
            )
            for ref_index in range(len(TONE_ORDER)):
                for hyp_index in range(len(TONE_ORDER)):
                    count = raw[ref_index][hyp_index]
                    rate = rates[ref_index][hyp_index]
                    annotation = "0" if count == 0 else f"{count}\n{rate:.1%}"
                    color = "white" if rate >= 0.5 else "black"
                    axis.text(
                        hyp_index,
                        ref_index,
                        annotation,
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=7,
                    )

    if image is None:
        plt.close(figure)
        raise ErrorArtifactError("cannot render a tone matrix without candidate runs")
    figure.suptitle(
        f"Ma trận nhầm lẫn 6 thanh tiếng Việt — {METRIC_VERSION}",
        fontsize=15,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.84,
        bottom=0.08,
        top=0.84,
        wspace=0.28,
        hspace=0.38,
    )
    colorbar_axis = figure.add_axes([0.89, 0.14, 0.018, 0.70])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Tỷ lệ theo thanh tham chiếu")
    try:
        figure.savefig(
            path,
            format="png",
            dpi=300,
            facecolor="white",
            metadata={"Software": f"VitoNESR {METRIC_VERSION}"},
        )
    finally:
        plt.close(figure)


def _coda_matrix_values(
    aggregation: CodaAggregation,
    run_key: RunKey,
    scope: str,
) -> tuple[list[list[int]], list[list[float]], int, int, int]:
    counter = aggregation.counts.get((run_key, scope), Counter())
    raw: list[list[int]] = []
    rates: list[list[float]] = []
    for ref_coda in CODA_ORDER:
        row = [counter[(ref_coda, hyp_coda)] for hyp_coda in CODA_ORDER]
        total = sum(row)
        raw.append(row)
        rates.append([value / total if total else 0.0 for value in row])
    matrix_total = sum(sum(row) for row in raw)
    off_diagonal = sum(
        raw[index][column]
        for index in range(len(CODA_ORDER))
        for column in range(len(CODA_ORDER))
        if index != column
    )
    word_deletions = int(aggregation.word_deletions.get((run_key, scope), 0))
    return raw, rates, matrix_total, off_diagonal, word_deletions


def render_coda_confusion_png(
    path: Path,
    aggregation: CodaAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
) -> None:
    labels = [CODA_DISPLAY[label] for label in CODA_ORDER]
    scopes = _scope_order(aggregation.low_snrs)
    figure, axes = plt.subplots(
        nrows=len(scopes),
        ncols=len(candidate_runs),
        squeeze=False,
        figsize=(
            max(7.5 * len(candidate_runs), 7.5),
            max(5.5 * len(scopes) + 1, 6.5),
        ),
    )
    image = None
    low_label = "/".join(aggregation.low_snrs)

    for row_index, scope in enumerate(scopes):
        for column_index, (candidate, run_key) in enumerate(candidate_runs):
            axis = axes[row_index][column_index]
            raw, rates, matrix_total, off_diagonal, word_deletions = (
                _coda_matrix_values(aggregation, run_key, scope)
            )
            image = axis.imshow(rates, cmap="Blues", vmin=0.0, vmax=1.0)
            axis.set_xticks(range(len(CODA_ORDER)), labels=labels, rotation=35, ha="right")
            axis.set_yticks(range(len(CODA_ORDER)), labels=labels)
            axis.set_xlabel("Âm cuối dự đoán")
            axis.set_ylabel("Âm cuối tham chiếu")
            scope_label = SCOPE_DISPLAY[scope]
            if scope == "low_snr":
                scope_label = f"{scope_label} ({low_label} dB)"
            axis.set_title(
                f"{_focus_display(candidate)} — {scope_label}\n"
                f"N={matrix_total:,}; nhầm={off_diagonal:,}; "
                f"xóa từ có âm cuối={word_deletions:,}",
                fontsize=11,
            )
            for ref_index in range(len(CODA_ORDER)):
                for hyp_index in range(len(CODA_ORDER)):
                    count = raw[ref_index][hyp_index]
                    rate = rates[ref_index][hyp_index]
                    annotation = "0" if count == 0 else f"{count}\n{rate:.1%}"
                    color = "white" if rate >= 0.5 else "black"
                    axis.text(
                        hyp_index,
                        ref_index,
                        annotation,
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=5.5,
                    )

    if image is None:
        plt.close(figure)
        raise ErrorArtifactError("cannot render a coda matrix without candidate runs")
    figure.suptitle(
        f"Ma trận nhầm lẫn âm cuối tiếng Việt — {METRIC_VERSION}",
        fontsize=15,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.84,
        bottom=0.08,
        top=0.84,
        wspace=0.24,
        hspace=0.34,
    )
    colorbar_axis = figure.add_axes([0.89, 0.14, 0.018, 0.70])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Tỷ lệ theo âm cuối tham chiếu")
    try:
        figure.savefig(
            path,
            format="png",
            dpi=300,
            facecolor="white",
            metadata={"Software": f"VitoNESR {METRIC_VERSION}"},
        )
    finally:
        plt.close(figure)


def _write_csv_temp(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    columns: Sequence[str] = TONE_MATRIX_COLUMNS,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def _write_png_temp(
    path: Path,
    aggregation: ToneAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        render_tone_confusion_png(temporary, aggregation, candidate_runs)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def _write_coda_png_temp(
    path: Path,
    aggregation: CodaAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        render_coda_confusion_png(temporary, aggregation, candidate_runs)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def _render_csv_bytes(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    columns: Sequence[str],
    resume: bool,
) -> bytes:
    if resume:
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
    temporary = _write_csv_temp(path, rows, columns=columns)
    try:
        return temporary.read_bytes()
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_tone_png_bytes(
    path: Path,
    aggregation: ToneAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
    *,
    resume: bool,
) -> bytes:
    if not resume:
        temporary = _write_png_temp(path, aggregation, candidate_runs)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.resume-render.{uuid.uuid4().hex}.png"
        )
        try:
            render_tone_confusion_png(temporary, aggregation, candidate_runs)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
    try:
        return temporary.read_bytes()
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_coda_png_bytes(
    path: Path,
    aggregation: CodaAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
    *,
    resume: bool,
) -> bytes:
    if not resume:
        temporary = _write_coda_png_temp(path, aggregation, candidate_runs)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.resume-render.{uuid.uuid4().hex}.png"
        )
        try:
            render_coda_confusion_png(temporary, aggregation, candidate_runs)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
    try:
        return temporary.read_bytes()
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_parameters(
    *,
    artifact: str,
    event_rows: int,
    low_snrs: Sequence[str],
    candidate_runs: Sequence[tuple[str, RunKey]],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "artifact": artifact,
        "metric_version": METRIC_VERSION,
        "event_rows": event_rows,
        "low_snrs": list(low_snrs),
        "focus_runs": [
            {"label": label, "run_key": list(run_key)}
            for label, run_key in candidate_runs
        ],
        **dict(extra or {}),
    }


def _destinations(output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    return directory / TONE_CSV_NAME, directory / TONE_PNG_NAME


def _coda_destinations(output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    return directory / CODA_CSV_NAME, directory / CODA_PNG_NAME


def _short_word_destination(output_dir: str | Path) -> Path:
    return Path(output_dir) / SHORT_WORD_CSV_NAME


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp")


def _backup_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.bak")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except (OSError, OverflowError):
        return False
    return True


def _lock_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
        return int(text.removeprefix("pid=")) if text.startswith("pid=") else None
    except (OSError, UnicodeDecodeError, ValueError):
        return None


@contextmanager
def _output_lock(
    output_dir: str | Path,
    *,
    lock_name: str = OUTPUT_LOCK_NAME,
    bundle_name: str = "tone_confusion_matrix",
    resume: bool = False,
) -> Iterator[None]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / lock_name
    journal_path = directory / f".{bundle_name}.bundle.transaction.json"
    marker_path = directory / f"{bundle_name}.bundle.commit.json"
    if (
        lock_path.exists()
        and resume
        and (journal_path.is_file() or marker_path.is_file())
    ):
        owner = _lock_pid(lock_path)
        if owner is not None and not _pid_is_alive(owner):
            lock_path.unlink()
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ErrorArtifactError(
            "output directory is locked by another build, or a stale lock remains: "
            f"{lock_path}"
        ) from exc

    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def ensure_outputs_available(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool,
    resume: bool = False,
) -> tuple[Path, Path]:
    destinations = _destinations(output_dir)
    _ensure_output_pair_available(
        event_path, destinations, overwrite=overwrite, resume=resume
    )
    return destinations


def ensure_coda_outputs_available(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool,
    resume: bool = False,
) -> tuple[Path, Path]:
    destinations = _coda_destinations(output_dir)
    _ensure_output_pair_available(
        event_path, destinations, overwrite=overwrite, resume=resume
    )
    return destinations


def ensure_short_word_output_available(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool,
    resume: bool = False,
) -> Path:
    destination = _short_word_destination(output_dir)
    _ensure_output_pair_available(
        event_path, (destination,), overwrite=overwrite, resume=resume
    )
    return destination


def _ensure_output_pair_available(
    event_path: str | Path,
    destinations: Sequence[Path],
    *,
    overwrite: bool,
    resume: bool = False,
) -> None:
    resolved_input = Path(event_path).resolve()
    if any(path.resolve() == resolved_input for path in destinations):
        raise ErrorArtifactError("refusing to overwrite the input event CSV")
    if resume:
        return
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise ErrorArtifactError(
            "output already exists; use a new --out-dir or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    auxiliaries = [
        auxiliary
        for path in destinations
        for auxiliary in (_temporary_path(path), _backup_path(path))
    ]
    stale = [path for path in auxiliaries if path.exists()]
    if stale:
        raise ErrorArtifactError(
            "temporary or backup output already exists; inspect or remove it before "
            "rerunning: "
            + ", ".join(str(path) for path in stale)
        )


def _commit_output_pair(
    temporary_paths: Sequence[Path],
    destinations: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    if len(temporary_paths) != len(destinations):
        raise ErrorArtifactError("output commit requires one temporary file per destination")
    missing_temporary = [path for path in temporary_paths if not path.is_file()]
    if missing_temporary:
        raise ErrorArtifactError(
            "temporary output is missing before commit: "
            + ", ".join(str(path) for path in missing_temporary)
        )

    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise ErrorArtifactError(
            "output appeared during generation; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    backups = tuple(_backup_path(path) for path in destinations)
    stale_backups = [path for path in backups if path.exists()]
    if stale_backups:
        raise ErrorArtifactError(
            "backup output appeared during generation; inspect or remove it before "
            "rerunning: "
            + ", ".join(str(path) for path in stale_backups)
        )

    backed_up: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        if overwrite:
            for destination, backup in zip(destinations, backups):
                if destination.exists():
                    destination.replace(backup)
                    backed_up.append((backup, destination))

        for temporary, destination in zip(temporary_paths, destinations):
            if overwrite:
                if destination.exists():
                    raise ErrorArtifactError(
                        f"output appeared during commit; refusing to overwrite: {destination}"
                    )
                temporary.replace(destination)
                committed.append(destination)
                continue

            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise ErrorArtifactError(
                    f"output appeared during commit; refusing to overwrite: {destination}"
                ) from exc
            committed.append(destination)
            temporary.unlink()
    except Exception as commit_error:
        rollback_errors: list[str] = []
        for destination in reversed(committed):
            try:
                if destination.exists():
                    destination.unlink()
            except OSError as exc:
                rollback_errors.append(f"remove {destination}: {exc}")
        for backup, destination in reversed(backed_up):
            try:
                if backup.exists():
                    backup.replace(destination)
            except OSError as exc:
                rollback_errors.append(f"restore {destination}: {exc}")
        if rollback_errors:
            raise ErrorArtifactError(
                "output pair commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from commit_error
        raise

    for backup, _ in backed_up:
        if backup.exists():
            backup.unlink()


def write_tone_outputs(
    output_dir: str | Path,
    rows: Sequence[dict[str, object]],
    aggregation: ToneAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
    *,
    overwrite: bool,
    resume: bool = False,
    input_bindings: Sequence[Mapping[str, object]] = (),
) -> tuple[Path, Path]:
    csv_path, png_path = _destinations(output_dir)
    csv_content = _render_csv_bytes(
        csv_path, rows, columns=TONE_MATRIX_COLUMNS, resume=resume
    )
    png_content = _render_tone_png_bytes(
        png_path, aggregation, candidate_runs, resume=resume
    )
    commit_artifact_bundle(
        bundle_name="tone_confusion_matrix",
        bundle_version=ERROR_ARTIFACT_BUNDLE_VERSION,
        data_destinations={"csv": csv_path, "png": png_path},
        data_contents={"csv": csv_content, "png": png_content},
        provenance_path=Path(output_dir) / "tone_confusion_matrix.provenance.json",
        input_bindings=input_bindings,
        parameters=_artifact_parameters(
            artifact="tone_confusion_matrix",
            event_rows=aggregation.event_rows,
            low_snrs=aggregation.low_snrs,
            candidate_runs=candidate_runs,
            extra={"matrix_rows": len(rows)},
        ),
        overwrite=overwrite,
        resume=resume,
        error_type=ErrorArtifactError,
    )
    return csv_path, png_path


def write_coda_outputs(
    output_dir: str | Path,
    rows: Sequence[dict[str, object]],
    aggregation: CodaAggregation,
    candidate_runs: Sequence[tuple[str, RunKey]],
    *,
    overwrite: bool,
    resume: bool = False,
    input_bindings: Sequence[Mapping[str, object]] = (),
) -> tuple[Path, Path]:
    csv_path, png_path = _coda_destinations(output_dir)
    csv_content = _render_csv_bytes(
        csv_path, rows, columns=CODA_MATRIX_COLUMNS, resume=resume
    )
    png_content = _render_coda_png_bytes(
        png_path, aggregation, candidate_runs, resume=resume
    )
    commit_artifact_bundle(
        bundle_name="final_coda_confusion_matrix",
        bundle_version=ERROR_ARTIFACT_BUNDLE_VERSION,
        data_destinations={"csv": csv_path, "png": png_path},
        data_contents={"csv": csv_content, "png": png_content},
        provenance_path=Path(output_dir)
        / "final_coda_confusion_matrix.provenance.json",
        input_bindings=input_bindings,
        parameters=_artifact_parameters(
            artifact="final_coda_confusion_matrix",
            event_rows=aggregation.event_rows,
            low_snrs=aggregation.low_snrs,
            candidate_runs=candidate_runs,
            extra={"matrix_rows": len(rows)},
        ),
        overwrite=overwrite,
        resume=resume,
        error_type=ErrorArtifactError,
    )
    return csv_path, png_path


def write_short_word_output(
    output_dir: str | Path,
    rows: Sequence[dict[str, object]],
    *,
    overwrite: bool,
    resume: bool = False,
    input_bindings: Sequence[Mapping[str, object]] = (),
    parameters: Mapping[str, object] | None = None,
) -> Path:
    csv_path = _short_word_destination(output_dir)
    csv_content = _render_csv_bytes(
        csv_path, rows, columns=SHORT_WORD_COLUMNS, resume=resume
    )
    commit_artifact_bundle(
        bundle_name="short_word_deletion_examples",
        bundle_version=ERROR_ARTIFACT_BUNDLE_VERSION,
        data_destinations={"csv": csv_path},
        data_contents={"csv": csv_content},
        provenance_path=Path(output_dir)
        / "short_word_deletion_examples.provenance.json",
        input_bindings=input_bindings,
        parameters=dict(parameters or {}),
        overwrite=overwrite,
        resume=resume,
        error_type=ErrorArtifactError,
    )
    return csv_path


def run_build(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_lambdas: Sequence[object] | None = None,
    focus_runs: Sequence[object] | None = None,
    low_snrs: Sequence[object],
    overall_only: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> BuildResult:
    low_snr_values = _canonical_scopes(low_snrs, overall_only=overall_only)
    bindings = bind_input_files((event_path,), root=ROOT)
    with _output_lock(
        output_dir,
        bundle_name="tone_confusion_matrix",
        resume=resume,
    ):
        ensure_outputs_available(
            event_path, output_dir, overwrite=overwrite, resume=resume
        )
        aggregation = load_tone_aggregation(
            event_path,
            low_snrs=low_snr_values,
            overall_only=overall_only,
        )
        verify_input_bindings(bindings, (event_path,), root=ROOT)
        candidate_runs = _resolve_report_runs(
            aggregation.run_keys,
            candidate_lambdas=candidate_lambdas,
            focus_runs=focus_runs,
        )
        rows = build_tone_matrix_rows(aggregation)
        csv_path, png_path = write_tone_outputs(
            output_dir,
            rows,
            aggregation,
            candidate_runs,
            overwrite=overwrite,
            resume=resume,
            input_bindings=bindings,
        )
        return BuildResult(
            csv_path=csv_path,
            png_path=png_path,
            matrix_rows=len(rows),
            aggregation=aggregation,
            candidate_runs=candidate_runs,
        )


def run_coda_build(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_lambdas: Sequence[object] | None = None,
    focus_runs: Sequence[object] | None = None,
    low_snrs: Sequence[object],
    overall_only: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> CodaBuildResult:
    low_snr_values = _canonical_scopes(low_snrs, overall_only=overall_only)
    bindings = bind_input_files((event_path,), root=ROOT)
    with _output_lock(
        output_dir,
        lock_name=CODA_OUTPUT_LOCK_NAME,
        bundle_name="final_coda_confusion_matrix",
        resume=resume,
    ):
        ensure_coda_outputs_available(
            event_path, output_dir, overwrite=overwrite, resume=resume
        )
        aggregation = load_coda_aggregation(
            event_path,
            low_snrs=low_snr_values,
            overall_only=overall_only,
        )
        verify_input_bindings(bindings, (event_path,), root=ROOT)
        candidate_runs = _resolve_report_runs(
            aggregation.run_keys,
            candidate_lambdas=candidate_lambdas,
            focus_runs=focus_runs,
        )
        rows = build_coda_matrix_rows(aggregation)
        csv_path, png_path = write_coda_outputs(
            output_dir,
            rows,
            aggregation,
            candidate_runs,
            overwrite=overwrite,
            resume=resume,
            input_bindings=bindings,
        )
        return CodaBuildResult(
            csv_path=csv_path,
            png_path=png_path,
            matrix_rows=len(rows),
            aggregation=aggregation,
            candidate_runs=candidate_runs,
        )


def run_short_word_build(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_lambdas: Sequence[object] | None = None,
    focus_runs: Sequence[object] | None = None,
    low_snrs: Sequence[object],
    context_window: int,
    overall_only: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> ShortWordBuildResult:
    low_snr_values = _canonical_scopes(low_snrs, overall_only=overall_only)
    bindings = bind_input_files((event_path,), root=ROOT)
    with _output_lock(
        output_dir,
        lock_name=SHORT_WORD_OUTPUT_LOCK_NAME,
        bundle_name="short_word_deletion_examples",
        resume=resume,
    ):
        ensure_short_word_output_available(
            event_path,
            output_dir,
            overwrite=overwrite,
            resume=resume,
        )
        aggregation = load_short_word_aggregation(
            event_path,
            low_snrs=low_snr_values,
            context_window=context_window,
            overall_only=overall_only,
        )
        verify_input_bindings(bindings, (event_path,), root=ROOT)
        candidate_runs = _resolve_report_runs(
            aggregation.run_keys,
            candidate_lambdas=candidate_lambdas,
            focus_runs=focus_runs,
        )
        csv_path = write_short_word_output(
            output_dir,
            aggregation.examples,
            overwrite=overwrite,
            resume=resume,
            input_bindings=bindings,
            parameters=_artifact_parameters(
                artifact="short_word_deletion_examples",
                event_rows=aggregation.event_rows,
                low_snrs=aggregation.low_snrs,
                candidate_runs=candidate_runs,
                extra={
                    "example_rows": len(aggregation.examples),
                    "context_window": context_window,
                },
            ),
        )
        return ShortWordBuildResult(
            csv_path=csv_path,
            example_rows=len(aggregation.examples),
            aggregation=aggregation,
            candidate_runs=candidate_runs,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build error-analysis artifacts from the shared aligned-v1 events."
        )
    )
    parser.add_argument(
        "--artifact",
        choices=("tone", "coda", "short-word"),
        default="tone",
        help="Artifact to build. Default: tone.",
    )
    parser.add_argument("--events", required=True, help="Gate C error_events.csv")
    parser.add_argument(
        "--candidate-lambda",
        action="append",
        default=None,
        help=(
            "Legacy tone-aware candidate lambda for focused reporting; repeat exactly "
            "twice. Mutually exclusive with --focus-run."
        ),
    )
    parser.add_argument(
        "--focus-run",
        action="append",
        default=None,
        help=(
            "Explicit focused run as TRAIN_TYPE:LAMBDA; repeatable. Example: "
            "ordinary_lora:0. Mutually exclusive with --candidate-lambda."
        ),
    )
    parser.add_argument(
        "--low-snr",
        action="append",
        default=None,
        help="SNR included in the low_snr scope; repeatable. Default: 0 and 5.",
    )
    parser.add_argument(
        "--overall-only",
        action="store_true",
        help="Build only the overall scope (for clean-only external datasets such as FLEURS).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Tokens on each side of a short-word deletion context. Default: 3.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--formal-paper-v2",
        action="store_true",
        help="Re-verify the error bundle and every source prediction before use.",
    )
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--split-lock")
    parser.add_argument("--decision-lock")
    parser.add_argument("--final-benchmark-lock")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Recover only a hash-verified interrupted artifact bundle.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.overall_only:
        low_snrs = args.low_snr or []
    else:
        low_snrs = args.low_snr or ["0", "5"]
    try:
        if args.formal_paper_v2:
            missing = [
                name
                for name, value in (
                    ("benchmark_manifest", args.benchmark_manifest),
                    ("split_lock", args.split_lock),
                    ("decision_lock", args.decision_lock),
                )
                if value is None
            ]
            if missing:
                raise ErrorArtifactError(
                    "formal paper-v2 artifacts require " + ", ".join(missing)
                )
            verify_formal_error_events(
                args.events,
                benchmark_path=args.benchmark_manifest,
                split_lock_path=args.split_lock,
                decision_path=args.decision_lock,
                final_benchmark_lock_path=args.final_benchmark_lock,
                root=ROOT,
            )
        if args.artifact == "short-word":
            result = run_short_word_build(
                args.events,
                args.out_dir,
                candidate_lambdas=args.candidate_lambda,
                focus_runs=args.focus_run,
                low_snrs=low_snrs,
                context_window=args.context_window,
                overall_only=args.overall_only,
                overwrite=args.overwrite,
                resume=args.resume,
            )
        elif args.artifact == "coda":
            result = run_coda_build(
                args.events,
                args.out_dir,
                candidate_lambdas=args.candidate_lambda,
                focus_runs=args.focus_run,
                low_snrs=low_snrs,
                overall_only=args.overall_only,
                overwrite=args.overwrite,
                resume=args.resume,
            )
        else:
            result = run_build(
                args.events,
                args.out_dir,
                candidate_lambdas=args.candidate_lambda,
                focus_runs=args.focus_run,
                low_snrs=low_snrs,
                overall_only=args.overall_only,
                overwrite=args.overwrite,
                resume=args.resume,
            )
    except (ErrorArtifactError, PredictionEvidenceError, OSError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(f"PASS event rows: {result.aggregation.event_rows}")
    print(f"PASS runs: {len(result.aggregation.run_keys)}")
    if result.aggregation.low_snrs:
        print(f"PASS low SNR: {','.join(result.aggregation.low_snrs)}")
    else:
        print("PASS scope: overall-only")
    scopes = _scope_order(result.aggregation.low_snrs)
    if args.artifact == "short-word":
        for candidate, run_key in result.candidate_runs:
            for scope in scopes:
                key = (run_key, scope)
                examples = int(result.aggregation.deletion_counts.get(key, 0))
                reference_units = int(result.aggregation.reference_units.get(key, 0))
                swdr = examples / reference_units if reference_units else 0.0
                print(
                    f"{_focus_console(candidate)} scope={scope} examples={examples} "
                    f"reference_units={reference_units} swdr={swdr:.12f}"
                )
        print(f"wrote {result.csv_path} ({result.example_rows} rows)")
        return

    for candidate, run_key in result.candidate_runs:
        for scope in scopes:
            if args.artifact == "coda":
                _, _, matrix_total, off_diagonal, word_deletions = (
                    _coda_matrix_values(result.aggregation, run_key, scope)
                )
                print(
                    f"{_focus_console(candidate)} scope={scope} matrix={matrix_total} "
                    f"off_diagonal={off_diagonal} word_deletions={word_deletions}"
                )
            else:
                _, _, matrix_total, off_diagonal, deletions = _matrix_values(
                    result.aggregation, run_key, scope
                )
                print(
                    f"{_focus_console(candidate)} scope={scope} matrix={matrix_total} "
                    f"off_diagonal={off_diagonal} excluded_deletions={deletions}"
                )
    print(f"wrote {result.csv_path} ({result.matrix_rows} rows)")
    print(f"wrote {result.png_path}")


if __name__ == "__main__":
    main()
