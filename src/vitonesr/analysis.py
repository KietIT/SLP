"""Canonical prediction analysis for the final Vietnamese ASR metrics.

The public metric contract in this module is versioned.  Unlike the legacy
position-based helpers in :mod:`src.vitonesr.metrics`, ``aligned_v1`` aligns
each utterance independently and reuses the same indexed events for scalar
metrics and qualitative error analysis.
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from .text_norm import normalize_vi_text
from .tone import ID_TO_TONE, extract_tone, strip_tone_marks


METRIC_VERSION = "aligned_v1"

CANONICAL_PREDICTION_COLUMNS = (
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
)

RUN_METADATA_COLUMNS = (
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
)

TONE_LABELS = tuple(ID_TO_TONE[index] for index in sorted(ID_TO_TONE))
FINAL_CODAS = ("ch", "ng", "nh", "n", "t", "c", "m", "p")
SHORT_WORDS = frozenset({"đã", "có", "là", "một", "và"})

_TRAIN_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_INTEGER_RE = re.compile(r"^[0-9]+$")


class PredictionValidationError(ValueError):
    """Raised when a prediction table violates the canonical schema."""


def _source_label(source: str | Path) -> str:
    return str(source)


def _canonical_float(value: str, *, field: str, source: str, row_number: int) -> str:
    try:
        number = float(value)
    except ValueError as error:
        raise PredictionValidationError(
            f"{source}: row {row_number} has non-numeric {field}={value!r}"
        ) from error
    if not math.isfinite(number):
        raise PredictionValidationError(
            f"{source}: row {row_number} has non-finite {field}={value!r}"
        )
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def validate_prediction_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    source: str | Path = "<memory>",
) -> list[dict[str, str]]:
    """Validate and return canonical string dictionaries.

    ``ref`` and ``hyp`` retain their whitespace and spelling; only their
    Unicode representation is normalized to NFC.  Metadata is stripped and
    numeric metadata is canonicalized.  The input mappings are never mutated.
    """

    label = _source_label(source)
    materialized = list(rows)
    if not materialized:
        raise PredictionValidationError(f"{label}: prediction table is empty")

    expected = set(CANONICAL_PREDICTION_COLUMNS)
    validated: list[dict[str, str]] = []
    seen_utt_ids: set[str] = set()

    for offset, raw_row in enumerate(materialized, start=2):
        actual = set(raw_row)
        missing = sorted(expected - actual)
        extra = sorted((repr(name) for name in actual - expected))
        if missing or extra:
            raise PredictionValidationError(
                f"{label}: row {offset} has non-canonical fields; "
                f"missing={missing}, extra={extra}"
            )

        missing_cells = [
            column
            for column in CANONICAL_PREDICTION_COLUMNS
            if raw_row[column] is None
        ]
        if missing_cells:
            raise PredictionValidationError(
                f"{label}: row {offset} has missing CSV cells: {missing_cells}"
            )

        row = {
            column: str(raw_row[column])
            for column in CANONICAL_PREDICTION_COLUMNS
        }
        for column in CANONICAL_PREDICTION_COLUMNS[:-2]:
            row[column] = row[column].strip()
        row["ref"] = unicodedata.normalize("NFC", row["ref"])
        row["hyp"] = unicodedata.normalize("NFC", row["hyp"])

        required = (
            "utt_id",
            "dataset",
            "model",
            "model_size",
            "train_type",
            "seed",
            "snr",
            "noise_type",
        )
        blank = [column for column in required if not row[column]]
        if not row["ref"].strip():
            blank.append("ref")
        if blank:
            raise PredictionValidationError(
                f"{label}: row {offset} has empty required fields: {blank}"
            )

        utt_id = row["utt_id"]
        if utt_id in seen_utt_ids:
            raise PredictionValidationError(
                f"{label}: duplicate utt_id {utt_id!r} at row {offset}"
            )
        seen_utt_ids.add(utt_id)

        if not _TRAIN_TYPE_RE.fullmatch(row["train_type"]):
            raise PredictionValidationError(
                f"{label}: row {offset} has invalid train_type={row['train_type']!r}"
            )
        if not _INTEGER_RE.fullmatch(row["seed"]):
            raise PredictionValidationError(
                f"{label}: row {offset} has invalid non-negative integer seed={row['seed']!r}"
            )
        row["seed"] = str(int(row["seed"]))

        if row["train_type"] == "zero_shot":
            if row["lambda"]:
                raise PredictionValidationError(
                    f"{label}: row {offset} zero_shot must use an empty lambda"
                )
        else:
            if not row["lambda"]:
                raise PredictionValidationError(
                    f"{label}: row {offset} non-zero-shot prediction requires lambda"
                )
            row["lambda"] = _canonical_float(
                row["lambda"], field="lambda", source=label, row_number=offset
            )
            if float(row["lambda"]) < 0:
                raise PredictionValidationError(
                    f"{label}: row {offset} lambda must be non-negative"
                )

        if row["snr"].lower() == "clean":
            row["snr"] = "clean"
            if row["noise_type"].lower() != "clean":
                raise PredictionValidationError(
                    f"{label}: row {offset} clean SNR requires noise_type='clean'"
                )
            row["noise_type"] = "clean"
        else:
            row["snr"] = _canonical_float(
                row["snr"], field="snr", source=label, row_number=offset
            )
            if row["noise_type"].lower() == "clean":
                raise PredictionValidationError(
                    f"{label}: row {offset} noisy SNR cannot use noise_type='clean'"
                )

        validated.append(row)

    baseline = {column: validated[0][column] for column in RUN_METADATA_COLUMNS}
    for row_number, row in enumerate(validated[1:], start=3):
        conflicts = [
            column
            for column in RUN_METADATA_COLUMNS
            if row[column] != baseline[column]
        ]
        if conflicts:
            raise PredictionValidationError(
                f"{label}: row {row_number} conflicts with run metadata in "
                f"columns {conflicts}"
            )
    return validated


def load_prediction_csv(path: str | Path) -> list[dict[str, str]]:
    """Load one exact-schema UTF-8 prediction CSV and validate its rows."""

    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != CANONICAL_PREDICTION_COLUMNS:
            raise PredictionValidationError(
                f"{prediction_path}: expected columns "
                f"{list(CANONICAL_PREDICTION_COLUMNS)}, found {list(columns)}"
            )
        rows = list(reader)
    return validate_prediction_rows(rows, source=prediction_path)


Operation = Literal["match", "substitution", "deletion", "insertion"]


@dataclass(frozen=True, slots=True)
class AlignmentEvent:
    """One deterministic, indexed operation in an utterance-level alignment."""

    utterance_index: int
    utt_id: str
    dataset: str
    model: str
    model_size: str
    train_type: str
    lambda_value: str
    seed: str
    snr: str
    noise_type: str
    operation: Operation
    ref_index: int | None
    hyp_index: int | None
    ref_token: str | None
    hyp_token: str | None
    ref_text: str
    hyp_text: str
    ref_tone_base: str | None
    hyp_tone_base: str | None
    ref_plain_base: str | None
    hyp_plain_base: str | None
    ref_tone: str | None
    hyp_tone: str | None
    ref_coda: str | None
    hyp_coda: str | None
    tone_eligible: bool
    tone_error: bool
    diacritic_eligible: bool
    diacritic_error: bool
    final_consonant_eligible: bool
    final_consonant_error: bool
    short_word_deletion: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize with canonical CSV-friendly field names."""

        return {
            "metric_version": METRIC_VERSION,
            "utterance_index": self.utterance_index,
            "utt_id": self.utt_id,
            "dataset": self.dataset,
            "model": self.model,
            "model_size": self.model_size,
            "train_type": self.train_type,
            "lambda": self.lambda_value,
            "seed": self.seed,
            "snr": self.snr,
            "noise_type": self.noise_type,
            "operation": self.operation,
            "ref_index": "" if self.ref_index is None else self.ref_index,
            "hyp_index": "" if self.hyp_index is None else self.hyp_index,
            "ref_token": self.ref_token or "",
            "hyp_token": self.hyp_token or "",
            "ref": self.ref_text,
            "hyp": self.hyp_text,
            "ref_tone_base": self.ref_tone_base or "",
            "hyp_tone_base": self.hyp_tone_base or "",
            "ref_plain_base": self.ref_plain_base or "",
            "hyp_plain_base": self.hyp_plain_base or "",
            "ref_tone": self.ref_tone or "",
            "hyp_tone": self.hyp_tone or "",
            "ref_coda": self.ref_coda or "",
            "hyp_coda": self.hyp_coda or "",
            "tone_eligible": self.tone_eligible,
            "tone_error": self.tone_error,
            "diacritic_eligible": self.diacritic_eligible,
            "diacritic_error": self.diacritic_error,
            "final_consonant_eligible": self.final_consonant_eligible,
            "final_consonant_error": self.final_consonant_error,
            "short_word_deletion": self.short_word_deletion,
        }


def _add_cost(cost: tuple[int, int], edits: int, semantic: int) -> tuple[int, int]:
    return cost[0] + edits, cost[1] + semantic


def _plain_base(token: str) -> str:
    translated = token.replace("đ", "d").replace("Đ", "D")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", translated)
        if unicodedata.category(character) != "Mn"
    ).lower()


def _tone_name(token: str | None) -> str | None:
    if token is None:
        return None
    tone_id, valid = extract_tone(token)
    return ID_TO_TONE[tone_id] if valid else None


def final_coda(token: str | None) -> str | None:
    """Return one of the eight final-coda labels, preferring longest matches."""

    if token is None:
        return None
    base = strip_tone_marks(token).lower()
    for coda in FINAL_CODAS:
        if base.endswith(coda):
            return coda
    return None


def _substitution_semantic_penalty(ref_token: str, hyp_token: str) -> int:
    if strip_tone_marks(ref_token) == strip_tone_marks(hyp_token):
        return 0
    if _plain_base(ref_token) == _plain_base(hyp_token):
        return 1
    return 3


def _indexed_operations(
    ref_tokens: Sequence[str], hyp_tokens: Sequence[str]
) -> list[tuple[Operation, int | None, int | None]]:
    """Levenshtein alignment with deterministic linguistic tie-breaking."""

    rows = len(ref_tokens) + 1
    columns = len(hyp_tokens) + 1
    costs = [[(0, 0)] * columns for _ in range(rows)]
    back: list[list[Operation | None]] = [[None] * columns for _ in range(rows)]
    for ref_index in range(1, rows):
        costs[ref_index][0] = (ref_index, ref_index * 2)
        back[ref_index][0] = "deletion"
    for hyp_index in range(1, columns):
        costs[0][hyp_index] = (hyp_index, hyp_index * 2)
        back[0][hyp_index] = "insertion"

    for ref_index in range(1, rows):
        for hyp_index in range(1, columns):
            ref_token = ref_tokens[ref_index - 1]
            hyp_token = hyp_tokens[hyp_index - 1]
            if ref_token == hyp_token:
                diagonal = _add_cost(costs[ref_index - 1][hyp_index - 1], 0, 0)
                diagonal_op: Operation = "match"
                diagonal_rank = 0
            else:
                diagonal = _add_cost(
                    costs[ref_index - 1][hyp_index - 1],
                    1,
                    _substitution_semantic_penalty(ref_token, hyp_token),
                )
                diagonal_op = "substitution"
                diagonal_rank = 1
            deletion = _add_cost(costs[ref_index - 1][hyp_index], 1, 2)
            insertion = _add_cost(costs[ref_index][hyp_index - 1], 1, 2)
            candidates = (
                (diagonal[0], diagonal[1], diagonal_rank, diagonal_op),
                (deletion[0], deletion[1], 2, "deletion"),
                (insertion[0], insertion[1], 3, "insertion"),
            )
            best = min(candidates)
            costs[ref_index][hyp_index] = best[0], best[1]
            back[ref_index][hyp_index] = best[3]  # type: ignore[assignment]

    operations: list[tuple[Operation, int | None, int | None]] = []
    ref_index = len(ref_tokens)
    hyp_index = len(hyp_tokens)
    while ref_index > 0 or hyp_index > 0:
        operation = back[ref_index][hyp_index]
        if operation in {"match", "substitution"}:
            operations.append((operation, ref_index - 1, hyp_index - 1))
            ref_index -= 1
            hyp_index -= 1
        elif operation == "deletion":
            operations.append((operation, ref_index - 1, None))
            ref_index -= 1
        elif operation == "insertion":
            operations.append((operation, None, hyp_index - 1))
            hyp_index -= 1
        else:  # pragma: no cover - defensive invariant
            raise RuntimeError("Alignment backtrace is incomplete")
    operations.reverse()
    return operations


def _metadata_value(metadata: Mapping[str, object], name: str) -> str:
    value = metadata.get(name, "")
    return "" if value is None else str(value)


def analyze_error_events(
    ref: str,
    hyp: str,
    *,
    utt_id: str = "",
    utterance_index: int = 0,
    metadata: Mapping[str, object] | None = None,
) -> list[AlignmentEvent]:
    """Return deterministic indexed word events for one utterance."""

    metadata = metadata or {}
    normalized_ref = normalize_vi_text(ref)
    normalized_hyp = normalize_vi_text(hyp)
    ref_tokens = normalized_ref.split()
    hyp_tokens = normalized_hyp.split()
    operations = _indexed_operations(ref_tokens, hyp_tokens)
    events: list[AlignmentEvent] = []

    for operation, ref_index, hyp_index in operations:
        ref_token = None if ref_index is None else ref_tokens[ref_index]
        hyp_token = None if hyp_index is None else hyp_tokens[hyp_index]
        ref_tone_base = None if ref_token is None else strip_tone_marks(ref_token)
        hyp_tone_base = None if hyp_token is None else strip_tone_marks(hyp_token)
        ref_plain_base = None if ref_token is None else _plain_base(ref_token)
        hyp_plain_base = None if hyp_token is None else _plain_base(hyp_token)
        ref_tone = _tone_name(ref_token)
        hyp_tone = _tone_name(hyp_token)
        ref_coda = final_coda(ref_token)
        hyp_coda = final_coda(hyp_token)

        same_tone_base = (
            ref_tone_base is not None
            and hyp_tone_base is not None
            and ref_tone_base == hyp_tone_base
        )
        tone_eligible = ref_tone is not None and (
            operation == "deletion" or same_tone_base
        )
        tone_error = tone_eligible and (
            hyp_tone is None or ref_tone != hyp_tone
        )

        diacritic_eligible = (
            ref_token is not None
            and hyp_token is not None
            and ref_plain_base == hyp_plain_base
        )
        diacritic_error = diacritic_eligible and ref_tone_base != hyp_tone_base

        if ref_token is None:
            final_eligible = False
        elif hyp_token is None:
            final_eligible = ref_coda is not None
        else:
            final_eligible = ref_coda is not None or hyp_coda is not None
        final_error = final_eligible and ref_coda != hyp_coda

        events.append(
            AlignmentEvent(
                utterance_index=utterance_index,
                utt_id=utt_id or _metadata_value(metadata, "utt_id"),
                dataset=_metadata_value(metadata, "dataset"),
                model=_metadata_value(metadata, "model"),
                model_size=_metadata_value(metadata, "model_size"),
                train_type=_metadata_value(metadata, "train_type"),
                lambda_value=_metadata_value(metadata, "lambda"),
                seed=_metadata_value(metadata, "seed"),
                snr=_metadata_value(metadata, "snr"),
                noise_type=_metadata_value(metadata, "noise_type"),
                operation=operation,
                ref_index=ref_index,
                hyp_index=hyp_index,
                ref_token=ref_token,
                hyp_token=hyp_token,
                ref_text=unicodedata.normalize("NFC", str(ref)),
                hyp_text=unicodedata.normalize("NFC", str(hyp)),
                ref_tone_base=ref_tone_base,
                hyp_tone_base=hyp_tone_base,
                ref_plain_base=ref_plain_base,
                hyp_plain_base=hyp_plain_base,
                ref_tone=ref_tone,
                hyp_tone=hyp_tone,
                ref_coda=ref_coda,
                hyp_coda=hyp_coda,
                tone_eligible=tone_eligible,
                tone_error=tone_error,
                diacritic_eligible=diacritic_eligible,
                diacritic_error=diacritic_error,
                final_consonant_eligible=final_eligible,
                final_consonant_error=final_error,
                short_word_deletion=(
                    operation == "deletion" and ref_token in SHORT_WORDS
                ),
            )
        )
    return events


def analyze_prediction_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    source: str | Path = "<memory>",
) -> list[AlignmentEvent]:
    """Validate canonical rows and emit all indexed events with run metadata."""

    validated = validate_prediction_rows(rows, source=source)
    events: list[AlignmentEvent] = []
    for utterance_index, row in enumerate(validated):
        events.extend(
            analyze_error_events(
                row["ref"],
                row["hyp"],
                utt_id=row["utt_id"],
                utterance_index=utterance_index,
                metadata=row,
            )
        )
    return events


def serialize_alignment_events(
    events: Iterable[AlignmentEvent],
) -> list[dict[str, object]]:
    return [event.to_dict() for event in events]


def _edit_distance(left: Sequence[object], right: Sequence[object]) -> int:
    """Memory-efficient Levenshtein distance."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def _coverage(eligible_units: int, reference_words: int) -> float:
    """Return the auditable share of reference-word positions in a metric.

    TER, DER, and FCER in ``aligned_v1`` are conditional diagnostics: their
    eligible denominators depend on the hypothesis produced at an aligned
    reference-word position.  Keeping this helper separate from ``_rate``
    makes that conditioning explicit without changing the versioned scalar
    definitions.
    """

    return eligible_units / max(reference_words, 1)


@dataclass(frozen=True, slots=True)
class CorpusMetricResult:
    """Aligned-v1 metric values plus auditable corpus numerators/denominators.

    WER and CER use reference-only denominators.  TER, DER, and FCER are
    conditional diagnostics whose eligible denominator can change with the
    hypothesis.  Their ``*_coverage`` properties expose the eligible share of
    reference-word positions and must accompany between-system claims.
    """

    n_utterances: int
    word_errors: int
    word_reference_units: int
    character_errors: int
    character_reference_units: int
    tone_errors: int
    tone_reference_units: int
    diacritic_errors: int
    diacritic_reference_units: int
    final_consonant_errors: int
    final_consonant_reference_units: int
    short_word_deletions: int
    short_word_reference_units: int

    @property
    def wer(self) -> float:
        return _rate(self.word_errors, self.word_reference_units)

    @property
    def cer(self) -> float:
        return _rate(self.character_errors, self.character_reference_units)

    @property
    def ter(self) -> float:
        return _rate(self.tone_errors, self.tone_reference_units)

    @property
    def der(self) -> float:
        return _rate(self.diacritic_errors, self.diacritic_reference_units)

    @property
    def fcer(self) -> float:
        return _rate(
            self.final_consonant_errors, self.final_consonant_reference_units
        )

    @property
    def swdr(self) -> float:
        return _rate(self.short_word_deletions, self.short_word_reference_units)

    @property
    def ter_coverage(self) -> float:
        return _coverage(self.tone_reference_units, self.word_reference_units)

    @property
    def der_coverage(self) -> float:
        return _coverage(self.diacritic_reference_units, self.word_reference_units)

    @property
    def fcer_coverage(self) -> float:
        return _coverage(
            self.final_consonant_reference_units,
            self.word_reference_units,
        )

    def to_dict(self, *, include_counts: bool = False) -> dict[str, object]:
        values: dict[str, object] = {
            "metric_version": METRIC_VERSION,
            "wer": self.wer,
            "cer": self.cer,
            "ter": self.ter,
            "der": self.der,
            "fcer": self.fcer,
            "swdr": self.swdr,
        }
        if include_counts:
            values.update(
                {
                    "n": self.n_utterances,
                    "wer_numerator": self.word_errors,
                    "wer_denominator": self.word_reference_units,
                    "cer_numerator": self.character_errors,
                    "cer_denominator": self.character_reference_units,
                    "ter_numerator": self.tone_errors,
                    "ter_denominator": self.tone_reference_units,
                    "ter_coverage": self.ter_coverage,
                    "der_numerator": self.diacritic_errors,
                    "der_denominator": self.diacritic_reference_units,
                    "der_coverage": self.der_coverage,
                    "fcer_numerator": self.final_consonant_errors,
                    "fcer_denominator": self.final_consonant_reference_units,
                    "fcer_coverage": self.fcer_coverage,
                    "swdr_numerator": self.short_word_deletions,
                    "swdr_denominator": self.short_word_reference_units,
                }
            )
        return values


def compute_aligned_metric_result(
    refs: Sequence[str], hyps: Sequence[str]
) -> CorpusMetricResult:
    """Compute corpus totals without allowing alignment across utterances."""

    if len(refs) != len(hyps):
        raise ValueError(
            f"refs and hyps must have equal length, found {len(refs)} and {len(hyps)}"
        )
    if not refs:
        raise ValueError("Cannot compute aligned metrics for an empty corpus")

    word_errors = 0
    word_reference_units = 0
    character_errors = 0
    character_reference_units = 0
    tone_errors = 0
    tone_reference_units = 0
    diacritic_errors = 0
    diacritic_reference_units = 0
    final_errors = 0
    final_units = 0
    short_deletions = 0
    short_units = 0

    for utterance_index, (ref, hyp) in enumerate(zip(refs, hyps)):
        normalized_ref = normalize_vi_text(ref)
        normalized_hyp = normalize_vi_text(hyp)
        if not normalized_ref:
            raise ValueError(
                "reference at utterance index "
                f"{utterance_index} is empty after text normalization"
            )
        events = analyze_error_events(
            ref, hyp, utterance_index=utterance_index
        )
        word_errors += sum(event.operation != "match" for event in events)
        word_reference_units += len(normalized_ref.split())
        character_errors += _edit_distance(normalized_ref, normalized_hyp)
        character_reference_units += len(normalized_ref)
        tone_errors += sum(event.tone_error for event in events)
        tone_reference_units += sum(event.tone_eligible for event in events)
        diacritic_errors += sum(event.diacritic_error for event in events)
        diacritic_reference_units += sum(
            event.diacritic_eligible for event in events
        )
        final_errors += sum(event.final_consonant_error for event in events)
        final_units += sum(event.final_consonant_eligible for event in events)
        short_deletions += sum(event.short_word_deletion for event in events)
        short_units += sum(event.ref_token in SHORT_WORDS for event in events)

    return CorpusMetricResult(
        n_utterances=len(refs),
        word_errors=word_errors,
        word_reference_units=word_reference_units,
        character_errors=character_errors,
        character_reference_units=character_reference_units,
        tone_errors=tone_errors,
        tone_reference_units=tone_reference_units,
        diacritic_errors=diacritic_errors,
        diacritic_reference_units=diacritic_reference_units,
        final_consonant_errors=final_errors,
        final_consonant_reference_units=final_units,
        short_word_deletions=short_deletions,
        short_word_reference_units=short_units,
    )


def compute_aligned_metrics(
    refs: Sequence[str], hyps: Sequence[str]
) -> dict[str, object]:
    """Return the stable aligned-v1 scalar metric dictionary."""

    return compute_aligned_metric_result(refs, hyps).to_dict()


__all__ = [
    "AlignmentEvent",
    "CANONICAL_PREDICTION_COLUMNS",
    "CorpusMetricResult",
    "FINAL_CODAS",
    "METRIC_VERSION",
    "PredictionValidationError",
    "RUN_METADATA_COLUMNS",
    "SHORT_WORDS",
    "TONE_LABELS",
    "analyze_error_events",
    "analyze_prediction_rows",
    "compute_aligned_metric_result",
    "compute_aligned_metrics",
    "final_coda",
    "load_prediction_csv",
    "serialize_alignment_events",
    "validate_prediction_rows",
]
