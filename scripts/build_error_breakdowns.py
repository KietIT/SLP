from __future__ import annotations

import argparse
import csv
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.error_analysis import EVENT_COLUMNS  # noqa: E402
from src.vitonesr.analysis import METRIC_VERSION, RUN_METADATA_COLUMNS  # noqa: E402


RUN_COLUMNS = list(RUN_METADATA_COLUMNS)
VALID_OPERATIONS = ("match", "substitution", "deletion", "insertion")
VALID_TONES = ("ngang", "sac", "huyen", "hoi", "nga", "nang")
TER_CATEGORY_ORDER = (
    "word_deletion",
    "tone_loss",
    "tone_addition",
    "tone_substitution",
)
DER_CATEGORY_ORDER = (
    "vowel_quality_loss",
    "vowel_quality_addition",
    "vowel_quality_substitution",
    "d_stroke_loss_or_change",
    "mixed_or_other",
)
ORTHOGRAPHIC_CATEGORY_ORDER = (
    "missing_diacritic",
    "wrong_tone_mark",
    "wrong_vowel_mark",
)
ORTHOGRAPHIC_DEFINITIONS = {
    "missing_diacritic": "tone, vowel-quality, or d-stroke mark present in ref but absent in hyp",
    "wrong_tone_mark": "tone addition or tone substitution; tone loss is counted as missing_diacritic",
    "wrong_vowel_mark": "vowel-quality addition or substitution; vowel loss is counted as missing_diacritic",
}
VOWEL_QUALITY_TO_PLAIN = {
    "ă": "a",
    "â": "a",
    "ê": "e",
    "ô": "o",
    "ơ": "o",
    "ư": "u",
}

WER_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "n_utterances",
    "n_ref_tokens",
    "n_hyp_tokens",
    "matches",
    "substitutions",
    "deletions",
    "insertions",
    "word_errors",
    "substitution_rate",
    "deletion_rate",
    "insertion_rate",
    "wer",
]
TER_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "tone_eligible",
    "tone_errors",
    "category",
    "count",
    "rate",
]
DER_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "diacritic_eligible",
    "diacritic_errors",
    "category",
    "count",
    "rate",
]
ORTHOGRAPHIC_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "category",
    "count",
    "definition",
    "overlap_policy",
]
DIACRITIC_EVENT_COLUMNS = [
    "metric_version",
    *RUN_COLUMNS,
    "utt_id",
    "snr",
    "noise_type",
    "operation",
    "ref_token",
    "hyp_token",
    "ref_index",
    "hyp_index",
    "ref_tone",
    "hyp_tone",
    "ref_tone_base",
    "hyp_tone_base",
    "ref_plain_base",
    "hyp_plain_base",
    "tone_error",
    "diacritic_error",
    "tone_primary_category",
    "der_primary_category",
    "quality_transitions",
    "has_tone_loss",
    "has_vowel_quality_loss",
    "has_d_stroke_loss",
    "missing_diacritic",
    "ref",
    "hyp",
]

OUTPUT_NAMES = (
    "wer_decomposition.csv",
    "ter_breakdown.csv",
    "der_breakdown.csv",
    "orthographic_breakdown.csv",
    "diacritic_error_events.csv",
)
OUTPUT_LOCK_NAME = ".error_breakdowns.lock"

RunKey = tuple[str, ...]


class ErrorBreakdownError(ValueError):
    pass


@dataclass(frozen=True)
class BreakdownResult:
    wer_rows: tuple[dict[str, object], ...]
    ter_rows: tuple[dict[str, object], ...]
    der_rows: tuple[dict[str, object], ...]
    orthographic_rows: tuple[dict[str, object], ...]
    event_rows: tuple[dict[str, object], ...]
    input_event_rows: int
    run_keys: tuple[RunKey, ...]


def _sort_text(value: object) -> str:
    return str(value).casefold()


def _run_sort_key(run_key: RunKey) -> tuple[str, ...]:
    return tuple(_sort_text(value) for value in run_key)


def _parse_bool(value: object, *, path: Path, row_number: int, field: str) -> bool:
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ErrorBreakdownError(
        f"{path}: row {row_number}: {field} must be true or false, found {value!r}"
    )


def _parse_index(
    value: object,
    *,
    path: Path,
    row_number: int,
    field: str,
    required: bool,
) -> int | None:
    text = str(value).strip()
    if not text:
        if required:
            raise ErrorBreakdownError(
                f"{path}: row {row_number}: {field} must be a non-negative integer"
            )
        return None
    if not text.isdigit():
        raise ErrorBreakdownError(
            f"{path}: row {row_number}: {field} must be a non-negative integer, "
            f"found {value!r}"
        )
    return int(text)


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator / max(denominator, 1):.12f}"


def _tone_category(
    row: Mapping[str, str],
    *,
    path: Path,
    row_number: int,
) -> str:
    operation = row["operation"]
    if operation == "deletion":
        return "word_deletion"
    if operation not in {"match", "substitution"}:
        raise ErrorBreakdownError(
            f"{path}: row {row_number}: tone error has unsupported operation "
            f"{operation!r}"
        )
    ref_tone = row["ref_tone"]
    hyp_tone = row["hyp_tone"]
    if ref_tone not in VALID_TONES or hyp_tone not in VALID_TONES:
        raise ErrorBreakdownError(
            f"{path}: row {row_number}: paired tone error requires valid ref/hyp tones"
        )
    if ref_tone == hyp_tone:
        raise ErrorBreakdownError(
            f"{path}: row {row_number}: tone_error=true but ref/hyp tones agree"
        )
    if ref_tone != "ngang" and hyp_tone == "ngang":
        return "tone_loss"
    if ref_tone == "ngang" and hyp_tone != "ngang":
        return "tone_addition"
    return "tone_substitution"


def _normalized_base(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).casefold()


def _plain_symbol(value: str) -> str:
    if value == "đ":
        return "d"
    if value in VOWEL_QUALITY_TO_PLAIN:
        return VOWEL_QUALITY_TO_PLAIN[value]
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def _quality_transitions(ref_base: str, hyp_base: str) -> tuple[tuple[str, str], ...]:
    ref_symbols = tuple(_normalized_base(ref_base))
    hyp_symbols = tuple(_normalized_base(hyp_base))
    if len(ref_symbols) != len(hyp_symbols):
        return (
            (
                "<" + _normalized_base(ref_base) + ">",
                "<" + _normalized_base(hyp_base) + ">",
            ),
        )
    transitions: list[tuple[str, str]] = []
    for ref_symbol, hyp_symbol in zip(ref_symbols, hyp_symbols):
        if ref_symbol == hyp_symbol:
            continue
        if _plain_symbol(ref_symbol) != _plain_symbol(hyp_symbol):
            return (
                (
                    "<" + _normalized_base(ref_base) + ">",
                    "<" + _normalized_base(hyp_base) + ">",
                ),
            )
        transitions.append((ref_symbol, hyp_symbol))
    return tuple(transitions)


def _is_vowel_loss(transition: tuple[str, str]) -> bool:
    ref_symbol, hyp_symbol = transition
    return (
        ref_symbol in VOWEL_QUALITY_TO_PLAIN
        and VOWEL_QUALITY_TO_PLAIN[ref_symbol] == hyp_symbol
    )


def _is_vowel_addition(transition: tuple[str, str]) -> bool:
    ref_symbol, hyp_symbol = transition
    return (
        hyp_symbol in VOWEL_QUALITY_TO_PLAIN
        and VOWEL_QUALITY_TO_PLAIN[hyp_symbol] == ref_symbol
    )


def _is_vowel_transition(transition: tuple[str, str]) -> bool:
    ref_symbol, hyp_symbol = transition
    ref_plain = VOWEL_QUALITY_TO_PLAIN.get(ref_symbol, ref_symbol)
    hyp_plain = VOWEL_QUALITY_TO_PLAIN.get(hyp_symbol, hyp_symbol)
    return (
        ref_plain == hyp_plain
        and (
            ref_symbol in VOWEL_QUALITY_TO_PLAIN
            or hyp_symbol in VOWEL_QUALITY_TO_PLAIN
        )
    )


def _is_d_transition(transition: tuple[str, str]) -> bool:
    return transition in {("đ", "d"), ("d", "đ")}


def _der_category(transitions: Sequence[tuple[str, str]]) -> str:
    if transitions and all(_is_vowel_loss(item) for item in transitions):
        return "vowel_quality_loss"
    if transitions and all(_is_vowel_addition(item) for item in transitions):
        return "vowel_quality_addition"
    if transitions and all(_is_vowel_transition(item) for item in transitions):
        return "vowel_quality_substitution"
    if transitions and all(_is_d_transition(item) for item in transitions):
        return "d_stroke_loss_or_change"
    return "mixed_or_other"


def _transition_text(transitions: Sequence[tuple[str, str]]) -> str:
    return ";".join(f"{ref_symbol}→{hyp_symbol}" for ref_symbol, hyp_symbol in transitions)


def _event_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    def index_key(value: object) -> tuple[int, int]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, -1)

    operation_rank = {name: index for index, name in enumerate(VALID_OPERATIONS)}
    return (
        *(_sort_text(row[column]) for column in RUN_COLUMNS),
        _sort_text(row["utt_id"]),
        index_key(row["ref_index"]),
        index_key(row["hyp_index"]),
        operation_rank[str(row["operation"])],
    )


def build_breakdowns(event_path: str | Path) -> BreakdownResult:
    path = Path(event_path)
    if not path.is_file():
        raise ErrorBreakdownError(f"event CSV does not exist: {path}")

    operation_counts: defaultdict[RunKey, Counter[str]] = defaultdict(Counter)
    utterance_ids: defaultdict[RunKey, set[str]] = defaultdict(set)
    tone_counts: defaultdict[RunKey, Counter[str]] = defaultdict(Counter)
    der_counts: defaultdict[RunKey, Counter[str]] = defaultdict(Counter)
    orthographic_counts: defaultdict[RunKey, Counter[str]] = defaultdict(Counter)
    tone_totals: defaultdict[RunKey, Counter[str]] = defaultdict(Counter)
    der_totals: defaultdict[RunKey, Counter[str]] = defaultdict(Counter)
    event_rows: list[dict[str, object]] = []
    run_keys: set[RunKey] = set()
    seen_events: set[tuple[object, ...]] = set()
    utterance_conditions: dict[tuple[RunKey, str], tuple[str, str, str, str]] = {}
    input_event_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if columns != EVENT_COLUMNS:
            raise ErrorBreakdownError(
                f"{path}: expected exact aligned-v1 event columns {EVENT_COLUMNS}, "
                f"found {columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            input_event_rows += 1
            if None in row or any(row.get(column) is None for column in EVENT_COLUMNS):
                raise ErrorBreakdownError(
                    f"{path}: row {row_number} has missing or extra CSV cells"
                )
            if row["metric_version"] != METRIC_VERSION:
                raise ErrorBreakdownError(
                    f"{path}: row {row_number}: expected metric_version={METRIC_VERSION!r}, "
                    f"found {row['metric_version']!r}"
                )

            run_key = tuple(row[column] for column in RUN_COLUMNS)
            for column, value in zip(RUN_COLUMNS, run_key):
                if column != "lambda" and not value:
                    raise ErrorBreakdownError(
                        f"{path}: row {row_number}: run metadata {column!r} is empty"
                    )
            run_keys.add(run_key)

            utt_id = row["utt_id"]
            if not utt_id:
                raise ErrorBreakdownError(f"{path}: row {row_number}: utt_id is empty")
            condition_key = (run_key, utt_id)
            condition = (row["snr"], row["noise_type"], row["ref"], row["hyp"])
            if (
                condition_key in utterance_conditions
                and utterance_conditions[condition_key] != condition
            ):
                raise ErrorBreakdownError(
                    f"{path}: row {row_number}: inconsistent condition/text within utterance "
                    f"{utt_id!r}"
                )
            utterance_conditions[condition_key] = condition
            utterance_ids[run_key].add(utt_id)

            operation = row["operation"]
            if operation not in VALID_OPERATIONS:
                raise ErrorBreakdownError(
                    f"{path}: row {row_number}: unknown operation {operation!r}"
                )
            paired = operation in {"match", "substitution"}
            ref_required = operation != "insertion"
            hyp_required = operation != "deletion"
            ref_index = _parse_index(
                row["ref_index"],
                path=path,
                row_number=row_number,
                field="ref_index",
                required=ref_required,
            )
            hyp_index = _parse_index(
                row["hyp_index"],
                path=path,
                row_number=row_number,
                field="hyp_index",
                required=hyp_required,
            )
            if ref_required != bool(row["ref_token"]):
                raise ErrorBreakdownError(
                    f"{path}: row {row_number}: ref_token disagrees with operation {operation!r}"
                )
            if hyp_required != bool(row["hyp_token"]):
                raise ErrorBreakdownError(
                    f"{path}: row {row_number}: hyp_token disagrees with operation {operation!r}"
                )
            unique_event = (run_key, utt_id, operation, ref_index, hyp_index)
            if unique_event in seen_events:
                raise ErrorBreakdownError(
                    f"{path}: duplicate aligned event key at row {row_number}: {unique_event}"
                )
            seen_events.add(unique_event)
            operation_counts[run_key][operation] += 1

            tone_eligible = _parse_bool(
                row["tone_eligible"], path=path, row_number=row_number, field="tone_eligible"
            )
            tone_error = _parse_bool(
                row["tone_error"], path=path, row_number=row_number, field="tone_error"
            )
            diacritic_eligible = _parse_bool(
                row["diacritic_eligible"],
                path=path,
                row_number=row_number,
                field="diacritic_eligible",
            )
            diacritic_error = _parse_bool(
                row["diacritic_error"],
                path=path,
                row_number=row_number,
                field="diacritic_error",
            )
            if tone_error and not tone_eligible:
                raise ErrorBreakdownError(
                    f"{path}: row {row_number}: tone_error=true requires tone_eligible=true"
                )
            if diacritic_error and not diacritic_eligible:
                raise ErrorBreakdownError(
                    f"{path}: row {row_number}: diacritic_error=true requires "
                    "diacritic_eligible=true"
                )
            tone_totals[run_key]["eligible"] += int(tone_eligible)
            tone_totals[run_key]["errors"] += int(tone_error)
            der_totals[run_key]["eligible"] += int(diacritic_eligible)
            der_totals[run_key]["errors"] += int(diacritic_error)

            tone_category = ""
            if tone_error:
                tone_category = _tone_category(row, path=path, row_number=row_number)
                tone_counts[run_key][tone_category] += 1

            ref_tone_base = _normalized_base(row["ref_tone_base"])
            hyp_tone_base = _normalized_base(row["hyp_tone_base"])
            ref_plain_base = _normalized_base(row["ref_plain_base"])
            hyp_plain_base = _normalized_base(row["hyp_plain_base"])
            transitions = _quality_transitions(ref_tone_base, hyp_tone_base) if paired else ()
            der_category = ""
            if diacritic_error:
                if not paired:
                    raise ErrorBreakdownError(
                        f"{path}: row {row_number}: DER error must be a paired event"
                    )
                if ref_plain_base != hyp_plain_base:
                    raise ErrorBreakdownError(
                        f"{path}: row {row_number}: DER error requires equal plain bases"
                    )
                if ref_tone_base == hyp_tone_base:
                    raise ErrorBreakdownError(
                        f"{path}: row {row_number}: DER error requires different "
                        "tone-stripped bases"
                    )
                der_category = _der_category(transitions)
                der_counts[run_key][der_category] += 1

            ref_tone = row["ref_tone"]
            hyp_tone = row["hyp_tone"]
            has_tone_loss = (
                paired
                and ref_tone in VALID_TONES
                and hyp_tone in VALID_TONES
                and ref_tone != "ngang"
                and hyp_tone == "ngang"
                and ref_plain_base == hyp_plain_base
            )
            has_vowel_quality_loss = any(_is_vowel_loss(item) for item in transitions)
            has_d_stroke_loss = ("đ", "d") in transitions
            missing_diacritic = (
                has_tone_loss or has_vowel_quality_loss or has_d_stroke_loss
            )
            if missing_diacritic:
                orthographic_counts[run_key]["missing_diacritic"] += 1
            wrong_tone_mark = (
                paired
                and ref_plain_base == hyp_plain_base
                and ref_tone in VALID_TONES
                and hyp_tone in VALID_TONES
                and ref_tone != hyp_tone
                and hyp_tone != "ngang"
            )
            if wrong_tone_mark:
                orthographic_counts[run_key]["wrong_tone_mark"] += 1
            if der_category in {
                "vowel_quality_addition",
                "vowel_quality_substitution",
            }:
                orthographic_counts[run_key]["wrong_vowel_mark"] += 1

            if tone_error or diacritic_error or missing_diacritic:
                event_rows.append(
                    {
                        "metric_version": METRIC_VERSION,
                        **dict(zip(RUN_COLUMNS, run_key)),
                        "utt_id": utt_id,
                        "snr": row["snr"],
                        "noise_type": row["noise_type"],
                        "operation": operation,
                        "ref_token": unicodedata.normalize("NFC", row["ref_token"]),
                        "hyp_token": unicodedata.normalize("NFC", row["hyp_token"]),
                        "ref_index": "" if ref_index is None else ref_index,
                        "hyp_index": "" if hyp_index is None else hyp_index,
                        "ref_tone": ref_tone,
                        "hyp_tone": hyp_tone,
                        "ref_tone_base": unicodedata.normalize("NFC", row["ref_tone_base"]),
                        "hyp_tone_base": unicodedata.normalize("NFC", row["hyp_tone_base"]),
                        "ref_plain_base": unicodedata.normalize("NFC", row["ref_plain_base"]),
                        "hyp_plain_base": unicodedata.normalize("NFC", row["hyp_plain_base"]),
                        "tone_error": str(tone_error).lower(),
                        "diacritic_error": str(diacritic_error).lower(),
                        "tone_primary_category": tone_category,
                        "der_primary_category": der_category,
                        "quality_transitions": _transition_text(transitions),
                        "has_tone_loss": str(has_tone_loss).lower(),
                        "has_vowel_quality_loss": str(has_vowel_quality_loss).lower(),
                        "has_d_stroke_loss": str(has_d_stroke_loss).lower(),
                        "missing_diacritic": str(missing_diacritic).lower(),
                        "ref": unicodedata.normalize("NFC", row["ref"]),
                        "hyp": unicodedata.normalize("NFC", row["hyp"]),
                    }
                )

    if input_event_rows == 0:
        raise ErrorBreakdownError(f"event CSV is empty: {path}")

    ordered_runs = tuple(sorted(run_keys, key=_run_sort_key))
    wer_rows: list[dict[str, object]] = []
    ter_rows: list[dict[str, object]] = []
    der_rows: list[dict[str, object]] = []
    orthographic_rows: list[dict[str, object]] = []
    for run_key in ordered_runs:
        metadata = dict(zip(RUN_COLUMNS, run_key))
        operations = operation_counts[run_key]
        matches = operations["match"]
        substitutions = operations["substitution"]
        deletions = operations["deletion"]
        insertions = operations["insertion"]
        ref_units = matches + substitutions + deletions
        hyp_units = matches + substitutions + insertions
        word_errors = substitutions + deletions + insertions
        wer_rows.append(
            {
                "metric_version": METRIC_VERSION,
                **metadata,
                "n_utterances": len(utterance_ids[run_key]),
                "n_ref_tokens": ref_units,
                "n_hyp_tokens": hyp_units,
                "matches": matches,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "word_errors": word_errors,
                "substitution_rate": _rate(substitutions, ref_units),
                "deletion_rate": _rate(deletions, ref_units),
                "insertion_rate": _rate(insertions, ref_units),
                "wer": _rate(word_errors, ref_units),
            }
        )

        tone_eligible = tone_totals[run_key]["eligible"]
        tone_errors = tone_totals[run_key]["errors"]
        if sum(tone_counts[run_key].values()) != tone_errors:
            raise ErrorBreakdownError("TER primary categories do not reconcile")
        for category in TER_CATEGORY_ORDER:
            count = tone_counts[run_key][category]
            ter_rows.append(
                {
                    "metric_version": METRIC_VERSION,
                    **metadata,
                    "tone_eligible": tone_eligible,
                    "tone_errors": tone_errors,
                    "category": category,
                    "count": count,
                    "rate": _rate(count, tone_eligible),
                }
            )

        diacritic_eligible = der_totals[run_key]["eligible"]
        diacritic_errors = der_totals[run_key]["errors"]
        if sum(der_counts[run_key].values()) != diacritic_errors:
            raise ErrorBreakdownError("DER primary categories do not reconcile")
        for category in DER_CATEGORY_ORDER:
            count = der_counts[run_key][category]
            der_rows.append(
                {
                    "metric_version": METRIC_VERSION,
                    **metadata,
                    "diacritic_eligible": diacritic_eligible,
                    "diacritic_errors": diacritic_errors,
                    "category": category,
                    "count": count,
                    "rate": _rate(count, diacritic_eligible),
                }
            )

        for category in ORTHOGRAPHIC_CATEGORY_ORDER:
            orthographic_rows.append(
                {
                    "metric_version": METRIC_VERSION,
                    **metadata,
                    "category": category,
                    "count": orthographic_counts[run_key][category],
                    "definition": ORTHOGRAPHIC_DEFINITIONS[category],
                    "overlap_policy": "nonexclusive_feature_diagnostic",
                }
            )

    event_rows.sort(key=_event_sort_key)
    return BreakdownResult(
        wer_rows=tuple(wer_rows),
        ter_rows=tuple(ter_rows),
        der_rows=tuple(der_rows),
        orthographic_rows=tuple(orthographic_rows),
        event_rows=tuple(event_rows),
        input_event_rows=input_event_rows,
        run_keys=ordered_runs,
    )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _backup_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.bak")


@contextmanager
def _output_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / OUTPUT_LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ErrorBreakdownError(
            f"output directory is locked, or a stale lock remains: {lock_path}"
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


def _write_csv_temp(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> Path:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(columns),
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


def _commit_outputs(
    temporary_paths: Sequence[Path],
    destinations: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    backups = tuple(_backup_path(path) for path in destinations)
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
                temporary.replace(destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError as exc:
                    raise ErrorBreakdownError(
                        f"output appeared during commit; refusing to overwrite: {destination}"
                    ) from exc
                temporary.unlink()
            committed.append(destination)
    except Exception as error:
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
            raise ErrorBreakdownError(
                "output commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise
    for backup, _ in backed_up:
        if backup.exists():
            backup.unlink()


def write_breakdown_outputs(
    event_path: str | Path,
    output_dir: str | Path,
    result: BreakdownResult,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    directory = Path(output_dir)
    destinations = tuple(directory / name for name in OUTPUT_NAMES)
    source = Path(event_path).resolve()
    if any(path.resolve() == source for path in destinations):
        raise ErrorBreakdownError("refusing to overwrite the input event CSV")
    with _output_lock(directory):
        existing = [path for path in destinations if path.exists()]
        if existing and not overwrite:
            raise ErrorBreakdownError(
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
            raise ErrorBreakdownError(
                "temporary or backup output already exists: "
                + ", ".join(str(path) for path in stale)
            )
        specifications = (
            (result.wer_rows, WER_COLUMNS),
            (result.ter_rows, TER_COLUMNS),
            (result.der_rows, DER_COLUMNS),
            (result.orthographic_rows, ORTHOGRAPHIC_COLUMNS),
            (result.event_rows, DIACRITIC_EVENT_COLUMNS),
        )
        temporary_paths: list[Path] = []
        try:
            for destination, (rows, columns) in zip(destinations, specifications):
                temporary_paths.append(_write_csv_temp(destination, rows, columns))
            _commit_outputs(temporary_paths, destinations, overwrite=overwrite)
        except Exception:
            for temporary in temporary_paths:
                if temporary.exists():
                    temporary.unlink()
            raise
    return destinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic WER, TER, and DER breakdowns from aligned-v1 "
            "error_events.csv without re-aligning predictions."
        )
    )
    parser.add_argument("--events", required=True, help="aligned-v1 error_events.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = build_breakdowns(args.events)
        outputs = write_breakdown_outputs(
            args.events,
            args.out_dir,
            result,
            overwrite=args.overwrite,
        )
    except (ErrorBreakdownError, OSError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(f"PASS event rows: {result.input_event_rows}")
    print(f"PASS runs: {len(result.run_keys)}")
    print(f"PASS diacritic event rows: {len(result.event_rows)}")
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
