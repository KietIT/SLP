from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.analysis import (  # noqa: E402
    CANONICAL_PREDICTION_COLUMNS,
    METRIC_VERSION,
    RUN_METADATA_COLUMNS,
    PredictionValidationError,
    analyze_error_events,
    load_prediction_csv,
)


DEFAULT_PREDICTION_GLOB = "outputs/predictions/*/pred_*.csv"
DEFAULT_OUTPUT_DIR = Path("outputs/error_analysis")

RUN_COLUMNS = list(RUN_METADATA_COLUMNS)
CANONICAL_COLUMNS = list(CANONICAL_PREDICTION_COLUMNS)
EVENT_COLUMNS = [
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
SUMMARY_COLUMNS = [
    *RUN_COLUMNS,
    "n_utterances",
    "n_ref_tokens",
    "n_hyp_tokens",
    "n_events",
    "matches",
    "substitutions",
    "deletions",
    "insertions",
    "word_errors",
    "word_error_rate",
    "tone_eligible",
    "tone_errors",
    "tone_error_rate",
    "diacritic_eligible",
    "diacritic_errors",
    "diacritic_error_rate",
    "final_consonant_eligible",
    "final_consonant_errors",
    "final_consonant_error_rate",
]
OPERATIONS = ("match", "substitution", "deletion", "insertion")
OPERATION_COUNT_COLUMNS = {
    "match": "matches",
    "substitution": "substitutions",
    "deletion": "deletions",
    "insertion": "insertions",
}


class ErrorAnalysisError(ValueError):
    pass


def _sort_text(value: object) -> str:
    return str(value).casefold()


def discover_inputs(patterns: Sequence[str] | None = None) -> list[Path]:
    selected_patterns = list(patterns or [DEFAULT_PREDICTION_GLOB])
    candidates: list[Path] = []
    for pattern in selected_patterns:
        candidates.extend(Path(match) for match in glob.glob(pattern, recursive=True))

    inputs: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(candidates, key=lambda item: _sort_text(item.as_posix())):
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise ErrorAnalysisError(f"input is not a file: {path}")
        if path.suffix.lower() != ".csv":
            raise ErrorAnalysisError(f"input must be a CSV file: {path}")
        seen.add(resolved)
        inputs.append(path)
    if not inputs:
        joined = ", ".join(selected_patterns)
        raise ErrorAnalysisError(f"no prediction CSV files matched: {joined}")
    return inputs


def _read_prediction_file(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    if columns != CANONICAL_COLUMNS:
        raise ErrorAnalysisError(
            f"{path}: expected exact canonical columns {CANONICAL_COLUMNS}, found {columns}"
        )
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(row.get(column) is None for column in CANONICAL_COLUMNS):
            raise ErrorAnalysisError(f"{path}: row {row_number} has missing or extra CSV cells")
    try:
        return load_prediction_csv(path)
    except (PredictionValidationError, FileNotFoundError) as exc:
        raise ErrorAnalysisError(str(exc)) from exc


def load_prediction_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_keys: dict[tuple[str, ...], Path] = {}
    sample_conditions: dict[tuple[str, str], tuple[str, str, str, Path]] = {}
    for path in paths:
        current = _read_prediction_file(path)
        for row in current:
            key = tuple(row[column] for column in [*RUN_COLUMNS, "utt_id"])
            if key in seen_keys:
                raise ErrorAnalysisError(
                    f"duplicate prediction key across {seen_keys[key]} and {path}: {key}"
                )
            seen_keys[key] = path
            sample_key = (row["dataset"], row["utt_id"])
            condition = (row["ref"], row["snr"], row["noise_type"])
            if sample_key in sample_conditions:
                expected = sample_conditions[sample_key]
                if condition != expected[:3]:
                    raise ErrorAnalysisError(
                        f"inconsistent ref/snr/noise_type for {sample_key} across "
                        f"{expected[3]} and {path}"
                    )
            else:
                sample_conditions[sample_key] = (*condition, path)
            rows.append(row)
    rows.sort(key=lambda row: tuple(_sort_text(row[column]) for column in [*RUN_COLUMNS, "utt_id"]))
    return rows


def _event_value(event: object, name: str) -> Any:
    if isinstance(event, Mapping):
        if name not in event:
            raise ErrorAnalysisError(f"shared analysis event is missing field {name!r}")
        return event[name]
    if not hasattr(event, name):
        raise ErrorAnalysisError(f"shared analysis event is missing field {name!r}")
    return getattr(event, name)


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _operation(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "equal": "match",
        "replace": "substitution",
        "substitute": "substitution",
        "delete": "deletion",
        "insert": "insertion",
    }
    text = aliases.get(text, text)
    if text not in OPERATIONS:
        raise ErrorAnalysisError(f"shared analysis returned unknown operation {value!r}")
    return text


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator / max(denominator, 1):.12f}"


def build_error_analysis(
    prediction_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    event_rows: list[dict[str, object]] = []
    summaries: dict[tuple[str, ...], dict[str, object]] = {}

    for utterance_index, prediction in enumerate(prediction_rows):
        run_key = tuple(prediction[column] for column in RUN_COLUMNS)
        if run_key not in summaries:
            summaries[run_key] = {
                **{column: prediction[column] for column in RUN_COLUMNS},
                **{column: 0 for column in SUMMARY_COLUMNS if column not in RUN_COLUMNS},
            }
        summary = summaries[run_key]
        summary["n_utterances"] = int(summary["n_utterances"]) + 1

        shared_events = list(
            analyze_error_events(
                prediction["ref"],
                prediction["hyp"],
                utt_id=prediction["utt_id"],
                utterance_index=utterance_index,
                metadata=prediction,
            )
        )
        for shared_event in shared_events:
            operation = _operation(_event_value(shared_event, "operation"))
            event = {
                "metric_version": METRIC_VERSION,
                **{column: prediction[column] for column in RUN_COLUMNS},
                "utt_id": prediction["utt_id"],
                "snr": prediction["snr"],
                "noise_type": prediction["noise_type"],
                "operation": operation,
                "ref_token": _cell(_event_value(shared_event, "ref_token")),
                "hyp_token": _cell(_event_value(shared_event, "hyp_token")),
                "ref_index": _cell(_event_value(shared_event, "ref_index")),
                "hyp_index": _cell(_event_value(shared_event, "hyp_index")),
                "ref": _cell(_event_value(shared_event, "ref_text")),
                "hyp": _cell(_event_value(shared_event, "hyp_text")),
                "ref_tone_base": _cell(
                    _event_value(shared_event, "ref_tone_base")
                ),
                "hyp_tone_base": _cell(
                    _event_value(shared_event, "hyp_tone_base")
                ),
                "ref_plain_base": _cell(
                    _event_value(shared_event, "ref_plain_base")
                ),
                "hyp_plain_base": _cell(
                    _event_value(shared_event, "hyp_plain_base")
                ),
                "ref_tone": _cell(_event_value(shared_event, "ref_tone")),
                "hyp_tone": _cell(_event_value(shared_event, "hyp_tone")),
                "ref_coda": _cell(_event_value(shared_event, "ref_coda")),
                "hyp_coda": _cell(_event_value(shared_event, "hyp_coda")),
                "tone_eligible": _cell(
                    _event_value(shared_event, "tone_eligible")
                ),
                "tone_error": _cell(_event_value(shared_event, "tone_error")),
                "diacritic_eligible": _cell(
                    _event_value(shared_event, "diacritic_eligible")
                ),
                "diacritic_error": _cell(_event_value(shared_event, "diacritic_error")),
                "final_consonant_eligible": _cell(
                    _event_value(shared_event, "final_consonant_eligible")
                ),
                "final_consonant_error": _cell(
                    _event_value(shared_event, "final_consonant_error")
                ),
                "short_word_deletion": _cell(
                    _event_value(shared_event, "short_word_deletion")
                ),
            }
            event_rows.append(event)

            summary["n_events"] = int(summary["n_events"]) + 1
            count_column = OPERATION_COUNT_COLUMNS[operation]
            summary[count_column] = int(summary[count_column]) + 1
            if operation != "insertion":
                summary["n_ref_tokens"] = int(summary["n_ref_tokens"]) + 1
            if operation != "deletion":
                summary["n_hyp_tokens"] = int(summary["n_hyp_tokens"]) + 1
            for event_column, summary_column in (
                ("tone_error", "tone_errors"),
                ("diacritic_error", "diacritic_errors"),
                ("final_consonant_error", "final_consonant_errors"),
            ):
                if _event_value(shared_event, event_column) is True:
                    summary[summary_column] = int(summary[summary_column]) + 1
            for event_column, summary_column in (
                ("tone_eligible", "tone_eligible"),
                ("diacritic_eligible", "diacritic_eligible"),
                ("final_consonant_eligible", "final_consonant_eligible"),
            ):
                if _event_value(shared_event, event_column) is True:
                    summary[summary_column] = int(summary[summary_column]) + 1

    summary_rows = [summaries[key] for key in sorted(summaries, key=lambda item: tuple(map(_sort_text, item)))]
    for summary in summary_rows:
        edit_total = sum(int(summary[name]) for name in ("matches", "substitutions", "deletions", "insertions"))
        ref_total = sum(int(summary[name]) for name in ("matches", "substitutions", "deletions"))
        hyp_total = sum(int(summary[name]) for name in ("matches", "substitutions", "insertions"))
        if edit_total != int(summary["n_events"]):
            raise ErrorAnalysisError("summary edit operations do not reconcile with n_events")
        if ref_total != int(summary["n_ref_tokens"]):
            raise ErrorAnalysisError("summary edit operations do not reconcile with n_ref_tokens")
        if hyp_total != int(summary["n_hyp_tokens"]):
            raise ErrorAnalysisError("summary edit operations do not reconcile with n_hyp_tokens")
        word_errors = sum(int(summary[name]) for name in ("substitutions", "deletions", "insertions"))
        summary["word_errors"] = word_errors
        summary["word_error_rate"] = _rate(word_errors, int(summary["n_ref_tokens"]))
        summary["tone_error_rate"] = _rate(
            int(summary["tone_errors"]), int(summary["tone_eligible"])
        )
        summary["diacritic_error_rate"] = _rate(
            int(summary["diacritic_errors"]), int(summary["diacritic_eligible"])
        )
        summary["final_consonant_error_rate"] = _rate(
            int(summary["final_consonant_errors"]),
            int(summary["final_consonant_eligible"]),
        )
    return event_rows, summary_rows


def _write_csv_temp(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def write_outputs(
    output_dir: Path,
    event_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    *,
    overwrite: bool = False,
    protected_inputs: Sequence[Path] = (),
) -> tuple[Path, Path]:
    event_path = output_dir / "error_events.csv"
    summary_path = output_dir / "error_summary.csv"
    outputs = (event_path, summary_path)
    protected = {path.resolve() for path in protected_inputs}
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ErrorAnalysisError("output paths collide")
    for path in outputs:
        if path.resolve() in protected:
            raise ErrorAnalysisError(f"refusing to overwrite an input prediction: {path}")
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise ErrorAnalysisError(
            "output already exists; use a new --out-dir or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    temporary_paths: list[Path] = []
    try:
        temporary_paths.append(_write_csv_temp(event_path, event_rows, EVENT_COLUMNS))
        temporary_paths.append(_write_csv_temp(summary_path, summary_rows, SUMMARY_COLUMNS))
        for temporary, destination in zip(temporary_paths, outputs):
            temporary.replace(destination)
    except Exception:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
        raise
    return event_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic token-level ASR error events and per-run summaries."
    )
    parser.add_argument(
        "--pred-glob",
        action="append",
        default=None,
        help=(
            "Canonical prediction glob; repeatable. Defaults to "
            f"{DEFAULT_PREDICTION_GLOB!r}."
        ),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        inputs = discover_inputs(args.pred_glob)
        output_dir = Path(args.out_dir)
        rows = load_prediction_rows(inputs)
        event_rows, summary_rows = build_error_analysis(rows)
        event_path, summary_path = write_outputs(
            output_dir,
            event_rows,
            summary_rows,
            overwrite=args.overwrite,
            protected_inputs=inputs,
        )
        print(f"PASS prediction files: {len(inputs)}")
        print(f"PASS prediction rows: {len(rows)}")
        print(f"wrote {event_path} ({len(event_rows)} events)")
        print(f"wrote {summary_path} ({len(summary_rows)} runs)")
    except ErrorAnalysisError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
