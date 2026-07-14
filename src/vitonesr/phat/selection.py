from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import ABLATION_RESULT_COLUMNS


EXPECTED_LAMBDAS = (0.0, 0.05, 0.1, 0.3, 0.5)


@dataclass(frozen=True)
class LambdaSummary:
    lambda_value: float
    train_type: str
    clean_wer: float | None
    clean_cer: float | None
    guard_wer: float
    guard_cer: float
    wer_delta: float
    cer_delta: float
    low_snr_ter: float
    low_snr_der: float
    score: float
    eligible: bool
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    selected_lambda: float
    summaries: tuple[LambdaSummary, ...]
    warnings: tuple[str, ...]


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Ablation result file does not exist: {result_path}")
    with result_path.open("r", encoding="utf-8", newline="") as result_file:
        reader = csv.DictReader(result_file)
        columns = list(reader.fieldnames or [])
        missing = [column for column in ABLATION_RESULT_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"Ablation result is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Ablation result file is empty: {result_path}")
    return rows


def _as_float(row: Mapping[str, Any], name: str) -> float:
    value = str(row.get(name, "")).strip()
    if value in {"", "not_available", "nan", "NaN"}:
        raise ValueError(f"Metric {name} is unavailable for lambda={row.get('lambda')}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Metric {name} is non-finite for lambda={row.get('lambda')}")
    return number


def _find_single_row(
    rows: Sequence[dict[str, str]],
    *,
    split: str,
    snr: str,
    noise_type: str = "all",
) -> dict[str, str] | None:
    matches = [
        row
        for row in rows
        if row.get("split") == split and row.get("snr") == snr and row.get("noise_type") == noise_type
    ]
    if len(matches) > 1:
        raise ValueError(f"Duplicate aggregate rows for split={split}, snr={snr}, noise_type={noise_type}")
    return matches[0] if matches else None


def _weighted_metric(rows: Sequence[dict[str, str]], metric: str) -> float:
    weighted_sum = 0.0
    total_samples = 0
    for row in rows:
        count = int(row["num_samples"])
        weighted_sum += _as_float(row, metric) * count
        total_samples += count
    if total_samples == 0:
        raise ValueError(f"Cannot average {metric}: zero samples")
    return weighted_sum / total_samples


def select_best_lambda_from_rows(
    rows: Sequence[dict[str, str]],
    selection: Mapping[str, Any],
    *,
    required_lambdas: Sequence[float] = EXPECTED_LAMBDAS,
) -> SelectionResult:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["lambda"]), []).append(row)
    warnings: list[str] = []
    missing = sorted(set(float(value) for value in required_lambdas) - set(grouped))
    if missing:
        warnings.append(f"Missing lambda results: {missing}")
    if 0.0 not in grouped:
        raise ValueError("Ordinary LoRA baseline lambda=0 is required for selection")

    guard_split = str(selection.get("guard_split", "all"))
    guard_snr = str(selection.get("guard_snr", "all"))
    baseline_guard = _find_single_row(grouped[0.0], split=guard_split, snr=guard_snr)
    if baseline_guard is None:
        raise ValueError(f"Missing lambda=0 guard row split={guard_split}, snr={guard_snr}")
    baseline_wer = _as_float(baseline_guard, "wer")
    baseline_cer = _as_float(baseline_guard, "cer")
    max_wer_increase = float(selection.get("max_wer_absolute_increase", 0.05))
    max_cer_increase = float(selection.get("max_cer_absolute_increase", 0.03))
    low_snr_values = {str(int(float(value))) if float(value).is_integer() else str(float(value)) for value in selection.get("low_snr", [0, 5])}
    ter_weight = float(selection.get("ter_weight", 0.5))
    der_weight = float(selection.get("der_weight", 0.5))
    if ter_weight < 0 or der_weight < 0 or ter_weight + der_weight <= 0:
        raise ValueError("TER/DER selection weights must be non-negative with a positive sum")
    normalizer = ter_weight + der_weight
    ter_weight /= normalizer
    der_weight /= normalizer

    summaries: list[LambdaSummary] = []
    for lambda_value in sorted(grouped):
        lambda_rows = grouped[lambda_value]
        guard_row = _find_single_row(lambda_rows, split=guard_split, snr=guard_snr)
        if guard_row is None:
            warnings.append(f"lambda={lambda_value:g}: missing guard row")
            continue
        low_rows = [
            row
            for row in lambda_rows
            if row.get("split") == "noisy"
            and row.get("snr") in low_snr_values
            and row.get("noise_type") == "all"
        ]
        observed_low_snr = {row["snr"] for row in low_rows}
        if observed_low_snr != low_snr_values:
            warnings.append(
                f"lambda={lambda_value:g}: low-SNR rows are incomplete; expected={sorted(low_snr_values)}, observed={sorted(observed_low_snr)}"
            )
            continue
        guard_wer = _as_float(guard_row, "wer")
        guard_cer = _as_float(guard_row, "cer")
        wer_delta = guard_wer - baseline_wer
        cer_delta = guard_cer - baseline_cer
        low_ter = _weighted_metric(low_rows, "ter")
        low_der = _weighted_metric(low_rows, "der")
        score = ter_weight * low_ter + der_weight * low_der
        eligible = wer_delta <= max_wer_increase and cer_delta <= max_cer_increase
        if lambda_value == 0.0 and not bool(selection.get("allow_lambda_zero", True)):
            eligible = False
            reason = "lambda=0 excluded by config"
        elif not eligible:
            reason = "WER/CER degradation exceeds configured guard"
        else:
            reason = "passes WER/CER guard"
        clean_row = _find_single_row(lambda_rows, split="clean", snr="clean", noise_type="clean")
        if clean_row is None:
            warnings.append(f"lambda={lambda_value:g}: missing clean aggregate row")
        expected_total = selection.get("expected_total_samples")
        if expected_total is not None and int(guard_row["num_samples"]) != int(expected_total):
            warnings.append(
                f"lambda={lambda_value:g}: guard row has {guard_row['num_samples']} samples, expected {int(expected_total)}"
            )
        expected_condition = selection.get("expected_samples_per_condition")
        if expected_condition is not None:
            if clean_row is not None and int(clean_row["num_samples"]) != int(expected_condition):
                warnings.append(
                    f"lambda={lambda_value:g}: clean row has {clean_row['num_samples']} samples, expected {int(expected_condition)}"
                )
            for low_row in low_rows:
                if int(low_row["num_samples"]) != int(expected_condition):
                    warnings.append(
                        f"lambda={lambda_value:g}: SNR {low_row['snr']} row has {low_row['num_samples']} samples, expected {int(expected_condition)}"
                    )
        summaries.append(
            LambdaSummary(
                lambda_value=lambda_value,
                train_type=str(guard_row["train_type"]),
                clean_wer=_as_float(clean_row, "wer") if clean_row else None,
                clean_cer=_as_float(clean_row, "cer") if clean_row else None,
                guard_wer=guard_wer,
                guard_cer=guard_cer,
                wer_delta=wer_delta,
                cer_delta=cer_delta,
                low_snr_ter=low_ter,
                low_snr_der=low_der,
                score=score,
                eligible=eligible,
                reason=reason,
            )
        )
    candidates = [summary for summary in summaries if summary.eligible]
    if not candidates:
        raise ValueError("No lambda passes the configured WER/CER guard with complete low-SNR metrics")
    selected = min(
        candidates,
        key=lambda summary: (
            summary.score,
            summary.guard_wer,
            summary.guard_cer,
            summary.lambda_value,
        ),
    )
    return SelectionResult(
        selected_lambda=selected.lambda_value,
        summaries=tuple(summaries),
        warnings=tuple(warnings),
    )


def select_best_lambda(
    results_path: str | Path,
    selection: Mapping[str, Any],
    report_path: str | Path,
    *,
    require_complete: bool = True,
    overwrite: bool = False,
) -> SelectionResult:
    rows = _read_rows(results_path)
    result = select_best_lambda_from_rows(rows, selection)
    if require_complete and result.warnings:
        raise ValueError("Cannot produce a complete best-lambda report: " + "; ".join(result.warnings))
    if Path(report_path).exists() and not overwrite:
        raise FileExistsError(f"Best-lambda report already exists: {report_path}. Use --overwrite explicitly.")
    _write_report(report_path, results_path, result, selection)
    return result


def _format_metric(value: float | None) -> str:
    return "not_available" if value is None else f"{value:.6f}"


def _write_report(
    report_path: str | Path,
    results_path: str | Path,
    result: SelectionResult,
    selection: Mapping[str, Any],
) -> None:
    low_snr = ", ".join(str(value) for value in selection.get("low_snr", [0, 5]))
    lines = [
        "# Best Lambda Report",
        "",
        f"- Source results: `{results_path}`",
        f"- Low-SNR priority: `{low_snr} dB`",
        f"- Maximum absolute WER increase: `{float(selection.get('max_wer_absolute_increase', 0.05)):.6f}`",
        f"- Maximum absolute CER increase: `{float(selection.get('max_cer_absolute_increase', 0.03)):.6f}`",
        "",
        "## Lambda comparison",
        "",
        "| lambda | train type | clean WER | clean CER | guard WER | guard CER | delta WER | delta CER | low-SNR TER | low-SNR DER | score | eligible |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for summary in result.summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{summary.lambda_value:g}",
                    summary.train_type,
                    _format_metric(summary.clean_wer),
                    _format_metric(summary.clean_cer),
                    _format_metric(summary.guard_wer),
                    _format_metric(summary.guard_cer),
                    f"{summary.wer_delta:+.6f}",
                    f"{summary.cer_delta:+.6f}",
                    _format_metric(summary.low_snr_ter),
                    _format_metric(summary.low_snr_der),
                    _format_metric(summary.score),
                    "yes" if summary.eligible else "no",
                ]
            )
            + " |"
        )
    selected = next(summary for summary in result.summaries if summary.lambda_value == result.selected_lambda)
    lines.extend(
        [
            "",
            "## Selected lambda",
            "",
            f"**lambda = {result.selected_lambda:g}**",
            "",
            "This lambda passes the configured WER/CER degradation guard and has the best weighted low-SNR TER/DER score. Ties are resolved by lower WER, then lower CER, then smaller lambda.",
            "",
            f"- Low-SNR TER: `{selected.low_snr_ter:.6f}`",
            f"- Low-SNR DER: `{selected.low_snr_der:.6f}`",
            f"- WER delta versus lambda 0: `{selected.wer_delta:+.6f}`",
            f"- CER delta versus lambda 0: `{selected.cer_delta:+.6f}`",
        ]
    )
    if result.warnings:
        lines.extend(["", "## Limitations and warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.extend(["", "## Limitations and warnings", "", "- All five configured lambda values were present with the required aggregate rows."])
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
