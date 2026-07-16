from __future__ import annotations

import csv
import json
import math
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.vitonesr.analysis import METRIC_VERSION

from .evaluation import ABLATION_RESULT_COLUMNS
from .protocol import (
    DECISION_VERSION,
    canonical_sha256,
    checkpoint_inference_sha256,
    evaluation_contract_sha256,
    is_sha256,
    load_split_lock,
    selection_rule_sha256,
    sha256_file,
    source_test_evaluation_contract_sha256,
    training_contract_sha256,
    verify_checkpoint_config,
)


EXPECTED_LAMBDAS = (0.0, 0.05, 0.1, 0.3, 0.5)
SELECTION_METRICS = ("wer", "cer", "ter", "der", "fcer", "swdr")
LOCKED_ROLES = ("ordinary_baseline", "selected_method", "locked_control")
CONTROL_STRATEGIES = (
    "best_eligible_non_selected_tone_aware",
    "fixed_preregistered_tone_aware",
)


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
    low_snr_ter_coverage_ratio: float
    low_snr_der_coverage_ratio: float
    low_snr_fcer_coverage_ratio: float
    score: float
    eligible: bool
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    selected_lambda: float
    locked_control_lambda: float
    locked_control_strategy: str
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
    numerator = 0
    denominator = 0
    for row in rows:
        numerator += int(row[f"{metric}_numerator"])
        denominator += int(row[f"{metric}_denominator"])
    if denominator == 0:
        raise ValueError(f"Cannot aggregate {metric}: zero reference units")
    return numerator / denominator


def _metric_denominator(rows: Sequence[dict[str, str]], metric: str) -> int:
    denominator = sum(int(row[f"{metric}_denominator"]) for row in rows)
    if denominator <= 0:
        raise ValueError(f"Cannot compute {metric} coverage: zero denominator")
    return denominator


def _summary_rank(summary: LambdaSummary) -> tuple[float, float, float, float]:
    return (
        summary.score,
        summary.guard_wer,
        summary.guard_cer,
        summary.lambda_value,
    )


def choose_locked_control_lambda(
    summaries: Sequence[LambdaSummary],
    *,
    selected_lambda: float,
    selection: Mapping[str, Any],
) -> tuple[float, str]:
    """Apply an explicit, pre-registered control policy.

    The control is never inferred from historical defaults such as 0.05 or 0.1.
    It is either the next best eligible tone-aware candidate under the locked
    ranking rule, or an explicitly configured fixed tone-aware candidate.
    """

    strategy = str(selection.get("locked_control_strategy", "")).strip()
    if strategy not in CONTROL_STRATEGIES:
        raise ValueError(
            "selection.locked_control_strategy must be one of: "
            + ", ".join(CONTROL_STRATEGIES)
        )
    tone_aware = [
        summary
        for summary in summaries
        if summary.train_type == "tone_aware_lora"
        and summary.lambda_value > 0.0
        and not math.isclose(
            summary.lambda_value,
            selected_lambda,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if strategy == "best_eligible_non_selected_tone_aware":
        candidates = [summary for summary in tone_aware if summary.eligible]
        if not candidates:
            raise ValueError(
                "No eligible non-selected tone-aware lambda is available for "
                "the locked control"
            )
        return min(candidates, key=_summary_rank).lambda_value, strategy

    raw_fixed = selection.get("locked_control_lambda")
    try:
        fixed = float(raw_fixed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fixed_preregistered_tone_aware requires "
            "selection.locked_control_lambda"
        ) from exc
    if not math.isfinite(fixed) or fixed <= 0.0:
        raise ValueError("selection.locked_control_lambda must be finite and positive")
    matches = [
        summary
        for summary in tone_aware
        if math.isclose(summary.lambda_value, fixed, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            "The pre-registered locked control must be one evaluated, "
            "non-selected tone-aware lambda"
        )
    return matches[0].lambda_value, strategy


def _require_dev_selection_provenance(
    rows: Sequence[dict[str, str]],
    selection: Mapping[str, Any],
) -> tuple[str, str]:
    if any(str(row.get("split", "")).strip().casefold() == "test" for row in rows):
        raise ValueError("Lambda selection refuses rows whose split is test")
    expected = str(selection.get("required_evaluation_split", "dev")).strip().casefold()
    observed = {
        str(row.get("evaluation_split", "")).strip().casefold()
        for row in rows
    }
    if observed != {expected} or expected != "dev":
        raise ValueError(
            "Lambda selection requires evaluation_split=dev only; "
            f"observed={sorted(observed)}"
        )
    manifest_hashes = {
        str(row.get("manifest_sha256", "")).strip().casefold()
        for row in rows
    }
    if len(manifest_hashes) != 1 or any(not is_sha256(value) for value in manifest_hashes):
        raise ValueError(
            "Lambda selection requires one non-empty 64-character manifest_sha256"
        )
    observed_manifest_hash = next(iter(manifest_hashes))
    configured_manifest_hash = str(
        selection.get("expected_manifest_sha256", "")
    ).strip().casefold()
    if (
        not is_sha256(configured_manifest_hash)
        or observed_manifest_hash != configured_manifest_hash
    ):
        raise ValueError(
            "Lambda selection manifest_sha256 does not match the locked config"
        )
    if {str(row.get("metric_version", "")) for row in rows} != {METRIC_VERSION}:
        raise ValueError(
            f"Lambda selection requires metric_version={METRIC_VERSION}"
        )
    if selection.get("require_full_manifest") is not True:
        raise ValueError("Lambda selection requires require_full_manifest=true")
    if {str(row.get("evaluation_scope", "")) for row in rows} != {"full_manifest"}:
        raise ValueError("Lambda selection refuses partial evaluation scope")
    if {str(row.get("training_scope", "")) for row in rows} != {"formal"}:
        raise ValueError("Lambda selection refuses smoke-trained checkpoints")
    evaluation_contracts = {
        str(row.get("evaluation_contract_sha256", "")).strip().casefold()
        for row in rows
    }
    configured_evaluation_contract = str(
        selection.get("expected_evaluation_contract_sha256", "")
    ).strip().casefold()
    if (
        len(evaluation_contracts) != 1
        or any(not is_sha256(value) for value in evaluation_contracts)
        or not is_sha256(configured_evaluation_contract)
        or next(iter(evaluation_contracts)) != configured_evaluation_contract
    ):
        raise ValueError(
            "Lambda selection requires one locked evaluation_contract_sha256"
        )
    selected_hashes = {
        str(row.get("selected_rows_sha256", "")).strip().casefold()
        for row in rows
    }
    if len(selected_hashes) != 1 or any(not is_sha256(value) for value in selected_hashes):
        raise ValueError(
            "Lambda selection requires one valid selected_rows_sha256"
        )
    if any(not is_sha256(row.get("checkpoint_sha256")) for row in rows):
        raise ValueError("Lambda selection requires checkpoint_sha256 provenance")
    if any(not is_sha256(row.get("training_contract_sha256")) for row in rows):
        raise ValueError(
            "Lambda selection requires training_contract_sha256 provenance"
        )
    for field in (
        "method_lock_sha256",
        "method_identity_sha256",
        "environment_artifact_sha256",
        "environment_identity_sha256",
        "source_tree_sha256",
    ):
        identities = {
            str(row.get(field, "")).strip().casefold() for row in rows
        }
        if len(identities) != 1 or any(not is_sha256(value) for value in identities):
            raise ValueError(
                f"Lambda selection requires one valid {field} across all runs"
            )
    for row in rows:
        if int(row.get("num_samples", 0)) < 1:
            raise ValueError("Lambda selection requires positive num_samples")
        for metric in SELECTION_METRICS:
            numerator = int(row[f"{metric}_numerator"])
            denominator = int(row[f"{metric}_denominator"])
            if numerator < 0 or denominator < 0:
                raise ValueError(
                    f"Metric {metric} has negative numerator/denominator"
                )
            if metric in {"ter", "der", "fcer", "swdr"} and numerator > denominator:
                raise ValueError(
                    f"Metric {metric} numerator exceeds its eligible denominator"
                )
            expected_rate = numerator / max(denominator, 1)
            observed_rate = _as_float(row, metric)
            if not math.isclose(
                observed_rate,
                expected_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Metric {metric} does not match its numerator/denominator"
                )
        word_denominator = int(row["wer_denominator"])
        if word_denominator <= 0:
            raise ValueError("Metric coverage requires a positive WER denominator")
        for metric in ("ter", "der", "fcer"):
            expected_coverage = int(row[f"{metric}_denominator"]) / word_denominator
            observed_coverage = _as_float(row, f"{metric}_coverage")
            if not 0.0 <= observed_coverage <= 1.0 or not math.isclose(
                observed_coverage,
                expected_coverage,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Metric {metric} coverage does not match eligible/reference units"
                )
    return expected, observed_manifest_hash


def select_best_lambda_from_rows(
    rows: Sequence[dict[str, str]],
    selection: Mapping[str, Any],
    *,
    required_lambdas: Sequence[float] = EXPECTED_LAMBDAS,
) -> SelectionResult:
    _require_dev_selection_provenance(rows, selection)
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
    if (
        not math.isfinite(max_wer_increase)
        or not math.isfinite(max_cer_increase)
        or max_wer_increase < 0
        or max_cer_increase < 0
    ):
        raise ValueError("WER/CER degradation guards must be finite and non-negative")
    low_snr_numbers = [float(value) for value in selection.get("low_snr", [0, 5])]
    if not low_snr_numbers or any(
        not math.isfinite(value) for value in low_snr_numbers
    ):
        raise ValueError("low_snr must contain finite numeric values")
    low_snr_values = {
        str(int(value)) if value.is_integer() else str(value)
        for value in low_snr_numbers
    }
    baseline_low_rows = [
        row
        for row in grouped[0.0]
        if row.get("split") == "noisy"
        and row.get("snr") in low_snr_values
        and row.get("noise_type") == "all"
    ]
    if (
        {row.get("snr") for row in baseline_low_rows} != low_snr_values
        or len(baseline_low_rows) != len(low_snr_values)
    ):
        raise ValueError("Ordinary LoRA baseline lacks complete low-SNR rows")
    baseline_ter_denominator = _metric_denominator(baseline_low_rows, "ter")
    baseline_der_denominator = _metric_denominator(baseline_low_rows, "der")
    baseline_fcer_denominator = _metric_denominator(baseline_low_rows, "fcer")
    min_ter_coverage = float(
        selection.get("min_ter_coverage_ratio_vs_baseline", 0.98)
    )
    min_der_coverage = float(
        selection.get("min_der_coverage_ratio_vs_baseline", 0.98)
    )
    min_fcer_coverage = float(
        selection.get("min_fcer_coverage_ratio_vs_baseline", 0.98)
    )
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in (min_ter_coverage, min_der_coverage, min_fcer_coverage)
    ):
        raise ValueError(
            "TER/DER/FCER coverage ratios must be finite values in [0, 1]"
        )
    ter_weight = float(selection.get("ter_weight", 0.5))
    der_weight = float(selection.get("der_weight", 0.5))
    if (
        not math.isfinite(ter_weight)
        or not math.isfinite(der_weight)
        or ter_weight < 0
        or der_weight < 0
        or ter_weight + der_weight <= 0
    ):
        raise ValueError("TER/DER selection weights must be non-negative with a positive sum")
    normalizer = ter_weight + der_weight
    ter_weight /= normalizer
    der_weight /= normalizer

    summaries: list[LambdaSummary] = []
    for lambda_value in sorted(grouped):
        lambda_rows = grouped[lambda_value]
        for field in (
            "seed",
            "train_type",
            "checkpoint_sha256",
            "training_contract_sha256",
        ):
            observed_values = {str(row.get(field, "")) for row in lambda_rows}
            if len(observed_values) != 1:
                raise ValueError(
                    f"lambda={lambda_value:g} mixes multiple {field} values"
                )
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
        if (
            observed_low_snr != low_snr_values
            or len(low_rows) != len(low_snr_values)
        ):
            warnings.append(
                f"lambda={lambda_value:g}: low-SNR rows are incomplete or duplicated; "
                f"expected={sorted(low_snr_values)}, observed={sorted(observed_low_snr)}, "
                f"row_count={len(low_rows)}"
            )
            continue
        guard_wer = _as_float(guard_row, "wer")
        guard_cer = _as_float(guard_row, "cer")
        wer_delta = guard_wer - baseline_wer
        cer_delta = guard_cer - baseline_cer
        low_ter = _weighted_metric(low_rows, "ter")
        low_der = _weighted_metric(low_rows, "der")
        ter_coverage_ratio = (
            _metric_denominator(low_rows, "ter") / baseline_ter_denominator
        )
        der_coverage_ratio = (
            _metric_denominator(low_rows, "der") / baseline_der_denominator
        )
        fcer_coverage_ratio = (
            _metric_denominator(low_rows, "fcer") / baseline_fcer_denominator
        )
        score = ter_weight * low_ter + der_weight * low_der
        error_guard_passes = (
            wer_delta <= max_wer_increase and cer_delta <= max_cer_increase
        )
        coverage_guard_passes = (
            ter_coverage_ratio >= min_ter_coverage
            and der_coverage_ratio >= min_der_coverage
            and fcer_coverage_ratio >= min_fcer_coverage
        )
        eligible = error_guard_passes and coverage_guard_passes
        if lambda_value == 0.0 and not bool(selection.get("allow_lambda_zero", True)):
            eligible = False
            reason = "lambda=0 excluded by config"
        elif not error_guard_passes:
            reason = "WER/CER degradation exceeds configured guard"
        elif not coverage_guard_passes:
            reason = "TER/DER/FCER coverage falls below the ordinary-LoRA guard"
        else:
            reason = "passes WER/CER and TER/DER/FCER coverage guards"
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
                low_snr_ter_coverage_ratio=ter_coverage_ratio,
                low_snr_der_coverage_ratio=der_coverage_ratio,
                low_snr_fcer_coverage_ratio=fcer_coverage_ratio,
                score=score,
                eligible=eligible,
                reason=reason,
            )
        )
    candidates = [summary for summary in summaries if summary.eligible]
    if not candidates:
        raise ValueError(
            "No lambda passes the configured WER/CER and TER/DER/FCER "
            "coverage guards with complete low-SNR metrics"
        )
    selected = min(
        candidates,
        key=_summary_rank,
    )
    control_lambda, control_strategy = choose_locked_control_lambda(
        summaries,
        selected_lambda=selected.lambda_value,
        selection=selection,
    )
    return SelectionResult(
        selected_lambda=selected.lambda_value,
        locked_control_lambda=control_lambda,
        locked_control_strategy=control_strategy,
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
    evaluation_split, manifest_sha256 = _require_dev_selection_provenance(rows, selection)
    result = select_best_lambda_from_rows(rows, selection)
    if require_complete and result.warnings:
        raise ValueError("Cannot produce a complete best-lambda report: " + "; ".join(result.warnings))
    if Path(report_path).exists() and not overwrite:
        raise FileExistsError(f"Best-lambda report already exists: {report_path}. Use --overwrite explicitly.")
    _write_report(
        report_path,
        results_path,
        result,
        selection,
        evaluation_split=evaluation_split,
        manifest_sha256=manifest_sha256,
    )
    return result


def _format_metric(value: float | None) -> str:
    return "not_available" if value is None else f"{value:.6f}"


def _write_report(
    report_path: str | Path,
    results_path: str | Path,
    result: SelectionResult,
    selection: Mapping[str, Any],
    *,
    evaluation_split: str,
    manifest_sha256: str,
) -> None:
    low_snr = ", ".join(str(value) for value in selection.get("low_snr", [0, 5]))
    lines = [
        "# Best Lambda Report",
        "",
        f"- Source results: `{results_path}`",
        f"- Evaluation split: `{evaluation_split}`",
        f"- Evaluation manifest SHA-256: `{manifest_sha256}`",
        "- Evaluation contract SHA-256: `"
        + str(selection["expected_evaluation_contract_sha256"])
        + "`",
        f"- Low-SNR priority: `{low_snr} dB`",
        f"- Maximum absolute WER increase: `{float(selection.get('max_wer_absolute_increase', 0.05)):.6f}`",
        f"- Maximum absolute CER increase: `{float(selection.get('max_cer_absolute_increase', 0.03)):.6f}`",
        f"- Minimum TER denominator ratio versus ordinary LoRA: `{float(selection.get('min_ter_coverage_ratio_vs_baseline', 0.98)):.6f}`",
        f"- Minimum DER denominator ratio versus ordinary LoRA: `{float(selection.get('min_der_coverage_ratio_vs_baseline', 0.98)):.6f}`",
        f"- Minimum FCER denominator ratio versus ordinary LoRA: `{float(selection.get('min_fcer_coverage_ratio_vs_baseline', 0.98)):.6f}`",
        "",
        "## Lambda comparison",
        "",
        "| lambda | train type | clean WER | clean CER | guard WER | guard CER | delta WER | delta CER | low-SNR TER | low-SNR DER | TER coverage | DER coverage | FCER coverage | score | eligible |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
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
                    f"{summary.low_snr_ter_coverage_ratio:.6f}",
                    f"{summary.low_snr_der_coverage_ratio:.6f}",
                    f"{summary.low_snr_fcer_coverage_ratio:.6f}",
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
            f"- TER coverage ratio versus ordinary LoRA: `{selected.low_snr_ter_coverage_ratio:.6f}`",
            f"- DER coverage ratio versus ordinary LoRA: `{selected.low_snr_der_coverage_ratio:.6f}`",
            f"- FCER coverage ratio versus ordinary LoRA: `{selected.low_snr_fcer_coverage_ratio:.6f}`",
            f"- WER delta versus lambda 0: `{selected.wer_delta:+.6f}`",
            f"- CER delta versus lambda 0: `{selected.cer_delta:+.6f}`",
            f"- Locked control lambda: `{result.locked_control_lambda:g}`",
            f"- Locked control strategy: `{result.locked_control_strategy}`",
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


def _repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _repo_display(path: str | Path, repo_root: Path) -> str:
    resolved = _repo_path(path, repo_root)
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Decision artifact is outside the repository: {resolved}") from exc


def _config_by_lambda(
    experiment_configs: Sequence[Mapping[str, Any]],
) -> dict[float, Mapping[str, Any]]:
    by_lambda: dict[float, Mapping[str, Any]] = {}
    for config in experiment_configs:
        value = float(config.get("training", {}).get("lambda_tone", -1))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Experiment config has an invalid training.lambda_tone")
        if value in by_lambda:
            raise ValueError(f"Duplicate experiment config for lambda={value:g}")
        by_lambda[value] = config
    expected = set(EXPECTED_LAMBDAS)
    if set(by_lambda) != expected:
        raise ValueError(
            "Decision requires exactly the five pre-registered lambda configs; "
            f"expected={sorted(expected)}, observed={sorted(by_lambda)}"
        )
    return by_lambda


def _single_row_value(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    lambda_value: float | None = None,
) -> str:
    selected_rows = (
        list(rows)
        if lambda_value is None
        else [
            row
            for row in rows
            if math.isclose(
                float(row.get("lambda", -1)),
                lambda_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
    )
    values = {str(row.get(field, "")).strip().casefold() for row in selected_rows}
    if len(values) != 1 or "" in values:
        label = "all lambdas" if lambda_value is None else f"lambda={lambda_value:g}"
        raise ValueError(f"Selection results do not bind one {field} for {label}")
    return next(iter(values))


def _test_contract_for_source_manifest(
    config: Mapping[str, Any],
    *,
    source_manifest: str,
    source_manifest_sha256: str,
    source_rows: int,
) -> str:
    return source_test_evaluation_contract_sha256(
        config,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        source_rows=source_rows,
    )


def build_decision_lock(
    results_path: str | Path,
    selection: Mapping[str, Any],
    result: SelectionResult,
    experiment_configs: Sequence[Mapping[str, Any]],
    *,
    split_lock_path: str | Path,
    method_lock_path: str | Path,
    noisy_dev_lock_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build decision v3 from dev-only evidence without opening final-test rows."""

    from .evaluation import resolve_checkpoint
    from .method_contract import (
        verify_checkpoint_method_binding,
        verify_method_lock,
        verify_noisy_dev_lock,
    )

    if result.warnings:
        raise ValueError(
            "A LOCKED decision requires a complete five-lambda screen: "
            + "; ".join(result.warnings)
        )
    if result.selected_lambda <= 0.0:
        raise ValueError(
            "selected_method must be a positive tone-aware lambda; ordinary lambda=0 "
            "is locked separately as ordinary_baseline"
        )
    if math.isclose(
        result.selected_lambda,
        result.locked_control_lambda,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("selected_method and locked_control must be distinct")

    root = Path(repo_root).resolve()
    results_file = _repo_path(results_path, root)
    split_file = _repo_path(split_lock_path, root)
    method_file = _repo_path(method_lock_path, root)
    noisy_dev_file = _repo_path(noisy_dev_lock_path, root)
    rows = _read_rows(results_file)
    evaluation_split, manifest_sha256 = _require_dev_selection_provenance(
        rows, selection
    )
    observed = select_best_lambda_from_rows(rows, selection)
    if observed != result:
        raise ValueError("Selection result changed while building the decision lock")
    if {
        float(row["lambda"]) for row in rows
    } != set(EXPECTED_LAMBDAS):
        raise ValueError("Decision lock requires results for exactly five lambdas")

    configs = _config_by_lambda(experiment_configs)
    ordinary_config = configs[0.0]
    method_integrity = verify_method_lock(
        method_file,
        config=ordinary_config,
        repo_root=root,
        formal=True,
        verify_audio=True,
    )
    noisy_dev = verify_noisy_dev_lock(
        noisy_dev_file,
        repo_root=root,
        verify_audio=False,
    )
    if noisy_dev["manifest_sha256"] != manifest_sha256:
        raise ValueError(
            "Selection results do not bind the verified noisy-dev manifest"
        )
    for field in (
        "method_lock_sha256",
        "method_identity_sha256",
        "environment_artifact_sha256",
        "environment_identity_sha256",
        "source_tree_sha256",
    ):
        if _single_row_value(rows, field) != str(method_integrity[field]).casefold():
            raise ValueError(f"Selection results do not bind the verified {field}")

    split_lock = load_split_lock(split_file)
    source_test = split_lock["splits"]["test_locked"]
    source_manifest = str(source_test["manifest"])
    source_manifest_sha256 = str(source_test["manifest_sha256"]).casefold()
    source_rows = int(source_test["utterance_count"])
    role_lambdas = {
        "ordinary_baseline": 0.0,
        "selected_method": result.selected_lambda,
        "locked_control": result.locked_control_lambda,
    }
    locked_configurations: list[dict[str, Any]] = []
    checkpoint_hashes: set[str] = set()
    evaluation_contracts: set[str] = set()
    for role in LOCKED_ROLES:
        lambda_value = role_lambdas[role]
        config = configs[lambda_value]
        train_type = str(config.get("experiment", {}).get("train_type", ""))
        method_id = str(config.get("experiment", {}).get("method_id", ""))
        if role == "ordinary_baseline":
            expected_identity = ("ordinary_lora", "ordinary_lora")
        else:
            expected_identity = ("corrected_decoder_tone_lora", "tone_aware_lora")
        if (method_id, train_type) != expected_identity:
            raise ValueError(
                f"{role} config has an unexpected method/train identity"
            )
        output_dir = str(config.get("training", {}).get("output_dir", ""))
        if not output_dir:
            raise ValueError(f"lambda={lambda_value:g} config has no output_dir")
        checkpoint_root, _ = resolve_checkpoint(_repo_path(output_dir, root))
        checkpoint_identity = verify_checkpoint_config(checkpoint_root, config)
        verify_checkpoint_method_binding(checkpoint_root, method_integrity)
        if checkpoint_identity["checkpoint_sha256"] != checkpoint_inference_sha256(
            checkpoint_root
        ):
            raise ValueError("Checkpoint fingerprint changed during decision creation")
        if checkpoint_identity["checkpoint_sha256"] in checkpoint_hashes:
            raise ValueError("Decision roles must use three distinct checkpoints")
        checkpoint_hashes.add(checkpoint_identity["checkpoint_sha256"])
        if (
            _single_row_value(rows, "checkpoint_sha256", lambda_value=lambda_value)
            != checkpoint_identity["checkpoint_sha256"]
            or _single_row_value(
                rows, "training_contract_sha256", lambda_value=lambda_value
            )
            != checkpoint_identity["training_contract_sha256"]
        ):
            raise ValueError(
                f"Selection results/checkpoint provenance differs for lambda={lambda_value:g}"
            )
        if checkpoint_identity["training_contract_sha256"] != training_contract_sha256(
            config
        ):
            raise ValueError("Checkpoint training contract changed during decision creation")
        experiment_id = str(config.get("experiment", {}).get("id", "")).strip()
        if not experiment_id:
            raise ValueError(f"lambda={lambda_value:g} config has no experiment.id")
        model = config.get("model", {})
        locked_configurations.append(
            {
                "configuration_id": experiment_id,
                "role": role,
                "method_id": method_id,
                "train_type": train_type,
                "lambda": lambda_value,
                "seed": int(config.get("seed", -1)),
                "backbone": str(model.get("name_or_path", "")),
                "backbone_revision": str(model.get("revision", "")).casefold(),
                "checkpoint_path": _repo_display(checkpoint_root, root),
                **checkpoint_identity,
            }
        )
        evaluation_contracts.add(
            _test_contract_for_source_manifest(
                config,
                source_manifest=source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                source_rows=source_rows,
            )
        )

    selection_contract = _single_row_value(rows, "evaluation_contract_sha256")
    configured_selection_contract = str(
        selection.get("expected_evaluation_contract_sha256", "")
    ).casefold()
    if selection_contract != configured_selection_contract:
        raise ValueError("Selection evaluation contract differs from the locked rule")
    payload: dict[str, Any] = {
        "decision_version": DECISION_VERSION,
        "status": "LOCKED",
        "selection_complete": True,
        "test_unlocked": True,
        "split_lock": _repo_display(split_file, root),
        "split_lock_sha256": sha256_file(split_file),
        "method_lock": _repo_display(method_file, root),
        "method_lock_sha256": method_integrity["method_lock_sha256"],
        "method_identity_sha256": method_integrity["method_identity_sha256"],
        "noisy_dev_lock": _repo_display(noisy_dev_file, root),
        "noisy_dev_lock_sha256": noisy_dev["lock_sha256"],
        "noisy_dev_manifest_sha256": noisy_dev["manifest_sha256"],
        "selection_evaluation_split": evaluation_split,
        "selection_manifest_sha256": manifest_sha256,
        "selection_results": _repo_display(results_file, root),
        "selection_results_sha256": sha256_file(results_file),
        "selection_metric_version": METRIC_VERSION,
        "selection_rule": deepcopy(dict(selection)),
        "selection_rule_sha256": selection_rule_sha256(selection),
        "selection_evaluation_contract_sha256": selection_contract,
        "evaluated_lambdas": list(EXPECTED_LAMBDAS),
        "locked_control_strategy": result.locked_control_strategy,
        "selected_method_id": str(
            configs[result.selected_lambda]["experiment"]["method_id"]
        ),
        "selected_lambda": result.selected_lambda,
        "locked_control_lambda": result.locked_control_lambda,
        "source_test_manifest": source_manifest,
        "source_test_manifest_sha256": source_manifest_sha256,
        "source_test_utterance_count": source_rows,
        "allowed_test_evaluation_contract_sha256": sorted(evaluation_contracts),
        "final_benchmark_lock_status": "PENDING_AFTER_DECISION",
        "locked_configurations": locked_configurations,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def write_decision_lock(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write a complete decision; partial/unlocked payloads are refused."""

    if (
        payload.get("decision_version") != DECISION_VERSION
        or payload.get("status") != "LOCKED"
        or payload.get("selection_complete") is not True
        or payload.get("test_unlocked") is not True
    ):
        raise ValueError("Only a complete LOCKED decision can unlock final-test access")
    identity = str(payload.get("identity_sha256", "")).casefold()
    content = dict(payload)
    content.pop("identity_sha256", None)
    if not is_sha256(identity) or canonical_sha256(content) != identity:
        raise ValueError("Decision identity SHA-256 is invalid")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite decision lock: {destination}"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CONTROL_STRATEGIES",
    "EXPECTED_LAMBDAS",
    "LOCKED_ROLES",
    "LambdaSummary",
    "SelectionResult",
    "build_decision_lock",
    "choose_locked_control_lambda",
    "select_best_lambda",
    "select_best_lambda_from_rows",
    "write_decision_lock",
]
