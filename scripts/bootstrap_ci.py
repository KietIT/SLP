from __future__ import annotations

import argparse
import csv
import math
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.analysis import (  # noqa: E402
    METRIC_VERSION,
    RUN_METADATA_COLUMNS,
    PredictionValidationError,
    compute_aligned_metric_result,
    load_prediction_csv,
)


EXPECTED_METRIC_VERSION = "aligned_v1"
DEFAULT_OUTPUT = "outputs/external/fleurs/bootstrap_ci_results.csv"
METRIC_FIELDS = {
    "wer": ("word_errors", "word_reference_units"),
    "cer": ("character_errors", "character_reference_units"),
    "ter": ("tone_errors", "tone_reference_units"),
    "der": ("diacritic_errors", "diacritic_reference_units"),
}
ROLE_CONTRACTS = {
    "ordinary": ("ordinary_lora", "0"),
    "lambda_005": ("tone_aware_lora", "0.05"),
    "lambda_01": ("tone_aware_lora", "0.1"),
}
PAIR_SPECS = (
    ("ordinary_vs_lambda_005", "ordinary", "lambda_005"),
    ("ordinary_vs_lambda_01", "ordinary", "lambda_01"),
    ("lambda_005_vs_lambda_01", "lambda_005", "lambda_01"),
)
OUTPUT_COLUMNS = [
    "metric_version",
    "dataset",
    "split",
    "pair_id",
    "model_a",
    "model_size_a",
    "train_type_a",
    "lambda_a",
    "seed_a",
    "model_b",
    "model_size_b",
    "train_type_b",
    "lambda_b",
    "seed_b",
    "n_paired",
    "metric",
    "numerator_a",
    "denominator_a",
    "estimate_a",
    "numerator_b",
    "denominator_b",
    "estimate_b",
    "delta_b_minus_a",
    "n_bootstrap",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "ci_excludes_zero",
    "bootstrap_seed",
    "bootstrap_unit",
    "ci_method",
]


class BootstrapError(ValueError):
    """Raised when paired prediction inputs or bootstrap settings are invalid."""


@dataclass(frozen=True)
class PredictionRun:
    role: str
    source_path: Path
    metadata: Mapping[str, str]
    rows_by_id: Mapping[str, Mapping[str, str]]


def _load_run(
    role: str,
    path: str | Path,
    *,
    expected_rows: int | None,
) -> PredictionRun:
    if role not in ROLE_CONTRACTS:
        raise BootstrapError(f"unknown run role: {role}")
    source = Path(path)
    try:
        rows = load_prediction_csv(source)
    except (FileNotFoundError, PredictionValidationError) as exc:
        raise BootstrapError(str(exc)) from exc
    if expected_rows is not None and len(rows) != expected_rows:
        raise BootstrapError(
            f"{source}: expected {expected_rows} prediction rows, found {len(rows)}"
        )
    if not rows:
        raise BootstrapError(f"{source}: prediction file is empty")

    metadata: dict[str, str] = {}
    for column in RUN_METADATA_COLUMNS:
        values = {row[column] for row in rows}
        if len(values) != 1:
            raise BootstrapError(
                f"{source}: run metadata {column!r} must be constant"
            )
        metadata[column] = next(iter(values))
    expected_train_type, expected_lambda = ROLE_CONTRACTS[role]
    if metadata["train_type"] != expected_train_type:
        raise BootstrapError(
            f"{source}: {role} requires train_type={expected_train_type!r}, "
            f"found {metadata['train_type']!r}"
        )
    if metadata["lambda"] != expected_lambda:
        raise BootstrapError(
            f"{source}: {role} requires lambda={expected_lambda!r}, "
            f"found {metadata['lambda']!r}"
        )
    if metadata["dataset"] != "fleurs":
        raise BootstrapError(
            f"{source}: external bootstrap requires dataset='fleurs'"
        )

    rows_by_id: dict[str, Mapping[str, str]] = {}
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id in rows_by_id:
            raise BootstrapError(f"{source}: duplicate utt_id {utt_id!r}")
        if row["snr"] != "clean" or row["noise_type"] != "clean":
            raise BootstrapError(
                f"{source}: FLEURS rows must use snr=noise_type='clean'"
            )
        rows_by_id[utt_id] = row
    return PredictionRun(role, source, metadata, rows_by_id)


def load_paired_runs(
    ordinary_path: str | Path,
    lambda_005_path: str | Path,
    lambda_01_path: str | Path,
    *,
    expected_rows: int | None = 857,
) -> tuple[dict[str, PredictionRun], list[str]]:
    paths = [Path(ordinary_path), Path(lambda_005_path), Path(lambda_01_path)]
    if len({path.resolve() for path in paths}) != len(paths):
        raise BootstrapError("the three prediction inputs must be different files")
    runs = {
        "ordinary": _load_run("ordinary", paths[0], expected_rows=expected_rows),
        "lambda_005": _load_run(
            "lambda_005", paths[1], expected_rows=expected_rows
        ),
        "lambda_01": _load_run("lambda_01", paths[2], expected_rows=expected_rows),
    }

    ordinary = runs["ordinary"]
    expected_ids = set(ordinary.rows_by_id)
    for role, run in runs.items():
        if set(run.rows_by_id) != expected_ids:
            missing = sorted(expected_ids.difference(run.rows_by_id))[:5]
            extra = sorted(set(run.rows_by_id).difference(expected_ids))[:5]
            raise BootstrapError(
                f"{run.source_path}: utt_id set differs from ordinary; "
                f"missing={missing}, extra={extra}"
            )
        if run.metadata["dataset"] != ordinary.metadata["dataset"]:
            raise BootstrapError(f"{run.source_path}: dataset differs across runs")
        for utt_id in expected_ids:
            if run.rows_by_id[utt_id]["ref"] != ordinary.rows_by_id[utt_id]["ref"]:
                raise BootstrapError(
                    f"{run.source_path}: ref differs for utt_id={utt_id!r}"
                )
    return runs, sorted(expected_ids, key=str.casefold)


def _per_utterance_contributions(
    run: PredictionRun,
    utt_ids: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    numerators = {name: np.zeros(len(utt_ids), dtype=np.int64) for name in METRIC_FIELDS}
    denominators = {name: np.zeros(len(utt_ids), dtype=np.int64) for name in METRIC_FIELDS}
    for index, utt_id in enumerate(utt_ids):
        row = run.rows_by_id[utt_id]
        result = compute_aligned_metric_result([row["ref"]], [row["hyp"]])
        for metric, (numerator_field, denominator_field) in METRIC_FIELDS.items():
            numerators[metric][index] = int(getattr(result, numerator_field))
            denominators[metric][index] = int(getattr(result, denominator_field))
    return {
        metric: (numerators[metric], denominators[metric]) for metric in METRIC_FIELDS
    }


def _rate(numerator: np.ndarray | int, denominator: np.ndarray | int):
    return np.asarray(numerator, dtype=np.float64) / np.maximum(
        np.asarray(denominator, dtype=np.float64), 1.0
    )


def _format_rate(value: float) -> str:
    if not math.isfinite(value):
        raise BootstrapError("bootstrap produced a non-finite statistic")
    return f"{value:.12f}"


def build_bootstrap_rows(
    runs: Mapping[str, PredictionRun],
    utt_ids: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> list[dict[str, object]]:
    if METRIC_VERSION != EXPECTED_METRIC_VERSION:
        raise BootstrapError(
            f"expected metric_version={EXPECTED_METRIC_VERSION!r}, found {METRIC_VERSION!r}"
        )
    if n_bootstrap < 1:
        raise BootstrapError("n_bootstrap must be at least 1")
    if not 0.0 < ci_level < 1.0:
        raise BootstrapError("ci_level must be strictly between 0 and 1")
    if not utt_ids:
        raise BootstrapError("paired bootstrap requires at least one utterance")
    if bootstrap_seed < 0:
        raise BootstrapError("bootstrap_seed must be non-negative")

    contributions = {
        role: _per_utterance_contributions(run, utt_ids)
        for role, run in runs.items()
    }
    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0, len(utt_ids), size=(n_bootstrap, len(utt_ids)), dtype=np.int64
    )
    alpha = (1.0 - ci_level) / 2.0
    rows: list[dict[str, object]] = []

    for pair_id, role_a, role_b in PAIR_SPECS:
        run_a = runs[role_a]
        run_b = runs[role_b]
        for metric in METRIC_FIELDS:
            numerator_a, denominator_a = contributions[role_a][metric]
            numerator_b, denominator_b = contributions[role_b][metric]
            total_numerator_a = int(numerator_a.sum())
            total_denominator_a = int(denominator_a.sum())
            total_numerator_b = int(numerator_b.sum())
            total_denominator_b = int(denominator_b.sum())
            estimate_a = float(_rate(total_numerator_a, total_denominator_a))
            estimate_b = float(_rate(total_numerator_b, total_denominator_b))

            replicate_a = _rate(
                numerator_a[sample_indices].sum(axis=1),
                denominator_a[sample_indices].sum(axis=1),
            )
            replicate_b = _rate(
                numerator_b[sample_indices].sum(axis=1),
                denominator_b[sample_indices].sum(axis=1),
            )
            replicate_delta = replicate_b - replicate_a
            ci_lower, ci_upper = np.quantile(
                replicate_delta,
                [alpha, 1.0 - alpha],
                method="linear",
            )
            delta = estimate_b - estimate_a
            rows.append(
                {
                    "metric_version": METRIC_VERSION,
                    "dataset": run_a.metadata["dataset"],
                    "split": "test",
                    "pair_id": pair_id,
                    "model_a": run_a.metadata["model"],
                    "model_size_a": run_a.metadata["model_size"],
                    "train_type_a": run_a.metadata["train_type"],
                    "lambda_a": run_a.metadata["lambda"],
                    "seed_a": run_a.metadata["seed"],
                    "model_b": run_b.metadata["model"],
                    "model_size_b": run_b.metadata["model_size"],
                    "train_type_b": run_b.metadata["train_type"],
                    "lambda_b": run_b.metadata["lambda"],
                    "seed_b": run_b.metadata["seed"],
                    "n_paired": len(utt_ids),
                    "metric": metric,
                    "numerator_a": total_numerator_a,
                    "denominator_a": total_denominator_a,
                    "estimate_a": _format_rate(estimate_a),
                    "numerator_b": total_numerator_b,
                    "denominator_b": total_denominator_b,
                    "estimate_b": _format_rate(estimate_b),
                    "delta_b_minus_a": _format_rate(delta),
                    "n_bootstrap": n_bootstrap,
                    "ci_level": _format_rate(ci_level),
                    "ci_lower": _format_rate(float(ci_lower)),
                    "ci_upper": _format_rate(float(ci_upper)),
                    "ci_excludes_zero": str(
                        float(ci_upper) < 0.0 or float(ci_lower) > 0.0
                    ).lower(),
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_unit": "utt_id",
                    "ci_method": "paired_percentile",
                }
            )
    return rows


def write_output(
    output_path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    overwrite: bool = False,
    protected_inputs: Sequence[str | Path] = (),
) -> Path:
    destination = Path(output_path)
    protected = {Path(path).resolve() for path in protected_inputs}
    if destination.resolve() in protected:
        raise BootstrapError("refusing to overwrite a prediction input")
    if destination.exists() and not overwrite:
        raise BootstrapError(
            f"output already exists: {destination}; use --overwrite explicitly"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        if not overwrite and destination.exists():
            raise BootstrapError(f"output appeared while writing: {destination}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def run_bootstrap(
    ordinary_path: str | Path,
    lambda_005_path: str | Path,
    lambda_01_path: str | Path,
    output_path: str | Path,
    *,
    expected_rows: int | None = 857,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: int = 42,
    overwrite: bool = False,
) -> tuple[Path, list[dict[str, object]]]:
    input_paths = (ordinary_path, lambda_005_path, lambda_01_path)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise BootstrapError(
            f"output already exists: {destination}; use --overwrite explicitly"
        )
    runs, utt_ids = load_paired_runs(*input_paths, expected_rows=expected_rows)
    rows = build_bootstrap_rows(
        runs,
        utt_ids,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        bootstrap_seed=bootstrap_seed,
    )
    return (
        write_output(
            destination,
            rows,
            overwrite=overwrite,
            protected_inputs=input_paths,
        ),
        rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired utterance bootstrap CIs for three FLEURS LoRA runs."
    )
    parser.add_argument("--ordinary", required=True)
    parser.add_argument("--lambda-005", required=True)
    parser.add_argument("--lambda-01", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-rows", type=int, default=857)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output, rows = run_bootstrap(
            args.ordinary,
            args.lambda_005,
            args.lambda_01,
            args.output,
            expected_rows=args.expected_rows,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            bootstrap_seed=args.bootstrap_seed,
            overwrite=args.overwrite,
        )
    except (BootstrapError, OSError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"PASS paired_rows={rows[0]['n_paired']} pairs=3 metrics=4 "
        f"bootstrap={args.n_bootstrap} metric_version={METRIC_VERSION}"
    )
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
