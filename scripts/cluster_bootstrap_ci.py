from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.statistics import (  # noqa: E402
    CLUSTER_UNITS,
    COMPARISON_ROLES,
    METRIC_VERSION,
    ClusterBootstrapError,
    run_cluster_bootstrap,
)


def _assignments(
    values: list[str],
    *,
    label: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, assigned = value.partition("=")
        key = key.strip()
        assigned = assigned.strip()
        if not separator or not key or not assigned:
            raise ClusterBootstrapError(
                f"{label} must use KEY=VALUE syntax, found {value!r}"
            )
        if key in result:
            raise ClusterBootstrapError(f"duplicate {label} key: {key}")
        result[key] = assigned
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired source-utterance cluster bootstrap for the three "
            "decision-locked ASR comparison roles."
        )
    )
    parser.add_argument("--decision-lock", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument(
        "--formal-paper-v2",
        action="store_true",
        help="Verify the decision and every prediction sidecar before resampling.",
    )
    parser.add_argument("--split-lock")
    parser.add_argument(
        "--final-benchmark-lock",
        help="Required for final VIVOS predictions; omit for FLEURS replication.",
    )
    parser.add_argument(
        "--cluster-unit",
        choices=CLUSTER_UNITS,
        default="source_utt_id",
        help=(
            "source_utt_id for the replicated VIVOS robustness benchmark; use "
            "utt_id_singleton_external explicitly only for a one-condition "
            "external manifest such as clean FLEURS."
        ),
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="CONFIGURATION_ID=CSV",
        help="Repeat once for each of the three locked configurations.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="ROLE=CONFIGURATION_ID",
        help=(
            "Required only when a role has multiple locked runs (for example, "
            "multi-seed); repeat for all three roles."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=64)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Recover an interrupted formal artifact-bundle transaction.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        prediction_paths = _assignments(args.prediction, label="prediction")
        if not prediction_paths:
            raise ClusterBootstrapError("provide exactly three --prediction inputs")
        comparison_set = _assignments(args.comparison, label="comparison")
        if comparison_set and set(comparison_set) != set(COMPARISON_ROLES):
            raise ClusterBootstrapError(
                "explicit --comparison values must cover exactly "
                f"{list(COMPARISON_ROLES)}"
            )
        output, rows = run_cluster_bootstrap(
            args.decision_lock,
            args.benchmark_manifest,
            prediction_paths,
            args.output,
            comparison_set=comparison_set or None,
            cluster_unit=args.cluster_unit,
            formal_paper_v2=args.formal_paper_v2,
            split_lock_path=args.split_lock,
            final_benchmark_lock_path=args.final_benchmark_lock,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            bootstrap_seed=args.bootstrap_seed,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    except (ClusterBootstrapError, FileNotFoundError, OSError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"PASS pairs=3 metrics=4 clusters={rows[0]['n_source_clusters']} "
        f"conditions={rows[0]['n_paired_conditions']} "
        f"bootstrap={args.n_bootstrap} metric_version={METRIC_VERSION}"
    )
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
