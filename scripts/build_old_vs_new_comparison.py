from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.comparison import (  # noqa: E402
    ComparisonError,
    ComparisonInputs,
    build_comparison,
    write_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an auditable legacy-vs-paper-v2 comparison. Formal mode "
            "fails closed; use --diagnostic-allow-partial only for a preview."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--old-by-snr", type=Path, default=Path("outputs/analysis/results_by_snr.csv"))
    parser.add_argument("--old-by-noise-type", type=Path, default=Path("outputs/analysis/results_by_noise_type.csv"))
    parser.add_argument("--old-fleurs-results", type=Path, default=Path("outputs/external/fleurs/external_fleurs_results.csv"))
    parser.add_argument("--old-fleurs-bootstrap", type=Path, default=Path("outputs/external/fleurs/bootstrap_ci_results.csv"))
    parser.add_argument("--old-benchmark-manifest", type=Path, default=Path("outputs/benchmark/benchmark_manifest.csv"))
    parser.add_argument("--old-fleurs-predictions-dir", type=Path, default=Path("outputs/external/fleurs/predictions"))
    parser.add_argument("--fleurs-manifest", type=Path, default=Path("data/manifests/fleurs/paper_v2/test.jsonl"))
    parser.add_argument("--fleurs-preparation-lock", type=Path, default=Path("outputs/paper_v2/protocol/fleurs_test_lock.json"))
    parser.add_argument("--new-by-snr", type=Path, default=Path("outputs/paper_v2/analysis/final/results_by_snr.csv"))
    parser.add_argument("--new-by-noise-type", type=Path, default=Path("outputs/paper_v2/analysis/final/results_by_noise_type.csv"))
    parser.add_argument("--new-fleurs-results", type=Path, default=Path("outputs/paper_v2/external/fleurs/external_fleurs_results.csv"))
    parser.add_argument("--new-fleurs-provenance", type=Path, default=Path("outputs/paper_v2/external/fleurs/external_fleurs_results.csv.provenance.json"))
    parser.add_argument("--new-fleurs-bootstrap", type=Path, default=Path("outputs/paper_v2/external/fleurs/bootstrap_ci_results.csv"))
    parser.add_argument("--new-final-bootstrap", type=Path, default=Path("outputs/paper_v2/statistics/bootstrap_ci_final.csv"))
    parser.add_argument("--new-benchmark-manifest", type=Path, default=Path("outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl"))
    parser.add_argument("--decision-lock", type=Path, default=Path("outputs/paper_v2/protocol/best_lambda_decision.json"))
    parser.add_argument("--split-lock", type=Path, default=Path("outputs/paper_v2/protocol/split_lock.json"))
    parser.add_argument("--noise-split-lock", type=Path, default=Path("outputs/paper_v2/protocol/noise_split_lock.json"))
    parser.add_argument("--noisy-dev-lock", type=Path, default=Path("outputs/paper_v2/protocol/noisy_dev_lock.json"))
    parser.add_argument("--environment-lock", type=Path, default=Path("outputs/paper_v2/protocol/environment_lock.json"))
    parser.add_argument("--method-lock", type=Path, default=Path("outputs/paper_v2/protocol/method_lock.json"))
    parser.add_argument("--final-benchmark-lock", type=Path, default=Path("outputs/paper_v2/protocol/final_benchmark_lock.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/paper_v2/reports/old_vs_new_comparison.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/paper_v2/reports/old_vs_new_comparison.md"))
    parser.add_argument("--output-provenance", type=Path, default=Path("outputs/paper_v2/reports/old_vs_new_comparison.provenance.json"))
    parser.add_argument(
        "--diagnostic-allow-partial",
        action="store_true",
        help="Permit missing post-run artifacts and emit an explicitly non-formal preview.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Recover an interrupted output bundle. Existing canonical/staged "
            "bytes must match this exact deterministic recomputation."
        ),
    )
    return parser


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        inputs = ComparisonInputs(
            repo_root=root,
            old_by_snr=_rooted(args.old_by_snr, root),
            old_by_noise_type=_rooted(args.old_by_noise_type, root),
            old_fleurs_results=_rooted(args.old_fleurs_results, root),
            old_fleurs_bootstrap=_rooted(args.old_fleurs_bootstrap, root),
            old_benchmark_manifest=_rooted(args.old_benchmark_manifest, root),
            old_fleurs_predictions_dir=_rooted(args.old_fleurs_predictions_dir, root),
            fleurs_manifest=_rooted(args.fleurs_manifest, root),
            fleurs_preparation_lock=_rooted(args.fleurs_preparation_lock, root),
            new_by_snr=_rooted(args.new_by_snr, root),
            new_by_noise_type=_rooted(args.new_by_noise_type, root),
            new_fleurs_results=_rooted(args.new_fleurs_results, root),
            new_fleurs_provenance=_rooted(args.new_fleurs_provenance, root),
            new_fleurs_bootstrap=_rooted(args.new_fleurs_bootstrap, root),
            new_final_bootstrap=_rooted(args.new_final_bootstrap, root),
            new_benchmark_manifest=_rooted(args.new_benchmark_manifest, root),
            decision_lock=_rooted(args.decision_lock, root),
            split_lock=_rooted(args.split_lock, root),
            noise_split_lock=_rooted(args.noise_split_lock, root),
            noisy_dev_lock=_rooted(args.noisy_dev_lock, root),
            environment_lock=_rooted(args.environment_lock, root),
            method_lock=_rooted(args.method_lock, root),
            final_benchmark_lock=_rooted(args.final_benchmark_lock, root),
        )
        bundle = build_comparison(
            inputs,
            diagnostic_allow_partial=args.diagnostic_allow_partial,
        )
        outputs = write_comparison(
            bundle,
            csv_path=_rooted(args.output_csv, root),
            markdown_path=_rooted(args.output_md, root),
            provenance_path=_rooted(args.output_provenance, root),
            resume=args.resume,
        )
    except (ComparisonError, FileNotFoundError, OSError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        f"PASS mode={bundle.provenance['mode']} rows={len(bundle.rows)} "
        f"valid_deltas={bundle.provenance['valid_delta_row_count']}"
    )
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
