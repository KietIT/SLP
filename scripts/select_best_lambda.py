from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.selection import (  # noqa: E402
    build_decision_lock,
    select_best_lambda,
    write_decision_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best lambda from dev-only ablation results; test input is rejected."
    )
    parser.add_argument("--config", default="configs/phat/phat_pipeline.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Best-lambda Markdown report path (legacy option name).",
    )
    parser.add_argument(
        "--decision-output",
        default=None,
        help="Atomic method/lambda decision JSON path.",
    )
    parser.add_argument(
        "--noisy-dev-lock",
        default="outputs/paper_v2/protocol/noisy_dev_lock.json",
    )
    parser.add_argument(
        "--skip-decision",
        action="store_true",
        help="Write only the diagnostic report; final-test access remains locked.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as pipeline_file:
        pipeline = yaml.safe_load(pipeline_file) or {}
    experiment_configs = pipeline.get("experiment_configs") or []
    if not experiment_configs:
        raise ValueError("Pipeline config must contain experiment_configs")
    loaded_configs = [load_experiment_config(path) for path in experiment_configs]
    experiment_config = loaded_configs[0]
    results_path = args.results or pipeline["results_path"]
    report_path = args.output_dir or pipeline["best_lambda_report"]
    if args.allow_partial and not args.skip_decision:
        parser.error("--allow-partial requires --skip-decision")
    result = select_best_lambda(
        results_path,
        experiment_config["selection"],
        report_path,
        require_complete=not args.allow_partial,
        overwrite=args.overwrite,
    )
    decision_path = (
        args.decision_output
        or pipeline.get("decision_lock")
        or experiment_config["protocol"]["decision_lock"]
    )
    if not args.skip_decision:
        decision = build_decision_lock(
            results_path,
            experiment_config["selection"],
            result,
            loaded_configs,
            split_lock_path=experiment_config["protocol"]["split_lock"],
            method_lock_path=experiment_config["protocol"]["method_lock"],
            noisy_dev_lock_path=args.noisy_dev_lock,
            repo_root=ROOT,
        )
        write_decision_lock(decision_path, decision, overwrite=args.overwrite)
    else:
        decision_path = "not_written"
    print(
        f"selected_lambda={result.selected_lambda:g} "
        f"locked_control_lambda={result.locked_control_lambda:g} "
        f"report={report_path} decision={decision_path}"
    )


if __name__ == "__main__":
    main()
