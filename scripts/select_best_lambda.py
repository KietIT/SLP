from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.selection import select_best_lambda  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best lambda from real ablation results.")
    parser.add_argument("--config", default="configs/phat/phat_pipeline.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument("--output-dir", default=None, help="Best-lambda Markdown report path.")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as pipeline_file:
        pipeline = yaml.safe_load(pipeline_file) or {}
    experiment_configs = pipeline.get("experiment_configs") or []
    if not experiment_configs:
        raise ValueError("Pipeline config must contain experiment_configs")
    experiment_config = load_experiment_config(experiment_configs[0])
    results_path = args.results or pipeline["results_path"]
    report_path = args.output_dir or pipeline["best_lambda_report"]
    result = select_best_lambda(
        results_path,
        experiment_config["selection"],
        report_path,
        require_complete=not args.allow_partial,
        overwrite=args.overwrite,
    )
    print(f"selected_lambda={result.selected_lambda:g} report={report_path}")


if __name__ == "__main__":
    main()
