from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.evaluation import run_checkpoint_evaluation, write_ablation_results  # noqa: E402


def _read_pipeline(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as pipeline_file:
        pipeline = yaml.safe_load(pipeline_file) or {}
    if not pipeline.get("experiment_configs"):
        raise ValueError("Pipeline config must contain experiment_configs")
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every completed lambda checkpoint and aggregate real metrics.")
    parser.add_argument("--config", default="configs/phat/phat_pipeline.yaml")
    parser.add_argument("--lambda-value", type=float, action="append", default=None)
    parser.add_argument("--checkpoint", default=None, help="Only valid when one lambda is selected.")
    parser.add_argument("--checkpoint-root", default=None, help="Root used by train_all_lambdas.py --output-dir.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None, help="Prediction directory; required for limited smoke runs.")
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Verify/reuse completed lambda predictions and resume exact "
            "hash-bound partial predictions."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.limit is not None and args.output_dir is None:
        parser.error("--limit requires a separate --output-dir so partial smoke outputs cannot replace official outputs")
    if args.checkpoint and len(args.lambda_value or []) != 1:
        parser.error("--checkpoint requires exactly one --lambda-value")

    pipeline = _read_pipeline(args.config)
    selected = set(args.lambda_value or [])
    artifacts: list[tuple[Path, Path]] = []
    for config_path in pipeline["experiment_configs"]:
        config = load_experiment_config(config_path)
        lambda_value = float(config["training"]["lambda_tone"])
        if selected and lambda_value not in selected:
            continue
        if args.checkpoint:
            checkpoint = Path(args.checkpoint)
        elif args.checkpoint_root:
            checkpoint = (
                Path(args.checkpoint_root)
                / Path(str(config["training"]["output_dir"])).name
                / "best"
            )
        else:
            checkpoint = Path(str(config["training"]["output_dir"])) / "best"
        if args.output_dir:
            prediction_path = Path(args.output_dir) / Path(str(config["evaluation"]["prediction_path"])).name
        else:
            prediction_path = Path(str(config["evaluation"]["prediction_path"]))
        prediction_path = run_checkpoint_evaluation(
            config,
            checkpoint=checkpoint,
            output_path=prediction_path,
            manifest=args.manifest,
            limit=args.limit,
            batch_size=(
                int(config["evaluation"]["batch_size"])
                if args.batch_size is None
                else args.batch_size
            ),
            device_arg=args.device,
            overwrite=args.overwrite,
            resume=args.resume,
        )
        artifacts.append((checkpoint, prediction_path))
    if not artifacts:
        raise ValueError("No experiment config matched the requested lambda values")

    if args.results_path:
        results_path = Path(args.results_path)
    elif args.output_dir:
        results_path = Path(args.output_dir) / "lambda_ablation_results.csv"
    else:
        results_path = Path(str(pipeline["results_path"]))
    required = None if args.allow_partial or selected else (0.0, 0.05, 0.1, 0.3, 0.5)
    write_ablation_results(
        artifacts,
        results_path,
        require_lambdas=required,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(results_path)


if __name__ == "__main__":
    main()
