from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.evaluation import load_benchmark_rows  # noqa: E402
from src.vitonesr.prediction import normalize_snr  # noqa: E402


def _run(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print("running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _preflight_dev_selection(pipeline_path: str | Path) -> None:
    with Path(pipeline_path).open("r", encoding="utf-8") as handle:
        pipeline = yaml.safe_load(handle) or {}
    config_paths = pipeline.get("experiment_configs")
    if not isinstance(config_paths, list) or not config_paths:
        raise ValueError("Pipeline config must contain experiment_configs")
    locked_contract: tuple[str, str] | None = None
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        evaluation = config["evaluation"]
        selection = config["selection"]
        contract = (
            str(evaluation["manifest"]),
            str(evaluation["expected_manifest_sha256"]),
        )
        if locked_contract is None:
            locked_contract = contract
        elif contract != locked_contract:
            raise ValueError(
                "All lambda configs must share one locked dev evaluation manifest"
            )
        rows = load_benchmark_rows(evaluation["manifest"])
        observed_snrs = {normalize_snr(row["snr"]) for row in rows}
        required_snrs = {
            normalize_snr(value) for value in selection.get("low_snr", [])
        }
        missing = sorted(required_snrs - observed_snrs)
        if missing:
            raise ValueError(
                "Formal train/evaluate/select pipeline is not ready: the locked "
                f"dev manifest is missing required selection SNR values {missing}. "
                "Build and lock the Gate-2 noise-disjoint dev screen first. "
                "Definitive training also waits for the Gate-2 locked noise "
                "registry; never substitute the official test manifest."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Phat LoRA train/evaluate/select pipeline.")
    parser.add_argument("--config", default="configs/phat/phat_pipeline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", default=None, help="Override the training manifest.")
    parser.add_argument("--output-dir", default=None, help="Optional checkpoint root.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    _preflight_dev_selection(args.config)

    train_command = ["scripts/train_all_lambdas.py", "--config", args.config, "--device", args.device]
    if args.seed is not None:
        train_command.extend(["--seed", str(args.seed)])
    if args.resume:
        train_command.append("--resume")
    if args.manifest:
        train_command.extend(["--manifest", args.manifest])
    if args.output_dir:
        train_command.extend(["--output-dir", args.output_dir])
    if args.overwrite:
        train_command.append("--overwrite")
    _run(train_command)

    evaluate_command = ["scripts/evaluate_all_lambdas.py", "--config", args.config, "--device", args.device]
    if args.output_dir:
        evaluate_command.extend(["--checkpoint-root", args.output_dir])
    if args.overwrite:
        evaluate_command.append("--overwrite")
    _run(evaluate_command)
    select_command = ["scripts/select_best_lambda.py", "--config", args.config]
    if args.overwrite:
        select_command.append("--overwrite")
    _run(select_command)


if __name__ == "__main__":
    main()
