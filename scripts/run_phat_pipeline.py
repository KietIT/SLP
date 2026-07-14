from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print("running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


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
