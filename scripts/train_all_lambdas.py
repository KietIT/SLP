from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import apply_cli_overrides, load_experiment_config  # noqa: E402
from src.vitonesr.phat.trainer import train_experiment  # noqa: E402


def _pipeline_configs(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as pipeline_file:
        pipeline = yaml.safe_load(pipeline_file) or {}
    configs = pipeline.get("experiment_configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("Pipeline config must contain a non-empty experiment_configs list")
    return [str(config) for config in configs]


def _latest_resumable_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_step_*")) if output_dir.exists() else []
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all five configured PhoWhisper LoRA lambda values.")
    parser.add_argument("--config", default="configs/phat/phat_pipeline.yaml")
    parser.add_argument("--lambda-value", type=float, action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None, help="Optional root replacing outputs/phat/checkpoints.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    args = parser.parse_args()

    selected = set(args.lambda_value or [])
    completed: list[Path] = []
    for config_path in _pipeline_configs(args.config):
        config = load_experiment_config(config_path)
        lambda_value = float(config["training"]["lambda_tone"])
        if selected and lambda_value not in selected:
            continue
        output_dir = None
        if args.output_dir:
            output_dir = str(Path(args.output_dir) / Path(str(config["training"]["output_dir"])).name)
        config = apply_cli_overrides(
            config,
            seed=args.seed,
            train_manifest=args.manifest,
            output_dir=output_dir,
            max_train_samples=args.max_train_samples,
            max_train_steps=args.max_train_steps,
        )
        run_dir = Path(str(config["training"]["output_dir"]))
        final_checkpoint = run_dir / "final"
        if args.resume and final_checkpoint.exists() and not args.overwrite:
            print(f"skip completed lambda={lambda_value:g}: {final_checkpoint}")
            completed.append(final_checkpoint)
            continue
        resume_checkpoint = _latest_resumable_checkpoint(run_dir) if args.resume else None
        completed.append(
            train_experiment(
                config,
                device_arg=args.device,
                resume_checkpoint=resume_checkpoint,
                overwrite=args.overwrite,
            )
        )
    if not completed:
        raise ValueError("No experiment config matched the requested lambda values")
    for checkpoint in completed:
        print(checkpoint)


if __name__ == "__main__":
    main()
