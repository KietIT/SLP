from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import apply_cli_overrides, load_experiment_config  # noqa: E402
from src.vitonesr.phat.trainer import train_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ordinary or tone-aware PhoWhisper LoRA experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lambda-value", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory used with --resume.")
    parser.add_argument("--manifest", default=None, help="Override the training JSONL manifest.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Real-data smoke-test subset size.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Stop after this many optimizer steps.")
    args = parser.parse_args()
    if args.resume and not args.checkpoint:
        parser.error("--resume requires --checkpoint")
    if args.checkpoint and not args.resume:
        parser.error("--checkpoint is only valid together with --resume")

    config = load_experiment_config(args.config)
    if (
        args.lambda_value is not None
        and float(args.lambda_value) != float(config["training"]["lambda_tone"])
        and args.output_dir is None
    ):
        parser.error("Changing --lambda-value requires a separate --output-dir to protect the configured checkpoint")
    config = apply_cli_overrides(
        config,
        lambda_value=args.lambda_value,
        seed=args.seed,
        train_manifest=args.manifest,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_train_steps=args.max_train_steps,
    )
    final_checkpoint = train_experiment(
        config,
        device_arg=args.device,
        resume_checkpoint=args.checkpoint if args.resume else None,
        overwrite=args.overwrite,
    )
    print(final_checkpoint)


if __name__ == "__main__":
    main()
