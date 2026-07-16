from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.evaluation import aggregate_prediction_file, run_checkpoint_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one real PhoWhisper LoRA checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction CSV path for this single run (required for filtered/smoke evaluation).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Verify/reuse a completed prediction or resume an exact hash-bound "
            "partial prediction. Mutually exclusive with --overwrite."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subset", choices=["all", "clean", "noisy"], default="all")
    parser.add_argument("--snr", action="append", default=None)
    parser.add_argument("--noise-type", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    is_filtered = (
        args.subset != "all"
        or bool(args.snr)
        or bool(args.noise_type)
        or args.limit is not None
    )
    if is_filtered and args.output_dir is None:
        parser.error(
            "filtered/limited evaluation requires --output-dir so it cannot "
            "replace the configured full-manifest prediction"
        )
    config = load_experiment_config(args.config)
    output_path = args.output_dir or config["evaluation"]["prediction_path"]
    prediction_path = run_checkpoint_evaluation(
        config,
        checkpoint=args.checkpoint,
        output_path=output_path,
        manifest=args.manifest,
        subset=args.subset,
        snrs=args.snr,
        noise_types=args.noise_type,
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
    result_rows = aggregate_prediction_file(prediction_path, checkpoint_path=args.checkpoint)
    print(f"prediction={prediction_path} aggregate_rows={len(result_rows)}")


if __name__ == "__main__":
    main()
