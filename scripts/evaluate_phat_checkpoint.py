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
    parser.add_argument("--output-dir", default=None, help="Prediction CSV path for this single run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subset", choices=["all", "clean", "noisy"], default="all")
    parser.add_argument("--snr", action="append", default=None)
    parser.add_argument("--noise-type", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
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
        batch_size=args.batch_size,
        device_arg=args.device,
        overwrite=args.overwrite,
    )
    result_rows = aggregate_prediction_file(prediction_path, checkpoint_path=args.checkpoint)
    print(f"prediction={prediction_path} aggregate_rows={len(result_rows)}")


if __name__ == "__main__":
    main()
