from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.prediction import ZERO_SHOT_MODEL_SPECS, aggregate_zero_shot_results  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate zero-shot prediction CSV files into metric tables.")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_by_snr", required=True)
    p.add_argument("--out_by_noise_type", default=None)
    p.add_argument("--expected_rows", type=int, default=1500)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--models", nargs="+", choices=sorted(ZERO_SHOT_MODEL_SPECS), default=None)
    args = p.parse_args()

    result = aggregate_zero_shot_results(
        pred_dir=Path(args.pred_dir),
        out_by_snr=Path(args.out_by_snr),
        out_by_noise_type=Path(args.out_by_noise_type) if args.out_by_noise_type else None,
        expected_rows=args.expected_rows,
        allow_partial=args.smoke_test,
        model_keys=args.models,
    )
    print(f"wrote {result['out_by_snr']}")
    if result["out_by_noise_type"]:
        print(f"wrote {result['out_by_noise_type']}")


if __name__ == "__main__":
    main()
