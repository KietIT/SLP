from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.benchmark import validate_robust_benchmark_files  # noqa: E402
from src.vitonesr.prediction import ZERO_SHOT_MODEL_SPECS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Validate robust benchmark, predictions, and result files.")
    p.add_argument("--benchmark_manifest", required=True)
    p.add_argument("--pool_manifest", required=True)
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--expected_eval_size", type=int, default=300)
    p.add_argument("--expected_pool_size", type=int, default=500)
    p.add_argument("--snrs", type=float, nargs="+", default=[20, 10, 5, 0])
    p.add_argument("--models", nargs="+", choices=sorted(ZERO_SHOT_MODEL_SPECS), default=None)
    p.add_argument("--no_result_check", action="store_true")
    args = p.parse_args()

    result = validate_robust_benchmark_files(
        benchmark_manifest=Path(args.benchmark_manifest),
        pool_manifest=Path(args.pool_manifest),
        pred_dir=Path(args.pred_dir),
        expected_eval_size=args.expected_eval_size,
        expected_pool_size=args.expected_pool_size,
        snrs=tuple(args.snrs),
        model_keys=args.models,
        require_results=not args.no_result_check,
    )
    print("validation PASS")
    print(f"pool rows: {result['pool_rows']}")
    print(f"benchmark rows: {result['benchmark_rows']}")
    print(f"prediction files: {len(result['prediction_files'])}")


if __name__ == "__main__":
    main()
