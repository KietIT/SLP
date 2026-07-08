from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.benchmark import BenchmarkConfig, RobustBenchmarkBuilder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build a fixed robust ASR benchmark from VIVOS and MUSAN.")
    p.add_argument("--vivos_manifest", required=True)
    p.add_argument("--noise_manifest", required=True)
    p.add_argument("--out_manifest", required=True)
    p.add_argument("--pool_manifest", required=True)
    p.add_argument("--report_out", required=True)
    p.add_argument("--out_noisy_dir", required=True)
    p.add_argument("--pool_size", type=int, default=500)
    p.add_argument("--eval_size", type=int, default=300)
    p.add_argument("--snrs", type=float, nargs="+", default=[20, 10, 5, 0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    args = p.parse_args()

    config = BenchmarkConfig(
        vivos_manifest=Path(args.vivos_manifest),
        noise_manifest=Path(args.noise_manifest),
        out_manifest=Path(args.out_manifest),
        pool_manifest=Path(args.pool_manifest),
        report_out=Path(args.report_out),
        out_noisy_dir=Path(args.out_noisy_dir),
        pool_size=args.pool_size,
        eval_size=args.eval_size,
        snrs=tuple(args.snrs),
        seed=args.seed,
        sample_rate=args.sample_rate,
    )
    result = RobustBenchmarkBuilder(config).build()
    print(f"wrote pool manifest: {result['pool_manifest']}")
    print(f"wrote benchmark manifest: {result['out_manifest']}")
    print(f"wrote report: {result['report_out']}")


if __name__ == "__main__":
    main()
