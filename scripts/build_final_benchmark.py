from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.final_benchmark import (  # noqa: E402
    FINAL_PEAK_LIMIT,
    FinalBenchmarkConfig,
    FinalBenchmarkError,
    build_final_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "After the method/lambda decision is locked, build the final 460-clean "
            "+ 1,840 MUSAN-test VIVOS robustness benchmark and its integrity lock."
        )
    )
    parser.add_argument(
        "--split-lock",
        type=Path,
        default=Path("outputs/paper_v2/protocol/split_lock.json"),
    )
    parser.add_argument(
        "--decision-lock",
        type=Path,
        default=Path("outputs/paper_v2/protocol/best_lambda_decision.json"),
    )
    parser.add_argument(
        "--noise-split-lock",
        type=Path,
        default=Path("outputs/paper_v2/protocol/noise_split_lock.json"),
    )
    parser.add_argument(
        "--method-lock",
        type=Path,
        default=Path("outputs/paper_v2/protocol/method_lock.json"),
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        default=Path("configs/phat/lambda_0.yaml"),
        help="Any lambda-grid config whose contract is bound by method_lock.",
    )
    parser.add_argument(
        "--source-test-manifest",
        type=Path,
        default=Path("data/manifests/paper_v2/vivos_test_locked.jsonl"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-audio-dir",
        type=Path,
        default=Path("data/derived/paper_v2/final_benchmark"),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("outputs/paper_v2/protocol/final_benchmark_lock.json"),
    )
    parser.add_argument(
        "--protocol-audit",
        type=Path,
        default=Path("outputs/paper_v2/protocol/final_benchmark_audit.csv"),
    )
    parser.add_argument(
        "--peak-limit",
        type=float,
        default=FINAL_PEAK_LIMIT,
        help=(
            f"Fixed formal anti-clipping peak limit ({FINAL_PEAK_LIMIT}); "
            "other values are rejected."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a complete previous transaction only when explicitly requested.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = FinalBenchmarkConfig(
        split_lock=args.split_lock,
        decision_lock=args.decision_lock,
        noise_split_lock=args.noise_split_lock,
        method_lock=args.method_lock,
        method_config=args.method_config,
        source_test_manifest=args.source_test_manifest,
        output_manifest=args.output_manifest,
        output_audio_dir=args.output_audio_dir,
        protocol_lock=args.protocol_lock,
        protocol_audit=args.protocol_audit,
        peak_limit=args.peak_limit,
    )
    try:
        result = build_final_benchmark(config, overwrite=bool(args.overwrite))
    except (FileExistsError, FinalBenchmarkError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Final paper-v2 benchmark: {result['status']}")
    print(f"rows: {result['rows']}")
    print(config.output_manifest)
    print(config.protocol_lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
