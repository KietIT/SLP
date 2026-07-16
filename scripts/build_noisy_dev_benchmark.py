from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.noise_protocol import (  # noqa: E402
    NoiseProtocolError,
    NoisyDevConfig,
    build_noisy_dev_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a content-locked noisy-dev benchmark from VIVOS dev and the "
            "locked MUSAN dev partition."
        )
    )
    parser.add_argument(
        "--source-dev-manifest",
        default="data/manifests/paper_v2/vivos_dev.jsonl",
    )
    parser.add_argument(
        "--source-dev-sha256",
        required=True,
        help="Expected SHA-256 of the locked source dev manifest.",
    )
    parser.add_argument(
        "--noise-split-lock",
        default="outputs/paper_v2/protocol/noise_split_lock.json",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/manifests/paper_v2/vivos_dev_noisy.jsonl",
    )
    parser.add_argument(
        "--output-audio-dir", default="data/derived/paper_v2/noisy_dev"
    )
    parser.add_argument(
        "--protocol-lock",
        default="outputs/paper_v2/protocol/noisy_dev_lock.json",
    )
    parser.add_argument(
        "--protocol-audit",
        default="outputs/paper_v2/protocol/noisy_dev_audit.csv",
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=[20, 10, 5, 0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--peak-limit", type=float, default=0.999)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = NoisyDevConfig(
        source_dev_manifest=Path(args.source_dev_manifest),
        source_dev_sha256=args.source_dev_sha256,
        noise_split_lock=Path(args.noise_split_lock),
        output_manifest=Path(args.output_manifest),
        output_audio_dir=Path(args.output_audio_dir),
        protocol_lock=Path(args.protocol_lock),
        protocol_audit=Path(args.protocol_audit),
        snrs=tuple(args.snrs),
        seed=args.seed,
        sample_rate=args.sample_rate,
        peak_limit=args.peak_limit,
        include_clean=not args.no_clean,
    )
    try:
        result = build_noisy_dev_benchmark(config, overwrite=args.overwrite)
    except (OSError, NoiseProtocolError) as exc:
        parser.error(str(exc))
    print(f"Noisy-dev paper-v2 benchmark: {result['status']}")
    print(f"rows: {result['rows']}")
    print(config.output_manifest)
    print(config.protocol_lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
