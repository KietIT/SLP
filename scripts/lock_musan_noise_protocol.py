from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.noise_protocol import (  # noqa: E402
    MusanSourceMetadata,
    NoiseProtocolError,
    build_noise_protocol_outputs,
    write_locked_noise_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create content-hashed, file-disjoint MUSAN train/dev/test registries "
            "and a protocol lock."
        )
    )
    parser.add_argument("--musan-root", required=True)
    parser.add_argument(
        "--manifest-dir", default="data/manifests/noise/paper_v2"
    )
    parser.add_argument(
        "--protocol-dir", default="outputs/paper_v2/protocol"
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument(
        "--source-revision",
        required=True,
        help="SHA-256 of the exact acquired MUSAN archive/artifact.",
    )
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payloads = build_noise_protocol_outputs(
            args.musan_root,
            manifest_dir=args.manifest_dir,
            protocol_dir=args.protocol_dir,
            source=MusanSourceMetadata(
                source_url=args.source_url,
                source_revision=args.source_revision,
                license_id=args.license_id,
                license_url=args.license_url,
            ),
            seed=args.seed,
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
            test_fraction=args.test_fraction,
        )
        status = write_locked_noise_outputs(payloads, overwrite=args.overwrite)
    except (OSError, NoiseProtocolError) as exc:
        parser.error(str(exc))
    print(f"MUSAN paper-v2 noise protocol: {status}")
    for path in payloads:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
