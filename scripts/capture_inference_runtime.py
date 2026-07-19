from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.inference_runtime import (  # noqa: E402
    REQUIRED_INFERENCE_SOURCE_PATHS,
    InferenceRuntimeLockError,
    capture_inference_runtime_lock,
    verify_inference_runtime_lock,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "paper_v2" / "protocol" / "inference_runtime_lock.json"
DEFAULT_TRAINING_ENVIRONMENT = (
    ROOT / "outputs" / "paper_v2" / "protocol" / "environment_lock.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture or verify an append-only formal inference runtime lock while "
            "preserving the original training environment identity."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--training-environment",
        type=Path,
        default=DEFAULT_TRAINING_ENVIRONMENT,
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=Path,
        metavar="REPO_RELATIVE_PATH",
        help=(
            "Additional inference source to hash (repeatable). The formal module, "
            "final-LoRA wrapper and FLEURS wrapper are always included."
        ),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify --output and the current runtime instead of creating a lock.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify_existing:
            if args.source:
                parser.error("--source cannot be combined with --verify-existing")
            result = verify_inference_runtime_lock(
                args.output,
                args.training_environment,
                ROOT,
                verify_current=True,
            )
            print(
                f"status={result['status']} identity={result['identity_sha256']} "
                f"lock_sha256={result['lock_sha256']} output={args.output}"
            )
            return 0
        required_sources = [ROOT / path for path in REQUIRED_INFERENCE_SOURCE_PATHS]
        artifact = capture_inference_runtime_lock(
            args.output,
            args.training_environment,
            [*required_sources, *args.source],
            ROOT,
        )
    except (InferenceRuntimeLockError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(
        f"status={artifact['status']} identity={artifact['identity_sha256']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
