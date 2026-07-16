from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.method_contract import (  # noqa: E402
    DEFAULT_LAMBDA_GRID,
    DEFAULT_METHOD_LOCK,
    DEFAULT_SOURCE_COMPONENTS,
    MethodContractError,
    build_method_contract,
    write_method_lock,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and atomically lock the paper-v2 method, data/noise protocol, "
            "environment, source components, lambda grid, decode and metric contracts."
        )
    )
    parser.add_argument("--config", default="configs/phat/lambda_0.yaml")
    parser.add_argument(
        "--split-lock", default="outputs/paper_v2/protocol/split_lock.json"
    )
    parser.add_argument(
        "--noise-split-lock",
        default="outputs/paper_v2/protocol/noise_split_lock.json",
    )
    parser.add_argument(
        "--noisy-dev-lock",
        default="outputs/paper_v2/protocol/noisy_dev_lock.json",
    )
    parser.add_argument(
        "--environment",
        default="outputs/paper_v2/protocol/environment_lock.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_METHOD_LOCK)
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA_GRID),
    )
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help=(
            "Repository-relative source/config component to hash. Repeat as needed; "
            "the audited default inventory is used when omitted."
        ),
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Create a diagnostic lock. The default is formal and therefore requires "
            "a formal environment capture and every prerequisite lock."
        ),
    )
    parser.add_argument(
        "--skip-audio-verification",
        action="store_true",
        help="Diagnostic-only speed option; forbidden for formal locks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    formal = not bool(args.diagnostic)
    if formal and args.skip_audio_verification:
        parser.error("formal method locks cannot skip audio verification")
    try:
        config = load_experiment_config(args.config)
        artifact = build_method_contract(
            config,
            split_lock_path=args.split_lock,
            noise_split_lock_path=args.noise_split_lock,
            noisy_dev_lock_path=args.noisy_dev_lock,
            environment_path=args.environment,
            source_components=tuple(args.component or DEFAULT_SOURCE_COMPONENTS),
            lambda_grid=tuple(args.lambda_grid),
            repo_root=ROOT,
            formal=formal,
            verify_audio=not bool(args.skip_audio_verification),
        )
        write_method_lock(args.output, artifact, overwrite=bool(args.overwrite))
    except (FileExistsError, MethodContractError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"method={artifact['identity_sha256']} mode={artifact['mode']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
