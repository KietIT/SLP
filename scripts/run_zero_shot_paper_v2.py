from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.zero_shot_paper_v2 import (  # noqa: E402
    DEFAULT_CONFIG,
    load_suite_config,
    run_zero_shot_suite,
    validate_suite_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run hash-locked paper-v2 zero-shot baselines on the authorized final "
            "benchmark. This command refuses mutable model revisions and locked tests."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model keys from config; omit to run the complete six-model suite.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only a hash-bound partial output; never overwrites a completed run.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate selected model revisions/settings without opening locks or data.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_config:
        config = load_suite_config(args.config)
        specs = validate_suite_config(config, args.models)
        print(
            json.dumps(
                {"status": "valid", "models": [spec["key"] for spec in specs]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    result = run_zero_shot_suite(
        args.config,
        model_keys=args.models,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
