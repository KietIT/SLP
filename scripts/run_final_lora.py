from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.final_evaluation import (  # noqa: E402
    FinalLoraProtocolError,
    load_final_lora_config,
    run_final_lora_suite,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exactly ordinary_baseline, selected_method, and "
            "locked_control from decision v3 on the locked 2,300-row final "
            "VIVOS/MUSAN benchmark. No lambda is accepted on the command line."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/paper_v2_final_lora.yaml",
        help="Hash-pinned final LoRA suite config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Verify/reuse completed roles and continue an exact hash-bound "
            "per-role prediction prefix from the next inference batch. "
            "Orphaned or tampered partial artifacts fail closed."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_final_lora_config(args.config)
        result = run_final_lora_suite(config, resume=bool(args.resume))
    except (FileExistsError, FinalLoraProtocolError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print("Final LoRA roles: " + ", ".join(result["roles"]))
    print(f"rows per role: {result['prediction_rows_per_role']}")
    print(f"aggregate rows: {result['aggregate_rows']}")
    print(f"output: {result['output_directory']}")
    print(f"aggregate: {result['aggregate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
