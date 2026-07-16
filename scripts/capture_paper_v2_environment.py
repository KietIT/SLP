from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.phat.reproducibility import (  # noqa: E402
    DEFAULT_ENVIRONMENT_PACKAGES,
    DEFAULT_REQUIRED_REVISIONS,
    EnvironmentCaptureError,
    capture_environment,
    write_environment_artifact,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "paper_v2" / "protocol" / "environment_lock.json"


def _key_value(value: str) -> tuple[str, str]:
    key, separator, revision = value.partition("=")
    if not separator or not key.strip() or not revision.strip():
        raise argparse.ArgumentTypeError("revision must use NAME=IMMUTABLE_VALUE")
    return key.strip(), revision.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a canonical, path-free Python/PyTorch/CUDA/GPU/package/Git "
            "environment artifact for the paper_v2 protocol."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        type=_key_value,
        help=(
            "Bind an immutable model/data/method revision. Repeat for each lock; "
            "formal mode requires at least base_model by default."
        ),
    )
    parser.add_argument(
        "--require-revision",
        action="append",
        default=[],
        metavar="NAME",
        help="Require this revision key in formal mode (repeatable).",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        metavar="DISTRIBUTION",
        help="Capture an additional Python distribution version.",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help=(
            "Require every declared dependency, an exact Git commit, a clean tree, "
            "and all required revision locks."
        ),
    )
    parser.add_argument(
        "--allow-dirty-repository",
        action="store_true",
        help=(
            "Allow a formal capture from a dirty tree only when downstream method "
            "locking hashes every runtime source component. Git status and diff "
            "identities remain part of the environment hash."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact (default: fail without changing it).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.allow_dirty_repository and not args.formal:
        parser.error("--allow-dirty-repository is valid only together with --formal")
    revisions: dict[str, str] = {}
    for key, value in args.revision:
        normalized_key = key.strip().lower()
        if normalized_key in revisions:
            parser.error(f"duplicate --revision key: {normalized_key}")
        revisions[normalized_key] = value

    required_revisions = tuple(
        args.require_revision or DEFAULT_REQUIRED_REVISIONS
    )
    packages = tuple(sorted(set(DEFAULT_ENVIRONMENT_PACKAGES) | set(args.package)))
    safe_cli_args = {
        "allow_dirty_repository": bool(args.allow_dirty_repository),
        "formal": bool(args.formal),
        "packages": sorted(package.lower() for package in packages),
        "required_revisions": sorted(
            revision.lower() for revision in required_revisions
        ),
    }
    try:
        artifact = capture_environment(
            repo_root=ROOT,
            revisions=revisions,
            package_names=packages,
            required_packages=DEFAULT_ENVIRONMENT_PACKAGES,
            required_revisions=required_revisions,
            cli_args=safe_cli_args,
            formal=bool(args.formal),
            allow_dirty_repository=bool(args.allow_dirty_repository),
        )
        write_environment_artifact(
            args.output,
            artifact,
            overwrite=bool(args.overwrite),
        )
    except (EnvironmentCaptureError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(
        f"environment={artifact['identity_sha256']} "
        f"mode={artifact['environment']['capture_mode']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
