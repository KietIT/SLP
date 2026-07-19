"""Create or verify the immutable paper-v2 inference delivery receipt.

This is the final, generic hand-off boundary for Trung's analysis outputs.  It
does not trust files merely because they exist: the formal final-LoRA and
FLEURS execution receipts are verified first against the *current* inference
runtime, then every explicitly supplied downstream artifact is bound by its
repository-relative path, byte count, and SHA-256 digest.

``--skip-fleurs`` is deliberately verification-only.  It is useful while the
external run is still in progress, but can never create the final receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_external_fleurs_inference_runtime import (  # noqa: E402
    DEFAULT_EXECUTION_RECEIPT as DEFAULT_FLEURS_RECEIPT,
    DEFAULT_TRAINING_ENVIRONMENT_LOCK,
    verify_execution_receipt as verify_fleurs_execution_receipt,
)
from scripts.run_final_lora_inference_runtime import (  # noqa: E402
    DEFAULT_CONFIG as DEFAULT_FINAL_LORA_CONFIG,
    DEFAULT_INFERENCE_RUNTIME_LOCK,
    DEFAULT_RECEIPT as DEFAULT_FINAL_LORA_RECEIPT,
    verify_final_lora_execution_receipt,
)
from src.vitonesr.inference_runtime import (  # noqa: E402
    INFERENCE_RUNTIME_SCHEMA_VERSION,
)
from src.vitonesr.phat.protocol import is_sha256  # noqa: E402


DELIVERY_RECEIPT_VERSION = "paper_v2_trung_delivery_receipt_v1"
DEFAULT_OUTPUT = Path(
    "outputs/paper_v2/protocol/trung_delivery_receipt.json"
)

FinalLoraVerifier = Callable[..., Mapping[str, Any]]
FleursVerifier = Callable[..., Mapping[str, Any]]


class PaperV2DeliveryError(ValueError):
    """Raised when final delivery evidence is incomplete or has drifted."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _repo_path(value: str | Path, *, label: str) -> tuple[Path, str]:
    raw = str(value).strip()
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not raw
        or windows.is_absolute()
        or bool(windows.drive)
        or raw.startswith(("~", "//", "\\\\"))
        or "\\" in raw
        or ".." in posix.parts
    ):
        raise PaperV2DeliveryError(
            f"{label} must be a portable repository-relative POSIX path"
        )
    path = ROOT.joinpath(*posix.parts).resolve()
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PaperV2DeliveryError(f"{label} escapes the repository") from exc
    return path, relative


def _file_record(path: Path, reference: str, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PaperV2DeliveryError(f"Missing {label}: {reference}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return {
        "bytes": byte_count,
        "path": reference,
        "sha256": digest.hexdigest(),
    }


def _required_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise PaperV2DeliveryError(f"{label} must be a concrete SHA-256")
    return str(value).strip().casefold()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PaperV2DeliveryError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperV2DeliveryError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PaperV2DeliveryError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(
            f"Delivery receipt already exists; use --verify-existing: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"Delivery receipt already exists; use --verify-existing: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_records(
    artifacts: Sequence[str | Path], *, output_path: Path
) -> list[dict[str, Any]]:
    if not artifacts:
        raise PaperV2DeliveryError("At least one explicit --artifact is required")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in artifacts:
        path, reference = _repo_path(raw, label="artifact")
        if reference in seen:
            raise PaperV2DeliveryError(f"Duplicate artifact: {reference}")
        if path == output_path:
            raise PaperV2DeliveryError("Delivery receipt cannot bind itself")
        seen.add(reference)
        records.append(_file_record(path, reference, label="downstream artifact"))
    return sorted(records, key=lambda item: str(item["path"]))


def _verified_upstreams(
    *,
    final_lora_receipt: str,
    final_lora_config: str,
    fleurs_receipt: str,
    inference_runtime_lock: str,
    training_environment_lock: str,
    skip_fleurs: bool,
    final_lora_verifier: FinalLoraVerifier,
    fleurs_verifier: FleursVerifier,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    # Both formal verifiers require verify_current=True internally.  Passing it
    # explicitly for final LoRA prevents this hand-off from ever accepting a
    # receipt that was checked only against a historical environment snapshot.
    final_summary = dict(
        final_lora_verifier(
            final_lora_receipt,
            final_lora_config,
            inference_runtime_lock,
            verify_current=True,
        )
    )
    if (
        final_summary.get("status") != "VERIFIED"
        or final_summary.get("inference_runtime_verified") is not True
    ):
        raise PaperV2DeliveryError("Final-LoRA receipt was not formally verified")

    runtime_path, runtime_reference = _repo_path(
        inference_runtime_lock, label="inference runtime lock"
    )
    runtime_record = _file_record(
        runtime_path, runtime_reference, label="inference runtime lock"
    )
    if _required_sha256(
        final_summary.get("inference_runtime_lock_sha256"),
        label="Final-LoRA inference runtime lock",
    ) != runtime_record["sha256"]:
        raise PaperV2DeliveryError(
            "Final-LoRA verifier returned another inference runtime lock"
        )
    runtime_value = _load_json_object(
        runtime_path, label="inference runtime lock"
    )
    if runtime_value.get("schema_version") != INFERENCE_RUNTIME_SCHEMA_VERSION:
        raise PaperV2DeliveryError("Unsupported inference runtime lock version")
    runtime_identity = _required_sha256(
        runtime_value.get("identity_sha256"), label="inference runtime identity"
    )
    runtime_record["identity_sha256"] = runtime_identity

    final_path, final_reference = _repo_path(
        final_lora_receipt, label="Final-LoRA execution receipt"
    )
    final_record = _file_record(
        final_path, final_reference, label="Final-LoRA execution receipt"
    )
    if _required_sha256(
        final_summary.get("execution_receipt_sha256"),
        label="verified Final-LoRA receipt",
    ) != final_record["sha256"]:
        raise PaperV2DeliveryError("Final-LoRA receipt hash differs from verifier")
    final_binding = {
        "receipt": final_record,
        "verification": {
            "aggregate_rows": int(final_summary.get("aggregate_rows", -1)),
            "prediction_rows_per_role": int(
                final_summary.get("prediction_rows_per_role", -1)
            ),
            "roles": list(final_summary.get("roles", [])),
            "status": "VERIFIED",
        },
    }

    if skip_fleurs:
        return final_binding, None, runtime_record

    fleurs_summary = dict(
        fleurs_verifier(
            fleurs_receipt,
            inference_runtime_lock=inference_runtime_lock,
            training_environment_lock=training_environment_lock,
        )
    )
    if fleurs_summary.get("status") != "VERIFIED":
        raise PaperV2DeliveryError("FLEURS receipt was not formally verified")
    if _required_sha256(
        fleurs_summary.get("inference_runtime_identity_sha256"),
        label="FLEURS inference runtime identity",
    ) != runtime_identity:
        raise PaperV2DeliveryError(
            "FLEURS and Final-LoRA receipts use different inference runtimes"
        )
    fleurs_path, fleurs_reference = _repo_path(
        fleurs_receipt, label="FLEURS execution receipt"
    )
    fleurs_record = _file_record(
        fleurs_path, fleurs_reference, label="FLEURS execution receipt"
    )
    if _required_sha256(
        fleurs_summary.get("receipt_sha256"), label="verified FLEURS receipt"
    ) != fleurs_record["sha256"]:
        raise PaperV2DeliveryError("FLEURS receipt hash differs from verifier")
    fleurs_binding = {
        "receipt": fleurs_record,
        "verification": {
            "prediction_count": int(fleurs_summary.get("prediction_count", -1)),
            "status": "VERIFIED",
        },
    }
    return final_binding, fleurs_binding, runtime_record


def build_or_verify_delivery_receipt(
    *,
    artifacts: Sequence[str | Path],
    output: str | Path = DEFAULT_OUTPUT,
    final_lora_receipt: str | Path = DEFAULT_FINAL_LORA_RECEIPT,
    final_lora_config: str | Path = DEFAULT_FINAL_LORA_CONFIG,
    fleurs_receipt: str | Path = DEFAULT_FLEURS_RECEIPT,
    inference_runtime_lock: str | Path = DEFAULT_INFERENCE_RUNTIME_LOCK,
    training_environment_lock: str | Path = DEFAULT_TRAINING_ENVIRONMENT_LOCK,
    verify_existing: bool = False,
    skip_fleurs: bool = False,
    final_lora_verifier: FinalLoraVerifier = verify_final_lora_execution_receipt,
    fleurs_verifier: FleursVerifier = verify_fleurs_execution_receipt,
) -> dict[str, Any]:
    """Verify upstream evidence and atomically create/verify final delivery."""

    output_path, output_reference = _repo_path(output, label="delivery receipt")
    final_receipt_ref = _repo_path(
        final_lora_receipt, label="Final-LoRA execution receipt"
    )[1]
    final_config_ref = _repo_path(final_lora_config, label="Final-LoRA config")[1]
    fleurs_receipt_ref = _repo_path(
        fleurs_receipt, label="FLEURS execution receipt"
    )[1]
    runtime_ref = _repo_path(
        inference_runtime_lock, label="inference runtime lock"
    )[1]
    training_ref = _repo_path(
        training_environment_lock, label="training environment lock"
    )[1]

    if skip_fleurs and verify_existing:
        raise PaperV2DeliveryError(
            "--skip-fleurs is intermediate verification only and cannot verify "
            "the final delivery receipt"
        )

    final_binding, fleurs_binding, runtime_record = _verified_upstreams(
        final_lora_receipt=final_receipt_ref,
        final_lora_config=final_config_ref,
        fleurs_receipt=fleurs_receipt_ref,
        inference_runtime_lock=runtime_ref,
        training_environment_lock=training_ref,
        skip_fleurs=skip_fleurs,
        final_lora_verifier=final_lora_verifier,
        fleurs_verifier=fleurs_verifier,
    )
    downstream = _artifact_records(artifacts, output_path=output_path)

    if skip_fleurs:
        return {
            "artifact_count": len(downstream),
            "delivery_receipt": None,
            "final_lora": final_binding,
            "inference_runtime": runtime_record,
            "status": "INTERMEDIATE_VERIFIED",
        }
    if fleurs_binding is None:  # Defensive: a complete receipt always has both.
        raise PaperV2DeliveryError("Final delivery requires FLEURS evidence")

    expected: dict[str, Any] = {
        "artifacts": downstream,
        "contract": {
            "artifact_binding": "explicit_path_sha256_bytes_v1",
            "fleurs_required": True,
            "final_lora_required": True,
            "inference_runtime_current_verified": True,
        },
        "inference_runtime": runtime_record,
        "schema_version": DELIVERY_RECEIPT_VERSION,
        "status": "COMPLETE",
        "upstream": {
            "final_lora": final_binding,
            "fleurs": fleurs_binding,
        },
    }
    expected_bytes = _canonical_json_bytes(expected)

    if verify_existing:
        actual = _load_json_object(output_path, label="Trung delivery receipt")
        if output_path.read_bytes() != _canonical_json_bytes(actual):
            raise PaperV2DeliveryError("Existing delivery receipt is not canonical JSON")
        if actual != expected:
            raise PaperV2DeliveryError(
                "Existing delivery receipt differs from current upstream or artifacts"
            )
        receipt_status = "verified_existing"
    else:
        _atomic_write_new(output_path, expected_bytes)
        receipt_status = "written"

    return {
        "artifact_count": len(downstream),
        "delivery_receipt": output_reference,
        "delivery_receipt_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "receipt_status": receipt_status,
        "status": "VERIFIED",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify formal LoRA/FLEURS inference evidence and bind explicit "
            "paper-v2 delivery artifacts."
        )
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Repository-relative downstream artifact; repeat for every file.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument(
        "--final-lora-receipt", default=DEFAULT_FINAL_LORA_RECEIPT.as_posix()
    )
    parser.add_argument(
        "--final-lora-config", default=DEFAULT_FINAL_LORA_CONFIG.as_posix()
    )
    parser.add_argument("--fleurs-receipt", default=DEFAULT_FLEURS_RECEIPT.as_posix())
    parser.add_argument(
        "--inference-runtime-lock",
        default=DEFAULT_INFERENCE_RUNTIME_LOCK.as_posix(),
    )
    parser.add_argument(
        "--training-environment-lock",
        default=DEFAULT_TRAINING_ENVIRONMENT_LOCK.as_posix(),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify the immutable completed receipt instead of creating it.",
    )
    parser.add_argument(
        "--skip-fleurs",
        action="store_true",
        help=(
            "Intermediate verification only; verifies LoRA/artifacts and never "
            "writes the final receipt."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_or_verify_delivery_receipt(
            artifacts=args.artifact,
            output=args.output,
            final_lora_receipt=args.final_lora_receipt,
            final_lora_config=args.final_lora_config,
            fleurs_receipt=args.fleurs_receipt,
            inference_runtime_lock=args.inference_runtime_lock,
            training_environment_lock=args.training_environment_lock,
            verify_existing=bool(args.verify_existing),
            skip_fleurs=bool(args.skip_fleurs),
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DELIVERY_RECEIPT_VERSION",
    "PaperV2DeliveryError",
    "build_or_verify_delivery_receipt",
    "build_parser",
    "main",
]
