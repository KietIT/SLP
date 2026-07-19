from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.inference_runtime import (  # noqa: E402
    INFERENCE_RUNTIME_SCHEMA_VERSION,
    REQUIRED_INFERENCE_SOURCE_PATHS,
    InferenceRuntimeLockError,
    verify_inference_runtime_lock,
)
from src.vitonesr.phat import final_evaluation as _locked_final_evaluation  # noqa: E402
from src.vitonesr.phat.final_evaluation import (  # noqa: E402
    FinalLoraProtocolError,
    ROLE_ORDER,
    _default_predictor as _locked_default_predictor,
    load_final_lora_config,
    run_final_lora_suite,
)
from src.vitonesr.phat.method_contract import verify_method_lock  # noqa: E402
from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.protocol import is_sha256, sha256_file  # noqa: E402


RECEIPT_VERSION = "paper_v2_final_lora_execution_receipt_v1"
DEFAULT_CONFIG = Path("outputs/paper_v2/protocol/final_lora_runtime.yaml")
DEFAULT_INFERENCE_RUNTIME_LOCK = Path(
    "outputs/paper_v2/protocol/inference_runtime_lock.json"
)
DEFAULT_RECEIPT = Path(
    "outputs/paper_v2/protocol/final_lora_execution_receipt.json"
)


class FinalLoraInferenceRuntimeError(ValueError):
    """Raised when cross-machine final LoRA execution is not reproducible."""


def _is_transient_windows_publish_error(error: OSError) -> bool:
    """Return whether OneDrive may briefly be holding an atomic-write target."""

    return isinstance(error, PermissionError) or getattr(error, "winerror", None) == 5


def _retry_atomic_write_bytes(
    writer: Callable[[Path, bytes], None],
    path: Path,
    payload: bytes,
    *,
    timeout_seconds: float = 30.0,
    initial_delay_seconds: float = 0.05,
    max_delay_seconds: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Retry only transient Windows publish failures around the locked writer.

    The method-locked writer still creates, flushes, fsyncs and atomically
    replaces its temporary file on every attempt.  This wrapper only gives a
    OneDrive/antivirus file-handle race a bounded chance to clear.
    """

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    if initial_delay_seconds <= 0 or max_delay_seconds <= 0:
        raise ValueError("retry delays must be positive")

    deadline = monotonic() + timeout_seconds
    delay = initial_delay_seconds
    while True:
        try:
            writer(path, payload)
            return
        except OSError as error:
            if not _is_transient_windows_publish_error(error):
                raise
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            sleep(min(delay, remaining))
            delay = min(delay * 2.0, max_delay_seconds)


@contextmanager
def _scoped_final_atomic_write_retry(
    *,
    timeout_seconds: float = 30.0,
) -> Iterator[None]:
    """Temporarily add bounded OneDrive retries to the locked suite writer."""

    original_writer = _locked_final_evaluation._atomic_write_bytes

    def retrying_writer(path: Path, payload: bytes) -> None:
        _retry_atomic_write_bytes(
            original_writer,
            path,
            payload,
            timeout_seconds=timeout_seconds,
        )

    _locked_final_evaluation._atomic_write_bytes = retrying_writer
    try:
        yield
    finally:
        _locked_final_evaluation._atomic_write_bytes = original_writer


def _load_processor_with_fallback(
    processor_class: Any,
    role: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Load the checkpoint processor, or its immutable base equivalent.

    Phat's checkpoints were written by Transformers 5.12, whose local
    ``tokenizer_config.json`` is not readable by the 4.57 runtime on this
    inference host.  LoRA does not modify the processor, so falling back to
    the decision-locked base processor is semantically identical.  The
    fallback is always pinned to the backbone revision from the authorized
    role config; it never resolves a moving model revision.
    """

    local_processor = role.checkpoint_path / "processor"
    try:
        return processor_class.from_pretrained(
            str(local_processor), *args, **kwargs
        )
    except Exception as local_error:
        model_config = role.config.get("model")
        if not isinstance(model_config, Mapping):
            raise FinalLoraInferenceRuntimeError(
                f"Role {role.role!r} has no authorized model config"
            ) from local_error
        model_name = str(model_config.get("name_or_path", "")).strip()
        revision = str(model_config.get("revision", "")).strip()
        if not model_name or not revision:
            raise FinalLoraInferenceRuntimeError(
                f"Role {role.role!r} has no pinned base processor"
            ) from local_error
        warnings.warn(
            f"Checkpoint processor at {local_processor} is incompatible "
            f"({type(local_error).__name__}: {local_error}); falling back to "
            f"{model_name} at pinned revision {revision}.",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["revision"] = revision
        try:
            return processor_class.from_pretrained(
                model_name, *args, **fallback_kwargs
            )
        except Exception as fallback_error:
            raise FinalLoraInferenceRuntimeError(
                "Could not load either the checkpoint-local processor or the "
                f"pinned base processor {model_name!r}"
            ) from fallback_error


def _compatible_final_predictor(
    role: Any,
    rows: Sequence[Mapping[str, str]],
    inference_contract: Mapping[str, Any],
    **kwargs: Any,
) -> tuple[list[str], dict[str, Any]]:
    """Run the locked predictor with a compatibility-only processor shim.

    All audio, decoding, batching, resume, model and adapter behavior stays in
    the method-locked predictor.  The temporary shim intercepts only its one
    checkpoint-local processor load and is restored even if inference fails.
    """

    import transformers

    original_processor_class = transformers.WhisperProcessor
    local_processor = (role.checkpoint_path / "processor").resolve()

    class CompatibleWhisperProcessor:
        @staticmethod
        def from_pretrained(
            pretrained_model_name_or_path: str | Path,
            *load_args: Any,
            **load_kwargs: Any,
        ) -> Any:
            requested = Path(str(pretrained_model_name_or_path)).resolve()
            if requested != local_processor:
                return original_processor_class.from_pretrained(
                    pretrained_model_name_or_path, *load_args, **load_kwargs
                )
            return _load_processor_with_fallback(
                original_processor_class,
                role,
                *load_args,
                **load_kwargs,
            )

    transformers.WhisperProcessor = CompatibleWhisperProcessor
    try:
        return _locked_default_predictor(
            role,
            rows,
            inference_contract,
            **kwargs,
        )
    finally:
        transformers.WhisperProcessor = original_processor_class


def _repo_path(value: str | Path, *, label: str) -> Path:
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
        raise FinalLoraInferenceRuntimeError(
            f"{label} must be a portable repository-relative POSIX path"
        )
    candidate = ROOT.joinpath(*posix.parts).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise FinalLoraInferenceRuntimeError(f"{label} escapes the repository") from exc
    return candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FinalLoraInferenceRuntimeError(
            f"Execution artifact is outside the repository: {path}"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise FinalLoraInferenceRuntimeError(
                f"Duplicate JSON key in {key!r} while loading evidence"
            )
        value[key] = nested
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise FinalLoraInferenceRuntimeError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FinalLoraInferenceRuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().casefold()
    if not is_sha256(normalized):
        raise FinalLoraInferenceRuntimeError(f"{label} is not a SHA-256")
    return normalized


def _verified_runtime_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete verifier response, including its source profile."""

    if value.get("status") != "VERIFIED":
        raise FinalLoraInferenceRuntimeError(
            "Inference runtime verifier did not return VERIFIED"
        )
    if value.get("schema_version") != INFERENCE_RUNTIME_SCHEMA_VERSION:
        raise FinalLoraInferenceRuntimeError(
            "Inference runtime verifier returned another schema"
        )
    source_paths = value.get("source_component_paths")
    if not isinstance(source_paths, (list, tuple)) or tuple(source_paths) != tuple(
        REQUIRED_INFERENCE_SOURCE_PATHS
    ):
        raise FinalLoraInferenceRuntimeError(
            "Inference runtime verifier returned an incomplete source profile"
        )
    return {
        "status": "VERIFIED",
        "schema_version": INFERENCE_RUNTIME_SCHEMA_VERSION,
        "lock_sha256": _require_sha256(
            value.get("lock_sha256"), label="inference runtime lock SHA-256"
        ),
        "identity_sha256": _require_sha256(
            value.get("identity_sha256"), label="inference runtime identity SHA-256"
        ),
        "training_environment_sha256": _require_sha256(
            value.get("training_environment_sha256"),
            label="training environment artifact SHA-256",
        ),
        "training_environment_identity_sha256": _require_sha256(
            value.get("training_environment_identity_sha256"),
            label="training environment identity SHA-256",
        ),
        "inference_environment_identity_sha256": _require_sha256(
            value.get("inference_environment_identity_sha256"),
            label="inference environment identity SHA-256",
        ),
        "source_component_paths": list(source_paths),
        "source_tree_sha256": _require_sha256(
            value.get("source_tree_sha256"),
            label="inference wrapper source tree SHA-256",
        ),
    }


def _training_environment_path(config: Mapping[str, Any]) -> Path:
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise FinalLoraInferenceRuntimeError("Final LoRA config has no protocol object")
    method_lock_path = _repo_path(
        str(protocol.get("method_lock", "")), label="protocol.method_lock"
    )
    expected_method = _require_sha256(
        protocol.get("expected_method_lock_sha256"),
        label="protocol.expected_method_lock_sha256",
    )
    if not method_lock_path.is_file():
        raise FileNotFoundError(f"Method lock does not exist: {method_lock_path}")
    if sha256_file(method_lock_path) != expected_method:
        raise FinalLoraInferenceRuntimeError(
            "Method lock changed before inference-runtime authorization"
        )
    method_lock = _load_json(method_lock_path, label="Method lock")
    artifacts = method_lock.get("artifacts")
    environment = artifacts.get("environment") if isinstance(artifacts, Mapping) else None
    if not isinstance(environment, Mapping):
        raise FinalLoraInferenceRuntimeError(
            "Method lock has no training-environment artifact binding"
        )
    return _repo_path(
        str(environment.get("path", "")),
        label="method_lock.artifacts.environment.path",
    )


def _file_binding(path: Path, *, label: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return {"path": _display_path(path), "sha256": sha256_file(path)}


def _build_receipt(
    *,
    config: Mapping[str, Any],
    inference_lock_path: Path,
    inference_integrity: Mapping[str, Any],
    method_integrity: Mapping[str, Any],
    suite_result: Mapping[str, Any],
) -> dict[str, Any]:
    inference_integrity = _verified_runtime_summary(inference_integrity)
    expected_training_identity = _require_sha256(
        inference_integrity.get("training_environment_identity_sha256"),
        label="inference lock training_environment_identity_sha256",
    )
    observed_training_identity = _require_sha256(
        method_integrity.get("environment_identity_sha256"),
        label="verified method environment_identity_sha256",
    )
    if observed_training_identity != expected_training_identity:
        raise FinalLoraInferenceRuntimeError(
            "Inference lock is bound to another training environment identity"
        )
    expected_training_artifact = _require_sha256(
        inference_integrity.get("training_environment_sha256"),
        label="inference lock training environment artifact SHA-256",
    )
    observed_training_artifact = _require_sha256(
        method_integrity.get("environment_artifact_sha256"),
        label="verified method environment artifact SHA-256",
    )
    if observed_training_artifact != expected_training_artifact:
        raise FinalLoraInferenceRuntimeError(
            "Inference lock is bound to another training environment artifact"
        )

    runtime_config_sha256 = _require_sha256(
        config.get("_runtime_config_sha256"), label="runtime config SHA-256"
    )
    runtime_config_path = str(config.get("_runtime_config_path", "")).strip()
    if not runtime_config_path:
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA config was not loaded from a repository artifact"
        )

    output = config.get("output")
    if not isinstance(output, Mapping):
        raise FinalLoraInferenceRuntimeError("Final LoRA config has no output object")
    output_root = _repo_path(str(output.get("directory", "")), label="output.directory")

    roles = suite_result.get("roles")
    if not isinstance(roles, (list, tuple)) or tuple(roles) != tuple(ROLE_ORDER):
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA suite did not return exactly the three locked roles"
        )
    try:
        prediction_rows_per_role = int(suite_result["prediction_rows_per_role"])
        aggregate_rows = int(suite_result["aggregate_rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA suite returned invalid row counts"
        ) from exc
    if prediction_rows_per_role != int(config["benchmark"]["expected_rows"]):
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA prediction row count differs from the locked benchmark"
        )
    if aggregate_rows < 1:
        raise FinalLoraInferenceRuntimeError("Final LoRA aggregate is empty")
    output_display = _display_path(output_root)
    if str(suite_result.get("output_directory", "")) != output_display:
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA suite returned another output directory"
        )

    prediction_bindings: dict[str, dict[str, str]] = {}
    prediction_hashes: dict[str, str] = {}
    for role in ROLE_ORDER:
        role_root = output_root / role
        prediction = _file_binding(
            role_root / "predictions.csv", label=f"{role} prediction"
        )
        provenance_path = role_root / "provenance.json"
        provenance = _file_binding(
            provenance_path, label=f"{role} prediction provenance"
        )
        provenance_payload = _load_json(
            provenance_path, label=f"{role} prediction provenance"
        )
        claimed_prediction = _require_sha256(
            provenance_payload.get("prediction_sha256"),
            label=f"{role} provenance prediction_sha256",
        )
        if claimed_prediction != prediction["sha256"]:
            raise FinalLoraInferenceRuntimeError(
                f"{role} provenance does not bind its prediction bytes"
            )
        prediction_hashes[role] = prediction["sha256"]
        prediction_bindings[role] = {
            "prediction_path": prediction["path"],
            "prediction_sha256": prediction["sha256"],
            "provenance_path": provenance["path"],
            "provenance_sha256": provenance["sha256"],
        }

    aggregate_path = _repo_path(
        str(suite_result.get("aggregate", "")), label="suite aggregate"
    )
    expected_aggregate_path = (
        output_root / "aggregate" / str(output.get("aggregate_filename", ""))
    )
    if aggregate_path != expected_aggregate_path:
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA suite returned an unexpected aggregate path"
        )
    aggregate = _file_binding(aggregate_path, label="Final LoRA aggregate")
    aggregate_provenance_path = aggregate_path.parent / "provenance.json"
    aggregate_provenance = _file_binding(
        aggregate_provenance_path, label="Final LoRA aggregate provenance"
    )
    aggregate_payload = _load_json(
        aggregate_provenance_path, label="Final LoRA aggregate provenance"
    )
    if (
        _require_sha256(
            aggregate_payload.get("result_sha256"),
            label="aggregate provenance result_sha256",
        )
        != aggregate["sha256"]
        or aggregate_payload.get("prediction_sha256_by_role") != prediction_hashes
    ):
        raise FinalLoraInferenceRuntimeError(
            "Aggregate provenance does not bind the completed role predictions"
        )

    return {
        "receipt_version": RECEIPT_VERSION,
        "inference_runtime_verified": True,
        "training_runtime_verified_as_current": False,
        "semantic_separation": {
            "inference_runtime": "separately_verified_execution_environment",
            "training_environment": "immutable_checkpoint_provenance",
            "method_lock_source_components_modified": False,
        },
        "inference_runtime": {
            "lock_path": _display_path(inference_lock_path),
            "schema_version": str(inference_integrity.get("schema_version", "")),
            "lock_sha256": _require_sha256(
                inference_integrity.get("lock_sha256"),
                label="inference runtime lock SHA-256",
            ),
            "identity_sha256": _require_sha256(
                inference_integrity.get("identity_sha256"),
                label="inference runtime identity SHA-256",
            ),
            "inference_environment_identity_sha256": _require_sha256(
                inference_integrity.get("inference_environment_identity_sha256"),
                label="inference environment identity SHA-256",
            ),
            "training_environment_identity_sha256": expected_training_identity,
            "training_environment_sha256": _require_sha256(
                inference_integrity.get("training_environment_sha256"),
                label="training environment artifact SHA-256",
            ),
            "source_tree_sha256": _require_sha256(
                inference_integrity.get("source_tree_sha256"),
                label="inference wrapper source tree SHA-256",
            ),
            "source_component_paths": list(
                inference_integrity["source_component_paths"]
            ),
        },
        "training_method": {
            "method_lock_sha256": _require_sha256(
                method_integrity.get("method_lock_sha256"),
                label="verified method lock SHA-256",
            ),
            "method_identity_sha256": _require_sha256(
                method_integrity.get("method_identity_sha256"),
                label="verified method identity SHA-256",
            ),
            "environment_identity_sha256": observed_training_identity,
            "environment_artifact_sha256": observed_training_artifact,
            "source_tree_sha256": _require_sha256(
                method_integrity.get("source_tree_sha256"),
                label="verified training source tree SHA-256",
            ),
        },
        "runtime_config": {
            "path": runtime_config_path,
            "sha256": runtime_config_sha256,
        },
        "suite": {
            "roles": list(ROLE_ORDER),
            "prediction_rows_per_role": prediction_rows_per_role,
            "aggregate_rows": aggregate_rows,
            "output_directory": output_display,
        },
        "predictions": prediction_bindings,
        "aggregate": {
            "result_path": aggregate["path"],
            "result_sha256": aggregate["sha256"],
            "provenance_path": aggregate_provenance["path"],
            "provenance_sha256": aggregate_provenance["sha256"],
        },
    }


def _publish_receipt(path: Path, payload: bytes, *, resume: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"Execution receipt exists; use --resume to verify it: {path}"
            )
        if not path.is_file() or path.read_bytes() != payload:
            raise FinalLoraInferenceRuntimeError(
                "Existing final-LoRA execution receipt is stale or tampered"
            )
        return "verified_existing"

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not resume or not path.is_file() or path.read_bytes() != payload:
                raise
            return "verified_existing"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _csv_data_rows(path: Path, *, label: str) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise FinalLoraInferenceRuntimeError(f"{label} has no CSV header")
            return sum(1 for _ in reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise FinalLoraInferenceRuntimeError(f"{label} is invalid CSV") from exc


def _verify_method_without_current_training_runtime(
    config: Mapping[str, Any],
    inference_integrity: Mapping[str, Any],
    *,
    method_verifier: Callable[..., Mapping[str, str]],
) -> dict[str, Any]:
    protocol = config.get("protocol")
    runtime = config.get("runtime")
    if not isinstance(protocol, Mapping) or not isinstance(runtime, Mapping):
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA config has no protocol/runtime object"
        )
    method_lock = _repo_path(
        str(protocol.get("method_lock", "")), label="protocol.method_lock"
    )
    method_config_path = _repo_path(
        str(protocol.get("method_config", "")), label="protocol.method_config"
    )
    method_config = load_experiment_config(method_config_path)
    method = dict(
        method_verifier(
            method_lock,
            config=method_config,
            repo_root=ROOT,
            formal=False,
            verify_audio=bool(runtime.get("verify_method_audio_sha256", False)),
        )
    )
    if (
        _require_sha256(
            method.get("method_lock_sha256"), label="verified method lock SHA-256"
        )
        != _require_sha256(
            protocol.get("expected_method_lock_sha256"),
            label="protocol.expected_method_lock_sha256",
        )
    ):
        raise FinalLoraInferenceRuntimeError(
            "Verified method lock differs from the runtime config"
        )
    if (
        _require_sha256(
            method.get("environment_identity_sha256"),
            label="verified method training environment identity",
        )
        != _require_sha256(
            inference_integrity.get("training_environment_identity_sha256"),
            label="inference lock training environment identity",
        )
    ):
        raise FinalLoraInferenceRuntimeError(
            "Verified method and inference lock name different training environments"
        )
    if (
        _require_sha256(
            method.get("environment_artifact_sha256"),
            label="verified method training environment artifact",
        )
        != _require_sha256(
            inference_integrity.get("training_environment_sha256"),
            label="inference lock training environment artifact",
        )
    ):
        raise FinalLoraInferenceRuntimeError(
            "Verified method and inference lock bind different training artifacts"
        )
    return method


def verify_final_lora_execution_receipt(
    receipt_path: str | Path = DEFAULT_RECEIPT,
    config_path: str | Path = DEFAULT_CONFIG,
    inference_runtime_lock_path: str | Path = DEFAULT_INFERENCE_RUNTIME_LOCK,
    *,
    verify_current: bool = True,
    runtime_verifier: Callable[..., Mapping[str, Any]] = (
        verify_inference_runtime_lock
    ),
    method_verifier: Callable[..., Mapping[str, str]] = verify_method_lock,
) -> dict[str, Any]:
    """Verify the full final-LoRA execution evidence without rerunning inference."""

    config = load_final_lora_config(config_path)
    receipt = _repo_path(receipt_path, label="execution receipt")
    inference_lock = _repo_path(
        inference_runtime_lock_path, label="inference runtime lock"
    )
    training_environment = _training_environment_path(config)
    runtime_integrity = _verified_runtime_summary(
        dict(
            runtime_verifier(
                inference_lock,
                training_environment,
                ROOT,
                verify_current=verify_current,
            )
        )
    )
    if runtime_integrity["lock_sha256"] != sha256_file(inference_lock):
        raise FinalLoraInferenceRuntimeError(
            "Inference runtime verifier returned a different lock hash"
        )
    method_integrity = _verify_method_without_current_training_runtime(
        config,
        runtime_integrity,
        method_verifier=method_verifier,
    )

    output = config["output"]
    output_root = _repo_path(str(output["directory"]), label="output.directory")
    role_counts = {
        role: _csv_data_rows(
            output_root / role / "predictions.csv", label=f"{role} prediction"
        )
        for role in ROLE_ORDER
    }
    expected_rows = int(config["benchmark"]["expected_rows"])
    if set(role_counts.values()) != {expected_rows}:
        raise FinalLoraInferenceRuntimeError(
            f"Final LoRA role row counts differ from {expected_rows}: {role_counts}"
        )
    aggregate_path = (
        output_root / "aggregate" / str(output["aggregate_filename"])
    )
    aggregate_rows = _csv_data_rows(
        aggregate_path, label="Final LoRA aggregate"
    )
    suite_result = {
        "roles": list(ROLE_ORDER),
        "prediction_rows_per_role": expected_rows,
        "aggregate_rows": aggregate_rows,
        "output_directory": _display_path(output_root),
        "aggregate": _display_path(aggregate_path),
    }
    expected_receipt = _build_receipt(
        config=config,
        inference_lock_path=inference_lock,
        inference_integrity=runtime_integrity,
        method_integrity=method_integrity,
        suite_result=suite_result,
    )
    actual_receipt = _load_json(receipt, label="Final LoRA execution receipt")
    expected_bytes = _canonical_json_bytes(expected_receipt)
    if receipt.read_bytes() != _canonical_json_bytes(actual_receipt):
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA execution receipt is not canonical JSON"
        )
    if actual_receipt != expected_receipt:
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA execution receipt differs from its transitive evidence"
        )
    return {
        **suite_result,
        "status": "VERIFIED",
        "inference_runtime_verified": True,
        "training_runtime_verified_as_current": False,
        "inference_runtime_lock_sha256": runtime_integrity["lock_sha256"],
        "execution_receipt": _display_path(receipt),
        "execution_receipt_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "execution_receipt_status": "verified_existing",
    }


def run_final_lora_inference_runtime(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    inference_runtime_lock_path: str | Path = DEFAULT_INFERENCE_RUNTIME_LOCK,
    receipt_path: str | Path = DEFAULT_RECEIPT,
    resume: bool = False,
    runtime_verifier: Callable[..., Mapping[str, Any]] = (
        verify_inference_runtime_lock
    ),
    method_verifier: Callable[..., Mapping[str, str]] = verify_method_lock,
    suite_runner: Callable[..., Mapping[str, Any]] = run_final_lora_suite,
) -> dict[str, Any]:
    """Run final LoRA on a separately locked, formally verified inference host."""

    config = load_final_lora_config(config_path)
    inference_lock = _repo_path(
        inference_runtime_lock_path, label="inference runtime lock"
    )
    receipt = _repo_path(receipt_path, label="execution receipt")
    if receipt.exists():
        if not resume:
            raise FileExistsError(
                f"Execution receipt exists; use --resume to verify it: {receipt}"
            )
        return verify_final_lora_execution_receipt(
            receipt_path,
            config_path,
            inference_runtime_lock_path,
            verify_current=True,
            runtime_verifier=runtime_verifier,
            method_verifier=method_verifier,
        )

    training_environment = _training_environment_path(config)
    inference_integrity = _verified_runtime_summary(
        dict(
            runtime_verifier(
                inference_lock,
                training_environment,
                ROOT,
                verify_current=True,
            )
        )
    )
    verified_lock_sha256 = inference_integrity["lock_sha256"]
    if verified_lock_sha256 != sha256_file(inference_lock):
        raise FinalLoraInferenceRuntimeError(
            "Inference runtime verifier returned a different lock hash"
        )

    verified_method: dict[str, Any] = {}

    def verify_training_method(
        lock_path: str | Path,
        *,
        config: Mapping[str, Any],
        repo_root: str | Path,
        formal: bool,
        verify_audio: bool = True,
    ) -> Mapping[str, str]:
        if formal is not True:
            raise FinalLoraInferenceRuntimeError(
                "Final suite did not request formal method authorization"
            )
        result = dict(
            method_verifier(
                lock_path,
                config=config,
                repo_root=repo_root,
                formal=False,
                verify_audio=verify_audio,
            )
        )
        expected_training_identity = _require_sha256(
            inference_integrity.get("training_environment_identity_sha256"),
            label="inference lock training environment identity",
        )
        if (
            _require_sha256(
                result.get("environment_identity_sha256"),
                label="method training environment identity",
            )
            != expected_training_identity
        ):
            raise FinalLoraInferenceRuntimeError(
                "Verified method and inference lock name different training environments"
            )
        if (
            _require_sha256(
                result.get("environment_artifact_sha256"),
                label="method training environment artifact",
            )
            != inference_integrity["training_environment_sha256"]
        ):
            raise FinalLoraInferenceRuntimeError(
                "Verified method and inference lock bind different training artifacts"
            )
        verified_method.clear()
        verified_method.update(result)
        return result

    with _scoped_final_atomic_write_retry():
        suite_result = dict(
            suite_runner(
                config,
                resume=resume,
                predictor=_compatible_final_predictor,
                authorization_kwargs={"method_verifier": verify_training_method},
            )
        )
    if not verified_method:
        raise FinalLoraInferenceRuntimeError(
            "Final LoRA suite bypassed method-lock verification"
        )
    receipt_payload = _build_receipt(
        config=config,
        inference_lock_path=inference_lock,
        inference_integrity=inference_integrity,
        method_integrity=verified_method,
        suite_result=suite_result,
    )
    receipt_bytes = _canonical_json_bytes(receipt_payload)
    receipt_status = _publish_receipt(receipt, receipt_bytes, resume=resume)
    return {
        **suite_result,
        "inference_runtime_lock_sha256": str(inference_integrity["lock_sha256"]),
        "execution_receipt": _display_path(receipt),
        "execution_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "execution_receipt_status": receipt_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the three decision-locked final LoRA roles on a separately "
            "captured and locked inference machine."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG).replace("\\", "/"))
    parser.add_argument(
        "--inference-runtime-lock",
        default=str(DEFAULT_INFERENCE_RUNTIME_LOCK).replace("\\", "/"),
    )
    parser.add_argument(
        "--receipt", default=str(DEFAULT_RECEIPT).replace("\\", "/")
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify/reuse exact completed outputs and an exact execution receipt.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify locks, receipt, and every completed artifact without inference.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            result = verify_final_lora_execution_receipt(
                args.receipt,
                args.config,
                args.inference_runtime_lock,
                verify_current=True,
            )
        else:
            result = run_final_lora_inference_runtime(
                config_path=args.config,
                inference_runtime_lock_path=args.inference_runtime_lock,
                receipt_path=args.receipt,
                resume=bool(args.resume),
            )
    except (
        FileExistsError,
        FinalLoraInferenceRuntimeError,
        FinalLoraProtocolError,
        InferenceRuntimeLockError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print("Final LoRA roles: " + ", ".join(result["roles"]))
    print(f"rows per role: {result['prediction_rows_per_role']}")
    print(f"aggregate rows: {result['aggregate_rows']}")
    print(f"execution receipt: {result['execution_receipt']}")
    print(f"receipt status: {result['execution_receipt_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_INFERENCE_RUNTIME_LOCK",
    "DEFAULT_RECEIPT",
    "FinalLoraInferenceRuntimeError",
    "RECEIPT_VERSION",
    "_compatible_final_predictor",
    "_is_transient_windows_publish_error",
    "_load_processor_with_fallback",
    "_retry_atomic_write_bytes",
    "_scoped_final_atomic_write_retry",
    "build_parser",
    "main",
    "run_final_lora_inference_runtime",
    "verify_final_lora_execution_receipt",
]
