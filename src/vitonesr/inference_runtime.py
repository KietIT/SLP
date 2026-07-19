from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.vitonesr.phat.reproducibility import (
    DEFAULT_ENVIRONMENT_PACKAGES,
    EnvironmentCaptureError,
    capture_environment,
    validate_environment_artifact,
)


INFERENCE_RUNTIME_SCHEMA_VERSION = "paper_v2_inference_runtime_v1"
INFERENCE_RUNTIME_STATUS = "LOCKED"
REQUIRED_INFERENCE_SOURCE_PATHS = (
    "scripts/run_external_fleurs_inference_runtime.py",
    "scripts/run_final_lora_inference_runtime.py",
    "src/vitonesr/inference_runtime.py",
)


class InferenceRuntimeLockError(ValueError):
    """Raised when an inference-only runtime lock is invalid or has drifted."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise InferenceRuntimeLockError(f"Cannot read {label}: {path}") from error
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(content) != after.st_size:
        raise InferenceRuntimeLockError(f"{label} changed while it was being read: {path}")
    return content


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InferenceRuntimeLockError(f"Duplicate JSON key in runtime lock input: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    content = _read_stable_bytes(path, label=label)
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceRuntimeLockError(f"Invalid JSON in {label}: {path}") from error
    if not isinstance(value, dict):
        raise InferenceRuntimeLockError(f"{label} must contain a JSON object: {path}")
    return value, content


def _repo_relative_file(path: str | Path, repo_root: Path, *, label: str) -> tuple[Path, str]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(repo_root)
    except (OSError, ValueError) as error:
        raise InferenceRuntimeLockError(
            f"{label} must be an existing file inside the repository: {path}"
        ) from error
    if not resolved.is_file():
        raise InferenceRuntimeLockError(f"{label} is not a regular file: {path}")
    return resolved, relative.as_posix()


def _capture_current_environment(
    package_names: Sequence[str], repo_root: Path
) -> dict[str, Any]:
    try:
        captured = capture_environment(
            repo_root=repo_root,
            package_names=tuple(package_names),
            required_packages=(),
            required_revisions=(),
            revisions={},
            cli_args={"profile": "paper_v2_inference_runtime"},
            formal=False,
        )
    except EnvironmentCaptureError as error:
        raise InferenceRuntimeLockError(str(error)) from error
    environment = captured["environment"]
    stable: dict[str, Any] = {
        "capture_mode": "formal_inference",
        "packages": dict(environment["packages"]),
        "platform": dict(environment["platform"]),
        "python": dict(environment["python"]),
        "runtime": dict(environment["runtime"]),
    }
    missing = sorted(name for name, version in stable["packages"].items() if not version)
    if missing:
        raise InferenceRuntimeLockError(
            "Inference runtime is missing required package versions: " + ", ".join(missing)
        )
    stable["identity_sha256"] = _sha256_json({"inference_environment": stable})
    return stable


def _validate_environment_identity(environment: Mapping[str, Any]) -> None:
    required = {"capture_mode", "identity_sha256", "packages", "platform", "python", "runtime"}
    if set(environment) != required:
        raise InferenceRuntimeLockError(
            "Inference environment fields do not match the locked schema"
        )
    if environment.get("capture_mode") != "formal_inference":
        raise InferenceRuntimeLockError("Inference environment capture mode is invalid")
    for key in ("packages", "platform", "python", "runtime"):
        if not isinstance(environment.get(key), Mapping):
            raise InferenceRuntimeLockError(f"Inference environment {key} must be an object")
    packages = environment["packages"]
    if not packages or any(not isinstance(name, str) or not name for name in packages):
        raise InferenceRuntimeLockError("Inference environment package inventory is empty/invalid")
    if any(not isinstance(version, str) or not version for version in packages.values()):
        raise InferenceRuntimeLockError("Inference environment has an unversioned package")
    unsigned = dict(environment)
    recorded = unsigned.pop("identity_sha256", None)
    expected = _sha256_json({"inference_environment": unsigned})
    if recorded != expected:
        raise InferenceRuntimeLockError("Inference environment identity hash is invalid")


def _stable_environment_projection(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Return runtime fields that must remain invariant across an inference run.

    Deterministic-algorithm and cuDNN tuning flags are deliberately excluded: the
    formal runners set those process-local flags after their initial verification.
    They remain recorded in the full lock for auditability, but changing them does
    not falsely claim that Python, a package, CUDA, cuDNN or the GPU has drifted.
    """

    runtime = environment["runtime"]
    cuda = runtime.get("cuda")
    cudnn = runtime.get("cudnn")
    if not isinstance(cuda, Mapping):
        raise InferenceRuntimeLockError("Inference CUDA runtime record is invalid")
    if cudnn is not None and not isinstance(cudnn, Mapping):
        raise InferenceRuntimeLockError("Inference cuDNN runtime record is invalid")
    cudnn_mapping = {} if cudnn is None else cudnn
    return {
        "packages": dict(environment["packages"]),
        "platform": dict(environment["platform"]),
        "python": dict(environment["python"]),
        "runtime": {
            "cuda": {
                "available": cuda.get("available"),
                "compiled_version": cuda.get("compiled_version"),
                "device_count": cuda.get("device_count"),
                "devices": cuda.get("devices"),
                "query_status": cuda.get("query_status"),
            },
            "cudnn": {
                "available": cudnn_mapping.get("available"),
                "version": cudnn_mapping.get("version"),
            },
            "torch_version": runtime.get("torch_version"),
        },
    }


def _source_inventory(
    source_paths: Sequence[str | Path], repo_root: Path
) -> dict[str, Any]:
    if not source_paths:
        raise InferenceRuntimeLockError(
            "At least one explicit inference wrapper source path is required"
        )
    components: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in source_paths:
        resolved, relative = _repo_relative_file(raw_path, repo_root, label="source path")
        if relative in seen:
            raise InferenceRuntimeLockError(f"Duplicate source path: {relative}")
        seen.add(relative)
        content = _read_stable_bytes(resolved, label=f"source component {relative}")
        components.append(
            {"bytes": len(content), "path": relative, "sha256": _sha256_bytes(content)}
        )
    components.sort(key=lambda item: item["path"])
    component_paths = {str(item["path"]) for item in components}
    missing_required = sorted(set(REQUIRED_INFERENCE_SOURCE_PATHS) - component_paths)
    if missing_required:
        raise InferenceRuntimeLockError(
            "Inference source profile is missing required components: "
            + ", ".join(missing_required)
        )
    return {
        "components": components,
        "profile": "paper_v2_formal_inference_wrappers_v1",
        "required_paths": list(REQUIRED_INFERENCE_SOURCE_PATHS),
        "tree_sha256": _sha256_json({"components": components}),
    }


def _training_environment_binding(
    training_environment_path: str | Path, repo_root: Path
) -> dict[str, Any]:
    resolved, relative = _repo_relative_file(
        training_environment_path, repo_root, label="training environment"
    )
    artifact, content = _load_json(resolved, label="training environment")
    try:
        validate_environment_artifact(artifact)
    except EnvironmentCaptureError as error:
        raise InferenceRuntimeLockError(
            f"Training environment artifact is invalid: {error}"
        ) from error
    environment = artifact.get("environment")
    assert isinstance(environment, Mapping)
    return {
        "binding_status": "PRESERVED",
        "capture_mode": environment.get("capture_mode"),
        "identity_sha256": artifact["identity_sha256"],
        "path": relative,
        "schema_version": artifact["schema_version"],
        "sha256": _sha256_bytes(content),
    }


def _identity_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": artifact.get("contract"),
        "inference_environment": artifact.get("inference_environment"),
        "schema_version": artifact.get("schema_version"),
        "source": artifact.get("source"),
        "status": artifact.get("status"),
        "training_environment": artifact.get("training_environment"),
    }


def _validate_artifact_shape(artifact: Mapping[str, Any]) -> None:
    expected_fields = {
        "captured_at_utc",
        "contract",
        "identity_sha256",
        "inference_environment",
        "schema_version",
        "source",
        "status",
        "training_environment",
    }
    if set(artifact) != expected_fields:
        raise InferenceRuntimeLockError("Inference runtime lock fields do not match the schema")
    if artifact.get("schema_version") != INFERENCE_RUNTIME_SCHEMA_VERSION:
        raise InferenceRuntimeLockError("Unsupported inference runtime lock schema")
    if artifact.get("status") != INFERENCE_RUNTIME_STATUS:
        raise InferenceRuntimeLockError("Inference runtime lock is not LOCKED")
    contract = artifact.get("contract")
    if contract != {
        "mode": "inference_only",
        "runtime_policy": "capture_and_verify_exact_current_runtime_v1",
        "training_environment_policy": "preserve_original_training_identity_v1",
    }:
        raise InferenceRuntimeLockError("Inference runtime contract is invalid")
    timestamp = artifact.get("captured_at_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise InferenceRuntimeLockError("Inference runtime capture timestamp is invalid")
    try:
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise InferenceRuntimeLockError("Inference runtime capture timestamp is invalid") from error
    if not isinstance(artifact.get("training_environment"), Mapping):
        raise InferenceRuntimeLockError("Training environment binding is missing")
    if not isinstance(artifact.get("inference_environment"), Mapping):
        raise InferenceRuntimeLockError("Inference environment binding is missing")
    if not isinstance(artifact.get("source"), Mapping):
        raise InferenceRuntimeLockError("Inference source binding is missing")
    _validate_environment_identity(artifact["inference_environment"])
    if artifact.get("identity_sha256") != _sha256_json(_identity_payload(artifact)):
        raise InferenceRuntimeLockError("Inference runtime lock identity hash is invalid")


def _atomic_write_no_overwrite(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite existing inference runtime lock: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def capture_inference_runtime_lock(
    output_path: str | Path,
    training_environment_path: str | Path,
    source_paths: Sequence[str | Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Capture and atomically write an inference-only environment extension.

    The original training environment remains immutable and is bound by both its
    raw-file SHA-256 and its validated identity.  This lock records the machine
    that performs inference without claiming that it was the training machine.
    """

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise InferenceRuntimeLockError(f"Repository root is not a directory: {repo_root}")
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing inference runtime lock: {destination}"
        )

    training_environment = _training_environment_binding(training_environment_path, root)
    source = _source_inventory(source_paths, root)
    inference_environment = _capture_current_environment(
        DEFAULT_ENVIRONMENT_PACKAGES, root
    )
    artifact: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "contract": {
            "mode": "inference_only",
            "runtime_policy": "capture_and_verify_exact_current_runtime_v1",
            "training_environment_policy": "preserve_original_training_identity_v1",
        },
        "inference_environment": inference_environment,
        "schema_version": INFERENCE_RUNTIME_SCHEMA_VERSION,
        "source": source,
        "status": INFERENCE_RUNTIME_STATUS,
        "training_environment": training_environment,
    }
    artifact["identity_sha256"] = _sha256_json(_identity_payload(artifact))
    _validate_artifact_shape(artifact)
    _atomic_write_no_overwrite(destination, _canonical_json_bytes(artifact))
    return artifact


def verify_inference_runtime_lock(
    path: str | Path,
    training_environment_path: str | Path,
    repo_root: str | Path,
    verify_current: bool = True,
) -> dict[str, Any]:
    """Verify lock identity, old environment, wrapper sources and current runtime."""

    root = Path(repo_root).resolve(strict=True)
    resolved_lock = Path(path)
    if not resolved_lock.is_absolute():
        resolved_lock = root / resolved_lock
    artifact, lock_content = _load_json(resolved_lock, label="inference runtime lock")
    _validate_artifact_shape(artifact)

    expected_training = _training_environment_binding(training_environment_path, root)
    if artifact["training_environment"] != expected_training:
        raise InferenceRuntimeLockError("Training environment binding has changed")

    source = artifact["source"]
    if set(source) != {"components", "profile", "required_paths", "tree_sha256"}:
        raise InferenceRuntimeLockError("Inference source profile fields are invalid")
    if source.get("profile") != "paper_v2_formal_inference_wrappers_v1":
        raise InferenceRuntimeLockError("Inference source profile is invalid")
    if source.get("required_paths") != list(REQUIRED_INFERENCE_SOURCE_PATHS):
        raise InferenceRuntimeLockError("Inference source required-path profile is invalid")
    components = source.get("components")
    if not isinstance(components, list) or not components:
        raise InferenceRuntimeLockError("Inference runtime lock has no source components")
    locked_paths: list[str] = []
    prior_path = ""
    for component in components:
        if not isinstance(component, Mapping):
            raise InferenceRuntimeLockError("Invalid inference source component")
        if set(component) != {"bytes", "path", "sha256"}:
            raise InferenceRuntimeLockError("Invalid inference source component fields")
        relative = component.get("path")
        if not isinstance(relative, str) or not relative or relative <= prior_path:
            raise InferenceRuntimeLockError(
                "Inference source paths must be unique and canonically sorted"
            )
        prior_path = relative
        locked_paths.append(relative)
    current_source = _source_inventory(locked_paths, root)
    if source != current_source:
        raise InferenceRuntimeLockError("Locked inference wrapper source has changed")

    locked_environment = artifact["inference_environment"]
    if verify_current:
        current_environment = _capture_current_environment(
            tuple(sorted(locked_environment["packages"])), root
        )
        if _stable_environment_projection(current_environment) != (
            _stable_environment_projection(locked_environment)
        ):
            changed_packages = sorted(
                name
                for name in locked_environment["packages"]
                if current_environment["packages"].get(name)
                != locked_environment["packages"].get(name)
            )
            detail = (
                " (package drift: " + ", ".join(changed_packages) + ")"
                if changed_packages
                else ""
            )
            raise InferenceRuntimeLockError(
                "Current inference Python/package/platform/CUDA/GPU runtime does not "
                "match the lock" + detail
            )

    training = artifact["training_environment"]
    return {
        "identity_sha256": artifact["identity_sha256"],
        "inference_environment_identity_sha256": locked_environment["identity_sha256"],
        "lock_sha256": _sha256_bytes(lock_content),
        "schema_version": artifact["schema_version"],
        "source_component_paths": locked_paths,
        "source_tree_sha256": source["tree_sha256"],
        "status": "VERIFIED",
        "training_environment_identity_sha256": training["identity_sha256"],
        "training_environment_sha256": training["sha256"],
    }
