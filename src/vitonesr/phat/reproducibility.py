from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ENVIRONMENT_SCHEMA_VERSION = "paper_v2_environment_v1"
DEFAULT_ENVIRONMENT_PACKAGES = (
    "accelerate",
    "datasets",
    "huggingface-hub",
    "jiwer",
    "librosa",
    "matplotlib",
    "numpy",
    "pandas",
    "peft",
    "pyyaml",
    "regex",
    "soundfile",
    "torch",
    "torchaudio",
    "tqdm",
    "transformers",
    "unidecode",
)
DEFAULT_REQUIRED_REVISIONS = ("base_model",)

_REVISION_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{12,}", re.IGNORECASE
)
_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "password", "secret", "token")
_FLOATING_REVISIONS = {"", "head", "latest", "main", "master", "none", "null", "unknown"}


class EnvironmentCaptureError(ValueError):
    """Raised when an environment artifact cannot satisfy its contract."""


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


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalize_names(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized: set[str] = set()
    for raw_value in values:
        value = str(raw_value).strip().lower()
        if not value:
            raise EnvironmentCaptureError(f"{label} cannot contain an empty name")
        normalized.add(value)
    return tuple(sorted(normalized))


def normalize_revisions(revisions: Mapping[str, str] | None) -> dict[str, str]:
    """Return validated, sorted immutable-revision labels.

    Values may be commit hashes, content hashes, or immutable release identifiers.
    Known floating references are rejected so a formal artifact cannot silently bind
    to a moving target.
    """

    normalized: dict[str, str] = {}
    for raw_key, raw_value in (revisions or {}).items():
        key = str(raw_key).strip().lower()
        value = str(raw_value).strip()
        if not _REVISION_KEY_RE.fullmatch(key):
            raise EnvironmentCaptureError(
                f"Invalid revision key {raw_key!r}; use lowercase letters, digits, '.', '-' or '_'"
            )
        if value.lower() in _FLOATING_REVISIONS:
            raise EnvironmentCaptureError(
                f"Revision {key!r} must identify an immutable version, not {value!r}"
            )
        if any(character in value for character in ("\n", "\r", "\x00")):
            raise EnvironmentCaptureError(f"Revision {key!r} contains control characters")
        if _CREDENTIAL_VALUE_RE.search(value):
            raise EnvironmentCaptureError(
                f"Revision {key!r} resembles a credential and must not be stored"
            )
        if _looks_like_absolute_path(value):
            raise EnvironmentCaptureError(
                f"Revision {key!r} must not contain an absolute filesystem path"
            )
        normalized[key] = value
    return dict(sorted(normalized.items()))


def _looks_like_absolute_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("\\\\")
        or bool(_WINDOWS_ABSOLUTE_PATH_RE.match(value))
    )


def _sanitize_cli_value(value: Any, *, key: str) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _looks_like_absolute_path(value):
            raise EnvironmentCaptureError(
                f"CLI field {key!r} contains an absolute filesystem path"
            )
        if any(character in value for character in ("\n", "\r", "\x00")):
            raise EnvironmentCaptureError(f"CLI field {key!r} contains control characters")
        if _CREDENTIAL_VALUE_RE.search(value):
            raise EnvironmentCaptureError(
                f"CLI field {key!r} resembles a credential and must not be stored"
            )
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_cli_value(item, key=key) for item in value]
    raise EnvironmentCaptureError(
        f"CLI field {key!r} has unsupported type {type(value).__name__}"
    )


def sanitize_cli_args(cli_args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only path-free, non-secret CLI values in a canonical order."""

    sanitized: dict[str, Any] = {}
    for raw_key, raw_value in (cli_args or {}).items():
        key = str(raw_key).strip().lower()
        if not _REVISION_KEY_RE.fullmatch(key):
            raise EnvironmentCaptureError(f"Invalid CLI field name: {raw_key!r}")
        if any(part in key for part in _SENSITIVE_KEY_PARTS):
            raise EnvironmentCaptureError(
                f"Sensitive CLI field {raw_key!r} must not be stored in the artifact"
            )
        sanitized[key] = _sanitize_cli_value(raw_value, key=key)
    return dict(sorted(sanitized.items()))


def _capture_python_and_platform() -> dict[str, Any]:
    return {
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
            "version": platform.version(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": [
                int(sys.version_info.major),
                int(sys.version_info.minor),
                int(sys.version_info.micro),
            ],
        },
    }


def _safe_bool_call(callable_value: Callable[[], Any], default: bool = False) -> bool:
    try:
        return bool(callable_value())
    except Exception:
        return default


def _capture_torch_runtime(torch_module: Any) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda) and _safe_bool_call(cuda.is_available)
    cuda_query_status = "available" if cuda_available else "unavailable"
    devices: list[dict[str, Any]] = []
    if cuda_available:
        try:
            device_count = int(cuda.device_count())
            for index in range(device_count):
                properties = cuda.get_device_properties(index)
                capability = cuda.get_device_capability(index)
                devices.append(
                    {
                        "compute_capability": [int(capability[0]), int(capability[1])],
                        "index": index,
                        "name": str(properties.name),
                        "total_memory_bytes": int(properties.total_memory),
                    }
                )
        except Exception:
            cuda_query_status = "query_failed"
            devices = []

    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None) if backends is not None else None
    cudnn_version: int | None = None
    cudnn_available = False
    if cudnn is not None:
        cudnn_available = _safe_bool_call(cudnn.is_available)
        try:
            raw_cudnn_version = cudnn.version()
            cudnn_version = (
                None if raw_cudnn_version is None else int(raw_cudnn_version)
            )
        except Exception:
            cudnn_version = None

    torch_version = getattr(torch_module, "version", None)
    deterministic_enabled = False
    deterministic_query = getattr(
        torch_module, "are_deterministic_algorithms_enabled", None
    )
    if callable(deterministic_query):
        deterministic_enabled = _safe_bool_call(deterministic_query)

    return {
        "cuda": {
            "available": cuda_available,
            "compiled_version": getattr(torch_version, "cuda", None),
            "device_count": len(devices),
            "devices": devices,
            "query_status": cuda_query_status,
        },
        "cudnn": {
            "available": cudnn_available,
            "benchmark": bool(getattr(cudnn, "benchmark", False)),
            "deterministic": bool(getattr(cudnn, "deterministic", False)),
            "version": cudnn_version,
        },
        "deterministic_algorithms_enabled": deterministic_enabled,
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
    }


def _capture_package_versions(
    package_names: Sequence[str],
    version_getter: Callable[[str], str],
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in _normalize_names(package_names, label="package_names"):
        try:
            value = str(version_getter(package_name)).strip()
        except importlib.metadata.PackageNotFoundError:
            value = ""
        except Exception:
            value = ""
        versions[package_name] = value or None
    return versions


def _run_git(
    repo_root: Path,
    arguments: Sequence[str],
    runner: Callable[..., Any],
) -> str | None:
    try:
        completed = runner(
            [
                "git",
                "-c",
                f"safe.directory={repo_root.as_posix()}",
                "-C",
                str(repo_root),
                *arguments,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if int(getattr(completed, "returncode", 1)) != 0:
        return None
    stdout = getattr(completed, "stdout", "")
    if not isinstance(stdout, str):
        stdout = stdout.decode("utf-8", errors="replace")
    return stdout.replace("\r\n", "\n")


def _capture_repository(
    repo_root: Path,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    commit_output = _run_git(repo_root, ["rev-parse", "HEAD"], runner)
    status_output = _run_git(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        runner,
    )
    diff_output = _run_git(
        repo_root,
        ["diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        runner,
    )
    commit = (commit_output or "").strip().lower() or None
    if commit is not None and not _GIT_COMMIT_RE.fullmatch(commit):
        commit = None

    if status_output is None:
        return {
            "available": commit is not None,
            "commit": commit,
            "dirty": None,
            "changed_entry_count": None,
            "diff_sha256": None,
            "status_sha256": None,
        }

    status_lines = sorted(line for line in status_output.splitlines() if line)
    normalized_status = "\n".join(status_lines)
    normalized_diff = (diff_output or "").rstrip("\n")
    return {
        "available": commit is not None,
        "commit": commit,
        "dirty": bool(status_lines),
        "changed_entry_count": len(status_lines),
        "diff_sha256": hashlib.sha256(normalized_diff.encode("utf-8")).hexdigest(),
        "status_sha256": hashlib.sha256(normalized_status.encode("utf-8")).hexdigest(),
    }


def _captured_at_utc(now: datetime | None) -> str:
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise EnvironmentCaptureError("capture timestamp must be timezone-aware")
    return (
        timestamp.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _validate_formal_environment(
    identity: Mapping[str, Any],
    *,
    required_packages: Sequence[str],
    required_revisions: Sequence[str],
    allow_dirty_repository: bool,
) -> None:
    packages = identity["packages"]
    missing_packages = [
        name
        for name in _normalize_names(required_packages, label="required_packages")
        if not packages.get(name)
    ]
    if missing_packages:
        raise EnvironmentCaptureError(
            "Formal environment capture is missing required package versions: "
            + ", ".join(missing_packages)
        )

    repository = identity["repository"]
    if not repository.get("commit"):
        raise EnvironmentCaptureError(
            "Formal environment capture requires an exact repository commit"
        )
    dirty = repository.get("dirty")
    if dirty is not False and not allow_dirty_repository:
        raise EnvironmentCaptureError(
            "Formal environment capture requires a clean repository working tree"
        )
    if dirty is True and allow_dirty_repository:
        if (
            not repository.get("status_sha256")
            or not repository.get("diff_sha256")
            or int(repository.get("changed_entry_count") or 0) < 1
        ):
            raise EnvironmentCaptureError(
                "Dirty formal capture requires hashed repository status and diff state"
            )
    if dirty is None:
        raise EnvironmentCaptureError(
            "Formal environment capture could not determine repository state"
        )

    revisions = identity["revisions"]
    missing_revisions = [
        name
        for name in _normalize_names(required_revisions, label="required_revisions")
        if not revisions.get(name)
    ]
    if missing_revisions:
        raise EnvironmentCaptureError(
            "Formal environment capture is missing required revisions: "
            + ", ".join(missing_revisions)
        )


def capture_environment(
    *,
    repo_root: str | Path,
    revisions: Mapping[str, str] | None = None,
    package_names: Sequence[str] = DEFAULT_ENVIRONMENT_PACKAGES,
    required_packages: Sequence[str] = DEFAULT_ENVIRONMENT_PACKAGES,
    required_revisions: Sequence[str] = DEFAULT_REQUIRED_REVISIONS,
    cli_args: Mapping[str, Any] | None = None,
    formal: bool = False,
    allow_dirty_repository: bool = False,
    captured_at: datetime | None = None,
    torch_module: Any | None = None,
    version_getter: Callable[[str], str] | None = None,
    command_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Capture a path-free paper runtime record with a stable identity hash.

    ``captured_at_utc`` is deliberately outside the hashed identity.  Consequently,
    repeated captures of the same runtime, revisions, CLI contract and repository
    state produce the same ``identity_sha256`` even at different times.
    """

    normalized_packages = _normalize_names(package_names, label="package_names")
    required_package_names = _normalize_names(
        required_packages, label="required_packages"
    )
    packages_to_capture = tuple(
        sorted(set(normalized_packages) | set(required_package_names))
    )
    getter = version_getter or importlib.metadata.version
    runner = command_runner or subprocess.run
    selected_torch = torch if torch_module is None else torch_module

    system_identity = _capture_python_and_platform()
    identity: dict[str, Any] = {
        "capture_mode": "formal" if formal else "diagnostic",
        "repository_policy": (
            "content_locked_dirty_allowed"
            if formal and allow_dirty_repository
            else "clean_only"
        ),
        "cli_args": sanitize_cli_args(cli_args),
        "packages": _capture_package_versions(packages_to_capture, getter),
        "platform": system_identity["platform"],
        "python": system_identity["python"],
        "repository": _capture_repository(Path(repo_root), runner),
        "revisions": normalize_revisions(revisions),
        "runtime": _capture_torch_runtime(selected_torch),
    }
    if formal:
        _validate_formal_environment(
            identity,
            required_packages=required_package_names,
            required_revisions=required_revisions,
            allow_dirty_repository=allow_dirty_repository,
        )

    hash_payload = {
        "environment": identity,
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
    }
    return {
        "captured_at_utc": _captured_at_utc(captured_at),
        "environment": identity,
        "identity_sha256": _sha256_json(hash_payload),
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
    }


def validate_environment_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise EnvironmentCaptureError("Unsupported environment artifact schema")
    environment = artifact.get("environment")
    if not isinstance(environment, Mapping):
        raise EnvironmentCaptureError("Environment artifact has no environment mapping")
    expected_hash = _sha256_json(
        {
            "environment": dict(environment),
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        }
    )
    if artifact.get("identity_sha256") != expected_hash:
        raise EnvironmentCaptureError("Environment artifact identity hash is invalid")
    def reject_unsafe_value(value: Any) -> None:
        if isinstance(value, str):
            if _looks_like_absolute_path(value):
                raise EnvironmentCaptureError(
                    "Environment artifact contains an absolute path"
                )
            if _CREDENTIAL_VALUE_RE.search(value):
                raise EnvironmentCaptureError(
                    "Environment artifact contains a credential-like value"
                )
        elif isinstance(value, Mapping):
            for nested in value.values():
                reject_unsafe_value(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                reject_unsafe_value(nested)

    reject_unsafe_value(artifact)


def write_environment_artifact(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write canonical JSON, refusing overwrite by default."""

    validate_environment_artifact(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(dict(artifact)))
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Refusing to overwrite existing environment artifact: {destination}"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch and return the applied settings."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if torch.cuda.is_available() and deterministic:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return {
        "seed": seed,
        "deterministic": deterministic,
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
    }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
