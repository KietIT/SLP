"""Fail-closed provenance verification for formal paper-v2 predictions.

Analysis code must not turn an arbitrary canonical-looking CSV into paper
evidence.  This module verifies the reviewed decision first, then binds every
prediction to its sidecar, benchmark, protocol locks, locked configuration,
and immutable config/checkpoint identities.  Legacy callers remain available
by simply not opting into formal verification at their entry point.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .analysis import CANONICAL_PREDICTION_COLUMNS, METRIC_VERSION
from .artifact_bundle import BUNDLE_PROTOCOL_VERSION, canonical_json_bytes
from .final_benchmark import FinalBenchmarkError, verify_final_benchmark_lock
from .noise_protocol import NoiseProtocolError, verify_noise_split_lock
from .phat.method_contract import (
    MethodContractError,
    verify_method_artifact_bindings,
)
from .phat.protocol import ProtocolIntegrityError, verify_test_decision_lock


ROOT = Path(__file__).resolve().parents[2]
ZERO_SHOT_VERSION = "paper_v2_zero_shot_prediction_v1"
FINAL_LORA_VERSION = "paper_v2_final_lora_prediction_v1"
FLEURS_VERSION = "paper_v2_fleurs_prediction_v3"
NOISY_DEV_VERSIONS = frozenset({"prediction_evaluation_v3", "prediction_evaluation_v4"})
FORMAL_VERSIONS = frozenset(
    {ZERO_SHOT_VERSION, FINAL_LORA_VERSION, FLEURS_VERSION, *NOISY_DEV_VERSIONS}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class PredictionEvidenceError(ValueError):
    """Raised when a formal prediction or its transitive evidence is invalid."""


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    integrity: Mapping[str, Any]
    configurations: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class BenchmarkEvidence:
    path: Path
    sha256: str
    row_count: int
    final_benchmark_lock_path: Path | None
    final_benchmark_lock_sha256: str


@dataclass(frozen=True, slots=True)
class PredictionEvidence:
    prediction_path: Path
    prediction_sha256: str
    provenance_path: Path
    provenance_sha256: str
    provenance_version: str
    row_count: int
    configuration_id: str
    role: str


@dataclass(frozen=True, slots=True)
class FormalPredictionSet:
    decision: DecisionEvidence
    benchmark: BenchmarkEvidence
    predictions: tuple[PredictionEvidence, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PredictionEvidenceError("Provenance contract is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value).casefold()))


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value).casefold()
    if not _is_sha256(normalized):
        raise PredictionEvidenceError(f"{label} is not a SHA-256 digest")
    return normalized


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionEvidenceError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PredictionEvidenceError(f"{label} must be a JSON object: {path}")
    return value


def _repo_path(reference: object, *, root: Path, label: str) -> Path:
    text = str(reference).strip()
    candidate = Path(text)
    path_segments = text.split("/")
    if (
        not text
        or "\\" in text
        or "//" in text
        or any(segment in {"", "."} for segment in path_segments)
        or candidate.is_absolute()
        or _WINDOWS_ABSOLUTE_RE.match(text)
        or any(part == ".." for part in candidate.parts)
    ):
        raise PredictionEvidenceError(
            f"{label} must be a canonical repository-relative POSIX path"
        )
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PredictionEvidenceError(f"{label} escapes the repository") from exc
    return resolved


def _canonical_number(value: object, *, label: str) -> str:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise PredictionEvidenceError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise PredictionEvidenceError(f"{label} must be finite and non-negative")
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _read_csv_identity(path: Path) -> tuple[int, dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            expected = tuple(CANONICAL_PREDICTION_COLUMNS)
            if columns != expected:
                raise PredictionEvidenceError(
                    f"{path}: formal prediction schema differs; expected={list(expected)}, "
                    f"found={list(columns)}"
                )
            row_count = 0
            metadata: dict[str, str] | None = None
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(row.get(column) is None for column in expected):
                    raise PredictionEvidenceError(
                        f"{path}: row {row_number} has missing or extra CSV cells"
                    )
                current = {
                    field: str(row[field])
                    for field in (
                        "dataset",
                        "model",
                        "model_size",
                        "train_type",
                        "lambda",
                        "seed",
                    )
                }
                if metadata is None:
                    metadata = current
                elif current != metadata:
                    raise PredictionEvidenceError(
                        f"{path}: formal run metadata must be constant"
                    )
                row_count += 1
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PredictionEvidenceError(f"Cannot read prediction CSV {path}: {exc}") from exc
    if row_count < 1 or metadata is None:
        raise PredictionEvidenceError(f"Formal prediction is empty: {path}")
    return row_count, metadata


def _manifest_row_count(path: Path) -> int:
    try:
        if path.suffix.casefold() == ".jsonl":
            count = 0
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise PredictionEvidenceError(
                            f"Benchmark JSONL line {line_number} is blank"
                        )
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise PredictionEvidenceError(
                            f"Benchmark JSONL line {line_number} is not an object"
                        )
                    count += 1
            return count
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise PredictionEvidenceError("Benchmark CSV has no header")
                return sum(1 for _ in reader)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        raise PredictionEvidenceError(f"Cannot read benchmark manifest {path}: {exc}") from exc
    raise PredictionEvidenceError("Formal benchmark manifest must be CSV or JSONL")


def verify_decision_evidence(
    decision_path: str | Path,
    split_lock_path: str | Path,
    *,
    decision_verifier: Callable[..., Mapping[str, Any]] = verify_test_decision_lock,
) -> DecisionEvidence:
    decision_file = Path(decision_path)
    try:
        integrity = dict(
            decision_verifier(
                split_lock_path=split_lock_path,
                decision_lock_path=decision_file,
                verify_checkpoints=False,
            )
        )
    except (ProtocolIntegrityError, FileNotFoundError, OSError, ValueError) as exc:
        raise PredictionEvidenceError(f"Decision verification failed: {exc}") from exc
    actual_sha = sha256_file(decision_file)
    if actual_sha != _require_sha256(
        integrity.get("decision_lock_sha256"), label="verified decision lock"
    ):
        raise PredictionEvidenceError("Decision changed after verification")
    raw = _load_json(decision_file, label="method/lambda decision")
    configurations: dict[str, Mapping[str, Any]] = {}
    raw_configurations = integrity.get("locked_configurations")
    if not isinstance(raw_configurations, Sequence) or isinstance(
        raw_configurations, (str, bytes)
    ):
        raise PredictionEvidenceError("Verified decision has no locked configurations")
    for item in raw_configurations:
        if not isinstance(item, Mapping):
            raise PredictionEvidenceError("Verified decision configuration is malformed")
        configuration_id = str(item.get("configuration_id", "")).strip()
        if not configuration_id or configuration_id in configurations:
            raise PredictionEvidenceError("Verified decision configuration IDs are invalid")
        configurations[configuration_id] = dict(item)
    return DecisionEvidence(
        path=decision_file,
        sha256=actual_sha,
        raw=raw,
        integrity=integrity,
        configurations=configurations,
    )


def _method_and_noise_evidence(
    decision: DecisionEvidence, *, root: Path
) -> tuple[str, str, Mapping[str, Any]]:
    method_path = _repo_path(
        decision.raw.get("method_lock", ""), root=root, label="decision.method_lock"
    )
    expected_method_sha = _require_sha256(
        decision.integrity.get("method_lock_sha256"), label="decision method lock"
    )
    try:
        verified_method = verify_method_artifact_bindings(
            method_path,
            repo_root=root,
            formal=True,
        )
    except (MethodContractError, FileNotFoundError, OSError) as exc:
        raise PredictionEvidenceError(
            f"Decision-bound method lock is invalid: {exc}"
        ) from exc
    if str(verified_method.get("method_lock_sha256", "")).casefold() != (
        expected_method_sha
    ):
        raise PredictionEvidenceError("Decision-bound method lock is missing or changed")
    method_identity = _require_sha256(
        decision.integrity.get("method_identity_sha256"), label="decision method identity"
    )
    if str(verified_method.get("method_identity_sha256", "")).casefold() != (
        method_identity
    ):
        raise PredictionEvidenceError("Method lock identity differs from the decision")
    artifacts = verified_method.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PredictionEvidenceError("Method lock has no artifact bindings")
    noise_binding = artifacts.get("noise_split_lock")
    if not isinstance(noise_binding, Mapping):
        raise PredictionEvidenceError("Method lock has no noise-split binding")
    noise_path = _repo_path(
        noise_binding.get("path", ""), root=root, label="method.noise_split_lock"
    )
    expected_noise_sha = _require_sha256(
        noise_binding.get("sha256"), label="method noise-split lock"
    )
    try:
        noise = verify_noise_split_lock(noise_path, verify_audio=False)
    except (NoiseProtocolError, FileNotFoundError, OSError, ValueError) as exc:
        raise PredictionEvidenceError(f"Noise-split verification failed: {exc}") from exc
    if str(noise.get("lock_sha256", "")).casefold() != expected_noise_sha:
        raise PredictionEvidenceError("Verified noise-split lock differs from method lock")
    return expected_method_sha, expected_noise_sha, noise


def verify_benchmark_evidence(
    benchmark_path: str | Path,
    decision: DecisionEvidence,
    *,
    final_benchmark_lock_path: str | Path | None = None,
    root: str | Path = ROOT,
) -> BenchmarkEvidence:
    repository_root = Path(root).resolve()
    benchmark = Path(benchmark_path)
    benchmark_sha = sha256_file(benchmark)
    row_count = _manifest_row_count(benchmark)
    if final_benchmark_lock_path is None:
        return BenchmarkEvidence(
            path=benchmark,
            sha256=benchmark_sha,
            row_count=row_count,
            final_benchmark_lock_path=None,
            final_benchmark_lock_sha256="",
        )

    final_lock = Path(final_benchmark_lock_path)
    final_lock_sha = sha256_file(final_lock)
    method_sha, noise_sha, noise = _method_and_noise_evidence(
        decision, root=repository_root
    )
    try:
        verified = verify_final_benchmark_lock(
            final_lock,
            expected_lock_sha256=final_lock_sha,
            expected_manifest=benchmark,
            expected_manifest_sha256=benchmark_sha,
            expected_rows=row_count,
            split_lock_sha256=_require_sha256(
                decision.integrity.get("split_lock_sha256"),
                label="decision split lock",
            ),
            decision_lock_sha256=decision.sha256,
            source_test_manifest_sha256=_require_sha256(
                decision.integrity.get("test_manifest_sha256"),
                label="decision source-test manifest",
            ),
            method_lock_sha256=method_sha,
            method_identity_sha256=_require_sha256(
                decision.integrity.get("method_identity_sha256"),
                label="decision method identity",
            ),
            noise_split_lock_sha256=noise_sha,
            noise_integrity=noise,
        )
    except (FinalBenchmarkError, FileNotFoundError, OSError, ValueError) as exc:
        raise PredictionEvidenceError(f"Final benchmark verification failed: {exc}") from exc
    if (
        str(verified.get("manifest_sha256", "")).casefold() != benchmark_sha
        or int(verified.get("row_count", -1)) != row_count
        or str(verified.get("lock_sha256", "")).casefold() != final_lock_sha
    ):
        raise PredictionEvidenceError("Final benchmark verifier returned another artifact")
    return BenchmarkEvidence(
        path=benchmark,
        sha256=benchmark_sha,
        row_count=row_count,
        final_benchmark_lock_path=final_lock,
        final_benchmark_lock_sha256=final_lock_sha,
    )


def _sidecar_path(prediction: Path) -> Path:
    adjacent = prediction.with_suffix(prediction.suffix + ".provenance.json")
    sibling = prediction.parent / "provenance.json"
    found = [path for path in (adjacent, sibling) if path.is_file()]
    if len(found) != 1:
        raise PredictionEvidenceError(
            f"{prediction}: expected exactly one provenance sidecar, found={found}"
        )
    return found[0]


def _verify_actual_file_binding(
    provenance: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    root: Path,
) -> None:
    path = _repo_path(provenance.get(path_field, ""), root=root, label=path_field)
    expected = _require_sha256(provenance.get(hash_field), label=hash_field)
    if not path.is_file() or sha256_file(path) != expected:
        raise PredictionEvidenceError(f"{path_field}/{hash_field} binding changed")


def _verify_zero_shot_contract(
    provenance: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
    decision: DecisionEvidence,
    benchmark: BenchmarkEvidence,
    root: Path,
) -> None:
    contract = provenance.get("run_contract")
    if not isinstance(contract, Mapping):
        raise PredictionEvidenceError("Zero-shot provenance has no run contract")
    recorded_hash = _require_sha256(
        provenance.get("run_contract_sha256"), label="zero-shot run contract"
    )
    if _canonical_json_sha256(contract) != recorded_hash:
        raise PredictionEvidenceError("Zero-shot run contract SHA-256 is invalid")
    if contract.get("contract_version") != "paper_v2_zero_shot_run_v1" or (
        tuple(contract.get("schema", ())) != tuple(CANONICAL_PREDICTION_COLUMNS)
    ):
        raise PredictionEvidenceError("Zero-shot run contract version/schema is invalid")
    if str(contract.get("suite_config_sha256", "")).casefold() != str(
        provenance.get("suite_config_sha256", "")
    ).casefold():
        raise PredictionEvidenceError("Zero-shot run contract binds another suite config")
    try:
        contract_seed = str(int(contract.get("seed", -1)))
    except (TypeError, ValueError) as exc:
        raise PredictionEvidenceError("Zero-shot run contract seed is invalid") from exc
    if contract_seed != metadata["seed"] or contract_seed != str(provenance.get("seed", "")):
        raise PredictionEvidenceError("Zero-shot seed differs across CSV/provenance/contract")
    model = contract.get("model")
    if not isinstance(model, Mapping):
        raise PredictionEvidenceError("Zero-shot run contract has no model object")
    expected_model = {
        "key": str(provenance.get("model_key", "")),
        "repo_id": str(provenance.get("model_repo_id", "")),
        "revision": str(provenance.get("model_revision", "")),
        "model": metadata["model"],
        "model_size": metadata["model_size"],
    }
    for field, expected in expected_model.items():
        if not expected or str(model.get(field, "")) != expected:
            raise PredictionEvidenceError(
                f"Zero-shot run contract model.{field} differs from provenance/CSV"
            )
    protocol = contract.get("protocol")
    benchmark_contract = contract.get("benchmark")
    if not isinstance(protocol, Mapping) or not isinstance(benchmark_contract, Mapping):
        raise PredictionEvidenceError("Zero-shot run contract lacks protocol/benchmark")
    if (
        protocol.get("formal") is not True
        or protocol.get("final_test_unlocked") is not True
        or str(protocol.get("expected_split_lock_sha256", "")).casefold()
        != str(decision.integrity["split_lock_sha256"]).casefold()
        or str(protocol.get("expected_decision_lock_sha256", "")).casefold()
        != decision.sha256
    ):
        raise PredictionEvidenceError("Zero-shot run contract protocol binding is stale")
    if (
        str(benchmark_contract.get("expected_manifest_sha256", "")).casefold()
        != benchmark.sha256
        or str(benchmark_contract.get("expected_lock_sha256", "")).casefold()
        != benchmark.final_benchmark_lock_sha256
        or int(benchmark_contract.get("expected_rows", -1)) != benchmark.row_count
        or str(benchmark_contract.get("dataset", "")).casefold()
        != metadata["dataset"].casefold()
        or benchmark_contract.get("verify_audio_sha256") is not True
    ):
        raise PredictionEvidenceError("Zero-shot run contract benchmark binding is stale")

    config_path = _repo_path(
        provenance.get("suite_config", ""), root=root, label="suite_config"
    )
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PredictionEvidenceError(f"Cannot read zero-shot suite config: {exc}") from exc
    if not isinstance(config, Mapping):
        raise PredictionEvidenceError("Zero-shot suite config must be a mapping")
    models = config.get("models")
    configured_model = (
        models.get(expected_model["key"])
        if isinstance(models, Mapping)
        else None
    )
    if not isinstance(configured_model, Mapping):
        raise PredictionEvidenceError("Zero-shot model key is absent from suite config")
    for field in ("repo_id", "revision", "model", "model_size"):
        if str(configured_model.get(field, "")) != expected_model[field]:
            raise PredictionEvidenceError(
                f"Zero-shot suite config model.{field} differs from run contract"
            )
    if str(int(config.get("seed", -1))) != contract_seed:
        raise PredictionEvidenceError("Zero-shot suite config seed differs from run contract")
    config_protocol = config.get("protocol")
    config_benchmark = config.get("benchmark")
    if not isinstance(config_protocol, Mapping) or not isinstance(
        config_benchmark, Mapping
    ):
        raise PredictionEvidenceError("Zero-shot suite config lacks protocol/benchmark")
    for field in (
        "expected_split_lock_sha256",
        "expected_decision_lock_sha256",
    ):
        if str(config_protocol.get(field, "")).casefold() != str(
            protocol.get(field, "")
        ).casefold():
            raise PredictionEvidenceError(f"Zero-shot suite config {field} is stale")
    for field in ("expected_lock_sha256", "expected_manifest_sha256", "expected_rows"):
        if str(config_benchmark.get(field, "")).casefold() != str(
            benchmark_contract.get(field, "")
        ).casefold():
            raise PredictionEvidenceError(f"Zero-shot suite config {field} is stale")


def _default_fleurs_authorizer(registry_reference: object) -> Any:
    # Import lazily to keep ordinary analysis imports light and to avoid making
    # the script module part of every legacy invocation.
    from scripts.run_external_fleurs import (
        ExternalFleursError,
        authorize_external_suite,
    )

    def decision_without_checkpoint_io(**kwargs: Any) -> Mapping[str, Any]:
        return verify_test_decision_lock(**kwargs, verify_checkpoints=False)

    try:
        return authorize_external_suite(
            str(registry_reference),
            formal=True,
            verify_current_method=False,
            decision_verifier=decision_without_checkpoint_io,
        )
    except (ExternalFleursError, FileNotFoundError, OSError, ValueError) as exc:
        raise PredictionEvidenceError(
            f"FLEURS registry/preparation verification failed: {exc}"
        ) from exc


def _external_value(authorization: Any, field: str) -> Any:
    if isinstance(authorization, Mapping):
        return authorization.get(field)
    return getattr(authorization, field, None)


def _verify_fleurs_authorization(
    provenance: Mapping[str, Any],
    *,
    authorization: Any,
    decision: DecisionEvidence,
    benchmark: BenchmarkEvidence,
    root: Path,
) -> None:
    registry_path = _repo_path(
        provenance.get("registry", ""), root=root, label="registry"
    )
    manifest_path = Path(_external_value(authorization, "manifest_path") or "")
    authorized_registry = Path(
        _external_value(authorization, "registry_path") or ""
    )
    if (
        _external_value(authorization, "formal") is not True
        or authorized_registry.resolve() != registry_path.resolve()
        or manifest_path.resolve() != benchmark.path.resolve()
        or str(_external_value(authorization, "registry_sha256") or "").casefold()
        != sha256_file(registry_path)
        or str(_external_value(authorization, "manifest_sha256") or "").casefold()
        != benchmark.sha256
        or int(_external_value(authorization, "expected_rows") or -1)
        != benchmark.row_count
        or str(_external_value(authorization, "split_lock_sha256") or "").casefold()
        != str(decision.integrity["split_lock_sha256"]).casefold()
        or str(_external_value(authorization, "decision_lock_sha256") or "").casefold()
        != decision.sha256
        or str(_external_value(authorization, "method_lock_sha256") or "").casefold()
        != str(decision.integrity["method_lock_sha256"]).casefold()
        or str(_external_value(authorization, "method_identity_sha256") or "").casefold()
        != str(decision.integrity["method_identity_sha256"]).casefold()
        or _external_value(authorization, "method_runtime_verified") is not False
    ):
        raise PredictionEvidenceError(
            "FLEURS authorization differs from decision/manifest/registry evidence"
        )
    method_bindings = {
        "method_environment_identity_sha256": "method_environment_identity_sha256",
        "method_source_tree_sha256": "method_source_tree_sha256",
    }
    if provenance.get("method_runtime_verified") is not True:
        raise PredictionEvidenceError(
            "FLEURS inference provenance did not verify the formal method runtime"
        )
    for provenance_field, authorization_field in method_bindings.items():
        observed = _require_sha256(
            provenance.get(provenance_field), label=provenance_field
        )
        expected = _require_sha256(
            _external_value(authorization, authorization_field),
            label=f"authorization.{authorization_field}",
        )
        if observed != expected:
            raise PredictionEvidenceError(
                f"FLEURS provenance differs from method evidence: {provenance_field}"
            )
    sidecar_bindings = {
        "fleurs_preparation_lock_sha256": "fleurs_preparation_lock_sha256",
        "fleurs_preparation_identity_sha256": "fleurs_preparation_identity_sha256",
        "fleurs_dataset_revision": "fleurs_dataset_revision",
        "fleurs_audio_inventory_sha256": "fleurs_audio_inventory_sha256",
        "fleurs_audit_sha256": "fleurs_audit_sha256",
    }
    for provenance_field, authorization_field in sidecar_bindings.items():
        observed = str(provenance.get(provenance_field, "")).casefold()
        expected = str(
            _external_value(authorization, authorization_field) or ""
        ).casefold()
        if not observed or observed != expected:
            raise PredictionEvidenceError(
                f"FLEURS provenance differs from preparation evidence: {provenance_field}"
            )
    registry = _external_value(authorization, "registry")
    if not isinstance(registry, Mapping) or not isinstance(registry.get("runs"), list):
        raise PredictionEvidenceError("FLEURS authorization has no registry runs")
    configuration_id = str(provenance.get("configuration_id", ""))
    registered = [
        item
        for item in registry["runs"]
        if isinstance(item, Mapping)
        and str(item.get("configuration_id", "")) == configuration_id
    ]
    if len(registered) != 1:
        raise PredictionEvidenceError("FLEURS provenance is absent/ambiguous in registry")
    registered_run = registered[0]
    for provenance_field, registry_field in (
        ("config", "config_path"),
        ("config_sha256", "config_sha256"),
        ("checkpoint", "checkpoint_path"),
    ):
        if str(provenance.get(provenance_field, "")).casefold() != str(
            registered_run.get(registry_field, "")
        ).casefold():
            raise PredictionEvidenceError(
                f"FLEURS provenance {provenance_field} differs from registry"
            )
    locked = decision.configurations.get(configuration_id)
    if locked is None or str(provenance.get("checkpoint", "")).casefold() != str(
        locked.get("checkpoint_path", "")
    ).casefold():
        raise PredictionEvidenceError("FLEURS checkpoint path differs from decision")


def _verify_fleurs_run_contract(
    provenance: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
    decision: DecisionEvidence,
    benchmark: BenchmarkEvidence,
    authorization: Any,
    row_count: int,
) -> None:
    contract = provenance.get("run_contract")
    if not isinstance(contract, Mapping) or _canonical_json_sha256(contract) != (
        _require_sha256(
            provenance.get("run_contract_sha256"), label="FLEURS run contract"
        )
    ):
        raise PredictionEvidenceError("FLEURS run contract SHA-256 is invalid")
    if (
        contract.get("contract_version")
        != "paper_v2_fleurs_inference_contract_v1"
        or contract.get("evaluation_domain")
        != "legacy_exposed_external_replication"
        or provenance.get("evaluation_domain")
        != "legacy_exposed_external_replication"
    ):
        raise PredictionEvidenceError("FLEURS run contract domain/version is invalid")
    identity_fields = {
        "registry_sha256": _external_value(authorization, "registry_sha256"),
        "split_lock_sha256": decision.integrity["split_lock_sha256"],
        "decision_lock_sha256": decision.sha256,
        "method_lock_sha256": decision.integrity["method_lock_sha256"],
        "method_identity_sha256": decision.integrity["method_identity_sha256"],
        "method_environment_identity_sha256": _external_value(
            authorization, "method_environment_identity_sha256"
        ),
        "method_source_tree_sha256": _external_value(
            authorization, "method_source_tree_sha256"
        ),
        "manifest_sha256": benchmark.sha256,
        "fleurs_preparation_lock_sha256": _external_value(
            authorization, "fleurs_preparation_lock_sha256"
        ),
        "fleurs_preparation_identity_sha256": _external_value(
            authorization, "fleurs_preparation_identity_sha256"
        ),
        "fleurs_dataset_revision": _external_value(
            authorization, "fleurs_dataset_revision"
        ),
        "fleurs_audio_inventory_sha256": _external_value(
            authorization, "fleurs_audio_inventory_sha256"
        ),
        "fleurs_audit_sha256": _external_value(
            authorization, "fleurs_audit_sha256"
        ),
    }
    for field, expected_value in identity_fields.items():
        expected = str(expected_value or "").casefold()
        if (
            not expected
            or str(contract.get(field, "")).casefold() != expected
            or str(provenance.get(field, "")).casefold() != expected
        ):
            raise PredictionEvidenceError(
                f"FLEURS run contract {field} differs from formal authorization"
            )
    if contract.get("method_runtime_verified") is not True or (
        provenance.get("method_runtime_verified") is not True
    ):
        raise PredictionEvidenceError(
            "FLEURS inference did not lock a verified formal method runtime"
        )
    try:
        selected_rows = int(contract.get("selected_rows", -1))
    except (TypeError, ValueError) as exc:
        raise PredictionEvidenceError("FLEURS selected_rows is invalid") from exc
    if selected_rows != row_count or selected_rows != benchmark.row_count or not _is_sha256(
        contract.get("selected_rows_sha256")
    ):
        raise PredictionEvidenceError("FLEURS selected-row identity/count is invalid")
    expected = {
        "configuration_id": str(provenance.get("configuration_id", "")),
        "role": str(provenance.get("role", "")),
        "method_id": str(provenance.get("method_id", "")),
        "train_type": metadata["train_type"],
        "lambda": _canonical_number(metadata["lambda"], label="FLEURS CSV lambda"),
        "seed": str(int(metadata["seed"])),
        "checkpoint_sha256": str(provenance.get("checkpoint_sha256", "")).casefold(),
        "resolved_config_sha256": str(
            provenance.get("resolved_config_sha256", "")
        ).casefold(),
        "training_contract_sha256": str(
            provenance.get("training_contract_sha256", "")
        ).casefold(),
        "config_sha256": str(provenance.get("config_sha256", "")).casefold(),
        "backbone": str(provenance.get("backbone", "")),
        "backbone_revision": str(provenance.get("backbone_revision", "")).casefold(),
    }
    for field, expected_value in expected.items():
        observed: object = contract.get(field, "")
        if field == "lambda":
            observed = _canonical_number(observed, label="FLEURS contract lambda")
        elif field == "seed":
            observed = str(int(observed))
        elif field.endswith("sha256"):
            observed = str(observed).casefold()
        else:
            observed = str(observed)
        if observed != expected_value:
            raise PredictionEvidenceError(
                f"FLEURS run contract {field} differs from provenance/CSV"
            )
    if tuple(contract.get("prediction_schema", ())) != tuple(
        CANONICAL_PREDICTION_COLUMNS
    ):
        raise PredictionEvidenceError("FLEURS run contract schema is invalid")
    contract_decoding = contract.get("decoding")
    provenance_decoding = provenance.get("decoding")
    if not isinstance(contract_decoding, Mapping) or not isinstance(
        provenance_decoding, Mapping
    ) or any(
        provenance_decoding.get(field) != value
        for field, value in contract_decoding.items()
    ):
        raise PredictionEvidenceError("FLEURS decoding differs from run contract")


def _expected_configuration(
    provenance: Mapping[str, Any],
    decision: DecisionEvidence,
    *,
    required_configuration_id: str | None,
) -> Mapping[str, Any]:
    configuration_id = str(provenance.get("configuration_id", "")).strip()
    if required_configuration_id is not None and configuration_id != required_configuration_id:
        raise PredictionEvidenceError(
            f"Prediction provenance configuration_id={configuration_id!r} differs from "
            f"the requested {required_configuration_id!r}"
        )
    configuration = decision.configurations.get(configuration_id)
    if configuration is None:
        raise PredictionEvidenceError(
            f"Prediction uses a configuration absent from the decision: {configuration_id!r}"
        )
    expected = {
        "configuration_id": configuration_id,
        "role": str(configuration.get("role", "")),
        "method_id": str(configuration.get("method_id", "")),
        "train_type": str(configuration.get("train_type", "")),
        "lambda": _canonical_number(configuration.get("lambda"), label="decision lambda"),
        "seed": str(int(configuration.get("seed", -1))),
        "checkpoint_sha256": _require_sha256(
            configuration.get("checkpoint_sha256"), label="decision checkpoint"
        ),
        "resolved_config_sha256": _require_sha256(
            configuration.get("resolved_config_sha256"), label="decision resolved config"
        ),
        "training_contract_sha256": _require_sha256(
            configuration.get("training_contract_sha256"), label="decision training contract"
        ),
    }
    for field, value in expected.items():
        observed = provenance.get(field, "")
        if field == "lambda":
            observed = _canonical_number(observed, label="provenance lambda")
        elif field == "seed":
            try:
                observed = str(int(observed))
            except (TypeError, ValueError) as exc:
                raise PredictionEvidenceError("Provenance seed is invalid") from exc
        else:
            observed = str(observed).casefold() if field.endswith("sha256") else str(observed)
        comparison = value.casefold() if field.endswith("sha256") else value
        if observed != comparison:
            raise PredictionEvidenceError(
                f"Prediction provenance {field} differs from the locked configuration"
            )
    return expected


def verify_prediction_evidence(
    prediction_path: str | Path,
    *,
    decision: DecisionEvidence,
    benchmark: BenchmarkEvidence,
    required_configuration_id: str | None = None,
    root: str | Path = ROOT,
    fleurs_authorizer: Callable[[object], Any] = _default_fleurs_authorizer,
) -> PredictionEvidence:
    repository_root = Path(root).resolve()
    prediction = Path(prediction_path)
    prediction_hash_before = sha256_file(prediction)
    row_count, metadata = _read_csv_identity(prediction)
    sidecar = _sidecar_path(prediction)
    provenance = _load_json(sidecar, label="prediction provenance")
    version = str(provenance.get("provenance_version", ""))
    if version not in FORMAL_VERSIONS:
        raise PredictionEvidenceError(
            f"Unsupported formal prediction provenance version: {version!r}"
        )
    if _require_sha256(provenance.get("prediction_sha256"), label="prediction_sha256") != prediction_hash_before:
        raise PredictionEvidenceError("Prediction SHA-256 differs from its provenance")
    try:
        recorded_rows = int(provenance.get("num_rows", -1))
    except (TypeError, ValueError) as exc:
        raise PredictionEvidenceError("Prediction provenance row count is invalid") from exc
    if recorded_rows != row_count:
        raise PredictionEvidenceError("Prediction row count differs from its provenance")
    if str(provenance.get("metric_version", METRIC_VERSION)) != METRIC_VERSION:
        raise PredictionEvidenceError("Prediction provenance metric version is invalid")
    manifest_hash_field = (
        "final_manifest_sha256"
        if version == FINAL_LORA_VERSION
        else "manifest_sha256"
    )
    if _require_sha256(
        provenance.get(manifest_hash_field), label=manifest_hash_field
    ) != benchmark.sha256:
        raise PredictionEvidenceError("Prediction provenance binds another benchmark manifest")
    if row_count != benchmark.row_count:
        raise PredictionEvidenceError("Prediction row count differs from benchmark row count")

    schema = provenance.get("schema", provenance.get("prediction_columns"))
    if schema is not None and tuple(schema) != tuple(CANONICAL_PREDICTION_COLUMNS):
        raise PredictionEvidenceError("Prediction provenance schema is not canonical")
    split_sha = _require_sha256(
        decision.integrity.get("split_lock_sha256"), label="verified split lock"
    )
    for field, expected in (
        ("decision_lock_sha256", decision.sha256),
        ("split_lock_sha256", split_sha),
        (
            "method_lock_sha256",
            _require_sha256(
                decision.integrity.get("method_lock_sha256"),
                label="verified method lock",
            ),
        ),
        (
            "method_identity_sha256",
            _require_sha256(
                decision.integrity.get("method_identity_sha256"),
                label="verified method identity",
            ),
        ),
    ):
        # Zero-shot provenance binds method transitively through the verified
        # decision and therefore intentionally has no duplicated method fields.
        if (
            field == "decision_lock_sha256"
            and version in NOISY_DEV_VERSIONS
            and not str(provenance.get(field, "")).strip()
        ):
            continue
        if field in provenance and str(provenance[field]).casefold() != expected:
            raise PredictionEvidenceError(f"Prediction provenance binds another {field}")
        required_fields = {"split_lock_sha256"}
        if version not in NOISY_DEV_VERSIONS:
            required_fields.add("decision_lock_sha256")
        if field in required_fields and field not in provenance:
            raise PredictionEvidenceError(f"Prediction provenance is missing {field}")

    if version in {FINAL_LORA_VERSION, ZERO_SHOT_VERSION}:
        if benchmark.final_benchmark_lock_path is None:
            raise PredictionEvidenceError(
                "Final-test prediction verification requires --final-benchmark-lock"
            )
        lock_field = (
            "final_benchmark_lock_sha256"
            if version == FINAL_LORA_VERSION
            else "benchmark_lock_sha256"
        )
        if str(provenance.get(lock_field, "")).casefold() != benchmark.final_benchmark_lock_sha256:
            raise PredictionEvidenceError("Prediction binds another final benchmark lock")

    configuration_id = ""
    role = ""
    if version in {FINAL_LORA_VERSION, FLEURS_VERSION}:
        expected = _expected_configuration(
            provenance,
            decision,
            required_configuration_id=required_configuration_id,
        )
        configuration_id = str(expected["configuration_id"])
        role = str(expected["role"])
        for field in ("train_type", "lambda", "seed"):
            observed = metadata[field]
            wanted = str(expected[field])
            if field == "lambda":
                observed = _canonical_number(observed, label="prediction lambda")
            elif field == "seed":
                observed = str(int(observed))
            if observed != wanted:
                raise PredictionEvidenceError(
                    f"Prediction CSV {field} differs from the locked configuration"
                )
        if version == FINAL_LORA_VERSION:
            _verify_actual_file_binding(
                provenance,
                path_field="runtime_config",
                hash_field="runtime_config_sha256",
                root=repository_root,
            )
        else:
            _verify_actual_file_binding(
                provenance,
                path_field="config",
                hash_field="config_sha256",
                root=repository_root,
            )
            _verify_actual_file_binding(
                provenance,
                path_field="registry",
                hash_field="registry_sha256",
                root=repository_root,
            )
            fleurs_authorization = fleurs_authorizer(
                provenance.get("registry", "")
            )
            _verify_fleurs_authorization(
                provenance,
                authorization=fleurs_authorization,
                decision=decision,
                benchmark=benchmark,
                root=repository_root,
            )
            _verify_fleurs_run_contract(
                provenance,
                metadata=metadata,
                decision=decision,
                benchmark=benchmark,
                authorization=fleurs_authorization,
                row_count=row_count,
            )
    elif version == ZERO_SHOT_VERSION:
        if required_configuration_id is not None:
            raise PredictionEvidenceError("Zero-shot prediction has no LoRA configuration ID")
        if metadata["train_type"] != "zero_shot" or metadata["lambda"] != "":
            raise PredictionEvidenceError("Zero-shot CSV has invalid train_type/lambda")
        _verify_actual_file_binding(
            provenance,
            path_field="suite_config",
            hash_field="suite_config_sha256",
            root=repository_root,
        )
        _verify_zero_shot_contract(
            provenance,
            metadata=metadata,
            decision=decision,
            benchmark=benchmark,
            root=repository_root,
        )
    else:
        if str(provenance.get("training_scope", "")) != "formal":
            raise PredictionEvidenceError("Noisy-dev provenance is not a formal run")
        _verify_actual_file_binding(
            provenance,
            path_field="config_path",
            hash_field="config_file_sha256",
            root=repository_root,
        )

    prediction_hash_after = sha256_file(prediction)
    if prediction_hash_after != prediction_hash_before:
        raise PredictionEvidenceError("Prediction changed during formal verification")
    return PredictionEvidence(
        prediction_path=prediction,
        prediction_sha256=prediction_hash_after,
        provenance_path=sidecar,
        provenance_sha256=sha256_file(sidecar),
        provenance_version=version,
        row_count=row_count,
        configuration_id=configuration_id,
        role=role,
    )


def verify_formal_prediction_set(
    prediction_paths: Sequence[str | Path],
    *,
    benchmark_path: str | Path,
    split_lock_path: str | Path,
    decision_path: str | Path,
    final_benchmark_lock_path: str | Path | None = None,
    required_configuration_ids: Mapping[str | Path, str] | None = None,
    root: str | Path = ROOT,
    decision_verifier: Callable[..., Mapping[str, Any]] = verify_test_decision_lock,
    fleurs_authorizer: Callable[[object], Any] = _default_fleurs_authorizer,
    method_evidence_verifier: Callable[..., Any] = _method_and_noise_evidence,
) -> FormalPredictionSet:
    if not prediction_paths:
        raise PredictionEvidenceError("Formal prediction set is empty")
    decision = verify_decision_evidence(
        decision_path, split_lock_path, decision_verifier=decision_verifier
    )
    method_evidence_verifier(decision, root=Path(root).resolve())
    benchmark = verify_benchmark_evidence(
        benchmark_path,
        decision,
        final_benchmark_lock_path=final_benchmark_lock_path,
        root=root,
    )
    required = {
        Path(path).resolve(): configuration_id
        for path, configuration_id in (required_configuration_ids or {}).items()
    }
    evidence = tuple(
        verify_prediction_evidence(
            path,
            decision=decision,
            benchmark=benchmark,
            required_configuration_id=required.get(Path(path).resolve()),
            root=root,
            fleurs_authorizer=fleurs_authorizer,
        )
        for path in prediction_paths
    )
    return FormalPredictionSet(
        decision=decision,
        benchmark=benchmark,
        predictions=evidence,
    )


def formal_protocol_parameters(evidence: FormalPredictionSet) -> dict[str, Any]:
    """Return path-portable identities to embed in an analysis provenance."""

    return {
        "formal_paper_v2": True,
        "decision_lock_sha256": evidence.decision.sha256,
        "split_lock_sha256": str(
            evidence.decision.integrity["split_lock_sha256"]
        ).casefold(),
        "method_lock_sha256": str(
            evidence.decision.integrity["method_lock_sha256"]
        ).casefold(),
        "method_identity_sha256": str(
            evidence.decision.integrity["method_identity_sha256"]
        ).casefold(),
        "benchmark_manifest_sha256": evidence.benchmark.sha256,
        "benchmark_rows": evidence.benchmark.row_count,
        "final_benchmark_lock_sha256": evidence.benchmark.final_benchmark_lock_sha256,
        "predictions": [
            {
                "prediction_sha256": item.prediction_sha256,
                "provenance_sha256": item.provenance_sha256,
                "provenance_version": item.provenance_version,
                "configuration_id": item.configuration_id,
                "role": item.role,
                "rows": item.row_count,
            }
            for item in evidence.predictions
        ],
    }


def verify_formal_error_events(
    event_path: str | Path,
    *,
    benchmark_path: str | Path,
    split_lock_path: str | Path,
    decision_path: str | Path,
    final_benchmark_lock_path: str | Path | None = None,
    root: str | Path = ROOT,
    decision_verifier: Callable[..., Mapping[str, Any]] = verify_test_decision_lock,
    fleurs_authorizer: Callable[[object], Any] = _default_fleurs_authorizer,
    method_evidence_verifier: Callable[..., Any] = _method_and_noise_evidence,
) -> FormalPredictionSet:
    """Verify error-events and recursively re-verify all source predictions."""

    event = Path(event_path)
    provenance_path = event.parent / "error_analysis.provenance.json"
    provenance = _load_json(provenance_path, label="error-analysis provenance")
    if provenance.get("provenance_version") != "analysis_artifact_provenance_v1" or (
        provenance.get("bundle_name") != "error_analysis"
    ):
        raise PredictionEvidenceError("Events are not from the formal error-analysis bundle")
    marker_path = event.parent / "error_analysis.bundle.commit.json"
    marker = _load_json(marker_path, label="error-analysis commit marker")
    if (
        marker.get("status") != "COMMITTED"
        or marker.get("protocol_version") != BUNDLE_PROTOCOL_VERSION
        or marker.get("bundle_name") != "error_analysis"
        or marker.get("bundle_version") != provenance.get("bundle_version")
        or marker.get("inputs") != provenance.get("inputs")
    ):
        raise PredictionEvidenceError("Error-analysis commit marker is invalid")
    marker_outputs = marker.get("outputs")
    if not isinstance(marker_outputs, list) or {
        str(item.get("key", ""))
        for item in marker_outputs
        if isinstance(item, Mapping)
    } != {"events", "summary", "provenance"}:
        raise PredictionEvidenceError("Error-analysis commit marker outputs are invalid")
    for item in marker_outputs:
        if not isinstance(item, Mapping):
            raise PredictionEvidenceError("Error-analysis commit marker output is malformed")
        output = _repo_path(
            item.get("path", ""), root=event.parent, label="error bundle output"
        )
        try:
            recorded_bytes = int(item.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise PredictionEvidenceError(
                "Error-analysis marker output size is invalid"
            ) from exc
        if (
            not output.is_file()
            or output.stat().st_size != recorded_bytes
            or sha256_file(output)
            != _require_sha256(item.get("sha256"), label="error output SHA-256")
        ):
            raise PredictionEvidenceError("Error-analysis committed output changed")
    marker_identity = {
        field: marker[field]
        for field in (
            "protocol_version",
            "bundle_name",
            "bundle_version",
            "inputs",
            "outputs",
        )
    }
    expected_bundle_sha = hashlib.sha256(
        canonical_json_bytes(marker_identity)
    ).hexdigest()
    if str(marker.get("bundle_sha256", "")).casefold() != expected_bundle_sha:
        raise PredictionEvidenceError("Error-analysis commit marker identity is invalid")
    output_specs = provenance.get("data_outputs")
    if not isinstance(output_specs, list):
        raise PredictionEvidenceError("Error-analysis provenance has no data outputs")
    matching = [
        item
        for item in output_specs
        if isinstance(item, Mapping) and str(item.get("path", "")) == event.name
    ]
    if len(matching) != 1 or str(matching[0].get("sha256", "")).casefold() != sha256_file(event):
        raise PredictionEvidenceError("Error-events CSV differs from bundle provenance")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise PredictionEvidenceError("Error-analysis provenance has no prediction inputs")
    prediction_paths: list[Path] = []
    for item in inputs:
        if not isinstance(item, Mapping):
            raise PredictionEvidenceError("Error-analysis input binding is malformed")
        path = _repo_path(item.get("path", ""), root=Path(root), label="error input")
        expected = _require_sha256(item.get("sha256"), label="error input SHA-256")
        if not path.is_file() or sha256_file(path) != expected:
            raise PredictionEvidenceError("Error-analysis prediction input changed")
        prediction_paths.append(path)
    verified = verify_formal_prediction_set(
        prediction_paths,
        benchmark_path=benchmark_path,
        split_lock_path=split_lock_path,
        decision_path=decision_path,
        final_benchmark_lock_path=final_benchmark_lock_path,
        root=root,
        decision_verifier=decision_verifier,
        fleurs_authorizer=fleurs_authorizer,
        method_evidence_verifier=method_evidence_verifier,
    )
    expected_parameters = formal_protocol_parameters(verified)
    parameters = provenance.get("parameters")
    if not isinstance(parameters, Mapping) or parameters.get("formal_protocol") != expected_parameters:
        raise PredictionEvidenceError(
            "Error-analysis provenance does not match re-verified formal inputs"
        )
    return verified


__all__ = [
    "BenchmarkEvidence",
    "DecisionEvidence",
    "FINAL_LORA_VERSION",
    "FLEURS_VERSION",
    "FormalPredictionSet",
    "PredictionEvidence",
    "PredictionEvidenceError",
    "ZERO_SHOT_VERSION",
    "formal_protocol_parameters",
    "sha256_file",
    "verify_benchmark_evidence",
    "verify_decision_evidence",
    "verify_formal_error_events",
    "verify_formal_prediction_set",
    "verify_prediction_evidence",
]
