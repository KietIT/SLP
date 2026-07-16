"""Run the three decision-locked PhoWhisper LoRA methods on FLEURS.

FLEURS was already inspected by the first experiment cycle, so this runner
records it as a ``legacy_exposed_external_replication``.  It must never be
described as an untouched final test.  The evaluated configurations are not
hard-coded: an immutable method/lambda decision and an explicit run registry
resolve exactly one ordinary baseline, selected method, and locked control.

Audio is never truncated.  Utterances longer than Whisper's 30 second input
window are split into consecutive, non-overlapping chunks and decoded text is
concatenated in temporal order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.analysis import (  # noqa: E402
    CANONICAL_PREDICTION_COLUMNS,
    METRIC_VERSION,
    compute_aligned_metric_result,
    load_prediction_csv,
    validate_prediction_rows,
)
from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.phat.method_contract import (  # noqa: E402
    MethodContractError,
    verify_method_lock,
)
from src.vitonesr.phat.protocol import (  # noqa: E402
    canonical_sha256,
    is_immutable_revision,
    is_sha256,
    resolve_locked_roles,
    sha256_file,
    verify_checkpoint_config,
    verify_test_configuration_locked,
    verify_test_decision_lock,
)
from src.vitonesr.prediction import atomic_write_csv  # noqa: E402
from scripts.download_fleurs import (  # noqa: E402
    DATASET_NAME as FLEURS_DATASET_REPOSITORY,
    LANGUAGE_CONFIG as FLEURS_DATASET_CONFIG,
    MANIFEST_FIELDS as FLEURS_MANIFEST_FIELDS,
    PREPARATION_LOCK_VERSION,
    FleursPreparationError,
    verify_fleurs_preparation_lock,
)


DEFAULT_RUN_REGISTRY = Path("outputs/paper_v2/protocol/fleurs_run_registry.json")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_v2/external/fleurs")
DEFAULT_EXPECTED_ROWS = 857
SAMPLE_RATE = 16_000
MAX_CHUNK_SECONDS = 30.0
REGISTRY_VERSION = "paper_v2_fleurs_run_registry_v3"
PROVENANCE_VERSION = "paper_v2_fleurs_prediction_v3"
RESUME_VERSION = "paper_v2_fleurs_resume_v2"
RECOVERY_VERSION = "paper_v2_fleurs_recovery_v1"
RESULT_PROVENANCE_VERSION = "paper_v2_fleurs_results_v4"
EVALUATION_DOMAIN = "legacy_exposed_external_replication"
FORMAL_PATH_MODE = "repository_relative_v1"
DIAGNOSTIC_PATH_MODE = "diagnostic_legacy_paths_v1"
REQUIRED_ROLES = (
    "ordinary_baseline",
    "selected_method",
    "locked_control",
)

METRIC_NAMES = ("wer", "cer", "ter", "der", "fcer", "swdr")
METRIC_COUNT_COLUMNS = tuple(
    item
    for metric in METRIC_NAMES
    for item in (f"{metric}_numerator", f"{metric}_denominator")
)
METRIC_COVERAGE_COLUMNS = ("ter_coverage", "der_coverage", "fcer_coverage")
METRIC_EVIDENCE_COLUMNS = (*METRIC_COUNT_COLUMNS, *METRIC_COVERAGE_COLUMNS)

RESULT_COLUMNS = (
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "n",
    "wer",
    "cer",
    "ter",
    "der",
    "fcer",
    "swdr",
    "metric_version",
    *METRIC_EVIDENCE_COLUMNS,
)


class ExternalFleursError(ValueError):
    """Raised when the external-test contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class ExternalRun:
    configuration_id: str
    role: str
    method_id: str
    train_type: str
    lambda_value: str
    seed: str
    model_name_or_path: str
    backbone_revision: str
    language: str
    task: str
    checkpoint: Path
    checkpoint_sha256: str
    resolved_config_sha256: str
    training_contract_sha256: str
    config_path: Path
    config_sha256: str
    prediction_name: str

    @property
    def run_metadata(self) -> dict[str, str]:
        return {
            "dataset": "fleurs",
            "model": "phowhisper",
            "model_size": "base",
            "train_type": self.train_type,
            "lambda": self.lambda_value,
            "seed": self.seed,
        }


class ChunkTranscriber(Protocol):
    def transcribe_chunk(self, waveform: Any) -> str:
        """Transcribe one waveform whose duration is at most 30 seconds."""

    def close(self) -> None:
        """Release model resources."""


TranscriberFactory = Callable[[ExternalRun, str, int], ChunkTranscriber]
AudioLoader = Callable[[str, int], Any]
DecisionVerifier = Callable[..., Mapping[str, Any]]
CheckpointVerifier = Callable[[str | Path, Mapping[str, Any]], Mapping[str, str]]
PreparationVerifier = Callable[..., Mapping[str, Any]]
MethodConfigLoader = Callable[[str | Path], Mapping[str, Any]]
MethodVerifier = Callable[..., Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class ExternalAuthorization:
    registry_path: Path
    registry_sha256: str
    registry: Mapping[str, Any]
    split_lock_sha256: str
    decision_lock_sha256: str
    method_lock_sha256: str
    method_identity_sha256: str
    manifest_path: Path
    manifest_sha256: str
    expected_rows: int
    locked_by_role: Mapping[str, Mapping[str, Any]]
    method_environment_identity_sha256: str = ""
    method_source_tree_sha256: str = ""
    method_runtime_verified: bool = False
    fleurs_preparation_lock_path: Path | None = None
    fleurs_preparation_lock_sha256: str = ""
    fleurs_preparation_identity_sha256: str = ""
    fleurs_dataset_revision: str = ""
    fleurs_audio_inventory_sha256: str = ""
    fleurs_audit_sha256: str = ""
    formal: bool = False


def _canonical_lambda(value: object) -> str:
    try:
        number = float(str(value))
    except ValueError as error:
        raise ExternalFleursError(f"Invalid lambda value: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ExternalFleursError(f"Invalid lambda value: {value!r}")
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalFleursError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ExternalFleursError(f"{label} must be a JSON object: {path}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise ExternalFleursError(f"{label} must be a concrete SHA-256")
    return str(value).strip().casefold()


def _portable_repo_reference(
    value: object,
    *,
    label: str,
    repository_root: str | Path = ROOT,
) -> str:
    """Return one normalized repository-relative POSIX artifact reference.

    Formal artifacts intentionally reject absolute paths instead of silently
    relativizing them.  That makes it impossible to publish a registry which
    embeds a developer's drive letter, home directory, UNC share, or platform
    separator by accident.
    """

    raw = str(value).strip()
    if not raw:
        raise ExternalFleursError(f"{label} must be a non-empty path")
    windows = PureWindowsPath(raw)
    posix = PurePosixPath(raw)
    if (
        Path(raw).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or raw.startswith(("~", "//", "\\\\"))
        or "\\" in raw
        or ".." in posix.parts
    ):
        raise ExternalFleursError(
            f"Formal {label} must be repository-relative and portable: {raw}"
        )
    normalized = posix.as_posix()
    if normalized in {"", "."}:
        raise ExternalFleursError(f"Formal {label} cannot name the repository root")
    root = Path(repository_root).resolve()
    resolved = (root / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ExternalFleursError(
            f"Formal {label} resolves outside the repository: {raw}"
        ) from error
    return normalized


def _artifact_path(
    value: object,
    *,
    formal: bool = False,
    label: str = "artifact path",
) -> Path:
    if formal:
        reference = _portable_repo_reference(value, label=label)
        return ROOT / Path(*PurePosixPath(reference).parts)
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _stored_path_reference(
    path: str | Path,
    *,
    formal: bool,
    label: str,
) -> str:
    """Serialize a runtime path without leaking a formal host path."""

    if not formal:
        return str(path)
    candidate = Path(path)
    if not candidate.is_absolute():
        return _portable_repo_reference(candidate.as_posix(), label=label)
    try:
        relative = candidate.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise ExternalFleursError(
            f"Formal {label} is outside the repository: {candidate}"
        ) from error
    return _portable_repo_reference(relative.as_posix(), label=label)


def _verify_preparation_binding(
    registry: Mapping[str, Any],
    preparation: Mapping[str, Any],
) -> None:
    dataset = preparation.get("dataset")
    output = preparation.get("output")
    if not isinstance(dataset, Mapping) or not isinstance(output, Mapping):
        raise ExternalFleursError("FLEURS preparation verifier returned no contract")
    expected = {
        "fleurs_preparation_lock_sha256": preparation.get(
            "preparation_lock_sha256"
        ),
        "fleurs_preparation_identity_sha256": preparation.get("identity_sha256"),
        "fleurs_dataset_revision": dataset.get("revision"),
        "fleurs_audio_inventory_sha256": output.get("audio_inventory_sha256"),
        "fleurs_audit_sha256": output.get("audit_sha256"),
        "manifest_sha256": output.get("manifest_sha256"),
    }
    for field, observed in expected.items():
        if str(registry.get(field, "")).casefold() != str(observed or "").casefold():
            raise ExternalFleursError(
                f"Run registry differs from the FLEURS preparation lock: {field}"
            )
    if (
        dataset.get("repository") != registry.get("fleurs_dataset_repository")
        or dataset.get("config") != registry.get("fleurs_dataset_config")
        or dataset.get("split") != registry.get("fleurs_dataset_split")
        or int(output.get("row_count", -1)) != int(registry.get("expected_rows", -2))
    ):
        raise ExternalFleursError(
            "Run registry differs from the FLEURS preparation dataset/count"
        )
    manifest_path = Path(str(preparation.get("manifest_path", ""))).resolve()
    registered_manifest = _artifact_path(
        registry.get("manifest", ""),
        formal=registry.get("path_mode") == FORMAL_PATH_MODE,
        label="registry.manifest",
    ).resolve()
    if manifest_path != registered_manifest:
        raise ExternalFleursError(
            "Run registry manifest differs from the FLEURS preparation lock"
        )


def load_run_registry(
    path: str | Path = DEFAULT_RUN_REGISTRY,
    *,
    formal: bool = True,
) -> dict[str, Any]:
    """Validate the static registry without opening its manifest/checkpoints."""

    registry_path = _artifact_path(
        path,
        formal=formal,
        label="run registry",
    )
    registry = _load_json_object(registry_path, label="FLEURS run registry")
    if registry.get("registry_version") != REGISTRY_VERSION:
        raise ExternalFleursError("Unsupported FLEURS run registry version")
    if registry.get("evaluation_domain") != EVALUATION_DOMAIN:
        raise ExternalFleursError(
            "FLEURS registry must declare legacy_exposed_external_replication"
        )
    if str(registry.get("dataset", "")).casefold() != "fleurs":
        raise ExternalFleursError("FLEURS registry must use dataset=fleurs")
    path_mode = registry.get("path_mode")
    if formal and path_mode != FORMAL_PATH_MODE:
        raise ExternalFleursError(
            f"Formal FLEURS registry must use path_mode={FORMAL_PATH_MODE}"
        )
    if not formal and path_mode not in {
        None,
        FORMAL_PATH_MODE,
        DIAGNOSTIC_PATH_MODE,
    }:
        raise ExternalFleursError(f"Unsupported FLEURS registry path_mode: {path_mode}")
    registry_identity = str(registry.get("identity_sha256", "")).casefold()
    identity_payload = dict(registry)
    identity_payload.pop("identity_sha256", None)
    if (
        not is_sha256(registry_identity)
        or canonical_sha256(identity_payload) != registry_identity
    ):
        raise ExternalFleursError("FLEURS run registry identity is invalid/tampered")
    for field in (
        "manifest",
        "fleurs_preparation_lock",
        "split_lock",
        "decision_lock",
    ):
        if not str(registry.get(field, "")).strip():
            raise ExternalFleursError(f"FLEURS registry is missing {field}")
        if formal:
            _portable_repo_reference(registry[field], label=f"registry.{field}")
    for field in (
        "manifest_sha256",
        "fleurs_preparation_lock_sha256",
        "fleurs_preparation_identity_sha256",
        "fleurs_audio_inventory_sha256",
        "fleurs_audit_sha256",
        "split_lock_sha256",
        "decision_lock_sha256",
        "method_lock_sha256",
        "method_identity_sha256",
    ):
        _require_sha256(registry.get(field), label=f"registry.{field}")
    if registry.get("fleurs_preparation_lock_version") != PREPARATION_LOCK_VERSION:
        raise ExternalFleursError("Registry binds an unsupported FLEURS preparation lock")
    if registry.get("fleurs_dataset_repository") != FLEURS_DATASET_REPOSITORY or (
        registry.get("fleurs_dataset_config") != FLEURS_DATASET_CONFIG
        or registry.get("fleurs_dataset_split") != "test"
    ):
        raise ExternalFleursError("Registry has an invalid FLEURS dataset identity")
    if not is_immutable_revision(registry.get("fleurs_dataset_revision")):
        raise ExternalFleursError(
            "Registry must bind an immutable FLEURS dataset revision"
        )
    try:
        expected_rows = int(registry["expected_rows"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExternalFleursError("registry.expected_rows must be an integer") from error
    if expected_rows != DEFAULT_EXPECTED_ROWS:
        raise ExternalFleursError(
            f"Formal FLEURS replication must lock exactly {DEFAULT_EXPECTED_ROWS} rows"
        )

    decoding = registry.get("decoding")
    if not isinstance(decoding, Mapping):
        raise ExternalFleursError("registry.decoding must be an object")
    expected_decoding = {
        "language": "vi",
        "task": "transcribe",
        "sample_rate": SAMPLE_RATE,
        "do_sample": False,
        "num_beams": 1,
    }
    conflicts = [
        field
        for field, expected in expected_decoding.items()
        if decoding.get(field) != expected
    ]
    try:
        max_new_tokens = int(decoding["max_new_tokens"])
        max_chunk_seconds = float(decoding["max_chunk_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExternalFleursError("Invalid registry decoding limits") from error
    if conflicts or max_new_tokens < 1 or not (
        0 < max_chunk_seconds <= MAX_CHUNK_SECONDS
    ):
        raise ExternalFleursError(
            "FLEURS decoding must be deterministic Vietnamese greedy decoding "
            "with a valid <=30-second chunk limit; conflicts=" + str(conflicts)
        )

    entries = registry.get("runs")
    if not isinstance(entries, list) or len(entries) != len(REQUIRED_ROLES):
        raise ExternalFleursError("Registry must name exactly three runs")
    configuration_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ExternalFleursError(f"registry.runs[{index}] must be an object")
        configuration_id = str(entry.get("configuration_id", "")).strip()
        if not configuration_id or configuration_id in configuration_ids:
            raise ExternalFleursError(
                f"Blank/duplicate registry configuration_id: {configuration_id!r}"
            )
        configuration_ids.add(configuration_id)
        for field in ("config_path", "checkpoint_path"):
            if not str(entry.get(field, "")).strip():
                raise ExternalFleursError(
                    f"registry.runs[{index}] is missing {field}"
                )
            if formal:
                _portable_repo_reference(
                    entry[field],
                    label=f"registry.runs[{index}].{field}",
                )
        _require_sha256(
            entry.get("config_sha256"),
            label=f"registry.runs[{index}].config_sha256",
        )
    return registry


def _verify_formal_method_runtime(
    registry: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    decision_path: Path,
    expected_split_hash: str,
    expected_method_hash: str,
    expected_method_identity: str,
    verify_current_method: bool,
    method_config_loader: MethodConfigLoader,
    method_verifier: MethodVerifier,
) -> dict[str, str]:
    """Verify the decision-bound method against all three registered configs.

    This gate deliberately performs no FLEURS or checkpoint access.  A formal
    inference authorization verifies the current runtime/environment; portable
    post-hoc evidence verification can disable only that current-runtime
    comparison while retaining the formal lock, source, config, and dependency
    hash checks.
    """

    raw_decision = _load_json_object(decision_path, label="Method/lambda decision lock")
    if (
        str(raw_decision.get("method_lock_sha256", "")).casefold()
        != expected_method_hash
        or str(raw_decision.get("method_identity_sha256", "")).casefold()
        != expected_method_identity
    ):
        raise ExternalFleursError(
            "Decision method-lock fields differ from the verified decision"
        )
    method_reference = _portable_repo_reference(
        raw_decision.get("method_lock", ""),
        label="decision.method_lock",
    )
    method_path = _artifact_path(
        method_reference,
        formal=True,
        label="decision.method_lock",
    )
    if not method_path.is_file() or sha256_file(method_path) != expected_method_hash:
        raise ExternalFleursError(
            "Decision-bound formal method lock is missing or has changed"
        )

    entries = {
        str(entry["configuration_id"]): entry for entry in registry["runs"]
    }
    integrity_by_role: dict[str, dict[str, str]] = {}
    for role in REQUIRED_ROLES:
        locked = decision.get("locked_configurations")
        if not isinstance(locked, (list, tuple)):
            raise ExternalFleursError("Verified decision has no locked configurations")
        matching_roles = [
            item
            for item in locked
            if isinstance(item, Mapping) and str(item.get("role", "")) == role
        ]
        if len(matching_roles) != 1:
            raise ExternalFleursError(f"Verified decision has no unique role: {role}")
        locked_role = matching_roles[0]
        configuration_id = str(locked_role.get("configuration_id", ""))
        entry = entries.get(configuration_id)
        if entry is None:
            raise ExternalFleursError(
                f"Registry has no config for decision role {role}"
            )
        config_path = _artifact_path(
            entry["config_path"],
            formal=True,
            label=f"{role}.config_path",
        )
        expected_config_hash = _require_sha256(
            entry.get("config_sha256"),
            label=f"{role}.config_sha256",
        )
        if not config_path.is_file() or sha256_file(config_path) != expected_config_hash:
            raise ExternalFleursError(
                f"Registered formal config is missing or has changed: {config_path}"
            )
        try:
            config = dict(method_config_loader(config_path))
        except (OSError, UnicodeError, ValueError) as error:
            raise ExternalFleursError(
                f"Cannot load registered formal config for {role}: {error}"
            ) from error
        protocol = config.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ExternalFleursError(f"Registered config has no protocol: {role}")
        config_method_reference = _portable_repo_reference(
            protocol.get("method_lock", ""),
            label=f"{role}.protocol.method_lock",
        )
        config_method_path = _artifact_path(
            config_method_reference,
            formal=True,
            label=f"{role}.protocol.method_lock",
        )
        if config_method_path.resolve() != method_path.resolve():
            raise ExternalFleursError(
                f"Registered config {role} refers to another method lock"
            )
        if str(config.get("training", {}).get("run_scope", "")).casefold() != "formal":
            raise ExternalFleursError(
                f"Registered config {role} is not a formal training config"
            )
        checkpoint_identity = {
            field: str(locked_role.get(field, "")).casefold()
            for field in (
                "checkpoint_sha256",
                "resolved_config_sha256",
                "training_contract_sha256",
            )
        }
        try:
            matched = verify_test_configuration_locked(
                decision,
                config=config,
                checkpoint_identity=checkpoint_identity,
            )
        except Exception as error:
            raise ExternalFleursError(
                f"Registered config is not decision-locked for role {role}"
            ) from error
        if (
            str(matched.get("configuration_id", "")) != configuration_id
            or str(matched.get("role", "")) != role
        ):
            raise ExternalFleursError(
                f"Registered config resolves to another decision role: {role}"
            )
        try:
            method = dict(
                method_verifier(
                    method_path,
                    config=config,
                    repo_root=ROOT,
                    formal=verify_current_method,
                    verify_audio=False,
                )
            )
        except (MethodContractError, FileNotFoundError, OSError, ValueError) as error:
            gate = "runtime" if verify_current_method else "post-hoc artifact"
            raise ExternalFleursError(
                f"Formal method {gate} verification failed for {role}: {error}"
            ) from error
        if method.get("mode") != "formal":
            raise ExternalFleursError(
                f"FLEURS role {role} is not bound to a formal method lock"
            )
        if (
            str(method.get("method_lock_sha256", "")).casefold()
            != expected_method_hash
            or str(method.get("method_identity_sha256", "")).casefold()
            != expected_method_identity
            or str(method.get("protocol_split_lock_sha256", "")).casefold()
            != expected_split_hash
        ):
            raise ExternalFleursError(
                f"Verified method identity differs from decision/registry for {role}"
            )
        integrity_by_role[role] = {
            "environment_identity_sha256": _require_sha256(
                method.get("environment_identity_sha256"),
                label=f"{role}.method.environment_identity_sha256",
            ),
            "source_tree_sha256": _require_sha256(
                method.get("source_tree_sha256"),
                label=f"{role}.method.source_tree_sha256",
            ),
        }

    environment_identities = {
        value["environment_identity_sha256"] for value in integrity_by_role.values()
    }
    source_trees = {
        value["source_tree_sha256"] for value in integrity_by_role.values()
    }
    if len(environment_identities) != 1 or len(source_trees) != 1:
        raise ExternalFleursError(
            "The three FLEURS configs do not resolve to one method environment/source"
        )
    return {
        "environment_identity_sha256": environment_identities.pop(),
        "source_tree_sha256": source_trees.pop(),
    }


def authorize_external_suite(
    registry_path: str | Path = DEFAULT_RUN_REGISTRY,
    *,
    formal: bool = True,
    verify_current_method: bool = True,
    decision_verifier: DecisionVerifier = verify_test_decision_lock,
    preparation_verifier: PreparationVerifier = verify_fleurs_preparation_lock,
    method_config_loader: MethodConfigLoader = load_experiment_config,
    method_verifier: MethodVerifier = verify_method_lock,
) -> ExternalAuthorization:
    """Authorize decision/roles before any FLEURS row or model is opened."""

    if not isinstance(verify_current_method, bool):
        raise ExternalFleursError("verify_current_method must be boolean")

    path = _artifact_path(
        registry_path,
        formal=formal,
        label="run registry",
    )
    registry = load_run_registry(registry_path, formal=formal)
    split_path = _artifact_path(
        registry["split_lock"],
        formal=formal,
        label="registry.split_lock",
    )
    decision_path = _artifact_path(
        registry["decision_lock"],
        formal=formal,
        label="registry.decision_lock",
    )
    expected_split_hash = _require_sha256(
        registry["split_lock_sha256"], label="registry.split_lock_sha256"
    )
    expected_decision_hash = _require_sha256(
        registry["decision_lock_sha256"], label="registry.decision_lock_sha256"
    )
    if not split_path.is_file() or sha256_file(split_path) != expected_split_hash:
        raise ExternalFleursError("Configured split lock is missing or has changed")
    if not decision_path.is_file() or sha256_file(decision_path) != expected_decision_hash:
        raise ExternalFleursError("Configured decision lock is missing or has changed")
    decision = dict(
        decision_verifier(
            split_lock_path=split_path,
            decision_lock_path=decision_path,
        )
    )
    if str(decision.get("split_lock_sha256", "")).casefold() != expected_split_hash:
        raise ExternalFleursError("Decision verifier returned another split lock")
    if str(decision.get("decision_lock_sha256", "")).casefold() != expected_decision_hash:
        raise ExternalFleursError("Decision verifier returned another decision lock")
    method_lock_hash = _require_sha256(
        decision.get("method_lock_sha256"), label="decision.method_lock_sha256"
    )
    method_identity_hash = _require_sha256(
        decision.get("method_identity_sha256"),
        label="decision.method_identity_sha256",
    )
    if method_lock_hash != str(registry["method_lock_sha256"]).casefold() or (
        method_identity_hash
        != str(registry["method_identity_sha256"]).casefold()
    ):
        raise ExternalFleursError("Run registry does not bind the verified method lock")
    try:
        by_role = resolve_locked_roles(decision)
    except Exception as error:
        raise ExternalFleursError(
            "Decision must lock exactly one ordinary baseline, selected method, "
            "and locked control"
        ) from error
    registered_ids = {
        str(entry["configuration_id"]) for entry in registry["runs"]
    }
    locked_ids = {
        str(item["configuration_id"]) for item in by_role.values()
    }
    if registered_ids != locked_ids:
        raise ExternalFleursError(
            "Run registry does not exactly cover the three decision-locked configurations"
        )
    method_runtime = {
        "environment_identity_sha256": "",
        "source_tree_sha256": "",
    }
    if formal:
        method_runtime = _verify_formal_method_runtime(
            registry,
            decision,
            decision_path=decision_path,
            expected_split_hash=expected_split_hash,
            expected_method_hash=method_lock_hash,
            expected_method_identity=method_identity_hash,
            verify_current_method=verify_current_method,
            method_config_loader=method_config_loader,
            method_verifier=method_verifier,
        )
    # FLEURS metadata is intentionally touched only after the method decision
    # and current method runtime are authorized. Rows/audio remain unopened
    # until run_external_suite.
    preparation_lock_path = _artifact_path(
        registry["fleurs_preparation_lock"],
        formal=formal,
        label="registry.fleurs_preparation_lock",
    )
    expected_preparation_hash = _require_sha256(
        registry["fleurs_preparation_lock_sha256"],
        label="registry.fleurs_preparation_lock_sha256",
    )
    if (
        not preparation_lock_path.is_file()
        or sha256_file(preparation_lock_path) != expected_preparation_hash
    ):
        raise ExternalFleursError("FLEURS preparation lock is missing or changed")
    try:
        preparation = dict(
            preparation_verifier(
                preparation_lock_path,
                repository_root=ROOT,
                expected_count=DEFAULT_EXPECTED_ROWS,
                verify_artifacts=False,
                verify_audio=False,
            )
        )
    except (FleursPreparationError, FileNotFoundError) as error:
        raise ExternalFleursError(f"Invalid FLEURS preparation lock: {error}") from error
    _verify_preparation_binding(registry, preparation)
    return ExternalAuthorization(
        registry_path=path,
        registry_sha256=sha256_file(path),
        registry=registry,
        split_lock_sha256=expected_split_hash,
        decision_lock_sha256=expected_decision_hash,
        method_lock_sha256=method_lock_hash,
        method_identity_sha256=method_identity_hash,
        manifest_path=_artifact_path(
            registry["manifest"],
            formal=formal,
            label="registry.manifest",
        ),
        manifest_sha256=_require_sha256(
            registry["manifest_sha256"], label="registry.manifest_sha256"
        ),
        expected_rows=int(registry["expected_rows"]),
        locked_by_role=by_role,
        method_environment_identity_sha256=method_runtime[
            "environment_identity_sha256"
        ],
        method_source_tree_sha256=method_runtime["source_tree_sha256"],
        method_runtime_verified=bool(formal and verify_current_method),
        fleurs_preparation_lock_path=preparation_lock_path,
        fleurs_preparation_lock_sha256=expected_preparation_hash,
        fleurs_preparation_identity_sha256=_require_sha256(
            registry["fleurs_preparation_identity_sha256"],
            label="registry.fleurs_preparation_identity_sha256",
        ),
        fleurs_dataset_revision=str(registry["fleurs_dataset_revision"]).casefold(),
        fleurs_audio_inventory_sha256=_require_sha256(
            registry["fleurs_audio_inventory_sha256"],
            label="registry.fleurs_audio_inventory_sha256",
        ),
        fleurs_audit_sha256=_require_sha256(
            registry["fleurs_audit_sha256"],
            label="registry.fleurs_audit_sha256",
        ),
        formal=formal,
    )


def create_run_registry(
    output_path: str | Path,
    *,
    preparation_lock_path: str | Path,
    split_lock_path: str | Path,
    decision_lock_path: str | Path,
    config_paths_by_role: Mapping[str, str | Path],
    decision_verifier: DecisionVerifier = verify_test_decision_lock,
    checkpoint_verifier: CheckpointVerifier = verify_checkpoint_config,
    preparation_verifier: PreparationVerifier = verify_fleurs_preparation_lock,
    manifest_path: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
    formal: bool = True,
) -> Path:
    """Create the immutable Gate-6 registry after the Gate-4 decision exists.

    Decision verification deliberately precedes opening FLEURS.  Checkpoint
    paths come from decision v3; callers explicitly bind the matching resolved
    experiment config for each semantic role.
    """

    output_reference = (
        _portable_repo_reference(output_path, label="run registry")
        if formal
        else str(output_path)
    )
    output = _artifact_path(
        output_reference,
        formal=formal,
        label="run registry",
    )
    if output.exists():
        raise FileExistsError(f"Run registry already exists and is immutable: {output}")
    supplied_roles = set(config_paths_by_role)
    if supplied_roles != set(REQUIRED_ROLES):
        raise ExternalFleursError(
            f"Config mapping must contain exactly roles {REQUIRED_ROLES}; "
            f"found {sorted(supplied_roles)}"
        )
    split_reference = (
        _portable_repo_reference(split_lock_path, label="split lock")
        if formal
        else str(split_lock_path)
    )
    decision_reference = (
        _portable_repo_reference(decision_lock_path, label="decision lock")
        if formal
        else str(decision_lock_path)
    )
    preparation_reference = (
        _portable_repo_reference(
            preparation_lock_path,
            label="FLEURS preparation lock",
        )
        if formal
        else str(preparation_lock_path)
    )
    config_references = {
        role: (
            _portable_repo_reference(path, label=f"{role} config")
            if formal
            else str(path)
        )
        for role, path in config_paths_by_role.items()
    }
    split_path = _artifact_path(
        split_reference,
        formal=formal,
        label="split lock",
    )
    decision_path = _artifact_path(
        decision_reference,
        formal=formal,
        label="decision lock",
    )
    decision = dict(
        decision_verifier(
            split_lock_path=split_path,
            decision_lock_path=decision_path,
        )
    )
    try:
        by_role = resolve_locked_roles(decision)
    except Exception as error:
        raise ExternalFleursError("Decision has no exact three-role lock") from error

    entries: list[dict[str, str]] = []
    for role in REQUIRED_ROLES:
        locked = by_role[role]
        config_reference = config_references[role]
        config_path = _artifact_path(
            config_reference,
            formal=formal,
            label=f"{role} config",
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"Role config does not exist: {config_path}")
        config = load_experiment_config(config_path)
        checkpoint_reference = (
            _portable_repo_reference(
                locked.get("checkpoint_path", ""),
                label=f"{role} checkpoint",
            )
            if formal
            else str(locked.get("checkpoint_path", ""))
        )
        checkpoint = _artifact_path(
            checkpoint_reference,
            formal=formal,
            label=f"{role} checkpoint",
        )
        identity = dict(checkpoint_verifier(checkpoint, config))
        try:
            matched = verify_test_configuration_locked(
                decision,
                config=config,
                checkpoint_identity=identity,
            )
        except Exception as error:
            raise ExternalFleursError(
                f"Config/checkpoint does not match the {role} decision identity"
            ) from error
        if matched["configuration_id"] != locked["configuration_id"]:
            raise ExternalFleursError(f"Ambiguous decision identity for role {role}")
        entries.append(
            {
                "configuration_id": str(locked["configuration_id"]),
                "config_path": config_reference,
                "config_sha256": sha256_file(config_path),
                "checkpoint_path": checkpoint_reference,
            }
        )

    # This is the first FLEURS access in registry creation; decision and all
    # three checkpoint identities have already been authorized. The formal
    # preparation verifier checks all 857 manifest rows and raw WAV hashes.
    preparation_path = _artifact_path(
        preparation_reference,
        formal=formal,
        label="FLEURS preparation lock",
    )
    try:
        preparation = dict(
            preparation_verifier(
                preparation_path,
                repository_root=ROOT,
                expected_count=DEFAULT_EXPECTED_ROWS,
                verify_artifacts=True,
                verify_audio=True,
            )
        )
    except (FleursPreparationError, FileNotFoundError) as error:
        raise ExternalFleursError(f"Invalid FLEURS preparation: {error}") from error
    dataset_contract = preparation.get("dataset")
    output_contract = preparation.get("output")
    if not isinstance(dataset_contract, Mapping) or not isinstance(
        output_contract, Mapping
    ):
        raise ExternalFleursError("FLEURS preparation verifier returned no contract")
    manifest = Path(str(preparation.get("manifest_path", "")))
    expected_manifest_hash = _require_sha256(
        output_contract.get("manifest_sha256"),
        label="preparation.output.manifest_sha256",
    )
    prepared_manifest_reference = str(output_contract.get("manifest") or "").strip()
    if formal:
        prepared_manifest_reference = _portable_repo_reference(
            prepared_manifest_reference,
            label="preparation.output.manifest",
        )
        if (
            _artifact_path(
                prepared_manifest_reference,
                formal=True,
                label="preparation.output.manifest",
            ).resolve()
            != manifest.resolve()
        ):
            raise ExternalFleursError(
                "FLEURS preparation returned a non-portable manifest binding"
            )
    elif not prepared_manifest_reference:
        prepared_manifest_reference = str(manifest)
    if manifest_path is not None:
        legacy_manifest_reference = (
            _portable_repo_reference(manifest_path, label="legacy manifest")
            if formal
            else str(manifest_path)
        )
        legacy_manifest = _artifact_path(
            legacy_manifest_reference,
            formal=formal,
            label="legacy manifest",
        )
    else:
        legacy_manifest = None
    if legacy_manifest is not None and legacy_manifest.resolve() != manifest.resolve():
        raise ExternalFleursError(
            "Legacy --manifest argument differs from the formal preparation lock"
        )
    if expected_manifest_sha256 is not None and _require_sha256(
        expected_manifest_sha256,
        label="expected_manifest_sha256",
    ) != expected_manifest_hash:
        raise ExternalFleursError(
            "Legacy manifest hash differs from the formal preparation lock"
        )
    registry = {
        "registry_version": REGISTRY_VERSION,
        "path_mode": FORMAL_PATH_MODE if formal else DIAGNOSTIC_PATH_MODE,
        "evaluation_domain": EVALUATION_DOMAIN,
        "external_evidence_status": "legacy_exposed_replication",
        "dataset": "fleurs",
        "manifest": prepared_manifest_reference,
        "manifest_sha256": expected_manifest_hash,
        "expected_rows": DEFAULT_EXPECTED_ROWS,
        "fleurs_preparation_lock": preparation_reference,
        "fleurs_preparation_lock_version": PREPARATION_LOCK_VERSION,
        "fleurs_preparation_lock_sha256": _require_sha256(
            preparation.get("preparation_lock_sha256"),
            label="preparation.preparation_lock_sha256",
        ),
        "fleurs_preparation_identity_sha256": _require_sha256(
            preparation.get("identity_sha256"),
            label="preparation.identity_sha256",
        ),
        "fleurs_dataset_repository": str(dataset_contract.get("repository", "")),
        "fleurs_dataset_config": str(dataset_contract.get("config", "")),
        "fleurs_dataset_split": str(dataset_contract.get("split", "")),
        "fleurs_dataset_revision": str(dataset_contract.get("revision", "")).casefold(),
        "fleurs_audio_inventory_sha256": _require_sha256(
            output_contract.get("audio_inventory_sha256"),
            label="preparation.output.audio_inventory_sha256",
        ),
        "fleurs_audit_sha256": _require_sha256(
            output_contract.get("audit_sha256"),
            label="preparation.output.audit_sha256",
        ),
        "split_lock": split_reference,
        "split_lock_sha256": str(decision["split_lock_sha256"]).casefold(),
        "decision_lock": decision_reference,
        "decision_lock_sha256": str(
            decision["decision_lock_sha256"]
        ).casefold(),
        "method_lock_sha256": _require_sha256(
            decision.get("method_lock_sha256"), label="decision.method_lock_sha256"
        ),
        "method_identity_sha256": _require_sha256(
            decision.get("method_identity_sha256"),
            label="decision.method_identity_sha256",
        ),
        "decoding": {
            "language": "vi",
            "task": "transcribe",
            "sample_rate": SAMPLE_RATE,
            "max_new_tokens": 440,
            "max_chunk_seconds": MAX_CHUNK_SECONDS,
            "do_sample": False,
            "num_beams": 1,
        },
        "runs": entries,
    }
    registry["identity_sha256"] = canonical_sha256(registry)
    # Validate the complete object through the same loader before publication.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.registry.tmp")
    try:
        temporary.write_text(
            json.dumps(registry, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if formal:
            load_run_registry(
                _stored_path_reference(
                    temporary,
                    formal=True,
                    label="temporary run registry",
                ),
                formal=True,
            )
        else:
            load_run_registry(temporary, formal=False)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_write_new_json(output, registry)
    return output


def build_external_runs(
    authorization: ExternalAuthorization,
    *,
    checkpoint_verifier: CheckpointVerifier = verify_checkpoint_config,
) -> tuple[ExternalRun, ...]:
    """Resolve the authorized role set through the explicit run registry."""

    entries = {
        str(entry["configuration_id"]): entry
        for entry in authorization.registry["runs"]
    }
    runs: list[ExternalRun] = []
    for role in REQUIRED_ROLES:
        locked = authorization.locked_by_role[role]
        configuration_id = str(locked["configuration_id"])
        entry = entries[configuration_id]
        config_path = _artifact_path(
            entry["config_path"],
            formal=authorization.formal,
            label=f"{configuration_id}.config_path",
        )
        expected_config_hash = _require_sha256(
            entry["config_sha256"], label=f"{configuration_id}.config_sha256"
        )
        if not config_path.is_file() or sha256_file(config_path) != expected_config_hash:
            raise ExternalFleursError(
                f"Registered config is missing or has changed: {config_path}"
            )
        config = load_experiment_config(config_path)
        checkpoint = _artifact_path(
            entry["checkpoint_path"],
            formal=authorization.formal,
            label=f"{configuration_id}.checkpoint_path",
        )
        locked_checkpoint = _artifact_path(
            locked.get("checkpoint_path", ""),
            formal=authorization.formal,
            label=f"{configuration_id}.decision_checkpoint_path",
        )
        if not str(locked.get("checkpoint_path", "")).strip() or (
            checkpoint.resolve() != locked_checkpoint.resolve()
        ):
            raise ExternalFleursError(
                f"Registry checkpoint path disagrees with decision: {configuration_id}"
            )
        identity = dict(checkpoint_verifier(checkpoint, config))
        try:
            matched = verify_test_configuration_locked(
                {"locked_configurations": tuple(authorization.locked_by_role.values())},
                config=config,
                checkpoint_identity=identity,
            )
        except Exception as error:
            raise ExternalFleursError(
                f"Registered checkpoint/config is not decision-locked: {configuration_id}"
            ) from error
        if matched["configuration_id"] != configuration_id or matched["role"] != role:
            raise ExternalFleursError(
                f"Registry role/configuration mismatch: {configuration_id}"
            )
        model = config["model"]
        revision = str(model.get("revision", "")).strip().casefold()
        if not is_immutable_revision(revision) or revision != str(
            locked["backbone_revision"]
        ).casefold():
            raise ExternalFleursError(
                f"Configuration has no matching immutable backbone revision: {config_path}"
            )
        language = str(model.get("language", ""))
        task = str(model.get("task", ""))
        if language != "vi" or task != "transcribe":
            raise ExternalFleursError(
                f"Configuration must use language=vi, task=transcribe: {config_path}"
            )
        runs.append(
            ExternalRun(
                configuration_id=configuration_id,
                role=role,
                method_id=str(locked["method_id"]),
                train_type=str(locked["train_type"]),
                lambda_value=_canonical_lambda(locked["lambda"]),
                seed=str(int(locked["seed"])),
                model_name_or_path=str(locked["backbone"]),
                backbone_revision=revision,
                language=language,
                task=task,
                checkpoint=checkpoint,
                checkpoint_sha256=str(identity["checkpoint_sha256"]).casefold(),
                resolved_config_sha256=str(
                    identity["resolved_config_sha256"]
                ).casefold(),
                training_contract_sha256=str(
                    identity["training_contract_sha256"]
                ).casefold(),
                config_path=config_path,
                config_sha256=expected_config_hash,
                prediction_name=f"pred_{configuration_id}.csv",
            )
        )
    return tuple(runs)


def _read_manifest_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"FLEURS manifest does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ExternalFleursError(f"Unsupported FLEURS manifest format: {path}")


def load_fleurs_manifest(
    path: str | Path,
    *,
    expected_rows: int | None = DEFAULT_EXPECTED_ROWS,
    require_audio: bool = True,
    formal: bool = True,
    repository_root: str | Path = ROOT,
    verify_audio_hashes: bool = False,
) -> list[dict[str, str]]:
    """Load a materialized FLEURS test manifest with canonical clean metadata."""

    manifest_path = Path(path)
    repo_root = Path(repository_root).resolve()
    raw_rows = _read_manifest_records(manifest_path)
    if not raw_rows:
        raise ExternalFleursError(f"FLEURS manifest is empty: {manifest_path}")
    if expected_rows is not None and len(raw_rows) != expected_rows:
        raise ExternalFleursError(
            f"FLEURS test manifest has {len(raw_rows)} rows, expected {expected_rows}"
        )

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        if formal and tuple(raw) != FLEURS_MANIFEST_FIELDS:
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} does not match the locked "
                f"FLEURS schema {FLEURS_MANIFEST_FIELDS}"
            )
        audio_value = raw.get("audio_path") or raw.get("audio")
        reference = (
            raw.get("transcript")
            if raw.get("transcript") is not None
            else raw.get("transcription")
        )
        if reference is None:
            reference = raw.get("text") if raw.get("text") is not None else raw.get("ref")
        if not audio_value or reference is None or not str(reference).strip():
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} requires audio_path and a non-empty transcript"
            )

        audio_reference = str(audio_value).strip()
        audio_path = Path(audio_reference)
        if formal:
            if audio_path.is_absolute() or ".." in audio_path.parts:
                raise ExternalFleursError(
                    f"{manifest_path}: row {row_number} audio_path must be "
                    "repository-relative and portable"
                )
            audio_path = (repo_root / audio_path).resolve()
            try:
                audio_path.relative_to(repo_root)
            except ValueError as error:
                raise ExternalFleursError(
                    f"{manifest_path}: row {row_number} audio_path resolves "
                    "outside the repository"
                ) from error
        elif not audio_path.is_absolute() and not audio_path.exists():
            relative_candidate = manifest_path.parent / audio_path
            if relative_candidate.exists():
                audio_path = relative_candidate
        if require_audio and not audio_path.exists():
            raise FileNotFoundError(
                f"{manifest_path}: row {row_number} audio does not exist: {audio_path}"
            )
        audio_sha256 = str(raw.get("audio_sha256", "")).strip().casefold()
        if formal and not is_sha256(audio_sha256):
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} requires audio_sha256"
            )
        if verify_audio_hashes and (
            not audio_path.is_file() or sha256_file(audio_path) != audio_sha256
        ):
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} audio SHA-256 mismatch/tamper"
            )

        dataset = str(raw.get("dataset", "fleurs")).strip().casefold()
        split = str(raw.get("split", "test")).strip().casefold()
        snr = str(raw.get("snr", "clean")).strip().casefold()
        noise_type = str(raw.get("noise_type", "clean")).strip().casefold()
        if dataset != "fleurs" or split != "test":
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} must use dataset=fleurs and split=test"
            )
        if snr != "clean" or noise_type not in {"", "clean"}:
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} FLEURS external audio must be clean"
            )

        utt_id = str(raw.get("utt_id") or audio_path.stem).strip()
        if not utt_id:
            raise ExternalFleursError(f"{manifest_path}: row {row_number} has an empty utt_id")
        if utt_id in seen:
            raise ExternalFleursError(
                f"{manifest_path}: duplicate utt_id {utt_id!r} at row {row_number}"
            )
        seen.add(utt_id)
        if formal and audio_path.stem != utt_id:
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} utt_id/audio filename mismatch"
            )
        rows.append(
            {
                "utt_id": utt_id,
                "dataset": "fleurs",
                "audio_path": str(audio_path),
                "audio_reference": audio_reference,
                "audio_sha256": audio_sha256,
                "ref": unicodedata.normalize("NFC", str(reference)),
                "snr": "clean",
                "noise_type": "clean",
            }
        )
    return rows


def split_waveform(
    waveform: Any,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_chunk_seconds: float = MAX_CHUNK_SECONDS,
) -> list[Any]:
    """Split a waveform completely into balanced deterministic <=30s chunks.

    Balancing avoids a very short final chunk, which can make Whisper
    hallucinate text from sub-second audio when an utterance is only slightly
    longer than the 30-second feature window.
    """

    if sample_rate < 1:
        raise ExternalFleursError("sample_rate must be positive")
    if max_chunk_seconds <= 0 or max_chunk_seconds > MAX_CHUNK_SECONDS:
        raise ExternalFleursError(
            f"max_chunk_seconds must be in (0, {MAX_CHUNK_SECONDS:g}]"
        )
    samples_per_chunk = int(round(sample_rate * max_chunk_seconds))
    if samples_per_chunk < 1:
        raise ExternalFleursError("Chunk duration resolves to fewer than one sample")
    if len(waveform) < 1:
        raise ExternalFleursError("Cannot transcribe an empty waveform")
    chunk_count = (len(waveform) + samples_per_chunk - 1) // samples_per_chunk
    base_size, remainder = divmod(len(waveform), chunk_count)
    chunks: list[Any] = []
    start = 0
    for index in range(chunk_count):
        chunk_size = base_size + (1 if index < remainder else 0)
        end = start + chunk_size
        chunks.append(waveform[start:end])
        start = end
    return chunks


def join_chunk_hypotheses(hypotheses: Sequence[str]) -> str:
    """Join chunk hypotheses in time order with stable whitespace."""

    return " ".join(text.strip() for text in hypotheses if text.strip())


def _prediction_row(
    manifest_row: Mapping[str, str],
    run: ExternalRun,
    hypothesis: str,
) -> dict[str, str]:
    return {
        "utt_id": manifest_row["utt_id"],
        **run.run_metadata,
        "snr": "clean",
        "noise_type": "clean",
        "ref": manifest_row["ref"],
        "hyp": hypothesis,
    }


def _partial_path(prediction_path: Path) -> Path:
    return prediction_path.with_name(f".{prediction_path.stem}.partial.csv")


def _resume_path(prediction_path: Path) -> Path:
    return prediction_path.with_suffix(prediction_path.suffix + ".resume.json")


def _recovery_path(prediction_path: Path) -> Path:
    return prediction_path.with_suffix(prediction_path.suffix + ".recovery.json")


def _provenance_path(prediction_path: Path) -> Path:
    return prediction_path.with_suffix(prediction_path.suffix + ".provenance.json")


def _result_provenance_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".provenance.json")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"Refusing to overwrite immutable JSON artifact: {path}"
            ) from None
    finally:
        if temporary.exists():
            temporary.unlink()


def _selected_rows_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    selected = [
        {
            "utt_id": row["utt_id"],
            "audio_path": row.get("audio_reference", row["audio_path"]),
            "audio_sha256": row.get("audio_sha256", ""),
            "ref": row["ref"],
            "snr": row["snr"],
            "noise_type": row["noise_type"],
        }
        for row in rows
    ]
    return canonical_sha256(selected)


def build_run_contract(
    run: ExternalRun,
    authorization: ExternalAuthorization,
    manifest_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    decoding = dict(authorization.registry["decoding"])
    return {
        "contract_version": "paper_v2_fleurs_inference_contract_v1",
        "evaluation_domain": EVALUATION_DOMAIN,
        "registry_sha256": authorization.registry_sha256,
        "split_lock_sha256": authorization.split_lock_sha256,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "method_identity_sha256": authorization.method_identity_sha256,
        "method_environment_identity_sha256": (
            authorization.method_environment_identity_sha256
        ),
        "method_source_tree_sha256": authorization.method_source_tree_sha256,
        "method_runtime_verified": authorization.method_runtime_verified,
        "manifest_sha256": authorization.manifest_sha256,
        "fleurs_preparation_lock_sha256": authorization.fleurs_preparation_lock_sha256,
        "fleurs_preparation_identity_sha256": authorization.fleurs_preparation_identity_sha256,
        "fleurs_dataset_revision": authorization.fleurs_dataset_revision,
        "fleurs_audio_inventory_sha256": authorization.fleurs_audio_inventory_sha256,
        "fleurs_audit_sha256": authorization.fleurs_audit_sha256,
        "selected_rows_sha256": _selected_rows_sha256(manifest_rows),
        "selected_rows": len(manifest_rows),
        "configuration_id": run.configuration_id,
        "role": run.role,
        "method_id": run.method_id,
        "train_type": run.train_type,
        "lambda": run.lambda_value,
        "seed": run.seed,
        "backbone": run.model_name_or_path,
        "backbone_revision": run.backbone_revision,
        "checkpoint_sha256": run.checkpoint_sha256,
        "resolved_config_sha256": run.resolved_config_sha256,
        "training_contract_sha256": run.training_contract_sha256,
        "config_sha256": run.config_sha256,
        "prediction_schema": list(CANONICAL_PREDICTION_COLUMNS),
        "decoding": decoding,
    }


def _prediction_csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CANONICAL_PREDICTION_COLUMNS),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {name: row.get(name, "") for name in CANONICAL_PREDICTION_COLUMNS}
        )
    return buffer.getvalue().encode("utf-8")


def _resume_contract_fields(run_contract: Mapping[str, Any]) -> dict[str, object]:
    return {
        "run_contract_sha256": canonical_sha256(run_contract),
        "registry_sha256": run_contract["registry_sha256"],
        "manifest_sha256": run_contract["manifest_sha256"],
        "selected_rows_sha256": run_contract["selected_rows_sha256"],
        "configuration_id": run_contract["configuration_id"],
        "role": run_contract["role"],
        "prediction_schema_sha256": canonical_sha256(
            list(CANONICAL_PREDICTION_COLUMNS)
        ),
    }


def _resume_payload(
    *,
    partial: Path,
    rows: int,
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "resume_version": RESUME_VERSION,
        "partial_prediction_sha256": sha256_file(partial),
        "completed_rows": rows,
        **_resume_contract_fields(run_contract),
    }


def _validate_resume_state(
    state: Mapping[str, Any],
    *,
    partial: Path,
    rows: int,
    run_contract: Mapping[str, Any],
) -> None:
    if state.get("resume_version") != RESUME_VERSION:
        raise ExternalFleursError("Unsupported FLEURS resume state")
    try:
        completed_rows = int(state.get("completed_rows", -1))
    except (TypeError, ValueError) as error:
        raise ExternalFleursError("Resume state row count is invalid") from error
    if completed_rows != rows:
        raise ExternalFleursError("Resume row count does not match partial prediction")
    expected: dict[str, object] = {
        "partial_prediction_sha256": sha256_file(partial),
        **_resume_contract_fields(run_contract),
    }
    for field, value in expected.items():
        if str(state.get(field, "")).casefold() != value.casefold():
            raise ExternalFleursError(f"Resume state mismatch: {field}")


def _recovery_payload(
    *,
    prediction_rows: Sequence[Mapping[str, object]],
    previous_rows: int,
    previous_sha256: str,
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _prediction_csv_bytes(prediction_rows)
    return {
        "recovery_version": RECOVERY_VERSION,
        "completed_rows": len(prediction_rows),
        "partial_prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "previous_completed_rows": previous_rows,
        "previous_partial_prediction_sha256": previous_sha256,
        **_resume_contract_fields(run_contract),
    }


def _validate_recovery_contract(
    receipt: Mapping[str, Any],
    *,
    run_contract: Mapping[str, Any],
) -> None:
    if receipt.get("recovery_version") != RECOVERY_VERSION:
        raise ExternalFleursError("Unsupported FLEURS recovery receipt")
    for field, value in _resume_contract_fields(run_contract).items():
        if str(receipt.get(field, "")).casefold() != str(value).casefold():
            raise ExternalFleursError(f"Recovery receipt mismatch: {field}")
    for field in ("completed_rows", "previous_completed_rows"):
        try:
            if int(receipt.get(field, -1)) < 0:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ExternalFleursError(
                f"Recovery receipt has invalid {field}"
            ) from error
    if not is_sha256(receipt.get("partial_prediction_sha256")):
        raise ExternalFleursError("Recovery receipt prediction hash is invalid")
    previous_hash = str(
        receipt.get("previous_partial_prediction_sha256", "")
    ).casefold()
    if previous_hash and not is_sha256(previous_hash):
        raise ExternalFleursError("Recovery receipt previous hash is invalid")


def _publish_partial_checkpoint(
    *,
    partial: Path,
    progress: Path,
    recovery: Path,
    prediction_rows: Sequence[Mapping[str, object]],
    run_contract: Mapping[str, Any],
) -> None:
    """Publish CSV/state with a write-ahead receipt for the crash gap."""

    desired_sha256 = hashlib.sha256(
        _prediction_csv_bytes(prediction_rows)
    ).hexdigest()
    if partial.exists() and progress.exists() and (
        sha256_file(partial) == desired_sha256
    ):
        state = _load_json_object(progress, label="FLEURS resume state")
        _validate_resume_state(
            state,
            partial=partial,
            rows=len(prediction_rows),
            run_contract=run_contract,
        )
        return
    previous_rows = 0
    previous_sha256 = ""
    if partial.exists():
        previous_rows = len(load_prediction_csv(partial))
        previous_sha256 = sha256_file(partial)
    receipt = _recovery_payload(
        prediction_rows=prediction_rows,
        previous_rows=previous_rows,
        previous_sha256=previous_sha256,
        run_contract=run_contract,
    )
    _atomic_write_json(recovery, receipt)
    atomic_write_csv(partial, prediction_rows, CANONICAL_PREDICTION_COLUMNS)
    if sha256_file(partial) != receipt["partial_prediction_sha256"]:
        raise RuntimeError("Published FLEURS checkpoint differs from recovery receipt")
    _atomic_write_json(
        progress,
        _resume_payload(
            partial=partial,
            rows=len(prediction_rows),
            run_contract=run_contract,
        ),
    )
    recovery.unlink()


def _reconcile_recovery_receipt(
    *,
    partial: Path,
    progress: Path,
    recovery: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    run: ExternalRun,
    run_contract: Mapping[str, Any],
) -> None:
    """Resolve only the two atomic outcomes described by a trusted receipt."""

    receipt = _load_json_object(recovery, label="FLEURS recovery receipt")
    _validate_recovery_contract(receipt, run_contract=run_contract)
    current_count = int(receipt["completed_rows"])
    previous_count = int(receipt["previous_completed_rows"])
    current_hash = str(receipt["partial_prediction_sha256"]).casefold()
    previous_hash = str(
        receipt.get("previous_partial_prediction_sha256", "")
    ).casefold()
    if current_count <= previous_count:
        raise ExternalFleursError("Recovery receipt has a non-forward row transition")

    if not partial.exists():
        if progress.exists() or previous_count != 0 or previous_hash:
            raise ExternalFleursError(
                "Recovery receipt is inconsistent with missing partial artifacts"
            )
        # Receipt committed, CSV replace did not: retry from row zero.
        recovery.unlink()
        return

    rows = _validate_prediction_prefix(
        load_prediction_csv(partial),
        manifest_rows,
        run,
        source=partial,
        require_complete=False,
    )
    actual_hash = sha256_file(partial)
    if actual_hash == current_hash and len(rows) == current_count:
        if progress.exists():
            state = _load_json_object(progress, label="FLEURS resume state")
            try:
                _validate_resume_state(
                    state,
                    partial=partial,
                    rows=current_count,
                    run_contract=run_contract,
                )
            except ExternalFleursError:
                # The CSV advanced but the state still describes the previous
                # checkpoint.  Validate that exact old identity before repair.
                for field, value in _resume_contract_fields(run_contract).items():
                    if str(state.get(field, "")).casefold() != str(value).casefold():
                        raise ExternalFleursError(
                            f"Recovery previous state mismatch: {field}"
                        )
                if int(state.get("completed_rows", -1)) != previous_count or str(
                    state.get("partial_prediction_sha256", "")
                ).casefold() != previous_hash:
                    raise ExternalFleursError(
                        "Recovery previous state is stale or tampered"
                    )
                _atomic_write_json(
                    progress,
                    _resume_payload(
                        partial=partial,
                        rows=current_count,
                        run_contract=run_contract,
                    ),
                )
        else:
            if previous_count != 0 or previous_hash:
                raise ExternalFleursError(
                    "Recovery state is missing after a non-initial checkpoint"
                )
            _atomic_write_json(
                progress,
                _resume_payload(
                    partial=partial,
                    rows=current_count,
                    run_contract=run_contract,
                ),
            )
        recovery.unlink()
        return

    if actual_hash == previous_hash and len(rows) == previous_count:
        if not progress.exists():
            raise ExternalFleursError(
                "Recovery receipt found the previous CSV without its state"
            )
        state = _load_json_object(progress, label="FLEURS resume state")
        _validate_resume_state(
            state,
            partial=partial,
            rows=previous_count,
            run_contract=run_contract,
        )
        # Receipt committed, CSV replace did not: retain the prior checkpoint.
        recovery.unlink()
        return
    raise ExternalFleursError(
        "Partial prediction matches neither hash in its recovery receipt; "
        "refusing possible tamper"
    )


def _prediction_provenance(
    *,
    prediction: Path,
    run: ExternalRun,
    authorization: ExternalAuthorization,
    run_contract: Mapping[str, Any],
    num_rows: int,
) -> dict[str, Any]:
    return {
        "provenance_version": PROVENANCE_VERSION,
        "evaluation_domain": EVALUATION_DOMAIN,
        "evaluation_scope": (
            "full_fleurs_857"
            if num_rows == authorization.expected_rows
            else "deterministic_first_n_smoke"
        ),
        "external_evidence_status": "legacy_exposed_replication",
        "prediction": _stored_path_reference(
            prediction,
            formal=authorization.formal,
            label="prediction output",
        ),
        "prediction_sha256": sha256_file(prediction),
        "num_rows": num_rows,
        "manifest": (
            _portable_repo_reference(
                authorization.registry.get("manifest", ""),
                label="registry.manifest",
            )
            if authorization.formal
            else str(authorization.registry.get("manifest", ""))
        ),
        "manifest_sha256": authorization.manifest_sha256,
        "fleurs_preparation_lock_sha256": authorization.fleurs_preparation_lock_sha256,
        "fleurs_preparation_identity_sha256": authorization.fleurs_preparation_identity_sha256,
        "fleurs_dataset_revision": authorization.fleurs_dataset_revision,
        "fleurs_audio_inventory_sha256": authorization.fleurs_audio_inventory_sha256,
        "fleurs_audit_sha256": authorization.fleurs_audit_sha256,
        "registry": _stored_path_reference(
            authorization.registry_path,
            formal=authorization.formal,
            label="run registry",
        ),
        "registry_sha256": authorization.registry_sha256,
        "split_lock_sha256": authorization.split_lock_sha256,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "method_identity_sha256": authorization.method_identity_sha256,
        "method_environment_identity_sha256": (
            authorization.method_environment_identity_sha256
        ),
        "method_source_tree_sha256": authorization.method_source_tree_sha256,
        "method_runtime_verified": authorization.method_runtime_verified,
        "configuration_id": run.configuration_id,
        "role": run.role,
        "method_id": run.method_id,
        "train_type": run.train_type,
        "lambda": run.lambda_value,
        "seed": run.seed,
        "checkpoint": _stored_path_reference(
            run.checkpoint,
            formal=authorization.formal,
            label="checkpoint",
        ),
        "checkpoint_sha256": run.checkpoint_sha256,
        "config": _stored_path_reference(
            run.config_path,
            formal=authorization.formal,
            label="config",
        ),
        "config_sha256": run.config_sha256,
        "resolved_config_sha256": run.resolved_config_sha256,
        "training_contract_sha256": run.training_contract_sha256,
        "backbone": run.model_name_or_path,
        "backbone_revision": run.backbone_revision,
        "run_contract": dict(run_contract),
        "run_contract_sha256": canonical_sha256(run_contract),
        "decoding": {
            **dict(run_contract["decoding"]),
            "implementation": "whisper_greedy_chunked_30s_v1",
            "deterministic_algorithms": True,
        },
    }


def _validate_completed_provenance(
    provenance: Mapping[str, Any],
    *,
    prediction: Path,
    run: ExternalRun,
    authorization: ExternalAuthorization,
    run_contract: Mapping[str, Any],
    num_rows: int,
) -> None:
    if provenance.get("provenance_version") != PROVENANCE_VERSION:
        raise ExternalFleursError("Unsupported FLEURS prediction provenance")
    expected = {
        "evaluation_domain": EVALUATION_DOMAIN,
        "prediction_sha256": sha256_file(prediction),
        "manifest_sha256": authorization.manifest_sha256,
        "fleurs_preparation_lock_sha256": authorization.fleurs_preparation_lock_sha256,
        "fleurs_preparation_identity_sha256": authorization.fleurs_preparation_identity_sha256,
        "fleurs_dataset_revision": authorization.fleurs_dataset_revision,
        "fleurs_audio_inventory_sha256": authorization.fleurs_audio_inventory_sha256,
        "fleurs_audit_sha256": authorization.fleurs_audit_sha256,
        "registry_sha256": authorization.registry_sha256,
        "split_lock_sha256": authorization.split_lock_sha256,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "method_identity_sha256": authorization.method_identity_sha256,
        "method_environment_identity_sha256": (
            authorization.method_environment_identity_sha256
        ),
        "method_source_tree_sha256": authorization.method_source_tree_sha256,
        "method_runtime_verified": authorization.method_runtime_verified,
        "configuration_id": run.configuration_id,
        "role": run.role,
        "checkpoint_sha256": run.checkpoint_sha256,
        "backbone_revision": run.backbone_revision,
        "run_contract_sha256": canonical_sha256(run_contract),
    }
    for field, value in expected.items():
        if str(provenance.get(field, "")).casefold() != str(value).casefold():
            raise ExternalFleursError(f"Completed provenance mismatch: {field}")
    if int(provenance.get("num_rows", -1)) != num_rows:
        raise ExternalFleursError("Completed provenance row count mismatch")


def _validate_prediction_prefix(
    rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, str]],
    run: ExternalRun,
    *,
    source: str | Path,
    require_complete: bool,
) -> list[dict[str, str]]:
    validated = validate_prediction_rows(rows, source=source)
    if len(validated) > len(manifest_rows):
        raise ExternalFleursError(
            f"{source}: has {len(validated)} rows but selected manifest has {len(manifest_rows)}"
        )
    if require_complete and len(validated) != len(manifest_rows):
        raise ExternalFleursError(
            f"{source}: has {len(validated)} rows, expected {len(manifest_rows)}"
        )
    expected_metadata = run.run_metadata
    for index, row in enumerate(validated):
        manifest_row = manifest_rows[index]
        expected = {
            **expected_metadata,
            "utt_id": manifest_row["utt_id"],
            "snr": "clean",
            "noise_type": "clean",
            "ref": manifest_row["ref"],
        }
        conflicts = [name for name, value in expected.items() if row[name] != value]
        if conflicts:
            raise ExternalFleursError(
                f"{source}: row {index + 2} is not the expected manifest prefix; "
                f"conflicts={conflicts}"
            )
    return validated


def _default_audio_loader(path: str, sample_rate: int) -> Any:
    from src.vitonesr.noise import read_audio

    return read_audio(path, sr=sample_rate)


def _load_processor_with_fallback(
    processor_class: Any,
    run: ExternalRun,
) -> Any:
    """Load a checkpoint processor, falling back to the unchanged base tokenizer.

    Some checkpoints were saved by an older Transformers version whose
    ``tokenizer_config.json`` encoded ``extra_special_tokens`` as a list.
    Transformers 4.57 expects a mapping and raises while loading that local
    copy.  LoRA changes model weights only, so the base PhoWhisper processor is
    the canonical, safe fallback for every run in this external suite.
    """

    kwargs = {"language": run.language, "task": run.task}
    local_processor = run.checkpoint / "processor"
    local_error: Exception | None = None
    if local_processor.exists():
        try:
            return processor_class.from_pretrained(str(local_processor), **kwargs)
        except Exception as error:
            local_error = error
            warnings.warn(
                f"Checkpoint processor at {local_processor} is incompatible "
                f"({type(error).__name__}: {error}); falling back to "
                f"{run.model_name_or_path}.",
                RuntimeWarning,
                stacklevel=2,
            )

    try:
        return processor_class.from_pretrained(
            run.model_name_or_path,
            revision=run.backbone_revision,
            **kwargs,
        )
    except Exception as error:
        if local_error is not None:
            raise RuntimeError(
                "Could not load either the checkpoint-local processor or the "
                f"base processor {run.model_name_or_path!r}"
            ) from error
        raise


def _load_base_model(model_class: Any, run: ExternalRun) -> Any:
    """Resolve the backbone only at the immutable decision-locked revision."""

    return model_class.from_pretrained(
        run.model_name_or_path,
        revision=run.backbone_revision,
    )


class WhisperAdapterTranscriber:
    """Lazy PEFT/Transformers wrapper so analysis-only use needs no model deps."""

    def __init__(self, run: ExternalRun, device_arg: str, max_new_tokens: int) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
                WhisperTokenizer,
            )
        except ImportError as error:
            raise RuntimeError(
                "FLEURS inference requires torch, transformers, and peft"
            ) from error

        checkpoint = run.checkpoint
        adapter = checkpoint / "adapter"
        if not checkpoint.exists() or not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Missing completed PEFT checkpoint for lambda={run.lambda_value}: {checkpoint}"
            )
        if max_new_tokens < 1:
            raise ExternalFleursError("max_new_tokens must be at least 1")

        if device_arg == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        torch.manual_seed(int(run.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(run.seed))
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.processor = _load_processor_with_fallback(WhisperProcessor, run)
        base_model = _load_base_model(WhisperForConditionalGeneration, run)
        base_model.config.use_cache = True
        self.model = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=False)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.language = run.language
        self.task = run.task
        self.model_name_or_path = run.model_name_or_path
        self.backbone_revision = run.backbone_revision
        self.slow_tokenizer_class = WhisperTokenizer
        self.slow_tokenizer: Any | None = None

    def _decode_generated(self, generated: Any) -> str:
        decoded = str(
            self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        )
        if "\ufffd" not in decoded:
            return decoded

        if self.slow_tokenizer is None:
            self.slow_tokenizer = self.slow_tokenizer_class.from_pretrained(
                self.model_name_or_path,
                revision=self.backbone_revision,
                language=self.language,
                task=self.task,
                errors="strict",
            )
            if self.slow_tokenizer.get_vocab() != self.processor.tokenizer.get_vocab():
                raise ExternalFleursError(
                    "Cannot recover invalid byte-BPE output with a different tokenizer vocab"
                )

        sequence = generated[0]
        token_ids = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        try:
            strict_decoded = self.slow_tokenizer.decode(
                token_ids, skip_special_tokens=True
            )
        except UnicodeDecodeError:
            self.slow_tokenizer.errors = "ignore"
            try:
                recovered = str(
                    self.slow_tokenizer.decode(token_ids, skip_special_tokens=True)
                )
            finally:
                self.slow_tokenizer.errors = "strict"
            if "\ufffd" in recovered:
                raise ExternalFleursError(
                    "Slow-tokenizer byte recovery still contains U+FFFD"
                )
            warnings.warn(
                "Recovered invalid byte-BPE output with the same tokenizer vocab "
                "and errors='ignore'",
                UnicodeWarning,
                stacklevel=2,
            )
            return recovered
        return str(strict_decoded)

    def transcribe_chunk(self, waveform: Any) -> str:
        feature_batch = self.processor.feature_extractor(
            [waveform],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feature_batch.input_features.to(
            device=self.device, dtype=self.dtype
        )
        attention_mask = feature_batch.attention_mask.to(device=self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "language": self.language,
            "task": self.task,
            "do_sample": False,
            "num_beams": 1,
        }
        with self.torch.inference_mode():
            try:
                generated = self.model.generate(
                    input_features, attention_mask=attention_mask, **kwargs
                )
            except TypeError:
                generated = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                        language=self.language, task=self.task
                    ),
                )
        return self._decode_generated(generated)

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _default_transcriber_factory(
    run: ExternalRun, device_arg: str, max_new_tokens: int
) -> ChunkTranscriber:
    return WhisperAdapterTranscriber(run, device_arg, max_new_tokens)


def run_external_prediction(
    run: ExternalRun,
    manifest_rows: Sequence[Mapping[str, str]],
    prediction_path: str | Path,
    *,
    authorization: ExternalAuthorization,
    device_arg: str = "auto",
    checkpoint_every: int = 10,
    resume: bool = False,
    transcriber_factory: TranscriberFactory = _default_transcriber_factory,
    audio_loader: AudioLoader = _default_audio_loader,
) -> Path:
    """Run/resume one locked checkpoint and publish CSV plus provenance."""

    if checkpoint_every < 1:
        raise ExternalFleursError("checkpoint_every must be at least 1")
    if not manifest_rows:
        raise ExternalFleursError("Cannot run inference for an empty manifest")

    output_reference = (
        _portable_repo_reference(prediction_path, label="prediction output")
        if authorization.formal
        else str(prediction_path)
    )
    output = _artifact_path(
        output_reference,
        formal=authorization.formal,
        label="prediction output",
    )
    partial = _partial_path(output)
    progress = _resume_path(output)
    recovery = _recovery_path(output)
    sidecar = _provenance_path(output)
    run_contract = build_run_contract(run, authorization, manifest_rows)
    decoding = authorization.registry["decoding"]
    sample_rate = int(decoding["sample_rate"])
    max_new_tokens = int(decoding["max_new_tokens"])
    max_chunk_seconds = float(decoding["max_chunk_seconds"])

    if sidecar.exists() and not output.exists():
        raise ExternalFleursError(f"Provenance exists without prediction: {sidecar}")
    if recovery.exists() and not resume:
        raise FileExistsError(
            f"Recovery receipt exists: {recovery}; pass --resume after review"
        )
    if recovery.exists():
        _reconcile_recovery_receipt(
            partial=partial,
            progress=progress,
            recovery=recovery,
            manifest_rows=manifest_rows,
            run=run,
            run_contract=run_contract,
        )
    if progress.exists() and not partial.exists():
        raise ExternalFleursError(
            f"Resume state exists without a partial prediction: {output}"
        )
    if partial.exists() and not progress.exists():
        if not resume:
            raise FileExistsError(
                f"Partial prediction exists without its commit state: {partial}; "
                "pass --resume after review"
            )
        if authorization.formal:
            raise ExternalFleursError(
                "Formal state-less partial prediction has no write-ahead recovery "
                "receipt; refusing possible tamper"
            )
        recovered_rows = _validate_prediction_prefix(
            load_prediction_csv(partial),
            manifest_rows,
            run,
            source=partial,
            require_complete=False,
        )
        if not recovered_rows or (
            len(recovered_rows) != len(manifest_rows)
            and len(recovered_rows) % checkpoint_every != 0
        ):
            raise ExternalFleursError(
                "State-less partial prediction is not an exact checkpoint boundary"
            )
        # This is the narrow crash window after the atomic CSV checkpoint was
        # committed but before its state JSON was committed.  Exact schema,
        # manifest prefix, role metadata and the current registry-bound run
        # contract were validated above; rebuilding only the commit state is
        # therefore safe.  A pre-existing state is never repaired or weakened.
        _atomic_write_json(
            progress,
            _resume_payload(
                partial=partial,
                rows=len(recovered_rows),
                run_contract=run_contract,
            ),
        )
    if output.exists() and sidecar.exists():
        if not resume:
            raise FileExistsError(
                f"Completed prediction already exists: {output}; outputs are immutable"
            )
        rows = load_prediction_csv(output)
        validated = _validate_prediction_prefix(
            rows, manifest_rows, run, source=output, require_complete=True
        )
        provenance = _load_json_object(sidecar, label="FLEURS prediction provenance")
        _validate_completed_provenance(
            provenance,
            prediction=output,
            run=run,
            authorization=authorization,
            run_contract=run_contract,
            num_rows=len(validated),
        )
        if partial.exists() or progress.exists() or recovery.exists():
            raise ExternalFleursError(
                f"Completed and partial artifacts coexist: {output}"
            )
        return output
    if output.exists():
        if not resume or not partial.exists():
            raise ExternalFleursError(
                f"Prediction exists without completed provenance: {output}"
            )
        output_rows = _validate_prediction_prefix(
            load_prediction_csv(output),
            manifest_rows,
            run,
            source=output,
            require_complete=True,
        )
        partial_rows = _validate_prediction_prefix(
            load_prediction_csv(partial),
            manifest_rows,
            run,
            source=partial,
            require_complete=True,
        )
        if sha256_file(output) != sha256_file(partial):
            raise ExternalFleursError(
                "Unpublished prediction and its recovery partial disagree"
            )
        state = _load_json_object(progress, label="FLEURS resume state")
        _validate_resume_state(
            state,
            partial=partial,
            rows=len(partial_rows),
            run_contract=run_contract,
        )
        _atomic_write_new_json(
            sidecar,
            _prediction_provenance(
                prediction=output,
                run=run,
                authorization=authorization,
                run_contract=run_contract,
                num_rows=len(output_rows),
            ),
        )
        partial.unlink()
        progress.unlink()
        return output
    if sidecar.exists():
        raise ExternalFleursError(f"Provenance exists without prediction: {sidecar}")
    if partial.exists() and not resume:
        raise FileExistsError(
            f"Partial prediction exists: {partial}; pass --resume after review"
        )

    prediction_rows: list[dict[str, str]] = []
    if partial.exists() and resume:
        partial_rows = load_prediction_csv(partial)
        prediction_rows = _validate_prediction_prefix(
            partial_rows,
            manifest_rows,
            run,
            source=partial,
            require_complete=False,
        )
        state = _load_json_object(progress, label="FLEURS resume state")
        _validate_resume_state(
            state,
            partial=partial,
            rows=len(prediction_rows),
            run_contract=run_contract,
        )

    start = len(prediction_rows)
    transcriber: ChunkTranscriber | None = None
    try:
        if start < len(manifest_rows):
            transcriber = transcriber_factory(run, device_arg, max_new_tokens)
        for index in range(start, len(manifest_rows)):
            manifest_row = manifest_rows[index]
            waveform = audio_loader(manifest_row["audio_path"], sample_rate)
            chunks = split_waveform(
                waveform,
                sample_rate=sample_rate,
                max_chunk_seconds=max_chunk_seconds,
            )
            hypothesis = join_chunk_hypotheses(
                [
                    transcriber.transcribe_chunk(chunk)  # type: ignore[union-attr]
                    for chunk in chunks
                ]
            )
            if "\ufffd" in hypothesis:
                warnings.warn(
                    "Raw tokenizer output contains U+FFFD for "
                    f"utt_id={manifest_row['utt_id']}; preserving it in the prediction",
                    UnicodeWarning,
                    stacklevel=2,
                )
            prediction_rows.append(_prediction_row(manifest_row, run, hypothesis))
            if len(prediction_rows) % checkpoint_every == 0:
                _publish_partial_checkpoint(
                    partial=partial,
                    progress=progress,
                    recovery=recovery,
                    prediction_rows=prediction_rows,
                    run_contract=run_contract,
                )

        validated = _validate_prediction_prefix(
            prediction_rows,
            manifest_rows,
            run,
            source=output,
            require_complete=True,
        )
        _publish_partial_checkpoint(
            partial=partial,
            progress=progress,
            recovery=recovery,
            prediction_rows=validated,
            run_contract=run_contract,
        )
        atomic_write_csv(output, validated, CANONICAL_PREDICTION_COLUMNS)
        _atomic_write_new_json(
            sidecar,
            _prediction_provenance(
                prediction=output,
                run=run,
                authorization=authorization,
                run_contract=run_contract,
                num_rows=len(validated),
            ),
        )
        partial.unlink()
        progress.unlink()
        recovery.unlink(missing_ok=True)
        return output
    except Exception:
        if prediction_rows and not recovery.exists():
            _publish_partial_checkpoint(
                partial=partial,
                progress=progress,
                recovery=recovery,
                prediction_rows=prediction_rows,
                run_contract=run_contract,
            )
        raise
    finally:
        if transcriber is not None:
            transcriber.close()


def build_external_results(
    artifacts: Sequence[tuple[ExternalRun, str | Path]],
) -> list[dict[str, object]]:
    """Validate the three role-locked files and calculate aligned_v1 metrics."""

    if len(artifacts) != len(REQUIRED_ROLES):
        raise ExternalFleursError(
            f"External FLEURS results require exactly {len(REQUIRED_ROLES)} runs"
        )
    observed_roles = [run.role for run, _ in artifacts]
    if observed_roles != list(REQUIRED_ROLES):
        raise ExternalFleursError(
            f"External runs must be in decision-role order {REQUIRED_ROLES}, "
            f"found {observed_roles}"
        )

    output: list[dict[str, object]] = []
    paired_identity: list[tuple[str, str]] | None = None
    for run, path in artifacts:
        rows = load_prediction_csv(path)
        expected_metadata = run.run_metadata
        for row_number, row in enumerate(rows, start=2):
            conflicts = [
                field for field, value in expected_metadata.items() if row[field] != value
            ]
            if row["snr"] != "clean" or row["noise_type"] != "clean":
                conflicts.extend(["snr/noise_type"])
            if conflicts:
                raise ExternalFleursError(
                    f"{path}: row {row_number} conflicts with external run: {conflicts}"
                )
        identity = [(row["utt_id"], row["ref"]) for row in rows]
        if paired_identity is None:
            paired_identity = identity
        elif identity != paired_identity:
            raise ExternalFleursError(
                f"{path}: utterance order/reference does not match the paired FLEURS run"
            )

        metrics = compute_aligned_metric_result(
            [row["ref"] for row in rows], [row["hyp"] for row in rows]
        )
        evidence = metrics.to_dict(include_counts=True)
        output.append(
            {
                **expected_metadata,
                "n": len(rows),
                "wer": metrics.wer,
                "cer": metrics.cer,
                "ter": metrics.ter,
                "der": metrics.der,
                "fcer": metrics.fcer,
                "swdr": metrics.swdr,
                "metric_version": METRIC_VERSION,
                **{
                    column: evidence[column]
                    for column in METRIC_EVIDENCE_COLUMNS
                },
            }
        )
    return output


def _existing_result_matches(path: Path, rows: Sequence[Mapping[str, object]]) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
            return False
        existing = list(reader)
    if len(existing) != len(rows):
        return False
    for current, expected in zip(existing, rows):
        for column in RESULT_COLUMNS:
            if current[column] != str(expected[column]):
                return False
    return True


def _canonical_result_csv_bytes(
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(RESULT_COLUMNS),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in RESULT_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def _result_provenance_payload(
    *,
    result_output: Path,
    result_rows: Sequence[Mapping[str, object]],
    artifacts: Sequence[tuple[ExternalRun, Path]],
    authorization: ExternalAuthorization,
    selected_manifest_rows: int,
) -> dict[str, Any]:
    """Build the exact commit marker for a completed FLEURS result CSV."""

    if not _existing_result_matches(result_output, result_rows) or (
        result_output.read_bytes() != _canonical_result_csv_bytes(result_rows)
    ):
        raise ExternalFleursError(
            f"Result CSV differs from the computed decision-locked rows: {result_output}"
        )
    return {
        "provenance_version": RESULT_PROVENANCE_VERSION,
        "evaluation_domain": EVALUATION_DOMAIN,
        "evaluation_scope": (
            "full_fleurs_857"
            if selected_manifest_rows == authorization.expected_rows
            else "deterministic_first_n_smoke"
        ),
        "external_evidence_status": "legacy_exposed_replication",
        "results": _stored_path_reference(
            result_output,
            formal=authorization.formal,
            label="FLEURS results",
        ),
        "results_sha256": sha256_file(result_output),
        "metric_version": METRIC_VERSION,
        "result_columns": list(RESULT_COLUMNS),
        "manifest_sha256": authorization.manifest_sha256,
        "fleurs_preparation_lock_sha256": (
            authorization.fleurs_preparation_lock_sha256
        ),
        "fleurs_preparation_identity_sha256": (
            authorization.fleurs_preparation_identity_sha256
        ),
        "fleurs_dataset_revision": authorization.fleurs_dataset_revision,
        "fleurs_audio_inventory_sha256": authorization.fleurs_audio_inventory_sha256,
        "fleurs_audit_sha256": authorization.fleurs_audit_sha256,
        "registry": _stored_path_reference(
            authorization.registry_path,
            formal=authorization.formal,
            label="run registry",
        ),
        "registry_sha256": authorization.registry_sha256,
        "split_lock_sha256": authorization.split_lock_sha256,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "method_identity_sha256": authorization.method_identity_sha256,
        "method_environment_identity_sha256": (
            authorization.method_environment_identity_sha256
        ),
        "method_source_tree_sha256": authorization.method_source_tree_sha256,
        "method_runtime_verified": authorization.method_runtime_verified,
        "prediction_set_sha256": canonical_sha256(
            [sha256_file(path) for _, path in artifacts]
        ),
        "runs": [
            {
                "configuration_id": run.configuration_id,
                "role": run.role,
                "prediction": _stored_path_reference(
                    path,
                    formal=authorization.formal,
                    label=f"{run.role} prediction",
                ),
                "prediction_sha256": sha256_file(path),
                "provenance_sha256": sha256_file(_provenance_path(path)),
            }
            for run, path in artifacts
        ],
    }


def run_external_suite(
    registry_path: str | Path = DEFAULT_RUN_REGISTRY,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    results_path: str | Path | None = None,
    limit: int | None = None,
    device_arg: str = "auto",
    checkpoint_every: int = 10,
    resume: bool = False,
    transcriber_factory: TranscriberFactory = _default_transcriber_factory,
    audio_loader: AudioLoader = _default_audio_loader,
    authorizer: Callable[[str | Path], ExternalAuthorization] = authorize_external_suite,
    preparation_verifier: PreparationVerifier = verify_fleurs_preparation_lock,
    manifest_loader: Callable[..., list[dict[str, str]]] = load_fleurs_manifest,
    run_builder: Callable[[ExternalAuthorization], tuple[ExternalRun, ...]] = build_external_runs,
) -> tuple[list[Path], Path]:
    """Run the three decision roles and write the replication result table."""

    if limit is not None and limit < 1:
        raise ExternalFleursError("limit must be at least 1")
    # Protocol order is deliberate: decision/role authorization happens before
    # any FLEURS manifest row, checkpoint, processor, or model is opened.
    authorization = authorizer(registry_path)
    if authorization.formal:
        _portable_repo_reference(registry_path, label="run registry")
        output_reference = _portable_repo_reference(
            output_dir,
            label="FLEURS output directory",
        )
        predictions_reference = (
            PurePosixPath(output_reference) / "predictions"
        ).as_posix()
        result_reference = (
            _portable_repo_reference(results_path, label="FLEURS results")
            if results_path is not None
            else (
                PurePosixPath(output_reference) / "external_fleurs_results.csv"
            ).as_posix()
        )
    else:
        output_reference = str(output_dir)
        predictions_reference = str(Path(output_reference) / "predictions")
        result_reference = str(
            Path(results_path)
            if results_path is not None
            else Path(output_reference) / "external_fleurs_results.csv"
        )
    if authorization.formal and not authorization.method_runtime_verified:
        raise ExternalFleursError(
            "Formal FLEURS inference requires current method runtime/environment "
            "verification; post-hoc authorization cannot run a model"
        )
    predictions_dir = _artifact_path(
        predictions_reference,
        formal=authorization.formal,
        label="FLEURS predictions directory",
    )
    result_output = _artifact_path(
        result_reference,
        formal=authorization.formal,
        label="FLEURS results",
    )
    if authorization.fleurs_preparation_lock_path is None:
        raise ExternalFleursError("Authorization has no FLEURS preparation lock")
    try:
        preparation = dict(
            preparation_verifier(
                authorization.fleurs_preparation_lock_path,
                repository_root=ROOT,
                expected_count=authorization.expected_rows,
                verify_artifacts=True,
                verify_audio=True,
            )
        )
    except (FleursPreparationError, FileNotFoundError) as error:
        raise ExternalFleursError(
            f"FLEURS manifest/audio integrity verification failed: {error}"
        ) from error
    _verify_preparation_binding(authorization.registry, preparation)
    if (
        str(preparation.get("preparation_lock_sha256", "")).casefold()
        != authorization.fleurs_preparation_lock_sha256
    ):
        raise ExternalFleursError("FLEURS preparation lock changed after authorization")
    if (
        not authorization.manifest_path.is_file()
        or sha256_file(authorization.manifest_path) != authorization.manifest_sha256
    ):
        raise ExternalFleursError("Registered FLEURS manifest is missing or has changed")
    all_rows = manifest_loader(
        authorization.manifest_path,
        expected_rows=authorization.expected_rows,
    )
    manifest_rows = all_rows[:limit] if limit is not None else all_rows
    runs = run_builder(authorization)
    if [run.role for run in runs] != list(REQUIRED_ROLES):
        raise ExternalFleursError("Run builder returned a non-canonical role set")
    result_sidecar = _result_provenance_path(result_output)
    if result_sidecar.exists() and not result_output.exists():
        raise ExternalFleursError(
            f"Result provenance exists without result CSV: {result_sidecar}"
        )
    if result_output.exists() and not resume:
        raise FileExistsError(
            f"External result already exists: {result_output}; outputs are immutable"
        )

    if not resume:
        for run in runs:
            prediction = predictions_dir / run.prediction_name
            occupied = [
                path
                for path in (
                    prediction,
                    _partial_path(prediction),
                    _resume_path(prediction),
                    _recovery_path(prediction),
                    _provenance_path(prediction),
                )
                if path.exists()
            ]
            if occupied:
                raise FileExistsError(
                    f"Prediction artifacts already exist and are immutable: {occupied}"
                )
    artifacts: list[tuple[ExternalRun, Path]] = []
    for run in runs:
        prediction_reference = (
            (PurePosixPath(predictions_reference) / run.prediction_name).as_posix()
            if authorization.formal
            else str(predictions_dir / run.prediction_name)
        )
        prediction_path = run_external_prediction(
            run,
            manifest_rows,
            prediction_reference,
            authorization=authorization,
            device_arg=device_arg,
            checkpoint_every=checkpoint_every,
            resume=resume,
            transcriber_factory=transcriber_factory,
            audio_loader=audio_loader,
        )
        artifacts.append((run, prediction_path))

    result_rows = build_external_results(artifacts)
    if not result_output.exists():
        atomic_write_csv(result_output, result_rows, RESULT_COLUMNS)
    expected_result_provenance = _result_provenance_payload(
        result_output=result_output,
        result_rows=result_rows,
        artifacts=artifacts,
        authorization=authorization,
        selected_manifest_rows=len(manifest_rows),
    )
    if result_output.exists() and resume:
        if result_sidecar.is_file():
            result_provenance = _load_json_object(
                result_sidecar,
                label="FLEURS result provenance",
            )
            if result_provenance != expected_result_provenance:
                raise ExternalFleursError(
                    "Result provenance differs from the exact resumed result contract"
                )
        else:
            # The CSV is staged first and the provenance is its immutable
            # commit marker.  A crash between those two atomic publications is
            # recoverable only after the CSV is recomputed from all three
            # verified prediction/provenance pairs and matches byte-for-byte.
            _atomic_write_new_json(result_sidecar, expected_result_provenance)
    else:
        _atomic_write_new_json(result_sidecar, expected_result_provenance)
    return [path for _, path in artifacts], result_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the decision-locked ordinary baseline, selected method, and "
            "locked control on the legacy-exposed FLEURS Vietnamese replication."
        )
    )
    parser.add_argument("--run-registry", default=str(DEFAULT_RUN_REGISTRY))
    parser.add_argument(
        "--create-registry",
        action="store_true",
        help="Create the immutable registry, then exit without inference.",
    )
    parser.add_argument(
        "--fleurs-preparation-lock",
        default=None,
        help="Formal 857-row FLEURS preparation lock created by download_fleurs.py.",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--expected-manifest-sha256", default=None)
    parser.add_argument("--split-lock", default=None)
    parser.add_argument("--decision-lock", default=None)
    parser.add_argument(
        "--role-config",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "Config path for a locked semantic role; registry creation requires "
            "one value for each ordinary_baseline/selected_method/locked_control."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--results-path", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Deterministic first-N smoke test."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic-allow-external-paths",
        action="store_true",
        help=(
            "Explicitly retain legacy absolute/external path behavior for a "
            "diagnostic run. Never use this mode for formal paper artifacts."
        ),
    )
    return parser.parse_args(argv)


def _parse_role_configs(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        role = role.strip()
        path = path.strip()
        if not separator or role not in REQUIRED_ROLES or not path or role in parsed:
            raise ExternalFleursError(
                f"Invalid/duplicate --role-config {value!r}; expected ROLE=PATH"
            )
        parsed[role] = Path(path)
    if set(parsed) != set(REQUIRED_ROLES):
        raise ExternalFleursError(
            f"Registry creation requires --role-config for every role: {REQUIRED_ROLES}"
        )
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.create_registry:
        if (
            not args.fleurs_preparation_lock
            or not args.split_lock
            or not args.decision_lock
        ):
            raise ExternalFleursError(
                "--create-registry requires --fleurs-preparation-lock, "
                "--split-lock, and --decision-lock"
            )
        registry = create_run_registry(
            args.run_registry,
            preparation_lock_path=args.fleurs_preparation_lock,
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            split_lock_path=args.split_lock,
            decision_lock_path=args.decision_lock,
            config_paths_by_role=_parse_role_configs(args.role_config),
            formal=not args.diagnostic_allow_external_paths,
        )
        print(f"run_registry={registry}")
        return 0
    if (
        args.fleurs_preparation_lock
        or args.manifest
        or args.expected_manifest_sha256
        or args.split_lock
        or args.decision_lock
        or args.role_config
    ):
        raise ExternalFleursError(
            "Registry-creation arguments require --create-registry"
        )
    if args.limit is not None and Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
        raise ExternalFleursError(
            "A smoke --limit requires a separate --output-dir so partial results "
            "cannot replace the official FLEURS run"
        )
    predictions, result = run_external_suite(
        args.run_registry,
        output_dir=args.output_dir,
        results_path=args.results_path,
        limit=args.limit,
        device_arg=args.device,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        authorizer=lambda path: authorize_external_suite(
            path,
            formal=not args.diagnostic_allow_external_paths,
        ),
    )
    for prediction in predictions:
        print(f"prediction={prediction}")
    print(f"results={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
