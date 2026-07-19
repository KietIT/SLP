"""Run FLEURS under a separately locked inference environment.

The method lock remains immutable training provenance.  Consequently this
wrapper never claims that the current host verified the original training
runtime: every ordinary FLEURS provenance keeps
``method_runtime_verified=false``.  A separate, hash-bound extension records
``inference_runtime_verified=true`` for the current execution environment.

The wrapper reuses the existing FLEURS authorization, manifest, prediction,
metric, and provenance primitives without editing any method-lock source
component.  It adds one inference extension per prediction, one result
extension, and an immutable execution receipt that can be independently
verified on resume or hand-off.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_fleurs import (  # noqa: E402
    FleursPreparationError,
    verify_fleurs_preparation_lock,
)
from scripts.run_external_fleurs import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_REGISTRY,
    REQUIRED_ROLES,
    RESULT_COLUMNS,
    AudioLoader,
    ExternalAuthorization,
    ExternalFleursError,
    ExternalRun,
    PreparationVerifier,
    TranscriberFactory,
    _artifact_path,
    _canonical_result_csv_bytes,
    _default_audio_loader,
    _default_transcriber_factory,
    _load_json_object,
    _partial_path,
    _provenance_path,
    _recovery_path,
    _result_provenance_path,
    _result_provenance_payload,
    _resume_path,
    _stored_path_reference,
    _verify_preparation_binding,
    authorize_external_suite,
    build_external_results,
    build_external_runs,
    load_fleurs_manifest,
    run_external_prediction,
)
from src.vitonesr.inference_runtime import (  # noqa: E402
    InferenceRuntimeLockError,
    verify_inference_runtime_lock,
)
from src.vitonesr.phat.protocol import is_sha256, sha256_file  # noqa: E402
from src.vitonesr.prediction import atomic_write_csv  # noqa: E402


DEFAULT_INFERENCE_RUNTIME_LOCK = Path(
    "outputs/paper_v2/protocol/inference_runtime_lock.json"
)
DEFAULT_TRAINING_ENVIRONMENT_LOCK = Path(
    "outputs/paper_v2/protocol/environment_lock.json"
)
DEFAULT_EXECUTION_RECEIPT = Path(
    "outputs/paper_v2/protocol/fleurs_execution_receipt.json"
)
EXTENSION_VERSION = "paper_v2_fleurs_inference_runtime_extension_v1"
RECEIPT_VERSION = "paper_v2_fleurs_inference_runtime_execution_v2"

BaseAuthorizer = Callable[..., ExternalAuthorization]
RuntimeVerifier = Callable[..., Mapping[str, Any]]
ManifestLoader = Callable[..., list[dict[str, str]]]
RunBuilder = Callable[[ExternalAuthorization], tuple[ExternalRun, ...]]
SuiteRunner = Callable[..., tuple[list[Path], Path]]


def _required_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise ExternalFleursError(f"{label} must be a concrete SHA-256")
    return str(value).strip().casefold()


def _formal_argument(value: str | Path) -> str:
    """Make relative ``Path`` defaults portable without accepting bad strings."""

    return value.as_posix() if isinstance(value, Path) else value


def _verified_runtime_summary(value: Mapping[str, Any]) -> dict[str, str]:
    if value.get("status") != "VERIFIED":
        raise ExternalFleursError("Inference runtime lock was not VERIFIED")
    return {
        "schema_version": str(value.get("schema_version", "")),
        "lock_sha256": _required_sha256(
            value.get("lock_sha256"), label="inference runtime lock SHA-256"
        ),
        "identity_sha256": _required_sha256(
            value.get("identity_sha256"), label="inference runtime identity"
        ),
        "training_environment_sha256": _required_sha256(
            value.get("training_environment_sha256"),
            label="training environment lock SHA-256",
        ),
        "training_environment_identity_sha256": _required_sha256(
            value.get("training_environment_identity_sha256"),
            label="training environment identity",
        ),
        "inference_environment_identity_sha256": _required_sha256(
            value.get("inference_environment_identity_sha256"),
            label="inference environment identity",
        ),
        "source_tree_sha256": _required_sha256(
            value.get("source_tree_sha256"), label="inference source tree"
        ),
    }


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable JSON: {path}")
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
                f"Refusing to overwrite immutable JSON: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_record(path: Path, *, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ExternalFleursError(f"Missing completed {label}: {path}")
    return {
        "path": _stored_path_reference(path, formal=True, label=label),
        "sha256": sha256_file(path),
    }


def _record_path(record: Mapping[str, Any], *, label: str) -> Path:
    path = _artifact_path(record.get("path", ""), formal=True, label=label)
    expected = _required_sha256(record.get("sha256"), label=f"{label} SHA-256")
    if not path.is_file() or sha256_file(path) != expected:
        raise ExternalFleursError(f"{label} is missing or changed: {path}")
    return path


def _inference_extension_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".inference_runtime.json")


def _runtime_binding(
    runtime_lock: Path, runtime: Mapping[str, str]
) -> dict[str, str]:
    return {
        "lock": _stored_path_reference(
            runtime_lock, formal=True, label="inference runtime lock"
        ),
        **dict(runtime),
    }


def _training_binding(
    authorization: ExternalAuthorization,
    training_environment_lock: Path,
    runtime: Mapping[str, str],
) -> dict[str, str]:
    return {
        "environment_lock": _stored_path_reference(
            training_environment_lock,
            formal=True,
            label="training environment lock",
        ),
        "environment_lock_sha256": runtime["training_environment_sha256"],
        "method_environment_identity_sha256": (
            authorization.method_environment_identity_sha256
        ),
        "method_lock_sha256": authorization.method_lock_sha256,
        "method_identity_sha256": authorization.method_identity_sha256,
        "method_source_tree_sha256": authorization.method_source_tree_sha256,
    }


def _extension_payload(
    *,
    artifact_kind: str,
    artifact: Path,
    base_provenance: Path,
    authorization: ExternalAuthorization,
    runtime_lock: Path,
    training_environment_lock: Path,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    artifact_record = _artifact_record(artifact, label=artifact_kind)
    provenance_record = _artifact_record(
        base_provenance, label=f"{artifact_kind} base provenance"
    )
    provenance = _load_json_object(
        base_provenance, label=f"{artifact_kind} base provenance"
    )
    if provenance.get("method_runtime_verified") is not False:
        raise ExternalFleursError(
            f"{artifact_kind} must preserve method_runtime_verified=false"
        )
    claimed_field = (
        "prediction_sha256" if artifact_kind == "prediction" else "results_sha256"
    )
    if (
        str(provenance.get(claimed_field, "")).casefold()
        != artifact_record["sha256"]
    ):
        raise ExternalFleursError(
            f"{artifact_kind} base provenance does not bind the artifact"
        )
    if (
        str(provenance.get("method_environment_identity_sha256", "")).casefold()
        != authorization.method_environment_identity_sha256.casefold()
    ):
        raise ExternalFleursError(
            f"{artifact_kind} base provenance changed the training identity"
        )
    return {
        "extension_version": EXTENSION_VERSION,
        "artifact_kind": artifact_kind,
        "artifact": artifact_record,
        "base_provenance": provenance_record,
        "method_runtime_verified": False,
        "inference_runtime_verified": True,
        "registry": {
            "path": _stored_path_reference(
                authorization.registry_path,
                formal=True,
                label="FLEURS run registry",
            ),
            "sha256": authorization.registry_sha256,
        },
        "training": _training_binding(
            authorization, training_environment_lock, runtime
        ),
        "inference_runtime": _runtime_binding(runtime_lock, runtime),
    }


def _publish_or_verify_extension(
    path: Path, payload: Mapping[str, Any], *, resume: bool
) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"Inference extension already exists; pass --resume: {path}"
            )
        if _load_json_object(path, label="inference-runtime extension") != payload:
            raise ExternalFleursError(
                f"Inference-runtime extension differs from its artifacts: {path}"
            )
        return
    _atomic_write_new_json(path, payload)


def authorize_external_inference_runtime(
    registry_path: str | Path,
    *,
    inference_runtime_lock: str | Path,
    training_environment_lock: str | Path = DEFAULT_TRAINING_ENVIRONMENT_LOCK,
    base_authorizer: BaseAuthorizer = authorize_external_suite,
    runtime_verifier: RuntimeVerifier = verify_inference_runtime_lock,
) -> tuple[ExternalAuthorization, dict[str, str], Path, Path]:
    """Authorize immutable training evidence and the current runtime separately."""

    runtime_lock_path = _artifact_path(
        _formal_argument(inference_runtime_lock),
        formal=True,
        label="inference runtime lock",
    )
    training_path = _artifact_path(
        _formal_argument(training_environment_lock),
        formal=True,
        label="training environment lock",
    )
    authorization = base_authorizer(
        _formal_argument(registry_path),
        formal=True,
        verify_current_method=False,
    )
    if not authorization.formal or authorization.method_runtime_verified:
        raise ExternalFleursError(
            "Base authorization must be formal with method_runtime_verified=false"
        )
    try:
        runtime = _verified_runtime_summary(
            runtime_verifier(
                runtime_lock_path,
                training_path,
                repo_root=ROOT,
                verify_current=True,
            )
        )
    except (InferenceRuntimeLockError, FileNotFoundError, OSError, ValueError) as error:
        raise ExternalFleursError(
            f"Separate inference-runtime verification failed: {error}"
        ) from error
    if (
        runtime["training_environment_identity_sha256"]
        != authorization.method_environment_identity_sha256.casefold()
    ):
        raise ExternalFleursError(
            "Inference lock is bound to a different training environment"
        )
    if runtime["lock_sha256"] != sha256_file(runtime_lock_path):
        raise ExternalFleursError("Runtime verifier returned another lock hash")
    if runtime["training_environment_sha256"] != sha256_file(training_path):
        raise ExternalFleursError(
            "Runtime verifier returned another training-environment hash"
        )
    return authorization, runtime, runtime_lock_path, training_path


def _run_external_suite_separate_runtime(
    registry_path: str | Path,
    *,
    authorization: ExternalAuthorization,
    output_dir: str | Path,
    results_path: str | Path | None,
    limit: int | None,
    device_arg: str,
    checkpoint_every: int,
    resume: bool,
    preparation_verifier: PreparationVerifier = verify_fleurs_preparation_lock,
    manifest_loader: ManifestLoader = load_fleurs_manifest,
    run_builder: RunBuilder = build_external_runs,
    transcriber_factory: TranscriberFactory = _default_transcriber_factory,
    audio_loader: AudioLoader = _default_audio_loader,
) -> tuple[list[Path], Path]:
    """Local orchestration equivalent to the locked suite, without its old gate."""

    if limit is not None and limit < 1:
        raise ExternalFleursError("limit must be at least 1")
    registry_reference = _formal_argument(registry_path)
    output_reference = _formal_argument(output_dir)
    results_reference = (
        _formal_argument(results_path) if results_path is not None else None
    )
    # Validate formal path references before FLEURS rows/audio or models open.
    _artifact_path(registry_reference, formal=True, label="run registry")
    output_root = _artifact_path(
        output_reference, formal=True, label="FLEURS output directory"
    )
    predictions_root = output_root / "predictions"
    result = _artifact_path(
        results_reference
        or (PurePosixPath(output_reference) / "external_fleurs_results.csv").as_posix(),
        formal=True,
        label="FLEURS results",
    )
    if authorization.method_runtime_verified:
        raise ExternalFleursError(
            "Separate-runtime suite refuses method_runtime_verified=true"
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
        raise ExternalFleursError("FLEURS preparation changed after authorization")
    if (
        not authorization.manifest_path.is_file()
        or sha256_file(authorization.manifest_path) != authorization.manifest_sha256
    ):
        raise ExternalFleursError("FLEURS manifest is missing or changed")
    rows = manifest_loader(
        authorization.manifest_path, expected_rows=authorization.expected_rows
    )
    selected_rows = rows[:limit] if limit is not None else rows
    runs = run_builder(authorization)
    if [run.role for run in runs] != list(REQUIRED_ROLES):
        raise ExternalFleursError("Run builder returned a non-canonical role set")

    result_provenance = _result_provenance_path(result)
    if result_provenance.exists() and not result.exists():
        raise ExternalFleursError("Result provenance exists without result CSV")
    if result.exists() and not resume:
        raise FileExistsError(f"External result already exists: {result}")
    if _inference_extension_path(result).exists() and not resume:
        raise FileExistsError(
            "FLEURS result inference extension already exists; pass --resume: "
            f"{_inference_extension_path(result)}"
        )
    if not resume:
        occupied: list[Path] = []
        for run in runs:
            prediction = predictions_root / run.prediction_name
            occupied.extend(
                path
                for path in (
                    prediction,
                    _partial_path(prediction),
                    _resume_path(prediction),
                    _recovery_path(prediction),
                    _provenance_path(prediction),
                    _inference_extension_path(prediction),
                )
                if path.exists()
            )
        if occupied:
            raise FileExistsError(
                f"FLEURS prediction artifacts already exist: {occupied}"
            )

    artifacts: list[tuple[ExternalRun, Path]] = []
    for run in runs:
        prediction_reference = (
            PurePosixPath(output_reference)
            / "predictions"
            / run.prediction_name
        ).as_posix()
        prediction = run_external_prediction(
            run,
            selected_rows,
            prediction_reference,
            authorization=authorization,
            device_arg=device_arg,
            checkpoint_every=checkpoint_every,
            resume=resume,
            transcriber_factory=transcriber_factory,
            audio_loader=audio_loader,
        )
        artifacts.append((run, prediction))

    result_rows = build_external_results(artifacts)
    if not result.exists():
        atomic_write_csv(result, result_rows, RESULT_COLUMNS)
    expected_provenance = _result_provenance_payload(
        result_output=result,
        result_rows=result_rows,
        artifacts=artifacts,
        authorization=authorization,
        selected_manifest_rows=len(selected_rows),
    )
    if result_provenance.exists():
        if not resume:
            raise FileExistsError(
                f"FLEURS result provenance already exists: {result_provenance}"
            )
        if (
            _load_json_object(result_provenance, label="FLEURS result provenance")
            != expected_provenance
        ):
            raise ExternalFleursError(
                "FLEURS result provenance differs from the resumed result"
            )
    else:
        # Recover only the narrow atomic gap after the exact canonical CSV.
        if result.read_bytes() != _canonical_result_csv_bytes(result_rows):
            raise ExternalFleursError(
                "FLEURS result differs from the exact computed canonical CSV"
            )
        _atomic_write_new_json(result_provenance, expected_provenance)
    return [path for _, path in artifacts], result


def _build_execution_receipt(
    *,
    authorization: ExternalAuthorization,
    runtime: Mapping[str, str],
    runtime_lock: Path,
    training_lock: Path,
    predictions: Sequence[Path],
    result: Path,
) -> dict[str, Any]:
    if len(predictions) != len(REQUIRED_ROLES):
        raise ExternalFleursError("Execution receipt requires three predictions")
    prediction_records: list[dict[str, Any]] = []
    for role, prediction in zip(REQUIRED_ROLES, predictions):
        provenance = _provenance_path(prediction)
        extension = _inference_extension_path(prediction)
        provenance_payload = _load_json_object(
            provenance, label="FLEURS prediction provenance"
        )
        if provenance_payload.get("role") != role:
            raise ExternalFleursError("Prediction order differs from locked roles")
        prediction_records.append(
            {
                "role": role,
                "configuration_id": str(
                    provenance_payload.get("configuration_id", "")
                ),
                "prediction": _artifact_record(prediction, label="prediction"),
                "provenance": _artifact_record(
                    provenance, label="prediction provenance"
                ),
                "inference_extension": _artifact_record(
                    extension, label="prediction inference extension"
                ),
            }
        )
    result_provenance = _result_provenance_path(result)
    result_extension = _inference_extension_path(result)
    return {
        "receipt_version": RECEIPT_VERSION,
        "method_runtime_verified": False,
        "inference_runtime_verified": True,
        "semantic_separation": {
            "training_environment": "immutable_checkpoint_provenance",
            "inference_runtime": "separately_verified_execution_environment",
            "method_lock_source_components_modified": False,
        },
        "inference_runtime": _runtime_binding(runtime_lock, runtime),
        "training": _training_binding(authorization, training_lock, runtime),
        "registry": {
            "path": _stored_path_reference(
                authorization.registry_path,
                formal=True,
                label="FLEURS run registry",
            ),
            "sha256": authorization.registry_sha256,
        },
        "predictions": prediction_records,
        "aggregate": {
            "result": _artifact_record(result, label="FLEURS result"),
            "provenance": _artifact_record(
                result_provenance, label="FLEURS result provenance"
            ),
            "inference_extension": _artifact_record(
                result_extension, label="FLEURS result inference extension"
            ),
        },
    }


def _validate_extension(
    extension_record: Mapping[str, Any],
    *,
    artifact_record: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    training_binding: Mapping[str, Any],
    registry_binding: Mapping[str, Any],
    artifact_kind: str,
) -> None:
    extension_path = _record_path(
        extension_record, label="inference-runtime extension"
    )
    value = _load_json_object(extension_path, label="inference-runtime extension")
    if (
        value.get("extension_version") != EXTENSION_VERSION
        or value.get("artifact_kind") != artifact_kind
        or value.get("method_runtime_verified") is not False
        or value.get("inference_runtime_verified") is not True
        or value.get("artifact") != artifact_record
        or value.get("base_provenance") != provenance_record
        or value.get("inference_runtime") != runtime_binding
        or value.get("training") != training_binding
        or value.get("registry") != registry_binding
    ):
        raise ExternalFleursError(
            f"Invalid inference-runtime extension: {extension_path}"
        )


def verify_execution_receipt(
    receipt_path: str | Path = DEFAULT_EXECUTION_RECEIPT,
    *,
    inference_runtime_lock: str | Path = DEFAULT_INFERENCE_RUNTIME_LOCK,
    training_environment_lock: str | Path = DEFAULT_TRAINING_ENVIRONMENT_LOCK,
    runtime_verifier: RuntimeVerifier = verify_inference_runtime_lock,
) -> dict[str, Any]:
    """Verify receipt hashes and the current separately locked runtime."""

    receipt = _artifact_path(
        _formal_argument(receipt_path), formal=True, label="FLEURS execution receipt"
    )
    value = _load_json_object(receipt, label="FLEURS execution receipt")
    if (
        value.get("receipt_version") != RECEIPT_VERSION
        or value.get("method_runtime_verified") is not False
        or value.get("inference_runtime_verified") is not True
    ):
        raise ExternalFleursError("Unsupported or semantically invalid receipt")
    runtime_lock = _artifact_path(
        _formal_argument(inference_runtime_lock),
        formal=True,
        label="inference runtime lock",
    )
    training_lock = _artifact_path(
        _formal_argument(training_environment_lock),
        formal=True,
        label="training environment lock",
    )
    runtime = _verified_runtime_summary(
        runtime_verifier(
            runtime_lock,
            training_lock,
            repo_root=ROOT,
            verify_current=True,
        )
    )
    runtime_binding = _runtime_binding(runtime_lock, runtime)
    if value.get("inference_runtime") != runtime_binding:
        raise ExternalFleursError("Receipt differs from the current inference lock")
    training_binding = value.get("training")
    registry_binding = value.get("registry")
    if not isinstance(training_binding, Mapping) or not isinstance(
        registry_binding, Mapping
    ):
        raise ExternalFleursError("Receipt lacks training or registry binding")
    if (
        training_binding.get("environment_lock_sha256")
        != runtime["training_environment_sha256"]
        or training_binding.get("method_environment_identity_sha256")
        != runtime["training_environment_identity_sha256"]
    ):
        raise ExternalFleursError("Receipt training identity differs from runtime lock")
    expected_training_reference = _stored_path_reference(
        training_lock, formal=True, label="training environment lock"
    )
    if training_binding.get("environment_lock") != expected_training_reference:
        raise ExternalFleursError("Receipt points at another training environment lock")
    for field in (
        "method_environment_identity_sha256",
        "method_lock_sha256",
        "method_identity_sha256",
        "method_source_tree_sha256",
    ):
        _required_sha256(training_binding.get(field), label=f"training.{field}")
    _record_path(registry_binding, label="FLEURS run registry")

    predictions = value.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(REQUIRED_ROLES):
        raise ExternalFleursError("Receipt must bind exactly three predictions")
    for expected_role, entry in zip(REQUIRED_ROLES, predictions):
        if not isinstance(entry, Mapping) or entry.get("role") != expected_role:
            raise ExternalFleursError("Receipt prediction roles are not canonical")
        artifact_record = entry.get("prediction")
        provenance_record = entry.get("provenance")
        extension_record = entry.get("inference_extension")
        if not all(
            isinstance(item, Mapping)
            for item in (artifact_record, provenance_record, extension_record)
        ):
            raise ExternalFleursError("Receipt prediction binding is incomplete")
        prediction = _record_path(artifact_record, label="FLEURS prediction")
        provenance = _record_path(
            provenance_record, label="FLEURS prediction provenance"
        )
        base = _load_json_object(provenance, label="FLEURS prediction provenance")
        if (
            base.get("method_runtime_verified") is not False
            or base.get("role") != expected_role
            or str(base.get("prediction_sha256", "")).casefold()
            != sha256_file(prediction)
            or str(base.get("method_environment_identity_sha256", "")).casefold()
            != str(
                training_binding.get("method_environment_identity_sha256", "")
            ).casefold()
        ):
            raise ExternalFleursError("Invalid base prediction provenance semantics")
        _validate_extension(
            extension_record,
            artifact_record=artifact_record,
            provenance_record=provenance_record,
            runtime_binding=runtime_binding,
            training_binding=training_binding,
            registry_binding=registry_binding,
            artifact_kind="prediction",
        )

    aggregate = value.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ExternalFleursError("Receipt lacks aggregate binding")
    result_record = aggregate.get("result")
    provenance_record = aggregate.get("provenance")
    extension_record = aggregate.get("inference_extension")
    if not all(
        isinstance(item, Mapping)
        for item in (result_record, provenance_record, extension_record)
    ):
        raise ExternalFleursError("Receipt aggregate binding is incomplete")
    result = _record_path(result_record, label="FLEURS result")
    provenance = _record_path(provenance_record, label="FLEURS result provenance")
    base = _load_json_object(provenance, label="FLEURS result provenance")
    if (
        base.get("method_runtime_verified") is not False
        or str(base.get("results_sha256", "")).casefold() != sha256_file(result)
        or str(base.get("method_environment_identity_sha256", "")).casefold()
        != str(
            training_binding.get("method_environment_identity_sha256", "")
        ).casefold()
    ):
        raise ExternalFleursError("Invalid base result provenance semantics")
    _validate_extension(
        extension_record,
        artifact_record=result_record,
        provenance_record=provenance_record,
        runtime_binding=runtime_binding,
        training_binding=training_binding,
        registry_binding=registry_binding,
        artifact_kind="result",
    )
    return {
        "status": "VERIFIED",
        "receipt_sha256": sha256_file(receipt),
        "inference_runtime_identity_sha256": runtime["identity_sha256"],
        "training_environment_identity_sha256": runtime[
            "training_environment_identity_sha256"
        ],
        "prediction_count": len(predictions),
    }


def run_external_fleurs_inference_runtime(
    registry_path: str | Path = DEFAULT_RUN_REGISTRY,
    *,
    inference_runtime_lock: str | Path = DEFAULT_INFERENCE_RUNTIME_LOCK,
    training_environment_lock: str | Path = DEFAULT_TRAINING_ENVIRONMENT_LOCK,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    results_path: str | Path | None = None,
    receipt_path: str | Path = DEFAULT_EXECUTION_RECEIPT,
    limit: int | None = None,
    device_arg: str = "auto",
    checkpoint_every: int = 10,
    resume: bool = False,
    base_authorizer: BaseAuthorizer = authorize_external_suite,
    runtime_verifier: RuntimeVerifier = verify_inference_runtime_lock,
    suite_runner: SuiteRunner = _run_external_suite_separate_runtime,
    **suite_dependencies: Any,
) -> tuple[list[Path], Path, Path]:
    """Execute FLEURS and publish separate inference-runtime evidence."""

    receipt = _artifact_path(
        _formal_argument(receipt_path), formal=True, label="FLEURS execution receipt"
    )
    if receipt.exists() and not resume:
        raise FileExistsError(
            f"FLEURS execution receipt already exists; pass --resume: {receipt}"
        )
    authorization, runtime, runtime_lock, training_lock = (
        authorize_external_inference_runtime(
            registry_path,
            inference_runtime_lock=inference_runtime_lock,
            training_environment_lock=training_environment_lock,
            base_authorizer=base_authorizer,
            runtime_verifier=runtime_verifier,
        )
    )
    predictions, result = suite_runner(
        _formal_argument(registry_path),
        authorization=authorization,
        output_dir=_formal_argument(output_dir),
        results_path=(
            _formal_argument(results_path) if results_path is not None else None
        ),
        limit=limit,
        device_arg=device_arg,
        checkpoint_every=checkpoint_every,
        resume=resume,
        **suite_dependencies,
    )

    for prediction in predictions:
        extension = _inference_extension_path(prediction)
        payload = _extension_payload(
            artifact_kind="prediction",
            artifact=prediction,
            base_provenance=_provenance_path(prediction),
            authorization=authorization,
            runtime_lock=runtime_lock,
            training_environment_lock=training_lock,
            runtime=runtime,
        )
        _publish_or_verify_extension(extension, payload, resume=resume)
    result_extension = _inference_extension_path(result)
    _publish_or_verify_extension(
        result_extension,
        _extension_payload(
            artifact_kind="result",
            artifact=result,
            base_provenance=_result_provenance_path(result),
            authorization=authorization,
            runtime_lock=runtime_lock,
            training_environment_lock=training_lock,
            runtime=runtime,
        ),
        resume=resume,
    )
    expected_receipt = _build_execution_receipt(
        authorization=authorization,
        runtime=runtime,
        runtime_lock=runtime_lock,
        training_lock=training_lock,
        predictions=predictions,
        result=result,
    )
    if receipt.exists():
        if _load_json_object(receipt, label="FLEURS execution receipt") != expected_receipt:
            raise ExternalFleursError(
                "Existing FLEURS execution receipt differs from completed artifacts"
            )
    else:
        _atomic_write_new_json(receipt, expected_receipt)
    verify_execution_receipt(
        _formal_argument(receipt_path),
        inference_runtime_lock=_formal_argument(inference_runtime_lock),
        training_environment_lock=_formal_argument(training_environment_lock),
        runtime_verifier=runtime_verifier,
    )
    return list(predictions), result, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run formal FLEURS with separate inference-runtime evidence while "
            "preserving original training provenance."
        )
    )
    parser.add_argument("--run-registry", default=DEFAULT_RUN_REGISTRY.as_posix())
    parser.add_argument(
        "--inference-runtime-lock",
        default=DEFAULT_INFERENCE_RUNTIME_LOCK.as_posix(),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--receipt", default=DEFAULT_EXECUTION_RECEIPT.as_posix())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verify-receipt-only",
        action="store_true",
        help="Verify completed hashes/current runtime without running a model.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_receipt_only:
        verified = verify_execution_receipt(
            args.receipt,
            inference_runtime_lock=args.inference_runtime_lock,
        )
        print(json.dumps(verified, ensure_ascii=False, sort_keys=True))
        return 0
    if args.limit is not None and Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
        raise ExternalFleursError(
            "A smoke --limit requires a separate --output-dir"
        )
    predictions, result, receipt = run_external_fleurs_inference_runtime(
        args.run_registry,
        inference_runtime_lock=args.inference_runtime_lock,
        output_dir=args.output_dir,
        results_path=args.results_path,
        receipt_path=args.receipt,
        limit=args.limit,
        device_arg=args.device,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    for prediction in predictions:
        print(f"prediction={prediction}")
    print(f"results={result}")
    print(f"execution_receipt={receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
