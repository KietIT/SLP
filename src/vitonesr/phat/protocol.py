from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROTOCOL_VERSION = "vivos_exposure_aware_split_v2"
DECISION_VERSION = "paper_v2_method_lambda_decision_v3"
TRAINING_CONTRACT_VERSION = "paper_v2_training_contract_v1"
EVALUATION_CONTRACT_VERSION = "paper_v2_evaluation_contract_v2"
DEFAULT_SPLIT_LOCK = Path("outputs/paper_v2/protocol/split_lock.json")
DEFAULT_DECISION_LOCK = Path(
    "outputs/paper_v2/protocol/best_lambda_decision.json"
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class ProtocolIntegrityError(ValueError):
    """Raised when a paper-v2 data, checkpoint, or decision lock is invalid."""


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    text = str(value).strip().casefold()
    return len(text) == 64 and all(character in _HEX_DIGITS for character in text)


def is_immutable_revision(value: object) -> bool:
    text = str(value).strip().casefold()
    return len(text) in {40, 64} and all(
        character in _HEX_DIGITS for character in text
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value is not JSON-contract serializable: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    training.pop("output_dir", None)
    experiment = config.get("experiment", {})
    return _json_safe(
        {
            "contract_version": TRAINING_CONTRACT_VERSION,
            "seed": config.get("seed"),
            "experiment": {
                "method_id": experiment.get("method_id"),
                "train_type": experiment.get("train_type"),
            },
            "model": config.get("model", {}),
            "training": training,
            "data": config.get("data", {}),
            "noise": config.get("noise", {}),
        }
    )


def training_contract_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(training_contract_payload(config))


def evaluation_contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}))
    evaluation.pop("prediction_path", None)
    evaluation.pop("manifest", None)
    model = config.get("model", {})
    return _json_safe(
        {
            "contract_version": EVALUATION_CONTRACT_VERSION,
            "model": {
                "name_or_path": model.get("name_or_path"),
                "revision": model.get("revision"),
                "language": model.get("language"),
                "task": model.get("task"),
            },
            "evaluation": evaluation,
            "effective_audio": {
                "sample_rate": evaluation.get(
                    "sample_rate", config.get("data", {}).get("sample_rate")
                ),
                "max_audio_seconds": evaluation.get("max_audio_seconds", 15.0),
            },
            "decoding": {
                "implementation": "whisper_generate_greedy_v1",
                "max_new_tokens": evaluation.get("max_new_tokens", 128),
                "language": model.get("language", "vi"),
                "task": model.get("task", "transcribe"),
                "do_sample": False,
                "num_beams": 1,
            },
        }
    )


def evaluation_contract_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(evaluation_contract_payload(config))


_NOISY_DEV_EVALUATION_FIELDS = (
    "noisy_dev_lock",
    "expected_noisy_dev_lock_sha256",
    "expected_noise_split_lock_sha256",
    "expected_source_dev_sha256",
)


def source_test_evaluation_contract_payload(
    config: Mapping[str, Any],
    *,
    source_manifest: str,
    source_manifest_sha256: str,
    source_rows: int,
) -> dict[str, Any]:
    """Build the pre-registered source-test inference contract.

    The final robustness benchmark is created only after method selection, so its
    derived manifest cannot be named in the decision artifact.  The decision
    instead locks the complete inference/decoding contract against the sealed
    source-test identity.  A later final-benchmark lock binds the derived
    2,300-row manifest transitively to that source test.

    No source-test file is opened here; only its already-locked identity is used.
    """

    if not str(source_manifest).strip():
        raise ProtocolIntegrityError("Source-test manifest reference is required")
    if not is_sha256(source_manifest_sha256):
        raise ProtocolIntegrityError("Source-test manifest SHA-256 is invalid")
    try:
        row_count = int(source_rows)
    except (TypeError, ValueError) as exc:
        raise ProtocolIntegrityError("Source-test row count is invalid") from exc
    if row_count < 1:
        raise ProtocolIntegrityError("Source-test row count must be positive")

    candidate = deepcopy(dict(config))
    candidate.setdefault("protocol", {})["final_test_unlocked"] = True
    evaluation = candidate.setdefault("evaluation", {})
    for field in _NOISY_DEV_EVALUATION_FIELDS:
        evaluation.pop(field, None)
    evaluation.update(
        {
            "manifest": str(source_manifest),
            "data_split": "test",
            "benchmark_protocol": "locked_vivos",
            "locked_vivos_split": "test_locked",
            "expected_manifest_sha256": str(source_manifest_sha256).casefold(),
            "expected_total_rows": row_count,
        }
    )
    return evaluation_contract_payload(candidate)


def source_test_evaluation_contract_sha256(
    config: Mapping[str, Any],
    *,
    source_manifest: str,
    source_manifest_sha256: str,
    source_rows: int,
) -> str:
    return canonical_sha256(
        source_test_evaluation_contract_payload(
            config,
            source_manifest=source_manifest,
            source_manifest_sha256=source_manifest_sha256,
            source_rows=source_rows,
        )
    )


def selection_rule_sha256(selection: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "contract_version": "paper_v2_selection_rule_v1",
            "selection": selection,
        }
    )


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolIntegrityError(f"{label} is invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ProtocolIntegrityError(f"{label} must be a JSON object: {source}")
    return value


def _artifact_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path.cwd() / path


def load_split_lock(path: str | Path = DEFAULT_SPLIT_LOCK) -> dict[str, Any]:
    lock_path = Path(path)
    lock = _load_json_object(lock_path, label="Split lock")
    if lock.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolIntegrityError(
            f"Unsupported split protocol in {lock_path}: "
            f"{lock.get('protocol_version')!r}"
        )
    if lock.get("dataset") != "vivos":
        raise ProtocolIntegrityError(f"Split lock dataset must be 'vivos': {lock_path}")
    official_test = lock.get("official_test")
    if not isinstance(official_test, Mapping):
        raise ProtocolIntegrityError(f"Split lock has no official_test policy: {lock_path}")
    if (
        official_test.get("status") != "SEALED"
        or official_test.get("selection_eligible") is not False
        or official_test.get("sealed_partition") != "test_locked"
        or official_test.get("legacy_exposed_partition")
        != "test_legacy_exposed"
    ):
        raise ProtocolIntegrityError(
            "Official VIVOS unseen test must remain SEALED, selection-ineligible, "
            f"and exposure-partitioned: {lock_path}"
        )
    splits = lock.get("splits")
    if not isinstance(splits, Mapping):
        raise ProtocolIntegrityError(f"Split lock has no splits mapping: {lock_path}")
    for split_name in ("train", "dev", "test_legacy_exposed", "test_locked"):
        item = splits.get(split_name)
        if not isinstance(item, Mapping):
            raise ProtocolIntegrityError(
                f"Split lock is missing splits.{split_name}: {lock_path}"
            )
        if not is_sha256(item.get("manifest_sha256")):
            raise ProtocolIntegrityError(
                f"Invalid manifest hash for splits.{split_name}: {lock_path}"
            )
        if int(item.get("utterance_count", 0)) < 1:
            raise ProtocolIntegrityError(
                f"Invalid utterance count for splits.{split_name}: {lock_path}"
            )

    exposed_count = int(official_test.get("legacy_exposed_utterance_count", -1))
    unseen_count = int(official_test.get("unseen_locked_utterance_count", -1))
    if exposed_count != int(splits["test_legacy_exposed"]["utterance_count"]):
        raise ProtocolIntegrityError(
            f"Legacy-exposed count does not match its locked split: {lock_path}"
        )
    if unseen_count != int(splits["test_locked"]["utterance_count"]):
        raise ProtocolIntegrityError(
            f"Unseen-test count does not match its locked split: {lock_path}"
        )

    exposure = official_test.get("exposure_evidence")
    if not isinstance(exposure, Mapping):
        raise ProtocolIntegrityError(
            f"Split lock has no legacy exposure evidence: {lock_path}"
        )
    for field in (
        "benchmark_manifest_sha256",
        "source_utt_ids_sha256",
        "registry_sha256",
    ):
        if not is_sha256(exposure.get(field)):
            raise ProtocolIntegrityError(
                f"Invalid official_test.exposure_evidence.{field}: {lock_path}"
            )
    benchmark_path = _artifact_path(exposure.get("benchmark_manifest", ""))
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Legacy exposure benchmark evidence does not exist: {benchmark_path}"
        )
    if sha256_file(benchmark_path) != str(
        exposure["benchmark_manifest_sha256"]
    ).casefold():
        raise ProtocolIntegrityError(
            f"Legacy exposure benchmark SHA-256 does not match the lock: {benchmark_path}"
        )
    with benchmark_path.open("r", encoding="utf-8-sig", newline="") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    if len(benchmark_rows) != int(exposure.get("benchmark_row_count", -1)):
        raise ProtocolIntegrityError(
            f"Legacy exposure benchmark row count does not match the lock: {benchmark_path}"
        )

    registry_path = _artifact_path(exposure.get("registry", ""))
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Legacy exposure registry does not exist: {registry_path}"
        )
    if sha256_file(registry_path) != str(exposure["registry_sha256"]).casefold():
        raise ProtocolIntegrityError(
            f"Legacy exposure registry SHA-256 does not match the lock: {registry_path}"
        )
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    source_ids = [str(row.get("source_utt_id", "")).strip() for row in registry_rows]
    if (
        len(source_ids) != int(exposure.get("source_utterance_count", -1))
        or len(source_ids) != exposed_count
        or not all(source_ids)
        or len(set(source_ids)) != len(source_ids)
        or any(row.get("exposure_status") != "legacy_exposed" for row in registry_rows)
    ):
        raise ProtocolIntegrityError(
            f"Legacy exposure registry inventory does not match the lock: {registry_path}"
        )
    source_ids_payload = "".join(f"{source_id}\n" for source_id in sorted(source_ids))
    if hashlib.sha256(source_ids_payload.encode("utf-8")).hexdigest() != str(
        exposure["source_utt_ids_sha256"]
    ).casefold():
        raise ProtocolIntegrityError(
            f"Legacy exposure source-ID SHA-256 does not match the lock: {registry_path}"
        )

    audit = lock.get("audit")
    if not isinstance(audit, Mapping) or int(audit.get("failed_checks", -1)) != 0:
        raise ProtocolIntegrityError(f"Split lock audit is not clean: {lock_path}")
    if not is_sha256(audit.get("sha256")):
        raise ProtocolIntegrityError(f"Split lock audit hash is invalid: {lock_path}")
    audit_path = _artifact_path(audit.get("path", ""))
    if not audit_path.is_file():
        raise FileNotFoundError(f"Split audit does not exist: {audit_path}")
    if sha256_file(audit_path) != str(audit["sha256"]).casefold():
        raise ProtocolIntegrityError(
            f"Split audit SHA-256 does not match the lock: {audit_path}"
        )
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if len(audit_rows) != int(audit.get("checks", -1)):
        raise ProtocolIntegrityError(
            f"Split audit row count does not match the lock: {audit_path}"
        )
    failed = [row.get("check_id", "") for row in audit_rows if row.get("status") != "PASS"]
    if failed:
        raise ProtocolIntegrityError(
            f"Split audit contains failed checks: {failed[:5]}"
        )
    return lock


def _read_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if manifest_path.suffix.casefold() == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif manifest_path.suffix.casefold() in {".json", ".jsonl"}:
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ProtocolIntegrityError(
            f"Unsupported manifest format for integrity verification: {manifest_path}"
        )
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ProtocolIntegrityError(f"Manifest is empty or malformed: {manifest_path}")
    return rows


def verify_manifest_audio_hashes(
    path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    manifest_path = Path(path)
    manifest_rows = list(rows) if rows is not None else _read_manifest_rows(manifest_path)
    for row_number, row in enumerate(manifest_rows, start=1):
        audio_value = (
            row.get("audio_path")
            or row.get("audio")
            or row.get("noisy_path")
            or row.get("clean_path")
        )
        if not audio_value:
            raise ProtocolIntegrityError(
                f"{manifest_path}: row {row_number} has no audio path"
            )
        expected = str(row.get("audio_sha256", "")).strip().casefold()
        if not is_sha256(expected):
            raise ProtocolIntegrityError(
                f"{manifest_path}: row {row_number} has no valid audio_sha256"
            )
        audio_path = Path(str(audio_value))
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path}: audio does not exist at row {row_number}: {audio_path}"
            )
        actual = sha256_file(audio_path)
        if actual != expected:
            raise ProtocolIntegrityError(
                f"{manifest_path}: audio SHA-256 mismatch at row {row_number}: "
                f"{audio_path}"
            )
    return len(manifest_rows)


def verify_locked_vivos_manifest(
    manifest_path: str | Path,
    *,
    split_name: str,
    split_lock_path: str | Path = DEFAULT_SPLIT_LOCK,
    verify_audio: bool = True,
) -> dict[str, Any]:
    if split_name not in {
        "train",
        "dev",
        "test_legacy_exposed",
        "test_locked",
    }:
        raise ValueError(
            "split_name must be train, dev, test_legacy_exposed, or test_locked"
        )
    lock = load_split_lock(split_lock_path)
    split_item = lock["splits"][split_name]
    actual_hash = sha256_file(manifest_path)
    expected_hash = str(split_item["manifest_sha256"]).casefold()
    if actual_hash != expected_hash:
        raise ProtocolIntegrityError(
            f"{split_name} manifest SHA-256 does not match split lock: {manifest_path}"
        )
    rows = _read_manifest_rows(manifest_path)
    if len(rows) != int(split_item["utterance_count"]):
        raise ProtocolIntegrityError(
            f"{split_name} manifest row count does not match split lock: "
            f"{len(rows)} != {split_item['utterance_count']}"
        )
    expected_declared_split = {
        "test_legacy_exposed": "legacy_exposed",
        "test_locked": "test",
    }.get(split_name, split_name)
    observed_splits = {
        str(row.get("split", "")).strip().casefold() for row in rows
    }
    if observed_splits != {expected_declared_split}:
        raise ProtocolIntegrityError(
            f"{split_name} manifest declares unexpected split values: "
            f"{sorted(observed_splits)}"
        )
    if verify_audio:
        verify_manifest_audio_hashes(manifest_path, rows=rows)
    return {
        "split_name": split_name,
        "manifest_sha256": actual_hash,
        "utterance_count": len(rows),
        "split_lock_sha256": sha256_file(split_lock_path),
        "audio_hashes_verified": bool(verify_audio),
    }


def verify_expected_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int | None = None,
    verify_audio: bool = True,
) -> dict[str, Any]:
    if not is_sha256(expected_sha256):
        raise ProtocolIntegrityError("Configured expected manifest SHA-256 is invalid")
    actual_hash = sha256_file(manifest_path)
    if actual_hash != str(expected_sha256).casefold():
        raise ProtocolIntegrityError(
            f"Evaluation manifest SHA-256 does not match config: {manifest_path}"
        )
    rows = _read_manifest_rows(manifest_path)
    if expected_rows is not None and len(rows) != int(expected_rows):
        raise ProtocolIntegrityError(
            f"Evaluation manifest has {len(rows)} rows, expected {expected_rows}"
        )
    if verify_audio:
        verify_manifest_audio_hashes(manifest_path, rows=rows)
    return {
        "manifest_sha256": actual_hash,
        "utterance_count": len(rows),
        "audio_hashes_verified": bool(verify_audio),
    }


def verify_test_decision_lock(
    *,
    split_lock_path: str | Path = DEFAULT_SPLIT_LOCK,
    decision_lock_path: str | Path = DEFAULT_DECISION_LOCK,
    verify_checkpoints: bool = True,
) -> dict[str, Any]:
    split_lock = load_split_lock(split_lock_path)
    decision_path = Path(decision_lock_path)
    decision = _load_json_object(decision_path, label="Method/lambda decision lock")
    if decision.get("decision_version") != DECISION_VERSION:
        raise ProtocolIntegrityError(
            f"Unsupported decision lock version: {decision_path}"
        )
    if (
        decision.get("status") != "LOCKED"
        or decision.get("selection_complete") is not True
        or decision.get("test_unlocked") is not True
    ):
        raise ProtocolIntegrityError(
            f"Method/lambda decision is not locked for test access: {decision_path}"
        )
    decision_identity = str(decision.get("identity_sha256", "")).casefold()
    identity_payload = dict(decision)
    identity_payload.pop("identity_sha256", None)
    if (
        not is_sha256(decision_identity)
        or canonical_sha256(identity_payload) != decision_identity
    ):
        raise ProtocolIntegrityError(
            f"Method/lambda decision identity is invalid: {decision_path}"
        )
    expected_split_lock_hash = sha256_file(split_lock_path)
    if (
        str(decision.get("split_lock_sha256", "")).casefold()
        != expected_split_lock_hash
    ):
        raise ProtocolIntegrityError(
            f"Decision lock does not bind the current split lock: {decision_path}"
        )
    source_test = split_lock["splits"]["test_locked"]
    try:
        decision_source_rows = int(decision.get("source_test_utterance_count", -1))
    except (TypeError, ValueError) as exc:
        raise ProtocolIntegrityError(
            "Decision source-test utterance count is invalid"
        ) from exc
    if (
        str(decision.get("source_test_manifest", ""))
        != str(source_test["manifest"])
        or str(decision.get("source_test_manifest_sha256", "")).casefold()
        != str(source_test["manifest_sha256"]).casefold()
        or decision_source_rows != int(source_test["utterance_count"])
    ):
        raise ProtocolIntegrityError(
            "Decision does not bind the locked unseen source-test identity"
        )
    for path_field, hash_field, label in (
        ("method_lock", "method_lock_sha256", "method lock"),
        ("noisy_dev_lock", "noisy_dev_lock_sha256", "noisy-dev lock"),
    ):
        artifact_path = _artifact_path(decision.get(path_field, ""))
        expected_hash = str(decision.get(hash_field, "")).casefold()
        if not is_sha256(expected_hash) or not artifact_path.is_file():
            raise ProtocolIntegrityError(
                f"Decision {label} binding is missing: {artifact_path}"
            )
        if sha256_file(artifact_path) != expected_hash:
            raise ProtocolIntegrityError(
                f"Decision {label} SHA-256 mismatch: {artifact_path}"
            )
    if not is_sha256(decision.get("method_identity_sha256")):
        raise ProtocolIntegrityError("Decision has no valid method identity SHA-256")
    if str(decision.get("selection_evaluation_split", "")).casefold() != "dev":
        raise ProtocolIntegrityError(
            f"Decision lock must record dev-only selection: {decision_path}"
        )
    if not is_sha256(decision.get("selection_manifest_sha256")):
        raise ProtocolIntegrityError(
            f"Decision lock has no valid selection manifest hash: {decision_path}"
        )
    if (
        str(decision.get("noisy_dev_manifest_sha256", "")).casefold()
        != str(decision["selection_manifest_sha256"]).casefold()
    ):
        raise ProtocolIntegrityError(
            "Decision selection manifest is not the locked noisy-dev manifest"
        )
    selected_method_id = str(decision.get("selected_method_id", "")).strip()
    if not _IDENTIFIER_RE.fullmatch(selected_method_id):
        raise ProtocolIntegrityError(
            f"Decision lock has no valid selected_method_id: {decision_path}"
        )
    try:
        selected_lambda = float(decision["selected_lambda"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolIntegrityError(
            f"Decision lock has no numeric selected lambda: {decision_path}"
        ) from exc
    if not math.isfinite(selected_lambda) or selected_lambda < 0:
        raise ProtocolIntegrityError(
            f"Decision lock selected lambda is invalid: {decision_path}"
        )
    if decision.get("selection_metric_version") != "aligned_v1":
        raise ProtocolIntegrityError(
            f"Decision lock must bind selection_metric_version=aligned_v1: {decision_path}"
        )
    for field in (
        "selection_results_sha256",
        "selection_rule_sha256",
        "selection_evaluation_contract_sha256",
    ):
        if not is_sha256(decision.get(field)):
            raise ProtocolIntegrityError(
                f"Decision lock has no valid {field}: {decision_path}"
            )
    selection_rule = decision.get("selection_rule")
    if not isinstance(selection_rule, Mapping) or selection_rule_sha256(
        selection_rule
    ) != str(decision["selection_rule_sha256"]).casefold():
        raise ProtocolIntegrityError(
            f"Decision selection rule object/hash mismatch: {decision_path}"
        )
    try:
        evaluated_lambdas = tuple(
            float(value) for value in decision.get("evaluated_lambdas", [])
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolIntegrityError("Decision evaluated_lambdas is invalid") from exc
    expected_lambdas = (0.0, 0.05, 0.1, 0.3, 0.5)
    if evaluated_lambdas != expected_lambdas:
        raise ProtocolIntegrityError(
            "Decision must bind the complete ordered five-lambda screen"
        )
    strategy = str(decision.get("locked_control_strategy", ""))
    if strategy not in {
        "best_eligible_non_selected_tone_aware",
        "fixed_preregistered_tone_aware",
    }:
        raise ProtocolIntegrityError("Decision locked-control strategy is invalid")
    if str(selection_rule.get("locked_control_strategy", "")) != strategy:
        raise ProtocolIntegrityError(
            "Decision locked-control strategy differs from the selection rule"
        )
    results_path = _artifact_path(decision.get("selection_results", ""))
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Decision selection results do not exist: {results_path}"
        )
    if sha256_file(results_path) != str(
        decision["selection_results_sha256"]
    ).casefold():
        raise ProtocolIntegrityError(
            f"Decision selection results SHA-256 mismatch: {results_path}"
        )
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        result_reader = csv.DictReader(handle)
        result_columns = set(result_reader.fieldnames or [])
        required_result_columns = {
            "lambda",
            "evaluation_split",
            "manifest_sha256",
            "metric_version",
            "evaluation_contract_sha256",
            "method_lock_sha256",
            "method_identity_sha256",
            "checkpoint_sha256",
            "training_contract_sha256",
        }
        missing_result_columns = sorted(required_result_columns - result_columns)
        if missing_result_columns:
            raise ProtocolIntegrityError(
                "Decision selection results lack provenance columns: "
                + ", ".join(missing_result_columns)
            )
        result_rows = list(result_reader)
    if not result_rows:
        raise ProtocolIntegrityError("Decision selection results are empty")
    try:
        result_lambdas = {float(row["lambda"]) for row in result_rows}
    except (TypeError, ValueError) as exc:
        raise ProtocolIntegrityError(
            "Decision selection results contain an invalid lambda"
        ) from exc
    if result_lambdas != set(expected_lambdas):
        raise ProtocolIntegrityError(
            "Decision selection results do not contain exactly five lambdas"
        )
    required_result_identities = {
        "evaluation_split": "dev",
        "manifest_sha256": str(decision["selection_manifest_sha256"]).casefold(),
        "metric_version": "aligned_v1",
        "evaluation_contract_sha256": str(
            decision["selection_evaluation_contract_sha256"]
        ).casefold(),
        "method_lock_sha256": str(decision["method_lock_sha256"]).casefold(),
        "method_identity_sha256": str(
            decision["method_identity_sha256"]
        ).casefold(),
    }
    for field, expected_value in required_result_identities.items():
        observed = {str(row.get(field, "")).strip().casefold() for row in result_rows}
        if observed != {expected_value}:
            raise ProtocolIntegrityError(
                f"Decision selection results do not bind one verified {field}"
            )
    allowed_test_contracts = decision.get(
        "allowed_test_evaluation_contract_sha256"
    )
    if not isinstance(allowed_test_contracts, list) or not allowed_test_contracts:
        raise ProtocolIntegrityError(
            f"Decision lock has no allowed test evaluation contracts: {decision_path}"
        )
    if any(not is_sha256(value) for value in allowed_test_contracts):
        raise ProtocolIntegrityError(
            f"Decision lock has an invalid test evaluation contract hash: {decision_path}"
        )

    raw_configurations = decision.get("locked_configurations")
    if not isinstance(raw_configurations, list) or not raw_configurations:
        raise ProtocolIntegrityError(
            f"Decision lock has no locked_configurations: {decision_path}"
        )
    configurations: list[dict[str, Any]] = []
    configuration_ids: set[str] = set()
    checkpoint_hashes: set[str] = set()
    checkpoint_paths: set[str] = set()
    identities: set[tuple[str, str, float, int, str, str]] = set()
    roles = {"ordinary_baseline", "selected_method", "locked_control"}
    for index, raw in enumerate(raw_configurations):
        if not isinstance(raw, Mapping):
            raise ProtocolIntegrityError(
                f"locked_configurations[{index}] must be an object"
            )
        configuration_id = str(raw.get("configuration_id", "")).strip()
        method_id = str(raw.get("method_id", "")).strip()
        train_type = str(raw.get("train_type", "")).strip()
        role = str(raw.get("role", "")).strip()
        checkpoint_path_text = str(raw.get("checkpoint_path", "")).strip()
        backbone = str(raw.get("backbone", "")).strip()
        backbone_revision = str(raw.get("backbone_revision", "")).strip()
        if not _IDENTIFIER_RE.fullmatch(configuration_id):
            raise ProtocolIntegrityError(
                f"Invalid configuration_id at locked_configurations[{index}]"
            )
        if configuration_id in configuration_ids:
            raise ProtocolIntegrityError(
                f"Duplicate locked configuration_id: {configuration_id}"
            )
        if not _IDENTIFIER_RE.fullmatch(method_id) or not _IDENTIFIER_RE.fullmatch(
            train_type
        ):
            raise ProtocolIntegrityError(
                f"Invalid method_id/train_type for configuration {configuration_id}"
            )
        if role not in roles:
            raise ProtocolIntegrityError(
                f"Invalid role for configuration {configuration_id}: {role!r}"
            )
        if not checkpoint_path_text:
            raise ProtocolIntegrityError(
                f"Missing checkpoint_path for configuration {configuration_id}"
            )
        checkpoint_path = _artifact_path(checkpoint_path_text)
        normalized_checkpoint_path = str(checkpoint_path.resolve()).casefold()
        if normalized_checkpoint_path in checkpoint_paths:
            raise ProtocolIntegrityError(
                f"Checkpoint path is assigned more than once: {checkpoint_path}"
            )
        if not backbone or not is_immutable_revision(backbone_revision):
            raise ProtocolIntegrityError(
                f"Invalid backbone identity for configuration {configuration_id}"
            )
        try:
            lambda_value = float(raw["lambda"])
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolIntegrityError(
                f"Invalid lambda/seed for configuration {configuration_id}"
            ) from exc
        if not math.isfinite(lambda_value) or lambda_value < 0 or seed < 0:
            raise ProtocolIntegrityError(
                f"Invalid lambda/seed for configuration {configuration_id}"
            )
        if train_type == "ordinary_lora" and lambda_value != 0.0:
            raise ProtocolIntegrityError(
                f"ordinary_lora must use lambda=0: {configuration_id}"
            )
        for field in (
            "checkpoint_sha256",
            "resolved_config_sha256",
            "training_contract_sha256",
        ):
            if not is_sha256(raw.get(field)):
                raise ProtocolIntegrityError(
                    f"Invalid {field} for configuration {configuration_id}"
                )
        checkpoint_hash = str(raw["checkpoint_sha256"]).casefold()
        if verify_checkpoints:
            try:
                actual_checkpoint_hash = checkpoint_inference_sha256(checkpoint_path)
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise ProtocolIntegrityError(
                    f"Cannot verify checkpoint for configuration {configuration_id}: {exc}"
                ) from exc
            if actual_checkpoint_hash != checkpoint_hash:
                raise ProtocolIntegrityError(
                    f"Checkpoint SHA-256 mismatch for configuration {configuration_id}"
                )
        identity = (
            method_id,
            train_type,
            lambda_value,
            seed,
            backbone,
            backbone_revision.casefold(),
        )
        if checkpoint_hash in checkpoint_hashes:
            raise ProtocolIntegrityError(
                f"Checkpoint hash is assigned more than once: {checkpoint_hash}"
            )
        if identity in identities:
            raise ProtocolIntegrityError(
                f"Duplicate locked configuration identity: {configuration_id}"
            )
        configuration_ids.add(configuration_id)
        checkpoint_hashes.add(checkpoint_hash)
        checkpoint_paths.add(normalized_checkpoint_path)
        identities.add(identity)
        configurations.append(
            {
                "configuration_id": configuration_id,
                "role": role,
                "method_id": method_id,
                "train_type": train_type,
                "lambda": lambda_value,
                "seed": seed,
                "backbone": backbone,
                "backbone_revision": backbone_revision.casefold(),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_path": checkpoint_path_text,
                "resolved_config_sha256": str(
                    raw["resolved_config_sha256"]
                ).casefold(),
                "training_contract_sha256": str(
                    raw["training_contract_sha256"]
                ).casefold(),
            }
        )
    role_counts = {
        role: sum(item["role"] == role for item in configurations) for role in roles
    }
    if len(configurations) != 3 or role_counts != {
        "ordinary_baseline": 1,
        "selected_method": 1,
        "locked_control": 1,
    }:
        raise ProtocolIntegrityError(
            "Decision must contain exactly one ordinary_baseline, "
            "selected_method, and locked_control"
        )
    for item in configurations:
        lambda_rows = [
            row
            for row in result_rows
            if math.isclose(
                float(row["lambda"]),
                item["lambda"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if not lambda_rows or {
            str(row.get("checkpoint_sha256", "")).casefold()
            for row in lambda_rows
        } != {item["checkpoint_sha256"]} or {
            str(row.get("training_contract_sha256", "")).casefold()
            for row in lambda_rows
        } != {item["training_contract_sha256"]}:
            raise ProtocolIntegrityError(
                f"Decision configuration {item['configuration_id']} is not bound "
                "by selection-result provenance"
            )
    selected_entries = [
        item
        for item in configurations
        if item["role"] == "selected_method"
        and item["method_id"] == selected_method_id
        and math.isclose(
            item["lambda"], selected_lambda, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if not selected_entries:
        raise ProtocolIntegrityError(
            "Decision selected method/lambda has no selected_method configuration"
        )
    baseline_entries = [
        item for item in configurations if item["role"] == "ordinary_baseline"
    ]
    if not baseline_entries or any(
        item["method_id"] != "ordinary_lora"
        or item["train_type"] != "ordinary_lora"
        or item["lambda"] != 0.0
        for item in baseline_entries
    ):
        raise ProtocolIntegrityError(
            "Decision must lock at least one valid ordinary-LoRA baseline"
        )
    control_entries = [
        item for item in configurations if item["role"] == "locked_control"
    ]
    try:
        locked_control_lambda = float(decision["locked_control_lambda"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolIntegrityError(
            "Decision lock has no numeric locked-control lambda"
        ) from exc
    control = control_entries[0]
    selected = selected_entries[0]
    if (
        control["method_id"] != selected_method_id
        or control["train_type"] != "tone_aware_lora"
        or control["lambda"] <= 0.0
        or not math.isclose(
            control["lambda"], locked_control_lambda, rel_tol=0.0, abs_tol=1e-12
        )
        or math.isclose(
            control["lambda"], selected["lambda"], rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ProtocolIntegrityError(
            "Decision locked_control must be a distinct positive tone-aware configuration"
        )
    if strategy == "fixed_preregistered_tone_aware":
        try:
            preregistered_control = float(selection_rule["locked_control_lambda"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolIntegrityError(
                "Fixed locked-control policy has no pre-registered lambda"
            ) from exc
        if not math.isclose(
            preregistered_control,
            control["lambda"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ProtocolIntegrityError(
                "Decision control differs from its pre-registered fixed lambda"
            )
    if selected["train_type"] != "tone_aware_lora" or selected["lambda"] <= 0.0:
        raise ProtocolIntegrityError(
            "Decision selected_method must be a positive tone-aware configuration"
        )
    selected_identities = {
        (
            item["train_type"],
            item["backbone"],
            item["backbone_revision"],
            item["lambda"],
        )
        for item in selected_entries
    }
    if len(selected_identities) != 1:
        raise ProtocolIntegrityError(
            "Selected method entries disagree on train type, backbone, revision, or lambda"
        )
    return {
        "decision_lock_sha256": sha256_file(decision_path),
        "split_lock_sha256": expected_split_lock_hash,
        "selection_manifest_sha256": str(
            decision["selection_manifest_sha256"]
        ).casefold(),
        "selected_method_id": selected_method_id,
        "selected_lambda": selected_lambda,
        "locked_control_lambda": locked_control_lambda,
        "locked_control_strategy": strategy,
        "method_lock_sha256": str(decision["method_lock_sha256"]).casefold(),
        "method_identity_sha256": str(
            decision["method_identity_sha256"]
        ).casefold(),
        "noisy_dev_lock_sha256": str(
            decision["noisy_dev_lock_sha256"]
        ).casefold(),
        "selection_results_sha256": str(
            decision["selection_results_sha256"]
        ).casefold(),
        "selection_rule_sha256": str(
            decision["selection_rule_sha256"]
        ).casefold(),
        "selection_evaluation_contract_sha256": str(
            decision["selection_evaluation_contract_sha256"]
        ).casefold(),
        "allowed_test_evaluation_contract_sha256": tuple(
            sorted(str(value).casefold() for value in allowed_test_contracts)
        ),
        "locked_configurations": tuple(configurations),
        "test_manifest_sha256": str(
            split_lock["splits"]["test_locked"]["manifest_sha256"]
        ),
        "test_manifest": str(split_lock["splits"]["test_locked"]["manifest"]),
        "test_utterance_count": int(
            split_lock["splits"]["test_locked"]["utterance_count"]
        ),
    }


def verify_test_configuration_locked(
    decision_integrity: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    checkpoint_identity: Mapping[str, str],
) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    model = config.get("model", {})
    expected = {
        "method_id": str(experiment.get("method_id", "")),
        "train_type": str(experiment.get("train_type", "")),
        "lambda": float(config.get("training", {}).get("lambda_tone", -1)),
        "seed": int(config.get("seed", -1)),
        "backbone": str(model.get("name_or_path", "")),
        "backbone_revision": str(model.get("revision", "")).casefold(),
        "checkpoint_sha256": str(
            checkpoint_identity.get("checkpoint_sha256", "")
        ).casefold(),
        "resolved_config_sha256": str(
            checkpoint_identity.get("resolved_config_sha256", "")
        ).casefold(),
        "training_contract_sha256": str(
            checkpoint_identity.get("training_contract_sha256", "")
        ).casefold(),
    }
    matches = []
    for item in decision_integrity.get("locked_configurations", ()):
        if (
            item["method_id"] == expected["method_id"]
            and item["train_type"] == expected["train_type"]
            and math.isclose(
                item["lambda"], expected["lambda"], rel_tol=0.0, abs_tol=1e-12
            )
            and item["seed"] == expected["seed"]
            and item["backbone"] == expected["backbone"]
            and item["backbone_revision"] == expected["backbone_revision"]
            and item["checkpoint_sha256"] == expected["checkpoint_sha256"]
            and item["resolved_config_sha256"]
            == expected["resolved_config_sha256"]
            and item["training_contract_sha256"]
            == expected["training_contract_sha256"]
        ):
            matches.append(dict(item))
    if len(matches) != 1:
        raise ProtocolIntegrityError(
            "Test configuration is not uniquely bound by the locked decision"
        )
    return matches[0]


def resolve_locked_roles(
    decision_integrity: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the three already-validated decision roles without hardcoded lambdas."""

    required = {"ordinary_baseline", "selected_method", "locked_control"}
    resolved: dict[str, dict[str, Any]] = {}
    configurations = decision_integrity.get("locked_configurations", ())
    if not isinstance(configurations, (list, tuple)):
        raise ProtocolIntegrityError("Decision integrity has no locked configurations")
    for raw in configurations:
        if not isinstance(raw, Mapping):
            raise ProtocolIntegrityError("Decision integrity contains a malformed role")
        role = str(raw.get("role", ""))
        if role not in required or role in resolved:
            raise ProtocolIntegrityError(
                "Decision roles must be unique ordinary_baseline, selected_method, "
                "and locked_control"
            )
        resolved[role] = dict(raw)
    if set(resolved) != required:
        raise ProtocolIntegrityError(
            "Decision roles are incomplete; expected ordinary_baseline, "
            "selected_method, and locked_control"
        )
    return resolved


def _hash_named_files(files: Iterable[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    seen = False
    for relative_name, path in sorted(files, key=lambda item: item[0]):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint fingerprint file is missing: {path}")
        seen = True
        name_bytes = relative_name.replace("\\", "/").encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(bytes.fromhex(sha256_file(path)))
    if not seen:
        raise ProtocolIntegrityError("Checkpoint fingerprint contains no files")
    return digest.hexdigest()


def checkpoint_inference_sha256(checkpoint_root: str | Path) -> str:
    root = Path(checkpoint_root)
    files: list[tuple[str, Path]] = []
    for directory_name in ("adapter", "processor"):
        directory = root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Checkpoint is missing {directory_name}/: {root}"
            )
        files.extend(
            (path.relative_to(root).as_posix(), path)
            for path in directory.rglob("*")
            if path.is_file()
        )
    for filename in ("resolved_config.yaml", "trainer_state.json"):
        files.append((filename, root / filename))
    tone_head = root / "tone_head.pt"
    if tone_head.is_file():
        files.append(("tone_head.pt", tone_head))
    return _hash_named_files(files)


def verify_checkpoint_config(
    checkpoint_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, str]:
    root = Path(checkpoint_root)
    resolved_path = root / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint is missing resolved_config.yaml: {root}"
        )
    with resolved_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle) or {}
    if not isinstance(saved, Mapping):
        raise ProtocolIntegrityError(
            f"Checkpoint resolved config must be a mapping: {resolved_path}"
        )
    comparisons = (
        ("seed", saved.get("seed"), config.get("seed")),
        (
            "experiment.method_id",
            saved.get("experiment", {}).get("method_id"),
            config.get("experiment", {}).get("method_id"),
        ),
        (
            "experiment.train_type",
            saved.get("experiment", {}).get("train_type"),
            config.get("experiment", {}).get("train_type"),
        ),
        (
            "model.name_or_path",
            saved.get("model", {}).get("name_or_path"),
            config.get("model", {}).get("name_or_path"),
        ),
        (
            "model.revision",
            saved.get("model", {}).get("revision"),
            config.get("model", {}).get("revision"),
        ),
        (
            "training.lambda_tone",
            saved.get("training", {}).get("lambda_tone"),
            config.get("training", {}).get("lambda_tone"),
        ),
    )
    mismatches = [
        f"{name}: checkpoint={old!r}, config={new!r}"
        for name, old, new in comparisons
        if old != new
    ]
    if mismatches:
        raise ProtocolIntegrityError(
            "Evaluation checkpoint/config identity mismatch: " + "; ".join(mismatches)
        )
    saved_training_contract = training_contract_sha256(saved)
    current_training_contract = training_contract_sha256(config)
    if saved_training_contract != current_training_contract:
        raise ProtocolIntegrityError(
            "Evaluation checkpoint training contract does not match the current "
            f"config: checkpoint={saved_training_contract}, "
            f"config={current_training_contract}"
        )
    training_scope = str(
        saved.get("training", {}).get("run_scope", "")
    ).strip().casefold()
    if training_scope not in {"formal", "smoke"}:
        raise ProtocolIntegrityError(
            f"Checkpoint has invalid training.run_scope: {training_scope!r}"
        )
    protocol_config = config.get("protocol")
    if not isinstance(protocol_config, Mapping) or not protocol_config.get(
        "split_lock"
    ):
        raise ProtocolIntegrityError(
            "Evaluation config has no protocol.split_lock"
        )
    split_lock_path = Path(str(protocol_config["split_lock"]))
    split_lock = load_split_lock(split_lock_path)
    runtime = saved.get("runtime_protocol")
    if not isinstance(runtime, Mapping):
        raise ProtocolIntegrityError(
            f"Checkpoint has no runtime_protocol integrity record: {resolved_path}"
        )
    expected_runtime = {
        "split_lock_sha256": sha256_file(split_lock_path),
        "train_manifest_sha256": str(
            split_lock["splits"]["train"]["manifest_sha256"]
        ).casefold(),
        "dev_manifest_sha256": str(
            split_lock["splits"]["dev"]["manifest_sha256"]
        ).casefold(),
        "training_contract_sha256": current_training_contract,
        "training_scope": training_scope,
    }
    noise_config = config.get("noise", {})
    if bool(noise_config.get("enable_train_noise", False)):
        noise_manifest_path = Path(str(noise_config.get("noise_manifest", "")))
        expected_runtime.update(
            {
                "noise_enabled": True,
                "noise_manifest_sha256": sha256_file(noise_manifest_path),
            }
        )
    runtime_mismatches = []
    for name, expected in expected_runtime.items():
        observed = runtime.get(name)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = observed == expected
        else:
            matches = str(observed if observed is not None else "").casefold() == str(
                expected
            ).casefold()
        if not matches:
            runtime_mismatches.append(
                f"{name}: checkpoint={observed!r}, expected={expected!r}"
            )
    if runtime.get("audio_hashes_verified") is not True:
        runtime_mismatches.append(
            "audio_hashes_verified: checkpoint did not verify training audio"
        )
    if bool(noise_config.get("enable_train_noise", False)) and runtime.get(
        "noise_audio_paths_verified"
    ) is not True:
        runtime_mismatches.append(
            "noise_audio_paths_verified: checkpoint did not validate noise paths"
        )
    if runtime_mismatches:
        raise ProtocolIntegrityError(
            "Checkpoint training-data integrity mismatch: "
            + "; ".join(runtime_mismatches)
        )
    lambda_tone = float(config["training"]["lambda_tone"])
    if lambda_tone > 0 and not (root / "tone_head.pt").is_file():
        raise FileNotFoundError(
            f"Tone-aware checkpoint is missing tone_head.pt: {root}"
        )
    return {
        "checkpoint_sha256": checkpoint_inference_sha256(root),
        "resolved_config_sha256": sha256_file(resolved_path),
        "training_contract_sha256": saved_training_contract,
        "training_scope": training_scope,
    }


def selected_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "utt_id": str(row.get("utt_id", "")),
            "dataset": str(row.get("dataset", "")),
            "audio_path": str(row.get("audio_path", "")),
            "snr": str(row.get("snr", "")),
            "noise_type": str(row.get("noise_type", "")),
            "ref": str(row.get("ref", "")),
            "evaluation_split": str(row.get("evaluation_split", "")),
        }
        for row in rows
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "DECISION_VERSION",
    "DEFAULT_DECISION_LOCK",
    "DEFAULT_SPLIT_LOCK",
    "EVALUATION_CONTRACT_VERSION",
    "PROTOCOL_VERSION",
    "TRAINING_CONTRACT_VERSION",
    "ProtocolIntegrityError",
    "canonical_sha256",
    "checkpoint_inference_sha256",
    "evaluation_contract_payload",
    "evaluation_contract_sha256",
    "is_immutable_revision",
    "is_sha256",
    "load_split_lock",
    "resolve_locked_roles",
    "selection_rule_sha256",
    "selected_rows_sha256",
    "sha256_file",
    "source_test_evaluation_contract_payload",
    "source_test_evaluation_contract_sha256",
    "training_contract_payload",
    "training_contract_sha256",
    "verify_checkpoint_config",
    "verify_expected_manifest",
    "verify_locked_vivos_manifest",
    "verify_manifest_audio_hashes",
    "verify_test_configuration_locked",
    "verify_test_decision_lock",
]
