"""Deterministic, scope-aware comparison of legacy and paper-v2 artifacts.

This module deliberately withholds numeric deltas when two values were
measured on different utterance/noise scopes.  In particular, the historical
VIVOS benchmark used 300 now-exposed official-test utterances, whereas the
paper-v2 final benchmark uses the disjoint 460-utterance locked complement.
FLEURS deltas are emitted only after row-level ``utt_id``/``ref`` identity has
been proved from the prediction artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


COMPARISON_VERSION = "paper_v2_old_new_comparison_v1"
COMPARISON_BUNDLE_VERSION = "paper_v2_comparison_bundle_v1"
METRICS = ("wer", "cer", "ter", "der", "fcer", "swdr")
BOOTSTRAP_METRICS = ("wer", "cer", "ter", "der")
ROLES = ("ordinary_baseline", "selected_method", "locked_control")

COMPARISON_COLUMNS = (
    "section",
    "artifact",
    "old_scope",
    "new_scope",
    "run_identity",
    "metric",
    "statistic",
    "old_value",
    "new_value",
    "delta_new_minus_old",
    "comparability",
    "reason",
    "old_row_count",
    "new_row_count",
    "old_sample_count",
    "new_sample_count",
    "old_manifest_sha256",
    "new_manifest_sha256",
    "old_protocol_label",
    "new_protocol_label",
    "old_artifact_sha256",
    "new_artifact_sha256",
    "decision_lock_sha256",
)


class ComparisonError(ValueError):
    """Raised when a comparison would be incomplete or scientifically unsafe."""


@dataclass(frozen=True, slots=True)
class ComparisonInputs:
    repo_root: Path
    old_by_snr: Path
    old_by_noise_type: Path
    old_fleurs_results: Path
    old_fleurs_bootstrap: Path
    old_benchmark_manifest: Path
    old_fleurs_predictions_dir: Path
    fleurs_manifest: Path
    fleurs_preparation_lock: Path
    new_by_snr: Path
    new_by_noise_type: Path
    new_fleurs_results: Path
    new_fleurs_provenance: Path
    new_fleurs_bootstrap: Path
    new_final_bootstrap: Path
    new_benchmark_manifest: Path
    decision_lock: Path
    split_lock: Path
    noise_split_lock: Path
    noisy_dev_lock: Path
    environment_lock: Path
    method_lock: Path
    final_benchmark_lock: Path

    def files(self) -> dict[str, Path]:
        return {
            "old_by_snr": self.old_by_snr,
            "old_by_noise_type": self.old_by_noise_type,
            "old_fleurs_results": self.old_fleurs_results,
            "old_fleurs_bootstrap": self.old_fleurs_bootstrap,
            "old_benchmark_manifest": self.old_benchmark_manifest,
            "fleurs_manifest": self.fleurs_manifest,
            "fleurs_preparation_lock": self.fleurs_preparation_lock,
            "new_by_snr": self.new_by_snr,
            "new_by_noise_type": self.new_by_noise_type,
            "new_fleurs_results": self.new_fleurs_results,
            "new_fleurs_provenance": self.new_fleurs_provenance,
            "new_fleurs_bootstrap": self.new_fleurs_bootstrap,
            "new_final_bootstrap": self.new_final_bootstrap,
            "new_benchmark_manifest": self.new_benchmark_manifest,
            "decision_lock": self.decision_lock,
            "split_lock": self.split_lock,
            "noise_split_lock": self.noise_split_lock,
            "noisy_dev_lock": self.noisy_dev_lock,
            "environment_lock": self.environment_lock,
            "method_lock": self.method_lock,
            "final_benchmark_lock": self.final_benchmark_lock,
        }


@dataclass(frozen=True, slots=True)
class ComparisonBundle:
    rows: tuple[dict[str, str], ...]
    markdown: str
    provenance: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComparisonError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must be a JSON object: {path}")
    return value


def _load_csv(path: Path, *, required: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing = [field for field in required if field not in fields]
            if missing:
                raise ComparisonError(f"{label} lacks columns {missing}: {path}")
            rows = [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error) as error:
        raise ComparisonError(f"cannot read {label}: {path}") from error
    if not rows:
        raise ComparisonError(f"{label} is empty: {path}")
    return rows


def _canonical_lambda(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError as error:
        raise ComparisonError(f"invalid lambda value: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ComparisonError(f"invalid lambda value: {value!r}")
    return str(int(number)) if number.is_integer() else format(number, ".15g")


def _float(value: object, *, label: str) -> float:
    try:
        number = float(str(value))
    except ValueError as error:
        raise ComparisonError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ComparisonError(f"{label} is not finite: {value!r}")
    return number


def _number_text(value: float) -> str:
    if value == 0.0:
        return "0"
    return format(value, ".15g")


def _int_text(value: object, *, label: str) -> str:
    number = _float(value, label=label)
    if not number.is_integer() or number < 0:
        raise ComparisonError(f"{label} must be a non-negative integer")
    return str(int(number))


def _run_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return (
        row.get("dataset", "").strip().casefold(),
        row.get("model", "").strip().casefold(),
        row.get("model_size", "").strip().casefold(),
        row.get("train_type", "").strip().casefold(),
        _canonical_lambda(row.get("lambda", "")),
        row.get("seed", "").strip(),
    )


def _run_identity(key: Sequence[str], *, role: str = "") -> str:
    names = ("dataset", "model", "model_size", "train_type", "lambda", "seed")
    pieces = [f"{name}={value or '<blank>'}" for name, value in zip(names, key)]
    if role:
        pieces.insert(0, f"role={role}")
    return ";".join(pieces)


def _new_row(**values: object) -> dict[str, str]:
    row = {column: "" for column in COMPARISON_COLUMNS}
    unknown = set(values) - set(row)
    if unknown:
        raise AssertionError(f"unknown comparison columns: {sorted(unknown)}")
    for key, value in values.items():
        row[key] = "" if value is None else str(value)
    return row


def _decision_roles(decision: Mapping[str, Any]) -> tuple[dict[str, dict[str, str]], str]:
    if (
        decision.get("status") != "LOCKED"
        or decision.get("selection_complete") is not True
        or decision.get("test_unlocked") is not True
    ):
        raise ComparisonError("decision lock is not a complete LOCKED decision")
    raw = decision.get("locked_configurations")
    if not isinstance(raw, list):
        raise ComparisonError("decision lock has no locked_configurations list")
    roles: dict[str, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ComparisonError("decision locked configuration must be an object")
        role = str(item.get("role", "")).strip()
        if role in roles or role not in ROLES:
            raise ComparisonError(f"invalid/duplicate decision role: {role!r}")
        config_id = str(item.get("configuration_id", "")).strip()
        if not config_id:
            raise ComparisonError(f"decision role {role} lacks configuration_id")
        roles[role] = {
            "configuration_id": config_id,
            "role": role,
            "method_id": str(item.get("method_id", "")).strip(),
            "train_type": str(item.get("train_type", "")).strip().casefold(),
            "lambda": _canonical_lambda(item.get("lambda", "")),
            "seed": str(item.get("seed", "")).strip(),
        }
    if set(roles) != set(ROLES):
        raise ComparisonError(f"decision must contain exactly roles {list(ROLES)}")
    identity = str(decision.get("identity_sha256", "")).strip().casefold()
    if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ComparisonError("decision identity_sha256 is missing/invalid")
    identity_payload = dict(decision)
    identity_payload.pop("identity_sha256", None)
    if _canonical_sha256(identity_payload) != identity:
        raise ComparisonError("decision identity_sha256 does not match its content")
    return roles, identity


def _role_for_key(key: Sequence[str], roles: Mapping[str, Mapping[str, str]]) -> str:
    for role, config in roles.items():
        if (
            key[3] == config["train_type"]
            and key[4] == config["lambda"]
            and key[5] == config["seed"]
        ):
            return role
    return ""


def _count_records(path: Path) -> int:
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for row in reader if row)
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise ComparisonError(f"invalid JSONL at {path}:{line_number}") from error
            count += 1
    return count


def _aggregate_map(
    rows: Sequence[Mapping[str, str]], *, group_column: str, label: str
) -> dict[tuple[str, ...], Mapping[str, str]]:
    result: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = (*_run_key(row), row.get(group_column, "").strip().casefold())
        if not key[-1]:
            raise ComparisonError(f"{label} has a blank {group_column}")
        if key in result:
            raise ComparisonError(f"{label} has duplicate aggregate key: {key}")
        _int_text(row.get("n", ""), label=f"{label}.n")
        for metric in METRICS:
            _float(row.get(metric, ""), label=f"{label}.{metric}")
        result[key] = row
    return result


def _aggregate_rows(
    *,
    artifact: str,
    group_column: str,
    old_rows: Sequence[Mapping[str, str]],
    new_rows: Sequence[Mapping[str, str]],
    roles: Mapping[str, Mapping[str, str]],
    old_hash: str,
    new_hash: str,
    old_manifest_hash: str,
    new_manifest_hash: str,
    decision_hash: str,
) -> list[dict[str, str]]:
    old_map = _aggregate_map(old_rows, group_column=group_column, label=f"old {artifact}")
    new_map = _aggregate_map(new_rows, group_column=group_column, label=f"new {artifact}")
    output: list[dict[str, str]] = []
    for key in sorted(set(old_map) | set(new_map)):
        old = old_map.get(key)
        new = new_map.get(key)
        role = _role_for_key(key[:6], roles)
        identity = f"{_run_identity(key[:6], role=role)};{group_column}={key[6]}"
        if old is not None and new is not None:
            comparability = "not_comparable_scope"
            reason = (
                "delta withheld: legacy VIVOS uses 300 exposed source utterances; "
                "paper-v2 uses the disjoint 460-utterance locked test and MUSAN-test noise"
            )
        else:
            comparability = "not_comparable_missing_counterpart"
            reason = (
                "legacy-only selection-screen run" if old is not None else
                "paper-v2 run has no legacy aggregate counterpart"
            )
        for metric in METRICS:
            output.append(
                _new_row(
                    section="vivos_aggregate",
                    artifact=artifact,
                    old_scope=f"legacy_exposed_vivos_300/{group_column}={key[6]}",
                    new_scope=f"paper_v2_locked_vivos_460/{group_column}={key[6]}",
                    run_identity=identity,
                    metric=metric,
                    statistic="rate",
                    old_value=old.get(metric, "") if old else "",
                    new_value=new.get(metric, "") if new else "",
                    delta_new_minus_old="",
                    comparability=comparability,
                    reason=reason,
                    old_row_count=len(old_rows),
                    new_row_count=len(new_rows),
                    old_sample_count=old.get("n", "") if old else "",
                    new_sample_count=new.get("n", "") if new else "",
                    old_manifest_sha256=old_manifest_hash,
                    new_manifest_sha256=new_manifest_hash,
                    old_protocol_label="legacy_v1_exposed_test_selection_and_evaluation",
                    new_protocol_label="paper_v2_unseen_final_after_noisy_dev_decision_lock",
                    old_artifact_sha256=old_hash,
                    new_artifact_sha256=new_hash,
                    decision_lock_sha256=decision_hash,
                )
            )
    return output


def _prediction_identity(path: Path) -> tuple[tuple[str, ...], int, str]:
    rows = _load_csv(
        path,
        required=(
            "utt_id", "dataset", "model", "model_size", "train_type", "lambda",
            "seed", "snr", "noise_type", "ref", "hyp",
        ),
        label="FLEURS prediction",
    )
    run_keys = {_run_key(row) for row in rows}
    if len(run_keys) != 1:
        raise ComparisonError(f"prediction mixes run identities: {path}")
    identities: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        utt_id = row["utt_id"].strip()
        if not utt_id or utt_id in seen:
            raise ComparisonError(f"prediction has blank/duplicate utt_id: {path}")
        seen.add(utt_id)
        if row["dataset"].strip().casefold() != "fleurs":
            raise ComparisonError(f"non-FLEURS row in external prediction: {path}")
        if row["snr"].strip().casefold() != "clean" or row["noise_type"].strip().casefold() != "clean":
            raise ComparisonError(f"non-clean row in FLEURS prediction: {path}")
        identities.append((utt_id, unicodedata.normalize("NFC", row["ref"])))
    return next(iter(run_keys)), len(rows), _canonical_sha256(identities)


def _manifest_identity(path: Path) -> tuple[int, str]:
    identities: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ComparisonError(f"invalid FLEURS JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ComparisonError(f"FLEURS manifest row is not an object: {line_number}")
            utt_id = str(row.get("utt_id", "")).strip()
            ref = str(row.get("transcript", ""))
            if not utt_id or not ref or utt_id in seen:
                raise ComparisonError(f"invalid FLEURS identity at {path}:{line_number}")
            seen.add(utt_id)
            identities.append((utt_id, unicodedata.normalize("NFC", ref)))
    if not identities:
        raise ComparisonError(f"FLEURS manifest is empty: {path}")
    return len(identities), _canonical_sha256(identities)


def _prediction_map(paths: Iterable[Path]) -> dict[tuple[str, ...], tuple[int, str, str]]:
    result: dict[tuple[str, ...], tuple[int, str, str]] = {}
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        key, count, identity = _prediction_identity(path)
        if key in result:
            raise ComparisonError(f"duplicate FLEURS prediction run identity: {key}")
        result[key] = (count, identity, sha256_file(path))
    return result


def _resolve_recorded_path(value: object, *, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _fleurs_prediction_maps(
    inputs: ComparisonInputs,
    provenance: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, ...], tuple[int, str, str]],
    dict[tuple[str, ...], tuple[int, str, str]],
    int,
    str,
]:
    old_paths = list(inputs.old_fleurs_predictions_dir.glob("pred_*.csv"))
    if not old_paths:
        raise ComparisonError(
            f"no legacy FLEURS predictions in {inputs.old_fleurs_predictions_dir}"
        )
    runs = provenance.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise ComparisonError("new FLEURS provenance must list exactly three runs")
    new_paths: list[Path] = []
    for run in runs:
        if not isinstance(run, dict) or not run.get("prediction"):
            raise ComparisonError("new FLEURS provenance has an invalid run entry")
        path = _resolve_recorded_path(run["prediction"], repo_root=inputs.repo_root)
        if not path.is_file():
            raise ComparisonError(f"new FLEURS prediction is missing: {path}")
        recorded = str(run.get("prediction_sha256", "")).casefold()
        if recorded != sha256_file(path):
            raise ComparisonError(f"new FLEURS prediction hash mismatch: {path}")
        new_paths.append(path)
    manifest_count, manifest_identity = _manifest_identity(inputs.fleurs_manifest)
    old_map = _prediction_map(old_paths)
    new_map = _prediction_map(new_paths)
    for label, mapping in (("old", old_map), ("new", new_map)):
        for key, (count, identity, _) in mapping.items():
            if count != manifest_count or identity != manifest_identity:
                raise ComparisonError(
                    f"{label} FLEURS prediction identity differs from manifest: {key}"
                )
    return old_map, new_map, manifest_count, manifest_identity


def _result_map(
    rows: Sequence[Mapping[str, str]], *, label: str
) -> dict[tuple[str, ...], Mapping[str, str]]:
    result: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = _run_key(row)
        if key in result:
            raise ComparisonError(f"{label} has duplicate run identity: {key}")
        _int_text(row.get("n", ""), label=f"{label}.n")
        for metric in METRICS:
            _float(row.get(metric, ""), label=f"{label}.{metric}")
        result[key] = row
    return result


def _fleurs_rows(
    *,
    inputs: ComparisonInputs,
    roles: Mapping[str, Mapping[str, str]],
    decision_hash: str,
    diagnostic: bool,
) -> tuple[list[dict[str, str]], bool, str]:
    old_rows = _load_csv(
        inputs.old_fleurs_results,
        required=(*("dataset", "model", "model_size", "train_type", "lambda", "seed", "n"), *METRICS, "metric_version"),
        label="legacy FLEURS results",
    )
    new_rows = _load_csv(
        inputs.new_fleurs_results,
        required=(*("dataset", "model", "model_size", "train_type", "lambda", "seed", "n"), *METRICS, "metric_version"),
        label="paper-v2 FLEURS results",
    )
    provenance = _load_json(inputs.new_fleurs_provenance, label="new FLEURS provenance")
    if str(provenance.get("results_sha256", "")).casefold() != sha256_file(inputs.new_fleurs_results):
        raise ComparisonError("new FLEURS provenance does not bind its result CSV")
    if provenance.get("evaluation_domain") != "legacy_exposed_external_replication":
        raise ComparisonError("new FLEURS provenance has the wrong evaluation domain")
    if provenance.get("evaluation_scope") != "full_fleurs_857":
        raise ComparisonError("new FLEURS result is a smoke/partial scope")
    manifest_hash = sha256_file(inputs.fleurs_manifest)
    if str(provenance.get("manifest_sha256", "")).casefold() != manifest_hash:
        raise ComparisonError("new FLEURS provenance manifest hash mismatch")
    if str(provenance.get("decision_lock_sha256", "")).casefold() != decision_hash:
        raise ComparisonError("new FLEURS provenance decision-lock hash mismatch")

    identity_verified = False
    identity_reason = "row-level FLEURS identity was not verified"
    old_predictions: dict[tuple[str, ...], tuple[int, str, str]] = {}
    new_predictions: dict[tuple[str, ...], tuple[int, str, str]] = {}
    manifest_count = _count_records(inputs.fleurs_manifest)
    try:
        old_predictions, new_predictions, manifest_count, _ = _fleurs_prediction_maps(
            inputs, provenance
        )
        if not diagnostic and manifest_count != 857:
            raise ComparisonError(
                f"formal FLEURS comparison requires 857 utterances, found {manifest_count}"
            )
        identity_verified = True
        identity_reason = "same manifest and exact ordered utt_id/ref identity"
    except (ComparisonError, FileNotFoundError, OSError):
        if not diagnostic:
            raise

    old_map = _result_map(old_rows, label="legacy FLEURS results")
    new_map = _result_map(new_rows, label="paper-v2 FLEURS results")
    output: list[dict[str, str]] = []
    for role in ROLES:
        config = roles[role]
        candidates = [
            key for key in new_map
            if key[3] == config["train_type"] and key[4] == config["lambda"] and key[5] == config["seed"]
        ]
        if len(candidates) != 1:
            raise ComparisonError(f"new FLEURS results do not resolve decision role {role}")
        new_key = candidates[0]
        new = new_map[new_key]
        old = old_map.get(new_key)
        same_metric = old is not None and old.get("metric_version") == new.get("metric_version")
        same_prediction_identity = (
            identity_verified
            and old is not None
            and new_key in old_predictions
            and new_key in new_predictions
            and old_predictions[new_key][1] == new_predictions[new_key][1]
        )
        comparable = old is not None and same_metric and same_prediction_identity
        if comparable:
            comparability = "comparable_same_utterances"
            reason = (
                f"{identity_reason}; checkpoints differ by design, so delta measures retraining/protocol change"
            )
        elif old is None:
            comparability = "not_comparable_missing_legacy_run"
            reason = (
                f"legacy FLEURS has no run for dynamic decision role {role} at lambda={config['lambda']}"
            )
        elif not same_metric:
            comparability = "not_comparable_metric_version"
            reason = "metric versions differ"
        else:
            comparability = "not_comparable_unverified_identity"
            reason = identity_reason
        for metric in METRICS:
            old_value = old.get(metric, "") if old else ""
            new_value = new[metric]
            delta = ""
            if comparable:
                delta = _number_text(
                    _float(new_value, label=f"new FLEURS {metric}")
                    - _float(old_value, label=f"old FLEURS {metric}")
                )
            output.append(
                _new_row(
                    section="fleurs_metrics",
                    artifact="external_fleurs_results.csv",
                    old_scope="legacy_exposed_external_replication/full_fleurs_857",
                    new_scope="legacy_exposed_external_replication/full_fleurs_857",
                    run_identity=_run_identity(new_key, role=role),
                    metric=metric,
                    statistic="rate",
                    old_value=old_value,
                    new_value=new_value,
                    delta_new_minus_old=delta,
                    comparability=comparability,
                    reason=reason,
                    old_row_count=len(old_rows),
                    new_row_count=len(new_rows),
                    old_sample_count=old.get("n", "") if old else "",
                    new_sample_count=new["n"],
                    old_manifest_sha256=manifest_hash if identity_verified else "",
                    new_manifest_sha256=manifest_hash,
                    old_protocol_label="legacy_external_evaluation",
                    new_protocol_label="paper_v2_legacy_exposed_external_replication",
                    old_artifact_sha256=sha256_file(inputs.old_fleurs_results),
                    new_artifact_sha256=sha256_file(inputs.new_fleurs_results),
                    decision_lock_sha256=decision_hash,
                )
            )
    return output, identity_verified, manifest_hash


def _bootstrap_identity_old(row: Mapping[str, str], suffix: str) -> tuple[str, str, str]:
    return (
        row.get(f"train_type_{suffix}", "").strip().casefold(),
        _canonical_lambda(row.get(f"lambda_{suffix}", "")),
        row.get(f"seed_{suffix}", "").strip(),
    )


def _bootstrap_identity_new(row: Mapping[str, str], suffix: str) -> tuple[str, str, str]:
    return _bootstrap_identity_old(row, suffix)


def _bootstrap_rows(
    *,
    inputs: ComparisonInputs,
    identity_verified: bool,
    manifest_hash: str,
    decision_hash: str,
) -> list[dict[str, str]]:
    old_rows = _load_csv(
        inputs.old_fleurs_bootstrap,
        required=(
            "train_type_a", "lambda_a", "seed_a", "train_type_b", "lambda_b",
            "seed_b", "n_paired", "metric", "delta_b_minus_a", "ci_lower", "ci_upper",
            "n_bootstrap", "bootstrap_unit", "ci_method",
        ),
        label="legacy FLEURS bootstrap",
    )
    new_rows = _load_csv(
        inputs.new_fleurs_bootstrap,
        required=(
            "decision_sha256", "benchmark_sha256", "role_a", "train_type_a", "lambda_a", "seed_a", "role_b", "train_type_b",
            "lambda_b", "seed_b", "n_source_clusters", "metric", "delta_b_minus_a",
            "ci_lower", "ci_upper", "n_bootstrap", "bootstrap_unit", "ci_method",
        ),
        label="paper-v2 FLEURS bootstrap",
    )
    old_index: dict[tuple[tuple[str, str, str], tuple[str, str, str], str], Mapping[str, str]] = {}
    for row in old_rows:
        key = (_bootstrap_identity_old(row, "a"), _bootstrap_identity_old(row, "b"), row["metric"].casefold())
        if key in old_index:
            raise ComparisonError(f"duplicate legacy bootstrap identity: {key}")
        old_index[key] = row
    output: list[dict[str, str]] = []
    for new in new_rows:
        if str(new["decision_sha256"]).casefold() != decision_hash:
            raise ComparisonError("paper-v2 FLEURS bootstrap decision hash mismatch")
        if str(new["benchmark_sha256"]).casefold() != manifest_hash:
            raise ComparisonError("paper-v2 FLEURS bootstrap manifest hash mismatch")
        metric = new["metric"].casefold()
        if metric not in BOOTSTRAP_METRICS:
            raise ComparisonError(f"unexpected paper-v2 bootstrap metric: {metric}")
        identity_a = _bootstrap_identity_new(new, "a")
        identity_b = _bootstrap_identity_new(new, "b")
        old = old_index.get((identity_a, identity_b, metric))
        reversed_orientation = False
        if old is None:
            old = old_index.get((identity_b, identity_a, metric))
            reversed_orientation = old is not None
        values: dict[str, str] = {}
        if old is not None:
            if reversed_orientation:
                values = {
                    "delta_b_minus_a": _number_text(-_float(old["delta_b_minus_a"], label="legacy bootstrap delta")),
                    "ci_lower": _number_text(-_float(old["ci_upper"], label="legacy bootstrap upper")),
                    "ci_upper": _number_text(-_float(old["ci_lower"], label="legacy bootstrap lower")),
                }
            else:
                values = {field: old[field] for field in ("delta_b_minus_a", "ci_lower", "ci_upper")}
        comparable = old is not None and identity_verified
        reason = (
            "same 857 singleton utterances; CI implementation label changed from legacy paired percentile to audited ratio-of-totals cluster code"
            if comparable else
            ("legacy bootstrap has no matching dynamic lambda pair" if old is None else "FLEURS row identity was not verified")
        )
        comparability = (
            "comparable_with_bootstrap_method_note" if comparable else
            ("not_comparable_missing_legacy_pair" if old is None else "not_comparable_unverified_identity")
        )
        run_identity = (
            f"{new['role_a']}[{identity_a[0]},lambda={identity_a[1]},seed={identity_a[2]}]"
            f"->{new['role_b']}[{identity_b[0]},lambda={identity_b[1]},seed={identity_b[2]}]"
        )
        for statistic in ("delta_b_minus_a", "ci_lower", "ci_upper"):
            old_value = values.get(statistic, "")
            new_value = new[statistic]
            delta = ""
            if comparable:
                delta = _number_text(
                    _float(new_value, label=f"new bootstrap {statistic}")
                    - _float(old_value, label=f"old bootstrap {statistic}")
                )
            output.append(
                _new_row(
                    section="fleurs_bootstrap",
                    artifact="bootstrap_ci_results.csv",
                    old_scope="legacy_exposed_fleurs_857/utt_id_paired",
                    new_scope="legacy_exposed_fleurs_857/utt_id_singleton_external",
                    run_identity=run_identity,
                    metric=metric,
                    statistic=statistic,
                    old_value=old_value,
                    new_value=new_value,
                    delta_new_minus_old=delta,
                    comparability=comparability,
                    reason=reason,
                    old_row_count=len(old_rows),
                    new_row_count=len(new_rows),
                    old_sample_count=old.get("n_paired", "") if old else "",
                    new_sample_count=new["n_source_clusters"],
                    old_manifest_sha256=manifest_hash if identity_verified else "",
                    new_manifest_sha256=manifest_hash,
                    old_protocol_label="legacy_paired_percentile",
                    new_protocol_label="paper_v2_paired_cluster_percentile_ratio_of_totals",
                    old_artifact_sha256=sha256_file(inputs.old_fleurs_bootstrap),
                    new_artifact_sha256=sha256_file(inputs.new_fleurs_bootstrap),
                    decision_lock_sha256=decision_hash,
                )
            )
    return output


def _final_bootstrap_structural_rows(
    path: Path, *, new_manifest_hash: str, decision_hash: str
) -> list[dict[str, str]]:
    rows = _load_csv(
        path,
        required=("decision_sha256", "benchmark_sha256", "role_a", "role_b", "metric", "delta_b_minus_a", "ci_lower", "ci_upper", "n_source_clusters"),
        label="paper-v2 final VIVOS bootstrap",
    )
    output: list[dict[str, str]] = []
    for row in rows:
        if str(row["decision_sha256"]).casefold() != decision_hash:
            raise ComparisonError("paper-v2 final bootstrap decision hash mismatch")
        if str(row["benchmark_sha256"]).casefold() != new_manifest_hash:
            raise ComparisonError("paper-v2 final bootstrap manifest hash mismatch")
        output.append(
            _new_row(
                section="final_bootstrap_new_only",
                artifact="bootstrap_ci_final.csv",
                old_scope="legacy_vivos_300/no_equivalent_source_cluster_ci",
                new_scope="paper_v2_locked_vivos_460/source_utt_id_cluster",
                run_identity=f"{row['role_a']}->{row['role_b']}",
                metric=row["metric"],
                statistic="delta_b_minus_a",
                old_value="",
                new_value=row["delta_b_minus_a"],
                delta_new_minus_old="",
                comparability="not_comparable_new_protocol_only",
                reason="legacy final-test CI is not scope-compatible with the new unseen source-cluster benchmark",
                old_row_count="0",
                new_row_count=len(rows),
                old_sample_count="",
                new_sample_count=row["n_source_clusters"],
                old_manifest_sha256="",
                new_manifest_sha256=new_manifest_hash,
                old_protocol_label="legacy_exposed_test",
                new_protocol_label="paper_v2_unseen_final_cluster_bootstrap",
                old_artifact_sha256="",
                new_artifact_sha256=sha256_file(path),
                decision_lock_sha256=decision_hash,
            )
        )
    return output


def _protocol_structural_rows(
    *,
    split: Mapping[str, Any] | None,
    noisy_dev: Mapping[str, Any] | None,
    old_manifest_hash: str,
    new_manifest_hash: str,
    old_manifest_rows: int | None,
    new_manifest_rows: int | None,
    decision_hash: str,
) -> list[dict[str, str]]:
    old_sources = 300
    new_sources = 460
    if split:
        try:
            old_sources = int(split["official_test"]["legacy_exposed_utterance_count"])
            new_sources = int(split["official_test"]["unseen_locked_utterance_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonError("split lock lacks exposed/unseen counts") from error
    noisy_dev_rows = ""
    if noisy_dev:
        try:
            noisy_dev_rows = str(int(noisy_dev["output"]["row_count"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonError("noisy-dev lock lacks output.row_count") from error
    common = {
        "section": "protocol_structure",
        "artifact": "benchmark_and_selection_protocol",
        "old_scope": "legacy_v1",
        "new_scope": "paper_v2",
        "run_identity": "all_runs",
        "statistic": "protocol_value",
        "comparability": "structural_change_not_numeric_comparison",
        "old_manifest_sha256": old_manifest_hash,
        "new_manifest_sha256": new_manifest_hash,
        "old_protocol_label": "legacy_v1_exposed_test_selection_and_evaluation",
        "new_protocol_label": "paper_v2_noisy_dev_selection_then_unseen_final",
        "decision_lock_sha256": decision_hash,
    }
    definitions = (
        (
            "source_utterance_count", str(old_sources), str(new_sources),
            "300 legacy-exposed and 460 locked-unseen utterances are disjoint; no metric delta is valid",
            str(old_manifest_rows or ""), str(new_manifest_rows or ""),
        ),
        (
            "benchmark_observation_count", str(old_manifest_rows or ""), str(new_manifest_rows or ""),
            "both use clean plus four SNR conditions, but their source utterances and noise partitions differ",
            str(old_manifest_rows or ""), str(new_manifest_rows or ""),
        ),
        (
            "training_pool_protocol",
            "official_train+dev (11,660; dev included in historical fit pool)",
            "official_train only (8,835); dev 2,825 held out",
            "paper-v2 removes the historical dev-in-training validity blocker",
            "11660", "8835",
        ),
        (
            "lambda_selection_scope",
            "same 300-source exposed benchmark used for lambda screening",
            f"held-out noisy-dev ({noisy_dev_rows or 'missing'} rows) before final-test unlock",
            "the old selected checkpoint and new decision-locked checkpoint follow different selection protocols",
            "300", noisy_dev_rows,
        ),
        (
            "noise_partition_protocol",
            "legacy MUSAN pool without train/dev/test content lock",
            "SHA-locked, content-disjoint MUSAN train/dev/test partitions",
            "noise realizations are not exchangeable across protocol versions",
            "", "",
        ),
    )
    return [
        _new_row(
            **common,
            metric=metric,
            old_value=old_value,
            new_value=new_value,
            delta_new_minus_old="",
            reason=reason,
            old_sample_count=old_n,
            new_sample_count=new_n,
        )
        for metric, old_value, new_value, reason, old_n, new_n in definitions
    ]


def _missing_rows(missing: Sequence[str]) -> list[dict[str, str]]:
    return [
        _new_row(
            section="diagnostic_missing_artifacts",
            artifact=name,
            old_scope="",
            new_scope="",
            run_identity="",
            metric="availability",
            statistic="artifact_status",
            old_value="",
            new_value="missing",
            comparability="not_comparable_missing_artifact",
            reason="diagnostic preview only; formal mode fails closed when this artifact is missing",
        )
        for name in sorted(missing)
    ]


def _validate_formal_shapes(
    *,
    old_snr: Sequence[Mapping[str, str]],
    old_noise: Sequence[Mapping[str, str]],
    new_snr: Sequence[Mapping[str, str]],
    new_noise: Sequence[Mapping[str, str]],
    old_fleurs: Sequence[Mapping[str, str]],
    new_fleurs: Sequence[Mapping[str, str]],
    old_bootstrap: Sequence[Mapping[str, str]],
    new_fleurs_bootstrap: Sequence[Mapping[str, str]],
    new_final_bootstrap: Sequence[Mapping[str, str]],
    old_manifest_rows: int,
    new_manifest_rows: int,
) -> None:
    expected = {
        "legacy results_by_snr": (len(old_snr), 77),
        "legacy results_by_noise_type": (len(old_noise), 44),
        "paper-v2 results_by_snr": (len(new_snr), 63),
        "paper-v2 results_by_noise_type": (len(new_noise), 36),
        "legacy FLEURS results": (len(old_fleurs), 3),
        "paper-v2 FLEURS results": (len(new_fleurs), 3),
        "legacy FLEURS bootstrap": (len(old_bootstrap), 12),
        "paper-v2 FLEURS bootstrap": (len(new_fleurs_bootstrap), 12),
        "paper-v2 final bootstrap": (len(new_final_bootstrap), 12),
        "legacy benchmark manifest": (old_manifest_rows, 1500),
        "paper-v2 final benchmark manifest": (new_manifest_rows, 2300),
    }
    wrong = [f"{label}: got {actual}, expected {wanted}" for label, (actual, wanted) in expected.items() if actual != wanted]
    if wrong:
        raise ComparisonError("formal artifact shape mismatch: " + "; ".join(wrong))


def _markdown(rows: Sequence[Mapping[str, str]], *, diagnostic: bool, inventory: Mapping[str, Any]) -> str:
    comparable = [row for row in rows if row["comparability"].startswith("comparable")]
    noncomparable = [row for row in rows if not row["comparability"].startswith("comparable")]
    lines = [
        "# So sánh output legacy và paper-v2",
        "",
        f"- Chế độ: `{'diagnostic_allow_partial' if diagnostic else 'formal_fail_closed'}`",
        f"- Tổng số dòng: {len(rows)}; có delta hợp lệ: {sum(bool(row['delta_new_minus_old']) for row in rows)}.",
        f"- Comparable: {len(comparable)}; structural/không comparable: {len(noncomparable)}.",
        "",
        "## Quy tắc diễn giải",
        "",
        "VIVOS legacy (300 source utterance đã exposed) và paper-v2 (460 source utterance locked-unseen) không cùng scope. CSV vẫn giữ hai giá trị để audit, nhưng cố ý để trống `delta_new_minus_old`. FLEURS chỉ có delta khi exact ordered `utt_id/ref` của cả prediction cũ, prediction mới và manifest 857 câu khớp nhau.",
        "",
        "## Thay đổi protocol chính",
        "",
        "| Thuộc tính | Cũ | Mới | Kết luận |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["section"] == "protocol_structure":
            lines.append(
                f"| {row['metric']} | {row['old_value']} | {row['new_value']} | {row['reason']} |"
            )
    lines.extend(
        [
            "",
            "## FLEURS: các delta được phép",
            "",
            "| Run | Metric/statistic | Cũ | Mới | Delta mới-cũ | Mức so sánh |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparable:
        lines.append(
            f"| {row['run_identity']} | {row['metric']}/{row['statistic']} | {row['old_value']} | {row['new_value']} | {row['delta_new_minus_old']} | {row['comparability']} |"
        )
    if not comparable:
        lines.append("| — | — | — | — | — | Chưa đủ artifact để chứng minh comparability |")
    lines.extend(
        [
            "",
            "## Inventory đầu vào",
            "",
            "| Artifact | Trạng thái | Số dòng | SHA-256 |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for name, item in sorted(inventory.items()):
        lines.append(
            f"| {name} | {item.get('status', '')} | {item.get('row_count', '')} | {item.get('sha256', '')} |"
        )
    if diagnostic:
        lines.extend(
            [
                "",
                "> CẢNH BÁO: đây là preview diagnostic có artifact thiếu. Không dùng làm bảng kết quả paper.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_comparison(
    inputs: ComparisonInputs,
    *,
    diagnostic_allow_partial: bool = False,
) -> ComparisonBundle:
    """Build comparison rows without writing outputs.

    Formal mode (the default) requires every registered post-run artifact.
    Diagnostic mode reports missing inputs explicitly and only builds sections
    whose dependencies are complete.
    """

    files = inputs.files()
    missing = [name for name, path in files.items() if not path.is_file()]
    if not inputs.old_fleurs_predictions_dir.is_dir():
        missing.append("old_fleurs_predictions_dir")
    if missing and not diagnostic_allow_partial:
        raise ComparisonError("formal comparison is incomplete; missing: " + ", ".join(sorted(missing)))

    inventory: dict[str, dict[str, Any]] = {}
    for name, path in sorted(files.items()):
        if path.is_file():
            item: dict[str, Any] = {"status": "present", "sha256": sha256_file(path)}
            if path.suffix.casefold() in {".csv", ".jsonl"}:
                item["row_count"] = _count_records(path)
            inventory[name] = item
        else:
            inventory[name] = {"status": "missing", "sha256": "", "row_count": ""}

    rows: list[dict[str, str]] = _missing_rows(missing) if diagnostic_allow_partial else []
    decision: dict[str, Any] | None = None
    roles: dict[str, dict[str, str]] = {}
    decision_hash = ""
    if inputs.decision_lock.is_file():
        decision = _load_json(inputs.decision_lock, label="best-lambda decision lock")
        roles, _ = _decision_roles(decision)
        decision_hash = sha256_file(inputs.decision_lock)

    old_manifest_hash = sha256_file(inputs.old_benchmark_manifest) if inputs.old_benchmark_manifest.is_file() else ""
    new_manifest_hash = sha256_file(inputs.new_benchmark_manifest) if inputs.new_benchmark_manifest.is_file() else ""
    old_manifest_rows = _count_records(inputs.old_benchmark_manifest) if inputs.old_benchmark_manifest.is_file() else None
    new_manifest_rows = _count_records(inputs.new_benchmark_manifest) if inputs.new_benchmark_manifest.is_file() else None
    split = _load_json(inputs.split_lock, label="split lock") if inputs.split_lock.is_file() else None
    noisy_dev = _load_json(inputs.noisy_dev_lock, label="noisy-dev lock") if inputs.noisy_dev_lock.is_file() else None
    protocol_objects: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("noise split lock", inputs.noise_split_lock),
        ("environment lock", inputs.environment_lock),
        ("method lock", inputs.method_lock),
        ("final benchmark lock", inputs.final_benchmark_lock),
        ("FLEURS preparation lock", inputs.fleurs_preparation_lock),
    ):
        if path.is_file():
            protocol_objects[name] = _load_json(path, label=name)
    if split and old_manifest_hash:
        try:
            recorded_old_hash = str(split["official_test"]["exposure_evidence"]["benchmark_manifest_sha256"]).casefold()
        except (KeyError, TypeError) as error:
            raise ComparisonError("split lock lacks legacy benchmark evidence") from error
        if recorded_old_hash != old_manifest_hash:
            raise ComparisonError("legacy benchmark hash differs from split-lock exposure evidence")
    if new_manifest_hash and "final benchmark lock" in protocol_objects:
        try:
            recorded_new_hash = str(
                protocol_objects["final benchmark lock"]["output"]["manifest_sha256"]
            ).casefold()
        except (KeyError, TypeError) as error:
            raise ComparisonError("final benchmark lock lacks output.manifest_sha256") from error
        if recorded_new_hash != new_manifest_hash:
            raise ComparisonError("paper-v2 benchmark hash differs from final benchmark lock")
    if inputs.fleurs_manifest.is_file() and "FLEURS preparation lock" in protocol_objects:
        fleurs_lock = protocol_objects["FLEURS preparation lock"]
        if fleurs_lock.get("status") != "LOCKED":
            raise ComparisonError("FLEURS preparation lock is not LOCKED")
        try:
            fleurs_output = fleurs_lock["output"]
            locked_fleurs_hash = str(fleurs_output["manifest_sha256"]).casefold()
            locked_fleurs_rows = int(fleurs_output["row_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonError("FLEURS preparation lock lacks output binding") from error
        if locked_fleurs_hash != sha256_file(inputs.fleurs_manifest) or locked_fleurs_rows != 857:
            raise ComparisonError("FLEURS manifest differs from its 857-row preparation lock")
    rows.extend(
        _protocol_structural_rows(
            split=split,
            noisy_dev=noisy_dev,
            old_manifest_hash=old_manifest_hash,
            new_manifest_hash=new_manifest_hash,
            old_manifest_rows=old_manifest_rows,
            new_manifest_rows=new_manifest_rows,
            decision_hash=decision_hash,
        )
    )

    old_snr = old_noise = new_snr = new_noise = None
    if roles and all(path.is_file() for path in (inputs.old_by_snr, inputs.new_by_snr)):
        old_snr = _load_csv(inputs.old_by_snr, required=("dataset", "model", "model_size", "train_type", "lambda", "seed", "snr", "n", *METRICS), label="legacy results_by_snr")
        new_snr = _load_csv(inputs.new_by_snr, required=("dataset", "model", "model_size", "train_type", "lambda", "seed", "snr", "n", *METRICS), label="paper-v2 results_by_snr")
        rows.extend(_aggregate_rows(artifact="results_by_snr.csv", group_column="snr", old_rows=old_snr, new_rows=new_snr, roles=roles, old_hash=sha256_file(inputs.old_by_snr), new_hash=sha256_file(inputs.new_by_snr), old_manifest_hash=old_manifest_hash, new_manifest_hash=new_manifest_hash, decision_hash=decision_hash))
    if roles and all(path.is_file() for path in (inputs.old_by_noise_type, inputs.new_by_noise_type)):
        old_noise = _load_csv(inputs.old_by_noise_type, required=("dataset", "model", "model_size", "train_type", "lambda", "seed", "noise_type", "n", *METRICS), label="legacy results_by_noise_type")
        new_noise = _load_csv(inputs.new_by_noise_type, required=("dataset", "model", "model_size", "train_type", "lambda", "seed", "noise_type", "n", *METRICS), label="paper-v2 results_by_noise_type")
        rows.extend(_aggregate_rows(artifact="results_by_noise_type.csv", group_column="noise_type", old_rows=old_noise, new_rows=new_noise, roles=roles, old_hash=sha256_file(inputs.old_by_noise_type), new_hash=sha256_file(inputs.new_by_noise_type), old_manifest_hash=old_manifest_hash, new_manifest_hash=new_manifest_hash, decision_hash=decision_hash))

    identity_verified = False
    fleurs_manifest_hash = ""
    if roles and all(path.is_file() for path in (inputs.old_fleurs_results, inputs.new_fleurs_results, inputs.new_fleurs_provenance, inputs.fleurs_manifest)):
        fleurs_output, identity_verified, fleurs_manifest_hash = _fleurs_rows(inputs=inputs, roles=roles, decision_hash=decision_hash, diagnostic=diagnostic_allow_partial)
        rows.extend(fleurs_output)
    if all(path.is_file() for path in (inputs.old_fleurs_bootstrap, inputs.new_fleurs_bootstrap)):
        rows.extend(_bootstrap_rows(inputs=inputs, identity_verified=identity_verified, manifest_hash=fleurs_manifest_hash, decision_hash=decision_hash))
    if inputs.new_final_bootstrap.is_file():
        rows.extend(_final_bootstrap_structural_rows(inputs.new_final_bootstrap, new_manifest_hash=new_manifest_hash, decision_hash=decision_hash))

    if not diagnostic_allow_partial:
        assert old_snr is not None and old_noise is not None and new_snr is not None and new_noise is not None
        old_fleurs = _load_csv(inputs.old_fleurs_results, required=("dataset", "n", *METRICS), label="legacy FLEURS results")
        new_fleurs = _load_csv(inputs.new_fleurs_results, required=("dataset", "n", *METRICS), label="paper-v2 FLEURS results")
        old_bootstrap = _load_csv(inputs.old_fleurs_bootstrap, required=("metric",), label="legacy FLEURS bootstrap")
        new_fleurs_bootstrap = _load_csv(inputs.new_fleurs_bootstrap, required=("metric",), label="paper-v2 FLEURS bootstrap")
        new_final_bootstrap = _load_csv(inputs.new_final_bootstrap, required=("metric",), label="paper-v2 final bootstrap")
        _validate_formal_shapes(old_snr=old_snr, old_noise=old_noise, new_snr=new_snr, new_noise=new_noise, old_fleurs=old_fleurs, new_fleurs=new_fleurs, old_bootstrap=old_bootstrap, new_fleurs_bootstrap=new_fleurs_bootstrap, new_final_bootstrap=new_final_bootstrap, old_manifest_rows=int(old_manifest_rows or -1), new_manifest_rows=int(new_manifest_rows or -1))
        if not identity_verified:
            raise ComparisonError("formal FLEURS comparison requires exact row-level identity proof")

    rows.sort(key=lambda row: tuple(row[column] for column in ("section", "artifact", "run_identity", "metric", "statistic")))
    input_identity = _canonical_sha256(
        {name: item.get("sha256", "") for name, item in sorted(inventory.items())}
    )
    provenance: dict[str, Any] = {
        "comparison_version": COMPARISON_VERSION,
        "mode": "diagnostic_allow_partial" if diagnostic_allow_partial else "formal_fail_closed",
        "input_set_sha256": input_identity,
        "decision_lock_sha256": decision_hash,
        "selected_lambda": roles.get("selected_method", {}).get("lambda", ""),
        "locked_control_lambda": roles.get("locked_control", {}).get("lambda", ""),
        "fleurs_row_identity_verified": identity_verified,
        "row_count": len(rows),
        "valid_delta_row_count": sum(bool(row["delta_new_minus_old"]) for row in rows),
        "artifacts": inventory,
    }
    return ComparisonBundle(
        rows=tuple(rows),
        markdown=_markdown(rows, diagnostic=diagnostic_allow_partial, inventory=inventory),
        provenance=provenance,
    )


def _csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_COLUMNS), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _comparison_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _comparison_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _comparison_display_path(path: Path, anchor: Path) -> str:
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _comparison_transaction_paths(
    provenance_path: Path, bundle_sha256: str
) -> tuple[Path, Path, Path]:
    suffix = ".provenance.json"
    base = (
        provenance_path.name[: -len(suffix)]
        if provenance_path.name.endswith(suffix)
        else provenance_path.stem
    )
    parent = provenance_path.parent
    return (
        parent / f"{base}.bundle.commit.json",
        parent / f".{base}.bundle.transaction.json",
        parent / f".{base}.bundle.stage.{bundle_sha256}",
    )


def _comparison_descriptor(
    destinations: Mapping[str, Path], contents: Mapping[str, bytes], *, anchor: Path
) -> dict[str, Any]:
    outputs = [
        {
            "key": key,
            "path": _comparison_display_path(destinations[key], anchor),
            "bytes": len(contents[key]),
            "sha256": _comparison_sha256_bytes(contents[key]),
        }
        for key in sorted(destinations)
    ]
    identity = {
        "bundle_version": COMPARISON_BUNDLE_VERSION,
        "outputs": outputs,
    }
    return {
        **identity,
        "bundle_sha256": _comparison_sha256_bytes(
            _comparison_json_bytes(identity)
        ),
    }


def _comparison_atomic_metadata_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _comparison_load_metadata(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"{label} is unreadable or corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must contain a JSON object: {path}")
    return value


def _comparison_validate_descriptor(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key in ("bundle_version", "bundle_sha256", "outputs"):
        if value.get(key) != expected.get(key):
            raise ComparisonError(
                f"{label} does not match the requested deterministic bundle ({key})"
            )


def _comparison_validate_journal_integrity(journal: Mapping[str, Any]) -> None:
    recorded = journal.get("journal_sha256")
    unsigned = {
        key: value for key, value in journal.items() if key != "journal_sha256"
    }
    if recorded != _comparison_sha256_bytes(_comparison_json_bytes(unsigned)):
        raise ComparisonError("comparison transaction journal integrity check failed")


def _comparison_validate_marker(
    marker_path: Path,
    destinations: Mapping[str, Path],
    *,
    anchor: Path,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    marker = _comparison_load_metadata(
        marker_path, label="comparison commit marker"
    )
    if marker.get("status") != "COMMITTED":
        raise ComparisonError("comparison commit marker is not COMMITTED")
    if marker.get("bundle_version") != COMPARISON_BUNDLE_VERSION:
        raise ComparisonError("comparison commit marker uses an unsupported version")
    outputs = marker.get("outputs")
    if not isinstance(outputs, list):
        raise ComparisonError("comparison commit marker has invalid outputs")
    identity = {
        "bundle_version": marker.get("bundle_version"),
        "outputs": outputs,
    }
    if marker.get("bundle_sha256") != _comparison_sha256_bytes(
        _comparison_json_bytes(identity)
    ):
        raise ComparisonError("comparison commit marker identity is corrupt")
    if expected is not None:
        _comparison_validate_descriptor(
            marker, expected, label="comparison commit marker"
        )
    recorded = {
        str(item.get("key")): item
        for item in outputs
        if isinstance(item, dict) and item.get("key")
    }
    if set(recorded) != set(destinations):
        raise ComparisonError("comparison commit marker output set is invalid")
    for key, destination in destinations.items():
        item = recorded[key]
        if item.get("path") != _comparison_display_path(destination, anchor):
            raise ComparisonError("comparison commit marker destination set is invalid")
        if not destination.is_file():
            raise ComparisonError(f"committed comparison output is missing: {destination}")
        if destination.stat().st_size != item.get("bytes") or sha256_file(
            destination
        ) != item.get("sha256"):
            raise ComparisonError(
                f"committed comparison output was tampered with: {destination}"
            )
    return marker


def _comparison_stage_name(key: str) -> str:
    return {
        "csv": "000-comparison.csv",
        "markdown": "001-comparison.md",
        "provenance": "002-provenance.json",
    }[key]


def _comparison_validate_stage_inventory(
    stage_dir: Path, keys: Sequence[str]
) -> None:
    if not stage_dir.exists():
        return
    allowed = {_comparison_stage_name(key) for key in keys}
    unexpected = sorted(
        path.name for path in stage_dir.iterdir() if path.name not in allowed
    )
    if unexpected:
        raise ComparisonError(
            f"comparison recovery stage contains unexpected entries: {unexpected}"
        )


def _comparison_write_stage(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _promote_comparison_staged_file(stage_path: Path, destination: Path) -> None:
    """Small promotion seam used by crash/recovery tests."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage_path, destination)


def _comparison_cleanup_transaction(
    journal_path: Path, stage_dir: Path, keys: Sequence[str]
) -> None:
    if journal_path.exists():
        journal_path.unlink()
    for key in keys:
        staged = stage_dir / _comparison_stage_name(key)
        if staged.exists():
            staged.unlink()
    try:
        stage_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        # Do not recursively delete unknown residue.  The commit marker proves
        # the canonical bundle while the extra file remains available to audit.
        pass


def write_comparison(
    bundle: ComparisonBundle,
    *,
    csv_path: str | Path,
    markdown_path: str | Path,
    provenance_path: str | Path,
    resume: bool = False,
) -> tuple[Path, Path, Path]:
    """Write a recoverable, immutable CSV/Markdown/provenance bundle.

    The three canonical paths cannot be renamed atomically as one filesystem
    object, so a durable PREPARED journal is written before promotion and a
    hash-bound COMMITTED marker is written last.  ``resume=True`` only promotes
    missing files when all surviving canonical/staged bytes match this exact
    deterministic recomputation.
    """

    destination_tuple = tuple(
        Path(path) for path in (csv_path, markdown_path, provenance_path)
    )
    resolved = [path.resolve() for path in destination_tuple]
    if len(set(resolved)) != 3:
        raise ComparisonError("comparison output paths must be distinct")
    destinations = {
        "csv": destination_tuple[0],
        "markdown": destination_tuple[1],
        "provenance": destination_tuple[2],
    }
    csv_content = _csv_bytes(bundle.rows)
    markdown_content = bundle.markdown.encode("utf-8")
    provenance = dict(bundle.provenance)

    def display(path: Path) -> str:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.name

    provenance.update(
        {
            "comparison_csv": display(destination_tuple[0]),
            "comparison_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
            "comparison_markdown": display(destination_tuple[1]),
            "comparison_markdown_sha256": hashlib.sha256(markdown_content).hexdigest(),
        }
    )
    provenance_content = (
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    contents = {
        "csv": csv_content,
        "markdown": markdown_content,
        "provenance": provenance_content,
    }
    anchor = destinations["provenance"].parent
    descriptor = _comparison_descriptor(destinations, contents, anchor=anchor)
    marker_path, journal_path, stage_dir = _comparison_transaction_paths(
        destinations["provenance"], str(descriptor["bundle_sha256"])
    )
    keys = sorted(destinations)

    if not resume:
        occupied = [path for path in destinations.values() if path.exists()]
        transaction_residue = [
            path
            for path in (marker_path, journal_path, stage_dir)
            if path.exists()
        ]
        if occupied or transaction_residue:
            raise FileExistsError(
                "comparison outputs are immutable; already exist (use --resume "
                "only for an interrupted exact bundle): "
                f"{occupied + transaction_residue}"
            )

    if resume and not journal_path.exists() and marker_path.exists():
        _comparison_validate_marker(
            marker_path, destinations, anchor=anchor, expected=descriptor
        )
        return destination_tuple

    if journal_path.exists():
        if not resume:
            raise FileExistsError(
                f"unfinished comparison transaction exists; rerun with --resume: {journal_path}"
            )
        journal = _comparison_load_metadata(
            journal_path, label="comparison transaction journal"
        )
        _comparison_validate_journal_integrity(journal)
        if journal.get("status") != "PREPARED" or journal.get("mode") != "create":
            raise ComparisonError("comparison transaction journal has invalid state")
        _comparison_validate_descriptor(
            journal, descriptor, label="comparison transaction journal"
        )
    else:
        prior: dict[str, str | None] = {}
        for key, destination in destinations.items():
            expected_hash = _comparison_sha256_bytes(contents[key])
            if destination.exists():
                current_hash = sha256_file(destination)
                if not resume or current_hash != expected_hash:
                    raise ComparisonError(
                        f"cannot recover comparison bundle: unexpected canonical file {destination}"
                    )
                prior[key] = expected_hash
            else:
                prior[key] = None

        stage_dir.mkdir(parents=True, exist_ok=True)
        _comparison_validate_stage_inventory(stage_dir, keys)
        for key, content in contents.items():
            destination = destinations[key]
            expected_hash = _comparison_sha256_bytes(content)
            if destination.is_file() and sha256_file(destination) == expected_hash:
                continue
            staged = stage_dir / _comparison_stage_name(key)
            if staged.exists():
                if sha256_file(staged) != expected_hash:
                    raise ComparisonError(
                        f"comparison recovery stage was tampered with: {staged}"
                    )
            else:
                _comparison_write_stage(staged, content)
        journal_unsigned = {
            **descriptor,
            "status": "PREPARED",
            "mode": "create",
            "prior_sha256": prior,
        }
        journal = {
            **journal_unsigned,
            "journal_sha256": _comparison_sha256_bytes(
                _comparison_json_bytes(journal_unsigned)
            ),
        }
        _comparison_atomic_metadata_write(
            journal_path, _comparison_json_bytes(journal)
        )

    _comparison_validate_stage_inventory(stage_dir, keys)

    if marker_path.exists():
        _comparison_validate_marker(
            marker_path, destinations, anchor=anchor, expected=descriptor
        )
        _comparison_cleanup_transaction(journal_path, stage_dir, keys)
        return destination_tuple

    prior = journal.get("prior_sha256")
    if not isinstance(prior, dict) or set(prior) != set(destinations):
        raise ComparisonError("comparison transaction journal has invalid prior hashes")
    for key in keys:
        destination = destinations[key]
        expected_hash = _comparison_sha256_bytes(contents[key])
        if destination.exists():
            if sha256_file(destination) == expected_hash:
                continue
            raise ComparisonError(
                f"comparison canonical output changed during transaction: {destination}"
            )
        staged = stage_dir / _comparison_stage_name(key)
        if not staged.is_file() or sha256_file(staged) != expected_hash:
            raise ComparisonError(
                f"comparison staged output is missing or tampered: {staged}"
            )
        _promote_comparison_staged_file(staged, destination)

    marker = {**descriptor, "status": "COMMITTED"}
    _comparison_atomic_metadata_write(marker_path, _comparison_json_bytes(marker))
    _comparison_validate_marker(
        marker_path, destinations, anchor=anchor, expected=descriptor
    )
    _comparison_cleanup_transaction(journal_path, stage_dir, keys)
    return destination_tuple


__all__ = [
    "COMPARISON_COLUMNS",
    "COMPARISON_BUNDLE_VERSION",
    "COMPARISON_VERSION",
    "ComparisonBundle",
    "ComparisonError",
    "ComparisonInputs",
    "build_comparison",
    "sha256_file",
    "write_comparison",
]
