from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import transformers
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.vitonesr.analysis import (
    METRIC_VERSION,
    compute_aligned_metric_result,
    validate_prediction_rows,
)
from src.vitonesr.noise import read_audio
from src.vitonesr.prediction import atomic_write_csv, normalize_snr, read_csv_rows, validate_columns

from .config import ConfigDict
from .method_contract import (
    verify_checkpoint_method_binding,
    verify_method_lock,
    verify_noisy_dev_lock,
)
from .protocol import (
    canonical_sha256,
    checkpoint_inference_sha256,
    evaluation_contract_payload,
    evaluation_contract_sha256,
    is_sha256,
    load_split_lock,
    selection_rule_sha256,
    selected_rows_sha256,
    sha256_file,
    verify_checkpoint_config,
    verify_expected_manifest,
    verify_locked_vivos_manifest,
    verify_test_configuration_locked,
    verify_test_decision_lock,
)


PREDICTION_COLUMNS = [
    "utt_id",
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "snr",
    "noise_type",
    "ref",
    "hyp",
]
REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTION_PROVENANCE_VERSION = "prediction_evaluation_v4"
PREDICTION_RESUME_VERSION = "prediction_evaluation_resume_v1"
PREDICTION_RECOVERY_VERSION = "prediction_evaluation_recovery_v1"

ABLATION_RESULT_COLUMNS = [
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "evaluation_split",
    "manifest_sha256",
    "evaluation_scope",
    "selected_rows_sha256",
    "training_scope",
    "training_contract_sha256",
    "evaluation_contract_sha256",
    "method_lock_sha256",
    "method_identity_sha256",
    "environment_artifact_sha256",
    "environment_identity_sha256",
    "source_tree_sha256",
    "split",
    "snr",
    "noise_type",
    "num_samples",
    "metric_version",
    "wer",
    "wer_numerator",
    "wer_denominator",
    "cer",
    "cer_numerator",
    "cer_denominator",
    "ter",
    "ter_numerator",
    "ter_denominator",
    "ter_coverage",
    "der",
    "der_numerator",
    "der_denominator",
    "der_coverage",
    "fcer",
    "fcer_numerator",
    "fcer_denominator",
    "fcer_coverage",
    "swdr",
    "swdr_numerator",
    "swdr_denominator",
    "checkpoint_path",
    "checkpoint_sha256",
    "prediction_path",
]


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Evaluation manifest does not exist: {manifest}")
    if manifest.suffix.lower() == ".csv":
        with manifest.open("r", encoding="utf-8", newline="") as manifest_file:
            rows = list(csv.DictReader(manifest_file))
    elif manifest.suffix.lower() in {".jsonl", ".json"}:
        with manifest.open("r", encoding="utf-8") as manifest_file:
            rows = [json.loads(line) for line in manifest_file if line.strip()]
    else:
        raise ValueError(f"Unsupported manifest format: {manifest}")
    if not rows:
        raise ValueError(f"Evaluation manifest is empty: {manifest}")
    return rows


def _canonical_manifest_row(row: dict[str, Any]) -> dict[str, str]:
    audio_path = row.get("audio_path") or row.get("audio") or row.get("noisy_path") or row.get("clean_path")
    reference = row.get("transcript") or row.get("text") or row.get("ref")
    if not audio_path or reference is None:
        raise ValueError("Every benchmark row must contain an audio path and transcript/reference")
    if not Path(str(audio_path)).exists():
        raise FileNotFoundError(f"Benchmark audio does not exist: {audio_path}")
    evaluation_split = str(row.get("split", "")).strip().casefold()
    if evaluation_split not in {"dev", "test", "external"}:
        raise ValueError(
            "Every evaluation manifest row must declare split=dev, test, or external"
        )
    snr = normalize_snr(row.get("snr", "clean"))
    utt_id = str(row.get("utt_id") or Path(str(audio_path)).stem).strip()
    dataset = str(row.get("dataset", "")).strip()
    noise_type = str(
        row.get("noise_type", "clean" if snr == "clean" else "")
    ).strip()
    if not utt_id or not dataset or not str(reference).strip():
        raise ValueError(
            "Every evaluation row must have non-empty utt_id, dataset, and reference"
        )
    if snr == "clean" and noise_type.casefold() != "clean":
        raise ValueError("Clean evaluation rows must use noise_type=clean")
    if snr != "clean" and (not noise_type or noise_type.casefold() == "clean"):
        raise ValueError("Noisy evaluation rows must declare a non-clean noise_type")
    return {
        "utt_id": utt_id,
        "dataset": dataset,
        "audio_path": str(audio_path),
        "snr": snr,
        "noise_type": noise_type,
        "ref": unicodedata.normalize("NFC", str(reference)),
        "evaluation_split": evaluation_split,
    }


def load_benchmark_rows(
    manifest: str | Path,
    *,
    subset: str = "all",
    snrs: Sequence[str] | None = None,
    noise_types: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    if subset not in {"all", "clean", "noisy"}:
        raise ValueError("subset must be one of: all, clean, noisy")
    rows = [_canonical_manifest_row(row) for row in _read_manifest(manifest)]
    seen_utt_ids: set[str] = set()
    for row in rows:
        if row["utt_id"] in seen_utt_ids:
            raise ValueError(
                f"Evaluation manifest has duplicate utt_id: {row['utt_id']}"
            )
        seen_utt_ids.add(row["utt_id"])
    observed_splits = {row["evaluation_split"] for row in rows}
    if len(observed_splits) != 1:
        raise ValueError(
            f"Evaluation manifest must contain exactly one data split; observed={sorted(observed_splits)}"
        )
    if subset == "clean":
        rows = [row for row in rows if row["snr"] == "clean"]
    elif subset == "noisy":
        rows = [row for row in rows if row["snr"] != "clean"]
    if snrs:
        normalized = {normalize_snr(value) for value in snrs}
        rows = [row for row in rows if row["snr"] in normalized]
    if noise_types:
        selected_noise = {str(value) for value in noise_types}
        rows = [row for row in rows if row["noise_type"] in selected_noise]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No benchmark rows remain after applying evaluation filters")
    return rows


def prediction_provenance_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    return path.with_suffix(path.suffix + ".provenance.json")


def prediction_resume_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    return path.with_suffix(path.suffix + ".resume.json")


def prediction_recovery_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    return path.with_suffix(path.suffix + ".recovery.json")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completed provenance: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"Refusing to overwrite completed provenance: {path}"
            ) from None
    finally:
        if temporary.exists():
            temporary.unlink()


def _prediction_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=PREDICTION_COLUMNS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in PREDICTION_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def _atomic_write_prediction(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> str:
    payload = _prediction_csv_bytes(rows)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Atomic prediction write failed integrity check: {path}")
    return expected_sha256


def resolve_checkpoint(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")
    if (candidate / "adapter" / "adapter_config.json").exists():
        return candidate, candidate / "adapter"
    if (candidate / "best" / "adapter" / "adapter_config.json").exists():
        return candidate / "best", candidate / "best" / "adapter"
    if (candidate / "final" / "adapter" / "adapter_config.json").exists():
        return candidate / "final", candidate / "final" / "adapter"
    if (candidate / "adapter_config.json").exists():
        return candidate.parent, candidate
    raise FileNotFoundError(f"Could not find a PEFT adapter under checkpoint: {candidate}")


def _batched(rows: Sequence[dict[str, str]], batch_size: int) -> Iterable[Sequence[dict[str, str]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _resolved_repo_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _repo_relative_reference(value: object, *, label: str) -> str:
    resolved = _resolved_repo_path(value)
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Formal {label} must remain inside the repository: {resolved}"
        ) from exc
    reference = relative.as_posix()
    if not reference or "\\" in reference or ".." in Path(reference).parts:
        raise ValueError(f"Formal {label} is not a portable repository-relative path")
    return reference


def _assert_no_absolute_paths(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_paths(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, label=f"{label}[{index}]")
    elif isinstance(value, str):
        windows_absolute = len(value) >= 3 and value[1:3] in {":\\", ":/"}
        if windows_absolute or Path(value).is_absolute():
            raise ValueError(f"{label} contains a forbidden absolute path")


def _config_identity_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in config.items() if key != "_config_path"}
    )


def _read_partial_prediction(path: Path) -> list[dict[str, str]]:
    rows, columns = read_csv_rows(path)
    validate_columns(path, columns, PREDICTION_COLUMNS, exact=True)
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed prediction row {row_number}: {path}")
    return [dict(row) for row in rows]


def _validate_prediction_prefix(
    prediction_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    train_type: str,
    lambda_tone: float,
    seed: int,
) -> None:
    if len(prediction_rows) > len(manifest_rows):
        raise ValueError("Partial prediction has more rows than the selected manifest")
    seen: set[str] = set()
    for index, prediction in enumerate(prediction_rows):
        manifest = manifest_rows[index]
        expected = {
            "utt_id": manifest["utt_id"],
            "dataset": manifest["dataset"],
            "model": "phowhisper",
            "model_size": "base",
            "train_type": train_type,
            "lambda": f"{lambda_tone:g}",
            "seed": str(seed),
            "snr": manifest["snr"],
            "noise_type": manifest["noise_type"],
            "ref": manifest["ref"],
        }
        if prediction.get("utt_id", "") in seen:
            raise ValueError("Partial prediction contains duplicate utt_id values")
        seen.add(str(prediction.get("utt_id", "")))
        for field, value in expected.items():
            if str(prediction.get(field, "")) != str(value):
                raise ValueError(
                    f"Prediction prefix mismatch at row {index + 1}, field {field}"
                )


def _resume_identity(
    *,
    resume_contract_sha256: str,
    config_identity_sha256: str,
    manifest_sha256: str,
    selected_rows_sha256_value: str,
    checkpoint_identity: Mapping[str, str],
) -> dict[str, str]:
    return {
        "resume_contract_sha256": resume_contract_sha256,
        "config_identity_sha256": config_identity_sha256,
        "manifest_sha256": manifest_sha256,
        "selected_rows_sha256": selected_rows_sha256_value,
        "checkpoint_sha256": str(checkpoint_identity["checkpoint_sha256"]),
        "resolved_config_sha256": str(
            checkpoint_identity["resolved_config_sha256"]
        ),
        "training_contract_sha256": str(
            checkpoint_identity["training_contract_sha256"]
        ),
    }


def _validate_resume_identity(
    value: Mapping[str, Any], expected: Mapping[str, str], *, label: str
) -> None:
    for field, expected_value in expected.items():
        if str(value.get(field, "")).casefold() != expected_value.casefold():
            raise ValueError(f"{label} identity mismatch: {field}")


def _resume_payload(
    *,
    prediction_path: Path,
    row_count: int,
    identity: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "resume_version": PREDICTION_RESUME_VERSION,
        **dict(identity),
        "completed_rows": row_count,
        "prediction_sha256": sha256_file(prediction_path),
    }


def _validate_resume_state(
    state: Mapping[str, Any],
    *,
    prediction_path: Path,
    row_count: int,
    identity: Mapping[str, str],
) -> None:
    if state.get("resume_version") != PREDICTION_RESUME_VERSION:
        raise ValueError("Unsupported checkpoint-evaluation resume state")
    _validate_resume_identity(state, identity, label="Resume state")
    if int(state.get("completed_rows", -1)) != row_count:
        raise ValueError("Resume state row count differs from prediction")
    if str(state.get("prediction_sha256", "")).casefold() != sha256_file(
        prediction_path
    ):
        raise ValueError("Resume state prediction SHA-256 mismatch")


def _recovery_payload(
    *,
    prediction_sha256: str,
    row_count: int,
    previous_prediction_sha256: str,
    previous_row_count: int,
    identity: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "recovery_version": PREDICTION_RECOVERY_VERSION,
        **dict(identity),
        "completed_rows": row_count,
        "prediction_sha256": prediction_sha256,
        "previous_completed_rows": previous_row_count,
        "previous_prediction_sha256": previous_prediction_sha256,
    }


def _validate_recovery_state(
    recovery: Mapping[str, Any],
    *,
    identity: Mapping[str, str],
    maximum_rows: int,
) -> None:
    if recovery.get("recovery_version") != PREDICTION_RECOVERY_VERSION:
        raise ValueError("Unsupported checkpoint-evaluation recovery receipt")
    _validate_resume_identity(recovery, identity, label="Recovery receipt")
    try:
        completed_rows = int(recovery.get("completed_rows", -1))
        previous_rows = int(recovery.get("previous_completed_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Recovery receipt row count is invalid") from exc
    if not (1 <= completed_rows <= maximum_rows) or not (
        0 <= previous_rows < completed_rows
    ):
        raise ValueError("Recovery receipt row transition is invalid")
    target_sha = str(recovery.get("prediction_sha256", ""))
    previous_sha = str(recovery.get("previous_prediction_sha256", ""))
    if not is_sha256(target_sha) or (
        previous_rows == 0 and previous_sha
    ) or (previous_rows > 0 and not is_sha256(previous_sha)):
        raise ValueError("Recovery receipt prediction hash is invalid")


def _publish_prediction_progress(
    *,
    prediction_path: Path,
    resume_path: Path,
    recovery_path: Path,
    rows: Sequence[Mapping[str, Any]],
    previous_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, str],
) -> None:
    target_sha = hashlib.sha256(_prediction_csv_bytes(rows)).hexdigest()
    previous_sha = (
        hashlib.sha256(_prediction_csv_bytes(previous_rows)).hexdigest()
        if previous_rows
        else ""
    )
    _atomic_write_json(
        recovery_path,
        _recovery_payload(
            prediction_sha256=target_sha,
            row_count=len(rows),
            previous_prediction_sha256=previous_sha,
            previous_row_count=len(previous_rows),
            identity=identity,
        ),
    )
    published_sha = _atomic_write_prediction(prediction_path, rows)
    if published_sha != target_sha:
        raise RuntimeError("Prediction differs from its write-ahead recovery receipt")
    _atomic_write_json(
        resume_path,
        _resume_payload(
            prediction_path=prediction_path,
            row_count=len(rows),
            identity=identity,
        ),
    )
    recovery_path.unlink()


def _prepare_prediction_output(
    *,
    prediction_path: Path,
    provenance_path: Path,
    resume_state_path: Path,
    recovery_path: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    train_type: str,
    lambda_tone: float,
    seed: int,
    identity: Mapping[str, str],
    resume: bool,
    overwrite: bool,
) -> tuple[list[dict[str, str]], bool]:
    """Verify/recover existing state before any model or audio access.

    Returns ``(prefix, completed)``.  Recovery is allowed only when the
    write-ahead receipt authenticates the exact published CSV bytes.
    """

    artifacts = (
        prediction_path,
        provenance_path,
        resume_state_path,
        recovery_path,
    )
    existing = [path for path in artifacts if path.exists()]
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if overwrite:
        for path in existing:
            if not path.is_file():
                raise ValueError(f"Evaluation artifact is not a file: {path}")
            path.unlink()
        return [], False
    if not existing:
        return [], False
    if not resume:
        raise FileExistsError(
            "Evaluation output already exists. Use --resume to verify/recover it "
            "or --overwrite to start over explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    if provenance_path.exists() and not prediction_path.exists():
        raise ValueError("Orphan provenance exists without its prediction CSV")
    if resume_state_path.exists() and not prediction_path.exists():
        raise ValueError("Orphan resume state exists without its prediction CSV")
    if provenance_path.exists() and (
        resume_state_path.exists() or recovery_path.exists()
    ):
        raise ValueError("Completed provenance coexists with partial state")

    if provenance_path.exists():
        rows = _read_partial_prediction(prediction_path)
        _validate_prediction_prefix(
            rows,
            manifest_rows,
            train_type=train_type,
            lambda_tone=lambda_tone,
            seed=seed,
        )
        if len(rows) != len(manifest_rows):
            raise ValueError("Completed provenance is attached to a partial prediction")
        provenance = load_prediction_provenance(prediction_path)
        if provenance.get("provenance_version") != PREDICTION_PROVENANCE_VERSION:
            raise ValueError(
                "Completed artifact predates resumable provenance; refusing reuse"
            )
        _validate_resume_identity(provenance, identity, label="Provenance")
        if int(provenance.get("num_rows", -1)) != len(rows):
            raise ValueError("Completed provenance row count differs from prediction")
        return rows, True

    recovery: dict[str, Any] | None = None
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        if not isinstance(recovery, dict):
            raise ValueError("Recovery receipt must be a JSON object")
        _validate_recovery_state(
            recovery, identity=identity, maximum_rows=len(manifest_rows)
        )

    if not prediction_path.exists():
        if recovery is None:
            raise ValueError("Partial state exists without a prediction or receipt")
        if int(recovery.get("previous_completed_rows", -1)) != 0 or str(
            recovery.get("previous_prediction_sha256", "")
        ):
            raise ValueError("Recovery receipt expects a missing previous prediction")
        recovery_path.unlink()
        return [], False

    rows = _read_partial_prediction(prediction_path)
    _validate_prediction_prefix(
        rows,
        manifest_rows,
        train_type=train_type,
        lambda_tone=lambda_tone,
        seed=seed,
    )
    actual_sha = sha256_file(prediction_path)
    if recovery is not None:
        target_matches = (
            actual_sha == str(recovery["prediction_sha256"]).casefold()
            and len(rows) == int(recovery["completed_rows"])
        )
        previous_matches = (
            resume_state_path.exists()
            and actual_sha
            == str(recovery.get("previous_prediction_sha256", "")).casefold()
            and len(rows) == int(recovery.get("previous_completed_rows", -1))
        )
        if target_matches:
            _atomic_write_json(
                resume_state_path,
                _resume_payload(
                    prediction_path=prediction_path,
                    row_count=len(rows),
                    identity=identity,
                ),
            )
        elif previous_matches:
            state = json.loads(resume_state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("Resume state must be a JSON object")
            _validate_resume_state(
                state,
                prediction_path=prediction_path,
                row_count=len(rows),
                identity=identity,
            )
        else:
            raise ValueError(
                "Prediction matches neither exact hash/row state in its recovery "
                "receipt; refusing possible tamper"
            )
        recovery_path.unlink()
    else:
        if not resume_state_path.exists():
            raise ValueError(
                "Prediction has neither provenance, resume state, nor recovery receipt"
            )
        state = json.loads(resume_state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Resume state must be a JSON object")
        _validate_resume_state(
            state,
            prediction_path=prediction_path,
            row_count=len(rows),
            identity=identity,
        )
    return rows, False


def _verify_configured_noisy_dev(
    config: Mapping[str, Any],
    *,
    split_lock: Mapping[str, Any],
    method_integrity: Mapping[str, str],
) -> dict[str, Any]:
    """Bind derived noisy-dev to its source split, MUSAN lock and method lock."""

    evaluation = config.get("evaluation", {})
    protocol = config.get("protocol", {})
    if str(evaluation.get("benchmark_protocol", "locked_vivos")).casefold() != "noisy_dev":
        raise ValueError("Noisy-dev verification requires benchmark_protocol=noisy_dev")
    if str(evaluation.get("data_split", "")).casefold() != "dev":
        raise ValueError("The noisy-dev benchmark can only be evaluated as data_split=dev")

    source_dev_sha256 = str(
        split_lock.get("splits", {}).get("dev", {}).get("manifest_sha256", "")
    ).casefold()
    configured_source_sha256 = str(
        evaluation.get("expected_source_dev_sha256", "")
    ).casefold()
    if not is_sha256(configured_source_sha256) or configured_source_sha256 != source_dev_sha256:
        raise ValueError(
            "Noisy-dev expected_source_dev_sha256 differs from the VIVOS split lock"
        )

    expected_noise_lock_sha256 = str(
        evaluation.get("expected_noise_split_lock_sha256", "")
    ).casefold()
    expected_noisy_dev_lock_sha256 = str(
        evaluation.get("expected_noisy_dev_lock_sha256", "")
    ).casefold()
    if not is_sha256(expected_noise_lock_sha256) or not is_sha256(
        expected_noisy_dev_lock_sha256
    ):
        raise ValueError("Noisy-dev config is missing an immutable protocol lock hash")

    # Formal method verification has already checked every audio byte. Re-run the
    # lightweight lock/path audit here to bind the explicit evaluation config to
    # that same dependency without hashing 14,125 audio files twice.
    noisy_dev = verify_noisy_dev_lock(
        str(evaluation.get("noisy_dev_lock", "")),
        repo_root=REPO_ROOT,
        expected_noise_lock_sha256=expected_noise_lock_sha256,
        expected_source_dev_sha256=configured_source_sha256,
        verify_audio=(
            bool(protocol.get("verify_audio_sha256", True))
            and not bool(method_integrity)
        ),
    )
    if str(noisy_dev["lock_sha256"]).casefold() != expected_noisy_dev_lock_sha256:
        raise ValueError("Noisy-dev lock SHA-256 differs from the configured hash")

    expected_manifest_sha256 = str(
        evaluation.get("expected_manifest_sha256", "")
    ).casefold()
    configured_manifest = _resolved_repo_path(evaluation.get("manifest", ""))
    if (
        configured_manifest != Path(noisy_dev["manifest_path"]).resolve()
        or str(noisy_dev["manifest_sha256"]).casefold()
        != expected_manifest_sha256
        or int(noisy_dev["rows"])
        != int(evaluation.get("expected_total_rows", -1))
    ):
        raise ValueError(
            "Evaluation manifest/path/row count differs from the locked noisy-dev output"
        )

    if method_integrity:
        method_bindings = {
            "protocol_split_lock_sha256": sha256_file(
                _resolved_repo_path(protocol.get("split_lock", ""))
            ),
            "noise_split_lock_sha256": expected_noise_lock_sha256,
            "noisy_dev_lock_sha256": expected_noisy_dev_lock_sha256,
            "noisy_dev_manifest_sha256": expected_manifest_sha256,
        }
        for field, expected in method_bindings.items():
            if str(method_integrity.get(field, "")).casefold() != expected:
                raise ValueError(
                    f"Noisy-dev {field} differs from the verified method lock"
                )

    return {
        **noisy_dev,
        "audio_hashes_verified": bool(protocol.get("verify_audio_sha256", True)),
        "split_lock_sha256": sha256_file(
            _resolved_repo_path(protocol.get("split_lock", ""))
        ),
        "noise_split_lock_sha256": expected_noise_lock_sha256,
        "benchmark_protocol": "noisy_dev",
    }


def run_checkpoint_evaluation(
    config: ConfigDict,
    *,
    checkpoint: str | Path,
    output_path: str | Path | None = None,
    manifest: str | Path | None = None,
    subset: str = "all",
    snrs: Sequence[str] | None = None,
    noise_types: Sequence[str] | None = None,
    limit: int | None = None,
    batch_size: int = 1,
    device_arg: str = "auto",
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    evaluation = config["evaluation"]
    protocol = config["protocol"]
    model_config = config["model"]
    training_scope = str(
        config.get("training", {}).get("run_scope", "")
    ).strip().casefold()
    configured_data_split = str(
        evaluation.get("data_split", "")
    ).strip().casefold()
    benchmark_protocol = str(
        evaluation.get("benchmark_protocol", "locked_vivos")
    ).strip().casefold()
    if benchmark_protocol not in {"locked_vivos", "noisy_dev"}:
        raise ValueError(
            "Unsupported evaluation.benchmark_protocol; expected locked_vivos or noisy_dev"
        )
    if (
        configured_data_split == "test"
        and protocol.get("final_test_unlocked") is not True
    ):
        raise ValueError(
            "Final test is locked by protocol.final_test_unlocked=false. "
            "The evaluator refused access before reading the manifest, "
            "checkpoint, or output paths."
        )
    method_integrity: dict[str, str] = {}
    if training_scope == "formal":
        method_lock = protocol.get("method_lock")
        if not method_lock:
            raise ValueError(
                "Formal paper-v2 evaluation requires protocol.method_lock"
            )
        method_integrity = verify_method_lock(
            method_lock,
            config=config,
            repo_root=REPO_ROOT,
            formal=True,
            verify_audio=bool(protocol.get("verify_audio_sha256", True)),
        )
    split_lock = load_split_lock(protocol["split_lock"])
    noisy_dev_integrity: dict[str, Any] | None = None
    if benchmark_protocol == "noisy_dev":
        noisy_dev_integrity = _verify_configured_noisy_dev(
            config,
            split_lock=split_lock,
            method_integrity=method_integrity,
        )
    configured_batch_size = int(evaluation.get("batch_size", 1))
    if batch_size != configured_batch_size:
        raise ValueError(
            "Evaluation batch_size differs from the locked config: "
            f"runtime={batch_size}, config={configured_batch_size}"
        )
    checkpoint_root, adapter_path = resolve_checkpoint(checkpoint)
    if method_integrity:
        verify_checkpoint_method_binding(checkpoint_root, method_integrity)
    prediction_path = Path(output_path or evaluation["prediction_path"])
    provenance_path = prediction_provenance_path(prediction_path)
    resume_state_path = prediction_resume_path(prediction_path)
    recovery_path = prediction_recovery_path(prediction_path)

    configured_manifest_path = Path(str(evaluation["manifest"]))
    manifest_path = Path(manifest or configured_manifest_path)
    if (
        manifest is not None
        and configured_data_split in {"dev", "test"}
        and manifest_path.resolve() != configured_manifest_path.resolve()
    ):
        raise ValueError(
            "Locked VIVOS dev/test evaluation forbids manifest path overrides"
        )
    test_lock_item = split_lock["splits"]["test_locked"]
    canonical_test_manifest_path = Path(str(test_lock_item["manifest"]))
    if not canonical_test_manifest_path.is_absolute():
        canonical_test_manifest_path = Path.cwd() / canonical_test_manifest_path
    configured_expected_manifest_sha256 = str(
        evaluation.get("expected_manifest_sha256", "")
    ).strip().casefold()
    if configured_data_split != "test" and (
        manifest_path.resolve() == canonical_test_manifest_path.resolve()
        or configured_expected_manifest_sha256
        == str(test_lock_item["manifest_sha256"]).casefold()
    ):
        raise ValueError(
            "The sealed VIVOS test manifest cannot be relabeled as a non-test "
            "evaluation split"
        )
    locked_split = str(evaluation.get("locked_vivos_split", "")).strip()
    if configured_data_split in {"dev", "test"}:
        expected_locked_split = (
            "dev" if configured_data_split == "dev" else "test_locked"
        )
        if locked_split != expected_locked_split:
            raise ValueError(
                "Evaluation config does not bind the declared VIVOS split to "
                f"{expected_locked_split!r}"
            )
        if benchmark_protocol == "locked_vivos" or configured_data_split == "test":
            canonical_manifest_path = Path(
                str(split_lock["splits"][expected_locked_split]["manifest"])
            )
            if not canonical_manifest_path.is_absolute():
                canonical_manifest_path = Path.cwd() / canonical_manifest_path
            if manifest_path.resolve() != canonical_manifest_path.resolve():
                raise ValueError(
                    "Locked VIVOS dev/test manifest path differs from the canonical "
                    f"split-lock path for {expected_locked_split!r}"
                )

    runtime_evaluation_contract = evaluation_contract_payload(config)
    runtime_evaluation_contract_sha256 = evaluation_contract_sha256(config)
    configured_evaluation_contract_sha256 = str(
        config.get("selection", {}).get(
            "expected_evaluation_contract_sha256", ""
        )
    ).casefold()
    decision_integrity: dict[str, Any] | None = None
    checkpoint_identity: dict[str, str] | None = None
    if configured_data_split == "test":
        decision_integrity = verify_test_decision_lock(
            split_lock_path=protocol["split_lock"],
            decision_lock_path=protocol["decision_lock"],
        )
        if (
            decision_integrity["selection_manifest_sha256"]
            != str(config["selection"]["expected_manifest_sha256"]).casefold()
        ):
            raise ValueError(
                "Decision lock does not bind the configured dev selection manifest"
            )
        if (
            decision_integrity["selection_evaluation_contract_sha256"]
            != configured_evaluation_contract_sha256
        ):
            raise ValueError(
                "Decision lock does not bind the configured dev evaluation contract"
            )
        if decision_integrity["selection_rule_sha256"] != selection_rule_sha256(
            config["selection"]
        ):
            raise ValueError(
                "Decision lock does not bind the configured selection rule"
            )
        if (
            runtime_evaluation_contract_sha256
            not in decision_integrity[
                "allowed_test_evaluation_contract_sha256"
            ]
        ):
            raise ValueError(
                "Decision lock does not allow this test evaluation contract"
            )
        checkpoint_identity = verify_checkpoint_config(checkpoint_root, config)
        verify_test_configuration_locked(
            decision_integrity,
            config=config,
            checkpoint_identity=checkpoint_identity,
        )

    full_rows = load_benchmark_rows(manifest_path)
    rows = load_benchmark_rows(
        manifest_path,
        subset=subset,
        snrs=snrs,
        noise_types=noise_types,
        limit=limit,
    )
    observed_evaluation_split = rows[0]["evaluation_split"]
    expected_evaluation_split = configured_data_split
    if observed_evaluation_split != expected_evaluation_split:
        raise ValueError(
            "Evaluation manifest split does not match the locked config: "
            f"manifest={observed_evaluation_split!r}, config={expected_evaluation_split!r}"
        )
    verify_audio = bool(protocol.get("verify_audio_sha256", True))
    if observed_evaluation_split == "dev" and benchmark_protocol == "noisy_dev":
        if noisy_dev_integrity is None:
            raise RuntimeError("Noisy-dev protocol preflight did not complete")
        manifest_integrity = noisy_dev_integrity
    elif observed_evaluation_split in {"dev", "test"}:
        expected_locked_split = (
            "dev" if observed_evaluation_split == "dev" else "test_locked"
        )
        if locked_split != expected_locked_split:
            raise ValueError(
                "Evaluation config does not bind the declared VIVOS split to "
                f"{expected_locked_split!r}"
            )
        manifest_integrity = verify_locked_vivos_manifest(
            manifest_path,
            split_name=expected_locked_split,
            split_lock_path=protocol["split_lock"],
            verify_audio=verify_audio,
        )
        if (
            manifest_integrity["manifest_sha256"]
            != str(evaluation["expected_manifest_sha256"]).casefold()
        ):
            raise ValueError(
                "Evaluation manifest hash differs from the hash locked in config"
            )
    else:
        manifest_integrity = verify_expected_manifest(
            manifest_path,
            expected_sha256=str(evaluation["expected_manifest_sha256"]),
            expected_rows=evaluation.get("expected_total_rows"),
            verify_audio=verify_audio,
        )
    if (
        observed_evaluation_split == "dev"
        and runtime_evaluation_contract_sha256
        != configured_evaluation_contract_sha256
    ):
        raise ValueError(
            "Runtime dev evaluation contract differs from the locked selection config"
        )
    if observed_evaluation_split == "test":
        if protocol.get("final_test_unlocked") is not True:
            raise ValueError(
                "Final test is locked by protocol.final_test_unlocked=false. "
                "Only set it true after Gates 2-3 and the reviewed decision-v3 "
                "artifact are complete."
            )
        if decision_integrity is None or checkpoint_identity is None:
            raise RuntimeError("Test authorization preflight did not complete")
        if (
            decision_integrity["test_manifest_sha256"]
            != manifest_integrity["manifest_sha256"]
        ):
            raise ValueError("Decision lock does not unlock this test manifest")

    is_full_manifest_evaluation = (
        subset == "all"
        and not snrs
        and not noise_types
        and limit is None
        and len(rows) == len(full_rows)
    )
    configured_prediction_path = Path(str(evaluation["prediction_path"]))
    if (
        not is_full_manifest_evaluation
        and prediction_path.resolve() == configured_prediction_path.resolve()
    ):
        raise ValueError(
            "Filtered/limited evaluation must use a separate smoke output path"
        )
    row_selection_hash = selected_rows_sha256(rows)
    expected_total = evaluation.get("expected_total_rows")
    if (
        expected_total is not None
        and subset == "all"
        and not snrs
        and not noise_types
        and limit is None
        and len(rows) != int(expected_total)
    ):
        raise ValueError(f"Benchmark has {len(rows)} rows, expected {int(expected_total)}")
    if checkpoint_identity is None:
        checkpoint_identity = verify_checkpoint_config(checkpoint_root, config)
    if (
        checkpoint_identity["training_scope"] != "formal"
        and prediction_path.resolve() == configured_prediction_path.resolve()
    ):
        raise ValueError(
            "A smoke-trained checkpoint cannot write the configured formal "
            "evaluation output"
        )
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    precision = str(evaluation.get("inference_precision", "")).casefold()
    if precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError(
                "The locked evaluation contract requires fp16 on CUDA"
            )
        dtype = torch.float16
    elif precision == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported inference precision: {precision!r}")

    train_type = str(config["experiment"]["train_type"])
    lambda_tone = float(config["training"]["lambda_tone"])
    seed = int(config["seed"])
    config_identity_sha256 = _config_identity_sha256(config)
    raw_config_path = str(config.get("_config_path", "")).strip()
    if not raw_config_path or not Path(raw_config_path).is_file():
        raise ValueError("Evaluation config must retain its readable _config_path")
    config_file_sha256 = sha256_file(raw_config_path)
    if training_scope == "formal":
        config_reference = _repo_relative_reference(
            raw_config_path, label="config path"
        )
        manifest_reference = _repo_relative_reference(
            manifest_path, label="manifest path"
        )
        checkpoint_reference = _repo_relative_reference(
            checkpoint_root, label="checkpoint path"
        )
        _repo_relative_reference(prediction_path, label="prediction path")
    else:
        config_reference = raw_config_path
        manifest_reference = str(manifest_path)
        checkpoint_reference = str(checkpoint_root)
    resume_contract = {
        "contract_version": "prediction_evaluation_run_v1",
        "schema": PREDICTION_COLUMNS,
        "config_path": config_reference,
        "config_file_sha256": config_file_sha256,
        "config_identity_sha256": config_identity_sha256,
        "evaluation_contract_sha256": runtime_evaluation_contract_sha256,
        "manifest": manifest_reference,
        "manifest_sha256": str(manifest_integrity["manifest_sha256"]),
        "manifest_num_rows": len(full_rows),
        "selected_rows_sha256": row_selection_hash,
        "selected_num_rows": len(rows),
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "resolved_config_sha256": checkpoint_identity["resolved_config_sha256"],
        "training_contract_sha256": checkpoint_identity[
            "training_contract_sha256"
        ],
        "train_type": train_type,
        "lambda": f"{lambda_tone:g}",
        "seed": seed,
        "filters": {
            "subset": subset,
            "snrs": list(snrs or []),
            "noise_types": list(noise_types or []),
            "limit": limit,
        },
        "method_lock_sha256": method_integrity.get("method_lock_sha256", ""),
        "method_identity_sha256": method_integrity.get(
            "method_identity_sha256", ""
        ),
    }
    resume_contract_sha256 = canonical_sha256(resume_contract)
    resume_identity = _resume_identity(
        resume_contract_sha256=resume_contract_sha256,
        config_identity_sha256=config_identity_sha256,
        manifest_sha256=str(manifest_integrity["manifest_sha256"]),
        selected_rows_sha256_value=row_selection_hash,
        checkpoint_identity=checkpoint_identity,
    )
    prediction_rows, completed = _prepare_prediction_output(
        prediction_path=prediction_path,
        provenance_path=provenance_path,
        resume_state_path=resume_state_path,
        recovery_path=recovery_path,
        manifest_rows=rows,
        train_type=train_type,
        lambda_tone=lambda_tone,
        seed=seed,
        identity=resume_identity,
        resume=resume,
        overwrite=overwrite,
    )
    if completed:
        return prediction_path

    local_processor = checkpoint_root / "processor"
    processor_source = str(local_processor) if local_processor.exists() else str(model_config["name_or_path"])
    processor = WhisperProcessor.from_pretrained(
        processor_source,
        revision=(
            None
            if local_processor.exists()
            else str(model_config["revision"])
        ),
        language=str(model_config.get("language", "vi")),
        task=str(model_config.get("task", "transcribe")),
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        str(model_config["name_or_path"]),
        revision=str(model_config["revision"]),
    )
    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    sample_rate = int(evaluation.get("sample_rate", config["data"].get("sample_rate", 16000)))
    max_length = int(float(evaluation.get("max_audio_seconds", 15.0)) * sample_rate)
    max_new_tokens = int(evaluation.get("max_new_tokens", 128))
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "language": str(model_config.get("language", "vi")),
        "task": str(model_config.get("task", "transcribe")),
        "do_sample": False,
        "num_beams": 1,
    }
    with torch.inference_mode():
        remaining_rows = rows[len(prediction_rows) :]
        batches = list(_batched(remaining_rows, batch_size))
        for row_batch in tqdm(batches, desc=f"evaluate lambda={lambda_tone:g}"):
            previous_prediction_rows = list(prediction_rows)
            waveforms = []
            for row in row_batch:
                waveform = read_audio(row["audio_path"], sr=sample_rate)
                waveforms.append(waveform[:max_length])
            feature_batch = processor.feature_extractor(
                waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = feature_batch.input_features.to(device=device, dtype=dtype)
            attention_mask = feature_batch.attention_mask.to(device=device)
            try:
                generated = model.generate(input_features, attention_mask=attention_mask, **generate_kwargs)
            except TypeError:
                fallback = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                    "forced_decoder_ids": processor.get_decoder_prompt_ids(
                        language=generate_kwargs["language"],
                        task=generate_kwargs["task"],
                    ),
                }
                generated = model.generate(input_features, attention_mask=attention_mask, **fallback)
            hypotheses = processor.batch_decode(generated, skip_special_tokens=True)
            if len(hypotheses) != len(row_batch):
                raise RuntimeError(
                    "Decoder returned a different number of hypotheses than inputs: "
                    f"{len(hypotheses)} != {len(row_batch)}"
                )
            for row, hypothesis in zip(row_batch, hypotheses):
                prediction_rows.append(
                    {
                        "utt_id": row["utt_id"],
                        "dataset": row["dataset"],
                        "model": "phowhisper",
                        "model_size": "base",
                        "train_type": train_type,
                        "lambda": f"{lambda_tone:g}",
                        "seed": seed,
                        "snr": row["snr"],
                        "noise_type": row["noise_type"],
                        "ref": row["ref"],
                        "hyp": hypothesis,
                    }
                )
            _publish_prediction_progress(
                prediction_path=prediction_path,
                resume_path=resume_state_path,
                recovery_path=recovery_path,
                rows=prediction_rows,
                previous_rows=previous_prediction_rows,
                identity=resume_identity,
            )

    if len(prediction_rows) != len(rows):
        raise RuntimeError(
            f"Prediction row count mismatch: {len(prediction_rows)} != {len(rows)}"
        )
    if not resume_state_path.is_file():
        raise RuntimeError("Completed prediction is missing verified resume state")
    provenance = {
        "provenance_version": PREDICTION_PROVENANCE_VERSION,
        **resume_identity,
        "config_path": config_reference,
        "config_file_sha256": config_file_sha256,
        "evaluation_split": observed_evaluation_split,
        "manifest": manifest_reference,
        "manifest_sha256": manifest_integrity["manifest_sha256"],
        "manifest_num_rows": len(full_rows),
        "audio_hashes_verified": manifest_integrity["audio_hashes_verified"],
        "evaluation_scope": (
            "full_manifest" if is_full_manifest_evaluation else "partial"
        ),
        "selected_rows_sha256": row_selection_hash,
        "training_scope": checkpoint_identity["training_scope"],
        "training_contract_sha256": checkpoint_identity[
            "training_contract_sha256"
        ],
        "evaluation_contract": runtime_evaluation_contract,
        "evaluation_contract_sha256": runtime_evaluation_contract_sha256,
        "runtime_environment": {
            "batch_size": batch_size,
            "device_type": device.type,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
        },
        "filters": {
            "subset": subset,
            "snrs": list(snrs or []),
            "noise_types": list(noise_types or []),
            "limit": limit,
        },
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "resolved_config_sha256": checkpoint_identity["resolved_config_sha256"],
        "prediction_sha256": sha256_file(prediction_path),
        "num_rows": len(prediction_rows),
        "metric_version": METRIC_VERSION,
        "method_lock_sha256": method_integrity.get("method_lock_sha256", ""),
        "method_identity_sha256": method_integrity.get(
            "method_identity_sha256", ""
        ),
        "environment_artifact_sha256": method_integrity.get(
            "environment_artifact_sha256", ""
        ),
        "environment_identity_sha256": method_integrity.get(
            "environment_identity_sha256", ""
        ),
        "source_tree_sha256": method_integrity.get("source_tree_sha256", ""),
        "benchmark_protocol": benchmark_protocol,
        "split_lock_sha256": manifest_integrity.get("split_lock_sha256", ""),
        "noise_split_lock_sha256": manifest_integrity.get(
            "noise_split_lock_sha256", ""
        ),
        "noisy_dev_lock_sha256": (
            ""
            if noisy_dev_integrity is None
            else noisy_dev_integrity["lock_sha256"]
        ),
        "decision_lock_sha256": (
            ""
            if decision_integrity is None
            else decision_integrity["decision_lock_sha256"]
        ),
    }
    if training_scope == "formal":
        _assert_no_absolute_paths(provenance, label="formal provenance")
    _atomic_write_new_json(provenance_path, provenance)
    resume_state_path.unlink()
    return prediction_path


def validate_prediction_schema(path: str | Path) -> list[dict[str, str]]:
    rows, columns = read_csv_rows(path)
    validate_columns(path, columns, PREDICTION_COLUMNS, exact=True)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return validate_prediction_rows(rows, source=path)


def load_prediction_provenance(prediction_path: str | Path) -> dict[str, Any]:
    path = Path(prediction_path)
    provenance_path = prediction_provenance_path(path)
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Missing prediction evaluation provenance: {provenance_path}"
        )
    try:
        value = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid prediction provenance JSON: {provenance_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Prediction provenance must be a JSON object: {provenance_path}")
    provenance_version = str(value.get("provenance_version", ""))
    if provenance_version not in {
        "prediction_evaluation_v3",
        PREDICTION_PROVENANCE_VERSION,
    }:
        raise ValueError(f"Unsupported prediction provenance version: {provenance_path}")
    evaluation_split = str(value.get("evaluation_split", "")).strip().casefold()
    if evaluation_split not in {"dev", "test", "external"}:
        raise ValueError(f"Invalid evaluation_split in {provenance_path}")
    expected_prediction_hash = str(value.get("prediction_sha256", "")).casefold()
    if not is_sha256(expected_prediction_hash):
        raise ValueError(f"Invalid prediction_sha256 in {provenance_path}")
    actual_prediction_hash = sha256_file(path)
    if expected_prediction_hash != actual_prediction_hash:
        raise ValueError(
            f"Prediction SHA-256 does not match provenance: {path}"
        )
    manifest_hash = str(value.get("manifest_sha256", "")).casefold()
    if not is_sha256(manifest_hash):
        raise ValueError(f"Invalid manifest_sha256 in {provenance_path}")
    manifest_path = _resolved_repo_path(value.get("manifest", ""))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prediction provenance manifest does not exist: {manifest_path}"
        )
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError(
            f"Manifest SHA-256 does not match prediction provenance: {manifest_path}"
        )
    if value.get("metric_version") != METRIC_VERSION:
        raise ValueError(
            f"Prediction provenance must use metric_version={METRIC_VERSION}"
        )
    if value.get("audio_hashes_verified") is not True:
        raise ValueError(
            f"Prediction provenance must confirm audio SHA-256 verification: {provenance_path}"
        )
    if evaluation_split in {"dev", "test"} and not is_sha256(
        value.get("split_lock_sha256")
    ):
        raise ValueError(f"Invalid split_lock_sha256 in {provenance_path}")
    benchmark_protocol = str(
        value.get("benchmark_protocol", "locked_vivos")
    ).strip().casefold()
    if benchmark_protocol not in {"locked_vivos", "noisy_dev"}:
        raise ValueError(f"Invalid benchmark_protocol in {provenance_path}")
    if benchmark_protocol == "noisy_dev":
        if evaluation_split != "dev":
            raise ValueError(
                f"noisy_dev provenance must use evaluation_split=dev: {provenance_path}"
            )
        for field in ("noise_split_lock_sha256", "noisy_dev_lock_sha256"):
            if not is_sha256(value.get(field)):
                raise ValueError(f"Invalid {field} in {provenance_path}")
    if evaluation_split == "test" and not is_sha256(
        value.get("decision_lock_sha256")
    ):
        raise ValueError(f"Invalid decision_lock_sha256 in {provenance_path}")
    if not is_sha256(value.get("checkpoint_sha256")):
        raise ValueError(f"Invalid checkpoint_sha256 in {provenance_path}")
    if not is_sha256(value.get("resolved_config_sha256")):
        raise ValueError(f"Invalid resolved_config_sha256 in {provenance_path}")
    if not is_sha256(value.get("selected_rows_sha256")):
        raise ValueError(f"Invalid selected_rows_sha256 in {provenance_path}")
    training_scope = str(value.get("training_scope", "")).strip().casefold()
    if training_scope not in {"formal", "smoke"}:
        raise ValueError(f"Invalid training_scope in {provenance_path}")
    if training_scope == "formal":
        for field in (
            "method_lock_sha256",
            "method_identity_sha256",
            "environment_artifact_sha256",
            "environment_identity_sha256",
            "source_tree_sha256",
        ):
            if not is_sha256(value.get(field)):
                raise ValueError(
                    f"Invalid {field} in formal prediction provenance: {provenance_path}"
                )
        if provenance_version == PREDICTION_PROVENANCE_VERSION:
            for field in ("config_path", "manifest", "checkpoint"):
                reference = str(value.get(field, ""))
                if _repo_relative_reference(reference, label=field) != reference:
                    raise ValueError(
                        f"Formal provenance {field} is not canonical: {provenance_path}"
                    )
            config_path = _resolved_repo_path(value.get("config_path", ""))
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Formal provenance config does not exist: {config_path}"
                )
            if sha256_file(config_path) != str(
                value.get("config_file_sha256", "")
            ).casefold():
                raise ValueError(
                    f"Formal provenance config SHA-256 mismatch: {config_path}"
                )
            from .config import load_experiment_config

            recorded_config = load_experiment_config(config_path)
            if _config_identity_sha256(recorded_config) != str(
                value.get("config_identity_sha256", "")
            ).casefold():
                raise ValueError(
                    f"Formal provenance resolved config identity mismatch: "
                    f"{config_path}"
                )
            for field in ("resume_contract_sha256", "config_identity_sha256"):
                if not is_sha256(value.get(field)):
                    raise ValueError(
                        f"Invalid {field} in formal prediction provenance: "
                        f"{provenance_path}"
                    )
    if not is_sha256(value.get("training_contract_sha256")):
        raise ValueError(
            f"Invalid training_contract_sha256 in {provenance_path}"
        )
    evaluation_contract = value.get("evaluation_contract")
    if not isinstance(evaluation_contract, Mapping):
        raise ValueError(f"Missing evaluation_contract in {provenance_path}")
    recorded_evaluation_contract_hash = str(
        value.get("evaluation_contract_sha256", "")
    ).casefold()
    if (
        not is_sha256(recorded_evaluation_contract_hash)
        or canonical_sha256(evaluation_contract)
        != recorded_evaluation_contract_hash
    ):
        raise ValueError(
            f"Evaluation contract SHA-256 mismatch in {provenance_path}"
        )
    runtime_environment = value.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        raise ValueError(f"Missing runtime_environment in {provenance_path}")
    contract_evaluation = evaluation_contract.get("evaluation", {})
    if str(
        contract_evaluation.get("benchmark_protocol", "locked_vivos")
    ).strip().casefold() != benchmark_protocol:
        raise ValueError(
            f"Provenance benchmark protocol differs from evaluation contract: {provenance_path}"
        )
    if benchmark_protocol == "noisy_dev":
        lock_bindings = {
            "expected_noisy_dev_lock_sha256": "noisy_dev_lock_sha256",
            "expected_noise_split_lock_sha256": "noise_split_lock_sha256",
        }
        for contract_field, provenance_field in lock_bindings.items():
            if str(contract_evaluation.get(contract_field, "")).casefold() != str(
                value.get(provenance_field, "")
            ).casefold():
                raise ValueError(
                    f"Noisy-dev {provenance_field} differs from evaluation contract: "
                    f"{provenance_path}"
                )
    if int(runtime_environment.get("batch_size", 0)) != int(
        contract_evaluation.get("batch_size", 0)
    ):
        raise ValueError(
            f"Runtime batch size does not match evaluation contract: {provenance_path}"
        )
    precision = str(contract_evaluation.get("inference_precision", ""))
    expected_dtype = "torch.float16" if precision == "fp16" else "torch.float32"
    if str(runtime_environment.get("dtype", "")) != expected_dtype:
        raise ValueError(
            f"Runtime dtype does not match evaluation contract: {provenance_path}"
        )
    if precision == "fp16" and runtime_environment.get("device_type") != "cuda":
        raise ValueError(
            f"FP16 evaluation provenance must record CUDA: {provenance_path}"
        )
    for field in ("torch_version", "transformers_version"):
        if not str(runtime_environment.get(field, "")).strip():
            raise ValueError(
                f"Runtime environment is missing {field}: {provenance_path}"
            )
    scope = str(value.get("evaluation_scope", ""))
    if scope not in {"full_manifest", "partial"}:
        raise ValueError(f"Invalid evaluation_scope in {provenance_path}")
    filters = value.get("filters")
    if not isinstance(filters, Mapping):
        raise ValueError(f"Missing evaluation filters in {provenance_path}")
    subset = str(filters.get("subset", ""))
    snrs = filters.get("snrs")
    noise_types = filters.get("noise_types")
    limit = filters.get("limit")
    if not isinstance(snrs, list) or not isinstance(noise_types, list):
        raise ValueError(f"Invalid evaluation filters in {provenance_path}")
    selected_manifest_rows = load_benchmark_rows(
        manifest_path,
        subset=subset,
        snrs=[str(item) for item in snrs] or None,
        noise_types=[str(item) for item in noise_types] or None,
        limit=None if limit is None else int(limit),
    )
    full_scope = subset == "all" and not snrs and not noise_types and limit is None
    expected_scope = "full_manifest" if full_scope else "partial"
    if scope != expected_scope:
        raise ValueError(
            f"Evaluation scope does not match recorded filters: {provenance_path}"
        )
    if selected_rows_sha256(selected_manifest_rows) != str(
        value["selected_rows_sha256"]
    ).casefold():
        raise ValueError(
            f"Selected-row SHA-256 does not match provenance: {provenance_path}"
        )
    if int(value.get("manifest_num_rows", -1)) != len(
        load_benchmark_rows(manifest_path)
    ):
        raise ValueError(
            f"Manifest row count does not match provenance: {provenance_path}"
        )
    value["_selected_manifest_rows"] = selected_manifest_rows
    return value


def _result_row(
    rows: Sequence[dict[str, str]],
    *,
    split: str,
    snr: str,
    noise_type: str,
    checkpoint_path: str,
    prediction_path: str,
) -> dict[str, Any]:
    metrics = compute_aligned_metric_result(
        [row["ref"] for row in rows],
        [row["hyp"] for row in rows],
    ).to_dict(include_counts=True)
    first = rows[0]
    return {
        "model": first["model"],
        "model_size": first["model_size"],
        "train_type": first["train_type"],
        "lambda": first["lambda"],
        "seed": first["seed"],
        "split": split,
        "snr": snr,
        "noise_type": noise_type,
        "num_samples": len(rows),
        "metric_version": metrics["metric_version"],
        "wer": metrics["wer"],
        "wer_numerator": metrics["wer_numerator"],
        "wer_denominator": metrics["wer_denominator"],
        "cer": metrics["cer"],
        "cer_numerator": metrics["cer_numerator"],
        "cer_denominator": metrics["cer_denominator"],
        "ter": metrics["ter"],
        "ter_numerator": metrics["ter_numerator"],
        "ter_denominator": metrics["ter_denominator"],
        "ter_coverage": metrics["ter_coverage"],
        "der": metrics["der"],
        "der_numerator": metrics["der_numerator"],
        "der_denominator": metrics["der_denominator"],
        "der_coverage": metrics["der_coverage"],
        "fcer": metrics["fcer"],
        "fcer_numerator": metrics["fcer_numerator"],
        "fcer_denominator": metrics["fcer_denominator"],
        "fcer_coverage": metrics["fcer_coverage"],
        "swdr": metrics["swdr"],
        "swdr_numerator": metrics["swdr_numerator"],
        "swdr_denominator": metrics["swdr_denominator"],
        "checkpoint_path": checkpoint_path,
        "prediction_path": prediction_path,
    }


def aggregate_prediction_file(
    prediction_path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> list[dict[str, Any]]:
    rows = validate_prediction_schema(prediction_path)
    provenance = load_prediction_provenance(prediction_path)
    if int(provenance.get("num_rows", -1)) != len(rows):
        raise ValueError(
            f"Prediction row count does not match provenance: {prediction_path}"
        )
    selected_manifest_rows = provenance.pop("_selected_manifest_rows")
    if len(selected_manifest_rows) != len(rows):
        raise ValueError(
            f"Prediction rows do not cover the recorded manifest selection: {prediction_path}"
        )
    for row_number, (prediction_row, manifest_row) in enumerate(
        zip(rows, selected_manifest_rows),
        start=1,
    ):
        mismatches = [
            field
            for field in ("utt_id", "dataset", "snr", "noise_type", "ref")
            if str(prediction_row[field]) != str(manifest_row[field])
        ]
        if mismatches:
            raise ValueError(
                f"Prediction/manifest mismatch at row {row_number}: {mismatches}"
            )
    recorded_checkpoint = _resolved_repo_path(provenance.get("checkpoint", ""))
    supplied_checkpoint, _ = resolve_checkpoint(checkpoint_path)
    if recorded_checkpoint.resolve() != supplied_checkpoint.resolve():
        raise ValueError(
            "Aggregation checkpoint path does not match prediction provenance: "
            f"{checkpoint_path} != {recorded_checkpoint}"
        )
    actual_checkpoint_hash = checkpoint_inference_sha256(supplied_checkpoint)
    if actual_checkpoint_hash != str(provenance["checkpoint_sha256"]).casefold():
        raise ValueError(
            "Checkpoint fingerprint changed after prediction generation: "
            f"{supplied_checkpoint}"
        )
    actual_resolved_config_hash = sha256_file(
        supplied_checkpoint / "resolved_config.yaml"
    )
    if actual_resolved_config_hash != str(
        provenance["resolved_config_sha256"]
    ).casefold():
        raise ValueError(
            "Checkpoint resolved config changed after prediction generation: "
            f"{supplied_checkpoint}"
        )
    output = [
        _result_row(
            rows,
            split="all",
            snr="all",
            noise_type="all",
            checkpoint_path=str(checkpoint_path),
            prediction_path=str(prediction_path),
        )
    ]
    clean_rows = [row for row in rows if normalize_snr(row["snr"]) == "clean"]
    noisy_rows = [row for row in rows if normalize_snr(row["snr"]) != "clean"]
    if clean_rows:
        output.append(
            _result_row(
                clean_rows,
                split="clean",
                snr="clean",
                noise_type="clean",
                checkpoint_path=str(checkpoint_path),
                prediction_path=str(prediction_path),
            )
        )
    if noisy_rows:
        output.append(
            _result_row(
                noisy_rows,
                split="noisy",
                snr="noisy_all",
                noise_type="all",
                checkpoint_path=str(checkpoint_path),
                prediction_path=str(prediction_path),
            )
        )
        by_snr: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_noise: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_snr_noise: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in noisy_rows:
            snr = normalize_snr(row["snr"])
            noise_type = row["noise_type"] or "unknown"
            by_snr[snr].append(row)
            by_noise[noise_type].append(row)
            by_snr_noise[(snr, noise_type)].append(row)
        for snr in sorted(by_snr, key=lambda value: -float(value)):
            output.append(
                _result_row(
                    by_snr[snr],
                    split="noisy",
                    snr=snr,
                    noise_type="all",
                    checkpoint_path=str(checkpoint_path),
                    prediction_path=str(prediction_path),
                )
            )
        for noise_type in sorted(by_noise):
            output.append(
                _result_row(
                    by_noise[noise_type],
                    split="noisy",
                    snr="all",
                    noise_type=noise_type,
                    checkpoint_path=str(checkpoint_path),
                    prediction_path=str(prediction_path),
                )
            )
        for (snr, noise_type), group_rows in sorted(by_snr_noise.items()):
            output.append(
                _result_row(
                    group_rows,
                    split="noisy",
                    snr=snr,
                    noise_type=noise_type,
                    checkpoint_path=str(checkpoint_path),
                    prediction_path=str(prediction_path),
                )
            )
    for row in output:
        row["evaluation_split"] = str(provenance["evaluation_split"])
        row["manifest_sha256"] = str(provenance["manifest_sha256"])
        row["evaluation_scope"] = str(provenance["evaluation_scope"])
        row["selected_rows_sha256"] = str(provenance["selected_rows_sha256"])
        row["training_scope"] = str(provenance["training_scope"])
        row["training_contract_sha256"] = str(
            provenance["training_contract_sha256"]
        )
        row["evaluation_contract_sha256"] = str(
            provenance["evaluation_contract_sha256"]
        )
        row["method_lock_sha256"] = str(provenance.get("method_lock_sha256", ""))
        row["method_identity_sha256"] = str(
            provenance.get("method_identity_sha256", "")
        )
        row["environment_artifact_sha256"] = str(
            provenance.get("environment_artifact_sha256", "")
        )
        row["environment_identity_sha256"] = str(
            provenance.get("environment_identity_sha256", "")
        )
        row["source_tree_sha256"] = str(provenance.get("source_tree_sha256", ""))
        row["checkpoint_sha256"] = str(provenance["checkpoint_sha256"])
    return output


def write_ablation_results(
    experiment_artifacts: Sequence[tuple[str | Path, str | Path]],
    output_path: str | Path,
    *,
    require_lambdas: Sequence[float] | None = (0.0, 0.05, 0.1, 0.3, 0.5),
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    all_rows: list[dict[str, Any]] = []
    observed_lambdas: set[float] = set()
    shared_contract: tuple[str, str, str, str, str, str, str, str] | None = None
    for checkpoint_path, prediction_path in experiment_artifacts:
        rows = aggregate_prediction_file(prediction_path, checkpoint_path=checkpoint_path)
        contract = (
            str(rows[0]["seed"]),
            str(rows[0]["evaluation_split"]),
            str(rows[0]["manifest_sha256"]),
            str(rows[0]["evaluation_scope"]),
            str(rows[0]["selected_rows_sha256"]),
            str(rows[0]["training_scope"]),
            str(rows[0]["evaluation_contract_sha256"]),
            str(rows[0]["metric_version"]),
        )
        if shared_contract is None:
            shared_contract = contract
        elif contract != shared_contract:
            raise ValueError(
                "Lambda ablation predictions do not share one seed/evaluation contract"
            )
        lambda_value = float(rows[0]["lambda"])
        if lambda_value in observed_lambdas:
            raise ValueError(
                f"Duplicate lambda prediction in one ablation: {lambda_value:g}"
            )
        all_rows.extend(rows)
        observed_lambdas.add(lambda_value)
    if require_lambdas is not None:
        missing = sorted(set(float(value) for value in require_lambdas) - observed_lambdas)
        if missing:
            raise ValueError(f"Cannot write complete lambda ablation; missing real predictions for: {missing}")
    path = Path(output_path)
    if path.exists() and resume:
        existing_rows, columns = read_csv_rows(path)
        validate_columns(path, columns, ABLATION_RESULT_COLUMNS, exact=True)
        expected_rows = [
            {
                column: "" if row.get(column) is None else str(row.get(column, ""))
                for column in ABLATION_RESULT_COLUMNS
            }
            for row in all_rows
        ]
        if existing_rows != expected_rows:
            raise ValueError(
                "Existing lambda ablation result differs from the five verified "
                "prediction artifacts; refusing resume overwrite"
            )
        return path
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Ablation result already exists: {path}. Use --resume to verify it "
            "or --overwrite explicitly."
        )
    atomic_write_csv(path, all_rows, ABLATION_RESULT_COLUMNS)
    return path
