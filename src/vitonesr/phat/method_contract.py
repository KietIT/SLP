from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.vitonesr.analysis import METRIC_VERSION
from src.vitonesr.noise_protocol import (
    NOISY_DEV_PROTOCOL_VERSION,
    NoiseProtocolError,
    verify_noise_split_lock,
)
from src.vitonesr.tone import ID_TO_TONE, IGNORE_INDEX

from .protocol import (
    canonical_sha256,
    is_immutable_revision,
    load_split_lock,
    sha256_file,
    training_contract_sha256,
)
from .reproducibility import (
    EnvironmentCaptureError,
    capture_environment,
    validate_environment_artifact,
)


METHOD_CONTRACT_VERSION = "paper_v2_method_contract_v1"
DEFAULT_METHOD_LOCK = Path("outputs/paper_v2/protocol/method_lock.json")
DEFAULT_LAMBDA_GRID = (0.0, 0.05, 0.1, 0.3, 0.5)
DEFAULT_SOURCE_COMPONENTS = (
    "configs/paper_v2_final_lora.yaml",
    "configs/paper_v2_zero_shot.yaml",
    "configs/phat/base.yaml",
    "configs/phat/lambda_0.yaml",
    "configs/phat/lambda_005.yaml",
    "configs/phat/lambda_01.yaml",
    "configs/phat/lambda_03.yaml",
    "configs/phat/lambda_05.yaml",
    "configs/phat/phat_pipeline.yaml",
    "requirements.txt",
    "src/vitonesr/analysis.py",
    "src/vitonesr/artifact_bundle.py",
    "src/vitonesr/comparison.py",
    "src/vitonesr/data.py",
    "src/vitonesr/final_benchmark.py",
    "src/vitonesr/noise.py",
    "src/vitonesr/noise_protocol.py",
    "src/vitonesr/prediction.py",
    "src/vitonesr/prediction_evidence.py",
    "src/vitonesr/statistics.py",
    "src/vitonesr/text_norm.py",
    "src/vitonesr/tone.py",
    "src/vitonesr/phat/config.py",
    "src/vitonesr/phat/evaluation.py",
    "src/vitonesr/phat/final_evaluation.py",
    "src/vitonesr/phat/losses.py",
    "src/vitonesr/phat/method_contract.py",
    "src/vitonesr/phat/modeling.py",
    "src/vitonesr/phat/protocol.py",
    "src/vitonesr/phat/reproducibility.py",
    "src/vitonesr/phat/selection.py",
    "src/vitonesr/phat/trainer.py",
    "src/vitonesr/phat/training_data.py",
    "src/vitonesr/zero_shot_paper_v2.py",
    "scripts/aggregate_results.py",
    "scripts/audit_tone_alignment.py",
    "scripts/build_error_artifacts.py",
    "scripts/build_error_breakdowns.py",
    "scripts/build_final_benchmark.py",
    "scripts/build_noisy_dev_benchmark.py",
    "scripts/build_old_vs_new_comparison.py",
    "scripts/capture_paper_v2_environment.py",
    "scripts/cluster_bootstrap_ci.py",
    "scripts/download_fleurs.py",
    "scripts/error_analysis.py",
    "scripts/evaluate_all_lambdas.py",
    "scripts/evaluate_phat_checkpoint.py",
    "scripts/lock_musan_noise_protocol.py",
    "scripts/lock_paper_v2_method.py",
    "scripts/make_vivos_manifest.py",
    "scripts/prepare_final_lora_config.py",
    "scripts/run_external_fleurs.py",
    "scripts/select_best_lambda.py",
    "scripts/run_final_lora.py",
    "scripts/run_zero_shot_paper_v2.py",
    "scripts/train_all_lambdas.py",
    "scripts/train_phat_lora.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class MethodContractError(ValueError):
    """Raised when the paper-v2 method contract is absent or inconsistent."""


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


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value).strip().casefold()))


def _looks_absolute(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("\\\\")
        or bool(_WINDOWS_ABSOLUTE_RE.match(value))
    )


def _reject_absolute_paths(value: Any, *, field: str = "method lock") -> None:
    if isinstance(value, str):
        if _looks_absolute(value):
            raise MethodContractError(f"{field} contains an absolute path: {value!r}")
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_absolute_paths(nested, field=f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_absolute_paths(nested, field=f"{field}[{index}]")


def _repo_path(path: str | Path, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MethodContractError(
            f"Contract artifact must stay inside the repository: {resolved}"
        ) from exc
    return resolved


def _display_path(path: str | Path, repo_root: str | Path) -> str:
    return _repo_path(path, repo_root).relative_to(Path(repo_root).resolve()).as_posix()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MethodContractError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MethodContractError(f"{label} must be a JSON object: {path}")
    return value


def _artifact_binding(path: str | Path, repo_root: str | Path) -> dict[str, str]:
    resolved = _repo_path(path, repo_root)
    if not resolved.is_file():
        raise FileNotFoundError(f"Contract artifact does not exist: {resolved}")
    return {
        "path": _display_path(resolved, repo_root),
        "sha256": sha256_file(resolved),
    }


def _verify_binding(
    binding: Mapping[str, Any],
    *,
    repo_root: str | Path,
    label: str,
) -> Path:
    path_value = str(binding.get("path", ""))
    expected = str(binding.get("sha256", "")).casefold()
    if not path_value or _looks_absolute(path_value):
        raise MethodContractError(f"{label} must use a repository-relative path")
    if not _is_sha256(expected):
        raise MethodContractError(f"{label} has no valid SHA-256")
    path = _repo_path(path_value, repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise MethodContractError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MethodContractError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise MethodContractError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise MethodContractError(f"JSONL artifact is empty: {path}")
    return rows


def verify_noisy_dev_lock(
    lock_path: str | Path,
    *,
    repo_root: str | Path,
    expected_noise_lock_sha256: str | None = None,
    expected_source_dev_sha256: str | None = None,
    verify_audio: bool = True,
) -> dict[str, Any]:
    """Verify a noisy-dev lock, its manifest/audit, and optionally every audio byte."""

    lock_file = _repo_path(lock_path, repo_root)
    lock = _load_json(lock_file, label="Noisy-dev lock")
    if (
        lock.get("protocol_version") != NOISY_DEV_PROTOCOL_VERSION
        or lock.get("status") != "LOCKED"
        or lock.get("selection_eligible") is not True
        or lock.get("final_test_eligible") is not False
    ):
        raise MethodContractError(f"Unsupported or unlocked noisy-dev protocol: {lock_file}")

    source = lock.get("source_dev")
    noise = lock.get("noise")
    output = lock.get("output")
    audit = lock.get("audit")
    if not all(isinstance(item, Mapping) for item in (source, noise, output, audit)):
        raise MethodContractError("Noisy-dev lock is missing source/noise/output/audit metadata")
    source = dict(source)  # type: ignore[arg-type]
    noise = dict(noise)  # type: ignore[arg-type]
    output = dict(output)  # type: ignore[arg-type]
    audit = dict(audit)  # type: ignore[arg-type]

    if expected_source_dev_sha256 is not None and str(
        source.get("manifest_sha256", "")
    ).casefold() != str(expected_source_dev_sha256).casefold():
        raise MethodContractError("Noisy-dev lock binds a different source dev manifest")
    if expected_noise_lock_sha256 is not None and str(
        noise.get("split_lock_sha256", "")
    ).casefold() != str(expected_noise_lock_sha256).casefold():
        raise MethodContractError("Noisy-dev lock binds a different noise split lock")
    if noise.get("partition") != "dev":
        raise MethodContractError("Noisy-dev benchmark must use the MUSAN dev partition")

    source_path = _repo_path(str(source.get("manifest", "")), repo_root)
    if sha256_file(source_path) != str(source.get("manifest_sha256", "")).casefold():
        raise MethodContractError("Noisy-dev source manifest hash mismatch")
    manifest_path = _repo_path(str(output.get("manifest", "")), repo_root)
    expected_manifest_hash = str(output.get("manifest_sha256", "")).casefold()
    if not _is_sha256(expected_manifest_hash) or sha256_file(manifest_path) != expected_manifest_hash:
        raise MethodContractError("Noisy-dev output manifest hash mismatch")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(output.get("row_count", -1)):
        raise MethodContractError("Noisy-dev output row count mismatch")

    if verify_audio:
        for row_number, row in enumerate(rows, start=1):
            audio_path = _repo_path(str(row.get("audio_path", "")), repo_root)
            expected_audio_hash = str(row.get("audio_sha256", "")).casefold()
            if not _is_sha256(expected_audio_hash):
                raise MethodContractError(
                    f"Noisy-dev row {row_number} has no valid audio SHA-256"
                )
            if sha256_file(audio_path) != expected_audio_hash:
                raise MethodContractError(
                    f"Noisy-dev audio SHA-256 mismatch at row {row_number}: {audio_path}"
                )

    audit_path = _repo_path(str(audit.get("path", "")), repo_root)
    if sha256_file(audit_path) != str(audit.get("sha256", "")).casefold():
        raise MethodContractError("Noisy-dev audit hash mismatch")
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if len(audit_rows) != int(audit.get("checks", -1)) or any(
        row.get("status") != "PASS" for row in audit_rows
    ):
        raise MethodContractError("Noisy-dev audit is incomplete or contains failures")
    return {
        "lock": lock,
        "lock_sha256": sha256_file(lock_file),
        "manifest_path": manifest_path,
        "manifest_sha256": expected_manifest_hash,
        "rows": len(rows),
        "audio_hashes_verified": bool(verify_audio),
    }


def _lambda_key(value: float) -> str:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise MethodContractError("Every lambda must be finite and non-negative")
    return format(number, ".15g")


def _normalize_lambda_grid(values: Sequence[float]) -> tuple[float, ...]:
    by_key: dict[str, float] = {}
    for raw_value in values:
        value = float(raw_value)
        key = _lambda_key(value)
        if key in by_key:
            raise MethodContractError(f"Duplicate lambda in method grid: {key}")
        by_key[key] = value
    if not by_key or "0" not in by_key or not any(value > 0 for value in by_key.values()):
        raise MethodContractError("Method grid must contain ordinary lambda=0 and tone-aware lambda>0")
    return tuple(by_key[key] for key in sorted(by_key, key=lambda item: float(item)))


def _config_for_lambda(config: Mapping[str, Any], lambda_value: float) -> dict[str, Any]:
    candidate = deepcopy(dict(config))
    candidate.pop("_config_path", None)
    candidate.pop("_runtime_protocol", None)
    candidate.setdefault("training", {})["lambda_tone"] = float(lambda_value)
    ordinary = math.isclose(float(lambda_value), 0.0, rel_tol=0.0, abs_tol=0.0)
    candidate.setdefault("experiment", {})["method_id"] = (
        "ordinary_lora" if ordinary else "corrected_decoder_tone_lora"
    )
    candidate["experiment"]["train_type"] = (
        "ordinary_lora" if ordinary else "tone_aware_lora"
    )
    return candidate


def _decode_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    data = config.get("data", {})
    evaluation = config.get("evaluation", {})
    return {
        "implementation": "whisper_generate_greedy_v1",
        "language": model.get("language", "vi"),
        "task": model.get("task", "transcribe"),
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": int(evaluation.get("max_new_tokens", 128)),
        "sample_rate": int(evaluation.get("sample_rate", data.get("sample_rate", 16000))),
        "max_audio_seconds": float(evaluation.get("max_audio_seconds", 15.0)),
        "inference_precision": str(evaluation.get("inference_precision", "")).casefold(),
        "batch_size": int(evaluation.get("batch_size", 1)),
    }


def _selection_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    selection = config.get("selection", {})
    return {
        "checkpoint_selection_split": "dev",
        "checkpoint_metric": training.get("checkpoint_metric"),
        "checkpoint_mode": training.get("checkpoint_mode"),
        "lambda_selection_split": selection.get("required_evaluation_split"),
        "require_full_manifest": selection.get("require_full_manifest"),
        "selection_manifest_sha256": str(
            selection.get("expected_manifest_sha256", "")
        ).casefold(),
        "rule": dict(selection),
    }


def _source_tree_sha256(components: Sequence[Mapping[str, str]]) -> str:
    return canonical_sha256(
        [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in sorted(components, key=lambda item: item["path"])
        ]
    )


def _validate_environment(
    path: Path,
    *,
    formal: bool,
    repo_root: str | Path | None = None,
    verify_current: bool = False,
) -> dict[str, Any]:
    artifact = _load_json(path, label="Environment artifact")
    try:
        validate_environment_artifact(artifact)
    except EnvironmentCaptureError as exc:
        raise MethodContractError(str(exc)) from exc
    environment = artifact.get("environment", {})
    capture_mode = str(environment.get("capture_mode", ""))
    if formal and capture_mode != "formal":
        raise MethodContractError(
            "A formal method lock requires a formal environment artifact"
        )
    identity = str(artifact.get("identity_sha256", "")).casefold()
    if not _is_sha256(identity):
        raise MethodContractError("Environment artifact has no valid identity SHA-256")
    if verify_current:
        if repo_root is None:
            raise MethodContractError("repo_root is required for current-runtime verification")
        locked_packages = environment.get("packages")
        if not isinstance(locked_packages, Mapping) or not locked_packages:
            raise MethodContractError("Environment artifact has no package inventory")
        current = capture_environment(
            repo_root=repo_root,
            package_names=tuple(str(name) for name in locked_packages),
            required_packages=(),
            required_revisions=(),
            revisions={},
            cli_args={},
            formal=False,
        )
        current_environment = current["environment"]
        stable_locked_runtime = _stable_runtime_projection(environment.get("runtime"))
        stable_current_runtime = _stable_runtime_projection(
            current_environment.get("runtime")
        )
        comparisons = {
            "packages": (locked_packages, current_environment.get("packages")),
            "python": (environment.get("python"), current_environment.get("python")),
            "platform": (environment.get("platform"), current_environment.get("platform")),
            "runtime": (stable_locked_runtime, stable_current_runtime),
        }
        mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
        if mismatches:
            raise MethodContractError(
                "Current runtime differs from the formal environment artifact: "
                + ", ".join(mismatches)
            )
    return artifact


def _stable_runtime_projection(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError("Environment artifact has no runtime inventory")
    cuda = value.get("cuda")
    cudnn = value.get("cudnn")
    if not isinstance(cuda, Mapping) or not isinstance(cudnn, Mapping):
        raise MethodContractError("Environment runtime has no CUDA/cuDNN inventory")
    return {
        "torch_version": value.get("torch_version"),
        "cuda": {
            "available": cuda.get("available"),
            "compiled_version": cuda.get("compiled_version"),
            "device_count": cuda.get("device_count"),
            "devices": cuda.get("devices"),
            "query_status": cuda.get("query_status"),
        },
        "cudnn": {
            "available": cudnn.get("available"),
            "version": cudnn.get("version"),
        },
    }


def _verify_protocol_dependencies(
    *,
    split_lock_path: Path,
    noise_split_lock_path: Path,
    noisy_dev_lock_path: Path,
    repo_root: str | Path,
    verify_audio: bool,
) -> dict[str, Any]:
    split = load_split_lock(split_lock_path)
    try:
        noise = verify_noise_split_lock(noise_split_lock_path, verify_audio=verify_audio)
    except NoiseProtocolError as exc:
        raise MethodContractError(str(exc)) from exc
    noisy = verify_noisy_dev_lock(
        noisy_dev_lock_path,
        repo_root=repo_root,
        expected_noise_lock_sha256=str(noise["lock_sha256"]),
        expected_source_dev_sha256=str(split["splits"]["dev"]["manifest_sha256"]),
        verify_audio=verify_audio,
    )
    return {"split": split, "noise": noise, "noisy_dev": noisy}


def build_method_contract(
    config: Mapping[str, Any],
    *,
    split_lock_path: str | Path,
    noise_split_lock_path: str | Path,
    noisy_dev_lock_path: str | Path,
    environment_path: str | Path,
    source_components: Sequence[str | Path] = DEFAULT_SOURCE_COMPONENTS,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    repo_root: str | Path,
    formal: bool,
    verify_audio: bool = True,
) -> dict[str, Any]:
    """Build a fully verified, path-free contract for the five-lambda study."""

    root = Path(repo_root).resolve()
    split_path = _repo_path(split_lock_path, root)
    noise_path = _repo_path(noise_split_lock_path, root)
    noisy_path = _repo_path(noisy_dev_lock_path, root)
    env_path = _repo_path(environment_path, root)
    dependency = _verify_protocol_dependencies(
        split_lock_path=split_path,
        noise_split_lock_path=noise_path,
        noisy_dev_lock_path=noisy_path,
        repo_root=root,
        verify_audio=verify_audio,
    )
    environment = _validate_environment(
        env_path,
        formal=formal,
        repo_root=root,
        verify_current=formal,
    )
    grid = _normalize_lambda_grid(lambda_grid)

    component_paths = sorted({_display_path(path, root) for path in source_components})
    if not component_paths:
        raise MethodContractError("At least one source component must be locked")
    components = [_artifact_binding(path, root) for path in component_paths]

    split_lock = dependency["split"]
    noise_lock = dependency["noise"]["lock"]
    noisy_dev = dependency["noisy_dev"]
    data = config.get("data", {})
    noise_config = config.get("noise", {})
    train_manifest = _artifact_binding(str(data.get("train_manifest", "")), root)
    dev_manifest = _artifact_binding(str(data.get("valid_manifest", "")), root)
    if train_manifest["sha256"] != str(
        split_lock["splits"]["train"]["manifest_sha256"]
    ).casefold():
        raise MethodContractError("Training config does not bind the locked VIVOS train manifest")
    if dev_manifest["sha256"] != str(
        split_lock["splits"]["dev"]["manifest_sha256"]
    ).casefold():
        raise MethodContractError("Training config does not bind the locked VIVOS dev manifest")
    if noise_config.get("enable_train_noise") is not True:
        raise MethodContractError("Formal paper-v2 training requires locked train-noise augmentation")
    train_noise_manifest = _artifact_binding(
        str(noise_config.get("noise_manifest", "")), root
    )
    if train_noise_manifest["sha256"] != str(
        noise_lock["splits"]["train"]["manifest_sha256"]
    ).casefold():
        raise MethodContractError("Training config must use the locked MUSAN train partition")

    selection = _selection_contract(config)
    if selection["checkpoint_metric"] != "dev_asr_loss" or selection["checkpoint_mode"] != "min":
        raise MethodContractError("Best checkpoints must be selected by minimum dev_asr_loss")
    if (
        selection["checkpoint_selection_split"] != "dev"
        or selection["lambda_selection_split"] != "dev"
        or selection["require_full_manifest"] is not True
    ):
        raise MethodContractError("Checkpoint and lambda selection must be full-manifest dev-only")
    if selection["selection_manifest_sha256"] != noisy_dev["manifest_sha256"]:
        raise MethodContractError(
            "Lambda selection must bind the locked noisy-dev output manifest"
        )

    model = config.get("model", {})
    lora = model.get("lora", {})
    training = config.get("training", {})
    if not is_immutable_revision(model.get("revision")):
        raise MethodContractError("Backbone revision must be an immutable commit hash")
    if (
        int(lora.get("r", 0)) < 1
        or int(lora.get("lora_alpha", 0)) < 1
        or not 0.0 <= float(lora.get("lora_dropout", -1)) < 1.0
        or not list(lora.get("target_modules", []))
    ):
        raise MethodContractError("LoRA rank/alpha/dropout/target_modules are invalid")
    if training.get("tone_label_policy") not in {"last_subtoken", "all_subtokens"}:
        raise MethodContractError("Tone label policy is not supported")
    noise_probability = float(noise_config.get("prob", -1))
    noise_snrs = [float(value) for value in noise_config.get("snr_choices", [])]
    if (
        not math.isfinite(noise_probability)
        or not 0.0 <= noise_probability <= 1.0
        or not noise_snrs
        or any(not math.isfinite(value) for value in noise_snrs)
    ):
        raise MethodContractError("Training augmentation probability/SNR choices are invalid")
    contract: dict[str, Any] = {
        "schema_version": METHOD_CONTRACT_VERSION,
        "status": "LOCKED",
        "mode": "formal" if formal else "diagnostic",
        "artifacts": {
            "split_lock": _artifact_binding(split_path, root),
            "noise_split_lock": _artifact_binding(noise_path, root),
            "noisy_dev_lock": _artifact_binding(noisy_path, root),
            "environment": {
                **_artifact_binding(env_path, root),
                "identity_sha256": environment["identity_sha256"],
                "capture_mode": environment["environment"]["capture_mode"],
            },
        },
        "source": {
            "components": components,
            "tree_sha256": _source_tree_sha256(components),
        },
        "model": {
            "backbone": model.get("name_or_path"),
            "revision": model.get("revision"),
            "lora": {
                "r": int(lora.get("r", 0)),
                "alpha": int(lora.get("lora_alpha", 0)),
                "dropout": float(lora.get("lora_dropout", -1)),
                "target_modules": list(lora.get("target_modules", [])),
                "bias": "none",
            },
        },
        "tone_supervision": {
            "alignment": "exact_wordwise_bpe_ids_v1",
            "policy": training.get("tone_label_policy"),
            "classes": [
                {"id": int(index), "name": ID_TO_TONE[index]}
                for index in sorted(ID_TO_TONE)
            ],
            "ignore_index": IGNORE_INDEX,
            "head": "decoder_last_hidden_state_layernorm_linear_6",
            "loss": "cross_entropy_valid_positions_only",
            "loss_equation": "L_total = L_ASR + lambda_tone * L_tone",
            "shape_policy": "exact_batch_and_sequence_match_fail_closed",
        },
        "training": {
            "lambda_grid": [float(value) for value in grid],
            "contract_sha256_by_lambda": {
                _lambda_key(value): training_contract_sha256(
                    _config_for_lambda(config, value)
                )
                for value in grid
            },
            "seed": int(config.get("seed", -1)),
            "train_manifest": train_manifest,
            "dev_manifest": dev_manifest,
            "augmentation": {
                "implementation": "deterministic_item_epoch_noise_mix_v1",
                "noise_manifest": train_noise_manifest,
                "probability": noise_probability,
                "snr_choices_db": noise_snrs,
                "noise_partition": "train",
            },
        },
        "selection": selection,
        "decoding": _decode_contract(config),
        "metrics": {"version": METRIC_VERSION},
    }
    _reject_absolute_paths(contract)
    contract["identity_sha256"] = canonical_sha256(contract)
    return contract


def validate_method_contract(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != METHOD_CONTRACT_VERSION:
        raise MethodContractError("Unsupported method contract schema")
    if artifact.get("status") != "LOCKED":
        raise MethodContractError("Method contract status must be LOCKED")
    if artifact.get("mode") not in {"diagnostic", "formal"}:
        raise MethodContractError("Method contract mode must be diagnostic or formal")
    _reject_absolute_paths(artifact)
    identity = str(artifact.get("identity_sha256", "")).casefold()
    if not _is_sha256(identity):
        raise MethodContractError("Method contract has no valid identity SHA-256")
    payload = dict(artifact)
    payload.pop("identity_sha256", None)
    if canonical_sha256(payload) != identity:
        raise MethodContractError("Method contract identity SHA-256 is invalid")


def verify_method_artifact_bindings(
    lock_path: str | Path,
    *,
    repo_root: str | Path,
    formal: bool = True,
) -> dict[str, Any]:
    """Verify a method lock's immutable files without inspecting runtime state.

    This is the portable verifier for post-hoc analysis and zero-shot
    authorization.  It deliberately does not compare a runtime config,
    environment, audio corpus, or checkpoint.  It does verify the canonical
    method identity, every bound artifact byte, every source component byte,
    and the complete source-tree identity.
    """

    root = Path(repo_root).resolve()
    lock_file = _repo_path(lock_path, root)
    lock_sha_before = sha256_file(lock_file)
    artifact = _load_json(lock_file, label="Method lock")
    validate_method_contract(artifact)
    if formal and artifact.get("mode") != "formal":
        raise MethodContractError("Formal execution requires a formal method lock")

    artifacts = artifact.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise MethodContractError("Method lock has no artifact bindings")
    verified_artifacts: dict[str, dict[str, str]] = {}
    artifact_paths: set[str] = set()
    for name, raw_binding in sorted(artifacts.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_binding, Mapping):
            raise MethodContractError(f"Malformed method artifact binding: {name}")
        path = _verify_binding(
            raw_binding,
            repo_root=root,
            label=f"method artifact {name}",
        )
        display = _display_path(path, root)
        if display in artifact_paths:
            raise MethodContractError(f"Duplicate method artifact binding: {display}")
        artifact_paths.add(display)
        verified_artifacts[str(name)] = {
            "path": display,
            "sha256": str(raw_binding["sha256"]).casefold(),
        }

    source = artifact.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("components"), list):
        raise MethodContractError("Method lock has no source component inventory")
    if not source["components"]:
        raise MethodContractError("Method lock source component inventory is empty")
    verified_components: list[dict[str, str]] = []
    observed_paths: set[str] = set()
    for raw_binding in source["components"]:
        if not isinstance(raw_binding, Mapping):
            raise MethodContractError("Malformed source component binding")
        path = _verify_binding(raw_binding, repo_root=root, label="source component")
        display = _display_path(path, root)
        if display in observed_paths:
            raise MethodContractError(f"Duplicate source component: {display}")
        observed_paths.add(display)
        verified_components.append(
            {"path": display, "sha256": str(raw_binding["sha256"]).casefold()}
        )
    source_tree = str(source.get("tree_sha256", "")).casefold()
    if not _is_sha256(source_tree) or (
        _source_tree_sha256(verified_components) != source_tree
    ):
        raise MethodContractError("Source component tree hash differs from the method lock")
    lock_sha_after = sha256_file(lock_file)
    if lock_sha_after != lock_sha_before:
        raise MethodContractError("Method lock changed while it was being verified")
    return {
        "artifact": artifact,
        "method_lock_path": lock_file,
        "method_lock_sha256": lock_sha_after,
        "method_identity_sha256": str(artifact["identity_sha256"]).casefold(),
        "source_tree_sha256": source_tree,
        "artifacts": verified_artifacts,
        "mode": str(artifact["mode"]),
    }


def _compare_config_to_contract(
    config: Mapping[str, Any], artifact: Mapping[str, Any]
) -> None:
    model = config.get("model", {})
    lora = model.get("lora", {})
    locked_model = artifact.get("model", {})
    expected_model = {
        "backbone": model.get("name_or_path"),
        "revision": model.get("revision"),
        "lora": {
            "r": int(lora.get("r", 0)),
            "alpha": int(lora.get("lora_alpha", 0)),
            "dropout": float(lora.get("lora_dropout", -1)),
            "target_modules": list(lora.get("target_modules", [])),
            "bias": "none",
        },
    }
    if locked_model != expected_model:
        raise MethodContractError("Runtime backbone/LoRA config differs from the method lock")

    lambda_value = float(config.get("training", {}).get("lambda_tone", -1))
    lambda_key = _lambda_key(lambda_value)
    contracts = artifact.get("training", {}).get("contract_sha256_by_lambda", {})
    if not isinstance(contracts, Mapping) or lambda_key not in contracts:
        raise MethodContractError(f"Runtime lambda={lambda_key} is outside the locked grid")
    runtime_hash = training_contract_sha256(config)
    if runtime_hash != str(contracts[lambda_key]).casefold():
        raise MethodContractError(
            f"Runtime training contract differs for lambda={lambda_key}"
        )
    if _decode_contract(config) != artifact.get("decoding"):
        raise MethodContractError("Runtime decode contract differs from the method lock")
    if _selection_contract(config) != artifact.get("selection"):
        raise MethodContractError("Runtime dev-only selection contract differs from the method lock")
    if artifact.get("metrics", {}).get("version") != METRIC_VERSION:
        raise MethodContractError(
            f"Method lock must bind metric_version={METRIC_VERSION}"
        )


def verify_method_lock(
    lock_path: str | Path,
    *,
    config: Mapping[str, Any],
    repo_root: str | Path,
    formal: bool,
    verify_audio: bool = True,
) -> dict[str, str]:
    """Fail closed on a tampered method or any stale dependency/source/config."""

    root = Path(repo_root).resolve()
    lock_file = _repo_path(lock_path, root)
    artifact = _load_json(lock_file, label="Method lock")
    validate_method_contract(artifact)
    if formal and artifact.get("mode") != "formal":
        raise MethodContractError("Formal execution requires a formal method lock")

    artifacts = artifact.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MethodContractError("Method lock has no artifact bindings")
    split_path = _verify_binding(artifacts.get("split_lock", {}), repo_root=root, label="split lock")
    noise_path = _verify_binding(
        artifacts.get("noise_split_lock", {}), repo_root=root, label="noise split lock"
    )
    noisy_path = _verify_binding(
        artifacts.get("noisy_dev_lock", {}), repo_root=root, label="noisy-dev lock"
    )
    environment_binding = artifacts.get("environment", {})
    if not isinstance(environment_binding, Mapping):
        raise MethodContractError("Method lock has no environment binding")
    environment_path = _verify_binding(
        environment_binding, repo_root=root, label="environment artifact"
    )
    environment = _validate_environment(
        environment_path,
        formal=formal,
        repo_root=root,
        verify_current=formal,
    )
    if (
        environment.get("identity_sha256")
        != environment_binding.get("identity_sha256")
        or environment.get("environment", {}).get("capture_mode")
        != environment_binding.get("capture_mode")
    ):
        raise MethodContractError("Environment identity/mode differs from the method lock")

    dependency = _verify_protocol_dependencies(
        split_lock_path=split_path,
        noise_split_lock_path=noise_path,
        noisy_dev_lock_path=noisy_path,
        repo_root=root,
        verify_audio=verify_audio,
    )
    source = artifact.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("components"), list):
        raise MethodContractError("Method lock has no source component inventory")
    verified_components: list[dict[str, str]] = []
    observed_paths: set[str] = set()
    for binding in source["components"]:
        if not isinstance(binding, Mapping):
            raise MethodContractError("Malformed source component binding")
        path = _verify_binding(binding, repo_root=root, label="source component")
        display = _display_path(path, root)
        if display in observed_paths:
            raise MethodContractError(f"Duplicate source component: {display}")
        observed_paths.add(display)
        verified_components.append(
            {"path": display, "sha256": str(binding["sha256"]).casefold()}
        )
    if _source_tree_sha256(verified_components) != source.get("tree_sha256"):
        raise MethodContractError("Source component tree hash differs from the method lock")

    _compare_config_to_contract(config, artifact)
    return {
        "method_lock_sha256": sha256_file(lock_file),
        "method_identity_sha256": str(artifact["identity_sha256"]),
        "environment_artifact_sha256": sha256_file(environment_path),
        "environment_identity_sha256": str(environment["identity_sha256"]),
        "source_tree_sha256": str(source["tree_sha256"]),
        "protocol_split_lock_sha256": sha256_file(split_path),
        "noise_split_lock_sha256": str(dependency["noise"]["lock_sha256"]),
        "noisy_dev_lock_sha256": str(dependency["noisy_dev"]["lock_sha256"]),
        "noisy_dev_manifest_sha256": str(
            dependency["noisy_dev"]["manifest_sha256"]
        ),
        "mode": str(artifact["mode"]),
    }


def verify_checkpoint_method_binding(
    checkpoint_root: str | Path,
    method_integrity: Mapping[str, str],
) -> None:
    """Require a trained checkpoint to carry the exact verified method identity."""

    resolved = Path(checkpoint_root) / "resolved_config.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint is missing resolved_config.yaml: {resolved}")
    import yaml

    saved = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    runtime = saved.get("runtime_protocol") if isinstance(saved, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise MethodContractError("Checkpoint has no runtime method binding")
    expected_fields = (
        "method_lock_sha256",
        "method_identity_sha256",
        "environment_artifact_sha256",
        "environment_identity_sha256",
        "source_tree_sha256",
    )
    mismatches = [
        field
        for field in expected_fields
        if str(runtime.get(field, "")).casefold()
        != str(method_integrity.get(field, "")).casefold()
    ]
    if mismatches:
        raise MethodContractError(
            "Checkpoint method binding differs from the verified lock: "
            + ", ".join(mismatches)
        )


def write_method_lock(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write canonical JSON and refuse overwrite by default."""

    validate_method_contract(artifact)
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
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite existing method lock: {destination}"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_METHOD_LOCK",
    "DEFAULT_SOURCE_COMPONENTS",
    "METHOD_CONTRACT_VERSION",
    "MethodContractError",
    "build_method_contract",
    "validate_method_contract",
    "verify_method_artifact_bindings",
    "verify_checkpoint_method_binding",
    "verify_method_lock",
    "verify_noisy_dev_lock",
    "write_method_lock",
]
