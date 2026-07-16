from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from .protocol import evaluation_contract_sha256, is_immutable_revision, is_sha256


ConfigDict = dict[str, Any]
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    merged: ConfigDict = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> ConfigDict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def load_experiment_config(path: str | Path) -> ConfigDict:
    """Load a YAML experiment config with an optional relative base config."""
    config_path = Path(path).resolve()
    raw = _read_yaml(config_path)
    base_reference = raw.pop("base_config", None)
    if base_reference:
        base_path = (config_path.parent / str(base_reference)).resolve()
        config = _deep_merge(_read_yaml(base_path), raw)
    else:
        config = raw
    config["_config_path"] = str(config_path)
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    required_sections = (
        "experiment",
        "protocol",
        "model",
        "training",
        "data",
        "evaluation",
        "selection",
    )
    missing_sections = [name for name in required_sections if not isinstance(config.get(name), Mapping)]
    if missing_sections:
        raise ValueError(f"Missing config sections: {missing_sections}")

    training = config["training"]
    model = config["model"]
    data = config["data"]
    evaluation = config["evaluation"]
    protocol = config["protocol"]
    lambda_tone = float(training.get("lambda_tone", -1.0))
    if not math.isfinite(lambda_tone) or lambda_tone < 0.0:
        raise ValueError("training.lambda_tone must be finite and non-negative")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not model.get("name_or_path"):
        raise ValueError("model.name_or_path is required")
    if not is_immutable_revision(model.get("revision")):
        raise ValueError(
            "model.revision must be an immutable 40- or 64-character commit hash"
        )
    if not data.get("train_manifest") or not data.get("valid_manifest"):
        raise ValueError("data.train_manifest and data.valid_manifest are required")
    if not evaluation.get("manifest"):
        raise ValueError("evaluation.manifest is required")
    if not protocol.get("split_lock") or not protocol.get("decision_lock"):
        raise ValueError("protocol.split_lock and protocol.decision_lock are required")
    if protocol.get("verify_audio_sha256") is not True:
        raise ValueError("protocol.verify_audio_sha256 must be true")
    if not isinstance(protocol.get("formal_training_unlocked"), bool):
        raise ValueError("protocol.formal_training_unlocked must be boolean")
    if not isinstance(protocol.get("final_test_unlocked"), bool):
        raise ValueError("protocol.final_test_unlocked must be boolean")
    if int(training.get("gradient_accumulation_steps", 0)) < 1:
        raise ValueError("training.gradient_accumulation_steps must be at least 1")
    if int(training.get("per_device_train_batch_size", 0)) < 1:
        raise ValueError("training.per_device_train_batch_size must be at least 1")
    if int(training.get("per_device_eval_batch_size", 0)) < 1:
        raise ValueError("training.per_device_eval_batch_size must be at least 1")
    if training.get("checkpoint_metric") != "dev_asr_loss":
        raise ValueError("training.checkpoint_metric must be 'dev_asr_loss'")
    if training.get("checkpoint_mode") != "min":
        raise ValueError("training.checkpoint_mode must be 'min'")
    run_scope = str(training.get("run_scope", "")).strip().casefold()
    if run_scope not in {"formal", "smoke"}:
        raise ValueError("training.run_scope must be 'formal' or 'smoke'")
    if run_scope == "formal" and not protocol.get("method_lock"):
        raise ValueError("formal training/evaluation requires protocol.method_lock")
    has_smoke_limit = any(
        training.get(name) is not None
        for name in ("max_train_samples", "max_train_steps", "max_eval_samples")
    )
    if run_scope == "formal" and has_smoke_limit:
        raise ValueError(
            "formal training cannot set max_train_samples, max_train_steps, or "
            "max_eval_samples"
        )
    evaluation_split = str(evaluation.get("data_split", "")).strip().casefold()
    if evaluation_split not in {"dev", "test", "external"}:
        raise ValueError("evaluation.data_split must be one of: dev, test, external")
    benchmark_protocol = str(
        evaluation.get("benchmark_protocol", "locked_vivos")
    ).strip().casefold()
    if benchmark_protocol not in {"locked_vivos", "noisy_dev"}:
        raise ValueError(
            "evaluation.benchmark_protocol must be one of: locked_vivos, noisy_dev"
        )
    noisy_dev_fields = (
        "noisy_dev_lock",
        "expected_noisy_dev_lock_sha256",
        "expected_noise_split_lock_sha256",
        "expected_source_dev_sha256",
    )
    if benchmark_protocol == "noisy_dev":
        if evaluation_split != "dev":
            raise ValueError(
                "evaluation.benchmark_protocol=noisy_dev is valid only for data_split=dev"
            )
        if not str(evaluation.get("noisy_dev_lock", "")).strip():
            raise ValueError(
                "evaluation.noisy_dev_lock is required for the noisy-dev benchmark"
            )
        for field in noisy_dev_fields[1:]:
            if not is_sha256(evaluation.get(field)):
                raise ValueError(
                    f"evaluation.{field} must be a SHA-256 hex digest"
                )
    elif any(evaluation.get(field) not in (None, "") for field in noisy_dev_fields):
        raise ValueError(
            "Noisy-dev lock fields require evaluation.benchmark_protocol=noisy_dev"
        )
    required_selection_split = str(
        config["selection"].get("required_evaluation_split", "")
    ).strip().casefold()
    if required_selection_split != "dev":
        raise ValueError("selection.required_evaluation_split must be 'dev'")
    expected_manifest_hash = str(
        evaluation.get("expected_manifest_sha256", "")
    ).strip().casefold()
    selection_manifest_hash = str(
        config["selection"].get("expected_manifest_sha256", "")
    ).strip().casefold()
    if not is_sha256(expected_manifest_hash):
        raise ValueError("evaluation.expected_manifest_sha256 must be a SHA-256 hex digest")
    if (
        evaluation_split == "dev"
        and selection_manifest_hash != expected_manifest_hash
    ):
        raise ValueError(
            "selection.expected_manifest_sha256 must match evaluation.expected_manifest_sha256"
        )
    if not is_sha256(selection_manifest_hash):
        raise ValueError(
            "selection.expected_manifest_sha256 must be a SHA-256 hex digest"
        )
    configured_evaluation_contract = str(
        config["selection"].get("expected_evaluation_contract_sha256", "")
    ).strip().casefold()
    actual_evaluation_contract = evaluation_contract_sha256(config)
    if not is_sha256(configured_evaluation_contract):
        raise ValueError(
            "selection.expected_evaluation_contract_sha256 must be a SHA-256 digest"
        )
    if (
        evaluation_split == "dev"
        and configured_evaluation_contract != actual_evaluation_contract
    ):
        raise ValueError(
            "selection.expected_evaluation_contract_sha256 must match the "
            "canonical dev evaluation contract"
        )
    locked_vivos_split = str(
        evaluation.get("locked_vivos_split", "")
    ).strip().casefold()
    allowed_locked_splits = {"dev"} if evaluation_split == "dev" else {"test_locked"}
    if evaluation_split in {"dev", "test"} and locked_vivos_split not in allowed_locked_splits:
        raise ValueError(
            "evaluation.locked_vivos_split must bind dev to dev and test to test_locked"
        )
    if evaluation.get("expected_total_rows") is not None:
        if int(evaluation["expected_total_rows"]) < 1:
            raise ValueError("evaluation.expected_total_rows must be at least 1")
    if int(evaluation.get("batch_size", 0)) < 1:
        raise ValueError("evaluation.batch_size must be at least 1")
    if str(evaluation.get("inference_precision", "")).casefold() not in {
        "fp16",
        "fp32",
    }:
        raise ValueError("evaluation.inference_precision must be fp16 or fp32")
    if config["selection"].get("require_full_manifest") is not True:
        raise ValueError("selection.require_full_manifest must be true")
    for field in (
        "min_ter_coverage_ratio_vs_baseline",
        "min_der_coverage_ratio_vs_baseline",
        "min_fcer_coverage_ratio_vs_baseline",
    ):
        value = float(config["selection"].get(field, -1))
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"selection.{field} must be in [0, 1]")
    expected_train_type = "ordinary_lora" if lambda_tone == 0.0 else "tone_aware_lora"
    if config["experiment"].get("train_type") != expected_train_type:
        raise ValueError(
            f"experiment.train_type must be {expected_train_type!r} when lambda_tone={lambda_tone:g}"
        )
    method_id = str(config["experiment"].get("method_id", "")).strip()
    if not _IDENTIFIER_RE.fullmatch(method_id):
        raise ValueError("experiment.method_id must be a lowercase snake_case identifier")


def apply_cli_overrides(
    config: ConfigDict,
    *,
    lambda_value: float | None = None,
    seed: int | None = None,
    train_manifest: str | None = None,
    output_dir: str | None = None,
    max_train_samples: int | None = None,
    max_train_steps: int | None = None,
) -> ConfigDict:
    updated = deepcopy(config)
    if lambda_value is not None:
        updated["training"]["lambda_tone"] = float(lambda_value)
        updated["experiment"]["train_type"] = "ordinary_lora" if float(lambda_value) == 0.0 else "tone_aware_lora"
        if updated["experiment"].get("method_id") in {
            "ordinary_lora",
            "corrected_decoder_tone_lora",
        }:
            updated["experiment"]["method_id"] = (
                "ordinary_lora"
                if float(lambda_value) == 0.0
                else "corrected_decoder_tone_lora"
            )
    if seed is not None:
        updated["seed"] = int(seed)
    if train_manifest is not None:
        updated["data"]["train_manifest"] = train_manifest
    if output_dir is not None:
        updated["training"]["output_dir"] = output_dir
        updated.setdefault("logging", {})["file"] = str(Path(output_dir) / "training.log")
        updated.setdefault("logging", {})["metrics_csv"] = str(Path(output_dir) / "training_metrics.csv")
        updated.setdefault("logging", {})["dev_metrics_csv"] = str(
            Path(output_dir) / "dev_metrics.csv"
        )
    if max_train_samples is not None:
        if max_train_samples < 1:
            raise ValueError("max_train_samples must be at least 1")
        updated["training"]["max_train_samples"] = int(max_train_samples)
        updated["training"]["run_scope"] = "smoke"
    if max_train_steps is not None:
        if max_train_steps < 1:
            raise ValueError("max_train_steps must be at least 1")
        updated["training"]["max_train_steps"] = int(max_train_steps)
        updated["training"]["run_scope"] = "smoke"
    validate_experiment_config(updated)
    return updated
