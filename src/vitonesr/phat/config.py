from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


ConfigDict = dict[str, Any]


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
    required_sections = ("experiment", "model", "training", "data", "evaluation", "selection")
    missing_sections = [name for name in required_sections if not isinstance(config.get(name), Mapping)]
    if missing_sections:
        raise ValueError(f"Missing config sections: {missing_sections}")

    training = config["training"]
    model = config["model"]
    data = config["data"]
    evaluation = config["evaluation"]
    lambda_tone = float(training.get("lambda_tone", -1.0))
    if lambda_tone < 0.0:
        raise ValueError("training.lambda_tone must be non-negative")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not model.get("name_or_path"):
        raise ValueError("model.name_or_path is required")
    if not data.get("train_manifest") or not data.get("valid_manifest"):
        raise ValueError("data.train_manifest and data.valid_manifest are required")
    if not evaluation.get("manifest"):
        raise ValueError("evaluation.manifest is required")
    if int(training.get("gradient_accumulation_steps", 0)) < 1:
        raise ValueError("training.gradient_accumulation_steps must be at least 1")
    if int(training.get("per_device_train_batch_size", 0)) < 1:
        raise ValueError("training.per_device_train_batch_size must be at least 1")
    expected_train_type = "ordinary_lora" if lambda_tone == 0.0 else "tone_aware_lora"
    if config["experiment"].get("train_type") != expected_train_type:
        raise ValueError(
            f"experiment.train_type must be {expected_train_type!r} when lambda_tone={lambda_tone:g}"
        )


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
    if seed is not None:
        updated["seed"] = int(seed)
    if train_manifest is not None:
        updated["data"]["train_manifest"] = train_manifest
    if output_dir is not None:
        updated["training"]["output_dir"] = output_dir
        updated.setdefault("logging", {})["file"] = str(Path(output_dir) / "training.log")
        updated.setdefault("logging", {})["metrics_csv"] = str(Path(output_dir) / "training_metrics.csv")
    if max_train_samples is not None:
        if max_train_samples < 1:
            raise ValueError("max_train_samples must be at least 1")
        updated["training"]["max_train_samples"] = int(max_train_samples)
    if max_train_steps is not None:
        if max_train_steps < 1:
            raise ValueError("max_train_steps must be at least 1")
        updated["training"]["max_train_steps"] = int(max_train_steps)
    validate_experiment_config(updated)
    return updated
