from __future__ import annotations

import csv
import json
import logging
import math
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, get_linear_schedule_with_warmup

from src.vitonesr.data import DataCollatorSpeechSeq2SeqWithTone

from .config import ConfigDict
from .modeling import (
    PhoWhisperLoRA,
    assert_only_lora_and_tone_trainable,
    build_trainable_model,
    parameter_stats,
)
from .method_contract import verify_method_lock
from .reproducibility import seed_worker, set_global_seed
from .protocol import (
    PROTOCOL_VERSION,
    sha256_file,
    training_contract_sha256,
    verify_locked_vivos_manifest,
)
from .training_data import (
    DeterministicASRTrainingDataset,
    validate_jsonl_audio_manifest_fields,
    validate_manifest_declared_split,
)


LOGGER_NAME = "vitonesr.phat.training"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEV_METRIC_COLUMNS = [
    "epoch",
    "global_step",
    "num_samples",
    "asr_tokens",
    "tone_targets",
    "dev_asr_loss",
    "dev_tone_loss",
    "dev_total_loss",
    "is_best",
]


@dataclass
class TrainerProgress:
    epoch: int = 0
    next_batch_index: int = 0
    global_step: int = 0
    best_eval_asr_loss: float | None = None
    best_global_step: int | None = None
    last_eval_step: int = -1


@dataclass(frozen=True)
class DevMetrics:
    num_samples: int
    asr_tokens: int
    tone_targets: int
    asr_loss: float
    tone_loss: float
    total_loss: float


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return selected


def _assert_training_gate(config: ConfigDict) -> None:
    training_scope = str(
        config.get("training", {}).get("run_scope", "")
    ).strip().casefold()
    if (
        training_scope == "formal"
        and config.get("protocol", {}).get("formal_training_unlocked") is not True
    ):
        raise RuntimeError(
            "Formal paper-v2 training is locked until Gates 2-3 install and "
            "review the noise/benchmark/method contracts. Smoke diagnostics "
            "must use training.run_scope=smoke and isolated outputs."
        )


def _prepare_output_directory(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if resume:
            return
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --resume or --overwrite explicitly."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _configure_logger(log_path: Path, *, append: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _build_dataset(
    config: ConfigDict,
    processor: WhisperProcessor,
    *,
    split: str = "train",
) -> DeterministicASRTrainingDataset:
    if split not in {"train", "dev"}:
        raise ValueError("dataset split must be train or dev")
    training = config["training"]
    data = config["data"]
    noise = config.get("noise", {})
    is_train = split == "train"
    enable_noise = is_train and bool(noise.get("enable_train_noise", False))
    return DeterministicASRTrainingDataset(
        str(data["train_manifest"] if is_train else data["valid_manifest"]),
        processor,
        seed=int(config["seed"]),
        sample_rate=int(data.get("sample_rate", 16000)),
        max_audio_seconds=float(training.get("max_audio_seconds", 15.0)),
        max_label_length=training.get("max_label_length"),
        noise_manifest=str(noise["noise_manifest"]) if enable_noise else None,
        noise_prob=float(noise.get("prob", 0.0)) if enable_noise else 0.0,
        snr_choices=noise.get("snr_choices", [20, 10, 5, 0]),
        tone_label_policy=str(training.get("tone_label_policy", "last_subtoken")),
        max_samples=training.get("max_train_samples" if is_train else "max_eval_samples"),
        expected_split=split,
    )


def _build_train_loader(
    dataset: DeterministicASRTrainingDataset,
    processor: WhisperProcessor,
    decoder_start_token_id: int,
    config: ConfigDict,
    epoch: int,
) -> DataLoader:
    training = config["training"]
    dataset.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + epoch)
    return DataLoader(
        dataset,
        batch_size=int(training.get("per_device_train_batch_size", 1)),
        shuffle=True,
        collate_fn=DataCollatorSpeechSeq2SeqWithTone(processor, decoder_start_token_id),
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _build_eval_loader(
    dataset: DeterministicASRTrainingDataset,
    processor: WhisperProcessor,
    decoder_start_token_id: int,
    config: ConfigDict,
) -> DataLoader:
    training = config["training"]
    dataset.set_epoch(0)
    return DataLoader(
        dataset,
        batch_size=int(
            training.get(
                "per_device_eval_batch_size",
                training.get("per_device_train_batch_size", 1),
            )
        ),
        shuffle=False,
        collate_fn=DataCollatorSpeechSeq2SeqWithTone(processor, decoder_start_token_id),
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )


def _checkpoint_adapter_path(checkpoint: Path) -> Path:
    adapter_path = checkpoint / "adapter"
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing PEFT adapter in checkpoint: {adapter_path}")
    return adapter_path


def _assert_resume_compatible(config: ConfigDict, checkpoint: Path) -> None:
    saved_path = checkpoint / "resolved_config.yaml"
    if not saved_path.exists():
        raise FileNotFoundError(f"Missing resolved config in checkpoint: {saved_path}")
    with saved_path.open("r", encoding="utf-8") as saved_file:
        saved = yaml.safe_load(saved_file) or {}
    saved_contract = training_contract_sha256(saved)
    current_contract = training_contract_sha256(config)
    if saved_contract != current_contract:
        raise ValueError(
            "Resume training contract mismatch: "
            f"saved={saved_contract}, current={current_contract}"
        )
    comparisons = [
        ("seed", saved.get("seed"), config["seed"]),
        ("model.name_or_path", saved.get("model", {}).get("name_or_path"), config["model"]["name_or_path"]),
        ("training.lambda_tone", saved.get("training", {}).get("lambda_tone"), config["training"]["lambda_tone"]),
        (
            "training.checkpoint_metric",
            saved.get("training", {}).get("checkpoint_metric"),
            config["training"]["checkpoint_metric"],
        ),
        (
            "training.num_train_epochs",
            saved.get("training", {}).get("num_train_epochs"),
            config["training"].get("num_train_epochs"),
        ),
        (
            "training.per_device_train_batch_size",
            saved.get("training", {}).get("per_device_train_batch_size"),
            config["training"].get("per_device_train_batch_size"),
        ),
        (
            "training.gradient_accumulation_steps",
            saved.get("training", {}).get("gradient_accumulation_steps"),
            config["training"].get("gradient_accumulation_steps"),
        ),
        (
            "training.learning_rate",
            saved.get("training", {}).get("learning_rate"),
            config["training"].get("learning_rate"),
        ),
        (
            "training.max_train_samples",
            saved.get("training", {}).get("max_train_samples"),
            config["training"].get("max_train_samples"),
        ),
        (
            "training.max_train_steps",
            saved.get("training", {}).get("max_train_steps"),
            config["training"].get("max_train_steps"),
        ),
        (
            "training.tone_label_policy",
            saved.get("training", {}).get("tone_label_policy"),
            config["training"].get("tone_label_policy"),
        ),
        ("noise", saved.get("noise"), config.get("noise")),
        ("model.lora.r", saved.get("model", {}).get("lora", {}).get("r"), config["model"]["lora"]["r"]),
        (
            "data.train_manifest",
            saved.get("data", {}).get("train_manifest"),
            config["data"]["train_manifest"],
        ),
        (
            "data.valid_manifest",
            saved.get("data", {}).get("valid_manifest"),
            config["data"]["valid_manifest"],
        ),
        (
            "runtime_protocol.train_manifest_sha256",
            saved.get("runtime_protocol", {}).get("train_manifest_sha256"),
            config.get("_runtime_protocol", {}).get("train_manifest_sha256"),
        ),
        (
            "runtime_protocol.dev_manifest_sha256",
            saved.get("runtime_protocol", {}).get("dev_manifest_sha256"),
            config.get("_runtime_protocol", {}).get("dev_manifest_sha256"),
        ),
        (
            "runtime_protocol.split_lock_sha256",
            saved.get("runtime_protocol", {}).get("split_lock_sha256"),
            config.get("_runtime_protocol", {}).get("split_lock_sha256"),
        ),
        (
            "runtime_protocol",
            saved.get("runtime_protocol"),
            config.get("_runtime_protocol"),
        ),
    ]
    mismatches = [f"{name}: saved={old!r}, current={new!r}" for name, old, new in comparisons if old != new]
    if mismatches:
        raise ValueError("Resume config mismatch: " + "; ".join(mismatches))


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_metrics_header(path: Path, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "global_step", "learning_rate", "asr_loss", "tone_loss", "total_loss"],
            lineterminator="\n",
        )
        writer.writeheader()


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=list(row),
            lineterminator="\n",
        )
        writer.writerow(row)


def _write_dev_metrics_header(path: Path, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=DEV_METRIC_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()


def _append_dev_metrics(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as metrics_file:
            existing = [
                item
                for item in csv.DictReader(metrics_file)
                if int(item["global_step"]) == int(row["global_step"])
            ]
        if len(existing) > 1:
            raise ValueError(
                f"Duplicate dev metric rows already exist for step {row['global_step']}"
            )
        if existing:
            previous = existing[0]
            for column in DEV_METRIC_COLUMNS:
                expected = str(row[column])
                observed = str(previous[column])
                if column in {
                    "dev_asr_loss",
                    "dev_tone_loss",
                    "dev_total_loss",
                }:
                    matches = math.isclose(
                        float(observed),
                        float(expected),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                else:
                    matches = observed == expected
                if not matches:
                    raise ValueError(
                        "Resumed dev evaluation conflicts with the existing "
                        f"metric row at step {row['global_step']}: {column}"
                    )
            return
    with path.open("a", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=DEV_METRIC_COLUMNS,
            lineterminator="\n",
        )
        writer.writerow(row)


def _is_strictly_better(current: float, best: float | None) -> bool:
    if not math.isfinite(current):
        raise FloatingPointError(f"Non-finite dev ASR loss: {current}")
    return best is None or current < best


def _reached_max_train_steps(
    progress: TrainerProgress,
    max_train_steps: int | None,
) -> bool:
    return (
        max_train_steps is not None
        and progress.global_step >= int(max_train_steps)
    )


def _resume_needs_dev_evaluation(
    progress: TrainerProgress,
    *,
    is_resume: bool,
    reached_max_steps: bool,
) -> bool:
    return (
        is_resume
        and progress.global_step > progress.last_eval_step
        and (progress.next_batch_index == 0 or reached_max_steps)
    )


def _save_checkpoint(
    *,
    target: Path,
    model: PhoWhisperLoRA,
    processor: WhisperProcessor,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    progress: TrainerProgress,
    config: ConfigDict,
    overwrite: bool = False,
) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Checkpoint already exists: {target}")
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    backup = target.with_name(f".{target.name}.backup")
    if backup.exists():
        raise FileExistsError(f"Stale checkpoint backup requires inspection: {backup}")
    temporary.mkdir(parents=True)
    try:
        model.asr_model.save_pretrained(temporary / "adapter")
        processor.save_pretrained(temporary / "processor")
        if model.tone_head is not None:
            torch.save(model.tone_head.state_dict(), temporary / "tone_head.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(scaler.state_dict(), temporary / "scaler.pt")
        torch.save(_capture_rng_state(), temporary / "rng_state.pt")
        (temporary / "trainer_state.json").write_text(
            json.dumps(progress.__dict__, indent=2),
            encoding="utf-8",
        )
        serializable_config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        if config.get("_runtime_protocol"):
            serializable_config["runtime_protocol"] = dict(
                config["_runtime_protocol"]
            )
        with (temporary / "resolved_config.yaml").open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(serializable_config, config_file, sort_keys=False, allow_unicode=True)
        if target.exists():
            target.rename(backup)
        try:
            temporary.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_training_state(
    checkpoint: Path,
    *,
    model: PhoWhisperLoRA,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
) -> TrainerProgress:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if model.tone_head is not None:
        tone_path = checkpoint / "tone_head.pt"
        if not tone_path.exists():
            raise FileNotFoundError(f"Missing tone head for tone-aware checkpoint: {tone_path}")
        model.tone_head.load_state_dict(torch.load(tone_path, map_location="cpu", weights_only=True))
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True))
    scaler.load_state_dict(torch.load(checkpoint / "scaler.pt", map_location="cpu", weights_only=True))
    rng_state = torch.load(checkpoint / "rng_state.pt", map_location="cpu", weights_only=False)
    _restore_rng_state(rng_state)
    progress = TrainerProgress(
        epoch=int(state["epoch"]),
        next_batch_index=int(state["next_batch_index"]),
        global_step=int(state["global_step"]),
        best_eval_asr_loss=(
            None
            if state.get("best_eval_asr_loss") is None
            else float(state["best_eval_asr_loss"])
        ),
        best_global_step=(
            None if state.get("best_global_step") is None else int(state["best_global_step"])
        ),
        last_eval_step=int(state.get("last_eval_step", -1)),
    )
    if (
        progress.best_eval_asr_loss is not None
        and not math.isfinite(progress.best_eval_asr_loss)
    ):
        raise ValueError(f"Checkpoint has non-finite best dev loss: {checkpoint}")
    if progress.last_eval_step > progress.global_step:
        raise ValueError(
            f"Checkpoint last_eval_step exceeds global_step: {checkpoint}"
        )
    return progress


def _merge_saved_best(
    progress: TrainerProgress,
    best_checkpoint: Path,
    config: ConfigDict,
) -> None:
    state_path = best_checkpoint / "trainer_state.json"
    if not best_checkpoint.exists():
        return
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Existing best checkpoint is missing trainer_state.json: {best_checkpoint}"
        )
    if not (best_checkpoint / "adapter" / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Existing best checkpoint is missing its adapter: {best_checkpoint}"
        )
    if not (best_checkpoint / "processor").is_dir():
        raise FileNotFoundError(
            f"Existing best checkpoint is missing its processor: {best_checkpoint}"
        )
    if (
        float(config["training"]["lambda_tone"]) > 0
        and not (best_checkpoint / "tone_head.pt").is_file()
    ):
        raise FileNotFoundError(
            f"Existing best checkpoint is missing tone_head.pt: {best_checkpoint}"
        )
    resolved_path = best_checkpoint / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint is missing resolved_config.yaml: {best_checkpoint}"
        )
    with resolved_path.open("r", encoding="utf-8") as handle:
        saved_config = yaml.safe_load(handle) or {}
    saved_contract = training_contract_sha256(saved_config)
    current_contract = training_contract_sha256(config)
    if saved_contract != current_contract:
        raise ValueError(
            "Best checkpoint training contract does not match the resumed run"
        )
    saved_runtime = saved_config.get("runtime_protocol")
    current_runtime = config.get("_runtime_protocol")
    if not isinstance(saved_runtime, dict) or not isinstance(current_runtime, dict):
        raise ValueError("Best checkpoint has no compatible runtime protocol")
    for field in (
        "split_lock_sha256",
        "train_manifest_sha256",
        "dev_manifest_sha256",
    ):
        if saved_runtime.get(field) != current_runtime.get(field):
            raise ValueError(
                f"Best checkpoint runtime protocol mismatch: {field}"
            )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved_loss = state.get("best_eval_asr_loss")
    if saved_loss is None:
        raise ValueError(
            f"Best checkpoint has no best_eval_asr_loss: {best_checkpoint}"
        )
    saved_loss = float(saved_loss)
    if not math.isfinite(saved_loss):
        raise ValueError(f"Best checkpoint has non-finite dev loss: {best_checkpoint}")
    saved_step_value = state.get("best_global_step")
    if saved_step_value is None:
        raise ValueError(
            f"Best checkpoint has no best_global_step: {best_checkpoint}"
        )
    saved_step = int(saved_step_value)
    checkpoint_step = int(state.get("global_step", -1))
    saved_last_eval_step = int(state.get("last_eval_step", -1))
    if (
        saved_step < 0
        or checkpoint_step != saved_step
        or saved_last_eval_step != saved_step
    ):
        raise ValueError(
            "Best checkpoint state is not from its recorded best dev step: "
            f"global_step={checkpoint_step}, best_global_step={saved_step}, "
            f"last_eval_step={saved_last_eval_step}"
        )
    if saved_step is not None and saved_step > progress.global_step:
        raise ValueError(
            "Cannot resume an earlier checkpoint into an output directory that "
            f"contains a future best checkpoint: best_step={saved_step}, "
            f"resume_step={progress.global_step}"
        )
    if progress.best_eval_asr_loss is None or saved_loss < progress.best_eval_asr_loss:
        progress.best_eval_asr_loss = saved_loss
        progress.best_global_step = saved_step
    if saved_last_eval_step > progress.global_step:
        raise ValueError(
            "Cannot merge a best checkpoint evaluated after the resume state: "
            f"last_eval_step={saved_last_eval_step}, "
            f"resume_step={progress.global_step}"
        )
    progress.last_eval_step = max(progress.last_eval_step, saved_last_eval_step)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _move_optimizer_state(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _autocast_context(device: torch.device, mixed_precision: str) -> Any:
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _evaluate_dev(
    model: PhoWhisperLoRA,
    loader: DataLoader,
    *,
    device: torch.device,
    mixed_precision: str,
) -> DevMetrics:
    """Evaluate the declared dev manifest without gradients or train augmentation."""

    was_training = model.training
    model.eval()
    num_samples = 0
    asr_tokens = 0
    tone_targets = 0
    asr_loss_sum = 0.0
    tone_loss_sum = 0.0
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = _move_batch(batch, device)
                batch_size = int(batch["input_features"].size(0))
                batch_asr_tokens = int(batch["labels"].ne(-100).sum().item())
                batch_tone_targets = int(batch["tone_labels"].ne(-100).sum().item())
                if batch_asr_tokens == 0:
                    raise ValueError("Dev batch contains no supervised ASR tokens")
                with _autocast_context(device, mixed_precision):
                    outputs = model(**batch)
                asr_loss = outputs["asr_loss"]
                tone_loss = outputs["tone_loss"]
                total_loss = outputs["total_loss"]
                if float(model.lambda_tone) > 0.0 and tone_loss is None:
                    raise ValueError(
                        "Tone-aware model returned no tone loss during dev evaluation"
                    )
                if not torch.isfinite(asr_loss) or not torch.isfinite(total_loss):
                    raise FloatingPointError("Non-finite loss during dev evaluation")
                if tone_loss is not None and not torch.isfinite(tone_loss):
                    raise FloatingPointError("Non-finite tone loss during dev evaluation")
                num_samples += batch_size
                asr_tokens += batch_asr_tokens
                tone_targets += batch_tone_targets
                asr_loss_sum += float(asr_loss.detach().item()) * batch_asr_tokens
                if tone_loss is not None and batch_tone_targets > 0:
                    tone_loss_sum += float(tone_loss.detach().item()) * batch_tone_targets
    finally:
        model.train(was_training)
    if num_samples == 0 or asr_tokens == 0:
        raise ValueError("Dev loader is empty")
    corpus_asr_loss = asr_loss_sum / asr_tokens
    corpus_tone_loss = tone_loss_sum / tone_targets if tone_targets else 0.0
    corpus_total_loss = (
        corpus_asr_loss + float(model.lambda_tone) * corpus_tone_loss
    )
    return DevMetrics(
        num_samples=num_samples,
        asr_tokens=asr_tokens,
        tone_targets=tone_targets,
        asr_loss=corpus_asr_loss,
        tone_loss=corpus_tone_loss,
        total_loss=corpus_total_loss,
    )


def _evaluate_and_record_dev(
    *,
    model: PhoWhisperLoRA,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    progress: TrainerProgress,
    epoch: int,
    dev_metrics_path: Path,
    output_dir: Path,
    processor: WhisperProcessor,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: ConfigDict,
    logger: logging.Logger,
) -> bool:
    """Evaluate one pending model state exactly once and update best atomically."""

    if progress.global_step <= progress.last_eval_step:
        return False
    dev_metrics = _evaluate_dev(
        model,
        loader,
        device=device,
        mixed_precision=mixed_precision,
    )
    is_best = _is_strictly_better(
        dev_metrics.asr_loss,
        progress.best_eval_asr_loss,
    )
    progress.last_eval_step = progress.global_step
    if is_best:
        progress.best_eval_asr_loss = dev_metrics.asr_loss
        progress.best_global_step = progress.global_step
    _append_dev_metrics(
        dev_metrics_path,
        {
            "epoch": epoch,
            "global_step": progress.global_step,
            "num_samples": dev_metrics.num_samples,
            "asr_tokens": dev_metrics.asr_tokens,
            "tone_targets": dev_metrics.tone_targets,
            "dev_asr_loss": dev_metrics.asr_loss,
            "dev_tone_loss": dev_metrics.tone_loss,
            "dev_total_loss": dev_metrics.total_loss,
            "is_best": str(is_best).lower(),
        },
    )
    logger.info(
        "dev epoch=%d step=%d samples=%d asr_loss=%.6f "
        "tone_loss=%.6f total_loss=%.6f best=%s",
        epoch,
        progress.global_step,
        dev_metrics.num_samples,
        dev_metrics.asr_loss,
        dev_metrics.tone_loss,
        dev_metrics.total_loss,
        is_best,
    )
    if is_best:
        best_checkpoint = output_dir / "best"
        _save_checkpoint(
            target=best_checkpoint,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            progress=progress,
            config=config,
            overwrite=True,
        )
        logger.info(
            "Saved best checkpoint by dev_asr_loss=%.6f: %s",
            dev_metrics.asr_loss,
            best_checkpoint,
        )
    return is_best


def _iter_from_batch(loader: DataLoader, start_batch: int) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    for batch_index, batch in enumerate(loader):
        if batch_index < start_batch:
            continue
        yield batch_index, batch


def train_experiment(
    config: ConfigDict,
    *,
    device_arg: str = "auto",
    resume_checkpoint: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Train one lambda experiment and return its best dev-loss checkpoint."""

    _assert_training_gate(config)
    training = config["training"]
    model_config = config["model"]
    lora_config = model_config["lora"]
    data_config = config["data"]
    noise_config = config["noise"]
    protocol_config = config["protocol"]
    verify_audio = bool(protocol_config.get("verify_audio_sha256", True))
    method_integrity: dict[str, str] = {}
    if str(training.get("run_scope", "")).strip().casefold() == "formal":
        method_lock = protocol_config.get("method_lock")
        if not method_lock:
            raise RuntimeError(
                "Formal paper-v2 training requires protocol.method_lock"
            )
        method_integrity = verify_method_lock(
            method_lock,
            config=config,
            repo_root=REPO_ROOT,
            formal=True,
            verify_audio=verify_audio,
        )
    train_integrity = verify_locked_vivos_manifest(
        data_config["train_manifest"],
        split_name="train",
        split_lock_path=protocol_config["split_lock"],
        verify_audio=verify_audio,
    )
    dev_integrity = verify_locked_vivos_manifest(
        data_config["valid_manifest"],
        split_name="dev",
        split_lock_path=protocol_config["split_lock"],
        verify_audio=verify_audio,
    )
    if train_integrity["split_lock_sha256"] != dev_integrity["split_lock_sha256"]:
        raise ValueError("Train and dev manifests do not share one split lock")
    noise_runtime: dict[str, Any] = {
        "noise_enabled": bool(noise_config.get("enable_train_noise", False))
    }
    if noise_runtime["noise_enabled"]:
        noise_manifest = Path(str(noise_config.get("noise_manifest", "")))
        noise_rows = validate_jsonl_audio_manifest_fields(
            noise_manifest,
            required=("audio",),
        )
        noise_runtime.update(
            {
                "noise_manifest_sha256": sha256_file(noise_manifest),
                "noise_manifest_rows": len(noise_rows),
                "noise_audio_paths_verified": True,
                "noise_audio_hashes_verified": bool(method_integrity and verify_audio),
            }
        )
    config["_runtime_protocol"] = {
        "protocol_version": PROTOCOL_VERSION,
        "split_lock_sha256": train_integrity["split_lock_sha256"],
        "train_manifest_sha256": train_integrity["manifest_sha256"],
        "dev_manifest_sha256": dev_integrity["manifest_sha256"],
        "audio_hashes_verified": verify_audio,
        "training_contract_sha256": training_contract_sha256(config),
        "training_scope": str(training.get("run_scope", "")).strip().casefold(),
        **method_integrity,
        **noise_runtime,
    }
    validate_manifest_declared_split(
        data_config["train_manifest"],
        expected_split="train",
    )
    validate_manifest_declared_split(
        data_config["valid_manifest"],
        expected_split="dev",
    )
    output_dir = Path(str(training["output_dir"]))
    resume_path = Path(resume_checkpoint) if resume_checkpoint else None
    if resume_path is not None:
        _assert_resume_compatible(config, resume_path)
    _prepare_output_directory(output_dir, resume=resume_path is not None, overwrite=overwrite)
    log_path = Path(str(config.get("logging", {}).get("file", output_dir / "training.log")))
    if log_path.exists() and resume_path is None and not overwrite:
        raise FileExistsError(f"Training log already exists: {log_path}. Use --overwrite explicitly.")
    logger = _configure_logger(log_path, append=resume_path is not None)

    seed_settings = set_global_seed(
        int(config["seed"]),
        deterministic=bool(training.get("deterministic", True)),
    )
    device = choose_device(device_arg)
    mixed_precision = str(training.get("mixed_precision", "fp16")).lower()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision must be one of: no, fp16, bf16")
    if device.type != "cuda" and mixed_precision != "no":
        logger.warning("Mixed precision %s disabled because selected device is %s", mixed_precision, device)
        mixed_precision = "no"
    if mixed_precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but the CUDA device does not support it")

    logger.info("Python seed settings: %s", seed_settings)
    logger.info("Device: %s", device)
    logger.info("PyTorch: %s | CUDA build: %s", torch.__version__, torch.version.cuda)
    logger.info("Resolved config: %s", config.get("_config_path", "CLI overrides"))

    if resume_path is not None:
        adapter_checkpoint = _checkpoint_adapter_path(resume_path)
    else:
        adapter_checkpoint = None

    processor = WhisperProcessor.from_pretrained(
        str(model_config["name_or_path"]),
        revision=str(model_config["revision"]),
        language=str(model_config.get("language", "vi")),
        task=str(model_config.get("task", "transcribe")),
    )
    model = build_trainable_model(
        model_name_or_path=str(model_config["name_or_path"]),
        revision=str(model_config["revision"]),
        lambda_tone=float(training["lambda_tone"]),
        lora_r=int(lora_config["r"]),
        lora_alpha=int(lora_config["lora_alpha"]),
        lora_dropout=float(lora_config["lora_dropout"]),
        target_modules=lora_config["target_modules"],
        adapter_checkpoint=adapter_checkpoint,
    )
    if bool(training.get("gradient_checkpointing", True)):
        base_model = model.asr_model.get_base_model()
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
    assert_only_lora_and_tone_trainable(model)
    stats = parameter_stats(model)
    logger.info(
        "Parameters | total=%d trainable=%d ratio=%.6f%%",
        stats.total,
        stats.trainable,
        stats.trainable_ratio * 100.0,
    )

    train_dataset = _build_dataset(config, processor, split="train")
    dev_dataset = _build_dataset(config, processor, split="dev")
    decoder_start_token_id = int(model.asr_model.config.decoder_start_token_id)
    first_loader = _build_train_loader(
        train_dataset,
        processor,
        decoder_start_token_id,
        config,
        epoch=0,
    )
    dev_loader = _build_eval_loader(
        dev_dataset,
        processor,
        decoder_start_token_id,
        config,
    )
    gradient_accumulation = int(training.get("gradient_accumulation_steps", 1))
    updates_per_epoch = max(1, math.ceil(len(first_loader) / gradient_accumulation))
    epochs = int(training.get("num_train_epochs", 1))
    total_steps = updates_per_epoch * epochs
    if training.get("max_train_steps") is not None:
        total_steps = min(total_steps, int(training["max_train_steps"]))
    warmup_steps = training.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = round(total_steps * float(training.get("warmup_ratio", 0.0)))

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=max(total_steps, 1),
    )
    use_scaler = device.type == "cuda" and mixed_precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    progress = TrainerProgress()
    if resume_path is not None:
        progress = _load_training_state(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        _merge_saved_best(progress, output_dir / "best", config)
        logger.info("Resumed from %s at global_step=%d", resume_path, progress.global_step)

    model.to(device)
    _move_optimizer_state(optimizer, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = Path(str(config.get("logging", {}).get("metrics_csv", output_dir / "training_metrics.csv")))
    _write_metrics_header(metrics_path, append=resume_path is not None)
    dev_metrics_path = Path(
        str(config.get("logging", {}).get("dev_metrics_csv", output_dir / "dev_metrics.csv"))
    )
    _write_dev_metrics_header(dev_metrics_path, append=resume_path is not None)
    log_steps = max(1, int(training.get("log_steps", 10)))
    save_steps = max(1, int(training.get("save_steps", 500)))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    max_train_steps = training.get("max_train_steps")
    accumulated_asr = 0.0
    accumulated_tone = 0.0
    accumulated_total = 0.0
    accumulated_batches = 0
    stop_training = _reached_max_train_steps(progress, max_train_steps)
    pending_resume_evaluation = _resume_needs_dev_evaluation(
        progress,
        is_resume=resume_path is not None,
        reached_max_steps=stop_training,
    )
    if pending_resume_evaluation:
        completed_epoch = (
            max(progress.epoch - 1, 0)
            if progress.next_batch_index == 0
            else progress.epoch
        )
        _evaluate_and_record_dev(
            model=model,
            loader=dev_loader,
            device=device,
            mixed_precision=mixed_precision,
            progress=progress,
            epoch=completed_epoch,
            dev_metrics_path=dev_metrics_path,
            output_dir=output_dir,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            logger=logger,
        )

    epoch_range = range(progress.epoch, epochs) if not stop_training else ()
    for epoch in epoch_range:
        loader = _build_train_loader(
            train_dataset,
            processor,
            decoder_start_token_id,
            config,
            epoch=epoch,
        )
        start_batch = progress.next_batch_index if epoch == progress.epoch else 0
        for batch_index, batch in _iter_from_batch(loader, start_batch):
            batch = _move_batch(batch, device)
            with _autocast_context(device, mixed_precision):
                outputs = model(**batch)
                total_loss = outputs["total_loss"]
                asr_loss = outputs["asr_loss"]
                tone_loss = outputs["tone_loss"]
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"Non-finite total loss at epoch={epoch}, batch={batch_index}: {total_loss.item()}"
                    )
                scaled_loss = total_loss / gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_asr += float(asr_loss.detach().item())
            accumulated_tone += 0.0 if tone_loss is None else float(tone_loss.detach().item())
            accumulated_total += float(total_loss.detach().item())
            accumulated_batches += 1

            should_update = (batch_index + 1) % gradient_accumulation == 0 or (batch_index + 1) == len(loader)
            if not should_update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress.global_step += 1
            next_batch_index = batch_index + 1
            next_epoch = epoch
            if next_batch_index >= len(loader):
                next_epoch = epoch + 1
                next_batch_index = 0
            progress.epoch = next_epoch
            progress.next_batch_index = next_batch_index

            if progress.global_step % log_steps == 0 or progress.global_step == 1:
                denominator = max(accumulated_batches, 1)
                row = {
                    "epoch": epoch,
                    "global_step": progress.global_step,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "asr_loss": accumulated_asr / denominator,
                    "tone_loss": accumulated_tone / denominator,
                    "total_loss": accumulated_total / denominator,
                }
                _append_metrics(metrics_path, row)
                logger.info(
                    "epoch=%d step=%d asr_loss=%.6f tone_loss=%.6f total_loss=%.6f lr=%.8f",
                    epoch,
                    progress.global_step,
                    row["asr_loss"],
                    row["tone_loss"],
                    row["total_loss"],
                    row["learning_rate"],
                )
                accumulated_asr = accumulated_tone = accumulated_total = 0.0
                accumulated_batches = 0

            if progress.global_step % save_steps == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{progress.global_step:06d}"
                _save_checkpoint(
                    target=checkpoint_path,
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=progress,
                    config=config,
                )
                logger.info("Saved checkpoint: %s", checkpoint_path)

            if max_train_steps is not None and progress.global_step >= int(max_train_steps):
                stop_training = True
                break
        _evaluate_and_record_dev(
            model=model,
            loader=dev_loader,
            device=device,
            mixed_precision=mixed_precision,
            progress=progress,
            epoch=epoch,
            dev_metrics_path=dev_metrics_path,
            output_dir=output_dir,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            logger=logger,
        )
        if stop_training:
            break

    best_checkpoint = output_dir / "best"
    if not best_checkpoint.exists():
        raise RuntimeError("Training completed without a best dev checkpoint")
    final_checkpoint = output_dir / "final"
    _save_checkpoint(
        target=final_checkpoint,
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        progress=progress,
        config=config,
        overwrite=overwrite,
    )
    logger.info("Saved archival final checkpoint: %s", final_checkpoint)
    logger.info(
        "Selected best checkpoint at step=%s with dev_asr_loss=%s: %s",
        progress.best_global_step,
        progress.best_eval_asr_loss,
        best_checkpoint,
    )
    return best_checkpoint
