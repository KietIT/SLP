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
from .reproducibility import seed_worker, set_global_seed
from .training_data import DeterministicASRTrainingDataset


LOGGER_NAME = "vitonesr.phat.training"


@dataclass
class TrainerProgress:
    epoch: int = 0
    next_batch_index: int = 0
    global_step: int = 0


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return selected


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


def _build_dataset(config: ConfigDict, processor: WhisperProcessor) -> DeterministicASRTrainingDataset:
    training = config["training"]
    data = config["data"]
    noise = config.get("noise", {})
    enable_noise = bool(noise.get("enable_train_noise", False))
    return DeterministicASRTrainingDataset(
        str(data["train_manifest"]),
        processor,
        seed=int(config["seed"]),
        sample_rate=int(data.get("sample_rate", 16000)),
        max_audio_seconds=float(training.get("max_audio_seconds", 15.0)),
        max_label_length=training.get("max_label_length"),
        noise_manifest=str(noise["noise_manifest"]) if enable_noise else None,
        noise_prob=float(noise.get("prob", 0.0)) if enable_noise else 0.0,
        snr_choices=noise.get("snr_choices", [20, 10, 5, 0]),
        tone_label_policy=str(training.get("tone_label_policy", "last_subtoken")),
        max_samples=training.get("max_train_samples"),
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
    comparisons = [
        ("model.name_or_path", saved.get("model", {}).get("name_or_path"), config["model"]["name_or_path"]),
        ("training.lambda_tone", saved.get("training", {}).get("lambda_tone"), config["training"]["lambda_tone"]),
        ("model.lora.r", saved.get("model", {}).get("lora", {}).get("r"), config["model"]["lora"]["r"]),
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
        )
        writer.writeheader()


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(row))
        writer.writerow(row)


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
        shutil.rmtree(target)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
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
    serializable_config = {key: value for key, value in config.items() if not key.startswith("_")}
    with (temporary / "resolved_config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(serializable_config, config_file, sort_keys=False, allow_unicode=True)
    temporary.rename(target)


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
    return TrainerProgress(
        epoch=int(state["epoch"]),
        next_batch_index=int(state["next_batch_index"]),
        global_step=int(state["global_step"]),
    )


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
    """Train one lambda experiment and return the real final checkpoint path."""
    training = config["training"]
    model_config = config["model"]
    lora_config = model_config["lora"]
    output_dir = Path(str(training["output_dir"]))
    resume_path = Path(resume_checkpoint) if resume_checkpoint else None
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
        _assert_resume_compatible(config, resume_path)
        adapter_checkpoint = _checkpoint_adapter_path(resume_path)
    else:
        adapter_checkpoint = None

    processor = WhisperProcessor.from_pretrained(
        str(model_config["name_or_path"]),
        language=str(model_config.get("language", "vi")),
        task=str(model_config.get("task", "transcribe")),
    )
    model = build_trainable_model(
        model_name_or_path=str(model_config["name_or_path"]),
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

    dataset = _build_dataset(config, processor)
    first_loader = _build_train_loader(
        dataset,
        processor,
        int(model.asr_model.config.decoder_start_token_id),
        config,
        epoch=0,
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
        logger.info("Resumed from %s at global_step=%d", resume_path, progress.global_step)

    model.to(device)
    _move_optimizer_state(optimizer, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = Path(str(config.get("logging", {}).get("metrics_csv", output_dir / "training_metrics.csv")))
    _write_metrics_header(metrics_path, append=resume_path is not None)
    log_steps = max(1, int(training.get("log_steps", 10)))
    save_steps = max(1, int(training.get("save_steps", 500)))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    max_train_steps = training.get("max_train_steps")
    accumulated_asr = 0.0
    accumulated_tone = 0.0
    accumulated_total = 0.0
    accumulated_batches = 0
    stop_training = False

    for epoch in range(progress.epoch, epochs):
        loader = _build_train_loader(
            dataset,
            processor,
            int(model.asr_model.config.decoder_start_token_id),
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
        if stop_training:
            break

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
    logger.info("Training completed with real final checkpoint: %s", final_checkpoint)
    return final_checkpoint
