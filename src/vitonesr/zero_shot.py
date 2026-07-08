from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .noise import read_audio
from .prediction import (
    BENCHMARK_COLUMNS,
    PREDICTION_COLUMNS,
    atomic_write_csv,
    read_csv_rows,
    read_prediction_file,
    validate_columns,
)


@dataclass
class ZeroShotConfig:
    benchmark_manifest: Path
    model_name_or_path: str
    model: str
    model_size: str
    out: Path
    sample_rate: int = 16000
    batch_size: int = 4
    device: str = "auto"
    language: str = "vi"
    task: str = "transcribe"
    max_new_tokens: int = 128
    resume: bool = True
    overwrite: bool = False
    max_audio_seconds: float = 30.0


def choose_device(device_arg: str, torch_module=None):
    if torch_module is None:
        import torch as torch_module

    if device_arg != "auto":
        return torch_module.device(device_arg)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


class ZeroShotInferencer:
    def __init__(self, config: ZeroShotConfig):
        self.config = config

    def run(self) -> dict:
        import torch

        if self.config.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        with torch.no_grad():
            return self._run(torch)

    def _run(self, torch) -> dict:
        manifest_rows, columns = read_csv_rows(self.config.benchmark_manifest)
        validate_columns(self.config.benchmark_manifest, columns, BENCHMARK_COLUMNS)
        if not manifest_rows:
            raise ValueError("Benchmark manifest is empty.")

        manifest_ids = [row["utt_id"] for row in manifest_rows]
        manifest_id_set = set(manifest_ids)
        results_by_id: dict[str, dict] = {}
        existing_count = 0
        if self.config.out.exists() and not self.config.overwrite:
            existing_rows = read_prediction_file(self.config.out)
            existing_count = len(existing_rows)
            existing_ids = [row["utt_id"] for row in existing_rows]
            if len(set(existing_ids)) != len(existing_ids):
                raise ValueError(f"{self.config.out} contains duplicate utt_id values")
            if existing_count == len(manifest_rows) and set(existing_ids) == manifest_id_set:
                return {
                    "status": "skipped",
                    "reason": "prediction file is complete",
                    "out": str(self.config.out),
                    "rows": existing_count,
                }
            if not self.config.resume:
                detail = (
                    "utt_id values do not match the benchmark manifest"
                    if existing_count == len(manifest_rows)
                    else f"expected {len(manifest_rows)}"
                )
                raise ValueError(
                    f"{self.config.out} has {existing_count} rows, {detail}. "
                    "Use --resume or --overwrite."
                )
            for row in existing_rows:
                if row["utt_id"] in manifest_id_set:
                    results_by_id[row["utt_id"]] = row

        pending_rows = [row for row in manifest_rows if self.config.overwrite or row["utt_id"] not in results_by_id]
        if not pending_rows:
            ordered = self._ordered_results(manifest_rows, results_by_id)
            atomic_write_csv(self.config.out, ordered, PREDICTION_COLUMNS)
            return {
                "status": "complete",
                "out": str(self.config.out),
                "rows": len(ordered),
            }

        processor, model, device, dtype = self._load_model(torch)
        max_len = int(self.config.max_audio_seconds * self.config.sample_rate)
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = lambda values, **_: values

        progress = tqdm(
            range(0, len(pending_rows), self.config.batch_size),
            desc=f"infer {self.config.model}_{self.config.model_size}",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        )
        for start in progress:
            batch_rows = pending_rows[start:start + self.config.batch_size]
            wavs = []
            for row in batch_rows:
                wav = read_audio(row["audio_path"], sr=self.config.sample_rate)
                if len(wav) > max_len:
                    wav = wav[:max_len]
                wavs.append(wav)

            features = processor.feature_extractor(
                wavs,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt",
                padding=True,
            ).input_features
            features = features.to(device=device, dtype=dtype)
            pred_ids = self._generate(model, processor, features)
            hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)

            for row, hyp in zip(batch_rows, hyps):
                results_by_id[row["utt_id"]] = {
                    "utt_id": row["utt_id"],
                    "dataset": row.get("dataset", ""),
                    "model": self.config.model,
                    "model_size": self.config.model_size,
                    "snr": row.get("snr", ""),
                    "noise_type": row.get("noise_type", ""),
                    "ref": row.get("transcript", ""),
                    "hyp": hyp,
                }
            ordered = self._ordered_results(manifest_rows, results_by_id)
            atomic_write_csv(self.config.out, ordered, PREDICTION_COLUMNS)

        ordered = self._ordered_results(manifest_rows, results_by_id)
        if len(ordered) != len(manifest_rows):
            raise RuntimeError(f"Inference finished with {len(ordered)} rows, expected {len(manifest_rows)}")
        atomic_write_csv(self.config.out, ordered, PREDICTION_COLUMNS)
        return {
            "status": "complete",
            "out": str(self.config.out),
            "rows": len(ordered),
            "existing_rows": existing_count,
        }

    def _load_model(self, torch):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as transformers_logging

        device = choose_device(self.config.device, torch)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        try:
            processor = WhisperProcessor.from_pretrained(
                self.config.model_name_or_path,
                language=self.config.language,
                task=self.config.task,
            )
            model = WhisperForConditionalGeneration.from_pretrained(self.config.model_name_or_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model {self.config.model_name_or_path}: {exc}") from exc
        transformers_logging.set_verbosity_error()
        model = model.to(device=device, dtype=dtype)
        model.eval()
        return processor, model, device, dtype

    def _generate(self, model, processor, features):
        generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.config.max_new_tokens,
        }
        if self.config.language:
            generate_kwargs["language"] = self.config.language
        if self.config.task:
            generate_kwargs["task"] = self.config.task
        try:
            with _suppress_generation_length_warning():
                return model.generate(features, **generate_kwargs)
        except TypeError:
            fallback_kwargs = {
                "do_sample": False,
                "max_new_tokens": self.config.max_new_tokens,
                "forced_decoder_ids": processor.get_decoder_prompt_ids(
                    language=self.config.language,
                    task=self.config.task,
                ),
            }
            with _suppress_generation_length_warning():
                return model.generate(features, **fallback_kwargs)

    def _ordered_results(self, manifest_rows: list[dict], results_by_id: dict[str, dict]) -> list[dict]:
        ordered = []
        for row in manifest_rows:
            utt_id = row["utt_id"]
            if utt_id in results_by_id:
                ordered.append(results_by_id[utt_id])
        return ordered


class _GenerationLengthWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Both `max_new_tokens`" not in message or "`max_length`" not in message


class _suppress_generation_length_warning:
    def __enter__(self):
        self._filter = _GenerationLengthWarningFilter()
        self._loggers = [
            logging.getLogger("transformers"),
            logging.getLogger("transformers.generation.utils"),
        ]
        for logger in self._loggers:
            logger.addFilter(self._filter)
        return self

    def __exit__(self, exc_type, exc, tb):
        for logger in self._loggers:
            logger.removeFilter(self._filter)
        return False
