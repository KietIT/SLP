from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.vitonesr.metrics import compute_all
from src.vitonesr.noise import read_audio
from src.vitonesr.prediction import atomic_write_csv, normalize_snr, read_csv_rows, validate_columns

from .config import ConfigDict


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

ABLATION_RESULT_COLUMNS = [
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "split",
    "snr",
    "noise_type",
    "num_samples",
    "wer",
    "cer",
    "ter",
    "der",
    "fcer",
    "swdr",
    "checkpoint_path",
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
    snr = normalize_snr(row.get("snr", "clean"))
    return {
        "utt_id": str(row.get("utt_id") or Path(str(audio_path)).stem),
        "dataset": str(row.get("dataset", "")),
        "audio_path": str(audio_path),
        "snr": snr,
        "noise_type": str(row.get("noise_type", "clean" if snr == "clean" else "")),
        "ref": str(reference),
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


def resolve_checkpoint(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")
    if (candidate / "adapter" / "adapter_config.json").exists():
        return candidate, candidate / "adapter"
    if (candidate / "final" / "adapter" / "adapter_config.json").exists():
        return candidate / "final", candidate / "final" / "adapter"
    if (candidate / "adapter_config.json").exists():
        return candidate.parent, candidate
    raise FileNotFoundError(f"Could not find a PEFT adapter under checkpoint: {candidate}")


def _batched(rows: Sequence[dict[str, str]], batch_size: int) -> Iterable[Sequence[dict[str, str]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


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
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    evaluation = config["evaluation"]
    model_config = config["model"]
    checkpoint_root, adapter_path = resolve_checkpoint(checkpoint)
    prediction_path = Path(output_path or evaluation["prediction_path"])
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"Prediction file already exists: {prediction_path}. Use --overwrite explicitly.")

    rows = load_benchmark_rows(
        manifest or evaluation["manifest"],
        subset=subset,
        snrs=snrs,
        noise_types=noise_types,
        limit=limit,
    )
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
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    local_processor = checkpoint_root / "processor"
    processor_source = str(local_processor) if local_processor.exists() else str(model_config["name_or_path"])
    processor = WhisperProcessor.from_pretrained(
        processor_source,
        language=str(model_config.get("language", "vi")),
        task=str(model_config.get("task", "transcribe")),
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(str(model_config["name_or_path"]))
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
    }
    prediction_rows: list[dict[str, Any]] = []
    train_type = str(config["experiment"]["train_type"])
    lambda_tone = float(config["training"]["lambda_tone"])
    seed = int(config["seed"])

    with torch.inference_mode():
        batches = list(_batched(rows, batch_size))
        for row_batch in tqdm(batches, desc=f"evaluate lambda={lambda_tone:g}"):
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
                    "forced_decoder_ids": processor.get_decoder_prompt_ids(
                        language=generate_kwargs["language"],
                        task=generate_kwargs["task"],
                    ),
                }
                generated = model.generate(input_features, attention_mask=attention_mask, **fallback)
            hypotheses = processor.batch_decode(generated, skip_special_tokens=True)
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

    atomic_write_csv(prediction_path, prediction_rows, PREDICTION_COLUMNS)
    return prediction_path


def validate_prediction_schema(path: str | Path) -> list[dict[str, str]]:
    rows, columns = read_csv_rows(path)
    validate_columns(path, columns, PREDICTION_COLUMNS, exact=True)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _result_row(
    rows: Sequence[dict[str, str]],
    *,
    split: str,
    snr: str,
    noise_type: str,
    checkpoint_path: str,
    prediction_path: str,
) -> dict[str, Any]:
    metrics = compute_all([row["ref"] for row in rows], [row["hyp"] for row in rows])
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
        "wer": metrics.get("wer", ""),
        "cer": metrics.get("cer", ""),
        "ter": metrics.get("ter_simple", metrics.get("ter", "")),
        "der": metrics.get("der_simple", metrics.get("der", "")),
        "fcer": metrics.get("fcer_simple", metrics.get("fcer", "")),
        "swdr": metrics.get("swdr_simple", metrics.get("swdr", "")),
        "checkpoint_path": checkpoint_path,
        "prediction_path": prediction_path,
    }


def aggregate_prediction_file(
    prediction_path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> list[dict[str, Any]]:
    rows = validate_prediction_schema(prediction_path)
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
    return output


def write_ablation_results(
    experiment_artifacts: Sequence[tuple[str | Path, str | Path]],
    output_path: str | Path,
    *,
    require_lambdas: Sequence[float] | None = (0.0, 0.05, 0.1, 0.3, 0.5),
    overwrite: bool = False,
) -> Path:
    all_rows: list[dict[str, Any]] = []
    observed_lambdas: set[float] = set()
    for checkpoint_path, prediction_path in experiment_artifacts:
        rows = aggregate_prediction_file(prediction_path, checkpoint_path=checkpoint_path)
        all_rows.extend(rows)
        observed_lambdas.add(float(rows[0]["lambda"]))
    if require_lambdas is not None:
        missing = sorted(set(float(value) for value in require_lambdas) - observed_lambdas)
        if missing:
            raise ValueError(f"Cannot write complete lambda ablation; missing real predictions for: {missing}")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Ablation result already exists: {path}. Use --overwrite explicitly.")
    atomic_write_csv(path, all_rows, ABLATION_RESULT_COLUMNS)
    return path
