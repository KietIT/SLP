from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import compute_all


POOL_COLUMNS = [
    "source_utt_id",
    "dataset",
    "split",
    "clean_path",
    "transcript",
    "duration",
    "seed",
    "pool_rank",
]

BENCHMARK_COLUMNS = [
    "utt_id",
    "dataset",
    "split",
    "condition",
    "clean_path",
    "noisy_path",
    "audio_path",
    "snr",
    "noise_type",
    "noise_path",
    "transcript",
    "duration",
    "seed",
    "source_utt_id",
]

PREDICTION_COLUMNS = [
    "utt_id",
    "dataset",
    "model",
    "model_size",
    "snr",
    "noise_type",
    "ref",
    "hyp",
]

METRIC_COLUMNS_BY_SNR = [
    "model",
    "model_size",
    "snr",
    "n",
    "wer",
    "cer",
    "ter_simple",
    "der_simple",
    "fcer_simple",
    "swdr_simple",
]

METRIC_COLUMNS_BY_NOISE_TYPE = [
    "model",
    "model_size",
    "noise_type",
    "n",
    "wer",
    "cer",
    "ter_simple",
    "der_simple",
    "fcer_simple",
    "swdr_simple",
]

ZERO_SHOT_MODEL_SPECS = {
    "whisper_tiny": {
        "model_name_or_path": "openai/whisper-tiny",
        "model": "whisper",
        "model_size": "tiny",
        "filename": "pred_whisper_tiny.csv",
    },
    "whisper_base": {
        "model_name_or_path": "openai/whisper-base",
        "model": "whisper",
        "model_size": "base",
        "filename": "pred_whisper_base.csv",
    },
    "whisper_small": {
        "model_name_or_path": "openai/whisper-small",
        "model": "whisper",
        "model_size": "small",
        "filename": "pred_whisper_small.csv",
    },
    "phowhisper_tiny": {
        "model_name_or_path": "vinai/PhoWhisper-tiny",
        "model": "phowhisper",
        "model_size": "tiny",
        "filename": "pred_phowhisper_tiny.csv",
    },
    "phowhisper_base": {
        "model_name_or_path": "vinai/PhoWhisper-base",
        "model": "phowhisper",
        "model_size": "base",
        "filename": "pred_phowhisper_base.csv",
    },
    "phowhisper_small": {
        "model_name_or_path": "vinai/PhoWhisper-small",
        "model": "phowhisper",
        "model_size": "small",
        "filename": "pred_phowhisper_small.csv",
    },
}


def atomic_write_csv(path: str | Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    row_list = list(rows)
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in row_list:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp.replace(path)


def read_csv_rows(path: str | Path) -> tuple[list[dict], list[str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def validate_columns(path: str | Path, actual: Sequence[str], expected: Sequence[str], exact: bool = False) -> None:
    if exact:
        if list(actual) != list(expected):
            raise ValueError(f"{path} must have columns {list(expected)}, found {list(actual)}")
        return
    missing = [name for name in expected if name not in actual]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def normalize_snr(value: object) -> str:
    text = str(value).strip()
    if text == "":
        return "unknown"
    if text.lower() == "clean":
        return "clean"
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number)


def selected_model_specs(model_keys: Sequence[str] | None = None) -> dict[str, dict]:
    if model_keys is None or len(model_keys) == 0:
        return dict(ZERO_SHOT_MODEL_SPECS)
    unknown = [key for key in model_keys if key not in ZERO_SHOT_MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")
    return {key: ZERO_SHOT_MODEL_SPECS[key] for key in model_keys}


def prediction_path_for(pred_dir: str | Path, spec: dict) -> Path:
    return Path(pred_dir) / spec["filename"]


def read_prediction_file(path: str | Path) -> list[dict]:
    rows, columns = read_csv_rows(path)
    validate_columns(path, columns, PREDICTION_COLUMNS, exact=True)
    return rows


def _numeric_snr_sort_key(label: str) -> tuple[int, float | str]:
    if label == "clean":
        return (0, 0.0)
    try:
        return (1, -float(label))
    except ValueError:
        return (2, label)


def _metric_row(model: str, model_size: str, group_name: str, group_key: str, rows: Sequence[dict]) -> dict:
    refs = [row.get("ref", "") for row in rows]
    hyps = [row.get("hyp", "") for row in rows]
    metrics = compute_all(refs, hyps)
    return {
        "model": model,
        "model_size": model_size,
        group_name: group_key,
        "n": len(rows),
        **metrics,
    }


def aggregate_zero_shot_results(
    pred_dir: str | Path,
    out_by_snr: str | Path,
    out_by_noise_type: str | Path | None = None,
    expected_rows: int | None = 1500,
    allow_partial: bool = False,
    model_keys: Sequence[str] | None = None,
) -> dict:
    specs = selected_model_specs(model_keys)
    snr_results: list[dict] = []
    noise_type_results: list[dict] = []
    counts: dict[str, int] = {}

    for key, spec in specs.items():
        path = prediction_path_for(pred_dir, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {key}: {path}")
        rows = read_prediction_file(path)
        if expected_rows is not None and len(rows) != expected_rows and not allow_partial:
            raise ValueError(f"{path} has {len(rows)} rows, expected {expected_rows}")
        if not rows:
            raise ValueError(f"{path} is empty")
        counts[key] = len(rows)
        model = rows[0].get("model", spec["model"])
        model_size = rows[0].get("model_size", spec["model_size"])

        by_snr: dict[str, list[dict]] = defaultdict(list)
        noisy_rows: list[dict] = []
        for row in rows:
            snr = normalize_snr(row.get("snr", "unknown"))
            row["snr"] = snr
            by_snr[snr].append(row)
            if snr != "clean":
                noisy_rows.append(row)

        for snr in sorted(by_snr, key=_numeric_snr_sort_key):
            snr_results.append(_metric_row(model, model_size, "snr", snr, by_snr[snr]))
        if noisy_rows:
            snr_results.append(_metric_row(model, model_size, "snr", "noisy_all", noisy_rows))
        snr_results.append(_metric_row(model, model_size, "snr", "all", rows))

        by_noise_type: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            noise_type = str(row.get("noise_type", "unknown") or "unknown")
            by_noise_type[noise_type].append(row)
        for noise_type in sorted(by_noise_type):
            noise_type_results.append(
                _metric_row(model, model_size, "noise_type", noise_type, by_noise_type[noise_type])
            )

    atomic_write_csv(out_by_snr, snr_results, METRIC_COLUMNS_BY_SNR)
    if out_by_noise_type is not None:
        atomic_write_csv(out_by_noise_type, noise_type_results, METRIC_COLUMNS_BY_NOISE_TYPE)
    return {
        "models": list(specs),
        "counts": counts,
        "out_by_snr": str(out_by_snr),
        "out_by_noise_type": str(out_by_noise_type) if out_by_noise_type is not None else "",
    }
