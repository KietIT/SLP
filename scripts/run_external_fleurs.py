"""Run the three approved PhoWhisper LoRA checkpoints on FLEURS Vietnamese.

The runner deliberately has a narrower contract than the VIVOS ablation
pipeline: FLEURS is a clean, external test set and only the ordinary LoRA,
tone-aware lambda=0.05, and tone-aware lambda=0.1 checkpoints are allowed.
Audio is never truncated.  Utterances longer than Whisper's 30 second input
window are split into consecutive, non-overlapping chunks and their decoded
texts are concatenated in temporal order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.analysis import (  # noqa: E402
    CANONICAL_PREDICTION_COLUMNS,
    METRIC_VERSION,
    compute_aligned_metric_result,
    load_prediction_csv,
    validate_prediction_rows,
)
from src.vitonesr.phat.config import load_experiment_config  # noqa: E402
from src.vitonesr.prediction import atomic_write_csv  # noqa: E402


DEFAULT_MANIFEST = Path("data/manifests/fleurs/test.jsonl")
DEFAULT_CONFIG_DIR = Path("configs/phat")
DEFAULT_CHECKPOINT_ROOT = Path("outputs/phat/checkpoints")
DEFAULT_OUTPUT_DIR = Path("outputs/external/fleurs")
DEFAULT_EXPECTED_ROWS = 857
DEFAULT_MAX_NEW_TOKENS = 440
SAMPLE_RATE = 16_000
MAX_CHUNK_SECONDS = 30.0

RESULT_COLUMNS = (
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "n",
    "wer",
    "cer",
    "ter",
    "der",
    "fcer",
    "swdr",
    "metric_version",
)


class ExternalFleursError(ValueError):
    """Raised when the external-test contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class RunTemplate:
    config_name: str
    checkpoint_name: str
    train_type: str
    lambda_value: str
    prediction_name: str


RUN_TEMPLATES = (
    RunTemplate(
        config_name="lambda_0.yaml",
        checkpoint_name="ckpt_lora_ordinary_lambda0",
        train_type="ordinary_lora",
        lambda_value="0",
        prediction_name="pred_lora_ordinary_lambda0.csv",
    ),
    RunTemplate(
        config_name="lambda_005.yaml",
        checkpoint_name="ckpt_tone_lora_lambda_005",
        train_type="tone_aware_lora",
        lambda_value="0.05",
        prediction_name="pred_tone_lora_lambda_005.csv",
    ),
    RunTemplate(
        config_name="lambda_01.yaml",
        checkpoint_name="ckpt_tone_lora_lambda_01",
        train_type="tone_aware_lora",
        lambda_value="0.1",
        prediction_name="pred_tone_lora_lambda_01.csv",
    ),
)


@dataclass(frozen=True, slots=True)
class ExternalRun:
    train_type: str
    lambda_value: str
    seed: str
    model_name_or_path: str
    language: str
    task: str
    checkpoint: Path
    prediction_name: str

    @property
    def run_metadata(self) -> dict[str, str]:
        return {
            "dataset": "fleurs",
            "model": "phowhisper",
            "model_size": "base",
            "train_type": self.train_type,
            "lambda": self.lambda_value,
            "seed": self.seed,
        }


class ChunkTranscriber(Protocol):
    def transcribe_chunk(self, waveform: Any) -> str:
        """Transcribe one waveform whose duration is at most 30 seconds."""

    def close(self) -> None:
        """Release model resources."""


TranscriberFactory = Callable[[ExternalRun, str, int], ChunkTranscriber]
AudioLoader = Callable[[str, int], Any]


def _canonical_lambda(value: object) -> str:
    try:
        number = float(str(value))
    except ValueError as error:
        raise ExternalFleursError(f"Invalid lambda value: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ExternalFleursError(f"Invalid lambda value: {value!r}")
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def build_external_runs(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> tuple[ExternalRun, ...]:
    """Resolve and validate the three fixed external-test configurations."""

    config_root = Path(config_dir)
    checkpoint_base = Path(checkpoint_root)
    runs: list[ExternalRun] = []
    for template in RUN_TEMPLATES:
        config = load_experiment_config(config_root / template.config_name)
        observed_type = str(config["experiment"]["train_type"])
        observed_lambda = _canonical_lambda(config["training"]["lambda_tone"])
        if observed_type != template.train_type or observed_lambda != template.lambda_value:
            raise ExternalFleursError(
                f"{template.config_name} must describe {template.train_type} "
                f"lambda={template.lambda_value}, found {observed_type} "
                f"lambda={observed_lambda}"
            )
        runs.append(
            ExternalRun(
                train_type=template.train_type,
                lambda_value=template.lambda_value,
                seed=str(int(config["seed"])),
                model_name_or_path=str(config["model"]["name_or_path"]),
                language=str(config["model"].get("language", "vi")),
                task=str(config["model"].get("task", "transcribe")),
                checkpoint=checkpoint_base / template.checkpoint_name / "final",
                prediction_name=template.prediction_name,
            )
        )
    return tuple(runs)


def _read_manifest_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"FLEURS manifest does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ExternalFleursError(f"Unsupported FLEURS manifest format: {path}")


def load_fleurs_manifest(
    path: str | Path,
    *,
    expected_rows: int | None = DEFAULT_EXPECTED_ROWS,
    require_audio: bool = True,
) -> list[dict[str, str]]:
    """Load a materialized FLEURS test manifest with canonical clean metadata."""

    manifest_path = Path(path)
    raw_rows = _read_manifest_records(manifest_path)
    if not raw_rows:
        raise ExternalFleursError(f"FLEURS manifest is empty: {manifest_path}")
    if expected_rows is not None and len(raw_rows) != expected_rows:
        raise ExternalFleursError(
            f"FLEURS test manifest has {len(raw_rows)} rows, expected {expected_rows}"
        )

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        audio_value = raw.get("audio_path") or raw.get("audio")
        reference = (
            raw.get("transcript")
            if raw.get("transcript") is not None
            else raw.get("transcription")
        )
        if reference is None:
            reference = raw.get("text") if raw.get("text") is not None else raw.get("ref")
        if not audio_value or reference is None or not str(reference).strip():
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} requires audio_path and a non-empty transcript"
            )

        audio_path = Path(str(audio_value))
        if not audio_path.is_absolute() and not audio_path.exists():
            relative_candidate = manifest_path.parent / audio_path
            if relative_candidate.exists():
                audio_path = relative_candidate
        if require_audio and not audio_path.exists():
            raise FileNotFoundError(
                f"{manifest_path}: row {row_number} audio does not exist: {audio_path}"
            )

        dataset = str(raw.get("dataset", "fleurs")).strip().casefold()
        split = str(raw.get("split", "test")).strip().casefold()
        snr = str(raw.get("snr", "clean")).strip().casefold()
        noise_type = str(raw.get("noise_type", "clean")).strip().casefold()
        if dataset != "fleurs" or split != "test":
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} must use dataset=fleurs and split=test"
            )
        if snr != "clean" or noise_type not in {"", "clean"}:
            raise ExternalFleursError(
                f"{manifest_path}: row {row_number} FLEURS external audio must be clean"
            )

        utt_id = str(raw.get("utt_id") or audio_path.stem).strip()
        if not utt_id:
            raise ExternalFleursError(f"{manifest_path}: row {row_number} has an empty utt_id")
        if utt_id in seen:
            raise ExternalFleursError(
                f"{manifest_path}: duplicate utt_id {utt_id!r} at row {row_number}"
            )
        seen.add(utt_id)
        rows.append(
            {
                "utt_id": utt_id,
                "dataset": "fleurs",
                "audio_path": str(audio_path),
                "ref": unicodedata.normalize("NFC", str(reference)),
                "snr": "clean",
                "noise_type": "clean",
            }
        )
    return rows


def split_waveform(
    waveform: Any,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_chunk_seconds: float = MAX_CHUNK_SECONDS,
) -> list[Any]:
    """Split a waveform completely into balanced deterministic <=30s chunks.

    Balancing avoids a very short final chunk, which can make Whisper
    hallucinate text from sub-second audio when an utterance is only slightly
    longer than the 30-second feature window.
    """

    if sample_rate < 1:
        raise ExternalFleursError("sample_rate must be positive")
    if max_chunk_seconds <= 0 or max_chunk_seconds > MAX_CHUNK_SECONDS:
        raise ExternalFleursError(
            f"max_chunk_seconds must be in (0, {MAX_CHUNK_SECONDS:g}]"
        )
    samples_per_chunk = int(round(sample_rate * max_chunk_seconds))
    if samples_per_chunk < 1:
        raise ExternalFleursError("Chunk duration resolves to fewer than one sample")
    if len(waveform) < 1:
        raise ExternalFleursError("Cannot transcribe an empty waveform")
    chunk_count = (len(waveform) + samples_per_chunk - 1) // samples_per_chunk
    base_size, remainder = divmod(len(waveform), chunk_count)
    chunks: list[Any] = []
    start = 0
    for index in range(chunk_count):
        chunk_size = base_size + (1 if index < remainder else 0)
        end = start + chunk_size
        chunks.append(waveform[start:end])
        start = end
    return chunks


def join_chunk_hypotheses(hypotheses: Sequence[str]) -> str:
    """Join chunk hypotheses in time order with stable whitespace."""

    return " ".join(text.strip() for text in hypotheses if text.strip())


def _prediction_row(
    manifest_row: Mapping[str, str],
    run: ExternalRun,
    hypothesis: str,
) -> dict[str, str]:
    return {
        "utt_id": manifest_row["utt_id"],
        **run.run_metadata,
        "snr": "clean",
        "noise_type": "clean",
        "ref": manifest_row["ref"],
        "hyp": hypothesis,
    }


def _partial_path(prediction_path: Path) -> Path:
    return prediction_path.with_name(f".{prediction_path.stem}.partial.csv")


def _validate_prediction_prefix(
    rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, str]],
    run: ExternalRun,
    *,
    source: str | Path,
    require_complete: bool,
) -> list[dict[str, str]]:
    validated = validate_prediction_rows(rows, source=source)
    if len(validated) > len(manifest_rows):
        raise ExternalFleursError(
            f"{source}: has {len(validated)} rows but selected manifest has {len(manifest_rows)}"
        )
    if require_complete and len(validated) != len(manifest_rows):
        raise ExternalFleursError(
            f"{source}: has {len(validated)} rows, expected {len(manifest_rows)}"
        )
    expected_metadata = run.run_metadata
    for index, row in enumerate(validated):
        manifest_row = manifest_rows[index]
        expected = {
            **expected_metadata,
            "utt_id": manifest_row["utt_id"],
            "snr": "clean",
            "noise_type": "clean",
            "ref": manifest_row["ref"],
        }
        conflicts = [name for name, value in expected.items() if row[name] != value]
        if conflicts:
            raise ExternalFleursError(
                f"{source}: row {index + 2} is not the expected manifest prefix; "
                f"conflicts={conflicts}"
            )
    return validated


def _default_audio_loader(path: str, sample_rate: int) -> Any:
    from src.vitonesr.noise import read_audio

    return read_audio(path, sr=sample_rate)


def _load_processor_with_fallback(
    processor_class: Any,
    run: ExternalRun,
) -> Any:
    """Load a checkpoint processor, falling back to the unchanged base tokenizer.

    Some checkpoints were saved by an older Transformers version whose
    ``tokenizer_config.json`` encoded ``extra_special_tokens`` as a list.
    Transformers 4.57 expects a mapping and raises while loading that local
    copy.  LoRA changes model weights only, so the base PhoWhisper processor is
    the canonical, safe fallback for every run in this external suite.
    """

    kwargs = {"language": run.language, "task": run.task}
    local_processor = run.checkpoint / "processor"
    local_error: Exception | None = None
    if local_processor.exists():
        try:
            return processor_class.from_pretrained(str(local_processor), **kwargs)
        except Exception as error:
            local_error = error
            warnings.warn(
                f"Checkpoint processor at {local_processor} is incompatible "
                f"({type(error).__name__}: {error}); falling back to "
                f"{run.model_name_or_path}.",
                RuntimeWarning,
                stacklevel=2,
            )

    try:
        return processor_class.from_pretrained(run.model_name_or_path, **kwargs)
    except Exception as error:
        if local_error is not None:
            raise RuntimeError(
                "Could not load either the checkpoint-local processor or the "
                f"base processor {run.model_name_or_path!r}"
            ) from error
        raise


class WhisperAdapterTranscriber:
    """Lazy PEFT/Transformers wrapper so analysis-only use needs no model deps."""

    def __init__(self, run: ExternalRun, device_arg: str, max_new_tokens: int) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
                WhisperTokenizer,
            )
        except ImportError as error:
            raise RuntimeError(
                "FLEURS inference requires torch, transformers, and peft"
            ) from error

        checkpoint = run.checkpoint
        adapter = checkpoint / "adapter"
        if not checkpoint.exists() or not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Missing completed PEFT checkpoint for lambda={run.lambda_value}: {checkpoint}"
            )
        if max_new_tokens < 1:
            raise ExternalFleursError("max_new_tokens must be at least 1")

        if device_arg == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        self.processor = _load_processor_with_fallback(WhisperProcessor, run)
        base_model = WhisperForConditionalGeneration.from_pretrained(run.model_name_or_path)
        base_model.config.use_cache = True
        self.model = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=False)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.language = run.language
        self.task = run.task
        self.model_name_or_path = run.model_name_or_path
        self.slow_tokenizer_class = WhisperTokenizer
        self.slow_tokenizer: Any | None = None

    def _decode_generated(self, generated: Any) -> str:
        decoded = str(
            self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        )
        if "\ufffd" not in decoded:
            return decoded

        if self.slow_tokenizer is None:
            self.slow_tokenizer = self.slow_tokenizer_class.from_pretrained(
                self.model_name_or_path,
                language=self.language,
                task=self.task,
                errors="strict",
            )
            if self.slow_tokenizer.get_vocab() != self.processor.tokenizer.get_vocab():
                raise ExternalFleursError(
                    "Cannot recover invalid byte-BPE output with a different tokenizer vocab"
                )

        sequence = generated[0]
        token_ids = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        try:
            strict_decoded = self.slow_tokenizer.decode(
                token_ids, skip_special_tokens=True
            )
        except UnicodeDecodeError:
            self.slow_tokenizer.errors = "ignore"
            try:
                recovered = str(
                    self.slow_tokenizer.decode(token_ids, skip_special_tokens=True)
                )
            finally:
                self.slow_tokenizer.errors = "strict"
            if "\ufffd" in recovered:
                raise ExternalFleursError(
                    "Slow-tokenizer byte recovery still contains U+FFFD"
                )
            warnings.warn(
                "Recovered invalid byte-BPE output with the same tokenizer vocab "
                "and errors='ignore'",
                UnicodeWarning,
                stacklevel=2,
            )
            return recovered
        return str(strict_decoded)

    def transcribe_chunk(self, waveform: Any) -> str:
        feature_batch = self.processor.feature_extractor(
            [waveform],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feature_batch.input_features.to(
            device=self.device, dtype=self.dtype
        )
        attention_mask = feature_batch.attention_mask.to(device=self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "language": self.language,
            "task": self.task,
        }
        with self.torch.inference_mode():
            try:
                generated = self.model.generate(
                    input_features, attention_mask=attention_mask, **kwargs
                )
            except TypeError:
                generated = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                        language=self.language, task=self.task
                    ),
                )
        return self._decode_generated(generated)

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _default_transcriber_factory(
    run: ExternalRun, device_arg: str, max_new_tokens: int
) -> ChunkTranscriber:
    return WhisperAdapterTranscriber(run, device_arg, max_new_tokens)


def run_external_prediction(
    run: ExternalRun,
    manifest_rows: Sequence[Mapping[str, str]],
    prediction_path: str | Path,
    *,
    device_arg: str = "auto",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    checkpoint_every: int = 10,
    resume: bool = False,
    overwrite: bool = False,
    transcriber_factory: TranscriberFactory = _default_transcriber_factory,
    audio_loader: AudioLoader = _default_audio_loader,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Run or resume one fixed checkpoint and atomically publish its CSV."""

    if resume and overwrite:
        raise ExternalFleursError("--resume and --overwrite are mutually exclusive")
    if checkpoint_every < 1:
        raise ExternalFleursError("checkpoint_every must be at least 1")
    if sample_rate != SAMPLE_RATE:
        raise ExternalFleursError(
            f"FLEURS inference requires materialized {SAMPLE_RATE} Hz audio"
        )
    if not manifest_rows:
        raise ExternalFleursError("Cannot run inference for an empty manifest")

    output = Path(prediction_path)
    partial = _partial_path(output)
    if output.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"Prediction file already exists: {output}. Use --resume or --overwrite explicitly."
        )
    if partial.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"Partial prediction exists: {partial}. Use --resume or --overwrite explicitly."
        )

    if output.exists() and resume:
        rows = load_prediction_csv(output)
        _validate_prediction_prefix(
            rows, manifest_rows, run, source=output, require_complete=True
        )
        return output

    prediction_rows: list[dict[str, str]] = []
    if partial.exists() and resume:
        partial_rows = load_prediction_csv(partial)
        prediction_rows = _validate_prediction_prefix(
            partial_rows,
            manifest_rows,
            run,
            source=partial,
            require_complete=False,
        )

    start = len(prediction_rows)
    transcriber: ChunkTranscriber | None = None
    try:
        if start < len(manifest_rows):
            transcriber = transcriber_factory(run, device_arg, max_new_tokens)
        for index in range(start, len(manifest_rows)):
            manifest_row = manifest_rows[index]
            waveform = audio_loader(manifest_row["audio_path"], sample_rate)
            chunks = split_waveform(waveform, sample_rate=sample_rate)
            hypothesis = join_chunk_hypotheses(
                [transcriber.transcribe_chunk(chunk) for chunk in chunks]  # type: ignore[union-attr]
            )
            if "\ufffd" in hypothesis:
                warnings.warn(
                    "Raw tokenizer output contains U+FFFD for "
                    f"utt_id={manifest_row['utt_id']}; preserving it in the prediction",
                    UnicodeWarning,
                    stacklevel=2,
                )
            prediction_rows.append(_prediction_row(manifest_row, run, hypothesis))
            if len(prediction_rows) % checkpoint_every == 0:
                atomic_write_csv(partial, prediction_rows, CANONICAL_PREDICTION_COLUMNS)

        validated = _validate_prediction_prefix(
            prediction_rows,
            manifest_rows,
            run,
            source=output,
            require_complete=True,
        )
        atomic_write_csv(output, validated, CANONICAL_PREDICTION_COLUMNS)
        if partial.exists():
            partial.unlink()
        return output
    except Exception:
        if prediction_rows:
            atomic_write_csv(partial, prediction_rows, CANONICAL_PREDICTION_COLUMNS)
        raise
    finally:
        if transcriber is not None:
            transcriber.close()


def build_external_results(
    artifacts: Sequence[tuple[ExternalRun, str | Path]],
) -> list[dict[str, object]]:
    """Validate three paired prediction files and calculate aligned_v1 metrics."""

    if len(artifacts) != len(RUN_TEMPLATES):
        raise ExternalFleursError(
            f"External FLEURS results require exactly {len(RUN_TEMPLATES)} runs"
        )
    observed = [(run.train_type, run.lambda_value) for run, _ in artifacts]
    expected = [(item.train_type, item.lambda_value) for item in RUN_TEMPLATES]
    if observed != expected:
        raise ExternalFleursError(
            f"External runs must be in the fixed order {expected}, found {observed}"
        )

    output: list[dict[str, object]] = []
    paired_identity: list[tuple[str, str]] | None = None
    for run, path in artifacts:
        rows = load_prediction_csv(path)
        expected_metadata = run.run_metadata
        for row_number, row in enumerate(rows, start=2):
            conflicts = [
                field for field, value in expected_metadata.items() if row[field] != value
            ]
            if row["snr"] != "clean" or row["noise_type"] != "clean":
                conflicts.extend(["snr/noise_type"])
            if conflicts:
                raise ExternalFleursError(
                    f"{path}: row {row_number} conflicts with external run: {conflicts}"
                )
        identity = [(row["utt_id"], row["ref"]) for row in rows]
        if paired_identity is None:
            paired_identity = identity
        elif identity != paired_identity:
            raise ExternalFleursError(
                f"{path}: utterance order/reference does not match the paired FLEURS run"
            )

        metrics = compute_aligned_metric_result(
            [row["ref"] for row in rows], [row["hyp"] for row in rows]
        )
        output.append(
            {
                **expected_metadata,
                "n": len(rows),
                "wer": metrics.wer,
                "cer": metrics.cer,
                "ter": metrics.ter,
                "der": metrics.der,
                "fcer": metrics.fcer,
                "swdr": metrics.swdr,
                "metric_version": METRIC_VERSION,
            }
        )
    return output


def _existing_result_matches(path: Path, rows: Sequence[Mapping[str, object]]) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
            return False
        existing = list(reader)
    if len(existing) != len(rows):
        return False
    for current, expected in zip(existing, rows):
        for column in RESULT_COLUMNS:
            if current[column] != str(expected[column]):
                return False
    return True


def run_external_suite(
    manifest: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    results_path: str | Path | None = None,
    expected_rows: int | None = DEFAULT_EXPECTED_ROWS,
    limit: int | None = None,
    device_arg: str = "auto",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    checkpoint_every: int = 10,
    resume: bool = False,
    overwrite: bool = False,
    transcriber_factory: TranscriberFactory = _default_transcriber_factory,
    audio_loader: AudioLoader = _default_audio_loader,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[list[Path], Path]:
    """Run all three checkpoints and write the three-row external result table."""

    if limit is not None and limit < 1:
        raise ExternalFleursError("limit must be at least 1")
    all_rows = load_fleurs_manifest(manifest, expected_rows=expected_rows)
    manifest_rows = all_rows[:limit] if limit is not None else all_rows
    output_root = Path(output_dir)
    predictions_dir = output_root / "predictions"
    result_output = Path(results_path) if results_path else output_root / "external_fleurs_results.csv"
    if result_output.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"External result already exists: {result_output}. Use --resume or --overwrite explicitly."
        )

    runs = build_external_runs(config_dir=config_dir, checkpoint_root=checkpoint_root)
    if not (resume or overwrite):
        for run in runs:
            prediction = predictions_dir / run.prediction_name
            partial = _partial_path(prediction)
            if prediction.exists():
                raise FileExistsError(
                    f"Prediction file already exists: {prediction}. "
                    "Use --resume or --overwrite explicitly."
                )
            if partial.exists():
                raise FileExistsError(
                    f"Partial prediction exists: {partial}. "
                    "Use --resume or --overwrite explicitly."
                )
    artifacts: list[tuple[ExternalRun, Path]] = []
    for run in runs:
        prediction_path = run_external_prediction(
            run,
            manifest_rows,
            predictions_dir / run.prediction_name,
            device_arg=device_arg,
            max_new_tokens=max_new_tokens,
            checkpoint_every=checkpoint_every,
            resume=resume,
            overwrite=overwrite,
            transcriber_factory=transcriber_factory,
            audio_loader=audio_loader,
            sample_rate=sample_rate,
        )
        artifacts.append((run, prediction_path))

    result_rows = build_external_results(artifacts)
    if result_output.exists() and resume:
        if not _existing_result_matches(result_output, result_rows):
            raise ExternalFleursError(
                f"Existing result does not match resumed predictions: {result_output}"
            )
    else:
        atomic_write_csv(result_output, result_rows, RESULT_COLUMNS)
    return [path for _, path in artifacts], result_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ordinary LoRA, tone-aware lambda=0.05, and tone-aware "
            "lambda=0.1 on the clean Vietnamese FLEURS test split."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
        help="Expected full manifest rows; use 0 only for synthetic/local tests.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Deterministic first-N smoke test."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Maximum decoded tokens per chunk. The default leaves eight "
            "positions for Whisper decoder control tokens within its "
            "448-position target limit."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
        raise ExternalFleursError(
            "A smoke --limit requires a separate --output-dir so partial results "
            "cannot replace the official FLEURS run"
        )
    expected_rows = None if args.expected_rows == 0 else args.expected_rows
    predictions, result = run_external_suite(
        args.manifest,
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        checkpoint_root=args.checkpoint_root,
        results_path=args.results_path,
        expected_rows=expected_rows,
        limit=args.limit,
        device_arg=args.device,
        max_new_tokens=args.max_new_tokens,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    for prediction in predictions:
        print(f"prediction={prediction}")
    print(f"results={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
