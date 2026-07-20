"""Lightweight, non-formal inference backend for the interactive demo.

The formal paper-v2 runners intentionally accept only locked manifests.  This
module provides a separate boundary for microphone/uploaded audio while
reusing their audio preprocessing, greedy decoding, checkpoints and
``aligned_v1`` analysis.  It never writes paper-v2 artifacts.
"""

from __future__ import annotations

import json
import random
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from .analysis import (
    METRIC_VERSION,
    analyze_error_events,
    compute_aligned_metric_result,
)
from .noise import fit_noise, mix_at_snr, read_audio


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = "vinai/PhoWhisper-base"
DEFAULT_BASE_REVISION = "7ebdb9e88f5cc5271fb88f4d642c82ff9388650e"
DEFAULT_NOISE_MANIFEST = "data/manifests/noise/paper_v2/musan_test.jsonl"
DEFAULT_ROLE_SPECS: tuple[tuple[str, str, str, str, float], ...] = (
    (
        "ordinary",
        "Ordinary LoRA",
        "outputs/paper_v2/checkpoints/ckpt_lora_ordinary_lambda0/best",
        "ordinary_lora",
        0.0,
    ),
    (
        "tone_005",
        "Tone-aware LoRA (lambda=0.05)",
        "outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_005/best",
        "tone_aware_lora",
        0.05,
    ),
    (
        "tone_01",
        "Tone-aware LoRA (lambda=0.1)",
        "outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_01/best",
        "tone_aware_lora",
        0.1,
    ),
)
_NOISE_ALIASES = {
    "fan": "noise",
    "traffic": "noise",
    "cafe": "speech",
    "babble": "speech",
}
_SUPPORTED_NOISE_TYPES = frozenset(
    {"fan", "traffic", "cafe", "babble", "music", "noise", "speech"}
)


class DemoInferenceError(RuntimeError):
    """Raised for an invalid demo input or unavailable local model asset."""


@dataclass(frozen=True, slots=True)
class DemoRoleConfig:
    """One adapter exposed in the comparison UI."""

    role: str
    label: str
    checkpoint: Path
    train_type: str
    lambda_value: float


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Resolved local configuration for the interactive demo."""

    sample_rate: int = 16_000
    max_audio_seconds: float = 15.0
    base_model: str = DEFAULT_BASE_MODEL
    base_revision: str = DEFAULT_BASE_REVISION
    max_new_tokens: int = 128
    device: str = "auto"
    precision: str = "fp16"
    local_files_only: bool = True
    seed: int = 42
    noise_manifest: Path | None = field(
        default_factory=lambda: REPO_ROOT / DEFAULT_NOISE_MANIFEST
    )
    roles: tuple[DemoRoleConfig, ...] = field(
        default_factory=lambda: tuple(
            DemoRoleConfig(
                role=role,
                label=label,
                checkpoint=REPO_ROOT / checkpoint,
                train_type=train_type,
                lambda_value=lambda_value,
            )
            for role, label, checkpoint, train_type, lambda_value in DEFAULT_ROLE_SPECS
        )
    )

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.device.casefold() not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.precision.casefold() not in {"fp16", "fp32"}:
            raise ValueError("precision must be fp16 or fp32")
        role_names = [role.role for role in self.roles]
        if not role_names or len(role_names) != len(set(role_names)):
            raise ValueError("roles must be non-empty and uniquely named")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DemoConfig":
        """Load a demo YAML, resolving repository-relative asset paths.

        The parser accepts either a compact ``roles`` list or a
        ``model.checkpoints`` mapping.  Omitted values use the locked paper-v2
        defaults, making the UI config intentionally small.
        """

        source = Path(path)
        if not source.is_absolute():
            source = REPO_ROOT / source
        if not source.is_file():
            raise FileNotFoundError(f"Demo config does not exist: {source}")
        value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(value, Mapping):
            raise ValueError("Demo config must contain a YAML object")

        audio = _mapping(value.get("audio"), "audio")
        model = _mapping(value.get("model"), "model")
        decoding = _mapping(value.get("decoding"), "decoding")
        runtime = _mapping(value.get("runtime"), "runtime")
        noise = _mapping(value.get("noise"), "noise")

        roles_value = value.get("roles")
        if roles_value is None:
            roles_value = model.get("roles") or model.get("checkpoints")
        roles = _parse_roles(roles_value)

        raw_noise_manifest = noise.get(
            "manifest", value.get("noise_manifest", DEFAULT_NOISE_MANIFEST)
        )
        noise_manifest = (
            None
            if raw_noise_manifest in {None, ""}
            else _repo_path(raw_noise_manifest)
        )
        return cls(
            sample_rate=int(audio.get("sample_rate", value.get("sample_rate", 16_000))),
            max_audio_seconds=float(
                audio.get(
                    "max_audio_seconds", value.get("max_audio_seconds", 15.0)
                )
            ),
            base_model=str(
                model.get(
                    "name_or_path",
                    model.get("base_model", value.get("base_model", DEFAULT_BASE_MODEL)),
                )
            ),
            base_revision=str(
                model.get(
                    "revision",
                    value.get("base_revision", DEFAULT_BASE_REVISION),
                )
            ),
            max_new_tokens=int(
                decoding.get(
                    "max_new_tokens", value.get("max_new_tokens", 128)
                )
            ),
            device=str(runtime.get("device", value.get("device", "auto"))),
            precision=str(
                runtime.get("precision", value.get("precision", "fp16"))
            ),
            local_files_only=bool(
                runtime.get(
                    "local_files_only", value.get("local_files_only", True)
                )
            ),
            seed=int(value.get("seed", runtime.get("seed", 42))),
            noise_manifest=noise_manifest,
            roles=roles,
        )


@dataclass(slots=True)
class PreparedAudio:
    """Audio ready for Whisper plus auditable preprocessing metadata."""

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    source: str
    truncated: bool
    noise: dict[str, Any]

    @property
    def waveform(self) -> np.ndarray:
        """Compatibility alias used by the UI and tests."""

        return self.audio

    @property
    def metadata(self) -> dict[str, Any]:
        """Return preprocessing fields in one UI-friendly mapping."""

        return {
            "source": self.source,
            "truncated": self.truncated,
            "duration_seconds": self.duration_seconds,
            "noise": dict(self.noise),
        }

    def __iter__(self):
        """Allow ``sample_rate, audio = prepared`` for UI convenience."""

        yield self.sample_rate
        yield self.audio


@dataclass(slots=True)
class DemoResult:
    """Complete result returned to the UI without writing to disk."""

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    audio_metadata: dict[str, Any]
    rows: list[dict[str, Any]]

    def to_dict(self, *, include_audio: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "audio_metadata": dict(self.audio_metadata),
            "rows": self.rows,
        }
        if include_audio:
            result["audio"] = self.audio
        return result


# Fake predictors used by unit tests receive (waveform, sample_rate, roles).
Predictor = Callable[[np.ndarray, int, Sequence[str]], Any]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Demo config {label} must be an object")
    return value


def _repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _default_roles() -> tuple[DemoRoleConfig, ...]:
    return tuple(
        DemoRoleConfig(
            role=role,
            label=label,
            checkpoint=REPO_ROOT / checkpoint,
            train_type=train_type,
            lambda_value=lambda_value,
        )
        for role, label, checkpoint, train_type, lambda_value in DEFAULT_ROLE_SPECS
    )


def _parse_roles(value: object) -> tuple[DemoRoleConfig, ...]:
    if value is None:
        return _default_roles()
    items: list[tuple[str, Mapping[str, Any] | str]] = []
    if isinstance(value, Mapping):
        items = [(str(name), spec) for name, spec in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                raise ValueError(f"roles[{index}] must be an object")
            name = str(raw.get("role") or raw.get("id") or "").strip()
            if not name:
                raise ValueError(f"roles[{index}] requires role or id")
            items.append((name, raw))
    else:
        raise ValueError("roles must be a list or object")

    defaults = {spec[0]: spec for spec in DEFAULT_ROLE_SPECS}
    roles: list[DemoRoleConfig] = []
    for name, raw in items:
        if isinstance(raw, str):
            spec: Mapping[str, Any] = {"checkpoint": raw}
        elif isinstance(raw, Mapping):
            spec = raw
        else:
            raise ValueError(f"Role {name!r} must be a path or object")
        default = defaults.get(name)
        label = str(spec.get("label", default[1] if default else name))
        checkpoint = spec.get("checkpoint", spec.get("path"))
        if checkpoint is None and default is not None:
            checkpoint = default[2]
        if checkpoint is None:
            raise ValueError(f"Role {name!r} requires checkpoint")
        train_type = str(
            spec.get("train_type", default[3] if default else "tone_aware_lora")
        )
        lambda_value = float(spec.get("lambda", default[4] if default else 0.0))
        roles.append(
            DemoRoleConfig(
                role=name,
                label=label,
                checkpoint=_repo_path(checkpoint),
                train_type=train_type,
                lambda_value=lambda_value,
            )
        )
    return tuple(roles)


def _as_float_mono(waveform: np.ndarray) -> np.ndarray:
    audio = np.asarray(waveform)
    if audio.ndim == 0 or audio.ndim > 2:
        raise ValueError("Audio must be a one- or two-dimensional array")
    if audio.size == 0:
        raise ValueError("Audio is empty")
    if audio.ndim == 2:
        # SoundFile/Gradio use frames x channels; accept common channels-first
        # arrays too so notebook users do not need a separate transpose.
        channel_axis = 0 if audio.shape[0] <= 8 < audio.shape[1] else 1
        audio = audio.astype(np.float64).mean(axis=channel_axis)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        divisor = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / divisor
    else:
        audio = audio.astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return np.ascontiguousarray(audio, dtype=np.float32)


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Demo noise manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DemoInferenceError(
                    f"Invalid demo noise manifest JSON at line {line_number}: {path}"
                ) from error
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise DemoInferenceError(f"Demo noise manifest is empty: {path}")
    return rows


class DemoEngine:
    """Lazy, reusable three-adapter Whisper inference engine."""

    def __init__(self, config: DemoConfig, predictor: Predictor | None = None):
        self.config = config
        self._predictor = predictor
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None
        self._noise_rows: list[dict[str, Any]] | None = None

    @property
    def available_roles(self) -> tuple[str, ...]:
        return tuple(role.role for role in self.config.roles)

    def close(self) -> None:
        """Release model references and cached CUDA memory."""

        self._processor = None
        self._model = None
        torch = self._torch
        self._torch = None
        self._device = None
        self._dtype = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_audio(
        self,
        source: str | Path | np.ndarray | tuple[int, np.ndarray],
        sample_rate: int | None = None,
        *,
        noise_type: str = "none",
        snr: float | None = None,
        noise_path: str | Path | None = None,
        seed: int | None = None,
    ) -> PreparedAudio:
        """Downmix, resample, cap at 15 s, and optionally add controlled noise."""

        source_label: str
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if not source_path.is_file():
                raise FileNotFoundError(f"Demo audio does not exist: {source_path}")
            # Keep exact repository preprocessing for already-canonical 16 kHz
            # assets.  For browser/uploaded files at another rate, avoid
            # librosa's expensive first-call JIT by using the same deterministic
            # in-memory interpolation as microphone input.
            import soundfile as sf

            info = sf.info(str(source_path))
            if int(info.samplerate) == self.config.sample_rate:
                waveform = read_audio(str(source_path), sr=self.config.sample_rate)
            else:
                raw, input_rate = sf.read(str(source_path), always_2d=False)
                waveform = self._resample_memory_audio(raw, int(input_rate))
            source_label = str(source_path)
        else:
            if isinstance(source, tuple):
                if len(source) != 2:
                    raise ValueError("Audio tuple must be (sample_rate, waveform)")
                tuple_rate, tuple_audio = source
                if sample_rate is not None and int(sample_rate) != int(tuple_rate):
                    raise ValueError("Conflicting sample rates for audio tuple")
                sample_rate = int(tuple_rate)
                source = tuple_audio
            input_rate = self.config.sample_rate if sample_rate is None else int(sample_rate)
            waveform = self._resample_memory_audio(np.asarray(source), input_rate)
            source_label = "<memory>"

        waveform = _as_float_mono(waveform)
        maximum = int(self.config.max_audio_seconds * self.config.sample_rate)
        truncated = len(waveform) > maximum
        waveform = np.ascontiguousarray(waveform[:maximum], dtype=np.float32)
        if len(waveform) == 0:
            raise ValueError("Audio is empty after preprocessing")

        noise_metadata: dict[str, Any] = {
            "applied": False,
            "noise_type": "clean",
            "snr_db": None,
            "noise_path": None,
            "seed": None,
        }
        normalized_type = str(noise_type or "none").strip().casefold()
        if normalized_type not in {"", "none", "clean"} or noise_path is not None:
            if snr is None:
                raise ValueError("snr is required when controlled noise is enabled")
            waveform, noise_metadata = self.apply_noise(
                waveform,
                noise_type=normalized_type,
                snr=float(snr),
                noise_path=noise_path,
                seed=seed,
            )

        return PreparedAudio(
            audio=waveform,
            sample_rate=self.config.sample_rate,
            duration_seconds=len(waveform) / self.config.sample_rate,
            source=source_label,
            truncated=truncated,
            noise=noise_metadata,
        )

    def apply_noise(
        self,
        waveform: np.ndarray,
        noise_type: str,
        snr: float,
        *,
        noise_path: str | Path | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mix a deterministic noise segment with the existing SNR helpers."""

        clean = _as_float_mono(waveform)
        if not np.isfinite(float(snr)):
            raise ValueError("snr must be finite")
        requested_type = str(noise_type or "noise").strip().casefold()
        if requested_type in {"", "none", "clean"} and noise_path is None:
            raise ValueError("noise_type must be non-clean when applying noise")
        if requested_type not in _SUPPORTED_NOISE_TYPES:
            raise ValueError(
                f"Unsupported noise_type={requested_type!r}; "
                f"available={sorted(_SUPPORTED_NOISE_TYPES)}"
            )
        resolved_type = _NOISE_ALIASES.get(requested_type, requested_type)
        rng_seed = self.config.seed if seed is None else int(seed)
        rng = random.Random(rng_seed)

        if noise_path is None:
            if self.config.noise_manifest is None:
                raise DemoInferenceError(
                    "No noise_path was supplied and no noise manifest is configured"
                )
            if self._noise_rows is None:
                self._noise_rows = _load_manifest_rows(self.config.noise_manifest)
            candidates = [
                row
                for row in self._noise_rows
                if str(row.get("noise_type", "")).casefold() == resolved_type
            ]
            if not candidates:
                available = sorted(
                    {str(row.get("noise_type", "")) for row in self._noise_rows}
                )
                raise DemoInferenceError(
                    f"No {resolved_type!r} entry in demo noise manifest; "
                    f"available={available}"
                )
            row = rng.choice(candidates)
            raw_path = Path(str(row.get("audio", "")))
            chosen_path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        else:
            chosen_path = Path(noise_path)
            if not chosen_path.is_absolute():
                chosen_path = REPO_ROOT / chosen_path
        if not chosen_path.is_file():
            raise FileNotFoundError(f"Demo noise audio does not exist: {chosen_path}")

        noise = read_audio(str(chosen_path), sr=self.config.sample_rate)
        fitted = fit_noise(noise, len(clean), rng)
        mixed = mix_at_snr(clean, fitted, float(snr))
        try:
            display_path = chosen_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            display_path = str(chosen_path)
        return mixed, {
            "applied": True,
            "noise_type": requested_type,
            "resolved_noise_type": resolved_type,
            "requested_noise_type": requested_type,
            "snr_db": float(snr),
            "noise_path": display_path,
            "seed": rng_seed,
        }

    def _resample_memory_audio(
        self, waveform: np.ndarray, input_rate: int
    ) -> np.ndarray:
        """Downmix and deterministically resample an in-memory waveform."""

        audio = _as_float_mono(waveform)
        if input_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if input_rate == self.config.sample_rate:
            return audio
        # A deterministic interpolation is fast enough for a live upload and
        # avoids first-run JIT stalls. Formal benchmark audio is already 16 kHz
        # and therefore always follows the exact ``read_audio`` path above.
        output_length = max(
            1,
            int(round(len(audio) * self.config.sample_rate / input_rate)),
        )
        source_positions = np.arange(len(audio), dtype=np.float64)
        target_positions = np.linspace(
            0.0,
            max(len(audio) - 1, 0),
            output_length,
            dtype=np.float64,
        )
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    def run(
        self,
        source: str | Path | np.ndarray | tuple[int, np.ndarray],
        sample_rate: int | None = None,
        reference: str | None = None,
        *,
        noise_type: str = "none",
        snr: float | None = None,
        noise_path: str | Path | None = None,
        roles: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> DemoResult:
        """Transcribe one input with selected adapters and optionally score it."""

        prepared = self.prepare_audio(
            source,
            sample_rate,
            noise_type=noise_type,
            snr=snr,
            noise_path=noise_path,
            seed=seed,
        )
        selected = self._select_roles(roles)
        role_names = [role.role for role in selected]
        predictions = (
            self._call_injected_predictor(prepared.audio, role_names)
            if self._predictor is not None
            else self._predict(prepared.audio, selected)
        )

        score_reference = None
        if reference is not None and str(reference).strip():
            score_reference = unicodedata.normalize("NFC", str(reference))
        rows: list[dict[str, Any]] = []
        for role in selected:
            output = predictions[role.role]
            transcript = unicodedata.normalize("NFC", str(output["transcript"]))
            row: dict[str, Any] = {
                "role": role.role,
                "label": role.label,
                "train_type": role.train_type,
                "lambda": role.lambda_value,
                "transcript": transcript,
                "latency_seconds": float(output["latency_seconds"]),
                "metric_version": METRIC_VERSION if score_reference is not None else None,
                "metrics": None,
                "metrics_percent": None,
                "errors": None,
                "alignment": None,
            }
            if score_reference is not None:
                metric_result = compute_aligned_metric_result(
                    [score_reference], [transcript]
                )
                metrics = metric_result.to_dict(include_counts=True)
                scalar_names = ("wer", "cer", "ter", "der", "fcer", "swdr")
                events = analyze_error_events(score_reference, transcript)
                operations = {
                    name: sum(event.operation == name for event in events)
                    for name in ("match", "substitution", "deletion", "insertion")
                }
                row["metrics"] = metrics
                row["metrics_percent"] = {
                    name: float(metrics[name]) * 100.0 for name in scalar_names
                }
                row["errors"] = {
                    **operations,
                    "word_errors": sum(
                        operations[name]
                        for name in ("substitution", "deletion", "insertion")
                    ),
                    "tone": sum(event.tone_error for event in events),
                    "diacritic": sum(event.diacritic_error for event in events),
                    "final_consonant": sum(
                        event.final_consonant_error for event in events
                    ),
                    "short_word_deletion": sum(
                        event.short_word_deletion for event in events
                    ),
                }
                row["alignment"] = [event.to_dict() for event in events]
                # Flat percentage/error fields keep a Streamlit DataFrame and
                # CSV export simple; nested values remain the canonical API.
                row.update(row["metrics_percent"])
                row.update(
                    {
                        name: row["errors"][name]
                        for name in (
                            "substitution",
                            "deletion",
                            "insertion",
                            "tone",
                            "diacritic",
                            "final_consonant",
                            "short_word_deletion",
                        )
                    }
                )
            rows.append(row)

        return DemoResult(
            audio=prepared.audio,
            sample_rate=prepared.sample_rate,
            duration_seconds=prepared.duration_seconds,
            audio_metadata={
                "source": prepared.source,
                "truncated": prepared.truncated,
                "noise": prepared.noise,
            },
            rows=rows,
        )

    def _select_roles(
        self, requested: Sequence[str] | None
    ) -> tuple[DemoRoleConfig, ...]:
        if requested is None:
            return self.config.roles
        wanted = list(dict.fromkeys(str(role) for role in requested))
        known = {role.role: role for role in self.config.roles}
        unknown = sorted(set(wanted) - set(known))
        if unknown:
            raise ValueError(
                f"Unknown demo roles {unknown}; available={list(known)}"
            )
        if not wanted:
            raise ValueError("At least one demo role must be selected")
        return tuple(known[name] for name in wanted)

    def _call_injected_predictor(
        self, waveform: np.ndarray, roles: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        assert self._predictor is not None
        started = time.perf_counter()
        raw = self._predictor(waveform, self.config.sample_rate, roles)
        elapsed = time.perf_counter() - started
        if isinstance(raw, Mapping):
            items = dict(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != len(roles):
                raise DemoInferenceError(
                    "Injected predictor returned a sequence with the wrong length"
                )
            items = dict(zip(roles, raw))
        else:
            raise DemoInferenceError(
                "Injected predictor must return a role mapping or result sequence"
            )
        missing = sorted(set(roles) - set(items))
        if missing:
            raise DemoInferenceError(
                f"Injected predictor did not return roles: {missing}"
            )
        results: dict[str, dict[str, Any]] = {}
        for role in roles:
            item = items[role]
            if isinstance(item, Mapping):
                transcript = item.get("transcript", item.get("hyp", ""))
                latency = item.get("latency_seconds", elapsed / len(roles))
            else:
                transcript = item
                latency = elapsed / len(roles)
            results[role] = {
                "transcript": str(transcript),
                "latency_seconds": float(latency),
            }
        return results

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as error:  # pragma: no cover - environment dependent
            raise DemoInferenceError(
                "Demo inference requires torch, transformers and peft; install "
                "requirements-demo.txt"
            ) from error

        device_name = self.config.device.casefold()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise DemoInferenceError("CUDA was requested but is unavailable")
        device = torch.device(device_name)
        dtype = (
            torch.float16
            if self.config.precision.casefold() == "fp16" and device.type == "cuda"
            else torch.float32
        )

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        resolved: list[tuple[DemoRoleConfig, Path, Path]] = []
        from .phat.evaluation import resolve_checkpoint

        for role in self.config.roles:
            checkpoint, adapter = resolve_checkpoint(role.checkpoint)
            processor_path = checkpoint / "processor"
            if not processor_path.is_dir():
                raise FileNotFoundError(
                    f"Processor directory is missing for role {role.role}: {processor_path}"
                )
            resolved.append((role, adapter, processor_path))

        try:
            processor = WhisperProcessor.from_pretrained(
                str(resolved[0][2]),
                language="vi",
                task="transcribe",
                local_files_only=True,
            )
        except Exception as local_error:
            # The formal checkpoints were saved with Transformers 5.12.  Its
            # tokenizer_config cannot be parsed by the 4.57 inference host.
            # LoRA never changes the processor, so the pinned base processor is
            # the immutable equivalent used by the verified formal wrapper.
            warnings.warn(
                "Checkpoint-local processor is incompatible with this "
                "Transformers runtime; using the pinned local base processor "
                f"instead ({type(local_error).__name__}: {local_error}).",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                processor = WhisperProcessor.from_pretrained(
                    self.config.base_model,
                    revision=self.config.base_revision,
                    language="vi",
                    task="transcribe",
                    local_files_only=self.config.local_files_only,
                )
            except Exception as fallback_error:
                raise DemoInferenceError(
                    "Could not load either the checkpoint processor or the "
                    "pinned PhoWhisper processor from the local cache"
                ) from fallback_error
        try:
            base = WhisperForConditionalGeneration.from_pretrained(
                self.config.base_model,
                revision=self.config.base_revision,
                local_files_only=self.config.local_files_only,
            )
        except OSError as error:
            raise DemoInferenceError(
                "Pinned PhoWhisper-base revision is not available locally. "
                "Pull/download the model cache before starting the demo."
            ) from error
        base.config.use_cache = True
        first_role, first_adapter, _ = resolved[0]
        model = PeftModel.from_pretrained(
            base,
            str(first_adapter),
            adapter_name=first_role.role,
            is_trainable=False,
        )
        for role, adapter, _ in resolved[1:]:
            model.load_adapter(
                str(adapter), adapter_name=role.role, is_trainable=False
            )
        model.to(device=device, dtype=dtype)
        model.eval()

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device
        self._dtype = dtype

    def _predict(
        self, waveform: np.ndarray, roles: Sequence[DemoRoleConfig]
    ) -> dict[str, dict[str, Any]]:
        self._ensure_model()
        torch = self._torch
        processor = self._processor
        model = self._model
        device = self._device
        dtype = self._dtype
        assert torch is not None and processor is not None and model is not None

        features = processor.feature_extractor(
            [waveform],
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = features.input_features.to(device=device, dtype=dtype)
        attention = features.attention_mask.to(device=device)
        results: dict[str, dict[str, Any]] = {}
        with torch.inference_mode():
            for role in roles:
                model.set_adapter(role.role)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                kwargs = {
                    "max_new_tokens": self.config.max_new_tokens,
                    "language": "vi",
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                }
                try:
                    generated = model.generate(
                        inputs, attention_mask=attention, **kwargs
                    )
                except TypeError:
                    # Compatibility fallback matches the formal final evaluator.
                    generated = model.generate(
                        inputs,
                        attention_mask=attention,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(
                            language="vi", task="transcribe"
                        ),
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency = time.perf_counter() - started
                transcript = processor.batch_decode(
                    generated, skip_special_tokens=True
                )[0]
                results[role.role] = {
                    "transcript": unicodedata.normalize("NFC", transcript),
                    "latency_seconds": latency,
                }
        return results


def load_demo_engine(
    config_path: str | Path = "configs/demo.yaml",
    *,
    predictor: Predictor | None = None,
) -> DemoEngine:
    """Construct a lazy engine; model weights load on the first ``run`` call."""

    return DemoEngine(DemoConfig.from_yaml(config_path), predictor=predictor)


__all__ = [
    "DemoConfig",
    "DemoEngine",
    "DemoInferenceError",
    "DemoResult",
    "DemoRoleConfig",
    "PreparedAudio",
    "load_demo_engine",
]
