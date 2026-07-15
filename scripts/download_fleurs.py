"""Download and materialize the Vietnamese FLEURS evaluation split.

The Hugging Face ``audio.path`` value is metadata and is not guaranteed to be
a usable local file.  This module therefore decodes every example and writes a
deterministic, mono 16 kHz PCM WAV before publishing the JSONL manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


DATASET_NAME = "google/fleurs"
LANGUAGE_CONFIG = "vi_vn"
DEFAULT_SPLIT = "test"
DEFAULT_EXPECTED_COUNT = 857
TARGET_SAMPLE_RATE = 16_000
MANIFEST_FIELDS = (
    "utt_id",
    "dataset",
    "split",
    "audio_path",
    "transcript",
    "snr",
    "noise_type",
)


class FleursPreparationError(RuntimeError):
    """Raised when the downloaded data cannot satisfy the manifest contract."""


def load_fleurs_dataset(
    *, split: str, cache_dir: str | Path, revision: str
) -> Iterable[Mapping[str, Any]]:
    """Load one FLEURS split lazily so unit tests do not require network access."""

    from datasets import load_dataset

    return load_dataset(
        DATASET_NAME,
        LANGUAGE_CONFIG,
        split=split,
        cache_dir=str(cache_dir),
        revision=revision,
    )


def _source_stem(audio_path: object, *, row_number: int) -> str:
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise FleursPreparationError(
            f"row {row_number}: audio.path must be a non-empty filename"
        )
    # FLEURS currently exposes a numeric filename.  Refuse unsafe stems rather
    # than silently changing them, because the stem is the external-test ID.
    stem = Path(audio_path.replace("\\", "/")).stem.strip()
    if not stem or stem in {".", ".."}:
        raise FleursPreparationError(
            f"row {row_number}: could not derive utt_id from audio.path={audio_path!r}"
        )
    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if any(character not in safe_characters for character in stem):
        raise FleursPreparationError(
            f"row {row_number}: unsafe utt_id derived from audio.path: {stem!r}"
        )
    return stem


def _decoded_audio(row: Mapping[str, Any], *, row_number: int) -> tuple[str, np.ndarray, int]:
    audio = row.get("audio")
    if not isinstance(audio, Mapping):
        raise FleursPreparationError(f"row {row_number}: audio must be a decoded mapping")
    utt_id = _source_stem(audio.get("path"), row_number=row_number)
    if "array" not in audio or "sampling_rate" not in audio:
        raise FleursPreparationError(
            f"row {row_number}: decoded audio requires array and sampling_rate"
        )
    try:
        samples = np.asarray(audio["array"], dtype=np.float64)
        if samples.ndim == 2 and 1 in samples.shape:
            samples = samples.reshape(-1)
        sample_rate = int(audio["sampling_rate"])
    except (TypeError, ValueError, OverflowError) as error:
        raise FleursPreparationError(f"row {row_number}: invalid decoded audio") from error
    if samples.ndim != 1 or samples.size == 0:
        raise FleursPreparationError(
            f"row {row_number}: decoded audio must be a non-empty mono array"
        )
    if sample_rate <= 0:
        raise FleursPreparationError(f"row {row_number}: invalid sampling rate {sample_rate}")
    if not np.isfinite(samples).all():
        raise FleursPreparationError(f"row {row_number}: decoded audio contains NaN/Inf")
    return utt_id, samples, sample_rate


def _resample_linear(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """Deterministically resample mono audio to 16 kHz.

    FLEURS is already distributed at 16 kHz.  The interpolation branch makes
    the output contract explicit and protects against a future source change.
    """

    if source_rate == TARGET_SAMPLE_RATE:
        return samples
    output_length = max(1, int(round(samples.size * TARGET_SAMPLE_RATE / source_rate)))
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.arange(output_length, dtype=np.float64) * (
        source_rate / TARGET_SAMPLE_RATE
    )
    return np.interp(target_positions, source_positions, samples)


def _atomic_write_wav(path: Path, samples: np.ndarray, *, source_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        output = _resample_linear(samples, source_rate)
        if not np.isfinite(output).all():
            raise FleursPreparationError(f"resampling produced invalid audio: {path}")
        pcm = np.rint(np.clip(output, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(temporary), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(TARGET_SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_wav(path: Path) -> None:
    if not path.is_file():
        raise FleursPreparationError(f"audio_path does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise FleursPreparationError(f"audio_path is not a readable WAV: {path}") from error
    if channels != 1 or width != 2 or sample_rate != TARGET_SAMPLE_RATE or frames < 1:
        raise FleursPreparationError(
            f"audio_path must be non-empty mono PCM16 {TARGET_SAMPLE_RATE} Hz: {path}"
        )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                if not line.strip():
                    raise FleursPreparationError(
                        f"{path}:{line_number}: blank manifest lines are not allowed"
                    )
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise FleursPreparationError(
                        f"{path}:{line_number}: manifest row must be an object"
                    )
                rows.append(item)
    except (OSError, json.JSONDecodeError) as error:
        raise FleursPreparationError(f"could not read manifest {path}: {error}") from error
    return rows


def validate_manifest(
    path: str | Path,
    *,
    expected_count: int | None = DEFAULT_EXPECTED_COUNT,
    expected_split: str = DEFAULT_SPLIT,
) -> list[dict[str, Any]]:
    """Validate the canonical manifest and every materialized WAV."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FleursPreparationError(f"manifest does not exist: {manifest_path}")
    rows = _read_manifest(manifest_path)
    if expected_count is not None and len(rows) != expected_count:
        raise FleursPreparationError(
            f"manifest has {len(rows)} rows, expected {expected_count}"
        )
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if tuple(row) != MANIFEST_FIELDS:
            raise FleursPreparationError(
                f"row {row_number}: expected fields {MANIFEST_FIELDS}, got {tuple(row)}"
            )
        utt_id = row["utt_id"]
        transcript = row["transcript"]
        if not isinstance(utt_id, str) or not utt_id:
            raise FleursPreparationError(f"row {row_number}: utt_id must be non-empty")
        if utt_id in seen:
            raise FleursPreparationError(f"row {row_number}: duplicate utt_id {utt_id!r}")
        seen.add(utt_id)
        if row["dataset"] != "fleurs" or row["split"] != expected_split:
            raise FleursPreparationError(
                f"row {row_number}: dataset/split must be fleurs/{expected_split}"
            )
        if row["snr"] != "clean" or row["noise_type"] != "clean":
            raise FleursPreparationError(
                f"row {row_number}: FLEURS external data must be clean/clean"
            )
        if not isinstance(transcript, str) or not transcript.strip():
            raise FleursPreparationError(f"row {row_number}: transcript must be non-empty")
        audio_path = Path(str(row["audio_path"]))
        if audio_path.stem != utt_id:
            raise FleursPreparationError(
                f"row {row_number}: utt_id does not match audio filename stem"
            )
        _validate_wav(audio_path)
    return rows


def _atomic_write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as manifest_file:
            for row in rows:
                manifest_file.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _output_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FleursPreparationError(f"output is locked by another process: {path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def prepare_fleurs(
    *,
    out_dir: str | Path = "data/manifests/fleurs",
    cache_dir: str | Path = "data/raw/hf_cache",
    split: str = DEFAULT_SPLIT,
    expected_count: int | None = DEFAULT_EXPECTED_COUNT,
    revision: str = "main",
    resume: bool = False,
    overwrite: bool = False,
    dataset_loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> Path:
    """Materialize one FLEURS split and return its validated manifest path."""

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if expected_count is not None and expected_count < 1:
        raise ValueError("expected_count must be at least 1 or None")
    if not split or any(character in split for character in "/\\"):
        raise ValueError("split must be a simple non-empty name")

    output_root = Path(out_dir)
    manifest_path = output_root / f"{split}.jsonl"
    audio_dir = output_root / "audio" / split
    lock_path = output_root / f".{split}.lock"

    with _output_lock(lock_path):
        if manifest_path.exists():
            if resume:
                validate_manifest(
                    manifest_path,
                    expected_count=expected_count,
                    expected_split=split,
                )
                return manifest_path
            if not overwrite:
                raise FileExistsError(
                    f"manifest already exists: {manifest_path}; use --resume or --overwrite"
                )

        loader = dataset_loader or load_fleurs_dataset
        dataset = loader(split=split, cache_dir=cache_dir, revision=revision)
        if expected_count is not None:
            try:
                source_count = len(dataset)  # type: ignore[arg-type]
            except TypeError:
                source_count = None
            if source_count is not None and source_count != expected_count:
                raise FleursPreparationError(
                    f"source split has {source_count} rows, expected {expected_count}"
                )

        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for row_number, source_row in enumerate(dataset, start=1):
            if not isinstance(source_row, Mapping):
                raise FleursPreparationError(f"row {row_number}: dataset row must be a mapping")
            utt_id, samples, sample_rate = _decoded_audio(source_row, row_number=row_number)
            if utt_id in seen:
                raise FleursPreparationError(
                    f"row {row_number}: duplicate audio filename stem/utt_id {utt_id!r}"
                )
            seen.add(utt_id)
            transcript_value = source_row.get("transcription")
            if not isinstance(transcript_value, str) or not transcript_value.strip():
                raise FleursPreparationError(
                    f"row {row_number}: transcription must be a non-empty string"
                )
            transcript = unicodedata.normalize("NFC", transcript_value.strip())
            audio_path = audio_dir / f"{utt_id}.wav"
            if audio_path.exists() and resume:
                _validate_wav(audio_path)
            elif audio_path.exists() and not overwrite:
                raise FileExistsError(
                    f"audio already exists: {audio_path}; use --resume or --overwrite"
                )
            else:
                _atomic_write_wav(audio_path, samples, source_rate=sample_rate)
            rows.append(
                {
                    "utt_id": utt_id,
                    "dataset": "fleurs",
                    "split": split,
                    "audio_path": audio_path.resolve().as_posix(),
                    "transcript": transcript,
                    "snr": "clean",
                    "noise_type": "clean",
                }
            )

        if expected_count is not None and len(rows) != expected_count:
            raise FleursPreparationError(
                f"source split produced {len(rows)} rows, expected {expected_count}"
            )
        if not rows:
            raise FleursPreparationError("source split is empty")

        # Validate the complete staged contract before publishing it.  Audio
        # files intentionally remain after interruption so --resume can reuse
        # them; the manifest itself only appears after full validation.
        staged = manifest_path.with_name(f".{manifest_path.name}.staged")
        try:
            _atomic_write_manifest(staged, rows)
            validate_manifest(
                staged,
                expected_count=expected_count,
                expected_split=split,
            )
            os.replace(staged, manifest_path)
        finally:
            staged.unlink(missing_ok=True)
        return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the Vietnamese FLEURS external-test manifest."
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default="data/manifests/fleurs",
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        dest="cache_dir",
        default="data/raw/hf_cache",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--revision", default="main")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_fleurs(
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            split=args.split,
            expected_count=args.expected_count,
            revision=args.revision,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        rows = validate_manifest(
            manifest,
            expected_count=args.expected_count,
            expected_split=args.split,
        )
    except (FleursPreparationError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} rows to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
