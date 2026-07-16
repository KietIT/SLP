"""Download and materialize the Vietnamese FLEURS evaluation split.

The Hugging Face ``audio.path`` value is metadata and is not guaranteed to be
a usable local file.  This module therefore decodes every example and writes a
deterministic, mono 16 kHz PCM WAV before publishing the JSONL manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "google/fleurs"
LANGUAGE_CONFIG = "vi_vn"
DEFAULT_SPLIT = "test"
DEFAULT_EXPECTED_COUNT = 857
TARGET_SAMPLE_RATE = 16_000
DEFAULT_LICENSE_ID = "CC-BY-4.0"
DEFAULT_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEFAULT_OUTPUT_DIR = Path("data/manifests/fleurs/paper_v2")
DEFAULT_CACHE_DIR = Path("data/raw/hf_cache")
DEFAULT_PREPARATION_LOCK = Path(
    "outputs/paper_v2/protocol/fleurs_test_lock.json"
)
DEFAULT_PREPARATION_AUDIT = Path(
    "outputs/paper_v2/protocol/fleurs_test_audit.csv"
)
PREPARATION_LOCK_VERSION = "paper_v2_fleurs_preparation_v1"
PREPARATION_CONTRACT_VERSION = "paper_v2_fleurs_pcm16_v1"
AUDIT_FIELDS = (
    "utt_id",
    "audio_path",
    "audio_sha256",
    "num_bytes",
    "sample_rate",
    "channels",
    "sample_width_bytes",
    "num_frames",
    "duration_seconds",
)
MANIFEST_FIELDS = (
    "utt_id",
    "dataset",
    "split",
    "audio_path",
    "audio_sha256",
    "transcript",
    "snr",
    "noise_type",
)
LEGACY_MANIFEST_FIELDS = tuple(
    field for field in MANIFEST_FIELDS if field != "audio_sha256"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FleursPreparationError(RuntimeError):
    """Raised when the downloaded data cannot satisfy the manifest contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value).strip().casefold()))


def _is_immutable_revision(value: object) -> bool:
    text = str(value).strip().casefold()
    return len(text) in {40, 64} and all(character in "0123456789abcdef" for character in text)


def _repo_relative(path: Path, *, repository_root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise FleursPreparationError(
            f"formal {label} must be inside the repository root: {path}"
        ) from error


def _portable_reference(value: object, *, label: str) -> Path:
    raw = str(value).strip()
    candidate = Path(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        raise FleursPreparationError(
            f"formal {label} must be a non-empty repository-relative path"
        )
    return candidate


def _resolve_portable(
    value: object,
    *,
    repository_root: Path,
    label: str,
) -> Path:
    relative = _portable_reference(value, label=label)
    resolved = (repository_root / relative).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise FleursPreparationError(
            f"formal {label} resolves outside the repository root"
        ) from error
    return resolved


def _wav_metadata(path: Path) -> dict[str, int | str]:
    _validate_wav(path)
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
    return {
        "audio_sha256": sha256_file(path),
        "num_bytes": path.stat().st_size,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": width,
        "num_frames": frames,
        "duration_seconds": format(frames / sample_rate, ".9f"),
    }


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
    formal: bool = True,
    repository_root: str | Path = ROOT,
    verify_audio_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Validate the canonical manifest and every materialized WAV."""

    manifest_path = Path(path)
    repo_root = Path(repository_root)
    if not manifest_path.is_file():
        raise FleursPreparationError(f"manifest does not exist: {manifest_path}")
    rows = _read_manifest(manifest_path)
    if expected_count is not None and len(rows) != expected_count:
        raise FleursPreparationError(
            f"manifest has {len(rows)} rows, expected {expected_count}"
        )
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        allowed_fields = (MANIFEST_FIELDS,) if formal else (
            MANIFEST_FIELDS,
            LEGACY_MANIFEST_FIELDS,
        )
        if tuple(row) not in allowed_fields:
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
        audio_value = row["audio_path"]
        if formal:
            audio_path = _resolve_portable(
                audio_value,
                repository_root=repo_root,
                label=f"manifest row {row_number} audio_path",
            )
        else:
            audio_path = Path(str(audio_value))
            if not audio_path.is_absolute():
                candidate = manifest_path.parent / audio_path
                audio_path = candidate if candidate.exists() else audio_path
        if audio_path.stem != utt_id:
            raise FleursPreparationError(
                f"row {row_number}: utt_id does not match audio filename stem"
            )
        _validate_wav(audio_path)
        expected_audio_hash = str(row.get("audio_sha256", "")).strip().casefold()
        if not formal and not expected_audio_hash:
            continue
        if not _is_sha256(expected_audio_hash):
            raise FleursPreparationError(
                f"row {row_number}: audio_sha256 must be a concrete SHA-256"
            )
        if verify_audio_hashes and sha256_file(audio_path) != expected_audio_hash:
            raise FleursPreparationError(
                f"row {row_number}: audio SHA-256 mismatch/tamper: {audio_path}"
            )
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


def _atomic_write_new_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _audit_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=AUDIT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_audit(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != AUDIT_FIELDS:
                raise FleursPreparationError(
                    f"FLEURS audit schema mismatch: {path}"
                )
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise FleursPreparationError(f"could not read FLEURS audit {path}") from error


def _audio_inventory_from_audit(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str | int]]:
    inventory: list[dict[str, str | int]] = []
    for index, row in enumerate(rows, start=2):
        if not _is_sha256(row.get("audio_sha256")):
            raise FleursPreparationError(
                f"FLEURS audit row {index} has invalid audio_sha256"
            )
        try:
            num_bytes = int(row["num_bytes"])
            sample_rate = int(row["sample_rate"])
            channels = int(row["channels"])
            sample_width = int(row["sample_width_bytes"])
            frames = int(row["num_frames"])
            duration = float(row["duration_seconds"])
        except (KeyError, TypeError, ValueError) as error:
            raise FleursPreparationError(
                f"FLEURS audit row {index} has invalid numeric metadata"
            ) from error
        if (
            num_bytes < 1
            or sample_rate != TARGET_SAMPLE_RATE
            or channels != 1
            or sample_width != 2
            or frames < 1
            or not np.isfinite(duration)
            or abs(duration - frames / sample_rate) > 0.5e-9
        ):
            raise FleursPreparationError(
                f"FLEURS audit row {index} violates the PCM16 contract"
            )
        _portable_reference(row.get("audio_path"), label=f"audit row {index} audio_path")
        inventory.append(
            {
                "utt_id": str(row.get("utt_id", "")),
                "audio_path": str(row["audio_path"]),
                "audio_sha256": str(row["audio_sha256"]).casefold(),
                "num_bytes": num_bytes,
                "num_frames": frames,
            }
        )
    return inventory


def verify_fleurs_preparation_lock(
    path: str | Path = DEFAULT_PREPARATION_LOCK,
    *,
    repository_root: str | Path = ROOT,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
    verify_artifacts: bool = True,
    verify_audio: bool = True,
) -> dict[str, Any]:
    """Verify the immutable FLEURS preparation identity and optional bytes.

    With ``verify_artifacts=False`` this reads only lock metadata.  Formal
    inference uses that mode only after decision authorization, then performs
    the full manifest/audit/audio verification before loading any model.
    """

    root = Path(repository_root)
    raw_lock_path = Path(path)
    lock_path = raw_lock_path if raw_lock_path.is_absolute() else root / raw_lock_path
    _repo_relative(lock_path, repository_root=root, label="preparation lock")
    if not lock_path.is_file():
        raise FileNotFoundError(f"FLEURS preparation lock does not exist: {lock_path}")
    lock_hash_before = sha256_file(lock_path)
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FleursPreparationError(
            f"FLEURS preparation lock is not valid UTF-8 JSON: {lock_path}"
        ) from error
    if not isinstance(lock, dict):
        raise FleursPreparationError("FLEURS preparation lock must be a JSON object")
    if sha256_file(lock_path) != lock_hash_before:
        raise FleursPreparationError("FLEURS preparation lock changed while being read")
    if lock.get("lock_version") != PREPARATION_LOCK_VERSION or lock.get("status") != "LOCKED":
        raise FleursPreparationError("unsupported or unlocked FLEURS preparation lock")
    identity = str(lock.get("identity_sha256", "")).casefold()
    identity_payload = dict(lock)
    identity_payload.pop("identity_sha256", None)
    if not _is_sha256(identity) or _canonical_sha256(identity_payload) != identity:
        raise FleursPreparationError("FLEURS preparation lock identity is invalid/tampered")
    dataset = lock.get("dataset")
    if not isinstance(dataset, Mapping):
        raise FleursPreparationError("FLEURS preparation lock has no dataset contract")
    if (
        dataset.get("repository") != DATASET_NAME
        or dataset.get("config") != LANGUAGE_CONFIG
        or dataset.get("split") != DEFAULT_SPLIT
        or not _is_immutable_revision(dataset.get("revision"))
    ):
        raise FleursPreparationError(
            "FLEURS lock must bind google/fleurs vi_vn test to an immutable revision"
        )
    output = lock.get("output")
    if not isinstance(output, Mapping):
        raise FleursPreparationError("FLEURS preparation lock has no output contract")
    try:
        row_count = int(output["row_count"])
        total_bytes = int(output["audio_total_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise FleursPreparationError("FLEURS output count/size is invalid") from error
    if row_count != expected_count:
        raise FleursPreparationError(
            f"formal FLEURS lock has {row_count} rows, expected exactly {expected_count}"
        )
    if tuple(output.get("manifest_schema", ())) != MANIFEST_FIELDS:
        raise FleursPreparationError("FLEURS manifest schema is not locked")
    if tuple(output.get("audit_schema", ())) != AUDIT_FIELDS:
        raise FleursPreparationError("FLEURS audit schema is not locked")
    for field in (
        "manifest_sha256",
        "audit_sha256",
        "audio_inventory_sha256",
    ):
        if not _is_sha256(output.get(field)):
            raise FleursPreparationError(f"FLEURS output.{field} is invalid")
    manifest_path = _resolve_portable(
        output.get("manifest"), repository_root=root, label="lock output.manifest"
    )
    audit_path = _resolve_portable(
        output.get("audit"), repository_root=root, label="lock output.audit"
    )
    result = {
        **lock,
        "preparation_lock_path": lock_path,
        "preparation_lock_sha256": lock_hash_before,
        "manifest_path": manifest_path,
        "audit_path": audit_path,
    }
    if not verify_artifacts:
        return result
    if not manifest_path.is_file() or sha256_file(manifest_path) != str(
        output["manifest_sha256"]
    ).casefold():
        raise FleursPreparationError("locked FLEURS manifest is missing or changed")
    if not audit_path.is_file() or sha256_file(audit_path) != str(
        output["audit_sha256"]
    ).casefold():
        raise FleursPreparationError("locked FLEURS audit is missing or changed")
    manifest_rows = validate_manifest(
        manifest_path,
        expected_count=expected_count,
        expected_split=DEFAULT_SPLIT,
        formal=True,
        repository_root=root,
        verify_audio_hashes=verify_audio,
    )
    audit_rows = _read_audit(audit_path)
    if len(audit_rows) != expected_count:
        raise FleursPreparationError("FLEURS audit row count differs from its lock")
    inventory = _audio_inventory_from_audit(audit_rows)
    if _canonical_sha256(inventory) != str(output["audio_inventory_sha256"]).casefold():
        raise FleursPreparationError("FLEURS audio inventory hash is invalid/tampered")
    if sum(int(item["num_bytes"]) for item in inventory) != total_bytes:
        raise FleursPreparationError("FLEURS locked audio byte total is invalid")
    for index, (manifest_row, audit_row) in enumerate(
        zip(manifest_rows, audit_rows), start=1
    ):
        if any(
            str(manifest_row[field]) != str(audit_row[field])
            for field in ("utt_id", "audio_path", "audio_sha256")
        ):
            raise FleursPreparationError(
                f"FLEURS manifest/audit identity mismatch at row {index}"
            )
        if verify_audio:
            audio_path = _resolve_portable(
                manifest_row["audio_path"],
                repository_root=root,
                label=f"manifest row {index} audio_path",
            )
            if audio_path.stat().st_size != int(audit_row["num_bytes"]):
                raise FleursPreparationError(
                    f"FLEURS audio size mismatch/tamper at row {index}"
                )
    return result


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
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    split: str = DEFAULT_SPLIT,
    expected_count: int | None = DEFAULT_EXPECTED_COUNT,
    revision: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
    dataset_loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    formal: bool = True,
    repository_root: str | Path = ROOT,
    preparation_lock: str | Path = DEFAULT_PREPARATION_LOCK,
    preparation_audit: str | Path = DEFAULT_PREPARATION_AUDIT,
    license_id: str | None = DEFAULT_LICENSE_ID,
    license_url: str | None = DEFAULT_LICENSE_URL,
) -> Path:
    """Materialize one FLEURS split and return its validated manifest path."""

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if expected_count is not None and expected_count < 1:
        raise ValueError("expected_count must be at least 1 or None")
    if not split or any(character in split for character in "/\\"):
        raise ValueError("split must be a simple non-empty name")
    root = Path(repository_root).resolve()
    revision_value = "main" if revision is None and not formal else str(revision or "")
    if formal:
        if split != DEFAULT_SPLIT or expected_count != DEFAULT_EXPECTED_COUNT:
            raise FleursPreparationError(
                "formal paper_v2 FLEURS preparation requires test with exactly 857 rows"
            )
        if not _is_immutable_revision(revision_value):
            raise FleursPreparationError(
                "formal FLEURS preparation requires --revision as an immutable "
                "40/64-character commit hash"
            )
        if overwrite:
            raise FleursPreparationError(
                "formal FLEURS artifacts are immutable; use a fresh output path"
            )

    raw_output_root = Path(out_dir)
    output_root = (
        (root / raw_output_root).resolve()
        if formal and not raw_output_root.is_absolute()
        else raw_output_root.resolve()
    )
    if formal:
        _repo_relative(output_root, repository_root=root, label="output directory")
    manifest_path = output_root / f"{split}.jsonl"
    audio_dir = output_root / "audio" / split
    lock_path = output_root / f".{split}.lock"
    preparation_lock_path = Path(preparation_lock)
    preparation_audit_path = Path(preparation_audit)
    if formal:
        if not preparation_lock_path.is_absolute():
            preparation_lock_path = (root / preparation_lock_path).resolve()
        if not preparation_audit_path.is_absolute():
            preparation_audit_path = (root / preparation_audit_path).resolve()
        _repo_relative(
            preparation_lock_path,
            repository_root=root,
            label="preparation lock",
        )
        _repo_relative(
            preparation_audit_path,
            repository_root=root,
            label="preparation audit",
        )

    with _output_lock(lock_path):
        if formal and preparation_lock_path.exists():
            if not resume:
                raise FileExistsError(
                    f"immutable preparation lock already exists: {preparation_lock_path}"
                )
            verified = verify_fleurs_preparation_lock(
                preparation_lock_path,
                repository_root=root,
                expected_count=DEFAULT_EXPECTED_COUNT,
                verify_artifacts=True,
                verify_audio=True,
            )
            if Path(verified["manifest_path"]).resolve() != manifest_path.resolve():
                raise FleursPreparationError(
                    "existing FLEURS lock points to another manifest"
                )
            if str(verified["dataset"]["revision"]).casefold() != revision_value.casefold():
                raise FleursPreparationError(
                    "existing FLEURS lock uses another dataset revision"
                )
            return manifest_path
        if manifest_path.exists():
            if formal:
                raise FleursPreparationError(
                    "formal manifest exists without its preparation lock; keep it "
                    "quarantined and use a fresh --out-dir/--preparation-lock"
                )
            if resume:
                validate_manifest(
                    manifest_path,
                    expected_count=expected_count,
                    expected_split=split,
                    formal=False,
                )
                return manifest_path
            if not overwrite:
                raise FileExistsError(
                    f"manifest already exists: {manifest_path}; use --resume or --overwrite"
                )

        loader = dataset_loader or load_fleurs_dataset
        dataset = loader(
            split=split,
            cache_dir=cache_dir,
            revision=revision_value,
        )
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
        audit_rows: list[dict[str, str | int]] = []
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
            metadata = _wav_metadata(audio_path)
            audio_reference = (
                _repo_relative(
                    audio_path,
                    repository_root=root,
                    label=f"audio for {utt_id}",
                )
                if formal
                else audio_path.resolve().as_posix()
            )
            rows.append(
                {
                    "utt_id": utt_id,
                    "dataset": "fleurs",
                    "split": split,
                    "audio_path": audio_reference,
                    "audio_sha256": str(metadata["audio_sha256"]),
                    "transcript": transcript,
                    "snr": "clean",
                    "noise_type": "clean",
                }
            )
            audit_rows.append(
                {
                    "utt_id": utt_id,
                    "audio_path": audio_reference,
                    **metadata,
                }
            )

        if expected_count is not None and len(rows) != expected_count:
            raise FleursPreparationError(
                f"source split produced {len(rows)} rows, expected {expected_count}"
            )
        if not rows:
            raise FleursPreparationError("source split is empty")

        # Validate the complete staged contract before publishing it. Audio
        # files intentionally remain after interruption; formal consumers see
        # the preparation only after the identity lock is committed last.
        staged = manifest_path.with_name(f".{manifest_path.name}.staged")
        try:
            _atomic_write_manifest(staged, rows)
            validate_manifest(
                staged,
                expected_count=expected_count,
                expected_split=split,
                formal=formal,
                repository_root=root,
                verify_audio_hashes=True,
            )
            if not formal:
                os.replace(staged, manifest_path)
                return manifest_path

            audit_payload = _audit_bytes(audit_rows)
            audit_hash = hashlib.sha256(audit_payload).hexdigest()
            manifest_payload = staged.read_bytes()
            manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
            inventory = _audio_inventory_from_audit(
                [{key: str(value) for key, value in row.items()} for row in audit_rows]
            )
            lock_payload: dict[str, Any] = {
                "lock_version": PREPARATION_LOCK_VERSION,
                "status": "LOCKED",
                "dataset": {
                    "repository": DATASET_NAME,
                    "config": LANGUAGE_CONFIG,
                    "split": split,
                    "revision": revision_value.casefold(),
                },
                "license": {
                    "id": str(license_id).strip() if license_id else None,
                    "url": str(license_url).strip() if license_url else None,
                    "status": (
                        "dataset_card_declared_at_revision"
                        if str(license_id).strip().upper() == DEFAULT_LICENSE_ID
                        and str(license_url).strip() == DEFAULT_LICENSE_URL
                        else "caller_asserted"
                        if license_id
                        else "not_asserted"
                    ),
                },
                "preparation_contract": {
                    "contract_version": PREPARATION_CONTRACT_VERSION,
                    "decoder": "huggingface_datasets_audio",
                    "channel_policy": "decoded_mono_only",
                    "resampling": "linear_interpolation_float64",
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "sample_format": "PCM_S16LE",
                    "amplitude_policy": "clip_-1_1_round_times_32767",
                    "transcript_normalization": "strip_then_NFC",
                    "utterance_id": "source_audio_filename_stem",
                    "ordering": "source_split_iteration_order",
                },
                "output": {
                    "manifest": _repo_relative(
                        manifest_path,
                        repository_root=root,
                        label="manifest",
                    ),
                    "manifest_sha256": manifest_hash,
                    "manifest_schema": list(MANIFEST_FIELDS),
                    "audit": _repo_relative(
                        preparation_audit_path,
                        repository_root=root,
                        label="audit",
                    ),
                    "audit_sha256": audit_hash,
                    "audit_schema": list(AUDIT_FIELDS),
                    "row_count": len(rows),
                    "audio_inventory_sha256": _canonical_sha256(inventory),
                    "audio_total_bytes": sum(
                        int(item["num_bytes"]) for item in inventory
                    ),
                },
            }
            lock_payload["identity_sha256"] = _canonical_sha256(lock_payload)
            published: list[Path] = []
            try:
                _atomic_write_new_bytes(manifest_path, manifest_payload)
                published.append(manifest_path)
                _atomic_write_new_bytes(preparation_audit_path, audit_payload)
                published.append(preparation_audit_path)
                _atomic_write_new_bytes(
                    preparation_lock_path,
                    _json_bytes(lock_payload),
                )
                published.append(preparation_lock_path)
                verify_fleurs_preparation_lock(
                    preparation_lock_path,
                    repository_root=root,
                    expected_count=DEFAULT_EXPECTED_COUNT,
                    verify_artifacts=True,
                    verify_audio=True,
                )
            except Exception:
                for published_path in reversed(published):
                    published_path.unlink(missing_ok=True)
                raise
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
        default=None,
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        dest="cache_dir",
        default=str(DEFAULT_CACHE_DIR),
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable Hugging Face dataset commit required in formal mode.",
    )
    parser.add_argument(
        "--preparation-lock",
        default=str(DEFAULT_PREPARATION_LOCK),
    )
    parser.add_argument(
        "--preparation-audit",
        default=str(DEFAULT_PREPARATION_AUDIT),
    )
    parser.add_argument("--license-id", default=DEFAULT_LICENSE_ID)
    parser.add_argument("--license-url", default=DEFAULT_LICENSE_URL)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Allow legacy/floating-revision preparation without a formal lock; "
            "these artifacts are not paper evidence."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formal = not args.diagnostic
    out_dir = args.out_dir or (
        str(DEFAULT_OUTPUT_DIR) if formal else "data/manifests/fleurs"
    )
    try:
        manifest = prepare_fleurs(
            out_dir=out_dir,
            cache_dir=args.cache_dir,
            split=args.split,
            expected_count=args.expected_count,
            revision=args.revision,
            resume=args.resume,
            overwrite=args.overwrite,
            formal=formal,
            preparation_lock=args.preparation_lock,
            preparation_audit=args.preparation_audit,
            license_id=args.license_id,
            license_url=args.license_url,
        )
        rows = validate_manifest(
            manifest,
            expected_count=args.expected_count,
            expected_split=args.split,
            formal=formal,
        )
    except (FleursPreparationError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} rows to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
