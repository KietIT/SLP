from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
NOISE_PROTOCOL_VERSION = "musan_content_split_v1"
NOISY_DEV_PROTOCOL_VERSION = "paper_v2_noisy_dev_v1"
NOISE_SPLIT_ALGORITHM = "sha256_stratified_content_split_v1"
NOISY_DEV_ALGORITHM = "sha256_noise_assignment_power_mix_v1"
MUSAN_TYPES = ("music", "noise", "speech")
AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".ogg", ".mp3", ".m4a"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

NOISE_REGISTRY_COLUMNS = (
    "noise_id",
    "source_dataset",
    "source_url",
    "source_revision",
    "license_id",
    "license_url",
    "source_relative_path",
    "audio",
    "audio_sha256",
    "duration_seconds",
    "sample_rate",
    "channels",
    "frames",
    "noise_type",
    "noise_subtype",
    "split",
)

NOISE_AUDIT_COLUMNS = (
    "protocol_version",
    "check_id",
    "entity",
    "split_a",
    "split_b",
    "status",
    "overlap_count",
    "total_a",
    "total_b",
    "details",
)

NOISY_DEV_COLUMNS = (
    "utt_id",
    "source_utt_id",
    "speaker_id",
    "dataset",
    "split",
    "condition",
    "audio_path",
    "audio_sha256",
    "clean_path",
    "clean_audio_sha256",
    "transcript",
    "text_sha256",
    "snr",
    "target_snr_db",
    "measured_snr_db",
    "noise_id",
    "noise_type",
    "noise_path",
    "noise_audio_sha256",
    "noise_split",
    "seed",
    "sample_rate",
    "duration_seconds",
    "pre_scale_peak",
    "anti_clip_gain",
    "pre_scale_clipped_samples",
    "stored_peak",
    "clipped_sample_count",
)

NOISY_DEV_AUDIT_COLUMNS = (
    "protocol_version",
    "check_id",
    "status",
    "observed",
    "expected",
    "details",
)


class NoiseProtocolError(ValueError):
    """Raised when a noise or derived benchmark lock is not reproducible."""


@dataclass(frozen=True)
class MusanSourceMetadata:
    source_url: str
    source_revision: str
    license_id: str
    license_url: str
    source_dataset: str = "musan"

    def validate(self) -> None:
        if not self.source_dataset.strip():
            raise NoiseProtocolError("source_dataset must not be empty")
        if not self.source_url.strip():
            raise NoiseProtocolError("source_url must not be empty")
        if not _is_sha256(self.source_revision):
            raise NoiseProtocolError(
                "source_revision must be the SHA-256 of the acquired MUSAN artifact"
            )
        if not self.license_id.strip() or not self.license_url.strip():
            raise NoiseProtocolError("license_id and license_url must not be empty")


@dataclass(frozen=True)
class NoisyDevConfig:
    source_dev_manifest: Path
    source_dev_sha256: str
    noise_split_lock: Path
    output_manifest: Path
    output_audio_dir: Path
    protocol_lock: Path
    protocol_audit: Path
    snrs: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0)
    seed: int = 42
    sample_rate: int = 16000
    peak_limit: float = 0.999
    include_clean: bool = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value).strip().casefold()))


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _artifact_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _csv_bytes(
    rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NoiseProtocolError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise NoiseProtocolError(
                    f"Manifest row must be an object at {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise NoiseProtocolError(f"Manifest is empty: {path}")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Lock does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NoiseProtocolError(f"Lock is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise NoiseProtocolError(f"Lock must contain one JSON object: {path}")
    return value


def _audio_info(path: Path) -> tuple[float, int, int, int]:
    try:
        import soundfile as sf

        info = sf.info(path)
    except Exception as exc:
        raise NoiseProtocolError(f"Cannot inspect audio file: {path}") from exc
    if info.frames < 1 or info.samplerate < 1 or info.channels < 1:
        raise NoiseProtocolError(f"Audio file is empty or malformed: {path}")
    duration = float(info.frames) / float(info.samplerate)
    if not math.isfinite(duration) or duration <= 0.0:
        raise NoiseProtocolError(f"Audio duration is invalid: {path}")
    return duration, int(info.samplerate), int(info.channels), int(info.frames)


def _noise_subtype(relative_path: Path) -> str:
    parts = relative_path.parts
    return parts[1] if len(parts) > 2 else "unspecified"


def _validate_split_fractions(
    train_fraction: float, dev_fraction: float, test_fraction: float
) -> None:
    values = (train_fraction, dev_fraction, test_fraction)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise NoiseProtocolError("All split fractions must be finite and positive")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise NoiseProtocolError("Noise split fractions must sum to 1.0")


def _partition_type_rows(
    rows: Sequence[dict[str, object]],
    *,
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> list[dict[str, object]]:
    if len(rows) < 3:
        noise_type = rows[0]["noise_type"] if rows else "unknown"
        raise NoiseProtocolError(
            f"Noise type {noise_type!r} needs at least three files for train/dev/test"
        )

    def rank(row: Mapping[str, object]) -> tuple[str, str]:
        identity = (
            f"{NOISE_SPLIT_ALGORITHM}|seed={seed}|type={row['noise_type']}|"
            f"sha256={row['audio_sha256']}|id={row['noise_id']}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest(), str(
            row["noise_id"]
        )

    ranked = sorted(rows, key=rank)
    dev_count = max(1, int(math.floor(len(rows) * dev_fraction + 0.5)))
    test_count = max(1, int(math.floor(len(rows) * test_fraction + 0.5)))
    if dev_count + test_count >= len(rows):
        raise NoiseProtocolError(
            f"Split fractions leave no training files for {rows[0]['noise_type']!r}"
        )
    output: list[dict[str, object]] = []
    for index, row in enumerate(ranked):
        split = (
            "test"
            if index < test_count
            else "dev"
            if index < test_count + dev_count
            else "train"
        )
        output.append({**row, "split": split})
    return output


def _entity_values(
    rows: Sequence[Mapping[str, object]], entity: str
) -> set[str]:
    return {str(row[entity]) for row in rows}


def _noise_audit_row(
    check_id: str,
    entity: str,
    split_a: str,
    split_b: str,
    values_a: set[str],
    values_b: set[str],
    *,
    identity: bool = False,
) -> dict[str, object]:
    differences = (
        values_a.symmetric_difference(values_b)
        if identity
        else values_a.intersection(values_b)
    )
    return {
        "protocol_version": NOISE_PROTOCOL_VERSION,
        "check_id": check_id,
        "entity": entity,
        "split_a": split_a,
        "split_b": split_b,
        "status": "PASS" if not differences else "FAIL",
        "overlap_count": len(differences),
        "total_a": len(values_a),
        "total_b": len(values_b),
        "details": "none" if not differences else ";".join(sorted(differences)[:20]),
    }


def _build_noise_audit(
    registry_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_split = {
        split: [row for row in registry_rows if row["split"] == split]
        for split in ("train", "dev", "test")
    }
    audit: list[dict[str, object]] = []
    for entity in ("noise_id", "source_relative_path", "audio_sha256"):
        for split_a, split_b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            audit.append(
                _noise_audit_row(
                    f"no_{entity}_overlap_{split_a}_{split_b}",
                    entity,
                    split_a,
                    split_b,
                    _entity_values(by_split[split_a], entity),
                    _entity_values(by_split[split_b], entity),
                )
            )
        partition_values = set().union(
            *(_entity_values(by_split[split], entity) for split in by_split)
        )
        audit.append(
            _noise_audit_row(
                f"registry_partition_identity_{entity}",
                entity,
                "registry",
                "train_plus_dev_plus_test",
                _entity_values(registry_rows, entity),
                partition_values,
                identity=True,
            )
        )
    for split, rows in by_split.items():
        observed_types = {str(row["noise_type"]) for row in rows}
        missing = set(MUSAN_TYPES).difference(observed_types)
        audit.append(
            {
                "protocol_version": NOISE_PROTOCOL_VERSION,
                "check_id": f"all_noise_types_present_{split}",
                "entity": "noise_type",
                "split_a": split,
                "split_b": "required",
                "status": "PASS" if not missing else "FAIL",
                "overlap_count": len(missing),
                "total_a": len(observed_types),
                "total_b": len(MUSAN_TYPES),
                "details": "none" if not missing else ";".join(sorted(missing)),
            }
        )
    failures = [row["check_id"] for row in audit if row["status"] != "PASS"]
    if failures:
        raise NoiseProtocolError("Noise split audit failed: " + ", ".join(failures))
    return audit


def build_noise_protocol_outputs(
    musan_root: str | Path,
    *,
    manifest_dir: str | Path,
    protocol_dir: str | Path,
    source: MusanSourceMetadata,
    seed: int = 42,
    train_fraction: float = 0.8,
    dev_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[Path, bytes]:
    """Build immutable MUSAN registry/split payloads without writing them."""

    source.validate()
    _validate_split_fractions(train_fraction, dev_fraction, test_fraction)
    if seed < 0:
        raise NoiseProtocolError("seed must be non-negative")
    root = Path(musan_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MUSAN root does not exist: {root}")
    resolved_root = root.resolve()
    raw_rows: list[dict[str, object]] = []
    for noise_type in MUSAN_TYPES:
        type_root = root / noise_type
        if not type_root.is_dir():
            raise NoiseProtocolError(
                f"Official MUSAN type directory is missing: {type_root}"
            )
        paths = sorted(
            (
                path
                for path in type_root.rglob("*")
                if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        if not paths:
            raise NoiseProtocolError(f"No audio files found for MUSAN type {noise_type}")
        for path in paths:
            resolved_path = path.resolve()
            try:
                relative = resolved_path.relative_to(resolved_root)
            except ValueError as exc:
                raise NoiseProtocolError(
                    f"MUSAN symlink escapes the source root: {path}"
                ) from exc
            duration, sample_rate, channels, frames = _audio_info(resolved_path)
            audio_sha256 = sha256_file(resolved_path)
            noise_id = "musan_" + hashlib.sha256(
                relative.as_posix().encode("utf-8")
            ).hexdigest()[:24]
            raw_rows.append(
                {
                    "noise_id": noise_id,
                    "source_dataset": source.source_dataset,
                    "source_url": source.source_url,
                    "source_revision": source.source_revision.casefold(),
                    "license_id": source.license_id,
                    "license_url": source.license_url,
                    "source_relative_path": relative.as_posix(),
                    "audio": _display_path(resolved_path),
                    "audio_sha256": audio_sha256,
                    "duration_seconds": f"{duration:.9f}",
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "frames": frames,
                    "noise_type": noise_type,
                    "noise_subtype": _noise_subtype(relative),
                }
            )

    for entity in ("noise_id", "source_relative_path", "audio_sha256"):
        values = [str(row[entity]) for row in raw_rows]
        if len(values) != len(set(values)):
            raise NoiseProtocolError(
                f"MUSAN registry has duplicate {entity}; content aliases are forbidden"
            )

    registry_rows: list[dict[str, object]] = []
    for noise_type in MUSAN_TYPES:
        registry_rows.extend(
            _partition_type_rows(
                [row for row in raw_rows if row["noise_type"] == noise_type],
                seed=seed,
                dev_fraction=dev_fraction,
                test_fraction=test_fraction,
            )
        )
    registry_rows = sorted(registry_rows, key=lambda row: str(row["noise_id"]))
    audit_rows = _build_noise_audit(registry_rows)

    manifest_dir = Path(manifest_dir)
    protocol_dir = Path(protocol_dir)
    registry_path = manifest_dir / "musan_registry.jsonl"
    split_paths = {
        split: manifest_dir / f"musan_{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    registry_payload = _jsonl_bytes(registry_rows)
    split_rows = {
        split: [row for row in registry_rows if row["split"] == split]
        for split in split_paths
    }
    split_payloads = {
        split: _jsonl_bytes(rows) for split, rows in split_rows.items()
    }
    audit_path = protocol_dir / "noise_split_audit.csv"
    audit_payload = _csv_bytes(audit_rows, NOISE_AUDIT_COLUMNS)
    source_inventory = [
        {
            "source_relative_path": row["source_relative_path"],
            "audio_sha256": row["audio_sha256"],
            "duration_seconds": row["duration_seconds"],
            "sample_rate": row["sample_rate"],
            "channels": row["channels"],
            "noise_type": row["noise_type"],
        }
        for row in registry_rows
    ]
    lock = {
        "protocol_version": NOISE_PROTOCOL_VERSION,
        "dataset": "musan",
        "status": "LOCKED",
        "seed": seed,
        "split_algorithm": NOISE_SPLIT_ALGORITHM,
        "hash_algorithm": "sha256_raw_bytes",
        "source": {
            "dataset": source.source_dataset,
            "url": source.source_url,
            "revision": source.source_revision.casefold(),
            "license_id": source.license_id,
            "license_url": source.license_url,
            "inventory_sha256": _canonical_sha256(source_inventory),
        },
        "schema": list(NOISE_REGISTRY_COLUMNS),
        "fractions": {
            "train": train_fraction,
            "dev": dev_fraction,
            "test": test_fraction,
        },
        "registry": {
            "manifest": _display_path(registry_path),
            "manifest_sha256": _sha256_bytes(registry_payload),
            "file_count": len(registry_rows),
        },
        "splits": {
            split: {
                "manifest": _display_path(split_paths[split]),
                "manifest_sha256": _sha256_bytes(split_payloads[split]),
                "file_count": len(rows),
                "type_counts": {
                    noise_type: sum(
                        row["noise_type"] == noise_type for row in rows
                    )
                    for noise_type in MUSAN_TYPES
                },
            }
            for split, rows in split_rows.items()
        },
        "audit": {
            "path": _display_path(audit_path),
            "sha256": _sha256_bytes(audit_payload),
            "checks": len(audit_rows),
            "failed_checks": 0,
        },
    }
    lock_payload = (
        json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return {
        registry_path: registry_payload,
        **{
            split_paths[split]: payload
            for split, payload in split_payloads.items()
        },
        protocol_dir / "noise_split_lock.json": lock_payload,
        audit_path: audit_payload,
    }


def _remove_transaction_target(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _commit_payloads(
    payloads: Mapping[Path, bytes], *, overwrite: bool
) -> str:
    existing = [path for path in payloads if path.exists()]
    if len(existing) == len(payloads) and all(
        path.is_file() and path.read_bytes() == payload
        for path, payload in payloads.items()
    ):
        return "verified_existing"
    if existing and not overwrite:
        raise NoiseProtocolError(
            "Locked outputs already exist but differ; refusing to overwrite: "
            + ", ".join(_display_path(path) for path in existing)
        )

    temporary: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            backup = path.with_name(f".{path.name}.{os.getpid()}.backup")
            if tmp.exists() or backup.exists():
                raise NoiseProtocolError(f"Stale transaction artifact: {tmp} or {backup}")
            tmp.write_bytes(payload)
            temporary[path] = tmp
            if path.exists():
                backups[path] = backup
        for path, backup in backups.items():
            path.rename(backup)
        try:
            for path, tmp in temporary.items():
                tmp.rename(path)
                committed.append(path)
        except Exception:
            for path in reversed(committed):
                _remove_transaction_target(path)
            for path, backup in backups.items():
                if backup.exists() and not path.exists():
                    backup.rename(path)
            raise
        for backup in backups.values():
            _remove_transaction_target(backup)
    finally:
        for tmp in temporary.values():
            _remove_transaction_target(tmp)
        for path, backup in backups.items():
            if backup.exists() and not path.exists():
                backup.rename(path)
    return "written"


def write_locked_noise_outputs(
    payloads: Mapping[Path, bytes], *, overwrite: bool = False
) -> str:
    return _commit_payloads(payloads, overwrite=overwrite)


def _validate_noise_row(
    row: Mapping[str, Any], *, manifest: Path, row_number: int, verify_audio: bool
) -> None:
    missing = set(NOISE_REGISTRY_COLUMNS).difference(row)
    extra = set(row).difference(NOISE_REGISTRY_COLUMNS)
    if missing or extra:
        raise NoiseProtocolError(
            f"{manifest}:{row_number} has a registry schema mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if row["noise_type"] not in MUSAN_TYPES or row["split"] not in {
        "train",
        "dev",
        "test",
    }:
        raise NoiseProtocolError(f"Invalid type/split at {manifest}:{row_number}")
    if not _is_sha256(row["audio_sha256"]):
        raise NoiseProtocolError(f"Invalid audio SHA-256 at {manifest}:{row_number}")
    if not _is_sha256(row["source_revision"]):
        raise NoiseProtocolError(f"Invalid source revision at {manifest}:{row_number}")
    if any(
        not str(row[field]).strip()
        for field in (
            "noise_id",
            "source_dataset",
            "source_url",
            "license_id",
            "license_url",
            "source_relative_path",
            "audio",
        )
    ):
        raise NoiseProtocolError(f"Empty registry metadata at {manifest}:{row_number}")
    try:
        duration = float(row["duration_seconds"])
        sample_rate = int(row["sample_rate"])
        channels = int(row["channels"])
        frames = int(row["frames"])
    except (TypeError, ValueError) as exc:
        raise NoiseProtocolError(
            f"Invalid audio metadata at {manifest}:{row_number}"
        ) from exc
    if (
        not math.isfinite(duration)
        or duration <= 0.0
        or sample_rate < 1
        or channels < 1
        or frames < 1
    ):
        raise NoiseProtocolError(f"Invalid audio metadata at {manifest}:{row_number}")
    if verify_audio:
        audio = _artifact_path(row["audio"])
        if not audio.is_file():
            raise FileNotFoundError(f"Noise audio does not exist: {audio}")
        if sha256_file(audio) != str(row["audio_sha256"]).casefold():
            raise NoiseProtocolError(f"Noise audio SHA-256 mismatch: {audio}")
        actual_duration, actual_rate, actual_channels, actual_frames = _audio_info(audio)
        if (
            actual_rate != sample_rate
            or actual_channels != channels
            or actual_frames != frames
            or not math.isclose(actual_duration, duration, abs_tol=1e-9)
        ):
            raise NoiseProtocolError(f"Noise audio metadata mismatch: {audio}")


def verify_noise_split_lock(
    lock_path: str | Path, *, verify_audio: bool = True
) -> dict[str, Any]:
    """Verify the registry, all three partitions, audit and optional audio bytes."""

    lock_path = Path(lock_path)
    lock = _load_json(lock_path)
    if (
        lock.get("protocol_version") != NOISE_PROTOCOL_VERSION
        or lock.get("dataset") != "musan"
        or lock.get("status") != "LOCKED"
    ):
        raise NoiseProtocolError(f"Unsupported or unlocked noise protocol: {lock_path}")
    if lock.get("schema") != list(NOISE_REGISTRY_COLUMNS):
        raise NoiseProtocolError(f"Noise registry schema mismatch: {lock_path}")

    registry_meta = lock.get("registry", {})
    registry_path = _artifact_path(registry_meta.get("manifest", ""))
    if sha256_file(registry_path) != registry_meta.get("manifest_sha256"):
        raise NoiseProtocolError("Noise registry hash does not match its lock")
    registry_rows = _read_jsonl(registry_path)
    if len(registry_rows) != int(registry_meta.get("file_count", -1)):
        raise NoiseProtocolError("Noise registry row count does not match its lock")
    for number, row in enumerate(registry_rows, start=1):
        _validate_noise_row(
            row,
            manifest=registry_path,
            row_number=number,
            verify_audio=verify_audio,
        )

    source_meta = lock.get("source", {})
    expected_source = {
        "source_dataset": source_meta.get("dataset"),
        "source_url": source_meta.get("url"),
        "source_revision": source_meta.get("revision"),
        "license_id": source_meta.get("license_id"),
        "license_url": source_meta.get("license_url"),
    }
    for field, expected in expected_source.items():
        if not expected or any(row.get(field) != expected for row in registry_rows):
            raise NoiseProtocolError(
                f"Noise registry {field} does not match source metadata in the lock"
            )
    source_inventory = [
        {
            "source_relative_path": row["source_relative_path"],
            "audio_sha256": row["audio_sha256"],
            "duration_seconds": row["duration_seconds"],
            "sample_rate": row["sample_rate"],
            "channels": row["channels"],
            "noise_type": row["noise_type"],
        }
        for row in registry_rows
    ]
    if _canonical_sha256(source_inventory) != source_meta.get("inventory_sha256"):
        raise NoiseProtocolError("Noise source inventory hash does not match its lock")

    registry_ids = [str(row["noise_id"]) for row in registry_rows]
    registry_hashes = [str(row["audio_sha256"]) for row in registry_rows]
    registry_paths = [str(row["source_relative_path"]) for row in registry_rows]
    if (
        len(registry_ids) != len(set(registry_ids))
        or len(registry_hashes) != len(set(registry_hashes))
        or len(registry_paths) != len(set(registry_paths))
    ):
        raise NoiseProtocolError("Noise registry identities are not unique")

    partition_rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        meta = lock.get("splits", {}).get(split, {})
        path = _artifact_path(meta.get("manifest", ""))
        if sha256_file(path) != meta.get("manifest_sha256"):
            raise NoiseProtocolError(f"Noise {split} manifest hash mismatch")
        rows = _read_jsonl(path)
        if len(rows) != int(meta.get("file_count", -1)):
            raise NoiseProtocolError(f"Noise {split} row count mismatch")
        if any(row.get("split") != split for row in rows):
            raise NoiseProtocolError(f"Noise {split} manifest declares another split")
        expected_counts = {
            noise_type: sum(row.get("noise_type") == noise_type for row in rows)
            for noise_type in MUSAN_TYPES
        }
        if expected_counts != meta.get("type_counts"):
            raise NoiseProtocolError(f"Noise {split} type counts mismatch")
        partition_rows.extend(rows)
    if sorted(partition_rows, key=lambda row: row["noise_id"]) != sorted(
        registry_rows, key=lambda row: row["noise_id"]
    ):
        raise NoiseProtocolError("Noise partitions are not identical to the registry")
    _build_noise_audit(registry_rows)

    audit_meta = lock.get("audit", {})
    audit_path = _artifact_path(audit_meta.get("path", ""))
    if sha256_file(audit_path) != audit_meta.get("sha256"):
        raise NoiseProtocolError("Noise split audit hash mismatch")
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if len(audit_rows) != int(audit_meta.get("checks", -1)) or any(
        row.get("status") != "PASS" for row in audit_rows
    ):
        raise NoiseProtocolError("Noise split audit is incomplete or contains failures")
    return {
        "lock": lock,
        "lock_sha256": sha256_file(lock_path),
        "registry_path": registry_path,
        "registry_rows": registry_rows,
        "dev_manifest": _artifact_path(lock["splits"]["dev"]["manifest"]),
        "dev_rows": [row for row in registry_rows if row["split"] == "dev"],
        "audio_hashes_verified": verify_audio,
    }


def _validate_noisy_dev_config(config: NoisyDevConfig) -> dict[str, object]:
    if not _is_sha256(config.source_dev_sha256):
        raise NoiseProtocolError("source_dev_sha256 must be a valid SHA-256")
    if config.seed < 0 or config.sample_rate < 1:
        raise NoiseProtocolError("seed must be non-negative and sample_rate positive")
    if not 0.0 < config.peak_limit < 1.0:
        raise NoiseProtocolError("peak_limit must be between zero and one")
    if not config.snrs:
        raise NoiseProtocolError("At least one noisy-dev SNR is required")
    normalized_snrs = tuple(float(value) for value in config.snrs)
    if any(not math.isfinite(value) for value in normalized_snrs):
        raise NoiseProtocolError("Every noisy-dev SNR must be finite")
    if len(set(normalized_snrs)) != len(normalized_snrs):
        raise NoiseProtocolError("Noisy-dev SNR values must be unique")
    return {
        "algorithm": NOISY_DEV_ALGORITHM,
        "seed": config.seed,
        "snrs_db": list(normalized_snrs),
        "sample_rate": config.sample_rate,
        "peak_limit": config.peak_limit,
        "include_clean": config.include_clean,
        "audio_container": "WAV",
        "audio_subtype": "PCM_16",
        "input_sample_rate_policy": "require_exact",
        "channel_policy": "mean_to_mono",
        "snr_measurement": "component_power_after_anti_clip_before_pcm16",
        "clipping_measurement": "pre_scale_over_1_and_stored_full_scale",
    }


def _validate_source_dev_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for number, row in enumerate(rows, start=1):
        utt_id = str(row.get("source_utt_id") or row.get("utt_id") or "").strip()
        audio_value = row.get("audio") or row.get("audio_path")
        text = row.get("text") or row.get("transcript")
        audio_sha = str(row.get("audio_sha256", "")).casefold()
        if not utt_id or not audio_value or text is None or row.get("split") != "dev":
            raise NoiseProtocolError(
                f"Source dev row is missing identity/audio/text or is not dev: {path}:{number}"
            )
        if utt_id in seen_ids:
            raise NoiseProtocolError(f"Duplicate source dev utterance ID: {utt_id}")
        if not _is_sha256(audio_sha):
            raise NoiseProtocolError(f"Invalid source audio hash at {path}:{number}")
        audio = _artifact_path(audio_value)
        if not audio.is_file() or sha256_file(audio) != audio_sha:
            raise NoiseProtocolError(f"Source dev audio hash mismatch: {audio}")
        text_sha = str(row.get("text_sha256", "")).casefold()
        if text_sha and (
            not _is_sha256(text_sha)
            or _sha256_bytes(str(text).encode("utf-8")) != text_sha
        ):
            raise NoiseProtocolError(f"Source dev text hash mismatch at {path}:{number}")
        seen_ids.add(utt_id)
        seen_hashes.add(audio_sha)
    if len(seen_hashes) != len(rows):
        raise NoiseProtocolError("Source dev contains duplicate audio content")
    return sorted(
        rows,
        key=lambda row: str(row.get("source_utt_id") or row.get("utt_id")),
    )


def _stable_seed(master_seed: int, source_id: str, snr: float) -> int:
    payload = (
        f"{NOISY_DEV_ALGORITHM}|seed={master_seed}|source={source_id}|snr={snr:g}"
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized[:120] or "utterance"


def _snr_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _mix_with_measurements(
    clean: Any, noise: Any, *, target_snr_db: float, peak_limit: float
) -> tuple[Any, dict[str, float | int]]:
    import numpy as np

    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if len(clean) == 0 or len(noise) != len(clean):
        raise NoiseProtocolError("Clean/noise waveforms must be non-empty and aligned")
    clean_power = float(np.mean(clean**2))
    noise_power = float(np.mean(noise**2))
    if clean_power <= 1e-12 or noise_power <= 1e-12:
        raise NoiseProtocolError("Cannot mix silent clean or noise audio")
    noise_gain = math.sqrt(
        clean_power / (noise_power * (10.0 ** (target_snr_db / 10.0)))
    )
    noise_component = noise * noise_gain
    raw = clean + noise_component
    pre_peak = float(np.max(np.abs(raw)))
    pre_clipped = int(np.count_nonzero(np.abs(raw) > 1.0))
    anti_clip_gain = min(1.0, peak_limit / max(pre_peak, 1e-12))
    clean_component = clean * anti_clip_gain
    noise_component = noise_component * anti_clip_gain
    mixed = clean_component + noise_component
    measured = 10.0 * math.log10(
        float(np.mean(clean_component**2))
        / float(np.mean(noise_component**2))
    )
    return mixed.astype(np.float32), {
        "measured_snr_db": measured,
        "pre_scale_peak": pre_peak,
        "anti_clip_gain": anti_clip_gain,
        "pre_scale_clipped_samples": pre_clipped,
    }


def _builder_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    source_count: int,
    snrs: Sequence[float],
    include_clean: bool,
    audio_dir: Path,
) -> list[dict[str, object]]:
    noisy_rows = [row for row in rows if row["condition"] == "noisy"]
    expected_rows = source_count * (len(snrs) + int(include_clean))
    checks: list[tuple[str, bool, object, object, str]] = [
        (
            "expected_row_count",
            len(rows) == expected_rows,
            len(rows),
            expected_rows,
            "one row per source and condition",
        ),
        (
            "unique_utterance_ids",
            len({row["utt_id"] for row in rows}) == len(rows),
            len({row["utt_id"] for row in rows}),
            len(rows),
            "derived utterance IDs must be unique",
        ),
        (
            "noise_partition_is_dev",
            all(row.get("noise_split") == "dev" for row in noisy_rows),
            len([row for row in noisy_rows if row.get("noise_split") == "dev"]),
            len(noisy_rows),
            "no train/test noise may enter noisy-dev",
        ),
        (
            "measured_snr_tolerance",
            all(
                abs(float(row["measured_snr_db"]) - float(row["target_snr_db"]))
                <= 1e-6
                for row in noisy_rows
            ),
            max(
                (
                    abs(float(row["measured_snr_db"]) - float(row["target_snr_db"]))
                    for row in noisy_rows
                ),
                default=0.0,
            ),
            1e-6,
            "measured component-power SNR error",
        ),
        (
            "stored_audio_has_no_clipping",
            all(int(row["clipped_sample_count"]) == 0 for row in noisy_rows),
            sum(int(row["clipped_sample_count"]) for row in noisy_rows),
            0,
            "post-write samples at full scale",
        ),
        (
            "derived_audio_paths_contained",
            all(
                _artifact_path(row["audio_path"]).resolve().is_relative_to(
                    audio_dir.resolve()
                )
                for row in noisy_rows
            ),
            len(noisy_rows),
            len(noisy_rows),
            "every noisy artifact must live under output_audio_dir",
        ),
    ]
    audit = [
        {
            "protocol_version": NOISY_DEV_PROTOCOL_VERSION,
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
        for check_id, passed, observed, expected, details in checks
    ]
    failures = [row["check_id"] for row in audit if row["status"] != "PASS"]
    if failures:
        raise NoiseProtocolError("Noisy-dev audit failed: " + ", ".join(failures))
    return audit


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _verify_existing_noisy_dev(
    config: NoisyDevConfig,
    *,
    builder_sha256: str,
    noise_lock_sha256: str,
) -> dict[str, object]:
    lock = _load_json(config.protocol_lock)
    if lock.get("protocol_version") != NOISY_DEV_PROTOCOL_VERSION:
        raise NoiseProtocolError("Existing noisy-dev lock uses another protocol")
    if lock.get("source_dev", {}).get("manifest_sha256") != config.source_dev_sha256:
        raise NoiseProtocolError("Existing noisy-dev lock binds another source dev")
    if lock.get("noise", {}).get("split_lock_sha256") != noise_lock_sha256:
        raise NoiseProtocolError("Existing noisy-dev lock binds another noise lock")
    if lock.get("builder", {}).get("params_sha256") != builder_sha256:
        raise NoiseProtocolError("Existing noisy-dev lock binds other builder parameters")
    output = lock.get("output", {})
    manifest_path = _artifact_path(output.get("manifest", ""))
    audio_dir = _artifact_path(output.get("audio_dir", ""))
    if not _same_path(manifest_path, config.output_manifest) or not _same_path(
        audio_dir, config.output_audio_dir
    ):
        raise NoiseProtocolError("Existing noisy-dev lock binds other output paths")
    if sha256_file(manifest_path) != output.get("manifest_sha256"):
        raise NoiseProtocolError("Existing noisy-dev manifest hash mismatch")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(output.get("row_count", -1)):
        raise NoiseProtocolError("Existing noisy-dev row count mismatch")
    noisy_hashes: list[dict[str, str]] = []
    for row in rows:
        audio = _artifact_path(row["audio_path"])
        if not audio.is_file() or sha256_file(audio) != row.get("audio_sha256"):
            raise NoiseProtocolError(f"Existing noisy-dev audio hash mismatch: {audio}")
        if row["condition"] == "noisy":
            noisy_hashes.append(
                {"utt_id": str(row["utt_id"]), "audio_sha256": str(row["audio_sha256"])}
            )
    if _canonical_sha256(noisy_hashes) != output.get("audio_inventory_sha256"):
        raise NoiseProtocolError("Existing noisy-dev audio inventory hash mismatch")
    audit_meta = lock.get("audit", {})
    if not _same_path(_artifact_path(audit_meta.get("path", "")), config.protocol_audit):
        raise NoiseProtocolError("Existing noisy-dev lock binds another audit path")
    if sha256_file(config.protocol_audit) != audit_meta.get("sha256"):
        raise NoiseProtocolError("Existing noisy-dev audit hash mismatch")
    with config.protocol_audit.open("r", encoding="utf-8-sig", newline="") as handle:
        audit = list(csv.DictReader(handle))
    if len(audit) != int(audit_meta.get("checks", -1)) or any(
        row.get("status") != "PASS" for row in audit
    ):
        raise NoiseProtocolError("Existing noisy-dev audit is incomplete or failed")
    return {"status": "verified_existing", "rows": len(rows), "lock": lock}


def _commit_noisy_dev_transaction(
    *,
    staged_audio_dir: Path,
    output_audio_dir: Path,
    payloads: Mapping[Path, bytes],
    overwrite: bool,
) -> None:
    targets = [output_audio_dir, *payloads]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise NoiseProtocolError(
            "Noisy-dev outputs already exist; refusing to overwrite: "
            + ", ".join(_display_path(path) for path in existing)
        )
    temporary: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            if tmp.exists():
                raise NoiseProtocolError(f"Stale transaction artifact: {tmp}")
            tmp.write_bytes(payload)
            temporary[path] = tmp
        for target in targets:
            if target.exists():
                backup = target.with_name(f".{target.name}.{os.getpid()}.backup")
                if backup.exists():
                    raise NoiseProtocolError(f"Stale transaction backup: {backup}")
                backups[target] = backup
        for target, backup in backups.items():
            target.rename(backup)
        try:
            staged_audio_dir.rename(output_audio_dir)
            committed.append(output_audio_dir)
            for target, tmp in temporary.items():
                tmp.rename(target)
                committed.append(target)
        except Exception:
            for target in reversed(committed):
                _remove_transaction_target(target)
            for target, backup in backups.items():
                if backup.exists() and not target.exists():
                    backup.rename(target)
            raise
        for backup in backups.values():
            _remove_transaction_target(backup)
    finally:
        for tmp in temporary.values():
            _remove_transaction_target(tmp)
        for target, backup in backups.items():
            if backup.exists() and not target.exists():
                backup.rename(target)


def build_noisy_dev_benchmark(
    config: NoisyDevConfig, *, overwrite: bool = False
) -> dict[str, object]:
    """Build and lock a noisy benchmark derived exclusively from source dev."""

    builder_params = _validate_noisy_dev_config(config)
    if not config.source_dev_manifest.is_file():
        raise FileNotFoundError(
            f"Source dev manifest does not exist: {config.source_dev_manifest}"
        )
    if sha256_file(config.source_dev_manifest) != config.source_dev_sha256:
        raise NoiseProtocolError("Source dev manifest SHA-256 does not match config")
    source_rows = _validate_source_dev_rows(config.source_dev_manifest)
    noise_verified = verify_noise_split_lock(config.noise_split_lock, verify_audio=True)
    noise_lock = noise_verified["lock"]
    noise_rows = list(noise_verified["dev_rows"])
    if not noise_rows:
        raise NoiseProtocolError("Locked MUSAN dev partition is empty")
    builder_sha256 = _canonical_sha256(builder_params)
    noise_lock_sha256 = str(noise_verified["lock_sha256"])

    targets = (
        config.output_manifest,
        config.output_audio_dir,
        config.protocol_lock,
        config.protocol_audit,
    )
    if any(path.exists() for path in targets) and not overwrite:
        if all(path.exists() for path in targets):
            try:
                return _verify_existing_noisy_dev(
                    config,
                    builder_sha256=builder_sha256,
                    noise_lock_sha256=noise_lock_sha256,
                )
            except (OSError, NoiseProtocolError) as exc:
                raise NoiseProtocolError(
                    "Existing noisy-dev artifacts do not match this source/config; "
                    "refusing to overwrite"
                ) from exc
        raise NoiseProtocolError(
            "Partial noisy-dev outputs exist; refusing to overwrite without --overwrite"
        )

    stage = config.output_audio_dir.with_name(
        f".{config.output_audio_dir.name}.{os.getpid()}.tmp"
    )
    if stage.exists():
        raise NoiseProtocolError(f"Stale noisy-dev staging directory: {stage}")
    stage.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    try:
        from .noise import fit_noise, read_audio
        import numpy as np
        import soundfile as sf

        for source_row in source_rows:
            source_id = str(
                source_row.get("source_utt_id") or source_row.get("utt_id")
            )
            clean_path = _artifact_path(
                source_row.get("audio") or source_row.get("audio_path")
            )
            _, source_sample_rate, _, _ = _audio_info(clean_path)
            if source_sample_rate != config.sample_rate:
                raise NoiseProtocolError(
                    f"Source dev sample rate must be {config.sample_rate}: {clean_path}"
                )
            clean_sha = str(source_row["audio_sha256"]).casefold()
            transcript = str(source_row.get("text") or source_row.get("transcript"))
            text_sha = str(source_row.get("text_sha256") or _sha256_bytes(transcript.encode("utf-8")))
            if config.include_clean:
                duration, clean_rate, _, _ = _audio_info(clean_path)
                rows.append(
                    {
                        "utt_id": f"{source_id}_clean",
                        "source_utt_id": source_id,
                        "speaker_id": source_row.get("speaker_id", ""),
                        "dataset": source_row.get("dataset", "vivos"),
                        "split": "dev",
                        "condition": "clean",
                        "audio_path": _display_path(clean_path),
                        "audio_sha256": clean_sha,
                        "clean_path": _display_path(clean_path),
                        "clean_audio_sha256": clean_sha,
                        "transcript": transcript,
                        "text_sha256": text_sha,
                        "snr": "clean",
                        "target_snr_db": "",
                        "measured_snr_db": "",
                        "noise_id": "",
                        "noise_type": "clean",
                        "noise_path": "",
                        "noise_audio_sha256": "",
                        "noise_split": "",
                        "seed": config.seed,
                        "sample_rate": clean_rate,
                        "duration_seconds": f"{duration:.9f}",
                        "pre_scale_peak": "",
                        "anti_clip_gain": "",
                        "pre_scale_clipped_samples": "",
                        "stored_peak": "",
                        "clipped_sample_count": 0,
                    }
                )

            clean = read_audio(str(clean_path), sr=config.sample_rate)
            for snr in config.snrs:
                target_snr = float(snr)
                item_seed = _stable_seed(config.seed, source_id, target_snr)
                rng = random.Random(item_seed)
                noise_row = noise_rows[rng.randrange(len(noise_rows))]
                noise_path = _artifact_path(noise_row["audio"])
                if int(noise_row["sample_rate"]) != config.sample_rate:
                    raise NoiseProtocolError(
                        f"MUSAN dev sample rate must be {config.sample_rate}: {noise_path}"
                    )
                noise = read_audio(str(noise_path), sr=config.sample_rate)
                fitted_noise = fit_noise(noise, len(clean), rng)
                mixed, measurements = _mix_with_measurements(
                    clean,
                    fitted_noise,
                    target_snr_db=target_snr,
                    peak_limit=config.peak_limit,
                )
                label = _snr_label(target_snr)
                relative = Path(f"snr_{label}") / (
                    f"{_safe_stem(source_id)}_snr{label}.wav"
                )
                staged_path = stage / relative
                final_path = config.output_audio_dir / relative
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    staged_path,
                    mixed,
                    config.sample_rate,
                    format="WAV",
                    subtype="PCM_16",
                )
                stored, stored_rate = sf.read(
                    staged_path, dtype="float32", always_2d=False
                )
                if getattr(stored, "ndim", 1) > 1:
                    stored = stored.mean(axis=1)
                stored_peak = float(np.max(np.abs(stored))) if len(stored) else 0.0
                clipped_count = int(np.count_nonzero(np.abs(stored) >= 1.0))
                audio_sha = sha256_file(staged_path)
                rows.append(
                    {
                        "utt_id": f"{source_id}_snr{label}",
                        "source_utt_id": source_id,
                        "speaker_id": source_row.get("speaker_id", ""),
                        "dataset": source_row.get("dataset", "vivos"),
                        "split": "dev",
                        "condition": "noisy",
                        "audio_path": _display_path(final_path),
                        "audio_sha256": audio_sha,
                        "clean_path": _display_path(clean_path),
                        "clean_audio_sha256": clean_sha,
                        "transcript": transcript,
                        "text_sha256": text_sha,
                        "snr": label,
                        "target_snr_db": f"{target_snr:.6f}",
                        "measured_snr_db": f"{float(measurements['measured_snr_db']):.9f}",
                        "noise_id": noise_row["noise_id"],
                        "noise_type": noise_row["noise_type"],
                        "noise_path": noise_row["audio"],
                        "noise_audio_sha256": noise_row["audio_sha256"],
                        "noise_split": "dev",
                        "seed": item_seed,
                        "sample_rate": int(stored_rate),
                        "duration_seconds": f"{len(stored) / float(stored_rate):.9f}",
                        "pre_scale_peak": f"{float(measurements['pre_scale_peak']):.9f}",
                        "anti_clip_gain": f"{float(measurements['anti_clip_gain']):.9f}",
                        "pre_scale_clipped_samples": int(
                            measurements["pre_scale_clipped_samples"]
                        ),
                        "stored_peak": f"{stored_peak:.9f}",
                        "clipped_sample_count": clipped_count,
                    }
                )

        rows = sorted(rows, key=lambda row: str(row["utt_id"]))
        audit_rows = _builder_audit(
            rows,
            source_count=len(source_rows),
            snrs=config.snrs,
            include_clean=config.include_clean,
            audio_dir=config.output_audio_dir,
        )
        manifest_payload = _jsonl_bytes(rows)
        audit_payload = _csv_bytes(audit_rows, NOISY_DEV_AUDIT_COLUMNS)
        noisy_inventory = [
            {"utt_id": row["utt_id"], "audio_sha256": row["audio_sha256"]}
            for row in rows
            if row["condition"] == "noisy"
        ]
        lock = {
            "protocol_version": NOISY_DEV_PROTOCOL_VERSION,
            "status": "LOCKED",
            "selection_eligible": True,
            "final_test_eligible": False,
            "source_dev": {
                "manifest": _display_path(config.source_dev_manifest),
                "manifest_sha256": config.source_dev_sha256.casefold(),
                "utterance_count": len(source_rows),
            },
            "noise": {
                "split_lock": _display_path(config.noise_split_lock),
                "split_lock_sha256": noise_lock_sha256,
                "registry_manifest_sha256": noise_lock["registry"]["manifest_sha256"],
                "dev_manifest": noise_lock["splits"]["dev"]["manifest"],
                "dev_manifest_sha256": noise_lock["splits"]["dev"]["manifest_sha256"],
                "partition": "dev",
            },
            "builder": {
                "params": builder_params,
                "params_sha256": builder_sha256,
            },
            "schema": list(NOISY_DEV_COLUMNS),
            "output": {
                "manifest": _display_path(config.output_manifest),
                "manifest_sha256": _sha256_bytes(manifest_payload),
                "row_count": len(rows),
                "clean_row_count": sum(row["condition"] == "clean" for row in rows),
                "noisy_row_count": sum(row["condition"] == "noisy" for row in rows),
                "audio_dir": _display_path(config.output_audio_dir),
                "audio_inventory_sha256": _canonical_sha256(noisy_inventory),
            },
            "audit": {
                "path": _display_path(config.protocol_audit),
                "sha256": _sha256_bytes(audit_payload),
                "checks": len(audit_rows),
                "failed_checks": 0,
            },
        }
        lock_payload = (
            json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        _commit_noisy_dev_transaction(
            staged_audio_dir=stage,
            output_audio_dir=config.output_audio_dir,
            payloads={
                config.output_manifest: manifest_payload,
                config.protocol_audit: audit_payload,
                config.protocol_lock: lock_payload,
            },
            overwrite=overwrite,
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {"status": "written", "rows": len(rows), "lock": lock}
