from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "vivos_exposure_aware_split_v2"
SPLIT_ALGORITHM = "sha256_ranked_speaker_holdout_v1"
SPLIT_RANK_NAMESPACE = "vivos_speaker_split_v1"
TEST_PARTITION_ALGORITHM = "legacy_exposure_complement_v1"
DEFAULT_VIVOS_ROOT = "data/raw/vivos"
DEFAULT_MANIFEST_DIR = "data/manifests/paper_v2"
DEFAULT_PROTOCOL_DIR = "outputs/paper_v2/protocol"
DEFAULT_LEGACY_BENCHMARK_MANIFEST = "outputs/benchmark/benchmark_manifest.csv"
AUDIT_COLUMNS = [
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
]
EXPOSURE_COLUMNS = [
    "protocol_version",
    "source_utt_id",
    "speaker_id",
    "official_split",
    "exposure_status",
    "audio_sha256",
    "text_sha256",
    "evidence_manifest",
    "evidence_manifest_sha256",
    "evidence_replica_count",
    "evidence_conditions",
    "evidence_snrs",
]


class VivosProtocolError(ValueError):
    """Raised when a VIVOS split cannot be locked without leakage."""


@dataclass(frozen=True)
class SourceUtterance:
    official_split: str
    speaker_id: str
    utt_id: str
    audio_path: Path
    text: str
    audio_sha256: str
    text_sha256: str

    def manifest_row(self, split: str) -> dict[str, object]:
        return {
            "audio": _display_path(self.audio_path),
            "text": self.text,
            "utt_id": self.utt_id,
            "source_utt_id": self.utt_id,
            "speaker_id": self.speaker_id,
            "dataset": "vivos",
            "split": split,
            "official_split": self.official_split,
            "condition": "clean",
            "snr": "clean",
            "noise_type": "clean",
            "audio_sha256": self.audio_sha256,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True)
class LegacyExposureEvidence:
    manifest_path: Path
    manifest_sha256: str
    row_count: int
    source_details: Mapping[str, tuple[int, tuple[str, ...], tuple[str, ...]]]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def locate_official_root(value: str | Path) -> Path:
    requested = Path(value)
    if not requested.is_dir():
        raise VivosProtocolError(f"VIVOS root does not exist: {requested}")
    candidates: set[Path] = set()

    def is_layout(path: Path) -> bool:
        return (
            (path / "train" / "prompts.txt").is_file()
            and (path / "test" / "prompts.txt").is_file()
        )

    if is_layout(requested):
        candidates.add(requested.resolve())
    for prompts in requested.rglob("prompts.txt"):
        if prompts.parent.name not in {"train", "test"}:
            continue
        candidate = prompts.parent.parent
        if is_layout(candidate):
            candidates.add(candidate.resolve())
    if not candidates:
        raise VivosProtocolError(
            f"Could not find one official VIVOS train/test layout under {requested}"
        )
    if len(candidates) != 1:
        raise VivosProtocolError(
            "Ambiguous VIVOS layouts found: "
            + ", ".join(sorted(_display_path(path) for path in candidates))
        )
    return next(iter(candidates))


def read_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise VivosProtocolError(f"Prompt file must be UTF-8: {path}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise VivosProtocolError(f"Malformed prompt at {path}:{line_number}")
        utt_id, text = parts[0], parts[1].strip()
        if utt_id in prompts:
            raise VivosProtocolError(f"Duplicate prompt ID {utt_id!r} in {path}")
        prompts[utt_id] = text
    if not prompts:
        raise VivosProtocolError(f"Prompt file is empty: {path}")
    return prompts


def _speaker_from_wave_path(wave_path: Path, waves_dir: Path, utt_id: str) -> str:
    relative = wave_path.relative_to(waves_dir)
    if len(relative.parts) < 2:
        raise VivosProtocolError(
            f"VIVOS WAV must be nested under a speaker directory: {wave_path}"
        )
    speaker_id = relative.parts[0]
    if not utt_id.startswith(speaker_id + "_"):
        raise VivosProtocolError(
            f"Utterance ID {utt_id!r} does not match speaker directory {speaker_id!r}"
        )
    return speaker_id


def load_official_split(official_root: Path, split: str) -> list[SourceUtterance]:
    if split not in {"train", "test"}:
        raise ValueError("official split must be train or test")
    split_dir = official_root / split
    prompts_path = split_dir / "prompts.txt"
    waves_dir = split_dir / "waves"
    if not waves_dir.is_dir():
        raise VivosProtocolError(f"Missing VIVOS waves directory: {waves_dir}")
    prompts = read_prompts(prompts_path)
    wavs = sorted(
        (path for path in waves_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.as_posix().casefold(),
    )
    if not wavs:
        raise VivosProtocolError(f"No WAV files found in {waves_dir}")
    wav_by_id: dict[str, Path] = {}
    for wav_path in wavs:
        utt_id = wav_path.stem
        if utt_id in wav_by_id:
            raise VivosProtocolError(f"Duplicate WAV utterance ID {utt_id!r} in {waves_dir}")
        wav_by_id[utt_id] = wav_path
    missing_audio = sorted(set(prompts).difference(wav_by_id))
    missing_prompts = sorted(set(wav_by_id).difference(prompts))
    if missing_audio or missing_prompts:
        raise VivosProtocolError(
            f"Prompt/WAV inventory mismatch for official {split}: "
            f"missing_audio={missing_audio[:5]}, missing_prompts={missing_prompts[:5]}"
        )

    utterances: list[SourceUtterance] = []
    for utt_id in sorted(prompts):
        audio_path = wav_by_id[utt_id]
        speaker_id = _speaker_from_wave_path(audio_path, waves_dir, utt_id)
        text = prompts[utt_id]
        utterances.append(
            SourceUtterance(
                official_split=split,
                speaker_id=speaker_id,
                utt_id=utt_id,
                audio_path=audio_path,
                text=text,
                audio_sha256=_sha256_file(audio_path),
                text_sha256=_sha256_text(text),
            )
        )
    return utterances


def _normalize_evidence_snr(value: object) -> str:
    text = str(value).strip().casefold()
    if text == "clean":
        return "clean"
    try:
        numeric = float(text)
    except ValueError as exc:
        raise VivosProtocolError(f"Invalid SNR in legacy benchmark evidence: {value!r}") from exc
    if not math.isfinite(numeric):
        raise VivosProtocolError(f"Invalid SNR in legacy benchmark evidence: {value!r}")
    return f"{numeric:g}"


def load_legacy_exposure_evidence(
    path: str | Path,
    *,
    official_test: Sequence[SourceUtterance],
    expected_source_count: int,
) -> LegacyExposureEvidence:
    """Load the historical benchmark as immutable evidence of test exposure.

    The legacy benchmark contains one clean and four noisy replicas for every
    selected official-test utterance.  Only ``source_utt_id`` determines the
    exposure partition; derived noisy replica IDs are never treated as new
    utterances.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise VivosProtocolError(
            f"Legacy benchmark evidence does not exist: {manifest_path}"
        )
    if expected_source_count < 1:
        raise VivosProtocolError("expected legacy-exposed source count must be positive")
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "utt_id",
                "dataset",
                "split",
                "condition",
                "snr",
                "source_utt_id",
            }
            missing = sorted(required.difference(reader.fieldnames or ()))
            if missing:
                raise VivosProtocolError(
                    f"Legacy benchmark evidence is missing columns: {missing}"
                )
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise VivosProtocolError(
            f"Legacy benchmark evidence must be UTF-8: {manifest_path}"
        ) from exc
    if not rows:
        raise VivosProtocolError(f"Legacy benchmark evidence is empty: {manifest_path}")

    official_by_id = {row.utt_id: row for row in official_test}
    details: dict[str, list[tuple[str, str, str]]] = {}
    replica_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        source_utt_id = str(row.get("source_utt_id", "")).strip()
        replica_id = str(row.get("utt_id", "")).strip()
        dataset = str(row.get("dataset", "")).strip().casefold()
        split = str(row.get("split", "")).strip().casefold()
        condition = str(row.get("condition", "")).strip().casefold()
        snr = _normalize_evidence_snr(row.get("snr", ""))
        if not source_utt_id or not replica_id:
            raise VivosProtocolError(
                f"Legacy benchmark evidence has an empty utterance ID at row {row_number}"
            )
        if replica_id in replica_ids:
            raise VivosProtocolError(
                f"Duplicate replica ID in legacy benchmark evidence: {replica_id}"
            )
        replica_ids.add(replica_id)
        if dataset != "vivos" or split != "test":
            raise VivosProtocolError(
                "Legacy benchmark evidence must contain only VIVOS official-test rows: "
                f"row={row_number}, dataset={dataset!r}, split={split!r}"
            )
        if source_utt_id not in official_by_id:
            raise VivosProtocolError(
                f"Legacy benchmark source is not in official VIVOS test: {source_utt_id}"
            )
        if condition not in {"clean", "noisy"}:
            raise VivosProtocolError(
                f"Invalid legacy benchmark condition at row {row_number}: {condition!r}"
            )
        if (snr == "clean") != (condition == "clean"):
            raise VivosProtocolError(
                f"Legacy benchmark condition/SNR mismatch at row {row_number}"
            )
        details.setdefault(source_utt_id, []).append((replica_id, condition, snr))

    if len(details) != expected_source_count:
        raise VivosProtocolError(
            "Legacy benchmark exposed-source count differs from the locked expectation: "
            f"{len(details)} != {expected_source_count}"
        )
    expected_snrs = {"clean", "20", "10", "5", "0"}
    source_details: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {}
    for source_utt_id, replicas in sorted(details.items()):
        conditions = tuple(sorted({condition for _, condition, _ in replicas}))
        snrs = {snr for _, _, snr in replicas}
        if len(replicas) != 5 or snrs != expected_snrs or set(conditions) != {
            "clean",
            "noisy",
        }:
            raise VivosProtocolError(
                "Every legacy-exposed source must have exactly clean/20/10/5/0 "
                f"replicas: {source_utt_id}"
            )
        source_details[source_utt_id] = (
            len(replicas),
            conditions,
            ("clean", "20", "10", "5", "0"),
        )
    return LegacyExposureEvidence(
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        row_count=len(rows),
        source_details=source_details,
    )


def select_dev_speakers(
    train_rows: Sequence[SourceUtterance],
    *,
    seed: int,
    dev_speaker_fraction: float,
) -> tuple[str, ...]:
    if seed < 0:
        raise VivosProtocolError("seed must be non-negative")
    if not 0.0 < dev_speaker_fraction < 1.0:
        raise VivosProtocolError("dev_speaker_fraction must be between 0 and 1")
    speakers = sorted({row.speaker_id for row in train_rows})
    if len(speakers) < 2:
        raise VivosProtocolError("At least two official-train speakers are required")
    dev_count = max(1, math.ceil(len(speakers) * dev_speaker_fraction))
    if dev_count >= len(speakers):
        raise VivosProtocolError("Speaker fraction leaves no speakers for training")

    def rank(speaker_id: str) -> tuple[str, str]:
        # Keep the already-reviewed train/dev assignment stable while the
        # official-test protocol gains an exposure-aware partition.
        payload = f"{SPLIT_RANK_NAMESPACE}|seed={seed}|speaker={speaker_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), speaker_id

    return tuple(sorted(sorted(speakers, key=rank)[:dev_count]))


def validate_replica_split_consistency(rows: Iterable[Mapping[str, object]]) -> None:
    splits_by_source: dict[str, set[str]] = {}
    for row in rows:
        source_id = str(row.get("source_utt_id") or row.get("utt_id") or "").strip()
        split = str(row.get("split") or "").strip()
        if not source_id or not split:
            raise VivosProtocolError("Every replica row must have source_utt_id/utt_id and split")
        splits_by_source.setdefault(source_id, set()).add(split)
    leaked = {
        source_id: sorted(splits)
        for source_id, splits in splits_by_source.items()
        if len(splits) != 1
    }
    if leaked:
        first = next(iter(sorted(leaked.items())))
        raise VivosProtocolError(
            f"Replicas of one utterance cross split boundaries: {first[0]} -> {first[1]}"
        )


def _entity_set(rows: Sequence[SourceUtterance], entity: str) -> set[str]:
    if entity == "speaker":
        return {row.speaker_id for row in rows}
    if entity == "utterance":
        return {row.utt_id for row in rows}
    if entity == "audio_sha256":
        return {row.audio_sha256 for row in rows}
    raise ValueError(f"Unsupported audit entity: {entity}")


def _audit_row(
    check_id: str,
    entity: str,
    split_a: str,
    split_b: str,
    values_a: set[str],
    values_b: set[str],
    *,
    require_equal: bool = False,
) -> dict[str, object]:
    difference = values_a.symmetric_difference(values_b) if require_equal else values_a & values_b
    passed = not difference
    return {
        "protocol_version": PROTOCOL_VERSION,
        "check_id": check_id,
        "entity": entity,
        "split_a": split_a,
        "split_b": split_b,
        "status": "PASS" if passed else "FAIL",
        "overlap_count": len(difference),
        "total_a": len(values_a),
        "total_b": len(values_b),
        "details": "none" if passed else ";".join(sorted(difference)[:20]),
    }


def build_split_audit(
    official_train: Sequence[SourceUtterance],
    official_test: Sequence[SourceUtterance],
    train_rows: Sequence[SourceUtterance],
    dev_rows: Sequence[SourceUtterance],
    legacy_exposed_rows: Sequence[SourceUtterance],
    test_locked_rows: Sequence[SourceUtterance],
    evidence_source_ids: set[str],
) -> list[dict[str, object]]:
    named = {
        "train": train_rows,
        "dev": dev_rows,
        "official_test": official_test,
        "test_legacy_exposed": legacy_exposed_rows,
        "test_locked": test_locked_rows,
    }
    audit: list[dict[str, object]] = []
    for entity in ("speaker", "utterance", "audio_sha256"):
        for split_a, split_b in (
            ("train", "dev"),
            ("train", "official_test"),
            ("dev", "official_test"),
        ):
            audit.append(
                _audit_row(
                    f"no_{entity}_overlap_{split_a}_{split_b}",
                    entity,
                    split_a,
                    split_b,
                    _entity_set(named[split_a], entity),
                    _entity_set(named[split_b], entity),
                )
            )
    for entity in ("utterance", "audio_sha256"):
        audit.append(
            _audit_row(
                f"no_{entity}_overlap_test_legacy_exposed_test_locked",
                entity,
                "test_legacy_exposed",
                "test_locked",
                _entity_set(named["test_legacy_exposed"], entity),
                _entity_set(named["test_locked"], entity),
            )
        )
    train_partition = {row.utt_id for row in train_rows} | {row.utt_id for row in dev_rows}
    audit.append(
        _audit_row(
            "official_train_partition_identity",
            "utterance",
            "official_train",
            "train_plus_dev",
            {row.utt_id for row in official_train},
            train_partition,
            require_equal=True,
        )
    )
    audit.append(
        _audit_row(
            "official_test_partition_identity",
            "utterance",
            "official_test",
            "legacy_exposed_plus_test_locked",
            {row.utt_id for row in official_test},
            {
                row.utt_id
                for row in (*legacy_exposed_rows, *test_locked_rows)
            },
            require_equal=True,
        )
    )
    audit.append(
        _audit_row(
            "legacy_exposure_evidence_identity",
            "utterance",
            "legacy_benchmark_evidence",
            "test_legacy_exposed",
            evidence_source_ids,
            {row.utt_id for row in legacy_exposed_rows},
            require_equal=True,
        )
    )
    manifest_rows = [
        *(row.manifest_row("train") for row in train_rows),
        *(row.manifest_row("dev") for row in dev_rows),
        *(row.manifest_row("legacy_exposed") for row in legacy_exposed_rows),
        *(row.manifest_row("test") for row in test_locked_rows),
    ]
    try:
        validate_replica_split_consistency(manifest_rows)
        status, details = "PASS", "none"
    except VivosProtocolError as exc:
        status, details = "FAIL", str(exc)
    audit.append(
        {
            "protocol_version": PROTOCOL_VERSION,
            "check_id": "replica_family_split_uniqueness",
            "entity": "source_utt_id",
            "split_a": "all",
            "split_b": "all",
            "status": status,
            "overlap_count": 0 if status == "PASS" else 1,
            "total_a": len(manifest_rows),
            "total_b": len({row["source_utt_id"] for row in manifest_rows}),
            "details": details,
        }
    )
    failures = [row for row in audit if row["status"] != "PASS"]
    if failures:
        raise VivosProtocolError(
            "Split leakage audit failed: "
            + "; ".join(str(row["check_id"]) for row in failures)
        )
    return audit


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    return text.encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=AUDIT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _exposure_csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EXPOSURE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _source_ids_sha256(source_ids: Iterable[str]) -> str:
    payload = "".join(f"{source_id}\n" for source_id in sorted(source_ids))
    return _sha256_text(payload)


def _inventory_sha256(rows: Sequence[SourceUtterance]) -> str:
    inventory = [
        {
            "official_split": row.official_split,
            "speaker_id": row.speaker_id,
            "utt_id": row.utt_id,
            "audio_sha256": row.audio_sha256,
            "text_sha256": row.text_sha256,
        }
        for row in sorted(rows, key=lambda item: item.utt_id)
    ]
    payload = json.dumps(inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def build_protocol_outputs(
    official_root: Path,
    *,
    manifest_dir: Path,
    protocol_dir: Path,
    legacy_benchmark_manifest: Path,
    expected_legacy_exposed: int,
    seed: int,
    dev_speaker_fraction: float,
) -> dict[Path, bytes]:
    official_train = load_official_split(official_root, "train")
    official_test = load_official_split(official_root, "test")
    exposure_evidence = load_legacy_exposure_evidence(
        legacy_benchmark_manifest,
        official_test=official_test,
        expected_source_count=expected_legacy_exposed,
    )
    legacy_source_ids = set(exposure_evidence.source_details)
    dev_speakers = set(
        select_dev_speakers(
            official_train,
            seed=seed,
            dev_speaker_fraction=dev_speaker_fraction,
        )
    )
    train_rows = sorted(
        (row for row in official_train if row.speaker_id not in dev_speakers),
        key=lambda row: (row.speaker_id, row.utt_id),
    )
    dev_rows = sorted(
        (row for row in official_train if row.speaker_id in dev_speakers),
        key=lambda row: (row.speaker_id, row.utt_id),
    )
    legacy_exposed_rows = sorted(
        (row for row in official_test if row.utt_id in legacy_source_ids),
        key=lambda row: (row.speaker_id, row.utt_id),
    )
    test_rows = sorted(
        (row for row in official_test if row.utt_id not in legacy_source_ids),
        key=lambda row: (row.speaker_id, row.utt_id),
    )
    if not test_rows:
        raise VivosProtocolError(
            "Legacy exposure leaves no unseen official-test utterances to seal"
        )
    audit_rows = build_split_audit(
        official_train,
        official_test,
        train_rows,
        dev_rows,
        legacy_exposed_rows,
        test_rows,
        legacy_source_ids,
    )

    manifest_paths = {
        "train": manifest_dir / "vivos_train.jsonl",
        "dev": manifest_dir / "vivos_dev.jsonl",
        "test_legacy_exposed": manifest_dir / "vivos_test_legacy_exposed.jsonl",
        "test_locked": manifest_dir / "vivos_test_locked.jsonl",
    }
    manifest_payloads = {
        "train": _jsonl_bytes([row.manifest_row("train") for row in train_rows]),
        "dev": _jsonl_bytes([row.manifest_row("dev") for row in dev_rows]),
        "test_legacy_exposed": _jsonl_bytes(
            [row.manifest_row("legacy_exposed") for row in legacy_exposed_rows]
        ),
        "test_locked": _jsonl_bytes([row.manifest_row("test") for row in test_rows]),
    }
    audit_path = protocol_dir / "split_audit.csv"
    audit_payload = _csv_bytes(audit_rows)
    exposure_path = protocol_dir / "legacy_test_exposure.csv"
    official_test_by_id = {row.utt_id: row for row in official_test}
    exposure_rows: list[dict[str, object]] = []
    for source_utt_id in sorted(legacy_source_ids):
        source = official_test_by_id[source_utt_id]
        replica_count, conditions, snrs = exposure_evidence.source_details[
            source_utt_id
        ]
        exposure_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "source_utt_id": source_utt_id,
                "speaker_id": source.speaker_id,
                "official_split": "test",
                "exposure_status": "legacy_exposed",
                "audio_sha256": source.audio_sha256,
                "text_sha256": source.text_sha256,
                "evidence_manifest": _display_path(
                    exposure_evidence.manifest_path
                ),
                "evidence_manifest_sha256": exposure_evidence.manifest_sha256,
                "evidence_replica_count": replica_count,
                "evidence_conditions": ";".join(conditions),
                "evidence_snrs": ";".join(snrs),
            }
        )
    exposure_payload = _exposure_csv_bytes(exposure_rows)
    split_details = {
        "train": (train_rows, sorted({row.speaker_id for row in train_rows})),
        "dev": (dev_rows, sorted({row.speaker_id for row in dev_rows})),
        "test_legacy_exposed": (
            legacy_exposed_rows,
            sorted({row.speaker_id for row in legacy_exposed_rows}),
        ),
        "test_locked": (test_rows, sorted({row.speaker_id for row in test_rows})),
    }
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "vivos",
        "seed": seed,
        "dev_speaker_fraction": dev_speaker_fraction,
        "split_algorithm": SPLIT_ALGORITHM,
        "split_rank_namespace": SPLIT_RANK_NAMESPACE,
        "test_partition_algorithm": TEST_PARTITION_ALGORITHM,
        "hash_algorithm": "sha256_raw_bytes",
        "official_test": {
            "status": "SEALED",
            "selection_eligible": False,
            "sealed_partition": "test_locked",
            "legacy_exposed_partition": "test_legacy_exposed",
            "legacy_exposed_utterance_count": len(legacy_exposed_rows),
            "unseen_locked_utterance_count": len(test_rows),
            "unlock_condition": "method_and_lambda_decision_locked",
            "exposure_evidence": {
                "benchmark_manifest": _display_path(
                    exposure_evidence.manifest_path
                ),
                "benchmark_manifest_sha256": exposure_evidence.manifest_sha256,
                "benchmark_row_count": exposure_evidence.row_count,
                "source_utterance_count": len(legacy_exposed_rows),
                "source_utt_ids_sha256": _source_ids_sha256(
                    legacy_source_ids
                ),
                "registry": _display_path(exposure_path),
                "registry_sha256": _sha256_bytes(exposure_payload),
            },
        },
        "source": {
            "layout": "official_train_and_test",
            "train_prompts_sha256": _sha256_file(official_root / "train" / "prompts.txt"),
            "test_prompts_sha256": _sha256_file(official_root / "test" / "prompts.txt"),
            "official_train_inventory_sha256": _inventory_sha256(official_train),
            "official_test_inventory_sha256": _inventory_sha256(official_test),
        },
        "splits": {
            name: {
                "manifest": _display_path(manifest_paths[name]),
                "manifest_sha256": _sha256_bytes(manifest_payloads[name]),
                "utterance_count": len(rows),
                "speaker_count": len(speakers),
                "speakers": speakers,
            }
            for name, (rows, speakers) in split_details.items()
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
        manifest_paths["train"]: manifest_payloads["train"],
        manifest_paths["dev"]: manifest_payloads["dev"],
        manifest_paths["test_legacy_exposed"]: manifest_payloads[
            "test_legacy_exposed"
        ],
        manifest_paths["test_locked"]: manifest_payloads["test_locked"],
        protocol_dir / "split_lock.json": lock_payload,
        audit_path: audit_payload,
        exposure_path: exposure_payload,
    }


def write_locked_outputs(payloads: Mapping[Path, bytes], *, overwrite: bool) -> str:
    existing = [path for path in payloads if path.exists()]
    if existing and len(existing) == len(payloads):
        identical = all(path.read_bytes() == payload for path, payload in payloads.items())
        if identical:
            return "verified_existing"
    if existing and not overwrite:
        raise VivosProtocolError(
            "Protocol outputs already exist but do not exactly match this source/config; "
            "refusing to mutate the lock without --overwrite: "
            + ", ".join(_display_path(path) for path in existing)
        )

    temporary_paths: dict[Path, Path] = {}
    backup_paths: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            if temporary.exists():
                temporary.unlink()
            temporary.write_bytes(payload)
            temporary_paths[path] = temporary
            if path.exists():
                backup = path.with_name(f".{path.name}.{os.getpid()}.backup")
                if backup.exists():
                    raise VivosProtocolError(
                        f"Stale protocol backup requires inspection: {backup}"
                    )
                backup_paths[path] = backup
        for path, backup in backup_paths.items():
            path.rename(backup)
        try:
            for path, temporary in temporary_paths.items():
                temporary.rename(path)
                committed.append(path)
        except Exception:
            for path in reversed(committed):
                if path.exists():
                    path.unlink()
            for path, backup in backup_paths.items():
                if backup.exists() and not path.exists():
                    backup.rename(path)
            raise
        for backup in backup_paths.values():
            if backup.exists():
                backup.unlink()
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()
        for path, backup in backup_paths.items():
            if backup.exists() and not path.exists():
                backup.rename(path)
    return "written"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create speaker-disjoint VIVOS train/dev splits and seal the "
            "official-test complement not exposed by the legacy benchmark."
        )
    )
    parser.add_argument(
        "--vivos-root",
        "--vivos_root",
        dest="vivos_root",
        default=DEFAULT_VIVOS_ROOT,
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default=DEFAULT_MANIFEST_DIR,
    )
    parser.add_argument("--protocol-dir", default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument(
        "--legacy-benchmark-manifest",
        default=DEFAULT_LEGACY_BENCHMARK_MANIFEST,
        help="Historical benchmark used as immutable evidence of exposed test IDs.",
    )
    parser.add_argument(
        "--expected-legacy-exposed",
        type=int,
        default=300,
        help="Fail unless the evidence contains exactly this many source IDs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-speaker-fraction", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        official_root = locate_official_root(args.vivos_root)
        payloads = build_protocol_outputs(
            official_root,
            manifest_dir=Path(args.out_dir),
            protocol_dir=Path(args.protocol_dir),
            legacy_benchmark_manifest=Path(args.legacy_benchmark_manifest),
            expected_legacy_exposed=args.expected_legacy_exposed,
            seed=args.seed,
            dev_speaker_fraction=args.dev_speaker_fraction,
        )
        status = write_locked_outputs(payloads, overwrite=args.overwrite)
    except (OSError, VivosProtocolError) as exc:
        parser.error(str(exc))
    print(f"VIVOS paper-v2 protocol: {status}")
    for path in payloads:
        print(_display_path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
