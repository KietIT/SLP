from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


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

PROVENANCE_VERSION = "paper_v2_zero_shot_prediction_v1"
RESUME_VERSION = "paper_v2_zero_shot_resume_v2"
RECOVERY_VERSION = "paper_v2_zero_shot_recovery_v1"
DEFAULT_CONFIG = Path("configs/paper_v2_zero_shot.yaml")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_HEX_DIGITS = frozenset("0123456789abcdef")


class ZeroShotProtocolError(ValueError):
    """Raised when a paper-v2 zero-shot run is not fully authorized."""


def _portable_repository_path(value: object, *, label: str) -> str:
    """Return one canonical repository-relative POSIX path or fail closed."""

    text = value.as_posix() if isinstance(value, Path) else str(value).strip()
    if not text:
        raise ZeroShotProtocolError(f"{label} is required")
    windows_path = PureWindowsPath(text)
    posix_path = PurePosixPath(text)
    if (
        "\\" in text
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ZeroShotProtocolError(
            f"{label} must be a portable repository-relative POSIX path: {text!r}"
        )
    canonical = posix_path.as_posix()
    if text != canonical or any(":" in part for part in posix_path.parts):
        raise ZeroShotProtocolError(
            f"{label} is not a canonical portable POSIX path: {text!r}"
        )
    root = REPOSITORY_ROOT.resolve()
    resolved = root.joinpath(*posix_path.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ZeroShotProtocolError(
            f"{label} escapes the repository root: {text!r}"
        ) from exc
    return canonical


def _repository_path(value: object, *, label: str) -> Path:
    portable = _portable_repository_path(value, label=label)
    return REPOSITORY_ROOT.joinpath(*PurePosixPath(portable).parts)


def _assert_no_absolute_path_strings(value: Any, *, label: str) -> None:
    """Defence in depth: portable formal JSON must not disclose local paths."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_path_strings(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_absolute_path_strings(item, label=f"{label}[{index}]")
    elif isinstance(value, str) and (
        PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    ):
        raise ZeroShotProtocolError(
            f"{label} contains a forbidden absolute local path"
        )


@dataclass(frozen=True)
class AuthorizationEvidence:
    split_lock_sha256: str
    decision_lock_sha256: str
    benchmark_lock_sha256: str
    manifest_sha256: str
    manifest_num_rows: int
    source_test_manifest_sha256: str
    benchmark_lock_protocol_version: str


@dataclass
class LoadedZeroShotModel:
    processor: Any
    model: Any
    device: Any
    dtype_name: str
    torch_module: Any
    snapshot_path: str
    snapshot_sha256: str
    model_fingerprint_sha256: str
    processor_fingerprint_sha256: str
    runtime_environment: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON contracts may not contain NaN or infinity")
        return value
    return str(value)


def is_sha256(value: object) -> bool:
    text = str(value).strip().casefold()
    return len(text) == 64 and all(character in _HEX_DIGITS for character in text)


def is_immutable_revision(value: object) -> bool:
    text = str(value).strip().casefold()
    return len(text) in {40, 64} and all(
        character in _HEX_DIGITS for character in text
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise ZeroShotProtocolError(
            f"{label} must be a concrete 64-character SHA-256, not a placeholder"
        )
    return str(value).strip().casefold()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ZeroShotProtocolError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ZeroShotProtocolError(f"{label} must be an object: {path}")
    return value


def load_suite_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return _load_object(
        _repository_path(path, label="config_path"),
        label="Zero-shot suite config",
    )


def _selected_model_specs(
    config: Mapping[str, Any], model_keys: Sequence[str] | None
) -> list[dict[str, Any]]:
    raw_models = config.get("models")
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise ZeroShotProtocolError("models must be a non-empty mapping")
    selected = list(model_keys or raw_models.keys())
    if not selected:
        raise ZeroShotProtocolError("At least one zero-shot model must be selected")
    if len(selected) != len(set(selected)):
        raise ZeroShotProtocolError("Selected model keys must be unique")
    unknown = [key for key in selected if key not in raw_models]
    if unknown:
        raise ZeroShotProtocolError(f"Unknown zero-shot model keys: {unknown}")

    specs: list[dict[str, Any]] = []
    filenames: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for key in selected:
        raw = raw_models[key]
        if not isinstance(raw, Mapping):
            raise ZeroShotProtocolError(f"models.{key} must be an object")
        revision = str(raw.get("revision", "")).strip().casefold()
        if not is_immutable_revision(revision):
            raise ZeroShotProtocolError(
                f"models.{key}.revision must be a concrete immutable 40/64-hex "
                "revision; resolve and lock it before formal inference"
            )
        repo_id = str(raw.get("repo_id", "")).strip()
        family = str(raw.get("model", "")).strip()
        size = str(raw.get("model_size", "")).strip()
        filename = str(raw.get("filename", "")).strip()
        if (
            not repo_id
            or "\\" in repo_id
            or PureWindowsPath(repo_id).is_absolute()
            or PurePosixPath(repo_id).is_absolute()
            or family not in {"whisper", "phowhisper"}
        ):
            raise ZeroShotProtocolError(f"Invalid repo/model family for models.{key}")
        if size not in {"tiny", "base", "small"}:
            raise ZeroShotProtocolError(f"Invalid model_size for models.{key}")
        if not filename or Path(filename).name != filename or not filename.endswith(".csv"):
            raise ZeroShotProtocolError(
                f"models.{key}.filename must be a plain .csv filename"
            )
        if filename in filenames or (family, size) in identities:
            raise ZeroShotProtocolError(f"Duplicate output/model identity for {key}")
        filenames.add(filename)
        identities.add((family, size))
        specs.append(
            {
                "key": str(key),
                "repo_id": repo_id,
                "revision": revision,
                "model": family,
                "model_size": size,
                "filename": filename,
            }
        )
    return specs


def validate_suite_config(
    config: Mapping[str, Any], model_keys: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    protocol = config.get("protocol")
    benchmark = config.get("benchmark")
    decoding = config.get("decoding")
    runtime = config.get("runtime")
    if not all(isinstance(item, Mapping) for item in (protocol, benchmark, decoding, runtime)):
        raise ZeroShotProtocolError(
            "protocol, benchmark, decoding, and runtime must all be objects"
        )
    if protocol.get("formal") is not True:
        raise ZeroShotProtocolError("paper-v2 zero-shot inference requires protocol.formal=true")
    if protocol.get("final_test_unlocked") is not True:
        raise ZeroShotProtocolError(
            "Final test is still locked; set protocol.final_test_unlocked=true only "
            "after the reviewed method/lambda decision is LOCKED"
        )
    for field in ("split_lock", "decision_lock"):
        _portable_repository_path(
            protocol.get(field, ""), label=f"protocol.{field}"
        )
    _require_sha256(
        protocol.get("expected_split_lock_sha256"),
        label="protocol.expected_split_lock_sha256",
    )
    _require_sha256(
        protocol.get("expected_decision_lock_sha256"),
        label="protocol.expected_decision_lock_sha256",
    )
    for field in ("lock", "manifest"):
        _portable_repository_path(
            benchmark.get(field, ""), label=f"benchmark.{field}"
        )
    for field in ("lock_protocol_version", "dataset"):
        if not str(benchmark.get(field, "")).strip():
            raise ZeroShotProtocolError(f"benchmark.{field} is required")
    _require_sha256(
        benchmark.get("expected_lock_sha256"),
        label="benchmark.expected_lock_sha256",
    )
    _require_sha256(
        benchmark.get("expected_manifest_sha256"),
        label="benchmark.expected_manifest_sha256",
    )
    try:
        expected_rows = int(benchmark["expected_rows"])
        seed = int(config["seed"])
        sample_rate = int(decoding["sample_rate"])
        max_new_tokens = int(decoding["max_new_tokens"])
        max_audio_seconds = float(decoding["max_audio_seconds"])
        batch_size = int(runtime["batch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZeroShotProtocolError("Invalid numeric zero-shot suite setting") from exc
    if expected_rows < 1 or seed < 0 or sample_rate < 1 or max_new_tokens < 1:
        raise ZeroShotProtocolError("Counts, seed, sample rate, and token limit are invalid")
    if not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0 or batch_size < 1:
        raise ZeroShotProtocolError("max_audio_seconds and batch_size must be positive")
    if str(runtime.get("device", "")).strip() == "" or str(
        runtime.get("precision", "")
    ).casefold() not in {"fp16", "fp32"}:
        raise ZeroShotProtocolError("runtime.device/precision are invalid")
    if not isinstance(runtime.get("local_files_only"), bool):
        raise ZeroShotProtocolError("runtime.local_files_only must be boolean")
    if decoding.get("language") != "vi" or decoding.get("task") != "transcribe":
        raise ZeroShotProtocolError("Formal Vietnamese decoding requires language=vi, task=transcribe")
    if decoding.get("do_sample") is not False or int(decoding.get("num_beams", 0)) != 1:
        raise ZeroShotProtocolError("Formal zero-shot decoding must be greedy and deterministic")
    if benchmark.get("verify_audio_sha256") is not True:
        raise ZeroShotProtocolError("Formal benchmark access requires verify_audio_sha256=true")
    _portable_repository_path(config.get("output_dir", ""), label="output_dir")
    return _selected_model_specs(config, model_keys)


def authorize_final_benchmark(
    config: Mapping[str, Any],
    *,
    decision_verifier: Callable[..., Mapping[str, Any]] | None = None,
    method_artifact_verifier: Callable[..., Mapping[str, Any]] | None = None,
    noise_verifier: Callable[..., Mapping[str, Any]] | None = None,
    benchmark_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> AuthorizationEvidence:
    """Authorize final-test access without opening the benchmark manifest/model.

    The decision lock is the root of trust.  Its hash-pinned formal method lock
    names the MUSAN split lock; the complete final-benchmark verifier then binds
    the builder, schema, source test, MUSAN-test inventory, and audit.  Keeping
    this transitive verification here prevents a self-consistent but fabricated
    final manifest/lock pair from reaching inference.
    """

    protocol = config["protocol"]
    benchmark = config["benchmark"]
    split_path = _repository_path(
        protocol["split_lock"], label="protocol.split_lock"
    )
    decision_path = _repository_path(
        protocol["decision_lock"], label="protocol.decision_lock"
    )
    benchmark_lock_path = _repository_path(
        benchmark["lock"], label="benchmark.lock"
    )
    expected_split_hash = _require_sha256(
        protocol["expected_split_lock_sha256"],
        label="protocol.expected_split_lock_sha256",
    )
    expected_decision_hash = _require_sha256(
        protocol["expected_decision_lock_sha256"],
        label="protocol.expected_decision_lock_sha256",
    )
    expected_benchmark_hash = _require_sha256(
        benchmark["expected_lock_sha256"], label="benchmark.expected_lock_sha256"
    )
    expected_manifest_hash = _require_sha256(
        benchmark["expected_manifest_sha256"],
        label="benchmark.expected_manifest_sha256",
    )
    if not split_path.is_file() or sha256_file(split_path) != expected_split_hash:
        raise ZeroShotProtocolError("Configured split lock is missing or has changed")
    if not decision_path.is_file() or sha256_file(decision_path) != expected_decision_hash:
        raise ZeroShotProtocolError("Configured method/lambda decision lock is missing or has changed")

    if decision_verifier is None:
        from src.vitonesr.phat.protocol import verify_test_decision_lock

        decision_verifier = verify_test_decision_lock
    decision = dict(
        decision_verifier(
            split_lock_path=split_path,
            decision_lock_path=decision_path,
            verify_checkpoints=False,
        )
    )
    if str(decision.get("split_lock_sha256", "")).casefold() != expected_split_hash:
        raise ZeroShotProtocolError("Decision verifier returned another split lock")
    if str(decision.get("decision_lock_sha256", "")).casefold() != expected_decision_hash:
        raise ZeroShotProtocolError("Decision verifier returned another decision lock")
    source_test_hash = _require_sha256(
        decision.get("test_manifest_sha256"),
        label="decision.test_manifest_sha256",
    )

    # The verified decision hash-pins this method lock.  Validate its canonical
    # identity before using its noise binding as an authorization dependency.
    raw_decision = _load_object(decision_path, label="Method/lambda decision lock")
    method_lock_hash = _require_sha256(
        decision.get("method_lock_sha256"), label="decision.method_lock_sha256"
    )
    method_identity_hash = _require_sha256(
        decision.get("method_identity_sha256"),
        label="decision.method_identity_sha256",
    )
    if str(raw_decision.get("method_lock_sha256", "")).casefold() != method_lock_hash:
        raise ZeroShotProtocolError("Decision verifier and lock disagree on method lock")
    if (
        str(raw_decision.get("method_identity_sha256", "")).casefold()
        != method_identity_hash
    ):
        raise ZeroShotProtocolError("Decision verifier and lock disagree on method identity")
    method_lock_path = _repository_path(
        raw_decision.get("method_lock", ""), label="decision.method_lock"
    )
    if not method_lock_path.is_file() or sha256_file(method_lock_path) != method_lock_hash:
        raise ZeroShotProtocolError("Decision-bound method lock is missing or has changed")
    if method_artifact_verifier is None:
        from src.vitonesr.phat.method_contract import (
            verify_method_artifact_bindings,
        )

        method_artifact_verifier = verify_method_artifact_bindings
    try:
        method_evidence = dict(
            method_artifact_verifier(
                method_lock_path,
                repo_root=REPOSITORY_ROOT,
                formal=True,
            )
        )
    except (ImportError, OSError, ValueError) as exc:
        raise ZeroShotProtocolError(f"Decision-bound method lock is invalid: {exc}") from exc
    if str(method_evidence.get("method_lock_sha256", "")).casefold() != (
        method_lock_hash
    ):
        raise ZeroShotProtocolError("Method artifact verifier returned another lock")
    if method_evidence.get("mode") != "formal":
        raise ZeroShotProtocolError("Final zero-shot evaluation requires a formal method lock")
    if str(method_evidence.get("method_identity_sha256", "")).casefold() != (
        method_identity_hash
    ):
        raise ZeroShotProtocolError("Method lock identity differs from the decision lock")
    artifacts = method_evidence.get("artifacts")
    noise_binding = (
        artifacts.get("noise_split_lock") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(noise_binding, Mapping):
        raise ZeroShotProtocolError("Method lock has no MUSAN split-lock binding")
    noise_lock_hash = _require_sha256(
        noise_binding.get("sha256"), label="method artifacts.noise_split_lock.sha256"
    )
    noise_lock_path = _repository_path(
        noise_binding.get("path", ""),
        label="method artifacts.noise_split_lock.path",
    )
    if not noise_lock_path.is_file() or sha256_file(noise_lock_path) != noise_lock_hash:
        raise ZeroShotProtocolError("Method-bound MUSAN split lock is missing or has changed")
    if noise_verifier is None:
        from src.vitonesr.noise_protocol import verify_noise_split_lock

        noise_verifier = verify_noise_split_lock
    try:
        noise_integrity = dict(noise_verifier(noise_lock_path, verify_audio=False))
    except (OSError, ValueError) as exc:
        raise ZeroShotProtocolError(f"MUSAN split-lock verification failed: {exc}") from exc
    if str(noise_integrity.get("lock_sha256", "")).casefold() != noise_lock_hash:
        raise ZeroShotProtocolError("MUSAN verifier returned another split lock")

    configured_manifest = _repository_path(
        benchmark["manifest"], label="benchmark.manifest"
    )
    lock_version = str(benchmark["lock_protocol_version"])
    if benchmark_verifier is None:
        from src.vitonesr.final_benchmark import verify_final_benchmark_lock

        benchmark_verifier = verify_final_benchmark_lock
    try:
        verified_benchmark = dict(
            benchmark_verifier(
                benchmark_lock_path,
                expected_lock_sha256=expected_benchmark_hash,
                expected_manifest=configured_manifest,
                expected_manifest_sha256=expected_manifest_hash,
                expected_rows=int(benchmark["expected_rows"]),
                split_lock_sha256=expected_split_hash,
                decision_lock_sha256=expected_decision_hash,
                source_test_manifest_sha256=source_test_hash,
                method_lock_sha256=method_lock_hash,
                method_identity_sha256=method_identity_hash,
                noise_split_lock_sha256=noise_lock_hash,
                noise_integrity=noise_integrity,
            )
        )
    except (OSError, ValueError) as exc:
        raise ZeroShotProtocolError(
            f"Final benchmark authorization failed: {exc}"
        ) from exc
    if str(verified_benchmark.get("lock_sha256", "")).casefold() != expected_benchmark_hash:
        raise ZeroShotProtocolError("Final benchmark verifier returned another lock")
    if str(verified_benchmark.get("manifest_sha256", "")).casefold() != expected_manifest_hash:
        raise ZeroShotProtocolError("Final benchmark verifier returned another manifest")
    try:
        locked_rows = int(verified_benchmark["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZeroShotProtocolError("Final benchmark verifier returned invalid row_count") from exc
    if locked_rows != int(benchmark["expected_rows"]):
        raise ZeroShotProtocolError("Final benchmark verifier returned another row count")
    if str(verified_benchmark.get("protocol_version", "")) != lock_version:
        raise ZeroShotProtocolError("Final benchmark verifier returned another protocol version")
    return AuthorizationEvidence(
        split_lock_sha256=expected_split_hash,
        decision_lock_sha256=expected_decision_hash,
        benchmark_lock_sha256=expected_benchmark_hash,
        manifest_sha256=expected_manifest_hash,
        manifest_num_rows=locked_rows,
        source_test_manifest_sha256=source_test_hash,
        benchmark_lock_protocol_version=lock_version,
    )


def _normalise_snr(value: object) -> str:
    text = str(value).strip()
    if not text or text.casefold() == "clean":
        return "clean"
    try:
        number = float(text)
    except ValueError:
        raise ZeroShotProtocolError(f"Invalid SNR label: {value!r}") from None
    if not math.isfinite(number):
        raise ZeroShotProtocolError(f"Invalid SNR label: {value!r}")
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _read_manifest_records(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        if path.suffix.casefold() in {".jsonl", ".json"}:
            records = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ZeroShotProtocolError(
                            f"Manifest line {line_number} must be an object"
                        )
                    records.append(value)
            return records
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ZeroShotProtocolError(f"Invalid UTF-8 benchmark manifest: {path}") from exc
    raise ZeroShotProtocolError(f"Unsupported benchmark manifest format: {path}")


def load_authorized_benchmark(
    config: Mapping[str, Any], evidence: AuthorizationEvidence
) -> list[dict[str, str]]:
    """Open and validate the benchmark only after authorization succeeds."""

    benchmark = config["benchmark"]
    path = _repository_path(benchmark["manifest"], label="benchmark.manifest")
    if not path.is_file():
        raise FileNotFoundError(f"Authorized benchmark manifest is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != evidence.manifest_sha256:
        raise ZeroShotProtocolError("Authorized benchmark manifest has changed")
    raw_rows = _read_manifest_records(path)
    if len(raw_rows) != evidence.manifest_num_rows:
        raise ZeroShotProtocolError(
            f"Benchmark has {len(raw_rows)} rows, expected {evidence.manifest_num_rows}"
        )
    expected_dataset = str(benchmark["dataset"])
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, raw in enumerate(raw_rows, start=1):
        audio = raw.get("audio_path") or raw.get("audio") or raw.get("noisy_path") or raw.get("clean_path")
        reference = raw.get("transcript") or raw.get("text") or raw.get("ref")
        utt_id = str(raw.get("utt_id", "")).strip()
        dataset = str(raw.get("dataset", "")).strip()
        split = str(raw.get("split", "")).strip().casefold()
        audio_sha = str(raw.get("audio_sha256", "")).strip().casefold()
        if not utt_id or utt_id in seen:
            raise ZeroShotProtocolError(
                f"Benchmark row {row_number} has blank/duplicate utt_id: {utt_id!r}"
            )
        if dataset != expected_dataset or split != "test":
            raise ZeroShotProtocolError(
                f"Benchmark row {row_number} must use dataset={expected_dataset}, split=test"
            )
        if not audio or reference is None or not str(reference).strip():
            raise ZeroShotProtocolError(
                f"Benchmark row {row_number} lacks audio or reference"
            )
        if not is_sha256(audio_sha):
            raise ZeroShotProtocolError(
                f"Benchmark row {row_number} lacks an audio SHA-256"
            )
        audio_path = _repository_path(
            audio, label=f"benchmark row {row_number} audio_path"
        )
        if not audio_path.is_file():
            raise FileNotFoundError(f"Benchmark audio is missing: {audio_path}")
        if benchmark["verify_audio_sha256"] is True and sha256_file(audio_path) != audio_sha:
            raise ZeroShotProtocolError(f"Benchmark audio SHA-256 mismatch: {audio_path}")
        snr = _normalise_snr(raw.get("snr", "clean"))
        noise_type = str(raw.get("noise_type", "clean" if snr == "clean" else "")).strip()
        if (snr == "clean") != (noise_type.casefold() == "clean"):
            raise ZeroShotProtocolError(
                f"Benchmark row {row_number} has inconsistent snr/noise_type"
            )
        seen.add(utt_id)
        rows.append(
            {
                "utt_id": utt_id,
                "source_utt_id": str(raw.get("source_utt_id", utt_id)).strip(),
                "dataset": dataset,
                "audio_path": str(audio_path),
                "audio_sha256": audio_sha,
                "snr": snr,
                "noise_type": noise_type,
                "ref": unicodedata.normalize("NFC", str(reference)),
            }
        )
    return rows


def benchmark_rows_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    return canonical_sha256(
        [
            {
                "utt_id": row["utt_id"],
                "source_utt_id": row["source_utt_id"],
                "dataset": row["dataset"],
                "audio_sha256": row["audio_sha256"],
                "snr": row["snr"],
                "noise_type": row["noise_type"],
                "ref": row["ref"],
            }
            for row in rows
        ]
    )


def fingerprint_snapshot(path: str | Path) -> tuple[str, int, int]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Resolved model snapshot is missing: {root}")
    inventory: list[dict[str, Any]] = []
    total_bytes = 0
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total_bytes += size
        inventory.append(
            {
                "path": item.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(item),
            }
        )
    if not inventory:
        raise ZeroShotProtocolError(f"Resolved model snapshot is empty: {root}")
    return canonical_sha256(inventory), len(inventory), total_bytes


def _component_payload(component: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"class": type(component).__name__}
    config = getattr(component, "config", None)
    if config is not None and callable(getattr(config, "to_dict", None)):
        payload["config"] = config.to_dict()
    feature_extractor = getattr(component, "feature_extractor", None)
    if feature_extractor is not None and callable(getattr(feature_extractor, "to_dict", None)):
        payload["feature_extractor"] = feature_extractor.to_dict()
    tokenizer = getattr(component, "tokenizer", None)
    if tokenizer is not None:
        payload["tokenizer_class"] = type(tokenizer).__name__
        payload["tokenizer_init"] = getattr(tokenizer, "init_kwargs", {})
    return payload


def _configure_determinism(torch_module: Any, seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def load_huggingface_model(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
    processor_class: Any | None = None,
    model_class: Any | None = None,
    snapshot_resolver: Callable[..., str] | None = None,
) -> LoadedZeroShotModel:
    """Load one immutable HF revision and fingerprint its resolved snapshot."""

    if torch_module is None:
        import torch as torch_module
    if processor_class is None or model_class is None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor_class = processor_class or WhisperProcessor
        model_class = model_class or WhisperForConditionalGeneration
    if snapshot_resolver is None:
        from huggingface_hub import snapshot_download

        snapshot_resolver = snapshot_download

    revision = str(spec["revision"])
    if not is_immutable_revision(revision):
        raise ZeroShotProtocolError("Model loader refuses a mutable revision")
    runtime = config["runtime"]
    decoding = config["decoding"]
    seed = int(config["seed"])
    _configure_determinism(torch_module, seed)
    local_only = bool(runtime.get("local_files_only", False))
    processor = processor_class.from_pretrained(
        str(spec["repo_id"]),
        revision=revision,
        local_files_only=local_only,
        language=str(decoding["language"]),
        task=str(decoding["task"]),
    )
    model = model_class.from_pretrained(
        str(spec["repo_id"]),
        revision=revision,
        local_files_only=local_only,
    )
    snapshot_path = Path(
        snapshot_resolver(
            repo_id=str(spec["repo_id"]),
            revision=revision,
            local_files_only=True,
        )
    )
    if is_immutable_revision(snapshot_path.name) and snapshot_path.name.casefold() != revision.casefold():
        raise ZeroShotProtocolError(
            "Resolved Hugging Face snapshot commit differs from the locked revision"
        )
    snapshot_sha, file_count, total_bytes = fingerprint_snapshot(snapshot_path)
    model_commit = str(getattr(getattr(model, "config", None), "_commit_hash", "") or "")
    if model_commit and model_commit.casefold() != revision.casefold():
        raise ZeroShotProtocolError("Loaded model config reports another commit hash")

    device_arg = str(runtime.get("device", "auto"))
    device = torch_module.device(
        "cuda" if device_arg == "auto" and torch_module.cuda.is_available() else (
            "cpu" if device_arg == "auto" else device_arg
        )
    )
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = str(runtime.get("precision", "fp32")).casefold()
    if precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError("fp16 inference requires CUDA")
        dtype = torch_module.float16
    elif precision == "fp32":
        dtype = torch_module.float32
    else:
        raise ZeroShotProtocolError(f"Unsupported inference precision: {precision!r}")
    model.config.use_cache = True
    model.to(device=device, dtype=dtype)
    model.eval()
    return LoadedZeroShotModel(
        processor=processor,
        model=model,
        device=device,
        dtype_name=str(dtype),
        torch_module=torch_module,
        snapshot_path=str(snapshot_path),
        snapshot_sha256=snapshot_sha,
        model_fingerprint_sha256=canonical_sha256(
            {
                "repo_id": spec["repo_id"],
                "revision": revision,
                "snapshot_sha256": snapshot_sha,
                "component": _component_payload(model),
            }
        ),
        processor_fingerprint_sha256=canonical_sha256(
            {
                "repo_id": spec["repo_id"],
                "revision": revision,
                "snapshot_sha256": snapshot_sha,
                "component": _component_payload(processor),
            }
        ),
        runtime_environment={
            "device_type": device.type,
            "dtype": str(dtype),
            "torch_version": str(torch_module.__version__),
            "cuda_version": str(torch_module.version.cuda or ""),
            "snapshot_file_count": file_count,
            "snapshot_total_bytes": total_bytes,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "deterministic_algorithms": bool(
                torch_module.are_deterministic_algorithms_enabled()
            ),
        },
    )


def decode_batch(
    loaded: LoadedZeroShotModel,
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[str]:
    from src.vitonesr.noise import read_audio

    decoding = config["decoding"]
    sample_rate = int(decoding["sample_rate"])
    max_samples = int(float(decoding["max_audio_seconds"]) * sample_rate)
    waveforms = [
        read_audio(row["audio_path"], sr=sample_rate)[:max_samples] for row in rows
    ]
    feature_batch = loaded.processor.feature_extractor(
        waveforms,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
    )
    input_features = feature_batch.input_features.to(
        device=loaded.device,
        dtype=getattr(loaded.model, "dtype", None),
    )
    attention_mask = getattr(feature_batch, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=loaded.device)
    generate_kwargs = {
        "max_new_tokens": int(decoding["max_new_tokens"]),
        "language": str(decoding["language"]),
        "task": str(decoding["task"]),
        "do_sample": False,
        "num_beams": 1,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    with loaded.torch_module.inference_mode():
        generated = loaded.model.generate(input_features, **generate_kwargs)
    hypotheses = loaded.processor.batch_decode(generated, skip_special_tokens=True)
    if len(hypotheses) != len(rows):
        raise RuntimeError(
            f"Decoder returned {len(hypotheses)} hypotheses for {len(rows)} inputs"
        )
    return [str(value) for value in hypotheses]


def provenance_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    return path.with_suffix(path.suffix + ".provenance.json")


def resume_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    return path.with_suffix(path.suffix + ".resume.json")


def recovery_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    return path.with_suffix(path.suffix + ".recovery.json")


def _prediction_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=PREDICTION_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in PREDICTION_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = _prediction_csv_bytes(rows)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Atomic prediction write failed integrity check: {path}")
    return expected_sha256


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish JSON while refusing to replace an existing artifact."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite provenance: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"Refusing to overwrite provenance: {path}") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_prediction(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != PREDICTION_COLUMNS:
            raise ZeroShotProtocolError(
                f"Prediction must have the exact 11-column schema: {path}"
            )
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ZeroShotProtocolError(
                    f"Malformed prediction row {row_number}: {path}"
                )
            rows.append(dict(row))
    return rows


def _expected_prediction_prefix(
    benchmark_rows: Sequence[Mapping[str, str]],
    spec: Mapping[str, Any],
    seed: int,
    hypotheses: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "utt_id": benchmark["utt_id"],
            "dataset": benchmark["dataset"],
            "model": spec["model"],
            "model_size": spec["model_size"],
            "train_type": "zero_shot",
            "lambda": "",
            "seed": seed,
            "snr": benchmark["snr"],
            "noise_type": benchmark["noise_type"],
            "ref": benchmark["ref"],
            "hyp": hypothesis,
        }
        for benchmark, hypothesis in zip(benchmark_rows, hypotheses)
    ]


def _validate_prediction_prefix(
    rows: Sequence[Mapping[str, str]],
    benchmark_rows: Sequence[Mapping[str, str]],
    spec: Mapping[str, Any],
    seed: int,
) -> None:
    if len(rows) > len(benchmark_rows):
        raise ZeroShotProtocolError("Resume prediction has more rows than the benchmark")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        benchmark = benchmark_rows[index]
        expected = {
            "utt_id": benchmark["utt_id"],
            "dataset": benchmark["dataset"],
            "model": str(spec["model"]),
            "model_size": str(spec["model_size"]),
            "train_type": "zero_shot",
            "lambda": "",
            "seed": str(seed),
            "snr": benchmark["snr"],
            "noise_type": benchmark["noise_type"],
            "ref": benchmark["ref"],
        }
        if row["utt_id"] in seen:
            raise ZeroShotProtocolError("Resume prediction contains duplicate utt_id values")
        seen.add(row["utt_id"])
        for field, value in expected.items():
            if str(row.get(field, "")) != value:
                raise ZeroShotProtocolError(
                    f"Resume mismatch at row {index + 1}, field {field}: "
                    f"{row.get(field)!r} != {value!r}"
                )


def _run_contract(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    suite_config_sha256: str,
) -> dict[str, Any]:
    protocol = config["protocol"]
    benchmark = config["benchmark"]
    decoding = config["decoding"]
    runtime = config["runtime"]
    return {
        "contract_version": "paper_v2_zero_shot_run_v1",
        "suite_config_sha256": suite_config_sha256,
        "schema": PREDICTION_COLUMNS,
        "model": dict(spec),
        "seed": int(config["seed"]),
        "protocol": {
            field: protocol[field]
            for field in (
                "formal",
                "final_test_unlocked",
                "split_lock",
                "expected_split_lock_sha256",
                "decision_lock",
                "expected_decision_lock_sha256",
            )
        },
        "benchmark": {
            field: benchmark[field]
            for field in (
                "lock_protocol_version",
                "lock",
                "expected_lock_sha256",
                "manifest",
                "expected_manifest_sha256",
                "expected_rows",
                "dataset",
                "verify_audio_sha256",
            )
        },
        "decoding": {
            field: decoding[field]
            for field in (
                "language",
                "task",
                "sample_rate",
                "max_audio_seconds",
                "max_new_tokens",
                "do_sample",
                "num_beams",
            )
        },
        "runtime": {
            field: runtime[field]
            for field in (
                "batch_size",
                "device",
                "precision",
                "local_files_only",
            )
        },
    }


def _authorization_payload(evidence: AuthorizationEvidence) -> dict[str, Any]:
    return {
        "split_lock_sha256": evidence.split_lock_sha256,
        "decision_lock_sha256": evidence.decision_lock_sha256,
        "benchmark_lock_sha256": evidence.benchmark_lock_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "manifest_num_rows": evidence.manifest_num_rows,
        "source_test_manifest_sha256": evidence.source_test_manifest_sha256,
        "benchmark_lock_protocol_version": evidence.benchmark_lock_protocol_version,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return _load_object(path, label=label)


def _validate_resume_state(
    state: Mapping[str, Any],
    *,
    prediction_path: Path,
    row_count: int,
    run_contract_sha256: str,
    selected_rows_sha256: str,
    evidence: AuthorizationEvidence,
    loaded: LoadedZeroShotModel | None,
) -> None:
    if state.get("resume_version") != RESUME_VERSION:
        raise ZeroShotProtocolError("Unsupported zero-shot resume state")
    expected = {
        "run_contract_sha256": run_contract_sha256,
        "selected_rows_sha256": selected_rows_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "benchmark_lock_sha256": evidence.benchmark_lock_sha256,
        "decision_lock_sha256": evidence.decision_lock_sha256,
        "prediction_sha256": sha256_file(prediction_path),
    }
    for field, value in expected.items():
        if str(state.get(field, "")).casefold() != value.casefold():
            raise ZeroShotProtocolError(f"Resume state mismatch: {field}")
    if int(state.get("completed_rows", -1)) != row_count:
        raise ZeroShotProtocolError("Resume state row count does not match prediction")
    if loaded is not None:
        snapshot = state.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise ZeroShotProtocolError("Resume state has no snapshot fingerprint")
        for field, value in (
            ("snapshot_sha256", loaded.snapshot_sha256),
            ("model_fingerprint_sha256", loaded.model_fingerprint_sha256),
            ("processor_fingerprint_sha256", loaded.processor_fingerprint_sha256),
        ):
            if str(snapshot.get(field, "")).casefold() != value.casefold():
                raise ZeroShotProtocolError(f"Resume model snapshot mismatch: {field}")


def _validate_completed_provenance(
    provenance: Mapping[str, Any],
    *,
    prediction_path: Path,
    rows: Sequence[Mapping[str, str]],
    run_contract_sha256: str,
    selected_rows_sha256: str,
    evidence: AuthorizationEvidence,
    spec: Mapping[str, Any],
) -> None:
    if provenance.get("provenance_version") != PROVENANCE_VERSION:
        raise ZeroShotProtocolError("Unsupported completed zero-shot provenance")
    expected = {
        "prediction_sha256": sha256_file(prediction_path),
        "manifest_sha256": evidence.manifest_sha256,
        "benchmark_lock_sha256": evidence.benchmark_lock_sha256,
        "split_lock_sha256": evidence.split_lock_sha256,
        "decision_lock_sha256": evidence.decision_lock_sha256,
        "selected_rows_sha256": selected_rows_sha256,
        "run_contract_sha256": run_contract_sha256,
        "model_revision": str(spec["revision"]),
    }
    for field, value in expected.items():
        if str(provenance.get(field, "")).casefold() != value.casefold():
            raise ZeroShotProtocolError(f"Completed provenance mismatch: {field}")
    if int(provenance.get("num_rows", -1)) != len(rows):
        raise ZeroShotProtocolError("Completed provenance row count mismatch")
    for field in (
        "snapshot_sha256",
        "model_fingerprint_sha256",
        "processor_fingerprint_sha256",
    ):
        if not is_sha256(provenance.get(field)):
            raise ZeroShotProtocolError(f"Completed provenance lacks {field}")


def _resume_payload(
    *,
    prediction_path: Path,
    row_count: int,
    run_contract_sha256: str,
    selected_rows_sha256: str,
    evidence: AuthorizationEvidence,
    loaded: LoadedZeroShotModel,
) -> dict[str, Any]:
    return {
        "resume_version": RESUME_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "selected_rows_sha256": selected_rows_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "benchmark_lock_sha256": evidence.benchmark_lock_sha256,
        "decision_lock_sha256": evidence.decision_lock_sha256,
        "completed_rows": row_count,
        "prediction_sha256": sha256_file(prediction_path),
        "snapshot": {
            "snapshot_sha256": loaded.snapshot_sha256,
            "model_fingerprint_sha256": loaded.model_fingerprint_sha256,
            "processor_fingerprint_sha256": loaded.processor_fingerprint_sha256,
        },
    }


def _recovery_payload(
    *,
    prediction_sha256: str,
    row_count: int,
    previous_prediction_sha256: str,
    previous_row_count: int,
    run_contract_sha256: str,
    selected_rows_sha256: str,
    evidence: AuthorizationEvidence,
    loaded: LoadedZeroShotModel,
) -> dict[str, Any]:
    """Create a write-ahead receipt for one atomic CSV publication.

    The receipt is published before the CSV.  Therefore a crash after the CSV
    replace but before the normal resume-state update remains recoverable only
    when the exact expected bytes and all run/model/benchmark identities match.
    """

    if not is_sha256(prediction_sha256):
        raise ZeroShotProtocolError("Recovery prediction SHA-256 is invalid")
    if previous_prediction_sha256 and not is_sha256(previous_prediction_sha256):
        raise ZeroShotProtocolError("Recovery previous prediction SHA-256 is invalid")
    return {
        "recovery_version": RECOVERY_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "selected_rows_sha256": selected_rows_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "benchmark_lock_sha256": evidence.benchmark_lock_sha256,
        "decision_lock_sha256": evidence.decision_lock_sha256,
        "completed_rows": row_count,
        "prediction_sha256": prediction_sha256,
        "previous_completed_rows": previous_row_count,
        "previous_prediction_sha256": previous_prediction_sha256,
        "snapshot": {
            "snapshot_sha256": loaded.snapshot_sha256,
            "model_fingerprint_sha256": loaded.model_fingerprint_sha256,
            "processor_fingerprint_sha256": loaded.processor_fingerprint_sha256,
        },
    }


def _validate_recovery_state(
    recovery: Mapping[str, Any],
    *,
    run_contract_sha256: str,
    selected_rows_sha256: str,
    evidence: AuthorizationEvidence,
    maximum_rows: int,
    loaded: LoadedZeroShotModel | None,
) -> None:
    if recovery.get("recovery_version") != RECOVERY_VERSION:
        raise ZeroShotProtocolError("Unsupported zero-shot recovery receipt")
    expected = {
        "run_contract_sha256": run_contract_sha256,
        "selected_rows_sha256": selected_rows_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "benchmark_lock_sha256": evidence.benchmark_lock_sha256,
        "decision_lock_sha256": evidence.decision_lock_sha256,
    }
    for field, value in expected.items():
        if str(recovery.get(field, "")).casefold() != value.casefold():
            raise ZeroShotProtocolError(f"Recovery receipt mismatch: {field}")
    try:
        completed_rows = int(recovery.get("completed_rows", -1))
        previous_rows = int(recovery.get("previous_completed_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ZeroShotProtocolError("Recovery receipt row count is invalid") from exc
    if not (1 <= completed_rows <= maximum_rows) or not (
        0 <= previous_rows < completed_rows
    ):
        raise ZeroShotProtocolError("Recovery receipt row transition is invalid")
    if not is_sha256(recovery.get("prediction_sha256")):
        raise ZeroShotProtocolError("Recovery receipt prediction hash is invalid")
    previous_sha = str(recovery.get("previous_prediction_sha256", ""))
    if (previous_rows == 0 and previous_sha) or (
        previous_rows > 0 and not is_sha256(previous_sha)
    ):
        raise ZeroShotProtocolError("Recovery receipt previous prediction hash is invalid")
    if loaded is not None:
        snapshot = recovery.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise ZeroShotProtocolError("Recovery receipt has no snapshot fingerprint")
        for field, value in (
            ("snapshot_sha256", loaded.snapshot_sha256),
            ("model_fingerprint_sha256", loaded.model_fingerprint_sha256),
            ("processor_fingerprint_sha256", loaded.processor_fingerprint_sha256),
        ):
            if str(snapshot.get(field, "")).casefold() != value.casefold():
                raise ZeroShotProtocolError(
                    f"Recovery model snapshot mismatch: {field}"
                )


def _publish_prediction_checkpoint(
    *,
    output_path: Path,
    progress_path: Path,
    receipt_path: Path,
    rows: Sequence[Mapping[str, Any]],
    previous_rows: Sequence[Mapping[str, Any]],
    run_contract_sha256: str,
    selected_rows_sha256: str,
    evidence: AuthorizationEvidence,
    loaded: LoadedZeroShotModel,
) -> None:
    """Publish a CSV plus resume state with a tamper-evident crash receipt."""

    expected_sha = hashlib.sha256(_prediction_csv_bytes(rows)).hexdigest()
    previous_sha = (
        hashlib.sha256(_prediction_csv_bytes(previous_rows)).hexdigest()
        if previous_rows
        else ""
    )
    _atomic_write_json(
        receipt_path,
        _recovery_payload(
            prediction_sha256=expected_sha,
            row_count=len(rows),
            previous_prediction_sha256=previous_sha,
            previous_row_count=len(previous_rows),
            run_contract_sha256=run_contract_sha256,
            selected_rows_sha256=selected_rows_sha256,
            evidence=evidence,
            loaded=loaded,
        ),
    )
    published_sha = _atomic_write_csv(output_path, rows)
    if published_sha != expected_sha:
        raise RuntimeError("Published prediction differs from its recovery receipt")
    _atomic_write_json(
        progress_path,
        _resume_payload(
            prediction_path=output_path,
            row_count=len(rows),
            run_contract_sha256=run_contract_sha256,
            selected_rows_sha256=selected_rows_sha256,
            evidence=evidence,
            loaded=loaded,
        ),
    )
    receipt_path.unlink()


def _run_one_model(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    suite_config_sha256: str,
    spec: Mapping[str, Any],
    evidence: AuthorizationEvidence,
    benchmark_rows: Sequence[Mapping[str, str]],
    resume: bool,
    model_loader: Callable[[Mapping[str, Any], Mapping[str, Any]], LoadedZeroShotModel],
    decoder: Callable[
        [LoadedZeroShotModel, Sequence[Mapping[str, str]], Mapping[str, Any]],
        Sequence[str],
    ],
) -> dict[str, Any]:
    output_reference = PurePosixPath(
        _portable_repository_path(config["output_dir"], label="output_dir")
    ).joinpath(str(spec["filename"])).as_posix()
    output_path = _repository_path(output_reference, label="prediction output")
    sidecar_path = provenance_path(output_path)
    progress_path = resume_path(output_path)
    receipt_path = recovery_path(output_path)
    run_contract = _run_contract(
        config, spec, suite_config_sha256=suite_config_sha256
    )
    run_contract_sha = canonical_sha256(run_contract)
    selected_rows_sha = benchmark_rows_sha256(benchmark_rows)
    seed = int(config["seed"])

    if sidecar_path.exists() and not output_path.exists():
        raise ZeroShotProtocolError(
            f"Orphan provenance exists without prediction: {sidecar_path}"
        )
    if progress_path.exists() and not output_path.exists():
        raise ZeroShotProtocolError(
            f"Orphan resume state exists without prediction: {progress_path}"
        )
    if sidecar_path.exists() and (progress_path.exists() or receipt_path.exists()):
        raise ZeroShotProtocolError(
            f"Completed and partial state coexist; refusing ambiguous output: {output_path}"
        )

    existing_rows: list[dict[str, str]] = []
    state: dict[str, Any] | None = None
    recovery: dict[str, Any] | None = None
    recovery_mode: str | None = None
    if receipt_path.exists() and not resume:
        raise FileExistsError(
            f"Interrupted prediction transaction exists; pass --resume after reviewing it: "
            f"{receipt_path}"
        )
    if receipt_path.exists():
        recovery = _load_json(
            receipt_path, label="Zero-shot prediction recovery receipt"
        )
        _validate_recovery_state(
            recovery,
            run_contract_sha256=run_contract_sha,
            selected_rows_sha256=selected_rows_sha,
            evidence=evidence,
            maximum_rows=len(benchmark_rows),
            loaded=None,
        )
        if not output_path.exists():
            recovery_mode = "not_published"

    if output_path.exists():
        existing_rows = _read_prediction(output_path)
        _validate_prediction_prefix(existing_rows, benchmark_rows, spec, seed)
        if sidecar_path.exists():
            if len(existing_rows) != len(benchmark_rows):
                raise ZeroShotProtocolError(
                    "Completed provenance is attached to a partial prediction"
                )
            completed = _load_json(sidecar_path, label="Zero-shot provenance")
            _validate_completed_provenance(
                completed,
                prediction_path=output_path,
                rows=existing_rows,
                run_contract_sha256=run_contract_sha,
                selected_rows_sha256=selected_rows_sha,
                evidence=evidence,
                spec=spec,
            )
            return {
                "model_key": spec["key"],
                "status": "verified_existing",
                "prediction": output_reference,
                "provenance": output_reference + ".provenance.json",
                "rows": len(existing_rows),
            }
        if not progress_path.exists() and recovery is None:
            raise ZeroShotProtocolError(
                f"Prediction has neither provenance nor resume state: {output_path}"
            )
        if not resume:
            raise FileExistsError(
                f"Partial prediction exists; pass --resume after reviewing it: {output_path}"
            )
        actual_prediction_sha = sha256_file(output_path)
        if recovery is not None:
            target_sha = str(recovery["prediction_sha256"]).casefold()
            target_rows = int(recovery["completed_rows"])
            previous_sha = str(
                recovery.get("previous_prediction_sha256", "")
            ).casefold()
            previous_rows = int(recovery.get("previous_completed_rows", -1))
            if actual_prediction_sha == target_sha and len(existing_rows) == target_rows:
                recovery_mode = "published"
            elif (
                progress_path.exists()
                and actual_prediction_sha == previous_sha
                and len(existing_rows) == previous_rows
            ):
                recovery_mode = "not_published"
            else:
                raise ZeroShotProtocolError(
                    "Prediction does not match either exact hash/row state in its "
                    "recovery receipt; refusing possible tamper"
                )
        if progress_path.exists() and recovery_mode != "published":
            state = _load_json(progress_path, label="Zero-shot resume state")
            _validate_resume_state(
                state,
                prediction_path=output_path,
                row_count=len(existing_rows),
                run_contract_sha256=run_contract_sha,
                selected_rows_sha256=selected_rows_sha,
                evidence=evidence,
                loaded=None,
            )

    loaded = model_loader(spec, config)
    for field in (
        loaded.snapshot_sha256,
        loaded.model_fingerprint_sha256,
        loaded.processor_fingerprint_sha256,
    ):
        if not is_sha256(field):
            raise ZeroShotProtocolError("Model loader returned an invalid snapshot fingerprint")
    if state is not None:
        _validate_resume_state(
            state,
            prediction_path=output_path,
            row_count=len(existing_rows),
            run_contract_sha256=run_contract_sha,
            selected_rows_sha256=selected_rows_sha,
            evidence=evidence,
            loaded=loaded,
        )
    if recovery is not None:
        _validate_recovery_state(
            recovery,
            run_contract_sha256=run_contract_sha,
            selected_rows_sha256=selected_rows_sha,
            evidence=evidence,
            maximum_rows=len(benchmark_rows),
            loaded=loaded,
        )
        if recovery_mode == "published":
            _atomic_write_json(
                progress_path,
                _resume_payload(
                    prediction_path=output_path,
                    row_count=len(existing_rows),
                    run_contract_sha256=run_contract_sha,
                    selected_rows_sha256=selected_rows_sha,
                    evidence=evidence,
                    loaded=loaded,
                ),
            )
        elif recovery_mode != "not_published":
            raise RuntimeError("Unclassified zero-shot recovery state")
        receipt_path.unlink()

    rows: list[dict[str, Any]] = list(existing_rows)
    batch_size = int(config["runtime"]["batch_size"])
    for start in range(len(rows), len(benchmark_rows), batch_size):
        batch = benchmark_rows[start : start + batch_size]
        hypotheses = list(decoder(loaded, batch, config))
        if len(hypotheses) != len(batch):
            raise RuntimeError(
                f"Decoder returned {len(hypotheses)} hypotheses for {len(batch)} inputs"
            )
        previous_rows = list(rows)
        rows.extend(
            _expected_prediction_prefix(
                batch,
                spec,
                seed,
                [str(hypothesis) for hypothesis in hypotheses],
            )
        )
        _publish_prediction_checkpoint(
            output_path=output_path,
            progress_path=progress_path,
            receipt_path=receipt_path,
            rows=rows,
            previous_rows=previous_rows,
            run_contract_sha256=run_contract_sha,
            selected_rows_sha256=selected_rows_sha,
            evidence=evidence,
            loaded=loaded,
        )

    if len(rows) != len(benchmark_rows):
        raise RuntimeError(
            f"Zero-shot inference produced {len(rows)} rows, expected {len(benchmark_rows)}"
        )
    _validate_prediction_prefix(rows, benchmark_rows, spec, seed)
    if not progress_path.is_file():
        raise RuntimeError(
            "Completed prediction is missing its verified resume state"
        )
    prediction_sha = sha256_file(output_path)
    provenance = {
        "provenance_version": PROVENANCE_VERSION,
        "evaluation_split": "test",
        "evaluation_scope": "full_manifest",
        "metric_version": "aligned_v1",
        "schema": PREDICTION_COLUMNS,
        "prediction": output_reference,
        "prediction_sha256": prediction_sha,
        "num_rows": len(rows),
        "manifest": _portable_repository_path(
            config["benchmark"]["manifest"], label="benchmark.manifest"
        ),
        "manifest_sha256": evidence.manifest_sha256,
        "selected_rows_sha256": selected_rows_sha,
        "audio_hashes_verified": True,
        "benchmark_lock": _portable_repository_path(
            config["benchmark"]["lock"], label="benchmark.lock"
        ),
        **_authorization_payload(evidence),
        "suite_config": _portable_repository_path(
            config_path, label="config_path"
        ),
        "suite_config_sha256": suite_config_sha256,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha,
        "model_key": spec["key"],
        "model_repo_id": spec["repo_id"],
        "model_revision": spec["revision"],
        "model": spec["model"],
        "model_size": spec["model_size"],
        "train_type": "zero_shot",
        "lambda": "",
        "seed": seed,
        "snapshot_sha256": loaded.snapshot_sha256,
        "model_fingerprint_sha256": loaded.model_fingerprint_sha256,
        "processor_fingerprint_sha256": loaded.processor_fingerprint_sha256,
        "decoding": {
            **dict(config["decoding"]),
            "implementation": "whisper_generate_greedy_paper_v2_v1",
            "do_sample": False,
            "num_beams": 1,
            "deterministic_algorithms": True,
        },
        "runtime_environment": {
            **dict(loaded.runtime_environment),
            "batch_size": batch_size,
        },
    }
    _assert_no_absolute_path_strings(provenance, label="provenance")
    _atomic_write_new_json(sidecar_path, provenance)
    progress_path.unlink()
    return {
        "model_key": spec["key"],
        "status": "complete",
        "prediction": output_reference,
        "prediction_sha256": prediction_sha,
        "provenance": output_reference + ".provenance.json",
        "rows": len(rows),
    }


def run_zero_shot_suite(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    model_keys: Sequence[str] | None = None,
    resume: bool = False,
    authorizer: Callable[[Mapping[str, Any]], AuthorizationEvidence] = authorize_final_benchmark,
    manifest_loader: Callable[
        [Mapping[str, Any], AuthorizationEvidence], list[dict[str, str]]
    ] = load_authorized_benchmark,
    model_loader: Callable[
        [Mapping[str, Any], Mapping[str, Any]], LoadedZeroShotModel
    ] = load_huggingface_model,
    decoder: Callable[
        [LoadedZeroShotModel, Sequence[Mapping[str, str]], Mapping[str, Any]],
        Sequence[str],
    ] = decode_batch,
) -> dict[str, Any]:
    """Run selected baselines after one fail-closed final-test preflight."""

    config_reference = _portable_repository_path(config_path, label="config_path")
    config_file = _repository_path(config_reference, label="config_path")
    config = load_suite_config(config_reference)
    specs = validate_suite_config(config, model_keys)
    suite_config_sha = sha256_file(config_file)

    # This order is a protocol guarantee: no benchmark row/audio or model may be
    # accessed until split, method decision, and final benchmark locks all pass.
    evidence = authorizer(config)
    benchmark_rows = manifest_loader(config, evidence)
    if len(benchmark_rows) != evidence.manifest_num_rows:
        raise ZeroShotProtocolError("Manifest loader returned an unauthorized row count")

    results = [
        _run_one_model(
            config=config,
            config_path=Path(config_reference),
            suite_config_sha256=suite_config_sha,
            spec=spec,
            evidence=evidence,
            benchmark_rows=benchmark_rows,
            resume=resume,
            model_loader=model_loader,
            decoder=decoder,
        )
        for spec in specs
    ]
    return {
        "status": "complete",
        "models": results,
        "manifest_sha256": evidence.manifest_sha256,
        "benchmark_lock_sha256": evidence.benchmark_lock_sha256,
        "decision_lock_sha256": evidence.decision_lock_sha256,
    }


__all__ = [
    "AuthorizationEvidence",
    "DEFAULT_CONFIG",
    "LoadedZeroShotModel",
    "PREDICTION_COLUMNS",
    "PROVENANCE_VERSION",
    "RECOVERY_VERSION",
    "REPOSITORY_ROOT",
    "RESUME_VERSION",
    "ZeroShotProtocolError",
    "authorize_final_benchmark",
    "benchmark_rows_sha256",
    "decode_batch",
    "fingerprint_snapshot",
    "load_authorized_benchmark",
    "load_huggingface_model",
    "load_suite_config",
    "provenance_path",
    "recovery_path",
    "resume_path",
    "run_zero_shot_suite",
    "sha256_file",
    "validate_suite_config",
]
