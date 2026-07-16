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
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FINAL_BENCHMARK_VERSION = "paper_v2_final_benchmark_v1"
FINAL_BENCHMARK_ALGORITHM = "sha256_musan_test_power_mix_v1"
FINAL_SNRS = (20.0, 10.0, 5.0, 0.0)
FINAL_SOURCE_COUNT = 460
FINAL_ROW_COUNT = 2300
FINAL_SEED = 42
FINAL_SAMPLE_RATE = 16000
FINAL_PEAK_LIMIT = 0.999
FINAL_AUDIT_COLUMNS = (
    "protocol_version",
    "check_id",
    "status",
    "observed",
    "expected",
    "details",
)
FINAL_BENCHMARK_COLUMNS = (
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
    "selection_eligible",
    "final_test_eligible",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalBenchmarkError(ValueError):
    """Raised when the final benchmark cannot be reproduced safely."""


@dataclass(frozen=True)
class FinalBenchmarkConfig:
    split_lock: Path
    decision_lock: Path
    noise_split_lock: Path
    method_lock: Path
    method_config: Path
    source_test_manifest: Path
    output_manifest: Path
    output_audio_dir: Path
    protocol_lock: Path
    protocol_audit: Path
    snrs: tuple[float, ...] = FINAL_SNRS
    seed: int = FINAL_SEED
    sample_rate: int = FINAL_SAMPLE_RATE
    peak_limit: float = FINAL_PEAK_LIMIT
    include_clean: bool = True
    expected_source_count: int = FINAL_SOURCE_COUNT


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FinalBenchmarkError(
            "Formal final-benchmark artifacts may only contain paths inside "
            f"the repository root: {path.resolve()}"
        ) from exc


def _artifact_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _paths_overlap(left: Path, right: Path) -> bool:
    # Lexical absolute paths avoid touching a locked test path before authorization.
    left_resolved = Path(os.path.abspath(left))
    right_resolved = Path(os.path.abspath(right))
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _resolved_paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _require_portable_path(path: Path, *, label: str) -> None:
    # Keep pre-authorization validation lexical. Symlinks are resolved only after
    # the decision lock has granted test access.
    absolute = Path(os.path.abspath(path))
    try:
        absolute.relative_to(Path(os.path.abspath(ROOT)))
    except ValueError as exc:
        raise FinalBenchmarkError(
            f"Formal {label} must be inside the repository root: {absolute}"
        ) from exc


def _require_resolved_portable_path(path: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise FinalBenchmarkError(
            f"Formal {label} resolves outside the repository root: {path}"
        ) from exc


def _is_portable_reference(value: object, *, allow_empty: bool = False) -> bool:
    raw = str(value).strip()
    if not raw:
        return allow_empty
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        return False
    try:
        (ROOT / path).resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Locked artifact does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalBenchmarkError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise FinalBenchmarkError(f"JSON artifact must contain an object: {path}")
    return value


def _locked_int(value: object, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FinalBenchmarkError(f"Invalid integer in final benchmark lock: {label}") from exc


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
                raise FinalBenchmarkError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise FinalBenchmarkError(
                    f"Manifest row must be an object at {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise FinalBenchmarkError(f"Manifest is empty: {path}")
    return rows


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


def _audio_info(path: Path) -> tuple[float, int, int, int]:
    try:
        import soundfile as sf

        info = sf.info(path)
    except Exception as exc:
        raise FinalBenchmarkError(f"Cannot inspect audio file: {path}") from exc
    if info.frames < 1 or info.samplerate < 1 or info.channels < 1:
        raise FinalBenchmarkError(f"Audio file is empty or malformed: {path}")
    return (
        float(info.frames) / float(info.samplerate),
        int(info.samplerate),
        int(info.channels),
        int(info.frames),
    )


def _validate_config(config: FinalBenchmarkConfig) -> dict[str, object]:
    snrs = tuple(float(value) for value in config.snrs)
    if snrs != FINAL_SNRS:
        raise FinalBenchmarkError(
            f"Formal final benchmark requires SNRs {FINAL_SNRS}, got {snrs}"
        )
    if config.seed != FINAL_SEED or config.sample_rate != FINAL_SAMPLE_RATE:
        raise FinalBenchmarkError(
            "Formal final benchmark requires seed=42 and sample_rate=16000"
        )
    if not config.include_clean:
        raise FinalBenchmarkError("Formal final benchmark must include clean speech")
    if config.expected_source_count != FINAL_SOURCE_COUNT:
        raise FinalBenchmarkError(
            f"Formal final benchmark requires exactly {FINAL_SOURCE_COUNT} "
            "locked unseen utterances"
        )
    if float(config.peak_limit) != FINAL_PEAK_LIMIT:
        raise FinalBenchmarkError(
            f"Formal final benchmark requires peak_limit={FINAL_PEAK_LIMIT}"
        )
    expected_rows = config.expected_source_count * (len(snrs) + 1)
    if expected_rows != FINAL_ROW_COUNT:
        raise FinalBenchmarkError(
            f"Formal final benchmark requires exactly {FINAL_ROW_COUNT} rows"
        )

    inputs = {
        "split lock": config.split_lock,
        "decision lock": config.decision_lock,
        "noise split lock": config.noise_split_lock,
        "method lock": config.method_lock,
        "method config": config.method_config,
        "source test manifest": config.source_test_manifest,
    }
    outputs = {
        "output manifest": config.output_manifest,
        "output audio directory": config.output_audio_dir,
        "protocol lock": config.protocol_lock,
        "protocol audit": config.protocol_audit,
    }
    for label, path in {**inputs, **outputs}.items():
        _require_portable_path(path, label=label)
    output_items = list(outputs.items())
    for index, (left_label, left) in enumerate(output_items):
        for right_label, right in output_items[index + 1 :]:
            if _paths_overlap(left, right):
                raise FinalBenchmarkError(
                    "Final benchmark output paths must not overlap: "
                    f"{left_label} and {right_label}"
                )
        for input_label, input_path in inputs.items():
            if _paths_overlap(left, input_path):
                raise FinalBenchmarkError(
                    "Final benchmark outputs must not overlap locked inputs: "
                    f"{left_label} and {input_label}"
                )
    return {
        "algorithm": FINAL_BENCHMARK_ALGORITHM,
        "seed": config.seed,
        "snrs_db": list(snrs),
        "sample_rate": config.sample_rate,
        "peak_limit": config.peak_limit,
        "include_clean": True,
        "expected_source_count": config.expected_source_count,
        "expected_row_count": expected_rows,
        "audio_container": "WAV",
        "audio_subtype": "PCM_16",
        "input_sample_rate_policy": "require_exact",
        "channel_policy": "mean_to_mono",
        "snr_measurement": "component_power_after_anti_clip_before_pcm16",
        "clipping_measurement": "pre_scale_over_1_and_stored_full_scale",
        "source_partition": "vivos_test_locked",
        "noise_partition": "musan_test",
    }


def _validate_resolved_paths_after_authorization(
    config: FinalBenchmarkConfig,
) -> None:
    inputs = {
        "split lock": config.split_lock,
        "decision lock": config.decision_lock,
        "noise split lock": config.noise_split_lock,
        "method lock": config.method_lock,
        "method config": config.method_config,
        "source test manifest": config.source_test_manifest,
    }
    outputs = {
        "output manifest": config.output_manifest,
        "output audio directory": config.output_audio_dir,
        "protocol lock": config.protocol_lock,
        "protocol audit": config.protocol_audit,
    }
    for label, path in {**inputs, **outputs}.items():
        _require_resolved_portable_path(path, label=label)
    output_items = list(outputs.items())
    for index, (left_label, left) in enumerate(output_items):
        for right_label, right in output_items[index + 1 :]:
            if _resolved_paths_overlap(left, right):
                raise FinalBenchmarkError(
                    "Resolved final benchmark output paths must not overlap: "
                    f"{left_label} and {right_label}"
                )
        for input_label, input_path in inputs.items():
            if _resolved_paths_overlap(left, input_path):
                raise FinalBenchmarkError(
                    "Resolved final benchmark outputs must not overlap inputs: "
                    f"{left_label} and {input_label}"
                )


def _default_decision_verifier(**kwargs: Any) -> Mapping[str, Any]:
    from .phat.protocol import verify_test_decision_lock

    return verify_test_decision_lock(**kwargs)


def _default_source_verifier(
    manifest_path: Path, *, split_lock_path: Path
) -> Mapping[str, Any]:
    from .phat.protocol import verify_locked_vivos_manifest

    return verify_locked_vivos_manifest(
        manifest_path,
        split_name="test_locked",
        split_lock_path=split_lock_path,
        verify_audio=True,
    )


def _default_noise_verifier(lock_path: Path) -> Mapping[str, Any]:
    from .noise_protocol import verify_noise_split_lock

    return verify_noise_split_lock(lock_path, verify_audio=True)


def _default_method_verifier(config: FinalBenchmarkConfig) -> Mapping[str, Any]:
    from .phat.config import load_experiment_config
    from .phat.method_contract import verify_method_lock

    experiment = load_experiment_config(config.method_config)
    return verify_method_lock(
        config.method_lock,
        config=experiment,
        repo_root=ROOT,
        formal=True,
        verify_audio=True,
    )


def _authorize_before_test_access(
    config: FinalBenchmarkConfig,
    decision_verifier: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate decision first; this function never opens test manifests/audio."""

    decision = dict(
        decision_verifier(
            split_lock_path=config.split_lock,
            decision_lock_path=config.decision_lock,
        )
    )
    for field in (
        "split_lock_sha256",
        "decision_lock_sha256",
        "test_manifest_sha256",
        "method_lock_sha256",
        "method_identity_sha256",
    ):
        if not _is_sha256(decision.get(field)):
            raise FinalBenchmarkError(
                f"Decision authorization returned invalid {field}"
            )
    if not config.split_lock.is_file() or (
        sha256_file(config.split_lock)
        != str(decision["split_lock_sha256"]).casefold()
    ):
        raise FinalBenchmarkError("Decision does not bind the configured split lock")
    if not config.decision_lock.is_file() or (
        sha256_file(config.decision_lock)
        != str(decision["decision_lock_sha256"]).casefold()
    ):
        raise FinalBenchmarkError("Decision verifier did not bind its decision lock")
    return decision


def _validate_source_rows(
    path: Path,
    *,
    expected_hash: str,
    expected_count: int,
    sample_rate: int,
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_hash:
        raise FinalBenchmarkError(
            "Locked unseen-test manifest differs from the authorized decision"
        )
    rows = _read_jsonl(path)
    if len(rows) != expected_count:
        raise FinalBenchmarkError(
            f"Final source has {len(rows)} rows, expected {expected_count}"
        )
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        source_id = str(row.get("source_utt_id") or row.get("utt_id") or "").strip()
        audio_value = row.get("audio") or row.get("audio_path")
        transcript = row.get("text") or row.get("transcript")
        audio_sha = str(row.get("audio_sha256", "")).casefold()
        if (
            not source_id
            or source_id in seen_ids
            or not audio_value
            or transcript is None
            or row.get("dataset") != "vivos"
            or row.get("split") != "test"
            or not _is_sha256(audio_sha)
        ):
            raise FinalBenchmarkError(
                f"Invalid/duplicate locked test identity at {path}:{number}"
            )
        audio = _artifact_path(audio_value)
        _require_portable_path(audio, label="locked VIVOS test audio")
        if not audio.is_file() or sha256_file(audio) != audio_sha:
            raise FinalBenchmarkError(f"Locked test audio hash mismatch: {audio}")
        duration, rate, _, _ = _audio_info(audio)
        if rate != sample_rate:
            raise FinalBenchmarkError(
                f"Locked VIVOS test audio must be {sample_rate} Hz: {audio}"
            )
        text = str(transcript)
        text_sha = str(row.get("text_sha256", "")).casefold()
        actual_text_sha = _sha256_bytes(text.encode("utf-8"))
        if text_sha and (not _is_sha256(text_sha) or text_sha != actual_text_sha):
            raise FinalBenchmarkError(f"Locked test text hash mismatch: {source_id}")
        if audio_sha in seen_audio:
            raise FinalBenchmarkError("Locked test contains duplicate audio content")
        seen_ids.add(source_id)
        seen_audio.add(audio_sha)
        normalized.append(
            {
                **row,
                "source_utt_id": source_id,
                "audio_path": audio,
                "audio_sha256": audio_sha,
                "transcript": text,
                "text_sha256": text_sha or actual_text_sha,
                "duration_seconds": duration,
            }
        )
    return sorted(normalized, key=lambda row: row["source_utt_id"])


def _validate_noise_integrity(
    verified: Mapping[str, Any], config: FinalBenchmarkConfig
) -> tuple[dict[str, Any], str, list[dict[str, Any]], set[str], set[str]]:
    lock = verified.get("lock")
    registry = verified.get("registry_rows")
    lock_sha = str(verified.get("lock_sha256", "")).casefold()
    if not isinstance(lock, Mapping) or not isinstance(registry, list):
        raise FinalBenchmarkError("Noise verifier returned malformed integrity data")
    if not _is_sha256(lock_sha) or sha256_file(config.noise_split_lock) != lock_sha:
        raise FinalBenchmarkError("Noise verifier returned another split lock")
    test_rows = [dict(row) for row in registry if row.get("split") == "test"]
    forbidden = [row for row in registry if row.get("split") in {"train", "dev"}]
    if not test_rows or any(row.get("split") != "test" for row in test_rows):
        raise FinalBenchmarkError("Locked MUSAN test partition is empty or malformed")
    test_ids = {str(row.get("noise_id", "")) for row in test_rows}
    test_hashes = {str(row.get("audio_sha256", "")).casefold() for row in test_rows}
    forbidden_ids = {str(row.get("noise_id", "")) for row in forbidden}
    forbidden_hashes = {
        str(row.get("audio_sha256", "")).casefold() for row in forbidden
    }
    if (
        "" in test_ids
        or any(not _is_sha256(value) for value in test_hashes)
        or test_ids & forbidden_ids
        or test_hashes & forbidden_hashes
    ):
        raise FinalBenchmarkError("MUSAN test is not content-disjoint from train/dev")
    for row in test_rows:
        audio = _artifact_path(row.get("audio", ""))
        _require_portable_path(audio, label="locked MUSAN test audio")
        if not audio.is_file() or sha256_file(audio) != str(row["audio_sha256"]):
            raise FinalBenchmarkError(f"MUSAN test audio hash mismatch: {audio}")
        _, rate, _, _ = _audio_info(audio)
        if rate != config.sample_rate:
            raise FinalBenchmarkError(
                f"MUSAN test audio must be {config.sample_rate} Hz: {audio}"
            )
    return dict(lock), lock_sha, test_rows, forbidden_ids, forbidden_hashes


def _stable_seed(master_seed: int, source_id: str, snr: float) -> int:
    payload = (
        f"{FINAL_BENCHMARK_ALGORITHM}|seed={master_seed}|"
        f"source={source_id}|snr={snr:g}"
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    base = (normalized[:96] or "utterance").rstrip(". ") or "utterance"
    identity_suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{base}_{identity_suffix}"


def _snr_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}".replace(".", "p")


def _mix_with_measurements(
    clean: Any, noise: Any, *, target_snr_db: float, peak_limit: float
) -> tuple[Any, dict[str, float | int]]:
    import numpy as np

    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if len(clean) == 0 or len(noise) != len(clean):
        raise FinalBenchmarkError("Clean/noise waveforms must be non-empty and aligned")
    clean_power = float(np.mean(clean**2))
    noise_power = float(np.mean(noise**2))
    if clean_power <= 1e-12 or noise_power <= 1e-12:
        raise FinalBenchmarkError("Cannot mix silent clean or noise audio")
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


def _audit_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source_rows: Sequence[Mapping[str, object]],
    noise_rows: Sequence[Mapping[str, object]],
    snrs: Sequence[float],
    output_audio_dir: Path,
    forbidden_ids: set[str],
    forbidden_hashes: set[str],
    staged_audio_dir: Path | None = None,
) -> list[dict[str, object]]:
    source_count = len(source_rows)
    noisy = [row for row in rows if row["condition"] == "noisy"]
    clean = [row for row in rows if row["condition"] == "clean"]
    expected_rows = source_count * (len(snrs) + 1)
    expected_snr_labels = {_snr_label(float(value)) for value in snrs}
    expected_condition_keys = {
        (str(source["source_utt_id"]), condition)
        for source in source_rows
        for condition in {"clean", *expected_snr_labels}
    }
    observed_condition_keys = [
        (str(row["source_utt_id"]), str(row["snr"])) for row in rows
    ]
    source_by_id = {
        str(row["source_utt_id"]): row for row in source_rows
    }
    noise_by_id = {str(row["noise_id"]): row for row in noise_rows}

    def source_link_matches(row: Mapping[str, object]) -> bool:
        source = source_by_id.get(str(row["source_utt_id"]))
        if source is None:
            return False
        source_audio = Path(source["audio_path"])
        if (
            str(row["dataset"]) != "vivos"
            or str(row["split"]) != "test"
            or str(row["transcript"]) != str(source["transcript"])
            or str(row["text_sha256"]) != str(source["text_sha256"])
            or str(row["clean_audio_sha256"]) != str(source["audio_sha256"])
            or not _same_path(_artifact_path(row["clean_path"]), source_audio)
        ):
            return False
        if row["condition"] == "clean":
            return (
                str(row["snr"]) == "clean"
                and str(row["noise_type"]) == "clean"
                and str(row["audio_sha256"]) == str(source["audio_sha256"])
                and _same_path(_artifact_path(row["audio_path"]), source_audio)
            )
        return str(row["snr"]) in expected_snr_labels

    def noise_link_matches(row: Mapping[str, object]) -> bool:
        noise = noise_by_id.get(str(row["noise_id"]))
        if noise is None:
            return False
        return (
            str(row["noise_split"]) == "test"
            and str(row["noise_audio_sha256"]) == str(noise["audio_sha256"])
            and str(row["noise_type"]) == str(noise["noise_type"])
            and _same_path(
                _artifact_path(row["noise_path"]),
                _artifact_path(noise["audio"]),
            )
        )

    def expected_noisy_path(row: Mapping[str, object]) -> Path:
        source_id = str(row["source_utt_id"])
        label = str(row["snr"])
        return output_audio_dir / f"snr_{label}" / (
            f"{_safe_stem(source_id)}_snr{label}.wav"
        )

    def stored_audio_path(row: Mapping[str, object]) -> Path:
        final_path = _artifact_path(row["audio_path"])
        if row["condition"] != "noisy" or staged_audio_dir is None:
            return final_path
        relative = final_path.resolve().relative_to(output_audio_dir.resolve())
        return staged_audio_dir / relative

    def audio_hash_matches(row: Mapping[str, object]) -> bool:
        try:
            audio = stored_audio_path(row)
        except (TypeError, ValueError):
            return False
        return audio.is_file() and sha256_file(audio) == str(row["audio_sha256"])

    derived_paths = [_artifact_path(row["audio_path"]).resolve() for row in noisy]
    actual_audio_root = staged_audio_dir or output_audio_dir
    actual_derived_files = {
        path.resolve() for path in actual_audio_root.rglob("*") if path.is_file()
    }
    expected_derived_files = {
        stored_audio_path(row).resolve() for row in noisy
    }
    source_link_results = [source_link_matches(row) for row in rows]
    noise_link_results = [noise_link_matches(row) for row in noisy]
    audio_hash_results = [audio_hash_matches(row) for row in rows]
    portable_path_results = [
        _is_portable_reference(row["audio_path"])
        and _is_portable_reference(row["clean_path"])
        and _is_portable_reference(
            row["noise_path"], allow_empty=row["condition"] == "clean"
        )
        for row in rows
    ]
    checks: list[tuple[str, bool, object, object, str]] = [
        (
            "source_count",
            len({row["source_utt_id"] for row in rows}) == source_count,
            len({row["source_utt_id"] for row in rows}),
            source_count,
            "only locked unseen VIVOS source utterances",
        ),
        (
            "expected_row_count",
            len(rows) == expected_rows,
            len(rows),
            expected_rows,
            "clean plus four SNR conditions",
        ),
        (
            "clean_row_count",
            len(clean) == source_count,
            len(clean),
            source_count,
            "one clean row per source utterance",
        ),
        (
            "noisy_row_count",
            len(noisy) == source_count * len(snrs),
            len(noisy),
            source_count * len(snrs),
            "one noisy row per source utterance/SNR",
        ),
        (
            "condition_completeness",
            set(observed_condition_keys) == expected_condition_keys
            and len(observed_condition_keys) == len(expected_condition_keys),
            len(set(observed_condition_keys)),
            len(expected_condition_keys),
            "every source occurs exactly once in clean/20/10/5/0",
        ),
        (
            "unique_utterance_ids",
            len({row["utt_id"] for row in rows}) == len(rows),
            len({row["utt_id"] for row in rows}),
            len(rows),
            "derived utterance IDs are unique",
        ),
        (
            "source_identity_linkage",
            all(source_link_results),
            sum(source_link_results),
            len(rows),
            "every row preserves its locked source audio/text identity",
        ),
        (
            "noise_partition_is_test",
            all(row["noise_split"] == "test" for row in noisy),
            sum(row["noise_split"] == "test" for row in noisy),
            len(noisy),
            "no MUSAN train/dev may enter final test",
        ),
        (
            "noise_disjoint_from_train_dev",
            all(
                row["noise_id"] not in forbidden_ids
                and row["noise_audio_sha256"] not in forbidden_hashes
                for row in noisy
            ),
            sum(
                row["noise_id"] in forbidden_ids
                or row["noise_audio_sha256"] in forbidden_hashes
                for row in noisy
            ),
            0,
            "assigned test noise is file/content disjoint",
        ),
        (
            "noise_registry_linkage",
            all(noise_link_results),
            sum(noise_link_results),
            len(noisy),
            "every noisy row binds an exact locked MUSAN-test registry entry",
        ),
        (
            "measured_snr_tolerance",
            all(
                abs(float(row["measured_snr_db"]) - float(row["target_snr_db"]))
                <= 1e-6
                for row in noisy
            ),
            max(
                (
                    abs(
                        float(row["measured_snr_db"])
                        - float(row["target_snr_db"])
                    )
                    for row in noisy
                ),
                default=0.0,
            ),
            1e-6,
            "component-power SNR error before PCM16 quantization",
        ),
        (
            "stored_audio_has_no_clipping",
            all(int(row["clipped_sample_count"]) == 0 for row in noisy),
            sum(int(row["clipped_sample_count"]) for row in noisy),
            0,
            "post-write full-scale samples",
        ),
        (
            "derived_audio_paths_contained",
            all(
                _artifact_path(row["audio_path"]).resolve().is_relative_to(
                    output_audio_dir.resolve()
                )
                for row in noisy
            ),
            len(noisy),
            len(noisy),
            "all noisy artifacts are inside output_audio_dir",
        ),
        (
            "portable_manifest_paths",
            all(portable_path_results),
            sum(portable_path_results),
            len(rows),
            "manifest paths are repository-relative and contain no parent traversal",
        ),
        (
            "unique_collision_safe_audio_paths",
            len(set(derived_paths)) == len(derived_paths)
            and all(
                _same_path(_artifact_path(row["audio_path"]), expected_noisy_path(row))
                for row in noisy
            ),
            len(set(derived_paths)),
            len(noisy),
            "hashed source stems prevent derived path collisions",
        ),
        (
            "derived_audio_inventory_exact",
            actual_derived_files == expected_derived_files,
            len(actual_derived_files),
            len(expected_derived_files),
            "output audio directory has no missing or unrecorded files",
        ),
        (
            "audio_hash_integrity",
            all(audio_hash_results),
            sum(audio_hash_results),
            len(rows),
            "manifest hashes match every clean and derived audio artifact",
        ),
        (
            "eligibility_flags",
            all(
                row["selection_eligible"] is False
                and row["final_test_eligible"] is True
                for row in rows
            ),
            len(rows),
            len(rows),
            "final benchmark is never selection eligible",
        ),
        (
            "sample_rate",
            all(int(row["sample_rate"]) == FINAL_SAMPLE_RATE for row in rows),
            len([row for row in rows if int(row["sample_rate"]) == FINAL_SAMPLE_RATE]),
            len(rows),
            "all clean and derived audio is 16 kHz",
        ),
    ]
    audit = [
        {
            "protocol_version": FINAL_BENCHMARK_VERSION,
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
        raise FinalBenchmarkError(
            "Final benchmark audit failed: " + ", ".join(failures)
        )
    return audit


def _remove_target(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _commit_transaction(
    *,
    staged_audio_dir: Path,
    output_audio_dir: Path,
    payloads: Mapping[Path, bytes],
    overwrite: bool,
) -> None:
    targets = [output_audio_dir, *payloads]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FinalBenchmarkError(
            "Final benchmark outputs already exist; refusing to overwrite: "
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
                raise FinalBenchmarkError(f"Stale transaction artifact: {tmp}")
            tmp.write_bytes(payload)
            temporary[path] = tmp
        for target in targets:
            if target.exists():
                backup = target.with_name(f".{target.name}.{os.getpid()}.backup")
                if backup.exists():
                    raise FinalBenchmarkError(f"Stale transaction backup: {backup}")
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
                _remove_target(target)
            for target, backup in backups.items():
                if backup.exists() and not target.exists():
                    backup.rename(target)
            raise
        for backup in backups.values():
            _remove_target(backup)
    finally:
        for tmp in temporary.values():
            _remove_target(tmp)
        for target, backup in backups.items():
            if backup.exists() and not target.exists():
                backup.rename(target)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _verify_existing(
    config: FinalBenchmarkConfig,
    *,
    bindings: Mapping[str, str],
    builder_params: Mapping[str, object],
    builder_sha256: str,
    method_identity_sha256: str,
    source_rows: Sequence[Mapping[str, object]],
    noise_lock: Mapping[str, object],
    noise_rows: Sequence[Mapping[str, object]],
    forbidden_ids: set[str],
    forbidden_hashes: set[str],
) -> dict[str, object]:
    lock = _read_json(config.protocol_lock)
    if (
        lock.get("protocol_version") != FINAL_BENCHMARK_VERSION
        or lock.get("status") != "LOCKED"
        or lock.get("selection_eligible") is not False
        or lock.get("final_test_eligible") is not True
    ):
        raise FinalBenchmarkError("Existing final benchmark lock is invalid")
    for field, expected in bindings.items():
        if str(lock.get(field, "")).casefold() != expected.casefold():
            raise FinalBenchmarkError(f"Existing lock binds another {field}")
    if (
        str(lock.get("method_identity_sha256", "")).casefold()
        != method_identity_sha256.casefold()
    ):
        raise FinalBenchmarkError("Existing lock binds another method identity")
    builder = lock.get("builder", {})
    if (
        not isinstance(builder, Mapping)
        or builder.get("params") != dict(builder_params)
        or builder.get("params_sha256") != builder_sha256
    ):
        raise FinalBenchmarkError("Existing lock binds other builder parameters")
    if lock.get("schema") != list(FINAL_BENCHMARK_COLUMNS):
        raise FinalBenchmarkError("Existing final benchmark lock schema is invalid")
    output = lock.get("output", {})
    if not isinstance(output, Mapping):
        raise FinalBenchmarkError("Existing final benchmark lock has no output object")
    if not _is_portable_reference(output.get("manifest", "")) or not _is_portable_reference(
        output.get("audio_dir", "")
    ):
        raise FinalBenchmarkError("Existing final benchmark output paths are not portable")
    if not _same_path(
        _artifact_path(output.get("manifest", "")), config.output_manifest
    ) or not _same_path(
        _artifact_path(output.get("audio_dir", "")), config.output_audio_dir
    ):
        raise FinalBenchmarkError("Existing lock binds other output paths")
    if sha256_file(config.output_manifest) != output.get("manifest_sha256"):
        raise FinalBenchmarkError("Existing final manifest hash mismatch")
    if output.get("audio_hashes_recorded") is not True:
        raise FinalBenchmarkError("Existing final lock does not bind per-audio hashes")
    rows = _read_jsonl(config.output_manifest)
    expected_rows = config.expected_source_count * (len(config.snrs) + 1)
    expected_noisy = config.expected_source_count * len(config.snrs)
    if (
        expected_rows != FINAL_ROW_COUNT
        or len(rows) != expected_rows
        or _locked_int(output.get("row_count"), label="output.row_count")
        != expected_rows
        or _locked_int(
            output.get("clean_row_count"), label="output.clean_row_count"
        )
        != config.expected_source_count
        or _locked_int(
            output.get("noisy_row_count"), label="output.noisy_row_count"
        )
        != expected_noisy
    ):
        raise FinalBenchmarkError(
            "Existing final manifest violates the formal 460x5 row contract"
        )
    inventory: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=1):
        if tuple(row) != FINAL_BENCHMARK_COLUMNS:
            raise FinalBenchmarkError(
                f"Existing final manifest schema mismatch at row {number}"
            )
        audio = _artifact_path(row["audio_path"])
        if not audio.is_file() or sha256_file(audio) != row["audio_sha256"]:
            raise FinalBenchmarkError(f"Existing final audio hash mismatch: {audio}")
        inventory.append(
            {"utt_id": str(row["utt_id"]), "audio_sha256": str(row["audio_sha256"])}
        )
    if _canonical_sha256(inventory) != output.get("audio_inventory_sha256"):
        raise FinalBenchmarkError("Existing final audio inventory hash mismatch")

    source_inventory = [
        {
            "source_utt_id": row["source_utt_id"],
            "audio_sha256": row["audio_sha256"],
            "text_sha256": row["text_sha256"],
        }
        for row in source_rows
    ]
    source_meta = lock.get("source_test", {})
    if (
        not isinstance(source_meta, Mapping)
        or not _is_portable_reference(source_meta.get("manifest", ""))
        or not _same_path(
            _artifact_path(source_meta.get("manifest", "")),
            config.source_test_manifest,
        )
        or _locked_int(
            source_meta.get("utterance_count"),
            label="source_test.utterance_count",
        )
        != config.expected_source_count
        or str(source_meta.get("manifest_sha256", "")).casefold()
        != bindings["source_test_manifest_sha256"].casefold()
        or source_meta.get("audio_text_inventory_sha256")
        != _canonical_sha256(source_inventory)
    ):
        raise FinalBenchmarkError("Existing lock source-test provenance is invalid")

    noise_inventory = [
        {
            "noise_id": row["noise_id"],
            "audio_sha256": row["audio_sha256"],
            "noise_type": row["noise_type"],
        }
        for row in sorted(noise_rows, key=lambda row: str(row["noise_id"]))
    ]
    noise_meta = lock.get("noise", {})
    locked_registry = noise_lock.get("registry", {})
    locked_test_split = noise_lock.get("splits", {}).get("test", {})
    if (
        not isinstance(noise_meta, Mapping)
        or not isinstance(locked_registry, Mapping)
        or not isinstance(locked_test_split, Mapping)
        or not _is_portable_reference(noise_meta.get("split_lock", ""))
        or not _is_portable_reference(noise_meta.get("test_manifest", ""))
        or not _same_path(
            _artifact_path(noise_meta.get("split_lock", "")),
            config.noise_split_lock,
        )
        or not _same_path(
            _artifact_path(noise_meta.get("test_manifest", "")),
            _artifact_path(locked_test_split.get("manifest", "")),
        )
        or str(noise_meta.get("split_lock_sha256", "")).casefold()
        != bindings["noise_split_lock_sha256"].casefold()
        or noise_meta.get("registry_manifest_sha256")
        != locked_registry.get("manifest_sha256")
        or noise_meta.get("test_manifest_sha256")
        != locked_test_split.get("manifest_sha256")
        or noise_meta.get("partition") != "test"
        or _locked_int(noise_meta.get("file_count"), label="noise.file_count")
        != len(noise_rows)
        or noise_meta.get("audio_inventory_sha256")
        != _canonical_sha256(noise_inventory)
    ):
        raise FinalBenchmarkError("Existing lock MUSAN-test provenance is invalid")

    recomputed_audit = _audit_rows(
        rows,
        source_rows=source_rows,
        noise_rows=noise_rows,
        snrs=config.snrs,
        output_audio_dir=config.output_audio_dir,
        forbidden_ids=forbidden_ids,
        forbidden_hashes=forbidden_hashes,
    )
    recomputed_audit_payload = _csv_bytes(recomputed_audit, FINAL_AUDIT_COLUMNS)
    audit_meta = lock.get("audit", {})
    if not isinstance(audit_meta, Mapping) or not _is_portable_reference(
        audit_meta.get("path", "")
    ):
        raise FinalBenchmarkError("Existing final benchmark audit path is invalid")
    if not _same_path(
        _artifact_path(audit_meta.get("path", "")), config.protocol_audit
    ) or (
        config.protocol_audit.read_bytes() != recomputed_audit_payload
        or _sha256_bytes(recomputed_audit_payload) != audit_meta.get("sha256")
    ):
        raise FinalBenchmarkError("Existing final benchmark audit mismatch")
    if (
        len(recomputed_audit)
        != _locked_int(audit_meta.get("checks"), label="audit.checks")
        or _locked_int(
            audit_meta.get("failed_checks"), label="audit.failed_checks"
        )
        != 0
    ):
        raise FinalBenchmarkError("Existing final benchmark audit failed")
    return {"status": "verified_existing", "rows": len(rows), "lock": lock}


def verify_final_benchmark_lock(
    lock_path: str | Path,
    *,
    expected_lock_sha256: str,
    expected_manifest: str | Path,
    expected_manifest_sha256: str,
    expected_rows: int,
    split_lock_sha256: str,
    decision_lock_sha256: str,
    source_test_manifest_sha256: str,
    method_lock_sha256: str,
    method_identity_sha256: str,
    noise_split_lock_sha256: str,
    noise_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify final-benchmark metadata without opening its manifest or audio.

    This is the authorization boundary shared by final inference runners.  It
    checks the exact lock digest and every transitive protocol binding first.
    Callers may open the returned manifest only after this function succeeds.
    """

    digest_fields = {
        "expected_lock_sha256": expected_lock_sha256,
        "expected_manifest_sha256": expected_manifest_sha256,
        "split_lock_sha256": split_lock_sha256,
        "decision_lock_sha256": decision_lock_sha256,
        "source_test_manifest_sha256": source_test_manifest_sha256,
        "method_lock_sha256": method_lock_sha256,
        "method_identity_sha256": method_identity_sha256,
        "noise_split_lock_sha256": noise_split_lock_sha256,
    }
    invalid = [name for name, value in digest_fields.items() if not _is_sha256(value)]
    if invalid:
        raise FinalBenchmarkError(
            "Invalid SHA-256 value(s) for final benchmark authorization: "
            + ", ".join(invalid)
        )
    try:
        row_count = int(expected_rows)
    except (TypeError, ValueError) as exc:
        raise FinalBenchmarkError("Expected final benchmark row count is invalid") from exc
    if row_count != FINAL_ROW_COUNT:
        raise FinalBenchmarkError(
            f"Formal final benchmark requires exactly {FINAL_ROW_COUNT} rows"
        )

    lock_file = Path(lock_path)
    manifest_file = Path(expected_manifest)
    _require_portable_path(lock_file, label="final benchmark lock")
    _require_portable_path(manifest_file, label="final benchmark manifest")
    if not lock_file.is_file():
        raise FileNotFoundError(f"Final benchmark lock does not exist: {lock_file}")
    actual_lock_hash = sha256_file(lock_file)
    if actual_lock_hash != str(expected_lock_sha256).casefold():
        raise FinalBenchmarkError("Final benchmark lock SHA-256 has changed")

    lock = _read_json(lock_file)
    if (
        lock.get("protocol_version") != FINAL_BENCHMARK_VERSION
        or lock.get("status") != "LOCKED"
        or lock.get("selection_eligible") is not False
        or lock.get("final_test_eligible") is not True
    ):
        raise FinalBenchmarkError("Final benchmark lock status/policy is invalid")
    expected_bindings = {
        "split_lock_sha256": split_lock_sha256,
        "decision_lock_sha256": decision_lock_sha256,
        "source_test_manifest_sha256": source_test_manifest_sha256,
        "method_lock_sha256": method_lock_sha256,
        "method_identity_sha256": method_identity_sha256,
        "noise_split_lock_sha256": noise_split_lock_sha256,
    }
    for field, expected in expected_bindings.items():
        if str(lock.get(field, "")).casefold() != str(expected).casefold():
            raise FinalBenchmarkError(
                f"Final benchmark lock binds another {field}"
            )

    expected_builder = {
        "algorithm": FINAL_BENCHMARK_ALGORITHM,
        "seed": FINAL_SEED,
        "snrs_db": list(FINAL_SNRS),
        "sample_rate": FINAL_SAMPLE_RATE,
        "peak_limit": FINAL_PEAK_LIMIT,
        "include_clean": True,
        "expected_source_count": FINAL_SOURCE_COUNT,
        "expected_row_count": FINAL_ROW_COUNT,
        "audio_container": "WAV",
        "audio_subtype": "PCM_16",
        "input_sample_rate_policy": "require_exact",
        "channel_policy": "mean_to_mono",
        "snr_measurement": "component_power_after_anti_clip_before_pcm16",
        "clipping_measurement": "pre_scale_over_1_and_stored_full_scale",
        "source_partition": "vivos_test_locked",
        "noise_partition": "musan_test",
    }
    builder = lock.get("builder")
    if (
        not isinstance(builder, Mapping)
        or builder.get("params") != expected_builder
        or builder.get("params_sha256") != _canonical_sha256(expected_builder)
    ):
        raise FinalBenchmarkError("Final benchmark builder contract is invalid")
    if lock.get("schema") != list(FINAL_BENCHMARK_COLUMNS):
        raise FinalBenchmarkError("Final benchmark schema lock is invalid")

    output = lock.get("output")
    if not isinstance(output, Mapping):
        raise FinalBenchmarkError("Final benchmark lock has no output object")
    if not _is_portable_reference(output.get("manifest", "")) or not _is_portable_reference(
        output.get("audio_dir", "")
    ):
        raise FinalBenchmarkError("Final benchmark output paths are not portable")
    if not _same_path(_artifact_path(output["manifest"]), manifest_file):
        raise FinalBenchmarkError("Final benchmark lock binds another manifest path")
    if str(output.get("manifest_sha256", "")).casefold() != str(
        expected_manifest_sha256
    ).casefold():
        raise FinalBenchmarkError("Final benchmark manifest SHA-256 is not locked")
    if (
        _locked_int(output.get("row_count"), label="output.row_count")
        != FINAL_ROW_COUNT
        or _locked_int(output.get("clean_row_count"), label="output.clean_row_count")
        != FINAL_SOURCE_COUNT
        or _locked_int(output.get("noisy_row_count"), label="output.noisy_row_count")
        != FINAL_ROW_COUNT - FINAL_SOURCE_COUNT
        or output.get("audio_hashes_recorded") is not True
        or not _is_sha256(output.get("audio_inventory_sha256"))
    ):
        raise FinalBenchmarkError("Final benchmark output inventory is invalid")

    source = lock.get("source_test")
    if (
        not isinstance(source, Mapping)
        or not _is_portable_reference(source.get("manifest", ""))
        or str(source.get("manifest_sha256", "")).casefold()
        != str(source_test_manifest_sha256).casefold()
        or _locked_int(source.get("utterance_count"), label="source_test.utterance_count")
        != FINAL_SOURCE_COUNT
        or not _is_sha256(source.get("audio_text_inventory_sha256"))
    ):
        raise FinalBenchmarkError("Final benchmark source-test provenance is invalid")

    noise = lock.get("noise")
    noise_lock = noise_integrity.get("lock")
    if not isinstance(noise, Mapping) or not isinstance(noise_lock, Mapping):
        raise FinalBenchmarkError("Final benchmark MUSAN-test provenance is missing")
    registry = noise_lock.get("registry")
    test_split = noise_lock.get("splits", {}).get("test")
    if not isinstance(registry, Mapping) or not isinstance(test_split, Mapping):
        raise FinalBenchmarkError("Verified MUSAN lock has no registry/test split")
    try:
        verified_test_count = int(test_split.get("file_count", -1))
    except (TypeError, ValueError) as exc:
        raise FinalBenchmarkError("Verified MUSAN-test file count is invalid") from exc
    registry_rows = noise_integrity.get("registry_rows")
    if not isinstance(registry_rows, list) or any(
        not isinstance(row, Mapping) for row in registry_rows
    ):
        raise FinalBenchmarkError("Verified MUSAN registry rows are missing")
    verified_test_rows = [
        row for row in registry_rows if str(row.get("split", "")) == "test"
    ]
    if len(verified_test_rows) != verified_test_count:
        raise FinalBenchmarkError("Verified MUSAN-test inventory count is invalid")
    try:
        verified_noise_inventory_sha256 = _canonical_sha256(
            [
                {
                    "noise_id": str(row["noise_id"]),
                    "audio_sha256": str(row["audio_sha256"]).casefold(),
                    "noise_type": str(row["noise_type"]),
                }
                for row in sorted(
                    verified_test_rows, key=lambda item: str(item["noise_id"])
                )
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalBenchmarkError("Verified MUSAN-test inventory is invalid") from exc
    if (
        not _is_portable_reference(noise.get("split_lock", ""))
        or not _is_portable_reference(noise.get("test_manifest", ""))
        or str(noise.get("split_lock_sha256", "")).casefold()
        != str(noise_split_lock_sha256).casefold()
        or str(noise.get("registry_manifest_sha256", "")).casefold()
        != str(registry.get("manifest_sha256", "")).casefold()
        or str(noise.get("test_manifest_sha256", "")).casefold()
        != str(test_split.get("manifest_sha256", "")).casefold()
        or not _same_path(
            _artifact_path(noise.get("test_manifest", "")),
            _artifact_path(test_split.get("manifest", "")),
        )
        or noise.get("partition") != "test"
        or _locked_int(noise.get("file_count"), label="noise.file_count")
        != verified_test_count
        or str(noise.get("audio_inventory_sha256", "")).casefold()
        != verified_noise_inventory_sha256
    ):
        raise FinalBenchmarkError("Final benchmark MUSAN-test provenance is invalid")

    audit = lock.get("audit")
    if not isinstance(audit, Mapping) or not _is_portable_reference(audit.get("path", "")):
        raise FinalBenchmarkError("Final benchmark audit binding is invalid")
    audit_path = _artifact_path(audit["path"])
    if not audit_path.is_file() or sha256_file(audit_path) != str(
        audit.get("sha256", "")
    ).casefold():
        raise FinalBenchmarkError("Final benchmark audit SHA-256 mismatch")
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if (
        len(audit_rows) != _locked_int(audit.get("checks"), label="audit.checks")
        or _locked_int(audit.get("failed_checks"), label="audit.failed_checks") != 0
        or any(row.get("status") != "PASS" for row in audit_rows)
    ):
        raise FinalBenchmarkError("Final benchmark audit contains failed checks")

    return {
        "lock_sha256": actual_lock_hash,
        "protocol_version": FINAL_BENCHMARK_VERSION,
        "manifest_path": str(manifest_file),
        "manifest_sha256": str(expected_manifest_sha256).casefold(),
        "row_count": FINAL_ROW_COUNT,
        "audio_dir": str(_artifact_path(output["audio_dir"])),
        "audio_inventory_sha256": str(output["audio_inventory_sha256"]).casefold(),
        "audit_sha256": str(audit["sha256"]).casefold(),
    }


def build_final_benchmark(
    config: FinalBenchmarkConfig,
    *,
    overwrite: bool = False,
    decision_verifier: Callable[..., Mapping[str, Any]] | None = None,
    source_verifier: Callable[..., Mapping[str, Any]] | None = None,
    noise_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    method_verifier: Callable[[FinalBenchmarkConfig], Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Build final VIVOS robustness data after a valid method decision unlock."""

    builder_params = _validate_config(config)

    # Security boundary: no VIVOS-test manifest, MUSAN-test manifest/audio, or method
    # verifier is touched before this decision verifier succeeds.
    decision = _authorize_before_test_access(
        config, decision_verifier or _default_decision_verifier
    )
    _validate_resolved_paths_after_authorization(config)

    method = dict((method_verifier or _default_method_verifier)(config))
    method_hash = str(method.get("method_lock_sha256", "")).casefold()
    method_identity = str(method.get("method_identity_sha256", "")).casefold()
    if not _is_sha256(method_hash) or sha256_file(config.method_lock) != method_hash:
        raise FinalBenchmarkError("Method verifier returned another method lock")
    if not _is_sha256(method_identity):
        raise FinalBenchmarkError("Method verifier returned an invalid method identity")
    if (
        method_hash != str(decision["method_lock_sha256"]).casefold()
        or method_identity
        != str(decision["method_identity_sha256"]).casefold()
    ):
        raise FinalBenchmarkError(
            "Verified method lock/identity differs from the authorized decision"
        )
    source_integrity = dict(
        (source_verifier or _default_source_verifier)(
            config.source_test_manifest,
            split_lock_path=config.split_lock,
        )
    )
    source_hash = str(decision["test_manifest_sha256"]).casefold()
    if (
        str(source_integrity.get("manifest_sha256", "")).casefold() != source_hash
        or str(source_integrity.get("split_lock_sha256", "")).casefold()
        != str(decision["split_lock_sha256"]).casefold()
        or int(source_integrity.get("utterance_count", -1))
        != config.expected_source_count
    ):
        raise FinalBenchmarkError("Source verifier returned another unseen test split")
    source_rows = _validate_source_rows(
        config.source_test_manifest,
        expected_hash=source_hash,
        expected_count=config.expected_source_count,
        sample_rate=config.sample_rate,
    )
    noise_verified = dict(
        (noise_verifier or _default_noise_verifier)(config.noise_split_lock)
    )
    (
        noise_lock,
        noise_lock_sha,
        noise_rows,
        forbidden_ids,
        forbidden_hashes,
    ) = _validate_noise_integrity(noise_verified, config)

    decision_hash = str(decision["decision_lock_sha256"]).casefold()
    split_hash = str(decision["split_lock_sha256"]).casefold()
    bindings = {
        "split_lock_sha256": split_hash,
        "decision_lock_sha256": decision_hash,
        "source_test_manifest_sha256": source_hash,
        "noise_split_lock_sha256": noise_lock_sha,
        "method_lock_sha256": method_hash,
    }
    builder_sha = _canonical_sha256(builder_params)
    targets = (
        config.output_manifest,
        config.output_audio_dir,
        config.protocol_lock,
        config.protocol_audit,
    )
    if any(path.exists() for path in targets) and not overwrite:
        if all(path.exists() for path in targets):
            try:
                return _verify_existing(
                    config,
                    bindings=bindings,
                    builder_params=builder_params,
                    builder_sha256=builder_sha,
                    method_identity_sha256=method_identity,
                    source_rows=source_rows,
                    noise_lock=noise_lock,
                    noise_rows=noise_rows,
                    forbidden_ids=forbidden_ids,
                    forbidden_hashes=forbidden_hashes,
                )
            except (OSError, FinalBenchmarkError) as exc:
                raise FinalBenchmarkError(
                    "Existing final benchmark differs from authorized inputs; "
                    "refusing to overwrite"
                ) from exc
        raise FinalBenchmarkError(
            "Partial final benchmark exists; refusing to overwrite without --overwrite"
        )

    stage = config.output_audio_dir.with_name(
        f".{config.output_audio_dir.name}.{os.getpid()}.tmp"
    )
    if stage.exists():
        raise FinalBenchmarkError(f"Stale final benchmark stage: {stage}")
    stage.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    try:
        import numpy as np
        import soundfile as sf

        from .noise import fit_noise, read_audio

        for source in source_rows:
            source_id = str(source["source_utt_id"])
            clean_path = Path(source["audio_path"])
            clean_sha = str(source["audio_sha256"])
            transcript = str(source["transcript"])
            text_sha = str(source["text_sha256"])
            duration = float(source["duration_seconds"])
            rows.append(
                {
                    "utt_id": f"{source_id}_clean",
                    "source_utt_id": source_id,
                    "speaker_id": source.get("speaker_id", ""),
                    "dataset": "vivos",
                    "split": "test",
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
                    "sample_rate": config.sample_rate,
                    "duration_seconds": f"{duration:.9f}",
                    "pre_scale_peak": "",
                    "anti_clip_gain": "",
                    "pre_scale_clipped_samples": "",
                    "stored_peak": "",
                    "clipped_sample_count": 0,
                    "selection_eligible": False,
                    "final_test_eligible": True,
                }
            )
            clean = read_audio(str(clean_path), sr=config.sample_rate)
            for snr in config.snrs:
                target_snr = float(snr)
                item_seed = _stable_seed(config.seed, source_id, target_snr)
                rng = random.Random(item_seed)
                noise_row = noise_rows[rng.randrange(len(noise_rows))]
                noise_path = _artifact_path(noise_row["audio"])
                noise = read_audio(str(noise_path), sr=config.sample_rate)
                fitted = fit_noise(noise, len(clean), rng)
                mixed, measurements = _mix_with_measurements(
                    clean,
                    fitted,
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
                clipped = int(np.count_nonzero(np.abs(stored) >= 1.0))
                rows.append(
                    {
                        "utt_id": f"{source_id}_snr{label}",
                        "source_utt_id": source_id,
                        "speaker_id": source.get("speaker_id", ""),
                        "dataset": "vivos",
                        "split": "test",
                        "condition": "noisy",
                        "audio_path": _display_path(final_path),
                        "audio_sha256": sha256_file(staged_path),
                        "clean_path": _display_path(clean_path),
                        "clean_audio_sha256": clean_sha,
                        "transcript": transcript,
                        "text_sha256": text_sha,
                        "snr": label,
                        "target_snr_db": f"{target_snr:.6f}",
                        "measured_snr_db": (
                            f"{float(measurements['measured_snr_db']):.9f}"
                        ),
                        "noise_id": noise_row["noise_id"],
                        "noise_type": noise_row["noise_type"],
                        "noise_path": _display_path(noise_path),
                        "noise_audio_sha256": noise_row["audio_sha256"],
                        "noise_split": "test",
                        "seed": item_seed,
                        "sample_rate": int(stored_rate),
                        "duration_seconds": f"{len(stored) / float(stored_rate):.9f}",
                        "pre_scale_peak": (
                            f"{float(measurements['pre_scale_peak']):.9f}"
                        ),
                        "anti_clip_gain": (
                            f"{float(measurements['anti_clip_gain']):.9f}"
                        ),
                        "pre_scale_clipped_samples": int(
                            measurements["pre_scale_clipped_samples"]
                        ),
                        "stored_peak": f"{stored_peak:.9f}",
                        "clipped_sample_count": clipped,
                        "selection_eligible": False,
                        "final_test_eligible": True,
                    }
                )

        rows = sorted(rows, key=lambda row: str(row["utt_id"]))
        audit = _audit_rows(
            rows,
            source_rows=source_rows,
            noise_rows=noise_rows,
            snrs=config.snrs,
            output_audio_dir=config.output_audio_dir,
            forbidden_ids=forbidden_ids,
            forbidden_hashes=forbidden_hashes,
            staged_audio_dir=stage,
        )
        manifest_payload = _jsonl_bytes(rows)
        audit_payload = _csv_bytes(audit, FINAL_AUDIT_COLUMNS)
        audio_inventory = [
            {"utt_id": row["utt_id"], "audio_sha256": row["audio_sha256"]}
            for row in rows
        ]
        source_inventory = [
            {
                "source_utt_id": row["source_utt_id"],
                "audio_sha256": row["audio_sha256"],
                "text_sha256": row["text_sha256"],
            }
            for row in source_rows
        ]
        noise_inventory = [
            {
                "noise_id": row["noise_id"],
                "audio_sha256": row["audio_sha256"],
                "noise_type": row["noise_type"],
            }
            for row in sorted(noise_rows, key=lambda row: str(row["noise_id"]))
        ]
        lock = {
            "protocol_version": FINAL_BENCHMARK_VERSION,
            "status": "LOCKED",
            "selection_eligible": False,
            "final_test_eligible": True,
            **bindings,
            "method_identity_sha256": method_identity,
            "source_test": {
                "manifest": _display_path(config.source_test_manifest),
                "manifest_sha256": source_hash,
                "utterance_count": len(source_rows),
                "audio_text_inventory_sha256": _canonical_sha256(source_inventory),
            },
            "noise": {
                "split_lock": _display_path(config.noise_split_lock),
                "split_lock_sha256": noise_lock_sha,
                "registry_manifest_sha256": noise_lock["registry"][
                    "manifest_sha256"
                ],
                "test_manifest": _display_path(
                    _artifact_path(noise_lock["splits"]["test"]["manifest"])
                ),
                "test_manifest_sha256": noise_lock["splits"]["test"][
                    "manifest_sha256"
                ],
                "partition": "test",
                "file_count": len(noise_rows),
                "audio_inventory_sha256": _canonical_sha256(noise_inventory),
            },
            "builder": {
                "params": builder_params,
                "params_sha256": builder_sha,
            },
            "schema": list(FINAL_BENCHMARK_COLUMNS),
            "output": {
                "manifest": _display_path(config.output_manifest),
                "manifest_sha256": _sha256_bytes(manifest_payload),
                "row_count": len(rows),
                "clean_row_count": sum(row["condition"] == "clean" for row in rows),
                "noisy_row_count": sum(row["condition"] == "noisy" for row in rows),
                "audio_dir": _display_path(config.output_audio_dir),
                "audio_hashes_recorded": True,
                "audio_inventory_sha256": _canonical_sha256(audio_inventory),
            },
            "audit": {
                "path": _display_path(config.protocol_audit),
                "sha256": _sha256_bytes(audit_payload),
                "checks": len(audit),
                "failed_checks": 0,
            },
        }
        lock_payload = (
            json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        _commit_transaction(
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


__all__ = [
    "FINAL_BENCHMARK_ALGORITHM",
    "FINAL_BENCHMARK_COLUMNS",
    "FINAL_BENCHMARK_VERSION",
    "FINAL_PEAK_LIMIT",
    "FINAL_ROW_COUNT",
    "FINAL_SAMPLE_RATE",
    "FINAL_SEED",
    "FINAL_SNRS",
    "FINAL_SOURCE_COUNT",
    "FinalBenchmarkConfig",
    "FinalBenchmarkError",
    "build_final_benchmark",
    "sha256_file",
    "verify_final_benchmark_lock",
]
