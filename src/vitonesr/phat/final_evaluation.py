"""Decision-authorized LoRA inference on the locked paper-v2 final benchmark.

The security boundary is deliberate: split/method/noise/decision/final-benchmark
locks are verified before this module opens the derived manifest, any benchmark
audio, or a model for inference.
"""

from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
import math
import os
import shutil
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.vitonesr.analysis import METRIC_VERSION, compute_aligned_metric_result
from src.vitonesr.final_benchmark import (
    FINAL_BENCHMARK_COLUMNS,
    FINAL_BENCHMARK_VERSION,
    FINAL_ROW_COUNT,
    FINAL_SAMPLE_RATE,
    FINAL_SNRS,
    FINAL_SOURCE_COUNT,
    verify_portable_final_benchmark_bundle,
)
from src.vitonesr.noise_protocol import verify_noise_split_lock

from .config import load_experiment_config
from .evaluation import resolve_checkpoint
from .method_contract import (
    verify_checkpoint_method_binding,
    verify_method_lock,
)
from .protocol import (
    canonical_sha256,
    is_sha256,
    resolve_locked_roles,
    sha256_file,
    source_test_evaluation_contract_payload,
    source_test_evaluation_contract_sha256,
    verify_checkpoint_config,
    verify_test_configuration_locked,
    verify_test_decision_lock,
)


ROOT = Path(__file__).resolve().parents[3]
SUITE_VERSION = "paper_v2_final_lora_suite_v1"
PROVENANCE_VERSION = "paper_v2_final_lora_prediction_v1"
AGGREGATE_VERSION = "paper_v2_final_lora_aggregate_v1"
PARTIAL_VERSION = "paper_v2_final_lora_partial_v1"
RECOVERY_VERSION = "paper_v2_final_lora_recovery_v1"
ROLE_ORDER = ("ordinary_baseline", "selected_method", "locked_control")
PREDICTION_COLUMNS = (
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
)
METRIC_NAMES = ("wer", "cer", "ter", "der", "fcer", "swdr")
AGGREGATE_COLUMNS = (
    "role",
    "configuration_id",
    "method_id",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "dataset",
    "split",
    "snr",
    "noise_type",
    "n",
    "metric_version",
    "wer",
    "cer",
    "ter",
    "der",
    "fcer",
    "swdr",
    "decision_lock_sha256",
    "method_lock_sha256",
    "noise_split_lock_sha256",
    "final_benchmark_lock_sha256",
    "final_manifest_sha256",
    "checkpoint_sha256",
    "prediction_sha256",
)
_NOISY_LABELS = tuple(str(int(value)) for value in FINAL_SNRS)


class FinalLoraProtocolError(ValueError):
    """Raised when final LoRA inference is not completely authorized."""


@dataclass(frozen=True)
class FinalLoraRole:
    role: str
    configuration_id: str
    method_id: str
    train_type: str
    lambda_value: float
    seed: int
    checkpoint_path: Path
    checkpoint_display: str
    checkpoint_sha256: str
    resolved_config_sha256: str
    training_contract_sha256: str
    config: dict[str, Any]
    source_test_contract: dict[str, Any]
    source_test_contract_sha256: str


@dataclass(frozen=True)
class FinalLoraAuthorization:
    split_lock_sha256: str
    decision_lock_sha256: str
    method_lock_sha256: str
    method_identity_sha256: str
    noise_split_lock_sha256: str
    final_benchmark_lock_sha256: str
    final_manifest: Path
    final_manifest_sha256: str
    final_rows: int
    final_audio_inventory_sha256: str
    source_test_manifest: str
    source_test_manifest_sha256: str
    source_test_rows: int
    roles: tuple[FinalLoraRole, ...]
    inference_contract: dict[str, Any]
    method_integrity: dict[str, str]
    runtime_config_sha256: str
    runtime_config_path: str


def _repo_path(value: object, *, label: str) -> Path:
    raw = str(value).strip()
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not raw
        or Path(raw).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or raw.startswith(("~", "//", "\\\\"))
        or "\\" in raw
        or ".." in posix.parts
    ):
        raise FinalLoraProtocolError(
            f"{label} must be a portable repository-relative POSIX path"
        )
    resolved = (ROOT / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise FinalLoraProtocolError(f"{label} escapes the repository") from exc
    return resolved


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FinalLoraProtocolError(
            f"Formal provenance cannot contain an external path: {path}"
        ) from exc


def _require_hash(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise FinalLoraProtocolError(
            f"{label} must be a concrete 64-character SHA-256"
        )
    return str(value).strip().casefold()


def load_final_lora_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Final LoRA suite config does not exist: {source}")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FinalLoraProtocolError(f"Invalid final LoRA YAML: {source}") from exc
    if not isinstance(value, dict):
        raise FinalLoraProtocolError("Final LoRA suite config must be an object")
    validate_final_lora_config(value)
    resolved = source.resolve()
    try:
        display = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FinalLoraProtocolError(
            "Formal final LoRA config must be inside the repository"
        ) from exc
    value["_runtime_config_sha256"] = sha256_file(source)
    value["_runtime_config_path"] = display
    return value


def validate_final_lora_config(config: Mapping[str, Any]) -> None:
    if config.get("suite_version") != SUITE_VERSION:
        raise FinalLoraProtocolError(
            f"suite_version must be {SUITE_VERSION}"
        )
    protocol = config.get("protocol")
    benchmark = config.get("benchmark")
    output = config.get("output")
    runtime = config.get("runtime")
    if not all(isinstance(value, Mapping) for value in (protocol, benchmark, output, runtime)):
        raise FinalLoraProtocolError(
            "protocol, benchmark, output, and runtime must be objects"
        )
    if protocol.get("formal") is not True or protocol.get("final_test_unlocked") is not True:
        raise FinalLoraProtocolError(
            "Formal final LoRA inference requires an explicitly unlocked protocol"
        )
    path_fields = (
        "split_lock",
        "decision_lock",
        "method_lock",
        "method_config",
        "noise_split_lock",
        "final_benchmark_lock",
    )
    for field in path_fields:
        _repo_path(protocol.get(field, ""), label=f"protocol.{field}")
    hash_fields = (
        "expected_split_lock_sha256",
        "expected_decision_lock_sha256",
        "expected_method_lock_sha256",
        "expected_noise_split_lock_sha256",
        "expected_final_benchmark_lock_sha256",
    )
    for field in hash_fields:
        _require_hash(protocol.get(field), label=f"protocol.{field}")
    manifest = _repo_path(benchmark.get("manifest", ""), label="benchmark.manifest")
    _require_hash(
        benchmark.get("expected_manifest_sha256"),
        label="benchmark.expected_manifest_sha256",
    )
    if int(benchmark.get("expected_rows", -1)) != FINAL_ROW_COUNT:
        raise FinalLoraProtocolError(
            f"benchmark.expected_rows must be {FINAL_ROW_COUNT}"
        )
    if benchmark.get("verify_audio_sha256") is not True:
        raise FinalLoraProtocolError("Final benchmark audio SHA-256 verification is mandatory")
    output_dir = _repo_path(output.get("directory", ""), label="output.directory")
    if manifest == output_dir or manifest in output_dir.parents or output_dir in manifest.parents:
        raise FinalLoraProtocolError("Final outputs must not overlap the benchmark manifest")
    if str(output.get("aggregate_filename", "")) != "final_lora_results.csv":
        raise FinalLoraProtocolError(
            "output.aggregate_filename must be final_lora_results.csv"
        )
    device = str(runtime.get("device", "auto")).casefold()
    if device not in {"auto", "cpu", "cuda"}:
        raise FinalLoraProtocolError("runtime.device must be auto, cpu, or cuda")
    if runtime.get("verify_method_audio_sha256") not in {True, False}:
        raise FinalLoraProtocolError(
            "runtime.verify_method_audio_sha256 must be boolean"
        )


def _load_saved_checkpoint_config(checkpoint_root: Path) -> dict[str, Any]:
    path = checkpoint_root / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint has no resolved config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise FinalLoraProtocolError(
            f"Checkpoint resolved config must be an object: {path}"
        )
    return value


def _default_role_verifier(
    raw: Mapping[str, Any],
    *,
    role: str,
    decision: Mapping[str, Any],
    method_integrity: Mapping[str, str],
) -> FinalLoraRole:
    checkpoint_text = str(raw.get("checkpoint_path", "")).strip()
    checkpoint_candidate = _repo_path(
        checkpoint_text, label=f"decision.{role}.checkpoint_path"
    )
    checkpoint_root, _ = resolve_checkpoint(checkpoint_candidate)
    if checkpoint_root.resolve() != checkpoint_candidate.resolve():
        raise FinalLoraProtocolError(
            f"Decision role {role} must name the exact inference checkpoint root"
        )
    saved_config = _load_saved_checkpoint_config(checkpoint_root)
    identity = verify_checkpoint_config(checkpoint_root, saved_config)
    verify_checkpoint_method_binding(checkpoint_root, method_integrity)
    matched = verify_test_configuration_locked(
        decision,
        config=saved_config,
        checkpoint_identity=identity,
    )
    if matched.get("role") != role or matched.get("configuration_id") != raw.get(
        "configuration_id"
    ):
        raise FinalLoraProtocolError(
            f"Resolved checkpoint does not match the decision role {role}"
        )
    contract = source_test_evaluation_contract_payload(
        saved_config,
        source_manifest=str(decision["test_manifest"]),
        source_manifest_sha256=str(decision["test_manifest_sha256"]),
        source_rows=int(decision["test_utterance_count"]),
    )
    contract_hash = source_test_evaluation_contract_sha256(
        saved_config,
        source_manifest=str(decision["test_manifest"]),
        source_manifest_sha256=str(decision["test_manifest_sha256"]),
        source_rows=int(decision["test_utterance_count"]),
    )
    if contract_hash not in set(decision["allowed_test_evaluation_contract_sha256"]):
        raise FinalLoraProtocolError(
            f"Decision did not pre-register the inference contract for role {role}"
        )
    return FinalLoraRole(
        role=role,
        configuration_id=str(raw["configuration_id"]),
        method_id=str(raw["method_id"]),
        train_type=str(raw["train_type"]),
        lambda_value=float(raw["lambda"]),
        seed=int(raw["seed"]),
        checkpoint_path=checkpoint_root,
        checkpoint_display=_display_path(checkpoint_root),
        checkpoint_sha256=str(identity["checkpoint_sha256"]).casefold(),
        resolved_config_sha256=str(identity["resolved_config_sha256"]).casefold(),
        training_contract_sha256=str(identity["training_contract_sha256"]).casefold(),
        config=saved_config,
        source_test_contract=contract,
        source_test_contract_sha256=contract_hash,
    )


def authorize_final_lora(
    config: Mapping[str, Any],
    *,
    decision_verifier: Callable[..., Mapping[str, Any]] = verify_test_decision_lock,
    method_config_loader: Callable[[str | Path], Mapping[str, Any]] = load_experiment_config,
    method_verifier: Callable[..., Mapping[str, str]] = verify_method_lock,
    noise_verifier: Callable[..., Mapping[str, Any]] = verify_noise_split_lock,
    benchmark_verifier: Callable[..., Mapping[str, Any]] = (
        verify_portable_final_benchmark_bundle
    ),
    role_verifier: Callable[..., FinalLoraRole] = _default_role_verifier,
) -> FinalLoraAuthorization:
    """Verify every lock before returning any final-data/model access grant."""

    validate_final_lora_config(config)
    protocol = config["protocol"]
    benchmark = config["benchmark"]
    split_path = _repo_path(protocol["split_lock"], label="protocol.split_lock")
    decision_path = _repo_path(protocol["decision_lock"], label="protocol.decision_lock")
    method_path = _repo_path(protocol["method_lock"], label="protocol.method_lock")
    method_config_path = _repo_path(
        protocol["method_config"], label="protocol.method_config"
    )
    noise_path = _repo_path(
        protocol["noise_split_lock"], label="protocol.noise_split_lock"
    )
    final_lock_path = _repo_path(
        protocol["final_benchmark_lock"], label="protocol.final_benchmark_lock"
    )
    manifest_path = _repo_path(benchmark["manifest"], label="benchmark.manifest")

    expected_split = _require_hash(
        protocol["expected_split_lock_sha256"],
        label="protocol.expected_split_lock_sha256",
    )
    expected_decision = _require_hash(
        protocol["expected_decision_lock_sha256"],
        label="protocol.expected_decision_lock_sha256",
    )
    expected_method = _require_hash(
        protocol["expected_method_lock_sha256"],
        label="protocol.expected_method_lock_sha256",
    )
    expected_noise = _require_hash(
        protocol["expected_noise_split_lock_sha256"],
        label="protocol.expected_noise_split_lock_sha256",
    )
    expected_final = _require_hash(
        protocol["expected_final_benchmark_lock_sha256"],
        label="protocol.expected_final_benchmark_lock_sha256",
    )
    expected_manifest = _require_hash(
        benchmark["expected_manifest_sha256"],
        label="benchmark.expected_manifest_sha256",
    )

    # First gate: a reviewed decision must authorize the sealed source test.
    decision = dict(
        decision_verifier(
            split_lock_path=split_path,
            decision_lock_path=decision_path,
            verify_checkpoints=False,
        )
    )
    if (
        str(decision.get("split_lock_sha256", "")).casefold() != expected_split
        or str(decision.get("decision_lock_sha256", "")).casefold()
        != expected_decision
        or str(decision.get("method_lock_sha256", "")).casefold()
        != expected_method
    ):
        raise FinalLoraProtocolError("Decision lock/config hash binding mismatch")
    source_test_manifest_path = _repo_path(
        decision.get("test_manifest", ""),
        label="decision.test_manifest",
    )
    source_test_manifest = _display_path(source_test_manifest_path)
    if str(decision.get("test_manifest", "")) != source_test_manifest:
        raise FinalLoraProtocolError(
            "decision.test_manifest must already be canonical repository-relative POSIX"
        )

    # Method and MUSAN locks authorize the selected LoRA roles.  The benchmark
    # itself is a method-independent, self-contained data bundle that may have
    # been prepared before the lambda decision.
    method_config = dict(method_config_loader(method_config_path))
    method = dict(
        method_verifier(
            method_path,
            config=method_config,
            repo_root=ROOT,
            formal=True,
            verify_audio=bool(config["runtime"]["verify_method_audio_sha256"]),
        )
    )
    if (
        str(method.get("method_lock_sha256", "")).casefold() != expected_method
        or str(method.get("method_identity_sha256", "")).casefold()
        != str(decision.get("method_identity_sha256", "")).casefold()
        or str(method.get("protocol_split_lock_sha256", "")).casefold()
        != expected_split
    ):
        raise FinalLoraProtocolError("Verified method identity differs from the decision")
    noise = dict(noise_verifier(noise_path, verify_audio=False))
    if str(noise.get("lock_sha256", "")).casefold() != expected_noise:
        raise FinalLoraProtocolError("Verified MUSAN split lock differs from config")

    # This metadata-only portable verifier must succeed before manifest/audio/model
    # access.  Decision/method bindings stay in this LoRA authorization layer; they
    # are deliberately not embedded in the reusable benchmark lock.
    final = dict(
        benchmark_verifier(
            final_lock_path,
            expected_lock_sha256=expected_final,
            expected_manifest=manifest_path,
            expected_manifest_sha256=expected_manifest,
            expected_rows=int(benchmark["expected_rows"]),
        )
    )
    if str(final.get("lock_sha256", "")).casefold() != expected_final:
        raise FinalLoraProtocolError("Final benchmark verifier returned another lock")

    try:
        raw_roles = resolve_locked_roles(decision)
    except ValueError as exc:
        raise FinalLoraProtocolError(str(exc)) from exc
    roles = tuple(
        role_verifier(
            raw_roles[role],
            role=role,
            decision=decision,
            method_integrity=method,
        )
        for role in ROLE_ORDER
    )
    if tuple(role.role for role in roles) != ROLE_ORDER:
        raise FinalLoraProtocolError("Final suite did not resolve exactly three roles")
    contracts = {role.source_test_contract_sha256 for role in roles}
    contract_payload_hashes = {
        canonical_sha256(role.source_test_contract) for role in roles
    }
    if len(contracts) != 1 or contracts != contract_payload_hashes:
        raise FinalLoraProtocolError(
            "The three locked roles do not share one pre-registered inference contract"
        )
    inference_contract = dict(roles[0].source_test_contract)
    decoding = inference_contract.get("decoding")
    evaluation = inference_contract.get("evaluation")
    if not isinstance(decoding, Mapping) or not isinstance(evaluation, Mapping):
        raise FinalLoraProtocolError("Pre-registered inference contract is malformed")
    if (
        decoding.get("implementation") != "whisper_generate_greedy_v1"
        or decoding.get("do_sample") is not False
        or int(decoding.get("num_beams", 0)) != 1
        or decoding.get("language") != "vi"
        or decoding.get("task") != "transcribe"
        or int(inference_contract.get("effective_audio", {}).get("sample_rate", 0))
        != FINAL_SAMPLE_RATE
        or int(evaluation.get("batch_size", 0)) < 1
        or str(evaluation.get("inference_precision", "")).casefold()
        not in {"fp16", "fp32"}
    ):
        raise FinalLoraProtocolError(
            "Pre-registered inference contract violates the formal greedy decode policy"
        )
    return FinalLoraAuthorization(
        split_lock_sha256=expected_split,
        decision_lock_sha256=expected_decision,
        method_lock_sha256=expected_method,
        method_identity_sha256=str(decision["method_identity_sha256"]).casefold(),
        noise_split_lock_sha256=expected_noise,
        final_benchmark_lock_sha256=expected_final,
        final_manifest=manifest_path,
        final_manifest_sha256=expected_manifest,
        final_rows=int(benchmark["expected_rows"]),
        final_audio_inventory_sha256=str(final["audio_inventory_sha256"]).casefold(),
        source_test_manifest=source_test_manifest,
        source_test_manifest_sha256=str(decision["test_manifest_sha256"]).casefold(),
        source_test_rows=int(decision["test_utterance_count"]),
        roles=roles,
        inference_contract=inference_contract,
        method_integrity=method,
        runtime_config_sha256=(
            _require_hash(
                config["_runtime_config_sha256"],
                label="runtime config SHA-256",
            )
            if config.get("_runtime_config_sha256") is not None
            else canonical_sha256(
                {
                    key: value
                    for key, value in config.items()
                    if not str(key).startswith("_runtime_config_")
                }
            )
        ),
        runtime_config_path=str(config.get("_runtime_config_path", "in_memory")),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FinalLoraProtocolError(
                    f"Invalid final manifest JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise FinalLoraProtocolError(
                    f"Final manifest line {line_number} is not an object"
                )
            rows.append(row)
    return rows


def load_authorized_final_benchmark(
    authorization: FinalLoraAuthorization,
    *,
    verify_audio: bool = True,
) -> list[dict[str, str]]:
    """Open final data only after ``authorize_final_lora`` has succeeded."""

    path = authorization.final_manifest
    if not path.is_file() or sha256_file(path) != authorization.final_manifest_sha256:
        raise FinalLoraProtocolError("Authorized final manifest is missing or changed")
    raw_rows = _read_jsonl(path)
    if len(raw_rows) != authorization.final_rows or len(raw_rows) != FINAL_ROW_COUNT:
        raise FinalLoraProtocolError(
            f"Final manifest has {len(raw_rows)} rows, expected {FINAL_ROW_COUNT}"
        )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    source_conditions: dict[str, set[str]] = defaultdict(set)
    inventory: list[dict[str, str]] = []
    for number, raw in enumerate(raw_rows, start=1):
        if tuple(raw) != FINAL_BENCHMARK_COLUMNS:
            raise FinalLoraProtocolError(
                f"Final manifest schema/order mismatch at row {number}"
            )
        utt_id = str(raw["utt_id"]).strip()
        source_id = str(raw["source_utt_id"]).strip()
        snr = str(raw["snr"]).strip().casefold()
        noise_type = str(raw["noise_type"]).strip()
        audio_sha = _require_hash(
            raw["audio_sha256"], label=f"manifest row {number} audio_sha256"
        )
        if not utt_id or utt_id in seen or not source_id:
            raise FinalLoraProtocolError(
                f"Blank/duplicate final utterance identity at row {number}"
            )
        if (
            raw["dataset"] != "vivos"
            or raw["split"] != "test"
            or raw["selection_eligible"] is not False
            or raw["final_test_eligible"] is not True
            or int(raw["sample_rate"]) != FINAL_SAMPLE_RATE
        ):
            raise FinalLoraProtocolError(
                f"Final-only dataset policy mismatch at row {number}"
            )
        if snr not in {"clean", *_NOISY_LABELS}:
            raise FinalLoraProtocolError(f"Unexpected final SNR at row {number}: {snr}")
        if (snr == "clean") != (noise_type.casefold() == "clean"):
            raise FinalLoraProtocolError(
                f"SNR/noise_type mismatch at row {number}"
            )
        reference = unicodedata.normalize("NFC", str(raw["transcript"]).strip())
        if not reference:
            raise FinalLoraProtocolError(f"Blank reference at row {number}")
        audio_path = _repo_path(raw["audio_path"], label=f"manifest row {number} audio_path")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Final benchmark audio is missing: {audio_path}")
        if verify_audio and sha256_file(audio_path) != audio_sha:
            raise FinalLoraProtocolError(f"Final benchmark audio changed: {audio_path}")
        seen.add(utt_id)
        source_conditions[source_id].add(snr)
        inventory.append({"utt_id": utt_id, "audio_sha256": audio_sha})
        rows.append(
            {
                "utt_id": utt_id,
                "source_utt_id": source_id,
                "dataset": "vivos",
                "audio_path": str(audio_path),
                "audio_sha256": audio_sha,
                "snr": snr,
                "noise_type": noise_type,
                "ref": reference,
            }
        )
    expected_conditions = {"clean", *_NOISY_LABELS}
    if len(source_conditions) != FINAL_SOURCE_COUNT or any(
        conditions != expected_conditions for conditions in source_conditions.values()
    ):
        raise FinalLoraProtocolError("Final manifest is not a complete 460 x 5 design")
    if canonical_sha256(inventory) != authorization.final_audio_inventory_sha256:
        raise FinalLoraProtocolError("Final audio inventory differs from its lock")
    return rows


def _default_predictor(
    role: FinalLoraRole,
    rows: Sequence[Mapping[str, str]],
    inference_contract: Mapping[str, Any],
    *,
    device_arg: str,
    start_index: int = 0,
    on_batch: Callable[[Sequence[str], Mapping[str, Any]], None] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    import torch
    import transformers
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from src.vitonesr.noise import read_audio

    evaluation = inference_contract["evaluation"]
    decoding = inference_contract["decoding"]
    model_config = role.config["model"]
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = str(evaluation["inference_precision"]).casefold()
    if precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError("The pre-registered fp16 contract requires CUDA")
        dtype = torch.float16
    else:
        dtype = torch.float32

    torch.manual_seed(role.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(role.seed)
    local_processor = role.checkpoint_path / "processor"
    processor = WhisperProcessor.from_pretrained(
        str(local_processor),
        language="vi",
        task="transcribe",
    )
    base = WhisperForConditionalGeneration.from_pretrained(
        str(model_config["name_or_path"]),
        revision=str(model_config["revision"]),
    )
    base.config.use_cache = True
    _, adapter_path = resolve_checkpoint(role.checkpoint_path)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    sample_rate = int(inference_contract["effective_audio"]["sample_rate"])
    max_length = int(
        float(inference_contract["effective_audio"]["max_audio_seconds"])
        * sample_rate
    )
    batch_size = int(evaluation["batch_size"])
    if start_index < 0 or start_index > len(rows) or start_index % batch_size != 0:
        raise FinalLoraProtocolError(
            "Final-LoRA resume offset must be an exact inference-batch boundary"
        )
    runtime = {
        "batch_size": batch_size,
        "device_type": device.type,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
    }
    hypotheses: list[str] = []
    try:
        with torch.inference_mode():
            for start in tqdm(
                range(start_index, len(rows), batch_size),
                desc=f"final {role.role} lambda={role.lambda_value:g}",
            ):
                batch = rows[start : start + batch_size]
                waveforms = [
                    read_audio(str(row["audio_path"]), sr=sample_rate)[:max_length]
                    for row in batch
                ]
                features = processor.feature_extractor(
                    waveforms,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                inputs = features.input_features.to(device=device, dtype=dtype)
                attention = features.attention_mask.to(device=device)
                kwargs = {
                    "max_new_tokens": int(decoding["max_new_tokens"]),
                    "language": "vi",
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                }
                try:
                    generated = model.generate(inputs, attention_mask=attention, **kwargs)
                except TypeError:
                    generated = model.generate(
                        inputs,
                        attention_mask=attention,
                        max_new_tokens=kwargs["max_new_tokens"],
                        do_sample=False,
                        num_beams=1,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(
                            language="vi", task="transcribe"
                        ),
                    )
                decoded = [
                    unicodedata.normalize("NFC", value)
                    for value in processor.batch_decode(
                        generated, skip_special_tokens=True
                    )
                ]
                if len(decoded) != len(batch):
                    raise FinalLoraProtocolError(
                        "Decoder returned a hypothesis count different from its batch"
                    )
                hypotheses.extend(decoded)
                if on_batch is not None:
                    on_batch(decoded, runtime)
    finally:
        del model, base, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return hypotheses, runtime


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalLoraProtocolError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalLoraProtocolError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FinalLoraProtocolError(f"{label} must be a JSON object: {path}")
    return value


def _partial_role_directory(output_root: Path, role: FinalLoraRole) -> Path:
    return output_root / f".{role.role}.partial"


def _partial_artifacts(work_directory: Path) -> tuple[Path, Path, Path]:
    return (
        work_directory / "predictions.partial.csv",
        work_directory / "resume.json",
        work_directory / "recovery.json",
    )


def _role_resume_contract(
    role: FinalLoraRole,
    authorization: FinalLoraAuthorization,
) -> dict[str, Any]:
    payload = {
        "contract_version": PARTIAL_VERSION,
        "role": role.role,
        "configuration_id": role.configuration_id,
        "method_id": role.method_id,
        "train_type": role.train_type,
        "lambda": f"{role.lambda_value:g}",
        "seed": role.seed,
        "checkpoint_sha256": role.checkpoint_sha256,
        "resolved_config_sha256": role.resolved_config_sha256,
        "training_contract_sha256": role.training_contract_sha256,
        "source_test_evaluation_contract_sha256": (
            role.source_test_contract_sha256
        ),
        "inference_contract_sha256": canonical_sha256(
            authorization.inference_contract
        ),
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "noise_split_lock_sha256": authorization.noise_split_lock_sha256,
        "final_benchmark_lock_sha256": (
            authorization.final_benchmark_lock_sha256
        ),
        "final_manifest_sha256": authorization.final_manifest_sha256,
        "final_audio_inventory_sha256": (
            authorization.final_audio_inventory_sha256
        ),
        "final_rows": authorization.final_rows,
        "runtime_config_sha256": authorization.runtime_config_sha256,
        "prediction_schema_sha256": canonical_sha256(list(PREDICTION_COLUMNS)),
    }
    return {**payload, "role_contract_sha256": canonical_sha256(payload)}


def _runtime_payload(runtime: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value = json.loads(
        json.dumps(dict(runtime), ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    if not isinstance(value, dict):
        raise FinalLoraProtocolError("Predictor runtime must be a JSON object")
    return value, canonical_sha256(value)


def _state_payload(
    *,
    partial_prediction: Path,
    row_count: int,
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_value, runtime_sha256 = _runtime_payload(runtime)
    return {
        "partial_version": PARTIAL_VERSION,
        **dict(contract),
        "completed_rows": row_count,
        "partial_prediction_sha256": sha256_file(partial_prediction),
        "runtime_environment": runtime_value,
        "runtime_environment_sha256": runtime_sha256,
    }


def _recovery_payload(
    *,
    prediction_payload: bytes,
    row_count: int,
    previous_rows: int,
    previous_sha256: str,
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_value, runtime_sha256 = _runtime_payload(runtime)
    return {
        "recovery_version": RECOVERY_VERSION,
        **dict(contract),
        "completed_rows": row_count,
        "partial_prediction_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "previous_completed_rows": previous_rows,
        "previous_partial_prediction_sha256": previous_sha256,
        "runtime_environment": runtime_value,
        "runtime_environment_sha256": runtime_sha256,
    }


def _validate_contract_binding(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for field, expected in contract.items():
        if str(value.get(field, "")).casefold() != str(expected).casefold():
            raise FinalLoraProtocolError(f"{label} mismatch: {field}")
    runtime = value.get("runtime_environment")
    if not isinstance(runtime, Mapping) or canonical_sha256(runtime) != str(
        value.get("runtime_environment_sha256", "")
    ).casefold():
        raise FinalLoraProtocolError(f"{label} runtime environment is invalid/tampered")


def _validate_prediction_prefix(
    prediction_path: Path,
    role: FinalLoraRole,
    benchmark_rows: Sequence[Mapping[str, str]],
    *,
    require_complete: bool = False,
) -> list[dict[str, str]]:
    rows, columns = _read_prediction(prediction_path)
    if columns != PREDICTION_COLUMNS:
        raise FinalLoraProtocolError(
            f"Partial prediction schema mismatch: {prediction_path}"
        )
    if len(rows) > len(benchmark_rows) or (
        require_complete and len(rows) != len(benchmark_rows)
    ):
        raise FinalLoraProtocolError(
            f"Partial prediction row count mismatch: {prediction_path}"
        )
    metadata = {
        "model": "phowhisper",
        "model_size": "base",
        "train_type": role.train_type,
        "lambda": f"{role.lambda_value:g}",
        "seed": str(role.seed),
    }
    for index, prediction in enumerate(rows):
        benchmark = benchmark_rows[index]
        expected = {
            **metadata,
            "utt_id": benchmark["utt_id"],
            "dataset": benchmark["dataset"],
            "snr": benchmark["snr"],
            "noise_type": benchmark["noise_type"],
            "ref": benchmark["ref"],
        }
        conflicts = [
            field for field, expected_value in expected.items()
            if prediction.get(field) != expected_value
        ]
        if conflicts or prediction.get("hyp", "") != unicodedata.normalize(
            "NFC", prediction.get("hyp", "")
        ):
            raise FinalLoraProtocolError(
                f"Partial prediction is not the exact role/manifest prefix at "
                f"{prediction_path}:{index + 2}; conflicts={conflicts}"
            )
    return rows


def _validate_partial_state(
    state: Mapping[str, Any],
    *,
    partial_prediction: Path,
    row_count: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if state.get("partial_version") != PARTIAL_VERSION:
        raise FinalLoraProtocolError("Unsupported final-LoRA partial state")
    _validate_contract_binding(state, contract, label="Partial state")
    try:
        completed_rows = int(state.get("completed_rows", -1))
    except (TypeError, ValueError) as exc:
        raise FinalLoraProtocolError("Partial state row count is invalid") from exc
    if completed_rows != row_count:
        raise FinalLoraProtocolError("Partial state row count mismatch")
    if str(state.get("partial_prediction_sha256", "")).casefold() != sha256_file(
        partial_prediction
    ):
        raise FinalLoraProtocolError("Partial state prediction hash mismatch")
    return dict(state["runtime_environment"])


def _publish_partial_role(
    *,
    work_directory: Path,
    role: FinalLoraRole,
    benchmark_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    partial_prediction, state_path, recovery_path = _partial_artifacts(work_directory)
    prediction_payload = _csv_bytes(prediction_rows, PREDICTION_COLUMNS)
    desired_hash = hashlib.sha256(prediction_payload).hexdigest()
    if partial_prediction.is_file() and state_path.is_file() and (
        sha256_file(partial_prediction) == desired_hash
    ):
        state = _load_json_object(state_path, label="final-LoRA partial state")
        existing_runtime = _validate_partial_state(
            state,
            partial_prediction=partial_prediction,
            row_count=len(prediction_rows),
            contract=contract,
        )
        if _runtime_payload(existing_runtime)[1] != _runtime_payload(runtime)[1]:
            raise FinalLoraProtocolError(
                "Predictor runtime changed while publishing the same partial prefix"
            )
        return
    previous_rows = 0
    previous_sha256 = ""
    if partial_prediction.exists():
        if not state_path.is_file():
            raise FinalLoraProtocolError(
                "Partial prediction exists without state before publication"
            )
        previous = _validate_prediction_prefix(
            partial_prediction,
            role,
            benchmark_rows,
        )
        _validate_partial_state(
            _load_json_object(state_path, label="final-LoRA partial state"),
            partial_prediction=partial_prediction,
            row_count=len(previous),
            contract=contract,
        )
        previous_rows = len(previous)
        previous_sha256 = sha256_file(partial_prediction)
    receipt = _recovery_payload(
        prediction_payload=prediction_payload,
        row_count=len(prediction_rows),
        previous_rows=previous_rows,
        previous_sha256=previous_sha256,
        contract=contract,
        runtime=runtime,
    )
    _atomic_write_bytes(recovery_path, _json_bytes(receipt))
    _atomic_write_bytes(partial_prediction, prediction_payload)
    if sha256_file(partial_prediction) != receipt["partial_prediction_sha256"]:
        raise RuntimeError("Final-LoRA partial CSV differs from its recovery receipt")
    _atomic_write_bytes(
        state_path,
        _json_bytes(
            _state_payload(
                partial_prediction=partial_prediction,
                row_count=len(prediction_rows),
                contract=contract,
                runtime=runtime,
            )
        ),
    )
    recovery_path.unlink()


def _validate_recovery_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if receipt.get("recovery_version") != RECOVERY_VERSION:
        raise FinalLoraProtocolError("Unsupported final-LoRA recovery receipt")
    _validate_contract_binding(receipt, contract, label="Recovery receipt")
    for field in ("completed_rows", "previous_completed_rows"):
        try:
            if int(receipt.get(field, -1)) < 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise FinalLoraProtocolError(
                f"Recovery receipt has invalid {field}"
            ) from exc
    if not is_sha256(receipt.get("partial_prediction_sha256")):
        raise FinalLoraProtocolError("Recovery receipt prediction hash is invalid")
    previous_hash = str(
        receipt.get("previous_partial_prediction_sha256", "")
    ).casefold()
    if previous_hash and not is_sha256(previous_hash):
        raise FinalLoraProtocolError("Recovery receipt previous hash is invalid")
    current_rows = int(receipt["completed_rows"])
    previous_rows = int(receipt["previous_completed_rows"])
    if current_rows <= previous_rows:
        raise FinalLoraProtocolError("Recovery receipt is not a forward row transition")
    return dict(receipt["runtime_environment"])


def _load_or_recover_partial_role(
    *,
    work_directory: Path,
    role: FinalLoraRole,
    authorization: FinalLoraAuthorization,
    benchmark_rows: Sequence[Mapping[str, str]],
    resume: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    contract = _role_resume_contract(role, authorization)
    if not work_directory.exists():
        work_directory.mkdir(parents=True)
        return [], {}
    if not work_directory.is_dir():
        raise FinalLoraProtocolError(
            f"Partial role artifact is not a directory: {work_directory}"
        )
    if not resume:
        raise FileExistsError(
            f"Partial role output exists; use --resume after review: {work_directory}"
        )
    partial_prediction, state_path, recovery_path = _partial_artifacts(work_directory)
    allowed = {path.name for path in (partial_prediction, state_path, recovery_path)}
    unexpected = sorted(path.name for path in work_directory.iterdir() if path.name not in allowed)
    if unexpected:
        raise FinalLoraProtocolError(
            f"Partial role directory contains unexpected artifacts: {unexpected}"
        )

    if recovery_path.exists():
        receipt = _load_json_object(
            recovery_path, label="final-LoRA recovery receipt"
        )
        receipt_runtime = _validate_recovery_receipt(receipt, contract=contract)
        current_rows = int(receipt["completed_rows"])
        previous_rows = int(receipt["previous_completed_rows"])
        current_hash = str(receipt["partial_prediction_sha256"]).casefold()
        previous_hash = str(
            receipt.get("previous_partial_prediction_sha256", "")
        ).casefold()
        if not partial_prediction.exists():
            if state_path.exists() or previous_rows != 0 or previous_hash:
                raise FinalLoraProtocolError(
                    "Recovery receipt is inconsistent with missing partial CSV"
                )
            recovery_path.unlink()
        else:
            rows = _validate_prediction_prefix(
                partial_prediction, role, benchmark_rows
            )
            actual_hash = sha256_file(partial_prediction)
            if actual_hash == current_hash and len(rows) == current_rows:
                if state_path.exists():
                    state = _load_json_object(
                        state_path, label="final-LoRA partial state"
                    )
                    try:
                        current_runtime = _validate_partial_state(
                            state,
                            partial_prediction=partial_prediction,
                            row_count=current_rows,
                            contract=contract,
                        )
                        if _runtime_payload(current_runtime)[1] != _runtime_payload(
                            receipt_runtime
                        )[1]:
                            raise FinalLoraProtocolError(
                                "Recovery receipt runtime differs from committed state"
                            )
                    except FinalLoraProtocolError:
                        _validate_contract_binding(
                            state, contract, label="Recovery previous state"
                        )
                        if int(state.get("completed_rows", -1)) != previous_rows or str(
                            state.get("partial_prediction_sha256", "")
                        ).casefold() != previous_hash:
                            raise FinalLoraProtocolError(
                                "Recovery previous state is stale or tampered"
                            )
                        if _runtime_payload(
                            dict(state["runtime_environment"])
                        )[1] != _runtime_payload(receipt_runtime)[1]:
                            raise FinalLoraProtocolError(
                                "Runtime changed across the recovery transition"
                            )
                        _atomic_write_bytes(
                            state_path,
                            _json_bytes(
                                _state_payload(
                                    partial_prediction=partial_prediction,
                                    row_count=current_rows,
                                    contract=contract,
                                    runtime=receipt_runtime,
                                )
                            ),
                        )
                else:
                    # The write-ahead receipt binds both the previous and new
                    # transition plus exact new CSV bytes.  It is therefore
                    # sufficient to rebuild a missing state file at any batch,
                    # including the first publication where no old state exists.
                    _atomic_write_bytes(
                        state_path,
                        _json_bytes(
                            _state_payload(
                                partial_prediction=partial_prediction,
                                row_count=current_rows,
                                contract=contract,
                                runtime=receipt_runtime,
                            )
                        ),
                    )
                recovery_path.unlink()
            elif actual_hash == previous_hash and len(rows) == previous_rows:
                if not state_path.is_file():
                    raise FinalLoraProtocolError(
                        "Recovery found the previous CSV without its state"
                    )
                previous_runtime = _validate_partial_state(
                    _load_json_object(
                        state_path, label="final-LoRA partial state"
                    ),
                    partial_prediction=partial_prediction,
                    row_count=previous_rows,
                    contract=contract,
                )
                if _runtime_payload(previous_runtime)[1] != _runtime_payload(
                    receipt_runtime
                )[1]:
                    raise FinalLoraProtocolError(
                        "Runtime changed across the pending recovery transition"
                    )
                recovery_path.unlink()
            else:
                raise FinalLoraProtocolError(
                    "Partial role CSV matches neither recovery hash; refusing tamper"
                )

    if partial_prediction.exists() != state_path.exists():
        raise FinalLoraProtocolError(
            "Orphan final-LoRA partial CSV/state cannot be resumed"
        )
    if not partial_prediction.exists():
        return [], {}
    rows = _validate_prediction_prefix(partial_prediction, role, benchmark_rows)
    runtime = _validate_partial_state(
        _load_json_object(state_path, label="final-LoRA partial state"),
        partial_prediction=partial_prediction,
        row_count=len(rows),
        contract=contract,
    )
    if len(rows) >= len(benchmark_rows):
        if len(rows) != len(benchmark_rows):
            raise FinalLoraProtocolError("Partial role has more rows than final benchmark")
    else:
        batch_size = int(authorization.inference_contract["evaluation"]["batch_size"])
        if len(rows) % batch_size != 0:
            raise FinalLoraProtocolError(
                "Partial role row count is not an exact inference-batch boundary"
            )
    return rows, runtime


def _commit_directory(destination: Path, payloads: Mapping[str, bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {destination}")
    stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    stage.mkdir()
    try:
        for name, payload in payloads.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _role_prediction_rows(
    role: FinalLoraRole,
    benchmark_rows: Sequence[Mapping[str, str]],
    hypotheses: Sequence[str],
) -> list[dict[str, Any]]:
    if len(hypotheses) != len(benchmark_rows):
        raise FinalLoraProtocolError(
            f"Role {role.role} returned {len(hypotheses)} hypotheses for "
            f"{len(benchmark_rows)} rows"
        )
    return [
        {
            "utt_id": row["utt_id"],
            "dataset": row["dataset"],
            "model": "phowhisper",
            "model_size": "base",
            "train_type": role.train_type,
            "lambda": f"{role.lambda_value:g}",
            "seed": role.seed,
            "snr": row["snr"],
            "noise_type": row["noise_type"],
            "ref": row["ref"],
            "hyp": unicodedata.normalize("NFC", str(hypothesis)),
        }
        for row, hypothesis in zip(benchmark_rows, hypotheses)
    ]


def _provenance(
    role: FinalLoraRole,
    authorization: FinalLoraAuthorization,
    prediction_payload: bytes,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "provenance_version": PROVENANCE_VERSION,
        "role": role.role,
        "configuration_id": role.configuration_id,
        "method_id": role.method_id,
        "train_type": role.train_type,
        "lambda": role.lambda_value,
        "seed": role.seed,
        "checkpoint": _display_path(
            _repo_path(role.checkpoint_display, label="role.checkpoint")
        ),
        "checkpoint_sha256": role.checkpoint_sha256,
        "resolved_config_sha256": role.resolved_config_sha256,
        "training_contract_sha256": role.training_contract_sha256,
        "source_test_evaluation_contract": role.source_test_contract,
        "source_test_evaluation_contract_sha256": role.source_test_contract_sha256,
        "prediction_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "num_rows": authorization.final_rows,
        "metric_version": METRIC_VERSION,
        "runtime_environment": dict(runtime),
        "split_lock_sha256": authorization.split_lock_sha256,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "method_identity_sha256": authorization.method_identity_sha256,
        "noise_split_lock_sha256": authorization.noise_split_lock_sha256,
        "final_benchmark_lock_sha256": authorization.final_benchmark_lock_sha256,
        "final_manifest": _display_path(authorization.final_manifest),
        "final_manifest_sha256": authorization.final_manifest_sha256,
        "final_audio_inventory_sha256": authorization.final_audio_inventory_sha256,
        "source_test_manifest": _display_path(
            _repo_path(
                authorization.source_test_manifest,
                label="authorization.source_test_manifest",
            )
        ),
        "source_test_manifest_sha256": authorization.source_test_manifest_sha256,
        "source_test_rows": authorization.source_test_rows,
        "runtime_config": _display_path(
            _repo_path(
                authorization.runtime_config_path,
                label="authorization.runtime_config_path",
            )
        ),
        "runtime_config_sha256": authorization.runtime_config_sha256,
    }


def _read_prediction(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), tuple(reader.fieldnames or ())


def _load_completed_role(
    destination: Path,
    role: FinalLoraRole,
    authorization: FinalLoraAuthorization,
    benchmark_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    prediction_path = destination / "predictions.csv"
    provenance_path = destination / "provenance.json"
    if not prediction_path.is_file() or not provenance_path.is_file():
        raise FinalLoraProtocolError(
            f"Incomplete per-role output cannot be resumed: {destination}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or provenance.get("provenance_version") != PROVENANCE_VERSION:
        raise FinalLoraProtocolError(f"Invalid per-role provenance: {provenance_path}")
    expected = {
        "role": role.role,
        "configuration_id": role.configuration_id,
        "method_id": role.method_id,
        "train_type": role.train_type,
        "lambda": role.lambda_value,
        "seed": role.seed,
        "checkpoint": _display_path(
            _repo_path(role.checkpoint_display, label="role.checkpoint")
        ),
        "checkpoint_sha256": role.checkpoint_sha256,
        "resolved_config_sha256": role.resolved_config_sha256,
        "training_contract_sha256": role.training_contract_sha256,
        "source_test_evaluation_contract_sha256": role.source_test_contract_sha256,
        "num_rows": len(benchmark_rows),
        "metric_version": METRIC_VERSION,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "noise_split_lock_sha256": authorization.noise_split_lock_sha256,
        "final_benchmark_lock_sha256": authorization.final_benchmark_lock_sha256,
        "final_manifest": _display_path(authorization.final_manifest),
        "final_manifest_sha256": authorization.final_manifest_sha256,
        "source_test_manifest": _display_path(
            _repo_path(
                authorization.source_test_manifest,
                label="authorization.source_test_manifest",
            )
        ),
        "source_test_manifest_sha256": authorization.source_test_manifest_sha256,
        "runtime_config": _display_path(
            _repo_path(
                authorization.runtime_config_path,
                label="authorization.runtime_config_path",
            )
        ),
        "runtime_config_sha256": authorization.runtime_config_sha256,
    }
    if any(str(provenance.get(field, "")) != str(value) for field, value in expected.items()):
        raise FinalLoraProtocolError(f"Stale/tampered per-role provenance: {destination}")
    if sha256_file(prediction_path) != provenance.get("prediction_sha256"):
        raise FinalLoraProtocolError(f"Prediction hash mismatch: {prediction_path}")
    if provenance.get("prediction_columns") != list(PREDICTION_COLUMNS) or (
        not isinstance(provenance.get("runtime_environment"), Mapping)
    ):
        raise FinalLoraProtocolError(
            f"Invalid schema/runtime in per-role provenance: {provenance_path}"
        )
    rows = _validate_prediction_prefix(
        prediction_path,
        role,
        benchmark_rows,
        require_complete=True,
    )
    return rows, provenance


def _metric_row(
    role: FinalLoraRole,
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    snr: str,
    noise_type: str,
    authorization: FinalLoraAuthorization,
    prediction_sha256: str,
) -> dict[str, Any]:
    result = compute_aligned_metric_result(
        [str(row["ref"]) for row in rows],
        [str(row["hyp"]) for row in rows],
    ).to_dict(include_counts=True)
    value = {
        "role": role.role,
        "configuration_id": role.configuration_id,
        "method_id": role.method_id,
        "model": "phowhisper",
        "model_size": "base",
        "train_type": role.train_type,
        "lambda": f"{role.lambda_value:g}",
        "seed": role.seed,
        "dataset": "vivos",
        "split": split,
        "snr": snr,
        "noise_type": noise_type,
        "n": len(rows),
        "metric_version": METRIC_VERSION,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "noise_split_lock_sha256": authorization.noise_split_lock_sha256,
        "final_benchmark_lock_sha256": authorization.final_benchmark_lock_sha256,
        "final_manifest_sha256": authorization.final_manifest_sha256,
        "checkpoint_sha256": role.checkpoint_sha256,
        "prediction_sha256": prediction_sha256,
    }
    value.update({metric: result[metric] for metric in METRIC_NAMES})
    return value


def _aggregate_role(
    role: FinalLoraRole,
    rows: Sequence[Mapping[str, str]],
    *,
    authorization: FinalLoraAuthorization,
    prediction_sha256: str,
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, str, list[Mapping[str, str]]]] = [
        ("all", "all", "all", list(rows))
    ]
    clean = [row for row in rows if row["snr"] == "clean"]
    noisy = [row for row in rows if row["snr"] != "clean"]
    groups.extend(
        [
            ("clean", "clean", "clean", clean),
            ("noisy", "noisy_all", "all", noisy),
        ]
    )
    by_snr: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    by_noise: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in noisy:
        by_snr[row["snr"]].append(row)
        by_noise[row["noise_type"]].append(row)
    groups.extend(
        ("noisy", snr, "all", by_snr[snr]) for snr in _NOISY_LABELS
    )
    groups.extend(
        ("noisy", "all", noise_type, by_noise[noise_type])
        for noise_type in sorted(by_noise)
    )
    return [
        _metric_row(
            role,
            group,
            split=split,
            snr=snr,
            noise_type=noise_type,
            authorization=authorization,
            prediction_sha256=prediction_sha256,
        )
        for split, snr, noise_type, group in groups
        if group
    ]


def _predictor_supports_incremental_resume(predictor: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(predictor).parameters
    except (TypeError, ValueError):
        return False
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    return {"start_index", "on_batch"}.issubset(parameters)


def _run_or_resume_role(
    *,
    role: FinalLoraRole,
    authorization: FinalLoraAuthorization,
    benchmark_rows: Sequence[Mapping[str, str]],
    output_root: Path,
    device: str,
    resume: bool,
    predictor: Callable[..., tuple[list[str], dict[str, Any]]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    destination = output_root / role.role
    work_directory = _partial_role_directory(output_root, role)
    contract = _role_resume_contract(role, authorization)
    partial_rows, saved_runtime = _load_or_recover_partial_role(
        work_directory=work_directory,
        role=role,
        authorization=authorization,
        benchmark_rows=benchmark_rows,
        resume=resume,
    )
    hypotheses = [row["hyp"] for row in partial_rows]
    start_index = len(hypotheses)
    runtime: dict[str, Any] = dict(saved_runtime)

    def bind_runtime(observed: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal runtime
        candidate, candidate_sha256 = _runtime_payload(observed)
        if runtime and _runtime_payload(runtime)[1] != candidate_sha256:
            raise FinalLoraProtocolError(
                f"Runtime environment changed while resuming role {role.role}"
            )
        runtime = candidate
        return candidate

    if start_index < len(benchmark_rows):
        if _predictor_supports_incremental_resume(predictor):
            streamed: list[str] = []
            batch_size = int(
                authorization.inference_contract["evaluation"]["batch_size"]
            )

            def on_batch(
                batch_hypotheses: Sequence[str],
                observed_runtime: Mapping[str, Any],
            ) -> None:
                normalized = [
                    unicodedata.normalize("NFC", str(value))
                    for value in batch_hypotheses
                ]
                remaining = len(benchmark_rows) - start_index - len(streamed)
                expected_size = min(batch_size, remaining)
                if len(normalized) != expected_size:
                    raise FinalLoraProtocolError(
                        f"Incremental predictor returned {len(normalized)} rows; "
                        f"expected batch size {expected_size}"
                    )
                streamed.extend(normalized)
                current_hypotheses = hypotheses + streamed
                current_rows = _role_prediction_rows(
                    role,
                    benchmark_rows[: len(current_hypotheses)],
                    current_hypotheses,
                )
                _publish_partial_role(
                    work_directory=work_directory,
                    role=role,
                    benchmark_rows=benchmark_rows,
                    prediction_rows=current_rows,
                    contract=contract,
                    runtime=bind_runtime(observed_runtime),
                )

            returned, observed_runtime = predictor(
                role,
                benchmark_rows,
                authorization.inference_contract,
                device_arg=device,
                start_index=start_index,
                on_batch=on_batch,
            )
            returned_normalized = [
                unicodedata.normalize("NFC", str(value)) for value in returned
            ]
            bind_runtime(observed_runtime)
            if streamed:
                if returned_normalized != streamed:
                    raise FinalLoraProtocolError(
                        "Incremental predictor return differs from streamed batches"
                    )
                hypotheses.extend(streamed)
            else:
                hypotheses.extend(returned_normalized)
        else:
            returned, observed_runtime = predictor(
                role,
                benchmark_rows[start_index:],
                authorization.inference_contract,
                device_arg=device,
            )
            hypotheses.extend(
                unicodedata.normalize("NFC", str(value)) for value in returned
            )
            bind_runtime(observed_runtime)

        prediction_rows = _role_prediction_rows(role, benchmark_rows, hypotheses)
        _publish_partial_role(
            work_directory=work_directory,
            role=role,
            benchmark_rows=benchmark_rows,
            prediction_rows=prediction_rows,
            contract=contract,
            runtime=runtime,
        )
    elif not runtime:
        raise FinalLoraProtocolError(
            f"Complete partial role has no bound runtime environment: {role.role}"
        )

    prediction_rows = _role_prediction_rows(role, benchmark_rows, hypotheses)
    prediction_payload = _csv_bytes(prediction_rows, PREDICTION_COLUMNS)
    provenance = _provenance(role, authorization, prediction_payload, runtime)
    _commit_directory(
        destination,
        {
            "predictions.csv": prediction_payload,
            "provenance.json": _json_bytes(provenance),
        },
    )
    completed = _load_completed_role(
        destination, role, authorization, benchmark_rows
    )
    shutil.rmtree(work_directory)
    return completed


def _verify_and_cleanup_completed_partial(
    *,
    destination: Path,
    work_directory: Path,
    role: FinalLoraRole,
    authorization: FinalLoraAuthorization,
    benchmark_rows: Sequence[Mapping[str, str]],
) -> None:
    if not work_directory.exists():
        return
    rows, _runtime = _load_or_recover_partial_role(
        work_directory=work_directory,
        role=role,
        authorization=authorization,
        benchmark_rows=benchmark_rows,
        resume=True,
    )
    partial_prediction, _state, recovery = _partial_artifacts(work_directory)
    completed_prediction = destination / "predictions.csv"
    if (
        len(rows) != len(benchmark_rows)
        or recovery.exists()
        or not partial_prediction.is_file()
        or partial_prediction.read_bytes() != completed_prediction.read_bytes()
    ):
        raise FinalLoraProtocolError(
            f"Completed role coexists with stale/tampered partial output: {role.role}"
        )
    shutil.rmtree(work_directory)


def run_final_lora_suite(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    predictor: Callable[..., tuple[list[str], dict[str, Any]]] = _default_predictor,
    authorization_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize, evaluate exactly three roles, and atomically aggregate results."""

    authorization = authorize_final_lora(config, **dict(authorization_kwargs or {}))
    benchmark_rows = load_authorized_final_benchmark(
        authorization,
        verify_audio=bool(config["benchmark"]["verify_audio_sha256"]),
    )
    if len(benchmark_rows) != authorization.final_rows:
        raise FinalLoraProtocolError(
            "Authorized final benchmark row count changed before inference"
        )
    output_root = _repo_path(config["output"]["directory"], label="output.directory")
    device = str(config["runtime"]["device"]).casefold()
    completed: dict[str, tuple[list[dict[str, str]], dict[str, Any]]] = {}
    for role in authorization.roles:
        destination = output_root / role.role
        work_directory = _partial_role_directory(output_root, role)
        if destination.exists():
            if not resume:
                raise FileExistsError(
                    f"Role output exists; use --resume to verify/reuse it: {destination}"
                )
            completed[role.role] = _load_completed_role(
                destination, role, authorization, benchmark_rows
            )
            _verify_and_cleanup_completed_partial(
                destination=destination,
                work_directory=work_directory,
                role=role,
                authorization=authorization,
                benchmark_rows=benchmark_rows,
            )
            continue
        completed[role.role] = _run_or_resume_role(
            role=role,
            authorization=authorization,
            benchmark_rows=benchmark_rows,
            output_root=output_root,
            device=device,
            resume=resume,
            predictor=predictor,
        )

    aggregate_rows: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    for role in authorization.roles:
        rows, provenance = completed[role.role]
        prediction_hash = str(provenance["prediction_sha256"])
        prediction_hashes[role.role] = prediction_hash
        aggregate_rows.extend(
            _aggregate_role(
                role,
                rows,
                authorization=authorization,
                prediction_sha256=prediction_hash,
            )
        )
    aggregate_payload = _csv_bytes(aggregate_rows, AGGREGATE_COLUMNS)
    aggregate_provenance = {
        "aggregate_version": AGGREGATE_VERSION,
        "metric_version": METRIC_VERSION,
        "roles": list(ROLE_ORDER),
        "row_count": len(aggregate_rows),
        "result_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
        "prediction_sha256_by_role": prediction_hashes,
        "decision_lock_sha256": authorization.decision_lock_sha256,
        "method_lock_sha256": authorization.method_lock_sha256,
        "noise_split_lock_sha256": authorization.noise_split_lock_sha256,
        "final_benchmark_lock_sha256": authorization.final_benchmark_lock_sha256,
        "final_manifest_sha256": authorization.final_manifest_sha256,
        "runtime_config_sha256": authorization.runtime_config_sha256,
    }
    aggregate_dir = output_root / "aggregate"
    if aggregate_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Aggregate output exists; use --resume to verify it: {aggregate_dir}"
            )
        result_path = aggregate_dir / str(config["output"]["aggregate_filename"])
        provenance_path = aggregate_dir / "provenance.json"
        if (
            not result_path.is_file()
            or not provenance_path.is_file()
            or result_path.read_bytes() != aggregate_payload
            or provenance_path.read_bytes() != _json_bytes(aggregate_provenance)
        ):
            raise FinalLoraProtocolError("Existing aggregate output is stale or tampered")
    else:
        _commit_directory(
            aggregate_dir,
            {
                str(config["output"]["aggregate_filename"]): aggregate_payload,
                "provenance.json": _json_bytes(aggregate_provenance),
            },
        )
    return {
        "roles": list(ROLE_ORDER),
        "prediction_rows_per_role": len(benchmark_rows),
        "aggregate_rows": len(aggregate_rows),
        "output_directory": _display_path(output_root),
        "aggregate": _display_path(
            aggregate_dir / str(config["output"]["aggregate_filename"])
        ),
    }


__all__ = [
    "AGGREGATE_COLUMNS",
    "AGGREGATE_VERSION",
    "FinalLoraAuthorization",
    "FinalLoraProtocolError",
    "FinalLoraRole",
    "PREDICTION_COLUMNS",
    "PARTIAL_VERSION",
    "PROVENANCE_VERSION",
    "RECOVERY_VERSION",
    "ROLE_ORDER",
    "SUITE_VERSION",
    "authorize_final_lora",
    "load_authorized_final_benchmark",
    "load_final_lora_config",
    "run_final_lora_suite",
    "validate_final_lora_config",
]
