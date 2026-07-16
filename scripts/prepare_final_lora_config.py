from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.final_benchmark import FINAL_BENCHMARK_VERSION  # noqa: E402
from src.vitonesr.phat.final_evaluation import (  # noqa: E402
    FinalLoraProtocolError,
    validate_final_lora_config,
)
from src.vitonesr.phat.protocol import (  # noqa: E402
    is_sha256,
    sha256_file,
    verify_test_decision_lock,
)


def _repo_path(value: object, *, label: str) -> Path:
    raw = str(value).strip()
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise FinalLoraProtocolError(f"{label} must be repository-relative")
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise FinalLoraProtocolError(f"{label} escapes the repository") from exc
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Final LoRA template does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalLoraProtocolError("Final LoRA template must be an object")
    return value


def materialize_config(template: Mapping[str, Any]) -> dict[str, Any]:
    """Pin current lock hashes after decision, without opening final data/model."""

    config = deepcopy(dict(template))
    protocol = config.get("protocol")
    benchmark = config.get("benchmark")
    if not isinstance(protocol, dict) or not isinstance(benchmark, dict):
        raise FinalLoraProtocolError("Template protocol/benchmark must be objects")
    split = _repo_path(protocol.get("split_lock"), label="protocol.split_lock")
    decision_path = _repo_path(
        protocol.get("decision_lock"), label="protocol.decision_lock"
    )
    method = _repo_path(protocol.get("method_lock"), label="protocol.method_lock")
    noise = _repo_path(
        protocol.get("noise_split_lock"), label="protocol.noise_split_lock"
    )
    final_lock = _repo_path(
        protocol.get("final_benchmark_lock"),
        label="protocol.final_benchmark_lock",
    )
    hashes = {
        "expected_split_lock_sha256": sha256_file(split),
        "expected_decision_lock_sha256": sha256_file(decision_path),
        "expected_method_lock_sha256": sha256_file(method),
        "expected_noise_split_lock_sha256": sha256_file(noise),
    }
    # Decision authorization is intentionally first. It checks no checkpoint
    # bytes in this preparation step; full checkpoint verification happens only
    # after every lock passes in run_final_lora.py.
    decision = verify_test_decision_lock(
        split_lock_path=split,
        decision_lock_path=decision_path,
        verify_checkpoints=False,
    )
    if (
        str(decision["split_lock_sha256"]).casefold()
        != hashes["expected_split_lock_sha256"]
        or str(decision["decision_lock_sha256"]).casefold()
        != hashes["expected_decision_lock_sha256"]
        or str(decision["method_lock_sha256"]).casefold()
        != hashes["expected_method_lock_sha256"]
    ):
        raise FinalLoraProtocolError("Current locks differ from the reviewed decision")
    if not final_lock.is_file():
        raise FileNotFoundError(f"Final benchmark lock does not exist: {final_lock}")
    lock_hash = sha256_file(final_lock)
    try:
        lock = json.loads(final_lock.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalLoraProtocolError("Final benchmark lock is invalid JSON") from exc
    if not isinstance(lock, dict) or (
        lock.get("protocol_version") != FINAL_BENCHMARK_VERSION
        or lock.get("status") != "LOCKED"
    ):
        raise FinalLoraProtocolError("Final benchmark lock is not a locked paper-v2 artifact")
    bindings = {
        "split_lock_sha256": hashes["expected_split_lock_sha256"],
        "decision_lock_sha256": hashes["expected_decision_lock_sha256"],
        "method_lock_sha256": hashes["expected_method_lock_sha256"],
        "noise_split_lock_sha256": hashes["expected_noise_split_lock_sha256"],
        "source_test_manifest_sha256": str(decision["test_manifest_sha256"]),
    }
    if any(
        str(lock.get(field, "")).casefold() != str(expected).casefold()
        for field, expected in bindings.items()
    ):
        raise FinalLoraProtocolError("Final benchmark lock has stale protocol bindings")
    output = lock.get("output")
    if not isinstance(output, dict) or not is_sha256(output.get("manifest_sha256")):
        raise FinalLoraProtocolError("Final benchmark lock has no manifest identity")
    configured_manifest = _repo_path(
        benchmark.get("manifest"), label="benchmark.manifest"
    )
    locked_manifest = _repo_path(output.get("manifest"), label="lock.output.manifest")
    if configured_manifest != locked_manifest:
        raise FinalLoraProtocolError("Template and final lock name different manifests")
    protocol.update(hashes)
    protocol["expected_final_benchmark_lock_sha256"] = lock_hash
    benchmark["expected_manifest_sha256"] = str(output["manifest_sha256"]).casefold()
    validate_final_lora_config(config)
    return config


def _atomic_write_yaml(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite runtime config: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                yaml.safe_dump(
                    dict(value),
                    allow_unicode=True,
                    sort_keys=False,
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a hash-pinned final LoRA runtime config after the "
            "decision and final-benchmark locks exist."
        )
    )
    parser.add_argument(
        "--template",
        default="configs/paper_v2_final_lora.yaml",
    )
    parser.add_argument(
        "--output",
        default="outputs/paper_v2/protocol/final_lora_runtime.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runtime = materialize_config(_read_yaml(Path(args.template)))
        output = _repo_path(args.output, label="output")
        _atomic_write_yaml(output, runtime, overwrite=bool(args.overwrite))
    except (FileExistsError, FinalLoraProtocolError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(output.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
