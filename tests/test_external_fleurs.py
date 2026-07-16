from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import scripts.run_external_fleurs as fleurs_runner
from scripts.run_external_fleurs import (
    EVALUATION_DOMAIN,
    FORMAL_PATH_MODE,
    METRIC_EVIDENCE_COLUMNS,
    PROVENANCE_VERSION,
    REGISTRY_VERSION,
    REQUIRED_ROLES,
    RESULT_COLUMNS,
    RESULT_PROVENANCE_VERSION,
    ExternalAuthorization,
    ExternalFleursError,
    ExternalRun,
    _load_base_model,
    _load_processor_with_fallback,
    _partial_path,
    _provenance_path,
    _recovery_path,
    _result_provenance_path,
    _resume_path,
    authorize_external_suite,
    build_external_results,
    build_external_runs,
    create_run_registry,
    join_chunk_hypotheses,
    load_fleurs_manifest,
    load_run_registry,
    run_external_prediction,
    run_external_suite,
    split_waveform,
)
from src.vitonesr.analysis import (
    CANONICAL_PREDICTION_COLUMNS,
    METRIC_VERSION,
    load_prediction_csv,
)
from src.vitonesr.phat.protocol import canonical_sha256, sha256_file
from src.vitonesr.prediction import atomic_write_csv
from scripts.download_fleurs import FleursPreparationError


ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ebdb9e88f5cc5271fb88f4d642c82ff9388650e"


def _write_manifest(path: Path, audio_paths: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, audio in enumerate(audio_paths):
            handle.write(
                json.dumps(
                    {
                        "utt_id": f"fleurs-{index}",
                        "dataset": "fleurs",
                        "split": "test",
                        "audio_path": str(audio),
                        "transcript": "xin chào" if index == 0 else "tôi là trung",
                        "snr": "clean",
                        "noise_type": "clean",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _locked_configuration(
    configuration_id: str,
    role: str,
    train_type: str,
    lambda_value: float,
    marker: str,
) -> dict[str, Any]:
    method_id = "ordinary_lora" if role == "ordinary_baseline" else "corrected_decoder_tone_lora"
    return {
        "configuration_id": configuration_id,
        "role": role,
        "method_id": method_id,
        "train_type": train_type,
        "lambda": lambda_value,
        "seed": 42,
        "backbone": "vinai/PhoWhisper-base",
        "backbone_revision": REVISION,
        "checkpoint_sha256": marker * 64,
        "resolved_config_sha256": marker.upper().casefold() * 64,
        "training_contract_sha256": str((int(marker, 16) + 3) % 16).replace("10", "a") * 64,
    }


def _locked_runs() -> list[dict[str, Any]]:
    # The selected lambda is intentionally not 0.05/0.1.
    return [
        _locked_configuration(
            "ordinary_seed42", "ordinary_baseline", "ordinary_lora", 0.0, "1"
        ),
        _locked_configuration(
            "tone_selected_lambda03_seed42",
            "selected_method",
            "tone_aware_lora",
            0.3,
            "2",
        ),
        _locked_configuration(
            "tone_control_lambda01_seed42",
            "locked_control",
            "tone_aware_lora",
            0.1,
            "3",
        ),
    ]


def _run(locked: Mapping[str, Any], root: Path) -> ExternalRun:
    return ExternalRun(
        configuration_id=str(locked["configuration_id"]),
        role=str(locked["role"]),
        method_id=str(locked["method_id"]),
        train_type=str(locked["train_type"]),
        lambda_value=str(locked["lambda"]).rstrip("0").rstrip(".") or "0",
        seed=str(locked["seed"]),
        model_name_or_path=str(locked["backbone"]),
        backbone_revision=str(locked["backbone_revision"]),
        language="vi",
        task="transcribe",
        checkpoint=root / str(locked["configuration_id"]),
        checkpoint_sha256=str(locked["checkpoint_sha256"]),
        resolved_config_sha256=str(locked["resolved_config_sha256"]),
        training_contract_sha256=str(locked["training_contract_sha256"]),
        config_path=root / f"{locked['configuration_id']}.yaml",
        config_sha256="a" * 64,
        prediction_name=f"pred_{locked['configuration_id']}.csv",
    )


def _authorization(root: Path, manifest: Path) -> ExternalAuthorization:
    locked = _locked_runs()
    preparation_lock = root / "fleurs_test_lock.json"
    preparation_lock.write_text("{}\n", encoding="utf-8")
    manifest_hash = sha256_file(manifest)
    registry = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "expected_rows": 2,
        "fleurs_preparation_lock_sha256": sha256_file(preparation_lock),
        "fleurs_preparation_identity_sha256": "f" * 64,
        "fleurs_dataset_repository": "google/fleurs",
        "fleurs_dataset_config": "vi_vn",
        "fleurs_dataset_split": "test",
        "fleurs_dataset_revision": REVISION,
        "fleurs_audio_inventory_sha256": "8" * 64,
        "fleurs_audit_sha256": "9" * 64,
        "decoding": {
            "language": "vi",
            "task": "transcribe",
            "sample_rate": 16_000,
            "max_new_tokens": 440,
            "max_chunk_seconds": 30.0,
            "do_sample": False,
            "num_beams": 1,
        }
    }
    return ExternalAuthorization(
        registry_path=root / "registry.json",
        registry_sha256="a" * 64,
        registry=registry,
        split_lock_sha256="b" * 64,
        decision_lock_sha256="c" * 64,
        method_lock_sha256="d" * 64,
        method_identity_sha256="e" * 64,
        manifest_path=manifest,
        manifest_sha256=manifest_hash,
        expected_rows=2,
        locked_by_role={str(item["role"]): item for item in locked},
        fleurs_preparation_lock_path=preparation_lock,
        fleurs_preparation_lock_sha256=sha256_file(preparation_lock),
        fleurs_preparation_identity_sha256="f" * 64,
        fleurs_dataset_revision=REVISION,
        fleurs_audio_inventory_sha256="8" * 64,
        fleurs_audit_sha256="9" * 64,
    )


def _preparation(authorization: ExternalAuthorization) -> dict[str, Any]:
    return {
        "preparation_lock_sha256": authorization.fleurs_preparation_lock_sha256,
        "identity_sha256": authorization.fleurs_preparation_identity_sha256,
        "manifest_path": authorization.manifest_path,
        "dataset": {
            "repository": "google/fleurs",
            "config": "vi_vn",
            "split": "test",
            "revision": authorization.fleurs_dataset_revision,
        },
        "output": {
            "row_count": authorization.expected_rows,
            "manifest_sha256": authorization.manifest_sha256,
            "audio_inventory_sha256": authorization.fleurs_audio_inventory_sha256,
            "audit_sha256": authorization.fleurs_audit_sha256,
        },
    }


def _registry_object(
    root: Path,
    *,
    manifest: Path,
    split_lock: Path,
    decision_lock: Path,
) -> dict[str, Any]:
    preparation_lock = root / "fleurs_test_lock.json"
    preparation_lock.write_text("{}\n", encoding="utf-8")
    registry = {
        "registry_version": REGISTRY_VERSION,
        "evaluation_domain": EVALUATION_DOMAIN,
        "dataset": "fleurs",
        "manifest": str(manifest),
        "manifest_sha256": "d" * 64,
        "expected_rows": 857,
        "fleurs_preparation_lock": str(preparation_lock),
        "fleurs_preparation_lock_version": "paper_v2_fleurs_preparation_v1",
        "fleurs_preparation_lock_sha256": sha256_file(preparation_lock),
        "fleurs_preparation_identity_sha256": "f" * 64,
        "fleurs_dataset_repository": "google/fleurs",
        "fleurs_dataset_config": "vi_vn",
        "fleurs_dataset_split": "test",
        "fleurs_dataset_revision": REVISION,
        "fleurs_audio_inventory_sha256": "8" * 64,
        "fleurs_audit_sha256": "9" * 64,
        "split_lock": str(split_lock),
        "split_lock_sha256": sha256_file(split_lock),
        "decision_lock": str(decision_lock),
        "decision_lock_sha256": sha256_file(decision_lock),
        "method_lock_sha256": "a" * 64,
        "method_identity_sha256": "b" * 64,
        "decoding": {
            "language": "vi",
            "task": "transcribe",
            "sample_rate": 16_000,
            "max_new_tokens": 440,
            "max_chunk_seconds": 30.0,
            "do_sample": False,
            "num_beams": 1,
        },
        "runs": [
            {
                "configuration_id": item["configuration_id"],
                "config_path": str(root / f"{item['configuration_id']}.yaml"),
                "config_sha256": "e" * 64,
                "checkpoint_path": str(root / str(item["configuration_id"])),
            }
            for item in _locked_runs()
        ],
    }
    registry["identity_sha256"] = canonical_sha256(registry)
    return registry


def _formal_authorization_fixture(root: Path) -> dict[str, Any]:
    prefix = root.relative_to(ROOT).as_posix()
    split = root / "split.json"
    method_lock = root / "method_lock.json"
    decision = root / "decision.json"
    manifest = root / "not-opened.jsonl"
    split.write_text("{}\n", encoding="utf-8", newline="\n")
    method_lock.write_text('{"mode":"formal"}\n', encoding="utf-8", newline="\n")
    method_hash = sha256_file(method_lock)
    method_identity = "b" * 64
    decision.write_text(
        json.dumps(
            {
                "method_lock": f"{prefix}/method_lock.json",
                "method_lock_sha256": method_hash,
                "method_identity_sha256": method_identity,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    locked = _locked_runs()
    registry = _registry_object(
        root,
        manifest=manifest,
        split_lock=split,
        decision_lock=decision,
    )
    registry["path_mode"] = FORMAL_PATH_MODE
    registry["manifest"] = f"{prefix}/not-opened.jsonl"
    registry["fleurs_preparation_lock"] = f"{prefix}/fleurs_test_lock.json"
    registry["split_lock"] = f"{prefix}/split.json"
    registry["decision_lock"] = f"{prefix}/decision.json"
    registry["method_lock_sha256"] = method_hash
    registry["method_identity_sha256"] = method_identity

    configs: dict[Path, dict[str, Any]] = {}
    for entry, role in zip(registry["runs"], locked):
        config_path = root / f"{role['configuration_id']}.yaml"
        config_path.write_text("fixture: true\n", encoding="utf-8", newline="\n")
        entry["config_path"] = f"{prefix}/{config_path.name}"
        entry["config_sha256"] = sha256_file(config_path)
        entry["checkpoint_path"] = f"{prefix}/{role['configuration_id']}"
        configs[config_path.resolve()] = {
            "seed": role["seed"],
            "experiment": {
                "method_id": role["method_id"],
                "train_type": role["train_type"],
            },
            "protocol": {"method_lock": f"{prefix}/method_lock.json"},
            "model": {
                "name_or_path": role["backbone"],
                "revision": role["backbone_revision"],
            },
            "training": {
                "lambda_tone": role["lambda"],
                "run_scope": "formal",
            },
        }
    registry["split_lock_sha256"] = sha256_file(split)
    registry["decision_lock_sha256"] = sha256_file(decision)
    registry.pop("identity_sha256", None)
    registry["identity_sha256"] = canonical_sha256(registry)
    registry_path = root / "registry.json"
    registry_path.write_text(
        json.dumps(registry, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "prefix": prefix,
        "registry_reference": f"{prefix}/registry.json",
        "registry": registry,
        "split": split,
        "decision": decision,
        "manifest": manifest,
        "method_hash": method_hash,
        "method_identity": method_identity,
        "locked": locked,
        "configs": configs,
    }


class FakeTranscriber:
    def __init__(self, hypothesis: str, calls: list[int]) -> None:
        self.hypothesis = hypothesis
        self.calls = calls
        self.closed = False

    def transcribe_chunk(self, waveform: Any) -> str:
        self.calls.append(len(waveform))
        return self.hypothesis

    def close(self) -> None:
        self.closed = True


class ExternalFleursTests(unittest.TestCase):
    def test_registry_creation_is_decision_first_dynamic_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            decision = root / "decision.json"
            manifest = root / "fleurs.jsonl"
            split.write_text("{}", encoding="utf-8")
            decision.write_text("{}", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            locked = _locked_runs()
            config_paths = {
                "ordinary_baseline": ROOT / "configs" / "phat" / "lambda_0.yaml",
                "selected_method": ROOT / "configs" / "phat" / "lambda_03.yaml",
                "locked_control": ROOT / "configs" / "phat" / "lambda_01.yaml",
            }
            identities: dict[str, dict[str, str]] = {}
            for item in locked:
                checkpoint_path = root / str(item["configuration_id"])
                item["checkpoint_path"] = str(checkpoint_path)
                marker = str(item["checkpoint_sha256"])[0]
                item["resolved_config_sha256"] = marker * 64
                item["training_contract_sha256"] = marker * 64
                identities[checkpoint_path.name] = {
                    "checkpoint_sha256": str(item["checkpoint_sha256"]),
                    "resolved_config_sha256": marker * 64,
                    "training_contract_sha256": marker * 64,
                }
            events: list[str] = []

            def verifier(**_kwargs: object) -> dict[str, Any]:
                events.append("decision")
                return {
                    "split_lock_sha256": sha256_file(split),
                    "decision_lock_sha256": sha256_file(decision),
                    "method_lock_sha256": "a" * 64,
                    "method_identity_sha256": "b" * 64,
                    "locked_configurations": locked,
                }

            def checkpoint_verifier(
                checkpoint: str | Path, _config: Mapping[str, Any]
            ) -> Mapping[str, str]:
                events.append("checkpoint")
                return identities[Path(checkpoint).name]

            preparation_lock = root / "fleurs_test_lock.json"
            preparation_lock.write_text("{}\n", encoding="utf-8")

            def preparation_verifier(
                _path: str | Path, **_kwargs: object
            ) -> dict[str, Any]:
                events.append("preparation")
                return {
                    "preparation_lock_sha256": sha256_file(preparation_lock),
                    "identity_sha256": "f" * 64,
                    "manifest_path": manifest,
                    "dataset": {
                        "repository": "google/fleurs",
                        "config": "vi_vn",
                        "split": "test",
                        "revision": REVISION,
                    },
                    "output": {
                        "row_count": 857,
                        "manifest_sha256": sha256_file(manifest),
                        "audio_inventory_sha256": "8" * 64,
                        "audit_sha256": "9" * 64,
                    },
                }

            output = root / "registry.json"
            create_run_registry(
                output,
                preparation_lock_path=preparation_lock,
                manifest_path=manifest,
                expected_manifest_sha256=sha256_file(manifest),
                split_lock_path=split,
                decision_lock_path=decision,
                config_paths_by_role=config_paths,
                decision_verifier=verifier,
                checkpoint_verifier=checkpoint_verifier,
                preparation_verifier=preparation_verifier,
                formal=False,
            )
            self.assertEqual(events[0], "decision")
            self.assertEqual(events[-1], "preparation")
            registry = load_run_registry(output, formal=False)
            self.assertEqual(registry["expected_rows"], 857)
            self.assertEqual(
                [entry["configuration_id"] for entry in registry["runs"]],
                [item["configuration_id"] for item in locked],
            )
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                create_run_registry(
                    output,
                    preparation_lock_path=preparation_lock,
                    manifest_path=manifest,
                    expected_manifest_sha256=sha256_file(manifest),
                    split_lock_path=split,
                    decision_lock_path=decision,
                    config_paths_by_role=config_paths,
                    formal=False,
                )

    def test_authorization_resolves_arbitrary_selected_lambda_and_exact_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            decision = root / "decision.json"
            split.write_text("{}\n", encoding="utf-8")
            decision.write_text("{}\n", encoding="utf-8")
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    _registry_object(
                        root,
                        manifest=root / "not-opened.jsonl",
                        split_lock=split,
                        decision_lock=decision,
                    )
                ),
                encoding="utf-8",
            )
            calls: list[str] = []

            def verifier(**_kwargs: object) -> dict[str, Any]:
                calls.append("decision")
                return {
                    "split_lock_sha256": sha256_file(split),
                    "decision_lock_sha256": sha256_file(decision),
                    "method_lock_sha256": "a" * 64,
                    "method_identity_sha256": "b" * 64,
                    "selected_lambda": 0.3,
                    "locked_configurations": _locked_runs(),
                }

            authorization = authorize_external_suite(
                registry_path,
                formal=False,
                decision_verifier=verifier,
                preparation_verifier=lambda *_args, **_kwargs: (
                    calls.append("preparation")
                    or {
                        "preparation_lock_sha256": sha256_file(
                            root / "fleurs_test_lock.json"
                        ),
                        "identity_sha256": "f" * 64,
                        "manifest_path": root / "not-opened.jsonl",
                        "dataset": {
                            "repository": "google/fleurs",
                            "config": "vi_vn",
                            "split": "test",
                            "revision": REVISION,
                        },
                        "output": {
                            "row_count": 857,
                            "manifest_sha256": "d" * 64,
                            "audio_inventory_sha256": "8" * 64,
                            "audit_sha256": "9" * 64,
                        },
                    }
                ),
            )
            self.assertEqual(calls, ["decision", "preparation"])
            self.assertFalse(authorization.manifest_path.exists())
            self.assertEqual(tuple(authorization.locked_by_role), REQUIRED_ROLES)
            self.assertEqual(
                authorization.locked_by_role["selected_method"]["lambda"], 0.3
            )

    def test_formal_authorization_verifies_method_runtime_for_all_three_configs(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = _formal_authorization_fixture(Path(temporary))
            events: list[str] = []

            def decision_verifier(**_kwargs: object) -> dict[str, Any]:
                events.append("decision")
                return {
                    "split_lock_sha256": sha256_file(fixture["split"]),
                    "decision_lock_sha256": sha256_file(fixture["decision"]),
                    "method_lock_sha256": fixture["method_hash"],
                    "method_identity_sha256": fixture["method_identity"],
                    "locked_configurations": fixture["locked"],
                }

            def config_loader(path: str | Path) -> Mapping[str, Any]:
                return fixture["configs"][Path(path).resolve()]

            def method_verifier(
                _path: str | Path,
                *,
                config: Mapping[str, Any],
                formal: bool,
                **_kwargs: object,
            ) -> Mapping[str, str]:
                events.append(
                    f"method:{config['training']['lambda_tone']}:{formal}"
                )
                return {
                    "mode": "formal",
                    "method_lock_sha256": fixture["method_hash"],
                    "method_identity_sha256": fixture["method_identity"],
                    "protocol_split_lock_sha256": sha256_file(fixture["split"]),
                    "environment_identity_sha256": "c" * 64,
                    "source_tree_sha256": "d" * 64,
                }

            def preparation_verifier(
                _path: str | Path, **_kwargs: object
            ) -> Mapping[str, Any]:
                events.append("preparation")
                registry = fixture["registry"]
                return {
                    "preparation_lock_sha256": registry[
                        "fleurs_preparation_lock_sha256"
                    ],
                    "identity_sha256": registry[
                        "fleurs_preparation_identity_sha256"
                    ],
                    "manifest_path": fixture["manifest"],
                    "dataset": {
                        "repository": "google/fleurs",
                        "config": "vi_vn",
                        "split": "test",
                        "revision": REVISION,
                    },
                    "output": {
                        "row_count": 857,
                        "manifest_sha256": "d" * 64,
                        "audio_inventory_sha256": "8" * 64,
                        "audit_sha256": "9" * 64,
                    },
                }

            strict = authorize_external_suite(
                fixture["registry_reference"],
                formal=True,
                decision_verifier=decision_verifier,
                preparation_verifier=preparation_verifier,
                method_config_loader=config_loader,
                method_verifier=method_verifier,
            )
            self.assertEqual(
                events,
                [
                    "decision",
                    "method:0.0:True",
                    "method:0.3:True",
                    "method:0.1:True",
                    "preparation",
                ],
            )
            self.assertTrue(strict.method_runtime_verified)
            self.assertEqual(strict.method_environment_identity_sha256, "c" * 64)
            self.assertEqual(strict.method_source_tree_sha256, "d" * 64)
            self.assertFalse(fixture["manifest"].exists())
            contract = fleurs_runner.build_run_contract(
                _run(fixture["locked"][0], Path(temporary)), strict, []
            )
            self.assertIs(contract["method_runtime_verified"], True)
            self.assertEqual(
                contract["method_environment_identity_sha256"], "c" * 64
            )
            self.assertEqual(contract["method_source_tree_sha256"], "d" * 64)

            events.clear()
            posthoc = authorize_external_suite(
                fixture["registry_reference"],
                formal=True,
                verify_current_method=False,
                decision_verifier=decision_verifier,
                preparation_verifier=preparation_verifier,
                method_config_loader=config_loader,
                method_verifier=method_verifier,
            )
            self.assertEqual(
                events,
                [
                    "decision",
                    "method:0.0:False",
                    "method:0.3:False",
                    "method:0.1:False",
                    "preparation",
                ],
            )
            self.assertFalse(posthoc.method_runtime_verified)
            with self.assertRaisesRegex(ExternalFleursError, "post-hoc authorization"):
                run_external_suite(
                    fixture["registry_reference"],
                    output_dir=f"{fixture['prefix']}/output",
                    authorizer=lambda _path: posthoc,
                    preparation_verifier=lambda *_args, **_kwargs: self.fail(
                        "post-hoc authorization must fail before FLEURS access"
                    ),
                )

    def test_formal_method_mismatch_fails_before_fleurs_access(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = _formal_authorization_fixture(Path(temporary))
            events: list[str] = []

            def decision_verifier(**_kwargs: object) -> Mapping[str, Any]:
                events.append("decision")
                return {
                    "split_lock_sha256": sha256_file(fixture["split"]),
                    "decision_lock_sha256": sha256_file(fixture["decision"]),
                    "method_lock_sha256": fixture["method_hash"],
                    "method_identity_sha256": fixture["method_identity"],
                    "locked_configurations": fixture["locked"],
                }

            def method_verifier(*_args: object, **_kwargs: object) -> Mapping[str, str]:
                events.append("method")
                return {
                    "mode": "formal",
                    "method_lock_sha256": fixture["method_hash"],
                    "method_identity_sha256": "f" * 64,
                    "protocol_split_lock_sha256": sha256_file(fixture["split"]),
                    "environment_identity_sha256": "c" * 64,
                    "source_tree_sha256": "d" * 64,
                }

            with self.assertRaisesRegex(
                ExternalFleursError, "differs from decision/registry"
            ):
                authorize_external_suite(
                    fixture["registry_reference"],
                    formal=True,
                    decision_verifier=decision_verifier,
                    preparation_verifier=lambda *_args, **_kwargs: (
                        events.append("FLEURS") or self.fail("must stay unopened")
                    ),
                    method_config_loader=lambda path: fixture["configs"][
                        Path(path).resolve()
                    ],
                    method_verifier=method_verifier,
                )
            self.assertEqual(events, ["decision", "method"])

    def test_registry_identity_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            decision = root / "decision.json"
            split.write_text("{}\n", encoding="utf-8")
            decision.write_text("{}\n", encoding="utf-8")
            registry_path = root / "registry.json"
            registry = _registry_object(
                root,
                manifest=root / "fleurs.jsonl",
                split_lock=split,
                decision_lock=decision,
            )
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            load_run_registry(registry_path, formal=False)
            registry["decoding"]["max_new_tokens"] = 441
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(ExternalFleursError, "identity.*tampered"):
                load_run_registry(registry_path, formal=False)

    def test_formal_registry_rejects_host_paths_and_diagnostic_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            prefix = root.relative_to(ROOT).as_posix()
            split = root / "split.json"
            decision = root / "decision.json"
            split.write_text("{}\n", encoding="utf-8")
            decision.write_text("{}\n", encoding="utf-8")
            registry_path = root / "registry.json"

            def portable_registry() -> dict[str, Any]:
                registry = _registry_object(
                    root,
                    manifest=root / "fleurs.jsonl",
                    split_lock=split,
                    decision_lock=decision,
                )
                registry["path_mode"] = FORMAL_PATH_MODE
                registry["manifest"] = f"{prefix}/fleurs.jsonl"
                registry["fleurs_preparation_lock"] = (
                    f"{prefix}/fleurs_test_lock.json"
                )
                registry["split_lock"] = f"{prefix}/split.json"
                registry["decision_lock"] = f"{prefix}/decision.json"
                for entry in registry["runs"]:
                    name = str(entry["configuration_id"])
                    entry["config_path"] = f"{prefix}/{name}.yaml"
                    entry["checkpoint_path"] = f"{prefix}/{name}"
                registry.pop("identity_sha256", None)
                registry["identity_sha256"] = canonical_sha256(registry)
                return registry

            clean = portable_registry()
            registry_path.write_text(json.dumps(clean), encoding="utf-8")
            reference = registry_path.relative_to(ROOT).as_posix()
            self.assertEqual(
                load_run_registry(reference, formal=True)["path_mode"],
                FORMAL_PATH_MODE,
            )
            with self.assertRaisesRegex(ExternalFleursError, "repository-relative"):
                load_run_registry(registry_path, formal=True)
            # The old absolute-path behavior remains available only when the
            # caller opts into diagnostic mode explicitly.
            diagnostic = portable_registry()
            diagnostic["path_mode"] = "diagnostic_legacy_paths_v1"
            diagnostic["manifest"] = str(root / "fleurs.jsonl")
            diagnostic.pop("identity_sha256", None)
            diagnostic["identity_sha256"] = canonical_sha256(diagnostic)
            registry_path.write_text(json.dumps(diagnostic), encoding="utf-8")
            load_run_registry(registry_path, formal=False)

            mutations = (
                ("manifest", None),
                ("fleurs_preparation_lock", None),
                ("split_lock", None),
                ("decision_lock", None),
                ("config_path", 0),
                ("checkpoint_path", 0),
            )
            for field, run_index in mutations:
                with self.subTest(field=field):
                    registry = portable_registry()
                    if run_index is None:
                        registry[field] = r"C:\Users\Trung\private\artifact.json"
                    else:
                        registry["runs"][run_index][field] = (
                            r"C:\Users\Trung\private\artifact"
                        )
                    registry.pop("identity_sha256", None)
                    registry["identity_sha256"] = canonical_sha256(registry)
                    registry_path.write_text(json.dumps(registry), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ExternalFleursError,
                        "repository-relative and portable",
                    ):
                        load_run_registry(reference, formal=True)

    def test_authorization_rejects_missing_or_duplicate_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            decision = root / "decision.json"
            split.write_text("{}", encoding="utf-8")
            decision.write_text("{}", encoding="utf-8")
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    _registry_object(
                        root,
                        manifest=root / "fleurs.jsonl",
                        split_lock=split,
                        decision_lock=decision,
                    )
                ),
                encoding="utf-8",
            )

            for configurations, message in (
                (_locked_runs()[:2], "exactly one ordinary"),
                (
                    [
                        _locked_runs()[0],
                        {**_locked_runs()[1], "role": "ordinary_baseline"},
                        _locked_runs()[2],
                    ],
                    "exactly one ordinary",
                ),
            ):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ExternalFleursError, message):
                        authorize_external_suite(
                            registry_path,
                            formal=False,
                            decision_verifier=lambda **_kwargs: {
                                "split_lock_sha256": sha256_file(split),
                                "decision_lock_sha256": sha256_file(decision),
                                "method_lock_sha256": "a" * 64,
                                "method_identity_sha256": "b" * 64,
                                "locked_configurations": configurations,
                            },
                        )

    def test_build_runs_uses_registry_and_selected_lambda_not_fixed_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "fleurs.jsonl"
            manifest.write_text("", encoding="utf-8")
            locked = _locked_runs()
            config_paths = [
                ROOT / "configs" / "phat" / "lambda_0.yaml",
                ROOT / "configs" / "phat" / "lambda_03.yaml",
                ROOT / "configs" / "phat" / "lambda_01.yaml",
            ]
            registry_runs = []
            identities: dict[str, dict[str, str]] = {}
            for item, config_path in zip(locked, config_paths):
                item["checkpoint_path"] = str(
                    root / str(item["configuration_id"])
                )
                item["resolved_config_sha256"] = (
                    str(item["checkpoint_sha256"])[0] * 64
                )
                item["training_contract_sha256"] = (
                    str(item["checkpoint_sha256"])[0] * 64
                )
                registry_runs.append(
                    {
                        "configuration_id": item["configuration_id"],
                        "config_path": str(config_path),
                        "config_sha256": sha256_file(config_path),
                        "checkpoint_path": item["checkpoint_path"],
                    }
                )
                identities[str(item["configuration_id"])] = {
                    "checkpoint_sha256": str(item["checkpoint_sha256"]),
                    "resolved_config_sha256": str(item["resolved_config_sha256"]),
                    "training_contract_sha256": str(item["training_contract_sha256"]),
                }
            authorization = ExternalAuthorization(
                registry_path=root / "registry.json",
                registry_sha256="a" * 64,
                registry={"runs": registry_runs},
                split_lock_sha256="b" * 64,
                decision_lock_sha256="c" * 64,
                method_lock_sha256="d" * 64,
                method_identity_sha256="e" * 64,
                manifest_path=manifest,
                manifest_sha256="d" * 64,
                expected_rows=857,
                locked_by_role={str(item["role"]): item for item in locked},
            )

            def checkpoint_verifier(
                checkpoint: str | Path, _config: Mapping[str, Any]
            ) -> Mapping[str, str]:
                return identities[Path(checkpoint).name]

            runs = build_external_runs(
                authorization, checkpoint_verifier=checkpoint_verifier
            )
            self.assertEqual([run.role for run in runs], list(REQUIRED_ROLES))
            self.assertEqual([run.lambda_value for run in runs], ["0", "0.3", "0.1"])
            self.assertEqual(
                runs[1].prediction_name, "pred_tone_selected_lambda03_seed42.csv"
            )

    def test_immutable_revision_is_forwarded_to_processor_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run(_locked_runs()[1], Path(temporary))
            calls: list[tuple[str, dict[str, Any]]] = []

            class LoaderStub:
                @classmethod
                def from_pretrained(cls, source: str, **kwargs: Any) -> dict[str, Any]:
                    calls.append((source, kwargs))
                    return {"source": source}

            _load_processor_with_fallback(LoaderStub, run)
            _load_base_model(LoaderStub, run)
            self.assertEqual([call[1]["revision"] for call in calls], [REVISION, REVISION])
            self.assertEqual(calls[0][1]["language"], "vi")
            self.assertEqual(calls[0][1]["task"], "transcribe")

    def test_authorization_precedes_manifest_and_model_and_provenance_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audios = [root / "a.wav", root / "b.wav"]
            for audio in audios:
                audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, audios)
            authorization = _authorization(root, manifest)
            runs = tuple(_run(item, root) for item in _locked_runs())
            events: list[str] = []

            def authorizer(_path: str | Path) -> ExternalAuthorization:
                events.append("authorize")
                return authorization

            def manifest_loader(path: str | Path, **kwargs: object) -> list[dict[str, str]]:
                events.append("manifest")
                return load_fleurs_manifest(path, formal=False, **kwargs)

            def preparation_verifier(
                _path: str | Path, **_kwargs: object
            ) -> dict[str, Any]:
                events.append("preparation")
                return _preparation(authorization)

            def factory(run: ExternalRun, _device: str, _tokens: int) -> FakeTranscriber:
                events.append("model:" + run.role)
                return FakeTranscriber(run.role, [])

            predictions, results = run_external_suite(
                root / "ignored.json",
                output_dir=root / "output",
                authorizer=authorizer,
                preparation_verifier=preparation_verifier,
                manifest_loader=manifest_loader,
                run_builder=lambda _authorization: runs,
                transcriber_factory=factory,
                audio_loader=lambda _path, _sample_rate: [0.0] * 8,
                checkpoint_every=1,
            )
            self.assertEqual(events[:3], ["authorize", "preparation", "manifest"])
            self.assertTrue(all(event.startswith("model:") for event in events[3:]))
            self.assertEqual(len(predictions), 3)
            for run, prediction in zip(runs, predictions):
                with prediction.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(
                        tuple(reader.fieldnames or ()), CANONICAL_PREDICTION_COLUMNS
                    )
                    self.assertEqual(len(list(reader)), 2)
                provenance = json.loads(
                    _provenance_path(prediction).read_text(encoding="utf-8")
                )
                self.assertEqual(provenance["provenance_version"], PROVENANCE_VERSION)
                self.assertEqual(provenance["evaluation_domain"], EVALUATION_DOMAIN)
                self.assertEqual(provenance["configuration_id"], run.configuration_id)
                self.assertEqual(provenance["prediction_sha256"], sha256_file(prediction))
                self.assertEqual(provenance["checkpoint_sha256"], run.checkpoint_sha256)
                self.assertEqual(provenance["decision_lock_sha256"], "c" * 64)
                self.assertEqual(provenance["manifest_sha256"], sha256_file(manifest))
                self.assertEqual(provenance["fleurs_dataset_revision"], REVISION)
                self.assertEqual(
                    provenance["fleurs_audio_inventory_sha256"], "8" * 64
                )
                self.assertFalse(_partial_path(prediction).exists())
                self.assertFalse(_resume_path(prediction).exists())
            with results.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), RESULT_COLUMNS)
                rows = list(reader)
            self.assertEqual(
                RESULT_COLUMNS[-len(METRIC_EVIDENCE_COLUMNS) :],
                METRIC_EVIDENCE_COLUMNS,
            )
            self.assertEqual(
                RESULT_COLUMNS[: -len(METRIC_EVIDENCE_COLUMNS)],
                (
                    "dataset", "model", "model_size", "train_type", "lambda", "seed",
                    "n", "wer", "cer", "ter", "der", "fcer", "swdr",
                    "metric_version",
                ),
            )
            self.assertEqual([row["lambda"] for row in rows], ["0", "0.3", "0.1"])
            self.assertTrue(all(row["metric_version"] == METRIC_VERSION for row in rows))
            result_provenance = json.loads(
                _result_provenance_path(results).read_text(encoding="utf-8")
            )
            self.assertEqual(
                RESULT_PROVENANCE_VERSION,
                "paper_v2_fleurs_results_v4",
            )
            self.assertEqual(
                result_provenance["provenance_version"],
                RESULT_PROVENANCE_VERSION,
            )
            self.assertEqual(
                result_provenance["result_columns"],
                list(RESULT_COLUMNS),
            )
            for row in rows:
                for metric in ("wer", "cer", "ter", "der", "fcer", "swdr"):
                    numerator = int(row[f"{metric}_numerator"])
                    denominator = int(row[f"{metric}_denominator"])
                    self.assertEqual(
                        float(row[metric]),
                        numerator / max(denominator, 1),
                    )
                word_units = int(row["wer_denominator"])
                self.assertGreater(word_units, 0)
                for metric in ("ter", "der", "fcer"):
                    self.assertEqual(
                        float(row[f"{metric}_coverage"]),
                        int(row[f"{metric}_denominator"]) / word_units,
                    )

    def test_audio_integrity_failure_precedes_manifest_and_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "fleurs.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            authorization = _authorization(root, manifest)
            events: list[str] = []

            def authorizer(_path: str | Path) -> ExternalAuthorization:
                events.append("authorize")
                return authorization

            def fail_preparation(*_args: object, **_kwargs: object) -> Mapping[str, Any]:
                events.append("verify_audio")
                raise FleursPreparationError("audio SHA-256 mismatch/tamper")

            with self.assertRaisesRegex(ExternalFleursError, "audio integrity"):
                run_external_suite(
                    root / "ignored.json",
                    output_dir=root / "output",
                    authorizer=authorizer,
                    preparation_verifier=fail_preparation,
                    manifest_loader=lambda *_args, **_kwargs: self.fail(
                        "manifest must not open"
                    ),
                    run_builder=lambda _authorization: self.fail("model must not open"),
                )
            self.assertEqual(events, ["authorize", "verify_audio"])

    def test_formal_run_rejects_absolute_registry_output_and_results_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "fleurs.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            authorization = replace(_authorization(root, manifest), formal=True)
            cases = (
                {
                    "registry_path": root / "registry.json",
                    "output_dir": "outputs/paper_v2/external/fleurs",
                    "results_path": None,
                },
                {
                    "registry_path": "outputs/paper_v2/protocol/registry.json",
                    "output_dir": root / "absolute-output",
                    "results_path": None,
                },
                {
                    "registry_path": "outputs/paper_v2/protocol/registry.json",
                    "output_dir": "outputs/paper_v2/external/fleurs",
                    "results_path": root / "absolute-results.csv",
                },
            )
            for case in cases:
                with self.subTest(case=case):
                    with self.assertRaisesRegex(
                        ExternalFleursError,
                        "repository-relative and portable",
                    ):
                        run_external_suite(
                            case["registry_path"],
                            output_dir=case["output_dir"],
                            results_path=case["results_path"],
                            authorizer=lambda _path: authorization,
                            preparation_verifier=lambda *_args, **_kwargs: self.fail(
                                "path rejection must precede FLEURS access"
                            ),
                        )

    def test_result_commit_marker_recovers_exact_orphan_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audios = [root / "a.wav", root / "b.wav"]
            for audio in audios:
                audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, audios)
            authorization = _authorization(root, manifest)
            runs = tuple(_run(item, root) for item in _locked_runs())

            def manifest_loader(
                path: str | Path, **kwargs: object
            ) -> list[dict[str, str]]:
                return load_fleurs_manifest(path, formal=False, **kwargs)

            common: dict[str, Any] = {
                "output_dir": root / "output",
                "authorizer": lambda _path: authorization,
                "preparation_verifier": lambda *_args, **_kwargs: _preparation(
                    authorization
                ),
                "manifest_loader": manifest_loader,
                "run_builder": lambda _authorization: runs,
                "audio_loader": lambda *_args: [0.0] * 4,
                "checkpoint_every": 1,
            }
            _predictions, result = run_external_suite(
                root / "ignored.json",
                transcriber_factory=lambda run, *_args: FakeTranscriber(
                    run.role, []
                ),
                **common,
            )
            commit_marker = _result_provenance_path(result)
            commit_marker.unlink()
            run_external_suite(
                root / "ignored.json",
                resume=True,
                transcriber_factory=lambda *_args: self.fail(
                    "completed prediction must not reload a model"
                ),
                **common,
            )
            self.assertTrue(commit_marker.is_file())

            commit_marker.unlink()
            # Even a semantically harmless byte change is not accepted in the
            # narrow result-CSV/provenance crash recovery window.
            result.write_bytes(result.read_bytes() + b"\r\n")
            with self.assertRaisesRegex(
                ExternalFleursError,
                "differs from the computed decision-locked rows",
            ):
                run_external_suite(
                    root / "ignored.json",
                    resume=True,
                    transcriber_factory=lambda *_args: self.fail(
                        "tampered result must not reload a model"
                    ),
                    **common,
                )

    def test_completed_resume_detects_prediction_or_provenance_tamper_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "a.wav"
            audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, [audio])
            authorization = _authorization(root, manifest)
            authorization = replace(authorization, expected_rows=1)
            run = _run(_locked_runs()[1], root)
            output = root / run.prediction_name
            run_external_prediction(
                run,
                load_fleurs_manifest(manifest, expected_rows=1, formal=False),
                output,
                authorization=authorization,
                transcriber_factory=lambda *_args: FakeTranscriber("xin chào", []),
                audio_loader=lambda *_args: [0.0],
            )
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                run_external_prediction(
                    run,
                    load_fleurs_manifest(manifest, expected_rows=1, formal=False),
                    output,
                    authorization=authorization,
                )
            sidecar = _provenance_path(output)
            original_provenance = sidecar.read_text(encoding="utf-8")
            tampered = json.loads(original_provenance)
            tampered["decision_lock_sha256"] = "f" * 64
            sidecar.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ExternalFleursError, "decision_lock_sha256"):
                run_external_prediction(
                    run,
                    load_fleurs_manifest(manifest, expected_rows=1, formal=False),
                    output,
                    authorization=authorization,
                    resume=True,
                    transcriber_factory=lambda *_args: self.fail("model must not load"),
                )
            sidecar.write_text(original_provenance, encoding="utf-8")
            output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ExternalFleursError, "prediction_sha256"):
                run_external_prediction(
                    run,
                    load_fleurs_manifest(manifest, expected_rows=1, formal=False),
                    output,
                    authorization=authorization,
                    resume=True,
                    transcriber_factory=lambda *_args: self.fail("model must not load"),
                )

    def test_partial_resume_is_hash_bound_and_transcribes_only_remaining_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audios = [root / "a.wav", root / "b.wav"]
            for audio in audios:
                audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, audios)
            rows = load_fleurs_manifest(manifest, expected_rows=2, formal=False)
            authorization = _authorization(root, manifest)
            run = _run(_locked_runs()[0], root)
            output = root / run.prediction_name
            calls: list[int] = []

            class FailingTranscriber(FakeTranscriber):
                def transcribe_chunk(self, waveform: Any) -> str:
                    if self.calls:
                        raise RuntimeError("interrupt")
                    return super().transcribe_chunk(waveform)

            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                run_external_prediction(
                    run,
                    rows,
                    output,
                    authorization=authorization,
                    checkpoint_every=1,
                    transcriber_factory=lambda *_args: FailingTranscriber("xin", calls),
                    audio_loader=lambda *_args: [0.0] * 7,
                )
            self.assertTrue(_partial_path(output).exists())
            self.assertTrue(_resume_path(output).exists())
            resumed_calls: list[int] = []
            run_external_prediction(
                run,
                rows,
                output,
                authorization=authorization,
                resume=True,
                transcriber_factory=lambda *_args: FakeTranscriber("trung", resumed_calls),
                audio_loader=lambda *_args: [0.0] * 9,
            )
            self.assertEqual(resumed_calls, [9])
            self.assertTrue(_provenance_path(output).exists())

    def test_resume_recovers_only_valid_csv_written_state_missing_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audios = [root / "a.wav", root / "b.wav"]
            for audio in audios:
                audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, audios)
            rows = load_fleurs_manifest(manifest, expected_rows=2, formal=False)
            authorization = _authorization(root, manifest)
            run = _run(_locked_runs()[0], root)
            output = root / run.prediction_name
            partial = _partial_path(output)
            first = {
                "utt_id": rows[0]["utt_id"],
                **run.run_metadata,
                "snr": "clean",
                "noise_type": "clean",
                "ref": rows[0]["ref"],
                "hyp": "xin",
            }
            # Simulate process death after the atomic CSV checkpoint but before
            # the matching state JSON commit.
            atomic_write_csv(partial, [first], CANONICAL_PREDICTION_COLUMNS)
            calls: list[int] = []
            run_external_prediction(
                run,
                rows,
                output,
                authorization=authorization,
                checkpoint_every=1,
                resume=True,
                transcriber_factory=lambda *_args: FakeTranscriber("trung", calls),
                audio_loader=lambda *_args: [0.0] * 5,
            )
            self.assertEqual(calls, [5])
            self.assertFalse(_resume_path(output).exists())
            self.assertTrue(_provenance_path(output).is_file())

    def test_formal_write_ahead_receipt_recovers_exact_csv_and_rejects_hyp_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            prefix = root.relative_to(ROOT).as_posix()
            audio = root / "a.wav"
            audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, [audio])
            rows = load_fleurs_manifest(manifest, expected_rows=1, formal=False)
            diagnostic = _authorization(root, manifest)
            registry = dict(diagnostic.registry)
            registry["manifest"] = f"{prefix}/fleurs.jsonl"
            authorization = replace(
                diagnostic,
                registry=registry,
                registry_path=Path(f"{prefix}/registry.json"),
                expected_rows=1,
                formal=True,
            )
            run = _run(_locked_runs()[0], root)
            original_write_json = fleurs_runner._atomic_write_json

            def leave_csv_without_state(
                progress_to_crash: Path,
            ) -> Any:
                def write(path: Path, value: Mapping[str, Any]) -> None:
                    if Path(path) == progress_to_crash:
                        raise KeyboardInterrupt("simulated process death")
                    original_write_json(path, value)

                return write

            output_reference = f"{prefix}/formal-prediction.csv"
            output = ROOT / output_reference
            with mock.patch.object(
                fleurs_runner,
                "_atomic_write_json",
                side_effect=leave_csv_without_state(_resume_path(output)),
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "process death"):
                    run_external_prediction(
                        run,
                        rows,
                        output_reference,
                        authorization=authorization,
                        checkpoint_every=1,
                        transcriber_factory=lambda *_args: FakeTranscriber("xin", []),
                        audio_loader=lambda *_args: [0.0],
                    )
            self.assertTrue(_partial_path(output).is_file())
            self.assertTrue(_recovery_path(output).is_file())
            self.assertFalse(_resume_path(output).exists())
            run_external_prediction(
                run,
                rows,
                output_reference,
                authorization=authorization,
                checkpoint_every=1,
                resume=True,
                transcriber_factory=lambda *_args: self.fail(
                    "receipt recovered the complete CSV; model must not load"
                ),
            )
            self.assertTrue(_provenance_path(output).is_file())
            self.assertFalse(_recovery_path(output).exists())

            tampered_reference = f"{prefix}/tampered-prediction.csv"
            tampered_output = ROOT / tampered_reference
            with mock.patch.object(
                fleurs_runner,
                "_atomic_write_json",
                side_effect=leave_csv_without_state(
                    _resume_path(tampered_output)
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_external_prediction(
                        run,
                        rows,
                        tampered_reference,
                        authorization=authorization,
                        checkpoint_every=1,
                        transcriber_factory=lambda *_args: FakeTranscriber("xin", []),
                        audio_loader=lambda *_args: [0.0],
                    )
            partial_rows = load_prediction_csv(_partial_path(tampered_output))
            partial_rows[0]["hyp"] = "tampered hypothesis"
            atomic_write_csv(
                _partial_path(tampered_output),
                partial_rows,
                CANONICAL_PREDICTION_COLUMNS,
            )
            with self.assertRaisesRegex(
                ExternalFleursError,
                "neither hash.*tamper",
            ):
                run_external_prediction(
                    run,
                    rows,
                    tampered_reference,
                    authorization=authorization,
                    checkpoint_every=1,
                    resume=True,
                    transcriber_factory=lambda *_args: self.fail(
                        "tamper must fail before model load"
                    ),
                )

    def test_resume_rejects_state_less_prefix_tamper_and_bound_state_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audios = [root / "a.wav", root / "b.wav"]
            for audio in audios:
                audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, audios)
            rows = load_fleurs_manifest(manifest, expected_rows=2, formal=False)
            authorization = _authorization(root, manifest)
            run = _run(_locked_runs()[0], root)
            output = root / run.prediction_name
            partial = _partial_path(output)
            tampered = {
                "utt_id": rows[0]["utt_id"],
                **run.run_metadata,
                "snr": "clean",
                "noise_type": "clean",
                "ref": "not the locked reference",
                "hyp": "xin",
            }
            atomic_write_csv(partial, [tampered], CANONICAL_PREDICTION_COLUMNS)
            with self.assertRaisesRegex(ExternalFleursError, "manifest prefix"):
                run_external_prediction(
                    run,
                    rows,
                    output,
                    authorization=authorization,
                    checkpoint_every=1,
                    resume=True,
                    transcriber_factory=lambda *_args: self.fail(
                        "model must not load for tampered prefix"
                    ),
                )

            partial.unlink()
            calls: list[int] = []

            class FailingTranscriber(FakeTranscriber):
                def transcribe_chunk(self, waveform: Any) -> str:
                    if self.calls:
                        raise RuntimeError("interrupt")
                    return super().transcribe_chunk(waveform)

            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                run_external_prediction(
                    run,
                    rows,
                    output,
                    authorization=authorization,
                    checkpoint_every=1,
                    transcriber_factory=lambda *_args: FailingTranscriber(
                        "xin", calls
                    ),
                    audio_loader=lambda *_args: [0.0],
                )
            state_path = _resume_path(output)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["registry_sha256"] = "f" * 64
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ExternalFleursError, "registry_sha256"):
                run_external_prediction(
                    run,
                    rows,
                    output,
                    authorization=authorization,
                    resume=True,
                    transcriber_factory=lambda *_args: self.fail(
                        "model must not load for tampered state"
                    ),
                )

    def test_manifest_chunking_and_paired_results_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "a.wav"
            audio.touch()
            manifest = root / "fleurs.jsonl"
            _write_manifest(manifest, [audio])
            rows = load_fleurs_manifest(manifest, expected_rows=1, formal=False)
            self.assertEqual(rows[0]["snr"], "clean")
            waveform = list(range(125))
            chunks = split_waveform(waveform, sample_rate=2)
            self.assertEqual([len(chunk) for chunk in chunks], [42, 42, 41])
            self.assertEqual([value for chunk in chunks for value in chunk], waveform)
            self.assertEqual(join_chunk_hypotheses([" xin ", "", " chào "]), "xin chào")

            runs = tuple(_run(item, root) for item in _locked_runs())
            artifacts: list[tuple[ExternalRun, Path]] = []
            for index, run in enumerate(runs):
                path = root / run.prediction_name
                row = {
                    "utt_id": "same",
                    **run.run_metadata,
                    "snr": "clean",
                    "noise_type": "clean",
                    "ref": "khác" if index == 2 else "một",
                    "hyp": "một",
                }
                atomic_write_csv(path, [row], CANONICAL_PREDICTION_COLUMNS)
                artifacts.append((run, path))
            with self.assertRaisesRegex(ExternalFleursError, "paired FLEURS run"):
                build_external_results(artifacts)


if __name__ == "__main__":
    unittest.main()
