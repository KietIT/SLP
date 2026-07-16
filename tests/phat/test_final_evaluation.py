from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import src.vitonesr.phat.final_evaluation as final_module
from src.vitonesr.phat.final_evaluation import (
    PREDICTION_COLUMNS,
    ROLE_ORDER,
    SUITE_VERSION,
    FinalLoraAuthorization,
    FinalLoraProtocolError,
    FinalLoraRole,
    authorize_final_lora,
    load_final_lora_config,
    run_final_lora_suite,
    validate_final_lora_config,
)
from src.vitonesr.phat.protocol import (
    canonical_sha256,
    sha256_file,
    source_test_evaluation_contract_payload,
)


def _hash(character: str) -> str:
    return character * 64


def _config() -> dict[str, object]:
    return {
        "suite_version": SUITE_VERSION,
        "protocol": {
            "formal": True,
            "final_test_unlocked": True,
            "split_lock": "protocol/split.json",
            "expected_split_lock_sha256": _hash("1"),
            "decision_lock": "protocol/decision.json",
            "expected_decision_lock_sha256": _hash("2"),
            "method_lock": "protocol/method.json",
            "expected_method_lock_sha256": _hash("3"),
            "method_config": "configs/lambda0.yaml",
            "noise_split_lock": "protocol/noise.json",
            "expected_noise_split_lock_sha256": _hash("4"),
            "final_benchmark_lock": "protocol/final.json",
            "expected_final_benchmark_lock_sha256": _hash("5"),
        },
        "benchmark": {
            "manifest": "benchmark/final.jsonl",
            "expected_manifest_sha256": _hash("6"),
            "expected_rows": 2300,
            "verify_audio_sha256": True,
        },
        "runtime": {
            "device": "cpu",
            "verify_method_audio_sha256": False,
        },
        "output": {
            "directory": "outputs/final_predictions",
            "aggregate_filename": "final_lora_results.csv",
        },
    }


def _contract(batch_size: int = 1) -> dict[str, object]:
    return {
        "contract_version": "paper_v2_evaluation_contract_v2",
        "model": {
            "name_or_path": "vinai/PhoWhisper-base",
            "revision": "a" * 40,
            "language": "vi",
            "task": "transcribe",
        },
        "evaluation": {"batch_size": batch_size, "inference_precision": "fp32"},
        "effective_audio": {"sample_rate": 16000, "max_audio_seconds": 15.0},
        "decoding": {
            "implementation": "whisper_generate_greedy_v1",
            "max_new_tokens": 128,
            "language": "vi",
            "task": "transcribe",
            "do_sample": False,
            "num_beams": 1,
        },
    }


def _role(name: str, index: int, contract: dict[str, object] | None = None) -> FinalLoraRole:
    value = contract or _contract()
    return FinalLoraRole(
        role=name,
        configuration_id=f"configuration_{index}",
        method_id="ordinary_lora" if index == 0 else "corrected_decoder_tone_lora",
        train_type="ordinary_lora" if index == 0 else "tone_aware_lora",
        lambda_value=(0.0, 0.05, 0.1)[index],
        seed=42,
        checkpoint_path=Path(f"checkpoints/{name}"),
        checkpoint_display=f"checkpoints/{name}",
        checkpoint_sha256=str(index + 7) * 64,
        resolved_config_sha256=str(index + 4) * 64,
        training_contract_sha256=str(index + 1) * 64,
        config={},
        source_test_contract=value,
        source_test_contract_sha256=canonical_sha256(value),
    )


def _decision() -> dict[str, object]:
    roles = []
    for index, role in enumerate(ROLE_ORDER):
        roles.append(
            {
                "role": role,
                "configuration_id": f"configuration_{index}",
                "method_id": "ordinary_lora" if index == 0 else "corrected_decoder_tone_lora",
                "train_type": "ordinary_lora" if index == 0 else "tone_aware_lora",
                "lambda": (0.0, 0.05, 0.1)[index],
                "seed": 42,
                "checkpoint_path": f"checkpoints/{role}",
            }
        )
    return {
        "split_lock_sha256": _hash("1"),
        "decision_lock_sha256": _hash("2"),
        "method_lock_sha256": _hash("3"),
        "method_identity_sha256": _hash("7"),
        "test_manifest": "manifests/source_test.jsonl",
        "test_manifest_sha256": _hash("8"),
        "test_utterance_count": 460,
        "allowed_test_evaluation_contract_sha256": (canonical_sha256(_contract()),),
        "locked_configurations": tuple(roles),
    }


def _authorization(
    root: Path | None = None,
    *,
    final_rows: int = 5,
) -> FinalLoraAuthorization:
    roles = tuple(_role(name, index) for index, name in enumerate(ROLE_ORDER))
    return FinalLoraAuthorization(
        split_lock_sha256=_hash("1"),
        decision_lock_sha256=_hash("2"),
        method_lock_sha256=_hash("3"),
        method_identity_sha256=_hash("7"),
        noise_split_lock_sha256=_hash("4"),
        final_benchmark_lock_sha256=_hash("5"),
        final_manifest=(root / "benchmark/final.jsonl") if root else Path("benchmark/final.jsonl"),
        final_manifest_sha256=_hash("6"),
        final_rows=final_rows,
        final_audio_inventory_sha256=_hash("9"),
        source_test_manifest="manifests/source_test.jsonl",
        source_test_manifest_sha256=_hash("8"),
        source_test_rows=460,
        roles=roles,
        inference_contract=_contract(),
        method_integrity={"method_lock_sha256": _hash("3")},
        runtime_config_sha256=_hash("f"),
        runtime_config_path="outputs/protocol/final_lora_runtime.yaml",
    )


class FinalEvaluationTests(unittest.TestCase):
    def test_source_test_contract_preregisters_decode_without_final_manifest(self) -> None:
        config = {
            "model": {
                "name_or_path": "vinai/PhoWhisper-base",
                "revision": "a" * 40,
                "language": "vi",
                "task": "transcribe",
            },
            "data": {"sample_rate": 16000},
            "evaluation": {
                "manifest": "derived/noisy_dev.jsonl",
                "prediction_path": "outputs/dev.csv",
                "data_split": "dev",
                "benchmark_protocol": "noisy_dev",
                "locked_vivos_split": "dev",
                "noisy_dev_lock": "protocol/noisy_dev.json",
                "expected_noisy_dev_lock_sha256": _hash("a"),
                "expected_noise_split_lock_sha256": _hash("b"),
                "expected_source_dev_sha256": _hash("c"),
                "expected_manifest_sha256": _hash("d"),
                "expected_total_rows": 14125,
                "batch_size": 1,
                "inference_precision": "fp16",
                "max_new_tokens": 128,
                "sample_rate": 16000,
                "max_audio_seconds": 15.0,
            },
        }
        payload = source_test_evaluation_contract_payload(
            config,
            source_manifest="manifests/vivos_test_locked.jsonl",
            source_manifest_sha256=_hash("e"),
            source_rows=460,
        )
        evaluation = payload["evaluation"]
        self.assertEqual(evaluation["benchmark_protocol"], "locked_vivos")
        self.assertEqual(evaluation["expected_manifest_sha256"], _hash("e"))
        self.assertEqual(evaluation["expected_total_rows"], 460)
        self.assertNotIn("noisy_dev_lock", evaluation)
        self.assertNotIn("final_benchmark_manifest", json.dumps(payload))
        self.assertEqual(payload["decoding"]["do_sample"], False)
        changed = json.loads(json.dumps(config))
        changed["evaluation"]["batch_size"] = 2
        self.assertNotEqual(
            canonical_sha256(payload),
            canonical_sha256(
                source_test_evaluation_contract_payload(
                    changed,
                    source_manifest="manifests/vivos_test_locked.jsonl",
                    source_manifest_sha256=_hash("e"),
                    source_rows=460,
                )
            ),
        )

    def test_hashes_paths_and_placeholders_fail_closed(self) -> None:
        config = _config()
        validate_final_lora_config(config)
        config["protocol"]["expected_decision_lock_sha256"] = "REQUIRED_AFTER_DECISION"
        with self.assertRaisesRegex(FinalLoraProtocolError, "concrete"):
            validate_final_lora_config(config)
        config = _config()
        config["output"]["directory"] = "../external"
        with self.assertRaisesRegex(FinalLoraProtocolError, "repository-relative"):
            validate_final_lora_config(config)

    def test_materialized_runtime_config_exact_bytes_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "outputs/protocol/final_lora_runtime.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(
                yaml.safe_dump(_config(), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with patch.object(final_module, "ROOT", root):
                loaded = load_final_lora_config(path)
            self.assertEqual(loaded["_runtime_config_sha256"], sha256_file(path))
            self.assertEqual(
                loaded["_runtime_config_path"],
                "outputs/protocol/final_lora_runtime.yaml",
            )

    def test_authorization_order_is_decision_method_noise_final_then_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            events: list[str] = []

            def decision_verifier(**_kwargs):
                events.append("decision")
                self.assertIs(_kwargs.get("verify_checkpoints"), False)
                return _decision()

            def method_loader(_path):
                events.append("method_config")
                return {}

            def method_verifier(*_args, **_kwargs):
                events.append("method")
                return {
                    "method_lock_sha256": _hash("3"),
                    "method_identity_sha256": _hash("7"),
                    "protocol_split_lock_sha256": _hash("1"),
                }

            def noise_verifier(*_args, **_kwargs):
                events.append("noise")
                return {"lock_sha256": _hash("4"), "lock": {}}

            def benchmark_verifier(*_args, **_kwargs):
                events.append("final_lock")
                return {
                    "lock_sha256": _hash("5"),
                    "audio_inventory_sha256": _hash("9"),
                }

            def role_verifier(_raw, *, role, **_kwargs):
                events.append(role)
                return _role(role, ROLE_ORDER.index(role))

            with patch.object(final_module, "ROOT", root):
                authorization = authorize_final_lora(
                    _config(),
                    decision_verifier=decision_verifier,
                    method_config_loader=method_loader,
                    method_verifier=method_verifier,
                    noise_verifier=noise_verifier,
                    benchmark_verifier=benchmark_verifier,
                    role_verifier=role_verifier,
                )
            self.assertEqual(
                events,
                ["decision", "method_config", "method", "noise", "final_lock", *ROLE_ORDER],
            )
            self.assertEqual(tuple(role.role for role in authorization.roles), ROLE_ORDER)

    def test_failed_final_lock_prevents_role_or_manifest_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            role_called = False

            def fail_final(*_args, **_kwargs):
                raise FinalLoraProtocolError("final lock rejected")

            def role_verifier(*_args, **_kwargs):
                nonlocal role_called
                role_called = True
                raise AssertionError("role/model access occurred")

            with patch.object(final_module, "ROOT", root):
                with self.assertRaisesRegex(FinalLoraProtocolError, "final lock rejected"):
                    authorize_final_lora(
                        _config(),
                        decision_verifier=lambda **_kwargs: _decision(),
                        method_config_loader=lambda _path: {},
                        method_verifier=lambda *_args, **_kwargs: {
                            "method_lock_sha256": _hash("3"),
                            "method_identity_sha256": _hash("7"),
                            "protocol_split_lock_sha256": _hash("1"),
                        },
                        noise_verifier=lambda *_args, **_kwargs: {
                            "lock_sha256": _hash("4"),
                            "lock": {},
                        },
                        benchmark_verifier=fail_final,
                        role_verifier=role_verifier,
                    )
            self.assertFalse(role_called)

    def test_duplicate_role_and_semantic_contract_tamper_are_rejected(self) -> None:
        common = {
            "decision_verifier": lambda **_kwargs: _decision(),
            "method_config_loader": lambda _path: {},
            "method_verifier": lambda *_args, **_kwargs: {
                "method_lock_sha256": _hash("3"),
                "method_identity_sha256": _hash("7"),
                "protocol_split_lock_sha256": _hash("1"),
            },
            "noise_verifier": lambda *_args, **_kwargs: {
                "lock_sha256": _hash("4"),
                "lock": {},
            },
            "benchmark_verifier": lambda *_args, **_kwargs: {
                "lock_sha256": _hash("5"),
                "audio_inventory_sha256": _hash("9"),
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            decision = _decision()
            decision["locked_configurations"][2]["role"] = "selected_method"
            with patch.object(final_module, "ROOT", root):
                with self.assertRaisesRegex(FinalLoraProtocolError, "unique"):
                    authorize_final_lora(
                        _config(),
                        **{**common, "decision_verifier": lambda **_kwargs: decision},
                        role_verifier=lambda *_args, **_kwargs: _role("ordinary_baseline", 0),
                    )

            def changed_contract(_raw, *, role, **_kwargs):
                index = ROLE_ORDER.index(role)
                return _role(role, index, _contract(2) if index == 2 else _contract())

            with patch.object(final_module, "ROOT", root):
                with self.assertRaisesRegex(FinalLoraProtocolError, "do not share"):
                    authorize_final_lora(
                        _config(), **common, role_verifier=changed_contract
                    )

    def test_exact_three_role_atomic_outputs_resume_and_tamper_detection(self) -> None:
        rows = [
            {
                "utt_id": f"u_{snr}",
                "source_utt_id": "source",
                "dataset": "vivos",
                "audio_path": f"audio/{snr}.wav",
                "audio_sha256": _hash("a"),
                "snr": snr,
                "noise_type": "clean" if snr == "clean" else "noise",
                "ref": "đã có một và",
            }
            for snr in ("clean", "20", "10", "5", "0")
        ]
        predictor_calls: list[str] = []

        def predictor(role, benchmark_rows, _contract_value, *, device_arg):
            predictor_calls.append(role.role)
            self.assertEqual(device_arg, "cpu")
            return [row["ref"] for row in benchmark_rows], {
                "batch_size": 1,
                "device_type": "cpu",
                "dtype": "torch.float32",
                "torch_version": "test",
                "transformers_version": "test",
                "cuda_version": None,
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            with (
                patch.object(final_module, "ROOT", root),
                patch.object(final_module, "authorize_final_lora", return_value=_authorization(root)),
                patch.object(final_module, "load_authorized_final_benchmark", return_value=rows),
            ):
                result = run_final_lora_suite(config, predictor=predictor)
                self.assertEqual(result["roles"], list(ROLE_ORDER))
                self.assertEqual(predictor_calls, list(ROLE_ORDER))
                for role in ROLE_ORDER:
                    prediction = root / "outputs/final_predictions" / role / "predictions.csv"
                    provenance = prediction.with_name("provenance.json")
                    with prediction.open("r", encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        self.assertEqual(tuple(reader.fieldnames or ()), PREDICTION_COLUMNS)
                        self.assertEqual(len(list(reader)), 5)
                    payload = provenance.read_text(encoding="utf-8")
                    self.assertNotIn(str(root), payload)
                aggregate = (
                    root
                    / "outputs/final_predictions/aggregate/final_lora_results.csv"
                )
                self.assertTrue(aggregate.is_file())

                run_final_lora_suite(config, resume=True, predictor=predictor)
                self.assertEqual(predictor_calls, list(ROLE_ORDER))

                selected = (
                    root
                    / "outputs/final_predictions/selected_method/predictions.csv"
                )
                selected.write_bytes(selected.read_bytes() + b"tamper")
                with self.assertRaisesRegex(FinalLoraProtocolError, "hash mismatch"):
                    run_final_lora_suite(config, resume=True, predictor=predictor)

    def test_incremental_role_resume_continues_at_exact_next_batch(self) -> None:
        rows = [
            {
                "utt_id": f"u_{index}",
                "source_utt_id": f"source_{index}",
                "dataset": "vivos",
                "audio_path": f"audio/{index}.wav",
                "audio_sha256": _hash("a"),
                "snr": "clean" if index == 0 else str((20, 10, 5, 0)[(index - 1) % 4]),
                "noise_type": "clean" if index == 0 else "noise",
                "ref": f"câu số {index}",
            }
            for index in range(5)
        ]
        calls: list[tuple[str, int]] = []
        interrupt = {"enabled": True}
        runtime = {
            "batch_size": 1,
            "device_type": "cpu",
            "dtype": "torch.float32",
            "torch_version": "test",
            "transformers_version": "test",
            "cuda_version": None,
        }

        def predictor(
            role,
            benchmark_rows,
            _contract_value,
            *,
            device_arg,
            start_index,
            on_batch,
        ):
            self.assertEqual(device_arg, "cpu")
            calls.append((role.role, start_index))
            generated: list[str] = []
            for index in range(start_index, len(benchmark_rows)):
                hypothesis = benchmark_rows[index]["ref"]
                on_batch([hypothesis], runtime)
                generated.append(hypothesis)
                if (
                    interrupt["enabled"]
                    and role.role == "ordinary_baseline"
                    and len(generated) == 2
                ):
                    raise RuntimeError("simulated inference interruption")
            return generated, runtime

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            with (
                patch.object(final_module, "ROOT", root),
                patch.object(
                    final_module,
                    "authorize_final_lora",
                    return_value=_authorization(root, final_rows=len(rows)),
                ),
                patch.object(
                    final_module,
                    "load_authorized_final_benchmark",
                    return_value=rows,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    run_final_lora_suite(config, predictor=predictor)
                partial_dir = (
                    root
                    / "outputs/final_predictions/.ordinary_baseline.partial"
                )
                with (partial_dir / "predictions.partial.csv").open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 2)
                self.assertTrue((partial_dir / "resume.json").is_file())
                self.assertFalse((partial_dir / "recovery.json").exists())

                interrupt["enabled"] = False
                result = run_final_lora_suite(
                    config,
                    resume=True,
                    predictor=predictor,
                )
                self.assertEqual(result["prediction_rows_per_role"], 5)
                self.assertEqual(
                    calls,
                    [
                        ("ordinary_baseline", 0),
                        ("ordinary_baseline", 2),
                        ("selected_method", 0),
                        ("locked_control", 0),
                    ],
                )
                for role in ROLE_ORDER:
                    self.assertFalse(
                        (
                            root
                            / f"outputs/final_predictions/.{role}.partial"
                        ).exists()
                    )
                    prediction = (
                        root
                        / f"outputs/final_predictions/{role}/predictions.csv"
                    )
                    with prediction.open(
                        "r", encoding="utf-8", newline=""
                    ) as handle:
                        self.assertEqual(len(list(csv.DictReader(handle))), 5)

    def test_write_ahead_receipt_recovers_csv_before_state_and_rejects_tamper(
        self,
    ) -> None:
        rows = [
            {
                "utt_id": f"u_{index}",
                "source_utt_id": f"source_{index}",
                "dataset": "vivos",
                "audio_path": f"audio/{index}.wav",
                "audio_sha256": _hash("a"),
                "snr": "clean" if index == 0 else "20",
                "noise_type": "clean" if index == 0 else "noise",
                "ref": f"câu số {index}",
            }
            for index in range(3)
        ]
        runtime = {
            "batch_size": 1,
            "device_type": "cpu",
            "dtype": "torch.float32",
            "torch_version": "test",
            "transformers_version": "test",
            "cuda_version": None,
        }
        predictor_calls: list[tuple[str, int]] = []

        def predictor(
            role,
            benchmark_rows,
            _contract_value,
            *,
            device_arg,
            start_index,
            on_batch,
        ):
            predictor_calls.append((role.role, start_index))
            generated = []
            for index in range(start_index, len(benchmark_rows)):
                hypothesis = benchmark_rows[index]["ref"]
                on_batch([hypothesis], runtime)
                generated.append(hypothesis)
            return generated, runtime

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            original_atomic_write = final_module._atomic_write_bytes
            crashed = {"done": False}

            def crash_before_first_state(path, payload):
                if path.name == "resume.json" and not crashed["done"]:
                    crashed["done"] = True
                    raise KeyboardInterrupt("CSV committed before state")
                return original_atomic_write(path, payload)

            with (
                patch.object(final_module, "ROOT", root),
                patch.object(
                    final_module,
                    "authorize_final_lora",
                    return_value=_authorization(root, final_rows=len(rows)),
                ),
                patch.object(
                    final_module,
                    "load_authorized_final_benchmark",
                    return_value=rows,
                ),
            ):
                with patch.object(
                    final_module,
                    "_atomic_write_bytes",
                    side_effect=crash_before_first_state,
                ):
                    with self.assertRaisesRegex(
                        KeyboardInterrupt, "CSV committed"
                    ):
                        run_final_lora_suite(config, predictor=predictor)
                partial_dir = (
                    root
                    / "outputs/final_predictions/.ordinary_baseline.partial"
                )
                partial = partial_dir / "predictions.partial.csv"
                receipt = partial_dir / "recovery.json"
                state = partial_dir / "resume.json"
                self.assertTrue(partial.is_file())
                self.assertTrue(receipt.is_file())
                self.assertFalse(state.exists())
                original_partial = partial.read_bytes()
                tampered_rows, columns = final_module._read_prediction(partial)
                self.assertEqual(columns, PREDICTION_COLUMNS)
                tampered_rows[0]["hyp"] = "tampered hypothesis"
                partial.write_bytes(
                    final_module._csv_bytes(tampered_rows, PREDICTION_COLUMNS)
                )
                with self.assertRaisesRegex(
                    FinalLoraProtocolError,
                    "neither recovery hash.*tamper",
                ):
                    run_final_lora_suite(
                        config,
                        resume=True,
                        predictor=lambda *_args, **_kwargs: self.fail(
                            "tamper must fail before predictor access"
                        ),
                    )
                partial.write_bytes(original_partial)
                result = run_final_lora_suite(
                    config,
                    resume=True,
                    predictor=predictor,
                )
                self.assertEqual(result["prediction_rows_per_role"], 3)
                self.assertIn(("ordinary_baseline", 1), predictor_calls)

    def test_orphan_and_state_contract_tamper_fail_before_predictor(self) -> None:
        rows = [
            {
                "utt_id": f"u_{index}",
                "source_utt_id": f"source_{index}",
                "dataset": "vivos",
                "audio_path": f"audio/{index}.wav",
                "audio_sha256": _hash("a"),
                "snr": "clean" if index == 0 else "20",
                "noise_type": "clean" if index == 0 else "noise",
                "ref": f"câu số {index}",
            }
            for index in range(2)
        ]
        runtime = {
            "batch_size": 1,
            "device_type": "cpu",
            "dtype": "torch.float32",
            "torch_version": "test",
            "transformers_version": "test",
            "cuda_version": None,
        }

        def interrupted_predictor(
            role,
            benchmark_rows,
            _contract_value,
            *,
            device_arg,
            start_index,
            on_batch,
        ):
            on_batch([benchmark_rows[start_index]["ref"]], runtime)
            raise RuntimeError("stop after one committed batch")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            with (
                patch.object(final_module, "ROOT", root),
                patch.object(
                    final_module,
                    "authorize_final_lora",
                    return_value=_authorization(root, final_rows=len(rows)),
                ),
                patch.object(
                    final_module,
                    "load_authorized_final_benchmark",
                    return_value=rows,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "committed batch"):
                    run_final_lora_suite(
                        config, predictor=interrupted_predictor
                    )
                partial_dir = (
                    root
                    / "outputs/final_predictions/.ordinary_baseline.partial"
                )
                state_path = partial_dir / "resume.json"
                original_state = state_path.read_bytes()
                state_path.unlink()
                with self.assertRaisesRegex(
                    FinalLoraProtocolError,
                    "Orphan final-LoRA partial CSV/state",
                ):
                    run_final_lora_suite(
                        config,
                        resume=True,
                        predictor=lambda *_args, **_kwargs: self.fail(
                            "orphan must fail before predictor"
                        ),
                    )
                state_path.write_bytes(original_state)
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["decision_lock_sha256"] = _hash("e")
                state_path.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaisesRegex(
                    FinalLoraProtocolError,
                    "Partial state mismatch: decision_lock_sha256",
                ):
                    run_final_lora_suite(
                        config,
                        resume=True,
                        predictor=lambda *_args, **_kwargs: self.fail(
                            "state tamper must fail before predictor"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
