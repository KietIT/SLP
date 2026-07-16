from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.vitonesr.phat.evaluation import run_checkpoint_evaluation
from src.vitonesr.phat.method_contract import (
    DEFAULT_SOURCE_COMPONENTS,
    METHOD_CONTRACT_VERSION,
    MethodContractError,
    _validate_environment,
    build_method_contract,
    validate_method_contract,
    verify_checkpoint_method_binding,
    verify_method_lock,
    write_method_lock,
)
from src.vitonesr.phat.protocol import canonical_sha256, sha256_file
from src.vitonesr.phat.trainer import train_experiment


def _config(noisy_dev_sha256: str) -> dict[str, object]:
    return {
        "seed": 42,
        "experiment": {
            "method_id": "ordinary_lora",
            "train_type": "ordinary_lora",
        },
        "protocol": {
            "split_lock": "split.json",
            "noise_split_lock": "noise.json",
            "noisy_dev_lock": "noisy.json",
            "method_lock": "method.json",
            "decision_lock": "decision.json",
            "verify_audio_sha256": True,
            "formal_training_unlocked": True,
            "final_test_unlocked": False,
        },
        "model": {
            "name_or_path": "vinai/PhoWhisper-base",
            "revision": "a" * 40,
            "language": "vi",
            "task": "transcribe",
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        "training": {
            "run_scope": "formal",
            "lambda_tone": 0.0,
            "output_dir": "outputs/checkpoint",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1.0e-4,
            "tone_label_policy": "last_subtoken",
            "checkpoint_metric": "dev_asr_loss",
            "checkpoint_mode": "min",
        },
        "data": {
            "train_manifest": "train.jsonl",
            "valid_manifest": "dev.jsonl",
            "sample_rate": 16000,
        },
        "noise": {
            "enable_train_noise": True,
            "noise_manifest": "noise_train.jsonl",
            "snr_choices": [20, 10, 5, 0],
            "prob": 0.7,
        },
        "evaluation": {
            "manifest": "noisy_dev.jsonl",
            "prediction_path": "outputs/pred.csv",
            "data_split": "dev",
            "locked_vivos_split": "dev",
            "expected_manifest_sha256": noisy_dev_sha256,
            "expected_total_rows": 5,
            "max_new_tokens": 128,
            "batch_size": 1,
            "inference_precision": "fp16",
            "sample_rate": 16000,
            "max_audio_seconds": 15.0,
        },
        "selection": {
            "required_evaluation_split": "dev",
            "expected_manifest_sha256": noisy_dev_sha256,
            "expected_evaluation_contract_sha256": "b" * 64,
            "require_full_manifest": True,
            "low_snr": [0, 5],
            "ter_weight": 0.5,
            "der_weight": 0.5,
            "max_wer_absolute_increase": 0.05,
            "max_cer_absolute_increase": 0.03,
            "min_ter_coverage_ratio_vs_baseline": 0.98,
            "min_der_coverage_ratio_vs_baseline": 0.98,
            "min_fcer_coverage_ratio_vs_baseline": 0.98,
            "guard_split": "all",
            "guard_snr": "all",
            "allow_lambda_zero": True,
        },
    }


class MethodContractTests(unittest.TestCase):
    def test_default_source_component_inventory_exists(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        missing = [
            component
            for component in DEFAULT_SOURCE_COMPONENTS
            if not (repository_root / component).is_file()
        ]
        self.assertEqual(missing, [])

    def _fixture(self, root: Path) -> tuple[dict[str, object], dict[str, object]]:
        payloads = {
            "train.jsonl": b'{"utt_id":"train"}\n',
            "dev.jsonl": b'{"utt_id":"dev"}\n',
            "noise_train.jsonl": b'{"noise_id":"n"}\n',
            "noisy_dev.jsonl": b'{"utt_id":"dev_noisy"}\n',
            "split.json": b'{"kind":"split"}\n',
            "noise.json": b'{"kind":"noise"}\n',
            "noisy.json": b'{"kind":"noisy"}\n',
            "environment.json": b'{"kind":"environment"}\n',
            "component.py": b"VALUE = 1\n",
        }
        for relative, payload in payloads.items():
            (root / relative).write_bytes(payload)
        config = _config(sha256_file(root / "noisy_dev.jsonl"))
        dependency = {
            "split": {
                "splits": {
                    "train": {"manifest_sha256": sha256_file(root / "train.jsonl")},
                    "dev": {"manifest_sha256": sha256_file(root / "dev.jsonl")},
                }
            },
            "noise": {
                "lock": {
                    "splits": {
                        "train": {
                            "manifest_sha256": sha256_file(root / "noise_train.jsonl")
                        }
                    }
                },
                "lock_sha256": sha256_file(root / "noise.json"),
            },
            "noisy_dev": {
                "lock_sha256": sha256_file(root / "noisy.json"),
                "manifest_sha256": sha256_file(root / "noisy_dev.jsonl"),
            },
        }
        return config, dependency

    @staticmethod
    def _environment() -> dict[str, object]:
        return {
            "identity_sha256": "e" * 64,
            "environment": {"capture_mode": "formal"},
        }

    def _build(self, root: Path) -> tuple[dict[str, object], dict[str, object]]:
        config, dependency = self._fixture(root)
        with (
            patch(
                "src.vitonesr.phat.method_contract._verify_protocol_dependencies",
                return_value=dependency,
            ),
            patch(
                "src.vitonesr.phat.method_contract._validate_environment",
                return_value=self._environment(),
            ),
        ):
            artifact = build_method_contract(
                config,
                split_lock_path="split.json",
                noise_split_lock_path="noise.json",
                noisy_dev_lock_path="noisy.json",
                environment_path="environment.json",
                source_components=("component.py",),
                lambda_grid=(0.0, 0.05, 0.1, 0.3, 0.5),
                repo_root=root,
                formal=True,
                verify_audio=True,
            )
        return config, artifact

    def test_build_is_deterministic_path_free_and_binds_all_five_lambdas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, first = self._build(root)
            _, second = self._build(root)
            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], METHOD_CONTRACT_VERSION)
            self.assertEqual(
                list(first["training"]["contract_sha256_by_lambda"]),  # type: ignore[index]
                ["0", "0.05", "0.1", "0.3", "0.5"],
            )
            self.assertNotIn(str(root), json.dumps(first))
            validate_method_contract(first)
            self.assertEqual(config["training"]["lambda_tone"], 0.0)  # type: ignore[index]

    def test_identity_tamper_and_absolute_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, artifact = self._build(Path(temporary_directory))
            tampered = copy.deepcopy(artifact)
            tampered["model"]["lora"]["r"] = 16  # type: ignore[index]
            with self.assertRaisesRegex(MethodContractError, "identity"):
                validate_method_contract(tampered)

            unsafe = copy.deepcopy(artifact)
            unsafe["artifacts"]["split_lock"]["path"] = "C:/private/split.json"  # type: ignore[index]
            payload = dict(unsafe)
            payload.pop("identity_sha256")
            unsafe["identity_sha256"] = canonical_sha256(payload)
            with self.assertRaisesRegex(MethodContractError, "absolute path"):
                validate_method_contract(unsafe)

    def test_write_is_canonical_atomic_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, artifact = self._build(root)
            output = root / "nested" / "method.json"
            write_method_lock(output, artifact)
            expected = (
                json.dumps(
                    artifact,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            self.assertEqual(output.read_bytes(), expected)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_method_lock(output, artifact)
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_atomic_publish_failure_leaves_no_partial_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, artifact = self._build(root)
            output = root / "nested" / "method.json"
            with patch(
                "src.vitonesr.phat.method_contract.os.link",
                side_effect=OSError("simulated publish failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    write_method_lock(output, artifact)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_verify_fails_closed_on_component_or_config_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, artifact = self._build(root)
            lock = root / "method.json"
            write_method_lock(lock, artifact)
            _, dependency = self._fixture(root)
            patches = (
                patch(
                    "src.vitonesr.phat.method_contract._verify_protocol_dependencies",
                    return_value=dependency,
                ),
                patch(
                    "src.vitonesr.phat.method_contract._validate_environment",
                    return_value=self._environment(),
                ),
            )
            with patches[0], patches[1]:
                integrity = verify_method_lock(
                    "method.json",
                    config=config,
                    repo_root=root,
                    formal=True,
                )
            self.assertEqual(integrity["method_lock_sha256"], sha256_file(lock))

            changed_config = copy.deepcopy(config)
            changed_config["model"]["lora"]["r"] = 16  # type: ignore[index]
            with (
                patch(
                    "src.vitonesr.phat.method_contract._verify_protocol_dependencies",
                    return_value=dependency,
                ),
                patch(
                    "src.vitonesr.phat.method_contract._validate_environment",
                    return_value=self._environment(),
                ),
                self.assertRaisesRegex(MethodContractError, "LoRA"),
            ):
                verify_method_lock(
                    "method.json",
                    config=changed_config,
                    repo_root=root,
                    formal=True,
                )

            (root / "component.py").write_text("VALUE = 2\n", encoding="utf-8")
            with (
                patch(
                    "src.vitonesr.phat.method_contract._verify_protocol_dependencies",
                    return_value=dependency,
                ),
                patch(
                    "src.vitonesr.phat.method_contract._validate_environment",
                    return_value=self._environment(),
                ),
                self.assertRaisesRegex(MethodContractError, "source component SHA-256 mismatch"),
            ):
                verify_method_lock(
                    "method.json",
                    config=config,
                    repo_root=root,
                    formal=True,
                )

    def test_checkpoint_must_bind_exact_method_environment_and_source(self) -> None:
        integrity = {
            "method_lock_sha256": "1" * 64,
            "method_identity_sha256": "2" * 64,
            "environment_artifact_sha256": "3" * 64,
            "environment_identity_sha256": "4" * 64,
            "source_tree_sha256": "5" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            (checkpoint / "resolved_config.yaml").write_text(
                yaml.safe_dump({"runtime_protocol": integrity}), encoding="utf-8"
            )
            verify_checkpoint_method_binding(checkpoint, integrity)
            stale = dict(integrity)
            stale["source_tree_sha256"] = "6" * 64
            with self.assertRaisesRegex(MethodContractError, "source_tree_sha256"):
                verify_checkpoint_method_binding(checkpoint, stale)

    def test_formal_environment_is_compared_with_current_runtime(self) -> None:
        runtime = {
            "torch_version": "2.6.0",
            "cuda": {
                "available": True,
                "compiled_version": "12.4",
                "device_count": 1,
                "devices": [{"index": 0, "name": "GPU", "total_memory_bytes": 8}],
                "query_status": "available",
            },
            "cudnn": {"available": True, "version": 90100},
        }
        locked_environment = {
            "capture_mode": "formal",
            "packages": {"torch": "2.6.0"},
            "python": {"version": "3.12.0"},
            "platform": {"system": "Windows"},
            "runtime": runtime,
        }
        artifact = {
            "identity_sha256": "e" * 64,
            "environment": locked_environment,
        }
        current = {
            "environment": {
                **locked_environment,
                "capture_mode": "diagnostic",
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "environment.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with (
                patch(
                    "src.vitonesr.phat.method_contract.validate_environment_artifact"
                ),
                patch(
                    "src.vitonesr.phat.method_contract.capture_environment",
                    return_value=current,
                ),
            ):
                _validate_environment(
                    path,
                    formal=True,
                    repo_root=temporary_directory,
                    verify_current=True,
                )
                stale = copy.deepcopy(current)
                stale["environment"]["packages"]["torch"] = "9.9"  # type: ignore[index]
                with (
                    patch(
                        "src.vitonesr.phat.method_contract.capture_environment",
                        return_value=stale,
                    ),
                    self.assertRaisesRegex(MethodContractError, "packages"),
                ):
                    _validate_environment(
                        path,
                        formal=True,
                        repo_root=temporary_directory,
                        verify_current=True,
                    )

    def test_formal_train_and_evaluation_authorize_before_data_or_checkpoint(self) -> None:
        config = _config("a" * 64)
        with (
            patch(
                "src.vitonesr.phat.trainer.verify_method_lock",
                side_effect=MethodContractError("method denied"),
            ) as training_authorizer,
            patch(
                "src.vitonesr.phat.trainer.verify_locked_vivos_manifest"
            ) as training_manifest,
            self.assertRaisesRegex(MethodContractError, "method denied"),
        ):
            train_experiment(config)  # type: ignore[arg-type]
        training_authorizer.assert_called_once()
        training_manifest.assert_not_called()

        with (
            patch(
                "src.vitonesr.phat.evaluation.verify_method_lock",
                side_effect=MethodContractError("method denied"),
            ) as evaluation_authorizer,
            patch("src.vitonesr.phat.evaluation.resolve_checkpoint") as checkpoint,
            patch("src.vitonesr.phat.evaluation.load_split_lock") as manifest_lock,
            self.assertRaisesRegex(MethodContractError, "method denied"),
        ):
            run_checkpoint_evaluation(
                config,  # type: ignore[arg-type]
                checkpoint="missing",
                batch_size=1,
            )
        evaluation_authorizer.assert_called_once()
        checkpoint.assert_not_called()
        manifest_lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
