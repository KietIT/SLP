from __future__ import annotations

import csv
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.make_vivos_manifest import build_protocol_outputs, write_locked_outputs
from src.vitonesr.phat.evaluation import run_checkpoint_evaluation
from src.vitonesr.phat.protocol import (
    DECISION_VERSION,
    ProtocolIntegrityError,
    canonical_sha256,
    checkpoint_inference_sha256,
    evaluation_contract_sha256,
    load_split_lock,
    resolve_locked_roles,
    selection_rule_sha256,
    sha256_file,
    training_contract_sha256,
    verify_checkpoint_config,
    verify_locked_vivos_manifest,
    verify_test_configuration_locked,
    verify_test_decision_lock,
)
from src.vitonesr.phat.selection import write_decision_lock


MODEL_REVISION = "7ebdb9e88f5cc5271fb88f4d642c82ff9388650e"


def _official_fixture(root: Path) -> Path:
    official = root / "vivos"
    train_prompts: list[str] = []
    for speaker in ("TRAIN01", "TRAIN02"):
        wave_dir = official / "train" / "waves" / speaker
        wave_dir.mkdir(parents=True, exist_ok=True)
        for number in (1, 2):
            utt_id = f"{speaker}_{number:03d}"
            audio = f"audio:{utt_id}".encode()
            (wave_dir / f"{utt_id}.wav").write_bytes(audio)
            train_prompts.append(f"{utt_id} train sentence {number}")
    train_prompt_path = official / "train" / "prompts.txt"
    train_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    train_prompt_path.write_text(
        "\n".join(train_prompts) + "\n", encoding="utf-8"
    )

    speaker = "TEST01"
    wave_dir = official / "test" / "waves" / speaker
    wave_dir.mkdir(parents=True, exist_ok=True)
    test_prompts: list[str] = []
    for number in (1, 2):
        utt_id = f"{speaker}_{number:03d}"
        (wave_dir / f"{utt_id}.wav").write_bytes(f"audio:{utt_id}".encode())
        test_prompts.append(f"{utt_id} test sentence {number}")
    test_prompt_path = official / "test" / "prompts.txt"
    test_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    test_prompt_path.write_text(
        "\n".join(test_prompts) + "\n", encoding="utf-8"
    )
    return official


def _legacy_benchmark_fixture(root: Path) -> Path:
    path = root / "legacy_benchmark.csv"
    fieldnames = [
        "utt_id",
        "dataset",
        "split",
        "condition",
        "snr",
        "source_utt_id",
    ]
    source_id = "TEST01_001"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for snr in ("clean", "20", "10", "5", "0"):
            writer.writerow(
                {
                    "utt_id": f"{source_id}_{'clean' if snr == 'clean' else 'snr' + snr}",
                    "dataset": "vivos",
                    "split": "test",
                    "condition": "clean" if snr == "clean" else "noisy",
                    "snr": snr,
                    "source_utt_id": source_id,
                }
            )
    return path


def _locked_fixture(root: Path) -> tuple[Path, Path, Path]:
    official = _official_fixture(root)
    manifest_dir = root / "manifests"
    protocol_dir = root / "protocol"
    payloads = build_protocol_outputs(
        official,
        manifest_dir=manifest_dir,
        protocol_dir=protocol_dir,
        legacy_benchmark_manifest=_legacy_benchmark_fixture(root),
        expected_legacy_exposed=1,
        seed=42,
        dev_speaker_fraction=0.5,
    )
    write_locked_outputs(payloads, overwrite=False)
    return (
        manifest_dir / "vivos_dev.jsonl",
        manifest_dir / "vivos_test_locked.jsonl",
        protocol_dir / "split_lock.json",
    )


def _formal_config(
    *,
    split_lock: Path,
    manifest: Path,
    data_split: str,
    method_id: str,
    train_type: str,
    lambda_value: float,
    selection_evaluation_contract: str | None = None,
) -> dict[str, object]:
    lock = json.loads(split_lock.read_text(encoding="utf-8"))
    noise_audio = split_lock.parent / "noise.wav"
    if not noise_audio.exists():
        noise_audio.write_bytes(b"noise-audio")
    noise_manifest = split_lock.parent / "noise_manifest.jsonl"
    if not noise_manifest.exists():
        noise_manifest.write_text(
            json.dumps(
                {
                    "audio": str(noise_audio),
                    "audio_sha256": sha256_file(noise_audio),
                    "noise_type": "noise",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    config: dict[str, object] = {
        "seed": 42,
        "protocol": {
            "split_lock": str(split_lock),
            "method_lock": str(split_lock.parent / "method_lock.json"),
            "decision_lock": str(split_lock.parent / "best_lambda_decision.json"),
            "verify_audio_sha256": True,
            "final_test_unlocked": data_split == "test",
        },
        "experiment": {"method_id": method_id, "train_type": train_type},
        "model": {
            "name_or_path": "vinai/PhoWhisper-base",
            "revision": MODEL_REVISION,
            "language": "vi",
            "task": "transcribe",
            "lora": {"r": 8, "lora_alpha": 16},
        },
        "training": {
            "run_scope": "formal",
            "lambda_tone": lambda_value,
            "num_train_epochs": 3,
        },
        "data": {
            "train_manifest": str(manifest.parent / "vivos_train.jsonl"),
            "valid_manifest": str(manifest.parent / "vivos_dev.jsonl"),
            "sample_rate": 16000,
        },
        "noise": {
            "enable_train_noise": True,
            "noise_manifest": str(noise_manifest),
            "snr_choices": [20, 10, 5, 0],
        },
        "evaluation": {
            "manifest": str(manifest),
            "prediction_path": str(
                manifest.parent / f"prediction_{method_id}_{data_split}.csv"
            ),
            "data_split": data_split,
            "locked_vivos_split": (
                "dev" if data_split == "dev" else "test_locked"
            ),
            "expected_manifest_sha256": sha256_file(manifest),
            "expected_total_rows": int(
                lock["splits"][
                    "dev" if data_split == "dev" else "test_locked"
                ]["utterance_count"]
            ),
            "sample_rate": 16000,
            "max_audio_seconds": 15.0,
            "max_new_tokens": 128,
            "batch_size": 1,
            "inference_precision": "fp32",
        },
        "selection": {
            "required_evaluation_split": "dev",
            "expected_manifest_sha256": str(
                lock["splits"]["dev"]["manifest_sha256"]
            ),
            "require_full_manifest": True,
            "low_snr": [0, 5],
            "ter_weight": 0.5,
            "der_weight": 0.5,
            "max_wer_absolute_increase": 0.05,
            "max_cer_absolute_increase": 0.03,
            "guard_split": "all",
            "guard_snr": "all",
            "allow_lambda_zero": False,
            "locked_control_strategy": "best_eligible_non_selected_tone_aware",
        },
    }
    selection = config["selection"]
    assert isinstance(selection, dict)
    selection["expected_evaluation_contract_sha256"] = (
        selection_evaluation_contract or evaluation_contract_sha256(config)
    )
    return config


def _write_complete_checkpoint(
    root: Path,
    *,
    name: str,
    config: dict[str, object],
    split_lock: Path,
) -> tuple[Path, dict[str, str]]:
    checkpoint = root / name
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "processor").mkdir()
    (checkpoint / "adapter" / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "name": name}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(
        f"adapter:{name}".encode()
    )
    (checkpoint / "processor" / "preprocessor_config.json").write_text(
        '{"sampling_rate":16000}\n', encoding="utf-8"
    )
    saved = deepcopy(config)
    lock = json.loads(split_lock.read_text(encoding="utf-8"))
    noise = config["noise"]
    assert isinstance(noise, dict)
    saved["runtime_protocol"] = {
        "split_lock_sha256": sha256_file(split_lock),
        "train_manifest_sha256": lock["splits"]["train"]["manifest_sha256"],
        "dev_manifest_sha256": lock["splits"]["dev"]["manifest_sha256"],
        "training_contract_sha256": training_contract_sha256(config),
        "training_scope": "formal",
        "audio_hashes_verified": True,
        "noise_enabled": True,
        "noise_manifest_sha256": sha256_file(
            Path(str(noise["noise_manifest"]))
        ),
        "noise_audio_paths_verified": True,
    }
    (checkpoint / "resolved_config.yaml").write_text(
        yaml.safe_dump(saved, sort_keys=True), encoding="utf-8"
    )
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step":100}\n', encoding="utf-8"
    )
    training = config["training"]
    assert isinstance(training, dict)
    if float(training["lambda_tone"]) > 0:
        (checkpoint / "tone_head.pt").write_bytes(b"tone-head")
    return checkpoint, verify_checkpoint_config(checkpoint, config)


def _locked_configuration(
    *,
    configuration_id: str,
    role: str,
    config: dict[str, object],
    checkpoint: Path,
    identity: dict[str, str],
) -> dict[str, object]:
    experiment = config["experiment"]
    model = config["model"]
    training = config["training"]
    assert isinstance(experiment, dict)
    assert isinstance(model, dict)
    assert isinstance(training, dict)
    return {
        "configuration_id": configuration_id,
        "role": role,
        "method_id": experiment["method_id"],
        "train_type": experiment["train_type"],
        "lambda": training["lambda_tone"],
        "seed": config["seed"],
        "backbone": model["name_or_path"],
        "backbone_revision": model["revision"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_path": str(checkpoint),
        "resolved_config_sha256": identity["resolved_config_sha256"],
        "training_contract_sha256": identity["training_contract_sha256"],
    }


def _valid_decision_fixture(
    root: Path,
    *,
    split_lock: Path,
    dev: Path,
    test: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, str],
]:
    ordinary_config = _formal_config(
        split_lock=split_lock,
        manifest=dev,
        data_split="dev",
        method_id="ordinary_lora",
        train_type="ordinary_lora",
        lambda_value=0.0,
    )
    dev_contract = evaluation_contract_sha256(ordinary_config)
    selected_config = _formal_config(
        split_lock=split_lock,
        manifest=dev,
        data_split="dev",
        method_id="corrected_decoder_tone_lora",
        train_type="tone_aware_lora",
        lambda_value=0.05,
        selection_evaluation_contract=dev_contract,
    )
    control_config = _formal_config(
        split_lock=split_lock,
        manifest=dev,
        data_split="dev",
        method_id="corrected_decoder_tone_lora",
        train_type="tone_aware_lora",
        lambda_value=0.1,
        selection_evaluation_contract=dev_contract,
    )
    ordinary_checkpoint, ordinary_identity = _write_complete_checkpoint(
        root,
        name="ordinary_best",
        config=ordinary_config,
        split_lock=split_lock,
    )
    selected_checkpoint, selected_identity = _write_complete_checkpoint(
        root,
        name="selected_best",
        config=selected_config,
        split_lock=split_lock,
    )
    control_checkpoint, control_identity = _write_complete_checkpoint(
        root,
        name="control_best",
        config=control_config,
        split_lock=split_lock,
    )
    test_config = _formal_config(
        split_lock=split_lock,
        manifest=test,
        data_split="test",
        method_id="corrected_decoder_tone_lora",
        train_type="tone_aware_lora",
        lambda_value=0.05,
        selection_evaluation_contract=dev_contract,
    )
    selection_results = root / "protocol" / "lambda_ablation_dev.csv"
    method_lock = root / "protocol" / "method_lock.json"
    method_lock.write_text('{"status":"LOCKED"}\n', encoding="utf-8")
    noisy_dev_lock = root / "protocol" / "noisy_dev_lock.json"
    noisy_dev_lock.write_text('{"status":"LOCKED"}\n', encoding="utf-8")
    method_identity = "9" * 64
    result_fields = [
        "lambda",
        "evaluation_split",
        "manifest_sha256",
        "metric_version",
        "evaluation_contract_sha256",
        "method_lock_sha256",
        "method_identity_sha256",
        "checkpoint_sha256",
        "training_contract_sha256",
    ]
    identities = {
        0.0: ordinary_identity,
        0.05: selected_identity,
        0.1: control_identity,
        0.3: {
            "checkpoint_sha256": "3" * 64,
            "training_contract_sha256": "4" * 64,
        },
        0.5: {
            "checkpoint_sha256": "5" * 64,
            "training_contract_sha256": "6" * 64,
        },
    }
    with selection_results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields, lineterminator="\n")
        writer.writeheader()
        for lambda_value in (0.0, 0.05, 0.1, 0.3, 0.5):
            writer.writerow(
                {
                    "lambda": lambda_value,
                    "evaluation_split": "dev",
                    "manifest_sha256": sha256_file(dev),
                    "metric_version": "aligned_v1",
                    "evaluation_contract_sha256": dev_contract,
                    "method_lock_sha256": sha256_file(method_lock),
                    "method_identity_sha256": method_identity,
                    "checkpoint_sha256": identities[lambda_value][
                        "checkpoint_sha256"
                    ],
                    "training_contract_sha256": identities[lambda_value][
                        "training_contract_sha256"
                    ],
                }
            )
    selection = ordinary_config["selection"]
    assert isinstance(selection, dict)
    decision = {
        "decision_version": DECISION_VERSION,
        "status": "LOCKED",
        "selection_complete": True,
        "test_unlocked": True,
        "identity_sha256": "",
        "split_lock": str(split_lock),
        "split_lock_sha256": sha256_file(split_lock),
        "method_lock": str(method_lock),
        "method_lock_sha256": sha256_file(method_lock),
        "method_identity_sha256": method_identity,
        "noisy_dev_lock": str(noisy_dev_lock),
        "noisy_dev_lock_sha256": sha256_file(noisy_dev_lock),
        "noisy_dev_manifest_sha256": sha256_file(dev),
        "selection_evaluation_split": "dev",
        "selection_manifest_sha256": sha256_file(dev),
        "selection_results": str(selection_results),
        "selection_results_sha256": sha256_file(selection_results),
        "selection_metric_version": "aligned_v1",
        "selection_rule": dict(selection),
        "selection_rule_sha256": selection_rule_sha256(selection),
        "selection_evaluation_contract_sha256": dev_contract,
        "allowed_test_evaluation_contract_sha256": [
            evaluation_contract_sha256(test_config)
        ],
        "selected_method_id": "corrected_decoder_tone_lora",
        "selected_lambda": 0.05,
        "locked_control_lambda": 0.1,
        "locked_control_strategy": "best_eligible_non_selected_tone_aware",
        "evaluated_lambdas": [0.0, 0.05, 0.1, 0.3, 0.5],
        "source_test_manifest": str(
            json.loads(split_lock.read_text(encoding="utf-8"))["splits"][
                "test_locked"
            ]["manifest"]
        ),
        "source_test_manifest_sha256": sha256_file(test),
        "source_test_utterance_count": 1,
        "final_benchmark_lock_status": "PENDING_AFTER_DECISION",
        "locked_configurations": [
            _locked_configuration(
                configuration_id="ordinary_seed_42",
                role="ordinary_baseline",
                config=ordinary_config,
                checkpoint=ordinary_checkpoint,
                identity=ordinary_identity,
            ),
            _locked_configuration(
                configuration_id="tone_selected_seed_42",
                role="selected_method",
                config=selected_config,
                checkpoint=selected_checkpoint,
                identity=selected_identity,
            ),
            _locked_configuration(
                configuration_id="tone_control_seed_42",
                role="locked_control",
                config=control_config,
                checkpoint=control_checkpoint,
                identity=control_identity,
            ),
        ],
    }
    decision_without_identity = dict(decision)
    decision_without_identity.pop("identity_sha256")
    decision["identity_sha256"] = canonical_sha256(decision_without_identity)
    return decision, test_config, selected_identity


class ProtocolIntegrityTests(unittest.TestCase):
    @staticmethod
    def _refresh_decision_identity(payload: dict[str, object]) -> None:
        content = dict(payload)
        content.pop("identity_sha256", None)
        payload["identity_sha256"] = canonical_sha256(content)

    def test_exposure_registry_is_hash_bound_without_opening_unseen_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, test, split_lock = _locked_fixture(root)
            lock = json.loads(split_lock.read_text(encoding="utf-8"))
            registry = Path(
                str(lock["official_test"]["exposure_evidence"]["registry"])
            )
            with patch(
                "src.vitonesr.phat.protocol._read_manifest_rows"
            ) as manifest_reader:
                loaded = load_split_lock(split_lock)
                manifest_reader.assert_not_called()
            self.assertEqual(
                loaded["splits"]["test_locked"]["manifest_sha256"],
                sha256_file(test),
            )
            registry.write_text(
                registry.read_text(encoding="utf-8") + "tampered\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ProtocolIntegrityError, "registry SHA-256"
            ):
                load_split_lock(split_lock)

    def test_locked_manifest_hash_and_audio_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, _, split_lock = _locked_fixture(root)
            integrity = verify_locked_vivos_manifest(
                dev,
                split_name="dev",
                split_lock_path=split_lock,
                verify_audio=True,
            )
            self.assertTrue(integrity["audio_hashes_verified"])
            first = json.loads(dev.read_text(encoding="utf-8").splitlines()[0])
            Path(first["audio"]).write_bytes(b"mutated")
            with self.assertRaisesRegex(ProtocolIntegrityError, "audio SHA-256"):
                verify_locked_vivos_manifest(
                    dev,
                    split_name="dev",
                    split_lock_path=split_lock,
                    verify_audio=True,
                )

    def test_test_access_requires_bound_decision_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            decision = root / "protocol" / "best_lambda_decision.json"
            with self.assertRaises(FileNotFoundError):
                verify_test_decision_lock(
                    split_lock_path=split_lock,
                    decision_lock_path=decision,
                )
            payload, _, _ = _valid_decision_fixture(
                root,
                split_lock=split_lock,
                dev=dev,
                test=test,
            )
            decision.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            verified = verify_test_decision_lock(
                split_lock_path=split_lock,
                decision_lock_path=decision,
            )
            self.assertEqual(
                verified["test_manifest_sha256"], sha256_file(test)
            )
            configurations = verified["locked_configurations"]
            self.assertEqual(len(configurations), 3)
            self.assertEqual(
                {item["role"] for item in configurations},
                {"ordinary_baseline", "selected_method", "locked_control"},
            )
            roles = resolve_locked_roles(verified)
            self.assertEqual(roles["selected_method"]["lambda"], 0.05)
            self.assertEqual(roles["locked_control"]["lambda"], 0.1)
            value = json.loads(decision.read_text(encoding="utf-8"))
            value["split_lock_sha256"] = "b" * 64
            decision.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolIntegrityError, "identity"):
                verify_test_decision_lock(
                    split_lock_path=split_lock,
                    decision_lock_path=decision,
                )

    def test_decision_requires_exactly_three_unique_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            payload, _, _ = _valid_decision_fixture(
                root, split_lock=split_lock, dev=dev, test=test
            )
            configurations = payload["locked_configurations"]
            assert isinstance(configurations, list)
            configurations[2]["role"] = "selected_method"
            self._refresh_decision_identity(payload)
            decision = root / "protocol" / "best_lambda_decision.json"
            decision.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolIntegrityError, "exactly one"):
                verify_test_decision_lock(
                    split_lock_path=split_lock,
                    decision_lock_path=decision,
                )

    def test_incomplete_selection_cannot_write_an_unlocked_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            payload, _, _ = _valid_decision_fixture(
                root, split_lock=split_lock, dev=dev, test=test
            )
            payload["selection_complete"] = False
            self._refresh_decision_identity(payload)
            with self.assertRaisesRegex(ValueError, "complete LOCKED"):
                write_decision_lock(root / "decision.json", payload)

    def test_decision_rejects_missing_method_hash_or_identity(self) -> None:
        for missing_field, expected_message in (
            ("method_lock_sha256", "method lock binding"),
            ("method_identity_sha256", "method identity"),
        ):
            with self.subTest(field=missing_field), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                dev, test, split_lock = _locked_fixture(root)
                payload, _, _ = _valid_decision_fixture(
                    root, split_lock=split_lock, dev=dev, test=test
                )
                payload.pop(missing_field)
                self._refresh_decision_identity(payload)
                decision = root / "protocol" / "best_lambda_decision.json"
                decision.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ProtocolIntegrityError, expected_message):
                    verify_test_decision_lock(
                        split_lock_path=split_lock,
                        decision_lock_path=decision,
                    )

    def test_decision_artifact_tamper_is_detected_without_reading_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            payload, _, _ = _valid_decision_fixture(
                root, split_lock=split_lock, dev=dev, test=test
            )
            decision = root / "protocol" / "best_lambda_decision.json"
            decision.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "src.vitonesr.phat.protocol._read_manifest_rows"
            ) as manifest_reader:
                verified = verify_test_decision_lock(
                    split_lock_path=split_lock,
                    decision_lock_path=decision,
                )
                manifest_reader.assert_not_called()
            self.assertEqual(verified["selected_lambda"], 0.05)
            method_lock = Path(str(payload["method_lock"]))
            method_lock.write_text('{"status":"TAMPERED"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolIntegrityError, "method lock SHA-256"):
                verify_test_decision_lock(
                    split_lock_path=split_lock,
                    decision_lock_path=decision,
                )

    def test_evaluator_rejects_sealed_test_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            dev_config = _formal_config(
                split_lock=split_lock,
                manifest=dev,
                data_split="dev",
                method_id="ordinary_lora",
                train_type="ordinary_lora",
                lambda_value=0.0,
            )
            config = _formal_config(
                split_lock=split_lock,
                manifest=test,
                data_split="test",
                method_id="ordinary_lora",
                train_type="ordinary_lora",
                lambda_value=0.0,
                selection_evaluation_contract=evaluation_contract_sha256(
                    dev_config
                ),
            )
            checkpoint, _ = _write_complete_checkpoint(
                root,
                name="checkpoint",
                config=config,
                split_lock=split_lock,
            )
            protocol = config["protocol"]
            assert isinstance(protocol, dict)
            protocol["final_test_unlocked"] = False
            with patch(
                "src.vitonesr.phat.evaluation.load_benchmark_rows"
            ) as manifest_loader, patch(
                "src.vitonesr.phat.evaluation.resolve_checkpoint"
            ) as checkpoint_resolver:
                with self.assertRaisesRegex(ValueError, "Final test is locked"):
                    run_checkpoint_evaluation(
                        config,
                        checkpoint=checkpoint,
                        device_arg="cpu",
                    )
                manifest_loader.assert_not_called()
                checkpoint_resolver.assert_not_called()
            protocol["final_test_unlocked"] = True
            with patch(
                "src.vitonesr.phat.evaluation.load_benchmark_rows"
            ) as manifest_loader, patch(
                "src.vitonesr.phat.evaluation.verify_method_lock",
                return_value={},
            ):
                with self.assertRaises(FileNotFoundError):
                    run_checkpoint_evaluation(
                        config,
                        checkpoint=checkpoint,
                        device_arg="cpu",
                    )
                manifest_loader.assert_not_called()

    def test_locked_manifest_override_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            config = _formal_config(
                split_lock=split_lock,
                manifest=dev,
                data_split="dev",
                method_id="ordinary_lora",
                train_type="ordinary_lora",
                lambda_value=0.0,
            )
            checkpoint, _ = _write_complete_checkpoint(
                root,
                name="dev_checkpoint",
                config=config,
                split_lock=split_lock,
            )
            with patch(
                "src.vitonesr.phat.evaluation.load_benchmark_rows"
            ) as manifest_loader, patch(
                "src.vitonesr.phat.evaluation.verify_method_lock",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    ValueError, "forbids manifest path overrides"
                ):
                    run_checkpoint_evaluation(
                        config,
                        checkpoint=checkpoint,
                        manifest=test,
                        device_arg="cpu",
                    )
                manifest_loader.assert_not_called()

    def test_non_test_split_cannot_masquerade_as_test_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            base_config = _formal_config(
                split_lock=split_lock,
                manifest=dev,
                data_split="dev",
                method_id="ordinary_lora",
                train_type="ordinary_lora",
                lambda_value=0.0,
            )
            checkpoint, _ = _write_complete_checkpoint(
                root,
                name="dev_checkpoint",
                config=base_config,
                split_lock=split_lock,
            )
            for declared_split in ("dev", "external"):
                with self.subTest(data_split=declared_split):
                    config = deepcopy(base_config)
                    evaluation = config["evaluation"]
                    assert isinstance(evaluation, dict)
                    evaluation["data_split"] = declared_split
                    evaluation["manifest"] = str(test)
                    evaluation["expected_manifest_sha256"] = sha256_file(test)
                    if declared_split == "external":
                        evaluation.pop("locked_vivos_split", None)
                    with patch(
                        "src.vitonesr.phat.evaluation.load_benchmark_rows"
                    ) as manifest_loader, patch(
                        "src.vitonesr.phat.evaluation.verify_method_lock",
                        return_value={},
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "sealed VIVOS test manifest"
                        ):
                            run_checkpoint_evaluation(
                                config,
                                checkpoint=checkpoint,
                                device_arg="cpu",
                            )
                        manifest_loader.assert_not_called()

    def test_decision_lock_rejects_unbound_test_configuration_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            payload, selected_config, selected_identity = _valid_decision_fixture(
                root,
                split_lock=split_lock,
                dev=dev,
                test=test,
            )
            decision = root / "protocol" / "best_lambda_decision.json"
            decision.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            integrity = verify_test_decision_lock(
                split_lock_path=split_lock,
                decision_lock_path=decision,
            )
            matched = verify_test_configuration_locked(
                integrity,
                config=selected_config,
                checkpoint_identity=selected_identity,
            )
            self.assertEqual(matched["configuration_id"], "tone_selected_seed_42")

            wrong_method = deepcopy(selected_config)
            experiment = wrong_method["experiment"]
            assert isinstance(experiment, dict)
            experiment["method_id"] = "ordinary_lora"
            wrong_lambda = deepcopy(selected_config)
            training = wrong_lambda["training"]
            assert isinstance(training, dict)
            training["lambda_tone"] = 0.1
            wrong_checkpoint = dict(selected_identity)
            wrong_checkpoint["checkpoint_sha256"] = "f" * 64
            cases = (
                ("method", wrong_method, selected_identity),
                ("lambda", wrong_lambda, selected_identity),
                ("checkpoint", selected_config, wrong_checkpoint),
            )
            for label, config, identity in cases:
                with self.subTest(identity=label):
                    with self.assertRaisesRegex(
                        ProtocolIntegrityError, "not uniquely bound"
                    ):
                        verify_test_configuration_locked(
                            integrity,
                            config=config,
                            checkpoint_identity=identity,
                        )

    def test_evaluator_rejects_unbound_checkpoint_before_test_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, test, split_lock = _locked_fixture(root)
            payload, test_config, _ = _valid_decision_fixture(
                root,
                split_lock=split_lock,
                dev=dev,
                test=test,
            )
            decision = root / "protocol" / "best_lambda_decision.json"
            decision.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            unbound_config = deepcopy(test_config)
            training = unbound_config["training"]
            assert isinstance(training, dict)
            training["lambda_tone"] = 0.1
            unbound_checkpoint, _ = _write_complete_checkpoint(
                root,
                name="unbound_test_checkpoint",
                config=unbound_config,
                split_lock=split_lock,
            )
            with patch(
                "src.vitonesr.phat.evaluation.load_benchmark_rows"
            ) as manifest_loader, patch(
                "src.vitonesr.phat.evaluation.verify_method_lock",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    ProtocolIntegrityError, "not uniquely bound"
                ):
                    run_checkpoint_evaluation(
                        unbound_config,
                        checkpoint=unbound_checkpoint,
                        device_arg="cpu",
                    )
                manifest_loader.assert_not_called()

    def test_checkpoint_fingerprint_binds_resolved_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dev, _, split_lock = _locked_fixture(root)
            config = _formal_config(
                split_lock=split_lock,
                manifest=dev,
                data_split="dev",
                method_id="corrected_decoder_tone_lora",
                train_type="tone_aware_lora",
                lambda_value=0.05,
            )
            checkpoint, identity = _write_complete_checkpoint(
                root,
                name="best",
                config=config,
                split_lock=split_lock,
            )
            self.assertEqual(
                identity["checkpoint_sha256"],
                checkpoint_inference_sha256(checkpoint),
            )
            training = config["training"]
            assert isinstance(training, dict)
            training["lambda_tone"] = 0.1
            with self.assertRaisesRegex(
                ProtocolIntegrityError, "identity mismatch"
            ):
                verify_checkpoint_config(checkpoint, config)


if __name__ == "__main__":
    unittest.main()
