from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.vitonesr.analysis import METRIC_VERSION, compute_aligned_metric_result
from src.vitonesr.phat.evaluation import (
    PREDICTION_COLUMNS,
    _result_row,
    aggregate_prediction_file,
    load_benchmark_rows,
    prediction_provenance_path,
    validate_prediction_schema,
)
from src.vitonesr.phat.protocol import (
    checkpoint_inference_sha256,
    evaluation_contract_payload,
    evaluation_contract_sha256,
    selected_rows_sha256,
    sha256_file,
    training_contract_sha256,
)
from src.vitonesr.phat.selection import select_best_lambda_from_rows
from src.vitonesr.prediction import atomic_write_csv


MODEL_REVISION = "7ebdb9e88f5cc5271fb88f4d642c82ff9388650e"


def _formal_config(
    *,
    lambda_value: float = 0.0,
    manifest: Path | None = None,
) -> dict[str, object]:
    train_type = "ordinary_lora" if lambda_value == 0 else "tone_aware_lora"
    method_id = (
        "ordinary_lora" if lambda_value == 0 else "corrected_decoder_tone_lora"
    )
    manifest_path = str(manifest or Path("dev.jsonl"))
    return {
        "seed": 42,
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
            "train_manifest": "train.jsonl",
            "valid_manifest": manifest_path,
            "sample_rate": 16000,
        },
        "noise": {"enable_train_noise": True, "snr_choices": [20, 10, 5, 0]},
        "evaluation": {
            "manifest": manifest_path,
            "data_split": "dev",
            "locked_vivos_split": "dev",
            "expected_manifest_sha256": "a" * 64,
            "expected_total_rows": 1,
            "sample_rate": 16000,
            "max_audio_seconds": 15.0,
            "max_new_tokens": 128,
            "batch_size": 1,
            "inference_precision": "fp32",
        },
    }


def _write_complete_checkpoint(root: Path, config: dict[str, object]) -> Path:
    checkpoint = root / "checkpoint"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "processor").mkdir()
    (checkpoint / "adapter" / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}\n', encoding="utf-8"
    )
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(
        b"adapter-weights"
    )
    (checkpoint / "processor" / "preprocessor_config.json").write_text(
        '{"sampling_rate":16000}\n', encoding="utf-8"
    )
    (checkpoint / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step":100}\n', encoding="utf-8"
    )
    training = config["training"]
    assert isinstance(training, dict)
    if float(training["lambda_tone"]) > 0:
        (checkpoint / "tone_head.pt").write_bytes(b"tone-head")
    return checkpoint


class EvaluationAndSelectionTests(unittest.TestCase):
    def test_ablation_metrics_use_aligned_v1_counts(self) -> None:
        rows = [
            {
                "model": "phowhisper",
                "model_size": "base",
                "train_type": "ordinary_lora",
                "lambda": "0",
                "seed": "42",
                "ref": "má mà",
                "hyp": "x má mà",
            }
        ]
        result = _result_row(
            rows,
            split="all",
            snr="all",
            noise_type="all",
            checkpoint_path="checkpoint",
            prediction_path="prediction",
        )
        expected = compute_aligned_metric_result(["má mà"], ["x má mà"])
        self.assertEqual(result["metric_version"], METRIC_VERSION)
        self.assertEqual(result["ter_numerator"], expected.tone_errors)
        self.assertEqual(result["ter_denominator"], expected.tone_reference_units)
        self.assertEqual(result["ter_coverage"], expected.ter_coverage)
        self.assertEqual(result["der_numerator"], expected.diacritic_errors)
        self.assertEqual(result["der_coverage"], expected.der_coverage)
        self.assertEqual(result["fcer_coverage"], expected.fcer_coverage)

    def test_prediction_csv_has_exact_schema(self) -> None:
        row = {
            "utt_id": "utt-1",
            "dataset": "vivos",
            "model": "phowhisper",
            "model_size": "base",
            "train_type": "ordinary_lora",
            "lambda": "0",
            "seed": "42",
            "snr": "clean",
            "noise_type": "clean",
            "ref": "xin chào",
            "hyp": "xin chào",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.csv"
            atomic_write_csv(path, [row], PREDICTION_COLUMNS)
            validated = validate_prediction_schema(path)
            self.assertEqual(validated[0]["utt_id"], "utt-1")
            with path.open("r", encoding="utf-8", newline="") as prediction_file:
                self.assertEqual(list(csv.DictReader(prediction_file).fieldnames or []), PREDICTION_COLUMNS)

    def test_prediction_aggregation_requires_hash_bound_split_provenance(self) -> None:
        row = {
            "utt_id": "utt-1",
            "dataset": "vivos",
            "model": "phowhisper",
            "model_size": "base",
            "train_type": "ordinary_lora",
            "lambda": "0",
            "seed": "42",
            "snr": "clean",
            "noise_type": "clean",
            "ref": "xin chào",
            "hyp": "xin chào",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "audio": str(audio),
                        "text": row["ref"],
                        "utt_id": row["utt_id"],
                        "dataset": row["dataset"],
                        "split": "dev",
                        "snr": "clean",
                        "noise_type": "clean",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            config = _formal_config(manifest=manifest)
            checkpoint = _write_complete_checkpoint(root, config)
            path = root / "predictions.csv"
            atomic_write_csv(path, [row], PREDICTION_COLUMNS)
            prediction_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            selected = load_benchmark_rows(manifest)
            evaluation_contract = evaluation_contract_payload(config)
            prediction_provenance_path(path).write_text(
                json.dumps(
                    {
                        "provenance_version": "prediction_evaluation_v3",
                        "evaluation_split": "dev",
                        "manifest": str(manifest),
                        "manifest_sha256": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                        "manifest_num_rows": 1,
                        "audio_hashes_verified": True,
                        "evaluation_scope": "full_manifest",
                        "selected_rows_sha256": selected_rows_sha256(selected),
                        "filters": {
                            "subset": "all",
                            "snrs": [],
                            "noise_types": [],
                            "limit": None,
                        },
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_inference_sha256(
                            checkpoint
                        ),
                        "resolved_config_sha256": sha256_file(
                            checkpoint / "resolved_config.yaml"
                        ),
                        "training_scope": "formal",
                        "training_contract_sha256": training_contract_sha256(
                            config
                        ),
                        "evaluation_contract": evaluation_contract,
                        "evaluation_contract_sha256": evaluation_contract_sha256(
                            config
                        ),
                        "runtime_environment": {
                            "device_type": "cpu",
                            "dtype": "torch.float32",
                            "batch_size": 1,
                            "torch_version": "test",
                            "transformers_version": "test",
                        },
                        "prediction_sha256": prediction_hash,
                        "num_rows": 1,
                        "metric_version": METRIC_VERSION,
                        "method_lock_sha256": "1" * 64,
                        "method_identity_sha256": "2" * 64,
                        "environment_artifact_sha256": "3" * 64,
                        "environment_identity_sha256": "4" * 64,
                        "source_tree_sha256": "5" * 64,
                        "split_lock_sha256": "d" * 64,
                        "decision_lock_sha256": "",
                    }
                ),
                encoding="utf-8",
            )
            aggregated = aggregate_prediction_file(
                path, checkpoint_path=checkpoint
            )
            self.assertTrue(aggregated)
            self.assertTrue(all(item["evaluation_split"] == "dev" for item in aggregated))
            self.assertTrue(
                all(item["metric_version"] == METRIC_VERSION for item in aggregated)
            )
            adapter = checkpoint / "adapter" / "adapter_model.safetensors"
            adapter.write_bytes(b"mutated-adapter-weights")
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                aggregate_prediction_file(path, checkpoint_path=checkpoint)
            adapter.write_bytes(b"adapter-weights")
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                aggregate_prediction_file(path, checkpoint_path=checkpoint)

    def test_evaluation_manifest_requires_one_explicit_data_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio_a = root / "a.wav"
            audio_b = root / "b.wav"
            audio_a.write_bytes(b"a")
            audio_b.write_bytes(b"b")
            manifest = root / "manifest.jsonl"
            rows = [
                {
                    "audio": str(audio_a),
                    "text": "xin chào",
                    "utt_id": "a",
                    "dataset": "vivos",
                    "split": "dev",
                }
            ]
            manifest.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
                encoding="utf-8",
            )
            loaded = load_benchmark_rows(manifest)
            self.assertEqual(loaded[0]["evaluation_split"], "dev")
            rows.append(
                {
                    "audio": str(audio_b),
                    "text": "tạm biệt",
                    "utt_id": "b",
                    "dataset": "vivos",
                    "split": "test",
                }
            )
            manifest.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_benchmark_rows(manifest)

    def test_best_lambda_prioritizes_low_snr_with_guards(self) -> None:
        rows: list[dict[str, str]] = []
        evaluation_contract_hash = evaluation_contract_sha256(_formal_config())
        values = {
            0.0: (0.30, 0.18, 0.24, 0.20),
            0.05: (0.31, 0.18, 0.20, 0.17),
            0.1: (0.32, 0.19, 0.18, 0.15),
            0.3: (0.38, 0.25, 0.16, 0.14),
            0.5: (0.34, 0.20, 0.19, 0.16),
        }
        for lambda_value, (wer, cer, ter, der) in values.items():
            train_type = "ordinary_lora" if lambda_value == 0 else "tone_aware_lora"
            training_contract_hash = training_contract_sha256(
                _formal_config(lambda_value=lambda_value)
            )
            common = {
                "model": "phowhisper",
                "model_size": "base",
                "train_type": train_type,
                "lambda": str(lambda_value),
                "seed": "42",
                "evaluation_split": "dev",
                "manifest_sha256": "a" * 64,
                "evaluation_scope": "full_manifest",
                "selected_rows_sha256": "b" * 64,
                "training_scope": "formal",
                "training_contract_sha256": training_contract_hash,
                "evaluation_contract_sha256": evaluation_contract_hash,
                "method_lock_sha256": "1" * 64,
                "method_identity_sha256": "2" * 64,
                "environment_artifact_sha256": "3" * 64,
                "environment_identity_sha256": "4" * 64,
                "source_tree_sha256": "5" * 64,
                "noise_type": "all",
                "num_samples": "100",
                "metric_version": METRIC_VERSION,
                "wer": str(wer),
                "wer_numerator": str(round(wer * 1000)),
                "wer_denominator": "1000",
                "cer": str(cer),
                "cer_numerator": str(round(cer * 1000)),
                "cer_denominator": "1000",
                "ter": str(ter),
                "ter_numerator": str(round(ter * 1000)),
                "ter_denominator": "1000",
                "ter_coverage": "1.0",
                "der": str(der),
                "der_numerator": str(round(der * 1000)),
                "der_denominator": "1000",
                "der_coverage": "1.0",
                "fcer": "0.1",
                "fcer_numerator": "100",
                "fcer_denominator": "1000",
                "fcer_coverage": "1.0",
                "swdr": "0.1",
                "swdr_numerator": "100",
                "swdr_denominator": "1000",
                "checkpoint_path": "checkpoint",
                "checkpoint_sha256": hashlib.sha256(
                    f"checkpoint:{lambda_value:g}".encode()
                ).hexdigest(),
                "prediction_path": "prediction",
            }
            rows.append({**common, "split": "all", "snr": "all"})
            rows.append(
                {
                    **common,
                    "split": "clean",
                    "snr": "clean",
                    "noise_type": "clean",
                    "num_samples": "50",
                }
            )
            rows.append({**common, "split": "noisy", "snr": "0", "num_samples": "50"})
            rows.append({**common, "split": "noisy", "snr": "5", "num_samples": "50"})
        for row in rows:
            if row["lambda"] == "0.05" and row["snr"] in {"0", "5"}:
                row["ter_denominator"] = "500"
                row["ter_numerator"] = str(round(float(row["ter"]) * 500))
                row["ter_coverage"] = "0.5"
                row["der_denominator"] = "500"
                row["der_numerator"] = str(round(float(row["der"]) * 500))
                row["der_coverage"] = "0.5"
                row["fcer_denominator"] = "500"
                row["fcer_numerator"] = str(round(float(row["fcer"]) * 500))
                row["fcer_coverage"] = "0.5"
        result = select_best_lambda_from_rows(
            rows,
            {
                "low_snr": [0, 5],
                "ter_weight": 0.5,
                "der_weight": 0.5,
                "max_wer_absolute_increase": 0.05,
                "max_cer_absolute_increase": 0.03,
                "guard_split": "all",
                "guard_snr": "all",
                "allow_lambda_zero": False,
                "locked_control_strategy": "best_eligible_non_selected_tone_aware",
                "expected_manifest_sha256": "a" * 64,
                "expected_evaluation_contract_sha256": evaluation_contract_hash,
                "require_full_manifest": True,
            },
        )
        self.assertEqual(result.selected_lambda, 0.1)
        self.assertEqual(result.locked_control_lambda, 0.5)
        lambda_03 = next(summary for summary in result.summaries if summary.lambda_value == 0.3)
        self.assertFalse(lambda_03.eligible)
        lambda_005 = next(summary for summary in result.summaries if summary.lambda_value == 0.05)
        self.assertFalse(lambda_005.eligible)
        self.assertIn("coverage", lambda_005.reason)
        self.assertEqual(lambda_005.low_snr_fcer_coverage_ratio, 0.5)

        tampered = [dict(row) for row in rows]
        tampered[0]["fcer_coverage"] = "0.5"
        with self.assertRaisesRegex(ValueError, "coverage does not match"):
            select_best_lambda_from_rows(
                tampered,
                {
                    "low_snr": [0, 5],
                    "guard_split": "all",
                    "guard_snr": "all",
                    "allow_lambda_zero": False,
                    "locked_control_strategy": "best_eligible_non_selected_tone_aware",
                    "expected_manifest_sha256": "a" * 64,
                    "expected_evaluation_contract_sha256": evaluation_contract_hash,
                    "require_full_manifest": True,
                },
            )

    def test_best_lambda_rejects_test_missing_and_mixed_provenance(self) -> None:
        base = {
            "lambda": "0",
            "split": "all",
            "manifest_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "evaluation_split=dev"):
            select_best_lambda_from_rows(
                [{**base, "evaluation_split": "test"}],
                {"required_evaluation_split": "dev"},
            )
        with self.assertRaisesRegex(ValueError, "evaluation_split=dev"):
            select_best_lambda_from_rows(
                [base],
                {"required_evaluation_split": "dev"},
            )
        with self.assertRaisesRegex(ValueError, "evaluation_split=dev"):
            select_best_lambda_from_rows(
                [
                    {**base, "evaluation_split": "dev"},
                    {**base, "evaluation_split": "test"},
                ],
                {"required_evaluation_split": "dev"},
            )
        with self.assertRaisesRegex(ValueError, "split is test"):
            select_best_lambda_from_rows(
                [{**base, "split": "test", "evaluation_split": "dev"}],
                {"required_evaluation_split": "dev"},
            )

    def test_best_lambda_rejects_missing_or_mixed_manifest_hash(self) -> None:
        base = {
            "lambda": "0",
            "split": "all",
            "evaluation_split": "dev",
        }
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            select_best_lambda_from_rows(
                [base],
                {"required_evaluation_split": "dev"},
            )

    def test_best_lambda_rejects_partial_or_legacy_metric_scope(self) -> None:
        base = {
            "lambda": "0",
            "split": "all",
            "evaluation_split": "dev",
            "manifest_sha256": "a" * 64,
            "selected_rows_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
            "metric_version": METRIC_VERSION,
            "evaluation_scope": "partial",
        }
        selection = {
            "required_evaluation_split": "dev",
            "expected_manifest_sha256": "a" * 64,
            "require_full_manifest": True,
        }
        with self.assertRaisesRegex(ValueError, "partial"):
            select_best_lambda_from_rows([base], selection)
        with self.assertRaisesRegex(ValueError, "metric_version"):
            select_best_lambda_from_rows(
                [
                    {
                        **base,
                        "evaluation_scope": "full_manifest",
                        "metric_version": "legacy",
                    }
                ],
                selection,
            )
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            select_best_lambda_from_rows(
                [
                    {**base, "manifest_sha256": "a" * 64},
                    {**base, "manifest_sha256": "b" * 64},
                ],
                {"required_evaluation_split": "dev"},
            )


if __name__ == "__main__":
    unittest.main()
