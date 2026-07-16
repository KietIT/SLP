from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.vitonesr.analysis import CANONICAL_PREDICTION_COLUMNS
from src.vitonesr.prediction_evidence import (
    BenchmarkEvidence,
    DecisionEvidence,
    FINAL_LORA_VERSION,
    FLEURS_VERSION,
    PredictionEvidenceError,
    formal_protocol_parameters,
    sha256_file,
    verify_formal_error_events,
    verify_formal_prediction_set,
    verify_prediction_evidence,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


class PredictionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.split = self.root / "protocol" / "split.json"
        self.decision = self.root / "protocol" / "decision.json"
        self.manifest = self.root / "manifests" / "fleurs.jsonl"
        self.config = self.root / "configs" / "ordinary.yaml"
        self.registry = self.root / "protocol" / "fleurs_registry.json"
        self.prediction = self.root / "predictions" / "ordinary.csv"
        for path in (
            self.split,
            self.decision,
            self.manifest,
            self.config,
            self.registry,
            self.prediction,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.split.write_text("{}\n", encoding="utf-8")
        self.decision.write_text('{"status":"LOCKED"}\n', encoding="utf-8")
        self.config.write_text("seed: 42\n", encoding="utf-8")
        self.registry.write_text('{"registry_version":"fixture"}\n', encoding="utf-8")
        self.manifest.write_text(
            "".join(
                json.dumps(
                    {
                        "utt_id": f"fleurs-{index}",
                        "dataset": "fleurs",
                        "split": "test",
                        "snr": "clean",
                        "noise_type": "clean",
                        "ref": text,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for index, text in ((1, "má"), (2, "bà"))
            ),
            encoding="utf-8",
        )
        with self.prediction.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(CANONICAL_PREDICTION_COLUMNS),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "utt_id": f"fleurs-{index}",
                        "dataset": "fleurs",
                        "model": "phowhisper",
                        "model_size": "base",
                        "train_type": "ordinary_lora",
                        "lambda": "0",
                        "seed": "42",
                        "snr": "clean",
                        "noise_type": "clean",
                        "ref": text,
                        "hyp": text,
                    }
                    for index, text in ((1, "má"), (2, "bà"))
                ]
            )
        self._write_sidecar()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _configuration(self) -> dict[str, object]:
        return {
            "configuration_id": "ordinary_seed42",
            "role": "ordinary_baseline",
            "method_id": "ordinary_lora",
            "train_type": "ordinary_lora",
            "lambda": 0.0,
            "seed": 42,
            "checkpoint_sha256": HASH_A,
            "resolved_config_sha256": HASH_B,
            "training_contract_sha256": HASH_C,
            "checkpoint_path": "checkpoints/ordinary",
        }

    def _verifier(self, **_: object) -> dict[str, object]:
        return {
            "decision_lock_sha256": sha256_file(self.decision),
            "split_lock_sha256": sha256_file(self.split),
            "method_lock_sha256": HASH_D,
            "method_identity_sha256": HASH_C,
            "method_environment_identity_sha256": HASH_B,
            "method_source_tree_sha256": HASH_A,
            "method_runtime_verified": True,
            "locked_configurations": (self._configuration(),),
        }

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @property
    def sidecar(self) -> Path:
        return self.prediction.with_suffix(".csv.provenance.json")

    def _sidecar_object(self) -> dict[str, object]:
        value: dict[str, object] = {
            "provenance_version": FLEURS_VERSION,
            "evaluation_domain": "legacy_exposed_external_replication",
            "prediction_sha256": sha256_file(self.prediction),
            "num_rows": 2,
            "manifest_sha256": sha256_file(self.manifest),
            "metric_version": "aligned_v1",
            "split_lock_sha256": sha256_file(self.split),
            "decision_lock_sha256": sha256_file(self.decision),
            "method_lock_sha256": HASH_D,
            "method_identity_sha256": HASH_C,
            "method_environment_identity_sha256": HASH_B,
            "method_source_tree_sha256": HASH_A,
            "method_runtime_verified": True,
            "configuration_id": "ordinary_seed42",
            "role": "ordinary_baseline",
            "method_id": "ordinary_lora",
            "train_type": "ordinary_lora",
            "lambda": "0",
            "seed": "42",
            "checkpoint_sha256": HASH_A,
            "checkpoint": "checkpoints/ordinary",
            "resolved_config_sha256": HASH_B,
            "training_contract_sha256": HASH_C,
            "backbone": "vinai/PhoWhisper-base",
            "backbone_revision": HASH_A,
            "config": self._relative(self.config),
            "config_sha256": sha256_file(self.config),
            "registry": self._relative(self.registry),
            "registry_sha256": sha256_file(self.registry),
            "fleurs_preparation_lock_sha256": HASH_A,
            "fleurs_preparation_identity_sha256": HASH_B,
            "fleurs_dataset_revision": HASH_C,
            "fleurs_audio_inventory_sha256": HASH_D,
            "fleurs_audit_sha256": HASH_A,
            "decoding": {
                "language": "vi",
                "task": "transcribe",
                "implementation": "fixture",
            },
        }
        run_contract = {
            "contract_version": "paper_v2_fleurs_inference_contract_v1",
            "evaluation_domain": "legacy_exposed_external_replication",
            "registry_sha256": sha256_file(self.registry),
            "split_lock_sha256": sha256_file(self.split),
            "decision_lock_sha256": sha256_file(self.decision),
            "method_lock_sha256": HASH_D,
            "method_identity_sha256": HASH_C,
            "method_environment_identity_sha256": HASH_B,
            "method_source_tree_sha256": HASH_A,
            "method_runtime_verified": True,
            "manifest_sha256": sha256_file(self.manifest),
            "fleurs_preparation_lock_sha256": HASH_A,
            "fleurs_preparation_identity_sha256": HASH_B,
            "fleurs_dataset_revision": HASH_C,
            "fleurs_audio_inventory_sha256": HASH_D,
            "fleurs_audit_sha256": HASH_A,
            "selected_rows_sha256": HASH_B,
            "selected_rows": 2,
            "configuration_id": "ordinary_seed42",
            "role": "ordinary_baseline",
            "method_id": "ordinary_lora",
            "train_type": "ordinary_lora",
            "lambda": "0",
            "seed": "42",
            "checkpoint_sha256": HASH_A,
            "resolved_config_sha256": HASH_B,
            "training_contract_sha256": HASH_C,
            "config_sha256": sha256_file(self.config),
            "backbone": "vinai/PhoWhisper-base",
            "backbone_revision": HASH_A,
            "prediction_schema": list(CANONICAL_PREDICTION_COLUMNS),
            "decoding": {"language": "vi", "task": "transcribe"},
        }
        value["run_contract"] = run_contract
        value["run_contract_sha256"] = hashlib.sha256(
            json.dumps(
                run_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return value

    def _fleurs_authorizer(self, _: object) -> dict[str, object]:
        return {
            "formal": True,
            "registry_path": self.registry,
            "registry_sha256": sha256_file(self.registry),
            "manifest_path": self.manifest,
            "manifest_sha256": sha256_file(self.manifest),
            "expected_rows": 2,
            "split_lock_sha256": sha256_file(self.split),
            "decision_lock_sha256": sha256_file(self.decision),
            "method_lock_sha256": HASH_D,
            "method_identity_sha256": HASH_C,
            "method_environment_identity_sha256": HASH_B,
            "method_source_tree_sha256": HASH_A,
            "method_runtime_verified": False,
            "fleurs_preparation_lock_sha256": HASH_A,
            "fleurs_preparation_identity_sha256": HASH_B,
            "fleurs_dataset_revision": HASH_C,
            "fleurs_audio_inventory_sha256": HASH_D,
            "fleurs_audit_sha256": HASH_A,
            "registry": {
                "runs": [
                    {
                        "configuration_id": "ordinary_seed42",
                        "config_path": self._relative(self.config),
                        "config_sha256": sha256_file(self.config),
                        "checkpoint_path": "checkpoints/ordinary",
                    }
                ]
            },
        }

    def _write_sidecar(self, updates: dict[str, object] | None = None) -> None:
        value = self._sidecar_object()
        value.update(updates or {})
        self.sidecar.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _verify(self):
        return verify_formal_prediction_set(
            [self.prediction],
            benchmark_path=self.manifest,
            split_lock_path=self.split,
            decision_path=self.decision,
            required_configuration_ids={self.prediction: "ordinary_seed42"},
            root=self.root,
            decision_verifier=self._verifier,
            fleurs_authorizer=self._fleurs_authorizer,
            method_evidence_verifier=lambda *_args, **_kwargs: None,
        )

    def test_fleurs_prediction_is_bound_end_to_end(self) -> None:
        evidence = self._verify()
        item = evidence.predictions[0]
        self.assertEqual(item.configuration_id, "ordinary_seed42")
        self.assertEqual(item.role, "ordinary_baseline")
        self.assertEqual(item.prediction_sha256, sha256_file(self.prediction))
        self.assertEqual(evidence.benchmark.row_count, 2)

    def test_rejects_tampered_csv_sidecar_and_locked_role(self) -> None:
        original = self.prediction.read_bytes()
        self.prediction.write_bytes(original.replace("má".encode(), "ma".encode()))
        with self.assertRaisesRegex(PredictionEvidenceError, "Prediction SHA-256"):
            self._verify()
        self.prediction.write_bytes(original)

        self._write_sidecar({"config_sha256": HASH_A})
        with self.assertRaisesRegex(PredictionEvidenceError, "config/config_sha256"):
            self._verify()

        self._write_sidecar({"role": "selected_method"})
        with self.assertRaisesRegex(PredictionEvidenceError, "role differs"):
            self._verify()

    def test_rejects_fabricated_or_changed_decision(self) -> None:
        def wrong_verifier(**_: object) -> dict[str, object]:
            value = self._verifier()
            value["decision_lock_sha256"] = HASH_A
            return value

        with self.assertRaisesRegex(PredictionEvidenceError, "Decision changed"):
            verify_formal_prediction_set(
                [self.prediction],
                benchmark_path=self.manifest,
                split_lock_path=self.split,
                decision_path=self.decision,
                root=self.root,
                decision_verifier=wrong_verifier,
                fleurs_authorizer=self._fleurs_authorizer,
                method_evidence_verifier=lambda *_args, **_kwargs: None,
            )

    def test_error_events_recursively_reverify_predictions(self) -> None:
        evidence = self._verify()
        directory = self.root / "analysis"
        directory.mkdir()
        events = directory / "error_events.csv"
        summary = directory / "error_summary.csv"
        events.write_text("metric_version\naligned_v1\n", encoding="utf-8")
        summary.write_text("n\n1\n", encoding="utf-8")
        provenance = {
            "provenance_version": "analysis_artifact_provenance_v1",
            "bundle_name": "error_analysis",
            "bundle_version": "fixture",
            "inputs": [
                {
                    "path": self._relative(self.prediction),
                    "sha256": sha256_file(self.prediction),
                    "bytes": self.prediction.stat().st_size,
                }
            ],
            "parameters": {"formal_protocol": formal_protocol_parameters(evidence)},
            "data_outputs": [
                {
                    "key": "events",
                    "path": events.name,
                    "bytes": events.stat().st_size,
                    "sha256": sha256_file(events),
                },
                {
                    "key": "summary",
                    "path": summary.name,
                    "bytes": summary.stat().st_size,
                    "sha256": sha256_file(summary),
                },
            ],
        }
        provenance_path = directory / "error_analysis.provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        marker_identity = {
            "protocol_version": "analysis_artifact_bundle_v1",
            "bundle_name": "error_analysis",
            "bundle_version": "fixture",
            "inputs": provenance["inputs"],
            "outputs": [
                {
                    "key": key,
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for key, path in sorted(
                    {
                        "events": events,
                        "summary": summary,
                        "provenance": provenance_path,
                    }.items()
                )
            ],
        }
        marker = {
            **marker_identity,
            "bundle_sha256": hashlib.sha256(
                (
                    json.dumps(
                        marker_identity,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest(),
            "status": "COMMITTED",
        }
        (directory / "error_analysis.bundle.commit.json").write_text(
            json.dumps(marker, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        verified = verify_formal_error_events(
            events,
            benchmark_path=self.manifest,
            split_lock_path=self.split,
            decision_path=self.decision,
            root=self.root,
            decision_verifier=self._verifier,
            fleurs_authorizer=self._fleurs_authorizer,
            method_evidence_verifier=lambda *_args, **_kwargs: None,
        )
        self.assertEqual(verified.predictions[0].provenance_sha256, sha256_file(self.sidecar))

        events.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(PredictionEvidenceError, "committed output changed"):
            verify_formal_error_events(
                events,
                benchmark_path=self.manifest,
                split_lock_path=self.split,
                decision_path=self.decision,
                root=self.root,
                decision_verifier=self._verifier,
                fleurs_authorizer=self._fleurs_authorizer,
                method_evidence_verifier=lambda *_args, **_kwargs: None,
            )

    def test_zero_shot_final_lora_and_noisy_dev_sidecar_shapes(self) -> None:
        decision = DecisionEvidence(
            path=self.decision,
            sha256=sha256_file(self.decision),
            raw={"status": "LOCKED"},
            integrity=self._verifier(),
            configurations={"ordinary_seed42": self._configuration()},
        )
        benchmark = BenchmarkEvidence(
            path=self.manifest,
            sha256=sha256_file(self.manifest),
            row_count=2,
            final_benchmark_lock_path=self.root / "protocol" / "final_lock.json",
            final_benchmark_lock_sha256=HASH_B,
        )

        zero = self.root / "predictions" / "zero.csv"
        rows: list[dict[str, str]] = []
        for index, text in ((1, "má"), (2, "bà")):
            rows.append(
                {
                    "utt_id": f"fleurs-{index}",
                    "dataset": "fleurs",
                    "model": "whisper",
                    "model_size": "tiny",
                    "train_type": "zero_shot",
                    "lambda": "",
                    "seed": "42",
                    "snr": "clean",
                    "noise_type": "clean",
                    "ref": text,
                    "hyp": text,
                }
            )
        with zero.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(CANONICAL_PREDICTION_COLUMNS),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        zero_model = {
            "key": "whisper_tiny",
            "repo_id": "openai/whisper-tiny",
            "revision": HASH_A,
            "model": "whisper",
            "model_size": "tiny",
            "filename": "zero.csv",
        }
        zero_config = {
            "seed": 42,
            "protocol": {
                "formal": True,
                "final_test_unlocked": True,
                "expected_split_lock_sha256": sha256_file(self.split),
                "expected_decision_lock_sha256": decision.sha256,
            },
            "benchmark": {
                "expected_lock_sha256": HASH_B,
                "expected_manifest_sha256": benchmark.sha256,
                "expected_rows": 2,
                "dataset": "fleurs",
                "verify_audio_sha256": True,
            },
            "models": {"whisper_tiny": zero_model},
        }
        self.config.write_text(
            json.dumps(zero_config, sort_keys=True) + "\n", encoding="utf-8"
        )
        run_contract = {
            "contract_version": "paper_v2_zero_shot_run_v1",
            "suite_config_sha256": sha256_file(self.config),
            "schema": list(CANONICAL_PREDICTION_COLUMNS),
            "model": zero_model,
            "seed": 42,
            "protocol": zero_config["protocol"],
            "benchmark": zero_config["benchmark"],
        }
        run_contract_sha256 = hashlib.sha256(
            json.dumps(
                run_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        zero_sidecar = {
            "provenance_version": "paper_v2_zero_shot_prediction_v1",
            "prediction_sha256": sha256_file(zero),
            "num_rows": 2,
            "manifest_sha256": benchmark.sha256,
            "metric_version": "aligned_v1",
            "schema": list(CANONICAL_PREDICTION_COLUMNS),
            "split_lock_sha256": sha256_file(self.split),
            "decision_lock_sha256": decision.sha256,
            "benchmark_lock_sha256": HASH_B,
            "suite_config": self._relative(self.config),
            "suite_config_sha256": sha256_file(self.config),
            "run_contract": run_contract,
            "run_contract_sha256": run_contract_sha256,
            "model_key": "whisper_tiny",
            "model_repo_id": "openai/whisper-tiny",
            "model_revision": HASH_A,
            "seed": 42,
        }
        zero.with_suffix(".csv.provenance.json").write_text(
            json.dumps(zero_sidecar, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.assertEqual(
            verify_prediction_evidence(
                zero, decision=decision, benchmark=benchmark, root=self.root
            ).provenance_version,
            "paper_v2_zero_shot_prediction_v1",
        )

        final_dir = self.root / "final" / "ordinary_baseline"
        final_dir.mkdir(parents=True)
        final = final_dir / "predictions.csv"
        final.write_bytes(self.prediction.read_bytes())
        final_sidecar = {
            **self._sidecar_object(),
            "provenance_version": FINAL_LORA_VERSION,
            "prediction_sha256": sha256_file(final),
            "prediction_columns": list(CANONICAL_PREDICTION_COLUMNS),
            "final_manifest_sha256": benchmark.sha256,
            "final_benchmark_lock_sha256": HASH_B,
            "runtime_config": self._relative(self.config),
            "runtime_config_sha256": sha256_file(self.config),
        }
        final_sidecar.pop("manifest_sha256", None)
        final_sidecar.pop("config", None)
        final_sidecar.pop("config_sha256", None)
        final_sidecar.pop("registry", None)
        final_sidecar.pop("registry_sha256", None)
        (final_dir / "provenance.json").write_text(
            json.dumps(final_sidecar, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.assertEqual(
            verify_prediction_evidence(
                final,
                decision=decision,
                benchmark=benchmark,
                required_configuration_id="ordinary_seed42",
                root=self.root,
            ).role,
            "ordinary_baseline",
        )

        noisy = self.root / "predictions" / "noisy_dev.csv"
        noisy.write_bytes(self.prediction.read_bytes())
        noisy_sidecar = {
            "provenance_version": "prediction_evaluation_v4",
            "prediction_sha256": sha256_file(noisy),
            "num_rows": 2,
            "manifest_sha256": benchmark.sha256,
            "metric_version": "aligned_v1",
            "split_lock_sha256": sha256_file(self.split),
            "decision_lock_sha256": "",
            "method_lock_sha256": HASH_D,
            "method_identity_sha256": HASH_C,
            "training_scope": "formal",
            "config_path": self._relative(self.config),
            "config_file_sha256": sha256_file(self.config),
        }
        noisy.with_suffix(".csv.provenance.json").write_text(
            json.dumps(noisy_sidecar, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.assertEqual(
            verify_prediction_evidence(
                noisy, decision=decision, benchmark=benchmark, root=self.root
            ).provenance_version,
            "prediction_evaluation_v4",
        )


if __name__ == "__main__":
    unittest.main()
