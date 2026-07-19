from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

from scripts.run_final_lora_inference_runtime import (
    FinalLoraInferenceRuntimeError,
    RECEIPT_VERSION,
    _compatible_final_predictor,
    _load_processor_with_fallback,
    _retry_atomic_write_bytes,
    _scoped_final_atomic_write_retry,
    build_parser,
    run_final_lora_inference_runtime,
    verify_final_lora_execution_receipt,
)
from src.vitonesr.phat import final_evaluation as locked_final_evaluation
from src.vitonesr.inference_runtime import REQUIRED_INFERENCE_SOURCE_PATHS
from src.vitonesr.phat.final_evaluation import ROLE_ORDER, SUITE_VERSION
from src.vitonesr.phat.protocol import sha256_file


ROOT = Path(__file__).resolve().parents[1]
HASHES = {letter: letter * 64 for letter in "abcdef"}


def _repo_ref(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


class FinalLoraInferenceRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        self.training_environment = self.root / "training_environment.json"
        self.training_environment.write_text("{}\n", encoding="utf-8")
        self.method_lock = self.root / "method_lock.json"
        self.method_lock.write_text(
            json.dumps(
                {
                    "artifacts": {
                        "environment": {"path": _repo_ref(self.training_environment)}
                    }
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.inference_lock = self.root / "inference_runtime_lock.json"
        self.inference_lock.write_text("{}\n", encoding="utf-8")
        self.output_root = self.root / "final_predictions"
        self.config_path = self.root / "final_lora_runtime.yaml"
        self.receipt = self.root / "execution_receipt.json"
        config = {
            "suite_version": SUITE_VERSION,
            "protocol": {
                "formal": True,
                "final_test_unlocked": True,
                "split_lock": _repo_ref(self.root / "split_lock.json"),
                "expected_split_lock_sha256": HASHES["a"],
                "decision_lock": _repo_ref(self.root / "decision.json"),
                "expected_decision_lock_sha256": HASHES["b"],
                "method_lock": _repo_ref(self.method_lock),
                "expected_method_lock_sha256": sha256_file(self.method_lock),
                "method_config": "configs/phat/lambda_0.yaml",
                "noise_split_lock": _repo_ref(self.root / "noise_lock.json"),
                "expected_noise_split_lock_sha256": HASHES["c"],
                "final_benchmark_lock": _repo_ref(
                    self.root / "final_benchmark_lock.json"
                ),
                "expected_final_benchmark_lock_sha256": HASHES["d"],
            },
            "benchmark": {
                "manifest": _repo_ref(self.root / "manifest.jsonl"),
                "expected_manifest_sha256": HASHES["e"],
                "expected_rows": 2300,
                "verify_audio_sha256": True,
            },
            "runtime": {"device": "cuda", "verify_method_audio_sha256": False},
            "output": {
                "directory": _repo_ref(self.output_root),
                "aggregate_filename": "final_lora_results.csv",
            },
        }
        self.config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        self.runtime_calls: list[tuple[Any, ...]] = []
        self.method_calls: list[dict[str, Any]] = []
        self.suite_calls = 0
        self.predictors: list[Any] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def runtime_verifier(
        self,
        lock_path: Path,
        training_environment_path: Path,
        repo_root: Path,
        *,
        verify_current: bool,
    ) -> Mapping[str, Any]:
        self.runtime_calls.append(
            (lock_path, training_environment_path, repo_root, verify_current)
        )
        return {
            "status": "VERIFIED",
            "schema_version": "paper_v2_inference_runtime_v1",
            "lock_sha256": sha256_file(self.inference_lock),
            "identity_sha256": HASHES["a"],
            "training_environment_sha256": sha256_file(
                self.training_environment
            ),
            "training_environment_identity_sha256": HASHES["b"],
            "inference_environment_identity_sha256": HASHES["c"],
            "source_component_paths": list(REQUIRED_INFERENCE_SOURCE_PATHS),
            "source_tree_sha256": HASHES["f"],
        }

    def method_verifier(self, *args: Any, **kwargs: Any) -> Mapping[str, str]:
        self.method_calls.append(dict(kwargs))
        return {
            "method_lock_sha256": sha256_file(self.method_lock),
            "method_identity_sha256": HASHES["d"],
            "environment_artifact_sha256": sha256_file(
                self.training_environment
            ),
            "environment_identity_sha256": HASHES["b"],
            "source_tree_sha256": HASHES["e"],
        }

    def suite_runner(
        self,
        config: Mapping[str, Any],
        *,
        resume: bool,
        predictor: Any,
        authorization_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.suite_calls += 1
        self.predictors.append(predictor)
        authorization_kwargs["method_verifier"](
            self.method_lock,
            config={},
            repo_root=ROOT,
            formal=True,
            verify_audio=False,
        )
        prediction_hashes: dict[str, str] = {}
        for role in ROLE_ORDER:
            role_root = self.output_root / role
            role_root.mkdir(parents=True, exist_ok=True)
            prediction = role_root / "predictions.csv"
            if not prediction.exists():
                prediction.write_bytes(
                    (
                        "role,hyp\n"
                        + "".join(
                            f"{role},xin chao {index}\n" for index in range(2300)
                        )
                    ).encode("utf-8")
                )
            prediction_hashes[role] = sha256_file(prediction)
            provenance = role_root / "provenance.json"
            if not provenance.exists():
                provenance.write_text(
                    json.dumps(
                        {"prediction_sha256": prediction_hashes[role]}, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
        aggregate_root = self.output_root / "aggregate"
        aggregate_root.mkdir(parents=True, exist_ok=True)
        aggregate = aggregate_root / "final_lora_results.csv"
        if not aggregate.exists():
            aggregate.write_bytes(
                (
                    "role,wer\n"
                    + "".join(f"group-{index},1.0\n" for index in range(36))
                ).encode("utf-8")
            )
        aggregate_provenance = aggregate_root / "provenance.json"
        if not aggregate_provenance.exists():
            aggregate_provenance.write_text(
                json.dumps(
                    {
                        "result_sha256": sha256_file(aggregate),
                        "prediction_sha256_by_role": prediction_hashes,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "roles": list(ROLE_ORDER),
            "prediction_rows_per_role": 2300,
            "aggregate_rows": 36,
            "output_directory": _repo_ref(self.output_root),
            "aggregate": _repo_ref(aggregate),
        }

    def execute(self, *, resume: bool = False) -> dict[str, Any]:
        return run_final_lora_inference_runtime(
            config_path=_repo_ref(self.config_path),
            inference_runtime_lock_path=_repo_ref(self.inference_lock),
            receipt_path=_repo_ref(self.receipt),
            resume=resume,
            runtime_verifier=self.runtime_verifier,
            method_verifier=self.method_verifier,
            suite_runner=self.suite_runner,
        )

    def verify_receipt(self, *, verify_current: bool = True) -> dict[str, Any]:
        return verify_final_lora_execution_receipt(
            _repo_ref(self.receipt),
            _repo_ref(self.config_path),
            _repo_ref(self.inference_lock),
            verify_current=verify_current,
            runtime_verifier=self.runtime_verifier,
            method_verifier=self.method_verifier,
        )

    def test_runs_on_locked_inference_host_and_writes_bound_receipt(self) -> None:
        result = self.execute()
        self.assertEqual(result["execution_receipt_status"], "written")
        self.assertEqual(len(self.runtime_calls), 1)
        lock_path, training_path, repo_root, verify_current = self.runtime_calls[0]
        self.assertEqual(lock_path, self.inference_lock.resolve())
        self.assertEqual(training_path, self.training_environment.resolve())
        self.assertEqual(repo_root, ROOT)
        self.assertTrue(verify_current)
        self.assertEqual(self.method_calls[0]["formal"], False)
        self.assertIs(self.predictors[0], _compatible_final_predictor)

        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(receipt["receipt_version"], RECEIPT_VERSION)
        self.assertIs(receipt["inference_runtime_verified"], True)
        self.assertIs(receipt["training_runtime_verified_as_current"], False)
        self.assertEqual(
            receipt["inference_runtime"]["training_environment_identity_sha256"],
            HASHES["b"],
        )
        self.assertEqual(
            receipt["training_method"]["environment_identity_sha256"],
            HASHES["b"],
        )
        self.assertEqual(set(receipt["predictions"]), set(ROLE_ORDER))
        for role in ROLE_ORDER:
            binding = receipt["predictions"][role]
            self.assertEqual(
                binding["prediction_sha256"],
                sha256_file(self.output_root / role / "predictions.csv"),
            )
            self.assertEqual(
                binding["provenance_sha256"],
                sha256_file(self.output_root / role / "provenance.json"),
            )
        self.assertEqual(
            receipt["aggregate"]["result_sha256"],
            sha256_file(self.output_root / "aggregate/final_lora_results.csv"),
        )

    def test_resume_verifies_exact_receipt_and_outputs(self) -> None:
        first = self.execute()
        second = self.execute(resume=True)
        self.assertEqual(first["execution_receipt_sha256"], second["execution_receipt_sha256"])
        self.assertEqual(second["execution_receipt_status"], "verified_existing")
        self.assertEqual(self.suite_calls, 1)

    def test_independent_verifier_checks_complete_transitive_evidence(self) -> None:
        self.execute()
        verified = self.verify_receipt(verify_current=False)
        self.assertEqual(verified["status"], "VERIFIED")
        self.assertTrue(verified["inference_runtime_verified"])
        self.assertFalse(verified["training_runtime_verified_as_current"])
        self.assertEqual(verified["prediction_rows_per_role"], 2300)
        self.assertEqual(verified["aggregate_rows"], 36)
        self.assertFalse(self.runtime_calls[-1][-1])

    def test_existing_receipt_requires_resume_before_any_verifier(self) -> None:
        self.execute()
        calls = len(self.runtime_calls)
        with self.assertRaisesRegex(FileExistsError, "--resume"):
            self.execute()
        self.assertEqual(len(self.runtime_calls), calls)

    def test_resume_rejects_tampered_receipt(self) -> None:
        self.execute()
        self.receipt.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(
            FinalLoraInferenceRuntimeError, "canonical JSON|transitive evidence"
        ):
            self.execute(resume=True)

    def test_verify_only_rejects_deleted_receipt(self) -> None:
        self.execute()
        self.receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify_receipt()

    def test_resume_rejects_deleted_prediction_without_rerunning_suite(self) -> None:
        self.execute()
        (self.output_root / "selected_method/predictions.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute(resume=True)
        self.assertEqual(self.suite_calls, 1)

    def test_verify_only_rejects_tampered_prediction(self) -> None:
        self.execute()
        prediction = self.output_root / "locked_control/predictions.csv"
        prediction.write_bytes(prediction.read_bytes() + b"locked_control,tampered\n")
        with self.assertRaisesRegex(
            FinalLoraInferenceRuntimeError, "row counts differ"
        ):
            self.verify_receipt()

    def test_rejects_incomplete_runtime_source_profile(self) -> None:
        def incomplete_runtime(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            value = dict(self.runtime_verifier(*args, **kwargs))
            value["source_component_paths"] = list(REQUIRED_INFERENCE_SOURCE_PATHS[:-1])
            return value

        with self.assertRaisesRegex(
            FinalLoraInferenceRuntimeError, "incomplete source profile"
        ):
            run_final_lora_inference_runtime(
                config_path=_repo_ref(self.config_path),
                inference_runtime_lock_path=_repo_ref(self.inference_lock),
                receipt_path=_repo_ref(self.receipt),
                runtime_verifier=incomplete_runtime,
                method_verifier=self.method_verifier,
                suite_runner=self.suite_runner,
            )
        self.assertEqual(self.suite_calls, 0)

    def test_parser_exposes_verify_only(self) -> None:
        args = build_parser().parse_args(["--verify-only"])
        self.assertTrue(args.verify_only)

    def test_rejects_training_environment_identity_mismatch(self) -> None:
        def mismatched_method(*args: Any, **kwargs: Any) -> Mapping[str, str]:
            value = dict(self.method_verifier(*args, **kwargs))
            value["environment_identity_sha256"] = HASHES["f"]
            return value

        with self.assertRaisesRegex(
            FinalLoraInferenceRuntimeError, "different training environments"
        ):
            run_final_lora_inference_runtime(
                config_path=_repo_ref(self.config_path),
                inference_runtime_lock_path=_repo_ref(self.inference_lock),
                receipt_path=_repo_ref(self.receipt),
                runtime_verifier=self.runtime_verifier,
                method_verifier=mismatched_method,
                suite_runner=self.suite_runner,
            )
        self.assertFalse(self.receipt.exists())

    def test_rejects_aggregate_provenance_mismatch(self) -> None:
        def corrupting_suite(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            result = dict(self.suite_runner(*args, **kwargs))
            provenance = self.output_root / "aggregate/provenance.json"
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            payload["result_sha256"] = HASHES["f"]
            provenance.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            return result

        with self.assertRaisesRegex(
            FinalLoraInferenceRuntimeError, "Aggregate provenance"
        ):
            run_final_lora_inference_runtime(
                config_path=_repo_ref(self.config_path),
                inference_runtime_lock_path=_repo_ref(self.inference_lock),
                receipt_path=_repo_ref(self.receipt),
                runtime_verifier=self.runtime_verifier,
                method_verifier=self.method_verifier,
                suite_runner=corrupting_suite,
            )
        self.assertFalse(self.receipt.exists())

    def test_processor_falls_back_to_pinned_base_revision(self) -> None:
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        processor = object()

        class FakeProcessor:
            @classmethod
            def from_pretrained(
                cls, source: str, *args: Any, **kwargs: Any
            ) -> Any:
                calls.append((source, args, dict(kwargs)))
                if source.endswith("processor"):
                    raise AttributeError("'list' object has no attribute 'keys'")
                return processor

        role = SimpleNamespace(
            role="selected_method",
            checkpoint_path=self.root / "checkpoint",
            config={
                "model": {
                    "name_or_path": "vinai/PhoWhisper-base",
                    "revision": HASHES["a"],
                }
            },
        )
        with self.assertWarnsRegex(RuntimeWarning, "falling back"):
            loaded = _load_processor_with_fallback(
                FakeProcessor,
                role,
                language="vi",
                task="transcribe",
            )

        self.assertIs(loaded, processor)
        self.assertEqual(calls[0][0], str(role.checkpoint_path / "processor"))
        self.assertEqual(calls[1][0], "vinai/PhoWhisper-base")
        self.assertEqual(calls[1][2]["revision"], HASHES["a"])
        self.assertEqual(calls[1][2]["language"], "vi")
        self.assertEqual(calls[1][2]["task"], "transcribe")

    def test_atomic_publish_retries_transient_permission_error(self) -> None:
        attempts = 0
        now = 0.0
        delays: list[float] = []
        target = self.root / "resume.json"

        def writer(path: Path, payload: bytes) -> None:
            nonlocal attempts
            attempts += 1
            self.assertEqual(path, target)
            self.assertEqual(payload, b"evidence")
            if attempts < 3:
                raise PermissionError(13, "temporarily locked", str(path))

        def monotonic() -> float:
            return now

        def sleep(delay: float) -> None:
            nonlocal now
            delays.append(delay)
            now += delay

        _retry_atomic_write_bytes(
            writer,
            target,
            b"evidence",
            timeout_seconds=1.0,
            initial_delay_seconds=0.1,
            max_delay_seconds=0.5,
            monotonic=monotonic,
            sleep=sleep,
        )

        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [0.1, 0.2])

    def test_atomic_publish_does_not_retry_non_transient_error(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def writer(path: Path, payload: bytes) -> None:
            nonlocal attempts
            attempts += 1
            raise OSError(28, "disk full", str(path))

        with self.assertRaisesRegex(OSError, "disk full"):
            _retry_atomic_write_bytes(
                writer,
                self.root / "resume.json",
                b"evidence",
                monotonic=lambda: 0.0,
                sleep=sleeps.append,
            )

        self.assertEqual(attempts, 1)
        self.assertEqual(sleeps, [])

    def test_atomic_publish_shim_is_scoped_and_restored(self) -> None:
        original = locked_final_evaluation._atomic_write_bytes
        with _scoped_final_atomic_write_retry(timeout_seconds=0.0):
            self.assertIsNot(locked_final_evaluation._atomic_write_bytes, original)
        self.assertIs(locked_final_evaluation._atomic_write_bytes, original)


if __name__ == "__main__":
    unittest.main()
