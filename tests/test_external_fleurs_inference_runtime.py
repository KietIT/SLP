from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.run_external_fleurs import (
    REQUIRED_ROLES,
    ExternalAuthorization,
    ExternalFleursError,
    _provenance_path,
    _result_provenance_path,
)
from scripts.run_external_fleurs_inference_runtime import (
    EXTENSION_VERSION,
    RECEIPT_VERSION,
    _inference_extension_path,
    parse_args,
    run_external_fleurs_inference_runtime,
    verify_execution_receipt,
)
from src.vitonesr.phat.protocol import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _authorization(root: Path, registry: Path) -> ExternalAuthorization:
    manifest = root / "fleurs.jsonl"
    manifest.write_text("{}\n", encoding="utf-8", newline="\n")
    return ExternalAuthorization(
        registry_path=registry,
        registry_sha256=sha256_file(registry),
        registry={},
        split_lock_sha256="a" * 64,
        decision_lock_sha256="b" * 64,
        method_lock_sha256="c" * 64,
        method_identity_sha256="d" * 64,
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        expected_rows=857,
        locked_by_role={},
        method_environment_identity_sha256="1" * 64,
        method_source_tree_sha256="e" * 64,
        method_runtime_verified=False,
        formal=True,
    )


def _runtime_summary(
    runtime_lock: Path,
    training_environment: Path,
    *,
    training_identity: str = "1" * 64,
) -> dict[str, str]:
    return {
        "status": "VERIFIED",
        "schema_version": "paper_v2_inference_runtime_v1",
        "lock_sha256": sha256_file(runtime_lock),
        "identity_sha256": "2" * 64,
        "training_environment_sha256": sha256_file(training_environment),
        "training_environment_identity_sha256": training_identity,
        "inference_environment_identity_sha256": "3" * 64,
        "source_tree_sha256": "4" * 64,
    }


def _suite_runner_factory(
    *,
    expected_authorization: ExternalAuthorization,
    observations: dict[str, Any],
) -> Callable[..., tuple[list[Path], Path]]:
    def run(registry_path: str | Path, **kwargs: Any) -> tuple[list[Path], Path]:
        authorization = kwargs["authorization"]
        observations["delegated_authorization"] = authorization
        if authorization.method_runtime_verified:
            raise AssertionError("wrapper forged the old method-runtime gate")
        if (
            authorization.method_environment_identity_sha256
            != expected_authorization.method_environment_identity_sha256
        ):
            raise AssertionError("wrapper changed the training environment identity")

        output = ROOT / Path(str(kwargs["output_dir"]))
        output.mkdir(parents=True, exist_ok=True)
        predictions: list[Path] = []
        for index, role in enumerate(REQUIRED_ROLES):
            prediction = output / "predictions" / f"pred_{role}.csv"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            if not prediction.exists():
                prediction.write_text(
                    "utt_id,ref,hyp\nexample,xin chao,xin chao\n",
                    encoding="utf-8",
                    newline="\n",
                )
                _write_json(
                    _provenance_path(prediction),
                    {
                        "role": role,
                        "configuration_id": f"configuration-{index}",
                        "prediction_sha256": sha256_file(prediction),
                        "method_runtime_verified": False,
                        "method_environment_identity_sha256": (
                            authorization.method_environment_identity_sha256
                        ),
                    },
                )
            predictions.append(prediction)

        result = output / "external_fleurs_results.csv"
        if not result.exists():
            result.write_text(
                "role,wer\nordinary_baseline,0.1\n",
                encoding="utf-8",
                newline="\n",
            )
            _write_json(
                _result_provenance_path(result),
                {
                    "results_sha256": sha256_file(result),
                    "method_runtime_verified": False,
                    "method_environment_identity_sha256": (
                        authorization.method_environment_identity_sha256
                    ),
                },
            )
        return predictions, result

    return run


class ExternalFleursInferenceRuntimeTests(unittest.TestCase):
    def _workspace(self) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        root = Path(temporary.name)
        return temporary, root, root.relative_to(ROOT).as_posix()

    def test_separate_runtime_authorizes_suite_and_receipt_binds_every_artifact(
        self,
    ) -> None:
        temporary, root, prefix = self._workspace()
        with temporary:
            registry = root / "registry.json"
            runtime_lock = root / "inference_runtime_lock.json"
            training_environment = root / "environment_lock.json"
            registry.write_text("{}\n", encoding="utf-8")
            runtime_lock.write_text("{\"runtime\":true}\n", encoding="utf-8")
            training_environment.write_text(
                "{\"training\":true}\n", encoding="utf-8"
            )
            authorization = _authorization(root, registry)
            observations: dict[str, Any] = {}

            def base_authorizer(
                path: str | Path, **kwargs: Any
            ) -> ExternalAuthorization:
                observations["base_call"] = (path, kwargs)
                return authorization

            def verifier(
                lock: Path,
                training: Path,
                **kwargs: Any,
            ) -> Mapping[str, Any]:
                observations["runtime_call"] = (lock, training, kwargs)
                return _runtime_summary(lock, training)

            predictions, result, receipt = run_external_fleurs_inference_runtime(
                f"{prefix}/registry.json",
                inference_runtime_lock=f"{prefix}/inference_runtime_lock.json",
                training_environment_lock=f"{prefix}/environment_lock.json",
                output_dir=f"{prefix}/output",
                receipt_path=f"{prefix}/receipt.json",
                base_authorizer=base_authorizer,
                runtime_verifier=verifier,
                suite_runner=_suite_runner_factory(
                    expected_authorization=authorization,
                    observations=observations,
                ),
            )

            self.assertEqual(len(predictions), 3)
            self.assertTrue(result.is_file())
            self.assertTrue(receipt.is_file())
            self.assertEqual(
                observations["base_call"][1],
                {"formal": True, "verify_current_method": False},
            )
            self.assertTrue(observations["runtime_call"][2]["verify_current"])
            self.assertEqual(observations["runtime_call"][2]["repo_root"], ROOT)
            delegated = observations["delegated_authorization"]
            self.assertFalse(delegated.method_runtime_verified)
            self.assertEqual(
                delegated.method_environment_identity_sha256,
                authorization.method_environment_identity_sha256,
            )

            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["receipt_version"], RECEIPT_VERSION)
            self.assertFalse(payload["method_runtime_verified"])
            self.assertTrue(payload["inference_runtime_verified"])
            self.assertEqual(payload["inference_runtime"]["identity_sha256"], "2" * 64)
            self.assertEqual(
                payload["training"]["method_environment_identity_sha256"],
                "1" * 64,
            )
            self.assertFalse(
                payload["semantic_separation"]["method_lock_source_components_modified"]
            )
            self.assertEqual(
                [entry["role"] for entry in payload["predictions"]],
                list(REQUIRED_ROLES),
            )
            for entry, prediction in zip(payload["predictions"], predictions):
                self.assertEqual(
                    entry["prediction"]["sha256"], sha256_file(prediction)
                )
                self.assertEqual(
                    entry["provenance"]["sha256"],
                    sha256_file(_provenance_path(prediction)),
                )
                extension = _inference_extension_path(prediction)
                extension_payload = json.loads(extension.read_text(encoding="utf-8"))
                self.assertEqual(
                    extension_payload["extension_version"], EXTENSION_VERSION
                )
                self.assertFalse(extension_payload["method_runtime_verified"])
                self.assertTrue(extension_payload["inference_runtime_verified"])
            self.assertEqual(
                payload["aggregate"]["result"]["sha256"], sha256_file(result)
            )
            self.assertEqual(
                payload["aggregate"]["provenance"]["sha256"],
                sha256_file(_result_provenance_path(result)),
            )
            verified = verify_execution_receipt(
                f"{prefix}/receipt.json",
                inference_runtime_lock=f"{prefix}/inference_runtime_lock.json",
                training_environment_lock=f"{prefix}/environment_lock.json",
                runtime_verifier=verifier,
            )
            self.assertEqual(verified["status"], "VERIFIED")
            self.assertEqual(verified["prediction_count"], 3)

    def test_runtime_must_bind_the_original_training_environment(self) -> None:
        temporary, root, prefix = self._workspace()
        with temporary:
            registry = root / "registry.json"
            runtime_lock = root / "runtime.json"
            training = root / "training.json"
            for path in (registry, runtime_lock, training):
                path.write_text("{}\n", encoding="utf-8")
            authorization = _authorization(root, registry)

            def runner(registry_path: str | Path, **kwargs: Any) -> tuple[list[Path], Path]:
                self.fail("mismatched runtime must fail before any suite artifact")

            with self.assertRaisesRegex(
                ExternalFleursError, "different training environment"
            ):
                run_external_fleurs_inference_runtime(
                    f"{prefix}/registry.json",
                    inference_runtime_lock=f"{prefix}/runtime.json",
                    training_environment_lock=f"{prefix}/training.json",
                    output_dir=f"{prefix}/output",
                    receipt_path=f"{prefix}/receipt.json",
                    base_authorizer=lambda *_args, **_kwargs: authorization,
                    runtime_verifier=lambda lock, environment, **_kwargs: (
                        _runtime_summary(
                            lock,
                            environment,
                            training_identity="9" * 64,
                        )
                    ),
                    suite_runner=runner,
                )
            self.assertFalse((root / "receipt.json").exists())

    def test_receipt_is_no_overwrite_and_resume_requires_exact_artifacts(self) -> None:
        temporary, root, prefix = self._workspace()
        with temporary:
            registry = root / "registry.json"
            runtime_lock = root / "runtime.json"
            training = root / "training.json"
            for path in (registry, runtime_lock, training):
                path.write_text("{}\n", encoding="utf-8")
            authorization = _authorization(root, registry)
            calls = {"suite": 0}
            delegate = _suite_runner_factory(
                expected_authorization=authorization,
                observations={},
            )

            def suite(*args: Any, **kwargs: Any) -> tuple[list[Path], Path]:
                calls["suite"] += 1
                return delegate(*args, **kwargs)

            common: dict[str, Any] = {
                "inference_runtime_lock": f"{prefix}/runtime.json",
                "training_environment_lock": f"{prefix}/training.json",
                "output_dir": f"{prefix}/output",
                "receipt_path": f"{prefix}/receipt.json",
                "base_authorizer": lambda *_args, **_kwargs: authorization,
                "runtime_verifier": lambda lock, environment, **_kwargs: (
                    _runtime_summary(lock, environment)
                ),
                "suite_runner": suite,
            }
            _predictions, _result, receipt = (
                run_external_fleurs_inference_runtime(
                    f"{prefix}/registry.json", **common
                )
            )
            self.assertEqual(calls["suite"], 1)
            with self.assertRaisesRegex(FileExistsError, "pass --resume"):
                run_external_fleurs_inference_runtime(
                    f"{prefix}/registry.json", **common
                )
            self.assertEqual(calls["suite"], 1)

            run_external_fleurs_inference_runtime(
                f"{prefix}/registry.json", resume=True, **common
            )
            self.assertEqual(calls["suite"], 2)

            tampered = json.loads(receipt.read_text(encoding="utf-8"))
            tampered["registry"]["sha256"] = "f" * 64
            _write_json(receipt, tampered)
            with self.assertRaisesRegex(
                ExternalFleursError, "receipt differs"
            ):
                run_external_fleurs_inference_runtime(
                    f"{prefix}/registry.json", resume=True, **common
                )

    def test_cli_exposes_inference_lock_and_execution_essentials(self) -> None:
        args = parse_args(
            [
                "--run-registry",
                "registry.json",
                "--inference-runtime-lock",
                "runtime.json",
                "--output-dir",
                "out",
                "--results-path",
                "results.csv",
                "--receipt",
                "receipt.json",
                "--limit",
                "5",
                "--device",
                "cuda",
                "--checkpoint-every",
                "3",
                "--resume",
            ]
        )
        self.assertEqual(args.run_registry, "registry.json")
        self.assertEqual(args.inference_runtime_lock, "runtime.json")
        self.assertEqual(args.output_dir, "out")
        self.assertEqual(args.results_path, "results.csv")
        self.assertEqual(args.receipt, "receipt.json")
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.checkpoint_every, 3)
        self.assertTrue(args.resume)


if __name__ == "__main__":
    unittest.main()
