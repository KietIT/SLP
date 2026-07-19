from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vitonesr.inference_runtime import (
    INFERENCE_RUNTIME_SCHEMA_VERSION,
    REQUIRED_INFERENCE_SOURCE_PATHS,
    InferenceRuntimeLockError,
    capture_inference_runtime_lock,
    verify_inference_runtime_lock,
)
from src.vitonesr.phat.reproducibility import capture_environment, write_environment_artifact


def _fake_base_capture(
    *,
    package_version: str = "1.0",
    runtime_version: str | None = None,
    gpu_name: str = "Test GPU",
    deterministic: bool = False,
) -> dict:
    return {
        "captured_at_utc": "2026-07-19T00:00:00Z",
        "environment": {
            "packages": {"torch": package_version, "transformers": "4.0"},
            "platform": {"machine": "AMD64", "system": "Windows"},
            "python": {
                "implementation": "CPython",
                "version": "3.12.0",
                "version_info": [3, 12, 0],
            },
            "runtime": {
                "cuda": {
                    "available": True,
                    "compiled_version": "12.4",
                    "device_count": 1,
                    "devices": [{"index": 0, "name": gpu_name}],
                    "query_status": "available",
                },
                "cudnn": {
                    "available": True,
                    "benchmark": not deterministic,
                    "deterministic": deterministic,
                    "version": 90100,
                },
                "deterministic_algorithms_enabled": deterministic,
                "torch_version": runtime_version or package_version,
            },
        },
    }


def _training_artifact(repo: Path) -> dict:
    def runner(command: list[str], **_: object):
        class Result:
            returncode = 0
            stdout = ""

        result = Result()
        if "rev-parse" in command:
            result.stdout = "a" * 40 + "\n"
        return result

    return capture_environment(
        repo_root=repo,
        revisions={"base_model": "b" * 40},
        package_names=("torch",),
        required_packages=("torch",),
        required_revisions=("base_model",),
        formal=True,
        version_getter=lambda _: "2.11.0",
        command_runner=runner,
    )


class InferenceRuntimeLockTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[list[Path], Path, Path]:
        sources: list[Path] = []
        for relative in REQUIRED_INFERENCE_SOURCE_PATHS:
            source = root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# {relative}\n", encoding="utf-8")
            sources.append(source)
        training = root / "outputs" / "environment_lock.json"
        write_environment_artifact(training, _training_artifact(root))
        output = root / "outputs" / "inference_runtime_lock.json"
        return sources, training, output

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_capture_preserves_training_identity_and_binds_runtime_and_source(
        self, capture_mock
    ) -> None:
        capture_mock.return_value = _fake_base_capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            artifact = capture_inference_runtime_lock(
                output, training, sources, root
            )
            original_training = json.loads(training.read_text(encoding="utf-8"))
            self.assertEqual(artifact["schema_version"], INFERENCE_RUNTIME_SCHEMA_VERSION)
            self.assertEqual(artifact["status"], "LOCKED")
            self.assertEqual(
                artifact["training_environment"]["identity_sha256"],
                original_training["identity_sha256"],
            )
            self.assertEqual(artifact["training_environment"]["binding_status"], "PRESERVED")
            self.assertEqual(
                artifact["source"]["required_paths"],
                list(REQUIRED_INFERENCE_SOURCE_PATHS),
            )
            self.assertEqual(
                artifact["inference_environment"]["runtime"]["cuda"]["devices"][0]["name"],
                "Test GPU",
            )
            result = verify_inference_runtime_lock(output, training, root)
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(result["identity_sha256"], artifact["identity_sha256"])
            self.assertEqual(len(result["lock_sha256"]), 64)
            self.assertEqual(
                result["source_component_paths"],
                sorted(REQUIRED_INFERENCE_SOURCE_PATHS),
            )

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_no_overwrite_is_atomic(self, capture_mock) -> None:
        capture_mock.return_value = _fake_base_capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            capture_inference_runtime_lock(output, training, sources, root)
            original = output.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                capture_inference_runtime_lock(output, training, sources, root)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_tampered_source_and_training_environment_are_rejected(
        self, capture_mock
    ) -> None:
        capture_mock.return_value = _fake_base_capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            capture_inference_runtime_lock(output, training, sources, root)
            source = root / "scripts" / "run_final_lora_inference_runtime.py"
            source.write_text("print('changed')\n", encoding="utf-8")
            with self.assertRaisesRegex(InferenceRuntimeLockError, "source has changed"):
                verify_inference_runtime_lock(output, training, root, verify_current=False)

            source.write_text(
                "# scripts/run_final_lora_inference_runtime.py\n", encoding="utf-8"
            )
            training.write_bytes(training.read_bytes() + b" \n")
            with self.assertRaisesRegex(InferenceRuntimeLockError, "binding has changed"):
                verify_inference_runtime_lock(output, training, root, verify_current=False)

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_current_package_or_gpu_drift_is_rejected(self, capture_mock) -> None:
        capture_mock.return_value = _fake_base_capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            capture_inference_runtime_lock(output, training, sources, root)
            capture_mock.return_value = _fake_base_capture(package_version="2.0")
            with self.assertRaisesRegex(InferenceRuntimeLockError, "package drift: torch"):
                verify_inference_runtime_lock(output, training, root, verify_current=True)
            result = verify_inference_runtime_lock(
                output, training, root, verify_current=False
            )
            self.assertEqual(result["status"], "VERIFIED")

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_mutable_flags_may_change_but_gpu_and_torch_version_may_not(
        self, capture_mock
    ) -> None:
        capture_mock.return_value = _fake_base_capture(deterministic=False)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            capture_inference_runtime_lock(output, training, sources, root)

            capture_mock.return_value = _fake_base_capture(deterministic=True)
            verified = verify_inference_runtime_lock(output, training, root)
            self.assertEqual(verified["status"], "VERIFIED")

            capture_mock.return_value = _fake_base_capture(
                deterministic=True, gpu_name="Different GPU"
            )
            with self.assertRaisesRegex(InferenceRuntimeLockError, "does not match"):
                verify_inference_runtime_lock(output, training, root)

            capture_mock.return_value = _fake_base_capture(
                deterministic=True, runtime_version="9.9"
            )
            with self.assertRaisesRegex(InferenceRuntimeLockError, "does not match"):
                verify_inference_runtime_lock(output, training, root)

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_tampered_lock_identity_is_rejected(self, capture_mock) -> None:
        capture_mock.return_value = _fake_base_capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            capture_inference_runtime_lock(output, training, sources, root)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            artifact["inference_environment"]["packages"]["torch"] = "0.0"
            output.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(InferenceRuntimeLockError, "identity hash"):
                verify_inference_runtime_lock(output, training, root, verify_current=False)

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_sources_must_be_explicit_unique_and_inside_repo(self, capture_mock) -> None:
        capture_mock.return_value = _fake_base_capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            with self.assertRaisesRegex(InferenceRuntimeLockError, "At least one"):
                capture_inference_runtime_lock(output, training, [], root)
            with self.assertRaisesRegex(InferenceRuntimeLockError, "Duplicate"):
                capture_inference_runtime_lock(
                    output, training, [*sources, sources[0]], root
                )
            outside = root.parent / "outside-runtime-source.py"
            outside.write_text("x = 1\n", encoding="utf-8")
            try:
                with self.assertRaisesRegex(InferenceRuntimeLockError, "inside the repository"):
                    capture_inference_runtime_lock(output, training, [*sources, outside], root)
            finally:
                outside.unlink(missing_ok=True)

    @patch("src.vitonesr.inference_runtime.capture_environment")
    def test_unrelated_or_omitted_required_sources_are_rejected(
        self, capture_mock
    ) -> None:
        capture_mock.return_value = _fake_base_capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources, training, output = self._workspace(root)
            unrelated = root / "scripts" / "unrelated.py"
            unrelated.write_text("# unrelated\n", encoding="utf-8")
            with self.assertRaisesRegex(
                InferenceRuntimeLockError, "missing required components"
            ):
                capture_inference_runtime_lock(output, training, [unrelated], root)
            with self.assertRaisesRegex(
                InferenceRuntimeLockError, "missing required components"
            ):
                capture_inference_runtime_lock(output, training, sources[:-1], root)


if __name__ == "__main__":
    unittest.main()
