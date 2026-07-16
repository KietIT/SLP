from __future__ import annotations

import json
import importlib.metadata
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.vitonesr.phat.reproducibility import (
    ENVIRONMENT_SCHEMA_VERSION,
    EnvironmentCaptureError,
    capture_environment,
    validate_environment_artifact,
    write_environment_artifact,
)


COMMIT = "a" * 40


class _FakeCudnn:
    benchmark = False
    deterministic = True

    def __init__(self, *, available: bool, version: int | None) -> None:
        self._available = available
        self._version = version

    def is_available(self) -> bool:
        return self._available

    def version(self) -> int | None:
        return self._version


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 1 if self._available else 0

    def get_device_properties(self, index: int) -> SimpleNamespace:
        if not self._available or index != 0:
            raise AssertionError("unexpected device query")
        return SimpleNamespace(name="Test GPU", total_memory=6 * 1024**3)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        if not self._available or index != 0:
            raise AssertionError("unexpected capability query")
        return (8, 9)


def _fake_torch(*, cuda_available: bool) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.6.0+cu124",
        are_deterministic_algorithms_enabled=lambda: True,
        backends=SimpleNamespace(
            cudnn=_FakeCudnn(
                available=cuda_available,
                version=90100 if cuda_available else None,
            )
        ),
        cuda=_FakeCuda(available=cuda_available),
        version=SimpleNamespace(cuda="12.4" if cuda_available else None),
    )


def _git_runner(*, dirty: bool = False, available: bool = True):
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        if not available:
            return SimpleNamespace(returncode=128, stdout="")
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout=f"{COMMIT}\r\n")
        if "status" in command:
            status = " M src/example.py\r\n?? local.txt\r\n" if dirty else ""
            return SimpleNamespace(returncode=0, stdout=status)
        if "diff" in command:
            diff = "diff --git a/x b/x\n+changed\n" if dirty else ""
            return SimpleNamespace(returncode=0, stdout=diff)
        raise AssertionError(f"unexpected git command: {command}")

    return run, calls


def _capture(
    *,
    cuda_available: bool = False,
    dirty: bool = False,
    formal: bool = False,
    allow_dirty_repository: bool = False,
    captured_at: datetime | None = None,
    revisions: dict[str, str] | None = None,
) -> dict:
    runner, _ = _git_runner(dirty=dirty)
    return capture_environment(
        repo_root=Path("C:/private/workspace"),
        revisions=revisions or {"base_model": "b" * 40},
        package_names=("torch", "transformers"),
        required_packages=("torch", "transformers"),
        required_revisions=("base_model",),
        cli_args={
            "formal": formal,
            "profile": "paper_v2",
            "required_revisions": ["base_model"],
        },
        formal=formal,
        allow_dirty_repository=allow_dirty_repository,
        captured_at=captured_at,
        torch_module=_fake_torch(cuda_available=cuda_available),
        version_getter=lambda name: {
            "torch": "2.6.0",
            "transformers": "4.57.3",
        }[name],
        command_runner=runner,
    )


class EnvironmentCaptureTests(unittest.TestCase):
    def test_cpu_capture_is_path_free_and_records_git_without_file_names(self) -> None:
        artifact = _capture(cuda_available=False, dirty=True)
        environment = artifact["environment"]
        self.assertEqual(artifact["schema_version"], ENVIRONMENT_SCHEMA_VERSION)
        self.assertFalse(environment["runtime"]["cuda"]["available"])
        self.assertEqual(environment["runtime"]["cuda"]["devices"], [])
        self.assertIsNone(environment["runtime"]["cuda"]["compiled_version"])
        self.assertEqual(environment["repository"]["commit"], COMMIT)
        self.assertTrue(environment["repository"]["dirty"])
        self.assertEqual(environment["repository"]["changed_entry_count"], 2)
        serialized = json.dumps(artifact, sort_keys=True)
        self.assertNotIn("C:/private/workspace", serialized)
        self.assertNotIn("src/example.py", serialized)
        self.assertNotIn("local.txt", serialized)

    def test_gpu_capture_records_name_memory_capability_cuda_and_cudnn(self) -> None:
        artifact = _capture(cuda_available=True)
        runtime = artifact["environment"]["runtime"]
        self.assertEqual(runtime["torch_version"], "2.6.0+cu124")
        self.assertEqual(runtime["cuda"]["compiled_version"], "12.4")
        self.assertEqual(runtime["cuda"]["device_count"], 1)
        self.assertEqual(
            runtime["cuda"]["devices"],
            [
                {
                    "compute_capability": [8, 9],
                    "index": 0,
                    "name": "Test GPU",
                    "total_memory_bytes": 6 * 1024**3,
                }
            ],
        )
        self.assertTrue(runtime["cudnn"]["available"])
        self.assertEqual(runtime["cudnn"]["version"], 90100)

    def test_timestamp_is_excluded_from_stable_identity(self) -> None:
        first = _capture(
            captured_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        )
        second = _capture(
            captured_at=datetime(2026, 2, 2, 3, 4, tzinfo=timezone.utc)
        )
        self.assertNotEqual(first["captured_at_utc"], second["captured_at_utc"])
        self.assertEqual(first["identity_sha256"], second["identity_sha256"])
        self.assertEqual(first["environment"], second["environment"])

    def test_git_commands_are_read_only_and_root_never_enters_artifact(self) -> None:
        runner, calls = _git_runner()
        artifact = capture_environment(
            repo_root=Path("C:/private/workspace"),
            revisions={"base_model": "b" * 40},
            package_names=("torch",),
            required_packages=("torch",),
            cli_args={},
            command_runner=runner,
            torch_module=_fake_torch(cuda_available=False),
            version_getter=lambda _: "2.6.0",
        )
        self.assertEqual(len(calls), 3)
        self.assertTrue(
            all(
                call[:5]
                == [
                    "git",
                    "-c",
                    "safe.directory=C:/private/workspace",
                    "-C",
                    "C:\\private\\workspace",
                ]
                for call in calls
            )
        )
        self.assertNotIn("private", json.dumps(artifact))

    def test_formal_capture_accepts_complete_clean_immutable_environment(self) -> None:
        artifact = _capture(formal=True)
        self.assertEqual(artifact["environment"]["capture_mode"], "formal")
        validate_environment_artifact(artifact)

    def test_formal_capture_rejects_missing_dependency_version(self) -> None:
        runner, _ = _git_runner()

        def version(name: str) -> str:
            if name == "torch":
                return "2.6.0"
            raise importlib.metadata.PackageNotFoundError(name)

        with self.assertRaisesRegex(EnvironmentCaptureError, "transformers"):
            capture_environment(
                repo_root="repo",
                revisions={"base_model": "b" * 40},
                package_names=("torch", "transformers"),
                required_packages=("torch", "transformers"),
                formal=True,
                torch_module=_fake_torch(cuda_available=False),
                version_getter=version,
                command_runner=runner,
            )

    def test_formal_capture_rejects_missing_revision_and_dirty_tree(self) -> None:
        runner, _ = _git_runner()
        with self.assertRaisesRegex(EnvironmentCaptureError, "missing required revisions"):
            capture_environment(
                repo_root="repo",
                revisions={"method_lock": "c" * 64},
                package_names=("torch",),
                required_packages=("torch",),
                required_revisions=("base_model",),
                formal=True,
                torch_module=_fake_torch(cuda_available=False),
                version_getter=lambda _: "2.6.0",
                command_runner=runner,
            )
        with self.assertRaisesRegex(EnvironmentCaptureError, "clean repository"):
            _capture(formal=True, dirty=True)

        locked_dirty = _capture(
            formal=True,
            dirty=True,
            allow_dirty_repository=True,
        )
        self.assertEqual(
            locked_dirty["environment"]["repository_policy"],
            "content_locked_dirty_allowed",
        )
        self.assertTrue(locked_dirty["environment"]["repository"]["dirty"])

        unavailable_runner, _ = _git_runner(available=False)
        with self.assertRaisesRegex(EnvironmentCaptureError, "repository commit"):
            capture_environment(
                repo_root="repo",
                revisions={"base_model": "b" * 40},
                package_names=("torch",),
                required_packages=("torch",),
                formal=True,
                torch_module=_fake_torch(cuda_available=False),
                version_getter=lambda _: "2.6.0",
                command_runner=unavailable_runner,
            )

    def test_floating_revision_secret_or_absolute_cli_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(EnvironmentCaptureError, "immutable"):
            _capture(revisions={"base_model": "main"})
        with self.assertRaisesRegex(EnvironmentCaptureError, "credential"):
            _capture(revisions={"base_model": "hf_abcdefghijklmnopqrst"})
        runner, _ = _git_runner()
        for cli_args, message in (
            ({"api_token": "abc"}, "Sensitive CLI"),
            ({"manifest": "C:/private/data.csv"}, "absolute filesystem path"),
        ):
            with self.subTest(cli_args=cli_args):
                with self.assertRaisesRegex(EnvironmentCaptureError, message):
                    capture_environment(
                        repo_root="repo",
                        revisions={"base_model": "b" * 40},
                        package_names=("torch",),
                        required_packages=("torch",),
                        cli_args=cli_args,
                        torch_module=_fake_torch(cuda_available=False),
                        version_getter=lambda _: "2.6.0",
                        command_runner=runner,
                    )

    def test_write_is_canonical_atomic_and_no_overwrite_by_default(self) -> None:
        first = _capture(
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        second = _capture(
            captured_at=datetime(2026, 1, 2, tzinfo=timezone.utc)
        )
        second["environment"]["cli_args"]["profile"] = "paper_v2_changed"
        hash_payload = {
            "environment": second["environment"],
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        }
        import hashlib

        second["identity_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    hash_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "nested" / "environment.json"
            write_environment_artifact(output, first)
            original = output.read_bytes()
            expected = (
                json.dumps(
                    first,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            self.assertEqual(original, expected)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_environment_artifact(output, second)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

            write_environment_artifact(output, second, overwrite=True)
            self.assertNotEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_atomic_link_failure_leaves_no_partial_output_or_temp(self) -> None:
        artifact = _capture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "environment.json"
            with patch(
                "src.vitonesr.phat.reproducibility.os.link",
                side_effect=OSError("simulated link failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_environment_artifact(output, artifact)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_tampered_artifact_is_rejected(self) -> None:
        artifact = _capture()
        artifact["environment"]["packages"]["torch"] = "0.0"
        with self.assertRaisesRegex(EnvironmentCaptureError, "identity hash"):
            validate_environment_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
