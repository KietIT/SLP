from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.vitonesr.artifact_bundle import (
    bind_input_files,
    commit_artifact_bundle,
)


class BundleTestError(ValueError):
    pass


class SimulatedProcessCrash(BaseException):
    """Bypasses the normal Exception rollback path like a killed process."""


class ArtifactBundleTests(unittest.TestCase):
    @staticmethod
    def _call(
        root: Path,
        *,
        resume: bool = False,
        overwrite: bool = False,
        contents: tuple[bytes, bytes] = (b"alpha\n", b"beta\n"),
    ):
        source = root / "input.csv"
        bindings = bind_input_files((source,), root=root)
        return commit_artifact_bundle(
            bundle_name="fixture",
            bundle_version="fixture_v1",
            data_destinations={
                "a": root / "out/a.csv",
                "b": root / "out/b.png",
            },
            data_contents={"a": contents[0], "b": contents[1]},
            provenance_path=root / "out/fixture.provenance.json",
            input_bindings=bindings,
            parameters={"seed": 42},
            overwrite=overwrite,
            resume=resume,
            error_type=BundleTestError,
        )

    def test_crash_leaves_journal_and_exact_resume_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.csv").write_bytes(b"input\n")
            original = os.link

            def crash_on_second(source: str | Path, target: str | Path) -> None:
                if Path(source).name == ".b.png.tmp":
                    raise SimulatedProcessCrash()
                original(source, target)

            with mock.patch(
                "src.vitonesr.artifact_bundle.os.link",
                side_effect=crash_on_second,
            ), self.assertRaises(SimulatedProcessCrash):
                self._call(root)

            output = root / "out"
            self.assertTrue((output / ".fixture.bundle.transaction.json").is_file())
            self.assertFalse((output / "fixture.bundle.commit.json").exists())
            self.assertTrue((output / "a.csv").is_file())
            self.assertTrue((output / ".b.png.tmp").is_file())

            result = self._call(root, resume=True)
            self.assertTrue(result.marker_path.is_file())
            self.assertTrue(result.provenance_path.is_file())
            self.assertFalse((output / ".fixture.bundle.transaction.json").exists())
            before = tuple(path.read_bytes() for path in result.destinations)
            stale_lock = output / ".fixture.bundle.lock"
            stale_lock.write_text("pid=2147483647\n", encoding="ascii")
            again = self._call(root, resume=True)
            self.assertEqual(before, tuple(path.read_bytes() for path in again.destinations))
            self.assertFalse(stale_lock.exists())
            provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(provenance["inputs"][0]["sha256"], bind_input_files((root / "input.csv",), root=root)[0]["sha256"])

    def test_resume_rejects_changed_input_and_tampered_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            source.write_bytes(b"input\n")
            original = os.link

            def crash_on_second(stage: str | Path, destination: str | Path) -> None:
                if Path(stage).name == ".b.png.tmp":
                    raise SimulatedProcessCrash()
                original(stage, destination)

            with mock.patch(
                "src.vitonesr.artifact_bundle.os.link",
                side_effect=crash_on_second,
            ), self.assertRaises(SimulatedProcessCrash):
                self._call(root)
            source.write_bytes(b"changed input\n")
            with self.assertRaisesRegex(BundleTestError, "exact input/output bundle"):
                self._call(root, resume=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.csv").write_bytes(b"input\n")
            original = os.link

            def crash(stage: str | Path, destination: str | Path) -> None:
                if Path(stage).name == ".b.png.tmp":
                    raise SimulatedProcessCrash()
                original(stage, destination)

            with mock.patch(
                "src.vitonesr.artifact_bundle.os.link",
                side_effect=crash,
            ), self.assertRaises(SimulatedProcessCrash):
                self._call(root)
            (root / "out/.b.png.tmp").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(BundleTestError, "staged bundle output was tampered"):
                self._call(root, resume=True)

    def test_orphan_and_committed_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.csv").write_bytes(b"input\n")
            output = root / "out"
            output.mkdir()
            (output / ".a.csv.tmp").write_bytes(b"alpha\n")
            with self.assertRaisesRegex(BundleTestError, "orphan bundle stage"):
                self._call(root, resume=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.csv").write_bytes(b"input\n")
            result = self._call(root)
            result.destinations[0].write_bytes(b"tampered\n")
            with self.assertRaisesRegex(BundleTestError, "tampered"):
                self._call(root, resume=True)

    def test_overwrite_crash_resumes_forward_and_exception_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.csv").write_bytes(b"input\n")
            old = (b"old-a\n", b"old-b\n")
            new = (b"new-a\n", b"new-b\n")
            first = self._call(root, contents=old)
            original = Path.replace

            def hard_crash(stage: Path, destination: str | Path) -> Path:
                if stage.name == ".b.png.tmp":
                    raise SimulatedProcessCrash()
                return original(stage, destination)

            with mock.patch.object(
                Path, "replace", autospec=True, side_effect=hard_crash
            ), self.assertRaises(SimulatedProcessCrash):
                self._call(root, overwrite=True, contents=new)
            self.assertFalse(first.marker_path.exists())
            recovered = self._call(root, resume=True, contents=new)
            self.assertEqual(
                tuple(path.read_bytes() for path in recovered.destinations), new
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.csv").write_bytes(b"input\n")
            old = (b"old-a\n", b"old-b\n")
            new = (b"new-a\n", b"new-b\n")
            first = self._call(root, contents=old)
            marker_before = first.marker_path.read_bytes()
            original = Path.replace

            def ordinary_failure(stage: Path, destination: str | Path) -> Path:
                if stage.name == ".b.png.tmp":
                    raise OSError("injected replace failure")
                return original(stage, destination)

            with mock.patch.object(
                Path, "replace", autospec=True, side_effect=ordinary_failure
            ), self.assertRaisesRegex(OSError, "injected replace failure"):
                self._call(root, overwrite=True, contents=new)
            self.assertEqual(
                tuple(path.read_bytes() for path in first.destinations), old
            )
            self.assertEqual(first.marker_path.read_bytes(), marker_before)
            self._call(root, resume=True, contents=old)


if __name__ == "__main__":
    unittest.main()
