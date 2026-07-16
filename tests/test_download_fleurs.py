from __future__ import annotations

import json
import os
import tempfile
import unicodedata
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.download_fleurs import (
    DEFAULT_EXPECTED_COUNT,
    MANIFEST_FIELDS,
    PREPARATION_LOCK_VERSION,
    TARGET_SAMPLE_RATE,
    FleursPreparationError,
    load_fleurs_dataset,
    main,
    prepare_fleurs as prepare_fleurs_formal,
    validate_manifest as validate_manifest_impl,
    verify_fleurs_preparation_lock,
)


REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def prepare_fleurs(**kwargs: object) -> Path:
    """Exercise legacy mechanics only through the explicit diagnostic mode."""

    kwargs.setdefault("formal", False)
    return prepare_fleurs_formal(**kwargs)  # type: ignore[arg-type]


def validate_manifest(path: str | Path, **kwargs: object) -> list[dict[str, object]]:
    kwargs.setdefault("formal", False)
    return validate_manifest_impl(path, **kwargs)  # type: ignore[arg-type,return-value]


def fleurs_row(
    filename: str,
    transcript: str,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    samples: np.ndarray | None = None,
) -> dict[str, object]:
    if samples is None:
        samples = np.linspace(-0.25, 0.25, 160, dtype=np.float32)
    return {
        "id": 1,  # Deliberately not used as utt_id.
        "audio": {
            "path": filename,
            "array": samples,
            "sampling_rate": sample_rate,
        },
        "transcription": transcript,
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class DownloadFleursTest(unittest.TestCase):
    def test_hugging_face_loader_is_pinned_to_vietnamese_fleurs(self) -> None:
        expected = object()
        with mock.patch("datasets.load_dataset", return_value=expected) as load_dataset:
            observed = load_fleurs_dataset(
                split="test", cache_dir="cache", revision="revision-123"
            )
        self.assertIs(observed, expected)
        load_dataset.assert_called_once_with(
            "google/fleurs",
            "vi_vn",
            split="test",
            cache_dir="cache",
            revision="revision-123",
        )

    def test_materializes_test_split_with_canonical_deterministic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            source_rows = [
                fleurs_row("1002.wav", "Tôi đi học"),
                fleurs_row("1001.wav", "Xin chào"),
            ]
            loader = mock.Mock(return_value=source_rows)

            manifest = prepare_fleurs(
                out_dir=root,
                cache_dir=root / "cache",
                expected_count=2,
                revision="revision-123",
                dataset_loader=loader,
            )
            first_manifest_bytes = manifest.read_bytes()
            first_wav_bytes = (root / "audio" / "test" / "1002.wav").read_bytes()
            rows = read_jsonl(manifest)

            self.assertEqual(manifest, root / "test.jsonl")
            loader.assert_called_once_with(
                split="test", cache_dir=root / "cache", revision="revision-123"
            )
            self.assertEqual([tuple(row) for row in rows], [MANIFEST_FIELDS] * 2)
            self.assertEqual([row["utt_id"] for row in rows], ["1002", "1001"])
            self.assertEqual(rows[0]["transcript"], "Tôi đi học")
            self.assertTrue(unicodedata.is_normalized("NFC", str(rows[0]["transcript"])))
            self.assertEqual(
                {row["dataset"] for row in rows},
                {"fleurs"},
            )
            self.assertEqual({row["split"] for row in rows}, {"test"})
            self.assertEqual({row["snr"] for row in rows}, {"clean"})
            self.assertEqual({row["noise_type"] for row in rows}, {"clean"})
            self.assertTrue(Path(str(rows[0]["audio_path"])).is_absolute())
            validate_manifest(manifest, expected_count=2)

            with wave.open(str(rows[0]["audio_path"]), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), TARGET_SAMPLE_RATE)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getnframes(), 160)

            prepare_fleurs(
                out_dir=root,
                cache_dir=root / "cache",
                expected_count=2,
                revision="revision-123",
                overwrite=True,
                dataset_loader=mock.Mock(return_value=source_rows),
            )
            self.assertEqual(manifest.read_bytes(), first_manifest_bytes)
            self.assertEqual(
                (root / "audio" / "test" / "1002.wav").read_bytes(),
                first_wav_bytes,
            )
            self.assertFalse((root / ".test.lock").exists())

    def test_resamples_non_16k_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            manifest = prepare_fleurs(
                out_dir=root,
                expected_count=1,
                dataset_loader=mock.Mock(
                    return_value=[
                        fleurs_row(
                            "sample.wav",
                            "một câu",
                            sample_rate=8_000,
                            samples=np.linspace(-0.5, 0.5, 80, dtype=np.float32),
                        )
                    ]
                ),
            )
            audio_path = Path(str(read_jsonl(manifest)[0]["audio_path"]))
            with wave.open(str(audio_path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), TARGET_SAMPLE_RATE)
                self.assertEqual(wav_file.getnframes(), 160)

    def test_existing_output_requires_resume_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            rows = [fleurs_row("sample.wav", "một câu")]
            manifest = prepare_fleurs(
                out_dir=root,
                expected_count=1,
                dataset_loader=mock.Mock(return_value=rows),
            )
            original = manifest.read_bytes()

            never_called = mock.Mock(side_effect=AssertionError("must not load remote data"))
            resumed = prepare_fleurs(
                out_dir=root,
                expected_count=1,
                resume=True,
                dataset_loader=never_called,
            )
            self.assertEqual(resumed, manifest)
            never_called.assert_not_called()
            self.assertEqual(manifest.read_bytes(), original)

            with self.assertRaisesRegex(FileExistsError, "use --resume or --overwrite"):
                prepare_fleurs(
                    out_dir=root,
                    expected_count=1,
                    dataset_loader=mock.Mock(return_value=rows),
                )
            self.assertEqual(manifest.read_bytes(), original)
            self.assertFalse((root / ".test.lock").exists())

    def test_resume_reuses_valid_audio_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            rows = [fleurs_row("sample.wav", "một câu")]
            manifest = prepare_fleurs(
                out_dir=root,
                expected_count=1,
                dataset_loader=mock.Mock(return_value=rows),
            )
            wav_path = root / "audio" / "test" / "sample.wav"
            original_wav = wav_path.read_bytes()
            manifest.unlink()

            prepare_fleurs(
                out_dir=root,
                expected_count=1,
                resume=True,
                dataset_loader=mock.Mock(return_value=rows),
            )
            self.assertEqual(wav_path.read_bytes(), original_wav)
            self.assertTrue(manifest.exists())

    def test_duplicate_filename_stem_and_bad_rows_are_rejected(self) -> None:
        cases = [
            (
                "duplicate",
                [fleurs_row("a/42.wav", "one"), fleurs_row("b/42.flac", "two")],
                "duplicate audio filename stem",
            ),
            (
                "missing_transcript",
                [{**fleurs_row("42.wav", "one"), "transcription": "  "}],
                "transcription must be a non-empty string",
            ),
            (
                "unsafe_stem",
                [fleurs_row("xin chao.wav", "one")],
                "unsafe utt_id",
            ),
            (
                "invalid_audio",
                [fleurs_row("42.wav", "one", samples=np.array([np.nan]))],
                "contains NaN/Inf",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, rows, message in cases:
                with self.subTest(name=name):
                    out_dir = root / name
                    with self.assertRaisesRegex(FleursPreparationError, message):
                        prepare_fleurs(
                            out_dir=out_dir,
                            expected_count=len(rows),
                            dataset_loader=mock.Mock(return_value=rows),
                        )
                    self.assertFalse((out_dir / "test.jsonl").exists())
                    self.assertFalse((out_dir / ".test.lock").exists())

    def test_count_mismatch_fails_before_writing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            with self.assertRaisesRegex(FleursPreparationError, "source split has 1 rows"):
                prepare_fleurs(
                    out_dir=root,
                    expected_count=857,
                    dataset_loader=mock.Mock(return_value=[fleurs_row("42.wav", "one")]),
                )
            self.assertFalse((root / "audio").exists())
            self.assertFalse((root / "test.jsonl").exists())

    def test_lock_is_respected_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            root.mkdir(parents=True)
            lock = root / ".test.lock"
            lock.write_text("pid=external\n", encoding="utf-8")
            with self.assertRaisesRegex(FleursPreparationError, "locked by another process"):
                prepare_fleurs(
                    out_dir=root,
                    expected_count=1,
                    dataset_loader=mock.Mock(
                        return_value=[fleurs_row("42.wav", "one")]
                    ),
                )
            self.assertEqual(lock.read_text(encoding="utf-8"), "pid=external\n")

    def test_manifest_commit_failure_leaves_no_partial_manifest_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            rows = [fleurs_row("42.wav", "một câu")]
            manifest = root / "test.jsonl"
            original_replace = os.replace

            def fail_manifest_commit(source: str | Path, target: str | Path) -> None:
                if Path(target) == manifest:
                    raise OSError("injected manifest commit failure")
                original_replace(source, target)

            with mock.patch(
                "scripts.download_fleurs.os.replace", side_effect=fail_manifest_commit
            ):
                with self.assertRaisesRegex(OSError, "injected manifest commit failure"):
                    prepare_fleurs(
                        out_dir=root,
                        expected_count=1,
                        dataset_loader=mock.Mock(return_value=rows),
                    )

            self.assertFalse(manifest.exists())
            self.assertFalse((root / ".test.jsonl.staged").exists())
            self.assertFalse((root / ".test.lock").exists())
            self.assertTrue((root / "audio" / "test" / "42.wav").exists())

            prepare_fleurs(
                out_dir=root,
                expected_count=1,
                resume=True,
                dataset_loader=mock.Mock(return_value=rows),
            )
            self.assertTrue(manifest.exists())

    def test_validator_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            manifest = prepare_fleurs(
                out_dir=root,
                expected_count=1,
                dataset_loader=mock.Mock(
                    return_value=[fleurs_row("42.wav", "một câu")]
                ),
            )
            rows = read_jsonl(manifest)
            rows[0]["utt_id"] = "different"
            manifest.write_text(
                json.dumps(rows[0], ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FleursPreparationError, "does not match audio"):
                validate_manifest(manifest, expected_count=1)

    def test_formal_lock_is_portable_exact_revision_bound_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            out_dir = Path("data/manifests/fleurs/paper_v2")
            lock = Path("outputs/paper_v2/protocol/fleurs_test_lock.json")
            audit = Path("outputs/paper_v2/protocol/fleurs_test_audit.csv")
            source_rows = [
                fleurs_row(
                    f"{index:04d}.wav",
                    f"câu thử {index}",
                    samples=np.array([0.0], dtype=np.float32),
                )
                for index in range(DEFAULT_EXPECTED_COUNT)
            ]
            manifest = prepare_fleurs_formal(
                out_dir=out_dir,
                cache_dir=repo / "cache",
                revision=REVISION,
                repository_root=repo,
                preparation_lock=lock,
                preparation_audit=audit,
                dataset_loader=mock.Mock(return_value=source_rows),
            )
            lock_path = repo / lock
            verified = verify_fleurs_preparation_lock(
                lock_path,
                repository_root=repo,
            )
            rows = read_jsonl(manifest)
            self.assertEqual(len(rows), 857)
            self.assertEqual(verified["lock_version"], PREPARATION_LOCK_VERSION)
            self.assertEqual(verified["dataset"]["revision"], REVISION)
            self.assertFalse(Path(str(rows[0]["audio_path"])).is_absolute())
            self.assertNotIn("..", Path(str(rows[0]["audio_path"])).parts)
            self.assertEqual(len(str(rows[0]["audio_sha256"])), 64)
            self.assertTrue((repo / audit).is_file())

            absolute_rows = [dict(row) for row in rows]
            absolute_rows[0]["audio_path"] = str(
                (repo / str(rows[0]["audio_path"])).resolve()
            )
            unsafe_manifest = repo / "unsafe.jsonl"
            unsafe_manifest.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    for row in absolute_rows
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FleursPreparationError, "repository-relative"):
                validate_manifest_impl(
                    unsafe_manifest,
                    repository_root=repo,
                    expected_count=857,
                    formal=True,
                )

            audio = repo / str(rows[0]["audio_path"])
            original_audio = audio.read_bytes()
            audio.write_bytes(original_audio + b"tamper")
            with self.assertRaisesRegex(FleursPreparationError, "SHA-256 mismatch"):
                verify_fleurs_preparation_lock(lock_path, repository_root=repo)
            audio.write_bytes(original_audio)

            original_lock = lock_path.read_text(encoding="utf-8")
            tampered_lock = json.loads(original_lock)
            tampered_lock["dataset"]["revision"] = "f" * 40
            lock_path.write_text(json.dumps(tampered_lock), encoding="utf-8")
            with self.assertRaisesRegex(FleursPreparationError, "identity.*tampered"):
                verify_fleurs_preparation_lock(lock_path, repository_root=repo)

    def test_formal_mode_rejects_missing_revision_and_non_857_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FleursPreparationError, "exactly 857"):
                prepare_fleurs_formal(
                    out_dir=Path(temporary) / "fleurs",
                    expected_count=1,
                    revision=REVISION,
                    dataset_loader=mock.Mock(return_value=[]),
                )
            with self.assertRaisesRegex(FleursPreparationError, "immutable"):
                prepare_fleurs_formal(
                    out_dir=Path(temporary) / "fleurs",
                    revision=None,
                    dataset_loader=mock.Mock(return_value=[]),
                )

    def test_cli_uses_test_split_by_default_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fleurs"
            loader = mock.Mock(return_value=[fleurs_row("42.wav", "một câu")])
            with mock.patch("scripts.download_fleurs.load_fleurs_dataset", loader):
                return_code = main(
                    [
                        "--out-dir",
                        str(root),
                        "--cache-dir",
                        str(root / "cache"),
                        "--expected-count",
                        "1",
                        "--revision",
                        "revision-123",
                        "--diagnostic",
                    ]
                )
            self.assertEqual(return_code, 0)
            loader.assert_called_once_with(
                split="test", cache_dir=str(root / "cache"), revision="revision-123"
            )
            self.assertTrue((root / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
