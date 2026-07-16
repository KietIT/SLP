from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from src.vitonesr.noise_protocol import (
    MUSAN_TYPES,
    NOISE_REGISTRY_COLUMNS,
    NOISY_DEV_COLUMNS,
    MusanSourceMetadata,
    NoiseProtocolError,
    NoisyDevConfig,
    build_noise_protocol_outputs,
    build_noisy_dev_benchmark,
    sha256_file,
    verify_noise_split_lock,
    write_locked_noise_outputs,
)


def _write_wav(
    path: Path,
    *,
    frequency: float,
    amplitude: float = 0.4,
    seconds: float = 0.08,
    sample_rate: int = 16000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(int(seconds * sample_rate)):
        value = amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        samples.append(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(samples))


def _source_metadata() -> MusanSourceMetadata:
    return MusanSourceMetadata(
        source_url="https://example.invalid/musan.tar.gz",
        source_revision="a" * 64,
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    )


def _write_musan_fixture(root: Path, files_per_type: int = 3) -> Path:
    musan = root / "musan"
    frequency = 180.0
    for noise_type in MUSAN_TYPES:
        for index in range(files_per_type):
            _write_wav(
                musan / noise_type / f"subtype_{noise_type}" / f"{index}.wav",
                frequency=frequency,
            )
            frequency += 37.0
    return musan


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_source_dev(root: Path, count: int = 2) -> tuple[Path, str]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        audio = root / "vivos_dev" / f"spk_{index}" / f"utt_{index}.wav"
        _write_wav(audio, frequency=500.0 + index * 91.0, amplitude=0.9)
        text = f"câu kiểm thử {index}"
        rows.append(
            {
                "audio": audio.resolve().as_posix(),
                "text": text,
                "utt_id": f"utt_{index}",
                "source_utt_id": f"utt_{index}",
                "speaker_id": f"spk_{index}",
                "dataset": "vivos",
                "split": "dev",
                "audio_sha256": sha256_file(audio),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    manifest = root / "manifests" / "vivos_dev.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(_jsonl_bytes(rows))
    return manifest, sha256_file(manifest)


def _noise_paths(root: Path) -> tuple[Path, Path]:
    return root / "noise_manifests", root / "protocol"


def _build_and_write_noise(root: Path) -> Path:
    manifest_dir, protocol_dir = _noise_paths(root)
    payloads = build_noise_protocol_outputs(
        _write_musan_fixture(root),
        manifest_dir=manifest_dir,
        protocol_dir=protocol_dir,
        source=_source_metadata(),
        seed=42,
    )
    write_locked_noise_outputs(payloads)
    return protocol_dir / "noise_split_lock.json"


def _noisy_config(root: Path, noise_lock: Path) -> NoisyDevConfig:
    source_manifest, source_sha = _write_source_dev(root)
    return NoisyDevConfig(
        source_dev_manifest=source_manifest,
        source_dev_sha256=source_sha,
        noise_split_lock=noise_lock,
        output_manifest=root / "derived_manifests" / "vivos_dev_noisy.jsonl",
        output_audio_dir=root / "derived_audio" / "noisy_dev",
        protocol_lock=root / "protocol" / "noisy_dev_lock.json",
        protocol_audit=root / "protocol" / "noisy_dev_audit.csv",
        snrs=(5.0, 0.0),
        seed=42,
    )


class NoiseRegistryProtocolTests(unittest.TestCase):
    def test_registry_is_content_hashed_typed_and_file_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            musan = _write_musan_fixture(root)
            manifest_dir, protocol_dir = _noise_paths(root)
            first = build_noise_protocol_outputs(
                musan,
                manifest_dir=manifest_dir,
                protocol_dir=protocol_dir,
                source=_source_metadata(),
                seed=42,
            )
            second = build_noise_protocol_outputs(
                musan,
                manifest_dir=manifest_dir,
                protocol_dir=protocol_dir,
                source=_source_metadata(),
                seed=42,
            )
            self.assertEqual(first, second)
            self.assertEqual(write_locked_noise_outputs(first), "written")
            self.assertEqual(
                write_locked_noise_outputs(second), "verified_existing"
            )

            verified = verify_noise_split_lock(
                protocol_dir / "noise_split_lock.json", verify_audio=True
            )
            rows = verified["registry_rows"]
            self.assertEqual(len(rows), 9)
            self.assertTrue(
                all(tuple(row) == NOISE_REGISTRY_COLUMNS for row in rows)
            )
            by_split = {
                split: {row["audio_sha256"] for row in rows if row["split"] == split}
                for split in ("train", "dev", "test")
            }
            self.assertFalse(by_split["train"] & by_split["dev"])
            self.assertFalse(by_split["train"] & by_split["test"])
            self.assertFalse(by_split["dev"] & by_split["test"])
            for split in by_split:
                self.assertEqual(
                    {row["noise_type"] for row in rows if row["split"] == split},
                    set(MUSAN_TYPES),
                )

    def test_duplicate_audio_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            musan = _write_musan_fixture(root)
            original = musan / "music" / "subtype_music" / "0.wav"
            duplicate = musan / "noise" / "subtype_noise" / "duplicate.wav"
            duplicate.write_bytes(original.read_bytes())
            with self.assertRaisesRegex(NoiseProtocolError, "duplicate audio_sha256"):
                build_noise_protocol_outputs(
                    musan,
                    manifest_dir=root / "manifests",
                    protocol_dir=root / "protocol",
                    source=_source_metadata(),
                )

    def test_locked_outputs_refuse_changes_and_restore_on_commit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payloads = build_noise_protocol_outputs(
                _write_musan_fixture(root),
                manifest_dir=root / "manifests",
                protocol_dir=root / "protocol",
                source=_source_metadata(),
            )
            write_locked_noise_outputs(payloads)
            before = {path: path.read_bytes() for path in payloads}
            changed = {path: value + b"changed" for path, value in payloads.items()}
            with self.assertRaisesRegex(NoiseProtocolError, "refusing to overwrite"):
                write_locked_noise_outputs(changed)

            original_rename = Path.rename
            temporary_commits = 0

            def fail_second_temporary_commit(source: Path, target: Path) -> Path:
                nonlocal temporary_commits
                if source.name.endswith(".tmp"):
                    temporary_commits += 1
                    if temporary_commits == 2:
                        raise OSError("simulated commit failure")
                return original_rename(source, target)

            with patch.object(Path, "rename", new=fail_second_temporary_commit):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_locked_noise_outputs(changed, overwrite=True)
            self.assertEqual({path: path.read_bytes() for path in payloads}, before)


class NoisyDevProtocolTests(unittest.TestCase):
    def test_noisy_dev_is_bound_measured_clipping_safe_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            noise_lock = _build_and_write_noise(root)
            config = _noisy_config(root, noise_lock)
            first = build_noisy_dev_benchmark(config)
            self.assertEqual(first["status"], "written")
            self.assertEqual(first["rows"], 6)
            second = build_noisy_dev_benchmark(config)
            self.assertEqual(second["status"], "verified_existing")

            rows = _read_jsonl(config.output_manifest)
            self.assertTrue(all(tuple(row) == NOISY_DEV_COLUMNS for row in rows))
            noisy = [row for row in rows if row["condition"] == "noisy"]
            self.assertEqual(len(noisy), 4)
            self.assertTrue(all(row["noise_split"] == "dev" for row in noisy))
            self.assertTrue(
                all(
                    abs(float(row["target_snr_db"]) - float(row["measured_snr_db"]))
                    <= 1e-6
                    for row in noisy
                )
            )
            self.assertTrue(all(int(row["clipped_sample_count"]) == 0 for row in noisy))
            self.assertTrue(
                all(sha256_file(row["audio_path"]) == row["audio_sha256"] for row in noisy)
            )
            lock = json.loads(config.protocol_lock.read_text(encoding="utf-8"))
            self.assertTrue(lock["selection_eligible"])
            self.assertFalse(lock["final_test_eligible"])
            self.assertEqual(lock["noise"]["partition"], "dev")

    def test_noisy_dev_refuses_source_hash_and_existing_artifact_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            noise_lock = _build_and_write_noise(root)
            config = _noisy_config(root, noise_lock)
            wrong_hash = NoisyDevConfig(
                **{**config.__dict__, "source_dev_sha256": "f" * 64}
            )
            with self.assertRaisesRegex(NoiseProtocolError, "does not match config"):
                build_noisy_dev_benchmark(wrong_hash)

            build_noisy_dev_benchmark(config)
            config.output_manifest.write_bytes(
                config.output_manifest.read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(NoiseProtocolError, "refusing to overwrite"):
                build_noisy_dev_benchmark(config)

    def test_noisy_dev_transaction_restores_previous_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            noise_lock = _build_and_write_noise(root)
            config = _noisy_config(root, noise_lock)
            build_noisy_dev_benchmark(config)
            before_files = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
                and (
                    path == config.output_manifest
                    or path == config.protocol_lock
                    or path == config.protocol_audit
                    or config.output_audio_dir in path.parents
                )
            }
            changed = NoisyDevConfig(**{**config.__dict__, "seed": 43})
            original_rename = Path.rename
            temporary_commits = 0

            def fail_second_temporary_commit(source: Path, target: Path) -> Path:
                nonlocal temporary_commits
                if source.name.endswith(".tmp"):
                    temporary_commits += 1
                    if temporary_commits == 2:
                        raise OSError("simulated derived commit failure")
                return original_rename(source, target)

            with patch.object(Path, "rename", new=fail_second_temporary_commit):
                with self.assertRaisesRegex(OSError, "simulated derived"):
                    build_noisy_dev_benchmark(changed, overwrite=True)
            after_files = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
                and (
                    path == config.output_manifest
                    or path == config.protocol_lock
                    or path == config.protocol_audit
                    or config.output_audio_dir in path.parents
                )
            }
            self.assertEqual(after_files, before_files)
            self.assertEqual(
                build_noisy_dev_benchmark(config)["status"], "verified_existing"
            )


if __name__ == "__main__":
    unittest.main()
