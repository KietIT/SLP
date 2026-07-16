from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.make_vivos_manifest import (
    VivosProtocolError,
    build_protocol_outputs,
    locate_official_root,
    validate_replica_split_consistency,
    write_locked_outputs,
)


def _write_official_fixture(root: Path) -> Path:
    official = root / "vivos"
    train_prompts: list[str] = []
    for speaker_number in range(1, 6):
        speaker = f"VIVOSSPK{speaker_number:02d}"
        speaker_dir = official / "train" / "waves" / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for utterance_number in range(1, 3):
            utt_id = f"{speaker}_R{utterance_number:03d}"
            train_prompts.append(f"{utt_id} câu train {speaker_number} {utterance_number}")
            (speaker_dir / f"{utt_id}.wav").write_bytes(f"audio:{utt_id}".encode())
    train_prompt_path = official / "train" / "prompts.txt"
    train_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    train_prompt_path.write_text("\n".join(reversed(train_prompts)) + "\n", encoding="utf-8")

    test_prompts: list[str] = []
    for speaker_number in range(1, 3):
        speaker = f"VIVOSDEV{speaker_number:02d}"
        speaker_dir = official / "test" / "waves" / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        utt_id = f"{speaker}_001"
        test_prompts.append(f"{utt_id} câu test {speaker_number}")
        (speaker_dir / f"{utt_id}.wav").write_bytes(f"audio:{utt_id}".encode())
    test_prompt_path = official / "test" / "prompts.txt"
    test_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    test_prompt_path.write_text("\n".join(test_prompts) + "\n", encoding="utf-8")
    return official


def _jsonl(payload: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def _write_legacy_benchmark(path: Path, source_utt_id: str) -> Path:
    fieldnames = [
        "utt_id",
        "dataset",
        "split",
        "condition",
        "snr",
        "source_utt_id",
    ]
    rows = []
    for snr in ("clean", "20", "10", "5", "0"):
        rows.append(
            {
                "utt_id": f"{source_utt_id}_{'clean' if snr == 'clean' else 'snr' + snr}",
                "dataset": "vivos",
                "split": "test",
                "condition": "clean" if snr == "clean" else "noisy",
                "snr": snr,
                "source_utt_id": source_utt_id,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_fixture_outputs(
    official: Path,
    *,
    manifest_dir: Path,
    protocol_dir: Path,
    seed: int = 42,
    dev_speaker_fraction: float = 0.20,
) -> dict[Path, bytes]:
    legacy_benchmark = _write_legacy_benchmark(
        official.parent / "legacy_benchmark.csv", "VIVOSDEV01_001"
    )
    return build_protocol_outputs(
        official,
        manifest_dir=manifest_dir,
        protocol_dir=protocol_dir,
        legacy_benchmark_manifest=legacy_benchmark,
        expected_legacy_exposed=1,
        seed=seed,
        dev_speaker_fraction=dev_speaker_fraction,
    )


class VivosProtocolTests(unittest.TestCase):
    def test_speaker_disjoint_split_and_locked_test_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            official = _write_official_fixture(root)
            manifest_dir = root / "manifests"
            protocol_dir = root / "protocol"
            first = _build_fixture_outputs(
                official,
                manifest_dir=manifest_dir,
                protocol_dir=protocol_dir,
                seed=42,
                dev_speaker_fraction=0.20,
            )
            second = _build_fixture_outputs(
                official,
                manifest_dir=manifest_dir,
                protocol_dir=protocol_dir,
                seed=42,
                dev_speaker_fraction=0.20,
            )
            self.assertEqual(first, second)

            train = _jsonl(first[manifest_dir / "vivos_train.jsonl"])
            dev = _jsonl(first[manifest_dir / "vivos_dev.jsonl"])
            test = _jsonl(first[manifest_dir / "vivos_test_locked.jsonl"])
            legacy_exposed = _jsonl(
                first[manifest_dir / "vivos_test_legacy_exposed.jsonl"]
            )
            self.assertEqual(len(train) + len(dev), 10)
            self.assertEqual(len(test), 1)
            self.assertEqual(len(legacy_exposed), 1)
            self.assertFalse(
                {row["speaker_id"] for row in train}
                & {row["speaker_id"] for row in dev}
            )
            self.assertTrue(all(row["official_split"] == "test" for row in test))
            self.assertTrue(all(row["split"] == "test" for row in test))
            self.assertEqual(
                {row["utt_id"] for row in test},
                {"VIVOSDEV02_001"},
            )
            self.assertEqual(
                {row["utt_id"] for row in legacy_exposed},
                {"VIVOSDEV01_001"},
            )
            lock = json.loads(first[protocol_dir / "split_lock.json"])
            self.assertEqual(lock["official_test"]["status"], "SEALED")
            self.assertFalse(lock["official_test"]["selection_eligible"])
            self.assertEqual(
                lock["official_test"]["legacy_exposed_utterance_count"], 1
            )
            self.assertEqual(
                lock["official_test"]["unseen_locked_utterance_count"], 1
            )
            self.assertEqual(lock["audit"]["failed_checks"], 0)
            exposure_registry = first[protocol_dir / "legacy_test_exposure.csv"]
            self.assertIn(b"VIVOSDEV01_001", exposure_registry)
            self.assertNotIn(b"VIVOSDEV02_001", exposure_registry)

            self.assertEqual(write_locked_outputs(first, overwrite=False), "written")
            self.assertEqual(
                write_locked_outputs(second, overwrite=False),
                "verified_existing",
            )

    def test_existing_lock_refuses_changed_source_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            official = _write_official_fixture(root)
            manifest_dir = root / "manifests"
            protocol_dir = root / "protocol"
            original = _build_fixture_outputs(
                official,
                manifest_dir=manifest_dir,
                protocol_dir=protocol_dir,
                seed=42,
                dev_speaker_fraction=0.20,
            )
            write_locked_outputs(original, overwrite=False)
            prompt_path = official / "train" / "prompts.txt"
            prompt_path.write_text(
                prompt_path.read_text(encoding="utf-8").replace("câu train", "câu đã đổi", 1),
                encoding="utf-8",
            )
            changed = _build_fixture_outputs(
                official,
                manifest_dir=manifest_dir,
                protocol_dir=protocol_dir,
                seed=42,
                dev_speaker_fraction=0.20,
            )
            with self.assertRaisesRegex(VivosProtocolError, "refusing to mutate"):
                write_locked_outputs(changed, overwrite=False)

    def test_legacy_exposure_evidence_must_match_locked_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            official = _write_official_fixture(root)
            legacy_benchmark = _write_legacy_benchmark(
                root / "legacy_benchmark.csv", "VIVOSDEV01_001"
            )
            with self.assertRaisesRegex(
                VivosProtocolError, "exposed-source count"
            ):
                build_protocol_outputs(
                    official,
                    manifest_dir=root / "manifests",
                    protocol_dir=root / "protocol",
                    legacy_benchmark_manifest=legacy_benchmark,
                    expected_legacy_exposed=2,
                    seed=42,
                    dev_speaker_fraction=0.20,
                )

            contents = legacy_benchmark.read_text(encoding="utf-8").replace(
                "VIVOSDEV01_001", "UNKNOWN_001"
            )
            legacy_benchmark.write_text(contents, encoding="utf-8")
            with self.assertRaisesRegex(
                VivosProtocolError, "not in official VIVOS test"
            ):
                build_protocol_outputs(
                    official,
                    manifest_dir=root / "manifests",
                    protocol_dir=root / "protocol",
                    legacy_benchmark_manifest=legacy_benchmark,
                    expected_legacy_exposed=1,
                    seed=42,
                    dev_speaker_fraction=0.20,
                )

    def test_cross_split_duplicate_audio_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            official = _write_official_fixture(root)
            train_audio = official / "train" / "waves" / "VIVOSSPK01" / "VIVOSSPK01_R001.wav"
            test_audio = official / "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_001.wav"
            test_audio.write_bytes(train_audio.read_bytes())
            with self.assertRaisesRegex(VivosProtocolError, "audit failed"):
                _build_fixture_outputs(
                    official,
                    manifest_dir=root / "manifests",
                    protocol_dir=root / "protocol",
                    seed=42,
                    dev_speaker_fraction=0.20,
                )

    def test_prompt_wav_inventory_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            official = _write_official_fixture(root)
            missing = official / "train" / "waves" / "VIVOSSPK01" / "VIVOSSPK01_R001.wav"
            missing.unlink()
            with self.assertRaisesRegex(VivosProtocolError, "inventory mismatch"):
                _build_fixture_outputs(
                    official,
                    manifest_dir=root / "manifests",
                    protocol_dir=root / "protocol",
                    seed=42,
                    dev_speaker_fraction=0.20,
                )

    def test_ambiguous_layout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_official_fixture(root / "first")
            _write_official_fixture(root / "second")
            with self.assertRaisesRegex(VivosProtocolError, "Ambiguous"):
                locate_official_root(root)

    def test_replica_family_cannot_cross_splits(self) -> None:
        validate_replica_split_consistency(
            [
                {"source_utt_id": "utt-1", "split": "dev", "snr": "clean"},
                {"source_utt_id": "utt-1", "split": "dev", "snr": "0"},
            ]
        )
        with self.assertRaisesRegex(VivosProtocolError, "cross split"):
            validate_replica_split_consistency(
                [
                    {"source_utt_id": "utt-1", "split": "dev"},
                    {"source_utt_id": "utt-1", "split": "test"},
                ]
            )

    def test_multi_file_commit_failure_restores_previous_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            official = _write_official_fixture(root)
            manifest_dir = root / "manifests"
            protocol_dir = root / "protocol"
            original = _build_fixture_outputs(
                official,
                manifest_dir=manifest_dir,
                protocol_dir=protocol_dir,
                seed=42,
                dev_speaker_fraction=0.20,
            )
            write_locked_outputs(original, overwrite=False)
            before = {path: path.read_bytes() for path in original}
            changed = {
                path: payload + b"changed"
                for path, payload in original.items()
            }
            original_rename = Path.rename
            temporary_commit_count = 0

            def fail_second_temporary_commit(
                source: Path, target: Path
            ) -> Path:
                nonlocal temporary_commit_count
                if source.name.endswith(".tmp"):
                    temporary_commit_count += 1
                    if temporary_commit_count == 2:
                        raise OSError("simulated commit failure")
                return original_rename(source, target)

            with patch.object(Path, "rename", new=fail_second_temporary_commit):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_locked_outputs(changed, overwrite=True)
            self.assertEqual(
                {path: path.read_bytes() for path in original},
                before,
            )
            leftovers = [
                path
                for path in root.rglob("*")
                if path.is_file()
                and (path.name.endswith(".tmp") or path.name.endswith(".backup"))
            ]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
