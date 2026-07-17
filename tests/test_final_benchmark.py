from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import src.vitonesr.final_benchmark as final_benchmark_module
from src.vitonesr.final_benchmark import (
    FINAL_BENCHMARK_COLUMNS,
    FINAL_PEAK_LIMIT,
    FINAL_ROW_COUNT,
    FINAL_SOURCE_COUNT,
    FinalBenchmarkConfig,
    FinalBenchmarkError,
    build_final_benchmark,
    sha256_file,
    verify_portable_final_benchmark_bundle,
)
from src.vitonesr.noise_protocol import (
    MUSAN_TYPES,
    MusanSourceMetadata,
    build_noise_protocol_outputs,
    verify_noise_split_lock,
    write_locked_noise_outputs,
)


def _write_wav(
    path: Path,
    *,
    frequency: float,
    amplitude: float = 0.4,
    seconds: float = 0.035,
    sample_rate: int = 16000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(int(seconds * sample_rate)):
        value = amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        frames.append(struct.pack("<h", int(value * 32767)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(frames))


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


def _write_source(root: Path, count: int = 2) -> Path:
    rows: list[dict[str, object]] = []
    for index in range(count):
        audio = root / "vivos_test" / f"spk_{index}" / f"utt_{index}.wav"
        _write_wav(audio, frequency=410.0 + index * 71.0, amplitude=0.75)
        text = f"câu unseen {index}"
        rows.append(
            {
                "audio": audio.resolve().as_posix(),
                "text": text,
                "utt_id": f"utt_{index}",
                "source_utt_id": f"utt_{index}",
                "speaker_id": f"spk_{index}",
                "dataset": "vivos",
                "split": "test",
                "condition": "clean",
                "snr": "clean",
                "noise_type": "clean",
                "audio_sha256": sha256_file(audio),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    manifest = root / "manifests" / "vivos_test_locked.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(_jsonl_bytes(rows))
    return manifest


def _write_noise_protocol(root: Path) -> Path:
    musan = root / "musan"
    frequency = 120.0
    for noise_type in MUSAN_TYPES:
        for index in range(3):
            _write_wav(
                musan / noise_type / f"subtype_{noise_type}" / f"{index}.wav",
                frequency=frequency,
                amplitude=0.3,
                seconds=0.07,
            )
            frequency += 43.0
    payloads = build_noise_protocol_outputs(
        musan,
        manifest_dir=root / "noise_manifests",
        protocol_dir=root / "protocol",
        source=MusanSourceMetadata(
            source_url="https://example.invalid/musan.tar.gz",
            source_revision="a" * 64,
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
        ),
        seed=42,
    )
    write_locked_noise_outputs(payloads)
    return root / "protocol" / "noise_split_lock.json"


def _fixture(root: Path) -> tuple[FinalBenchmarkConfig, list[str]]:
    source = _write_source(root)
    noise_lock = _write_noise_protocol(root)
    split = root / "protocol" / "split_lock.json"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("{}\n", encoding="utf-8")
    config = FinalBenchmarkConfig(
        split_lock=split,
        noise_split_lock=noise_lock,
        source_test_manifest=source,
        output_manifest=root / "outputs" / "final_benchmark.jsonl",
        output_audio_dir=root / "derived" / "final_audio",
        protocol_lock=root / "outputs" / "final_benchmark_lock.json",
        protocol_audit=root / "outputs" / "final_benchmark_audit.csv",
        expected_source_count=2,
    )
    return config, []


def _verifiers(config: FinalBenchmarkConfig, access: list[str]):
    split_hash = sha256_file(config.split_lock)

    def source(path, *, split_lock_path):
        access.append("source")
        return {
            "manifest_sha256": sha256_file(path),
            "split_lock_sha256": sha256_file(split_lock_path),
            "utterance_count": len(_read_jsonl(path)),
        }

    def noise(path):
        access.append("noise")
        return verify_noise_split_lock(path, verify_audio=True)

    return source, noise


def _build(config: FinalBenchmarkConfig, access: list[str], **kwargs):
    source, noise = _verifiers(config, access)
    fixture_root = config.split_lock.parents[1]
    with (
        patch.object(final_benchmark_module, "ROOT", fixture_root),
        patch.object(final_benchmark_module, "FINAL_SOURCE_COUNT", 2),
        patch.object(final_benchmark_module, "FINAL_ROW_COUNT", 10),
    ):
        return build_final_benchmark(
            config,
            source_verifier=source,
            noise_verifier=noise,
            **kwargs,
        )


def _transaction_snapshot(config: FinalBenchmarkConfig) -> dict[str, bytes]:
    root = config.protocol_lock.parents[1]
    paths = [config.output_manifest, config.protocol_lock, config.protocol_audit]
    paths.extend(path for path in config.output_audio_dir.rglob("*") if path.is_file())
    return {
        path.resolve().relative_to(root.resolve()).as_posix(): path.read_bytes()
        for path in paths
    }


class FinalBenchmarkTests(unittest.TestCase):
    def test_locked_data_build_is_method_independent_and_deterministic(self) -> None:
        self.assertEqual(FINAL_SOURCE_COUNT, 460)
        self.assertEqual(FINAL_ROW_COUNT, 2300)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, access = _fixture(root)
            result = _build(config, access)
            self.assertEqual(result["status"], "written")
            self.assertEqual(result["rows"], 10)
            self.assertEqual(access, ["source", "noise"])

            rows = _read_jsonl(config.output_manifest)
            self.assertTrue(all(tuple(row) == FINAL_BENCHMARK_COLUMNS for row in rows))
            self.assertTrue(
                all(
                    not Path(str(row["audio_path"])).is_absolute()
                    and not Path(str(row["clean_path"])).is_absolute()
                    and (
                        not row["noise_path"]
                        or not Path(str(row["noise_path"])).is_absolute()
                    )
                    for row in rows
                )
            )
            self.assertEqual(sum(row["condition"] == "clean" for row in rows), 2)
            self.assertEqual(sum(row["condition"] == "noisy" for row in rows), 8)
            self.assertEqual(
                {str(row["snr"]) for row in rows}, {"clean", "20", "10", "5", "0"}
            )
            noisy = [row for row in rows if row["condition"] == "noisy"]
            clean = [row for row in rows if row["condition"] == "clean"]
            self.assertEqual(len({row["audio_path"] for row in noisy}), len(noisy))
            self.assertEqual(
                len(list(config.output_audio_dir.rglob("*.wav"))), len(rows)
            )
            self.assertTrue(
                all(
                    Path(str(row["audio_path"])).parts[-2] == "clean"
                    and sha256_file(root / str(row["audio_path"]))
                    == row["clean_audio_sha256"]
                    for row in clean
                )
            )
            self.assertTrue(all(row["noise_split"] == "test" for row in noisy))
            self.assertTrue(all(row["selection_eligible"] is False for row in rows))
            self.assertTrue(all(row["final_test_eligible"] is True for row in rows))
            locked = json.loads(config.protocol_lock.read_text(encoding="utf-8"))
            locked_paths = (
                locked["source_test"]["manifest"],
                locked["noise"]["split_lock"],
                locked["noise"]["test_manifest"],
                locked["output"]["manifest"],
                locked["output"]["audio_dir"],
                locked["audit"]["path"],
            )
            self.assertTrue(
                all(
                    not Path(value).is_absolute() and ".." not in Path(value).parts
                    for value in locked_paths
                )
            )
            registry = verify_noise_split_lock(
                config.noise_split_lock, verify_audio=True
            )["registry_rows"]
            forbidden_ids = {
                row["noise_id"] for row in registry if row["split"] != "test"
            }
            forbidden_hashes = {
                row["audio_sha256"] for row in registry if row["split"] != "test"
            }
            self.assertFalse({row["noise_id"] for row in noisy} & forbidden_ids)
            self.assertFalse(
                {row["noise_audio_sha256"] for row in noisy} & forbidden_hashes
            )
            lock_hash = sha256_file(config.protocol_lock)
            manifest_hash = sha256_file(config.output_manifest)
            locked = json.loads(config.protocol_lock.read_text(encoding="utf-8"))
            self.assertEqual(locked["protocol_version"], "paper_v2_final_benchmark_v2")
            self.assertNotIn("decision_lock_sha256", locked)
            self.assertNotIn("method_lock_sha256", locked)
            self.assertNotIn("method_identity_sha256", locked)
            self.assertEqual(locked["output"]["manifest_sha256"], manifest_hash)
            self.assertEqual(sha256_file(config.protocol_lock), lock_hash)
            manifest_before = config.output_manifest.read_bytes()
            inventory_before = sorted(
                (path.relative_to(config.output_audio_dir).as_posix(), sha256_file(path))
                for path in config.output_audio_dir.rglob("*.wav")
            )
            self.assertEqual(_build(config, [])["status"], "verified_existing")

            shutil.rmtree(config.output_audio_dir)
            config.output_manifest.unlink()
            config.protocol_lock.unlink()
            config.protocol_audit.unlink()
            self.assertEqual(_build(config, [])["status"], "written")
            self.assertEqual(config.output_manifest.read_bytes(), manifest_before)
            self.assertEqual(
                sorted(
                    (
                        path.relative_to(config.output_audio_dir).as_posix(),
                        sha256_file(path),
                    )
                    for path in config.output_audio_dir.rglob("*.wav")
                ),
                inventory_before,
            )
            shutil.rmtree(root / "vivos_test")
            shutil.rmtree(root / "musan")
            self.assertFalse((root / "vivos_test").exists())
            self.assertFalse((root / "musan").exists())
            with (
                patch.object(final_benchmark_module, "ROOT", root),
                patch.object(final_benchmark_module, "FINAL_SOURCE_COUNT", 2),
                patch.object(final_benchmark_module, "FINAL_ROW_COUNT", 10),
            ):
                portable = verify_portable_final_benchmark_bundle(
                    config.protocol_lock,
                    expected_lock_sha256=sha256_file(config.protocol_lock),
                    expected_manifest=config.output_manifest,
                    expected_manifest_sha256=sha256_file(config.output_manifest),
                    expected_rows=10,
                )
            self.assertEqual(portable["row_count"], 10)
            copied_rows = _read_jsonl(config.output_manifest)
            self.assertTrue(all(row["noise_path"] == "" for row in copied_rows))
            self.assertTrue(
                all(
                    (root / str(row["audio_path"])).resolve().is_relative_to(
                        config.output_audio_dir.resolve()
                    )
                    and (root / str(row["clean_path"])).resolve().is_relative_to(
                        config.output_audio_dir.resolve()
                    )
                    for row in copied_rows
                )
            )
            for row in copied_rows:
                with wave.open(str(root / str(row["audio_path"])), "rb") as handle:
                    self.assertGreater(handle.getnframes(), 0)

    def test_failed_split_verification_cannot_touch_noise_or_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = _fixture(Path(temporary_directory))
            touched: list[str] = []

            def rejected(*_args, **_kwargs):
                touched.append("source")
                raise FinalBenchmarkError("split lock rejected")

            def forbidden(*_args, **_kwargs):
                raise AssertionError("noise verifier ran after rejected split lock")

            with (
                patch.object(final_benchmark_module, "ROOT", config.split_lock.parents[1]),
                patch.object(final_benchmark_module, "FINAL_SOURCE_COUNT", 2),
                patch.object(final_benchmark_module, "FINAL_ROW_COUNT", 10),
            ):
                with self.assertRaisesRegex(FinalBenchmarkError, "split lock rejected"):
                    build_final_benchmark(
                        config,
                        source_verifier=rejected,
                        noise_verifier=forbidden,
                    )
            self.assertEqual(touched, ["source"])
            self.assertFalse(config.output_manifest.exists())
            self.assertFalse(config.output_audio_dir.exists())

    def test_tampering_and_partial_outputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, access = _fixture(Path(temporary_directory))
            _build(config, access)
            config.output_manifest.write_bytes(config.output_manifest.read_bytes() + b"\n")
            with self.assertRaisesRegex(FinalBenchmarkError, "refusing to overwrite"):
                _build(config, [])

        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = _fixture(Path(temporary_directory))
            config.output_manifest.parent.mkdir(parents=True, exist_ok=True)
            config.output_manifest.write_text("partial\n", encoding="utf-8")
            with self.assertRaisesRegex(FinalBenchmarkError, "Partial final benchmark"):
                _build(config, [])

        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = _fixture(Path(temporary_directory))
            verified = verify_noise_split_lock(config.noise_split_lock, verify_audio=True)
            test_noise = next(
                row for row in verified["registry_rows"] if row["split"] == "test"
            )
            Path(test_noise["audio"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(Exception, "audio|Audio"):
                _build(config, [])

    def test_portable_bundle_authorizer_rejects_semantic_lock_tampering(self) -> None:
        cases = {
            "builder_params": "builder contract",
            "schema": "schema lock",
            "audit": "audit contains failed",
            "noise_inventory": "MUSAN-test provenance",
            "method_binding": "method/decision independent",
            "benchmark_lock_hash": "SHA-256 has changed",
        }
        for case, message in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                config, _ = _fixture(root)
                _build(config, [])
                lock = json.loads(config.protocol_lock.read_text(encoding="utf-8"))
                original_lock_hash = sha256_file(config.protocol_lock)
                if case == "builder_params":
                    lock["builder"]["params"]["seed"] = 999
                elif case == "schema":
                    lock["schema"] = lock["schema"][:-1]
                elif case == "audit":
                    with config.protocol_audit.open(
                        "r", encoding="utf-8-sig", newline=""
                    ) as handle:
                        rows = list(csv.DictReader(handle))
                    rows[0]["status"] = "FAIL"
                    with config.protocol_audit.open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=list(rows[0]), lineterminator="\n"
                        )
                        writer.writeheader()
                        writer.writerows(rows)
                    lock["audit"]["sha256"] = sha256_file(config.protocol_audit)
                elif case == "noise_inventory":
                    lock["noise"]["audio_inventory_sha256"] = "invalid"
                elif case == "method_binding":
                    lock["method_lock_sha256"] = "f" * 64
                elif case == "benchmark_lock_hash":
                    config.protocol_lock.write_bytes(
                        config.protocol_lock.read_bytes() + b" "
                    )

                if case != "benchmark_lock_hash":
                    config.protocol_lock.write_text(
                        json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                expected_lock_hash = (
                    original_lock_hash
                    if case == "benchmark_lock_hash"
                    else sha256_file(config.protocol_lock)
                )
                with (
                    patch.object(final_benchmark_module, "ROOT", root),
                    patch.object(final_benchmark_module, "FINAL_SOURCE_COUNT", 2),
                    patch.object(final_benchmark_module, "FINAL_ROW_COUNT", 10),
                    self.assertRaisesRegex(FinalBenchmarkError, message),
                ):
                    verify_portable_final_benchmark_bundle(
                        config.protocol_lock,
                        expected_lock_sha256=expected_lock_hash,
                        expected_manifest=config.output_manifest,
                        expected_manifest_sha256=str(
                            lock["output"]["manifest_sha256"]
                        ),
                        expected_rows=10,
                    )

    def test_arbitrary_method_files_do_not_change_benchmark_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, _ = _fixture(root)
            _build(config, [])
            before = _transaction_snapshot(config)
            (root / "protocol" / "best_lambda_decision.json").write_text(
                '{"lambda":0.1}\n', encoding="utf-8"
            )
            (root / "protocol" / "method_lock.json").write_text(
                '{"method":"changed"}\n', encoding="utf-8"
            )
            self.assertEqual(_build(config, [])["status"], "verified_existing")
            self.assertEqual(_transaction_snapshot(config), before)

    def test_self_consistent_existing_output_tampering_fails_semantic_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = _fixture(Path(temporary_directory))
            _build(config, [])
            rows = _read_jsonl(config.output_manifest)[:-1]
            config.output_manifest.write_bytes(_jsonl_bytes(rows))
            lock = json.loads(config.protocol_lock.read_text(encoding="utf-8"))
            lock["output"]["manifest_sha256"] = sha256_file(config.output_manifest)
            lock["output"]["row_count"] = len(rows)
            lock["output"]["clean_row_count"] = sum(
                row["condition"] == "clean" for row in rows
            )
            lock["output"]["noisy_row_count"] = sum(
                row["condition"] == "noisy" for row in rows
            )
            lock["output"]["audio_inventory_sha256"] = (
                final_benchmark_module._canonical_sha256(
                    [
                        {
                            "utt_id": row["utt_id"],
                            "audio_sha256": row["audio_sha256"],
                        }
                        for row in rows
                    ]
                )
            )
            config.protocol_lock.write_text(
                json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FinalBenchmarkError, "refusing to overwrite"):
                _build(config, [])

        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = _fixture(Path(temporary_directory))
            _build(config, [])
            rows = _read_jsonl(config.output_manifest)
            rows[0]["transcript"] = "tampered but row count and hashes are coherent"
            config.output_manifest.write_bytes(_jsonl_bytes(rows))
            lock = json.loads(config.protocol_lock.read_text(encoding="utf-8"))
            lock["output"]["manifest_sha256"] = sha256_file(config.output_manifest)
            config.protocol_lock.write_text(
                json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FinalBenchmarkError, "refusing to overwrite"):
                _build(config, [])

    def test_formal_paths_peak_limit_and_output_overlap_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = _fixture(Path(temporary_directory))
            outside = config.split_lock.parents[1].parent / "outside_manifest.jsonl"
            nonportable = FinalBenchmarkConfig(
                **{**config.__dict__, "output_manifest": outside}
            )
            with self.assertRaisesRegex(FinalBenchmarkError, "inside the repository root"):
                _build(nonportable, [])

            changed_peak = FinalBenchmarkConfig(
                **{**config.__dict__, "peak_limit": 0.95}
            )
            with self.assertRaisesRegex(FinalBenchmarkError, "peak_limit=0.999"):
                _build(changed_peak, [])

            overlapping = FinalBenchmarkConfig(
                **{
                    **config.__dict__,
                    "output_manifest": config.output_audio_dir / "manifest.jsonl",
                }
            )
            with self.assertRaisesRegex(FinalBenchmarkError, "must not overlap"):
                _build(overlapping, [])

    def test_safe_stem_distinguishes_ids_that_sanitize_to_the_same_text(self) -> None:
        left = final_benchmark_module._safe_stem("speaker/a")
        right = final_benchmark_module._safe_stem("speaker_a")
        self.assertNotEqual(left, right)
        self.assertTrue(left.startswith("speaker_a_"))
        self.assertTrue(right.startswith("speaker_a_"))

    def test_overwrite_transaction_restores_previous_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = _fixture(Path(temporary_directory))
            _build(config, [])
            before = _transaction_snapshot(config)
            changed = FinalBenchmarkConfig(
                **{**config.__dict__, "peak_limit": FINAL_PEAK_LIMIT}
            )
            original_rename = Path.rename
            commits = 0

            def fail_second_commit(source: Path, target: Path) -> Path:
                nonlocal commits
                if source.name.endswith(".tmp"):
                    commits += 1
                    if commits == 2:
                        raise OSError("simulated final benchmark commit failure")
                return original_rename(source, target)

            with patch.object(Path, "rename", new=fail_second_commit):
                with self.assertRaisesRegex(OSError, "simulated final benchmark"):
                    _build(changed, [], overwrite=True)
            self.assertEqual(_transaction_snapshot(config), before)
            self.assertEqual(_build(config, [])["status"], "verified_existing")


if __name__ == "__main__":
    unittest.main()
