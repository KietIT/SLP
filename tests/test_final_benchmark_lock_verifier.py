from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.vitonesr.final_benchmark as final_module
from src.vitonesr.final_benchmark import (
    FINAL_BENCHMARK_ALGORITHM,
    FINAL_BENCHMARK_COLUMNS,
    FINAL_BENCHMARK_VERSION,
    FINAL_PEAK_LIMIT,
    FINAL_ROW_COUNT,
    FINAL_SAMPLE_RATE,
    FINAL_SEED,
    FINAL_SNRS,
    FINAL_SOURCE_COUNT,
    FinalBenchmarkError,
    sha256_file,
    verify_final_benchmark_lock,
)
from src.vitonesr.phat.protocol import canonical_sha256


def _h(character: str) -> str:
    return character * 64


class FinalBenchmarkLockVerifierTests(unittest.TestCase):
    def test_metadata_authorization_binds_data_protocols_without_manifest_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audit_path = root / "protocol/final_audit.csv"
            audit_path.parent.mkdir(parents=True)
            with audit_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["check_id", "status"], lineterminator="\n")
                writer.writeheader()
                writer.writerow({"check_id": "complete", "status": "PASS"})
            builder = {
                "algorithm": FINAL_BENCHMARK_ALGORITHM,
                "seed": FINAL_SEED,
                "snrs_db": list(FINAL_SNRS),
                "sample_rate": FINAL_SAMPLE_RATE,
                "peak_limit": FINAL_PEAK_LIMIT,
                "include_clean": True,
                "expected_source_count": FINAL_SOURCE_COUNT,
                "expected_row_count": FINAL_ROW_COUNT,
                "audio_container": "WAV",
                "audio_subtype": "PCM_16",
                "input_sample_rate_policy": "require_exact",
                "channel_policy": "mean_to_mono",
                "snr_measurement": "component_power_after_anti_clip_before_pcm16",
                "clipping_measurement": "pre_scale_over_1_and_stored_full_scale",
                "source_partition": "vivos_test_locked",
                "noise_partition": "musan_test",
            }
            registry_rows = [
                {
                    "noise_id": f"musan-test-{index}",
                    "audio_sha256": _h(character),
                    "noise_type": noise_type,
                    "split": "test",
                }
                for index, (character, noise_type) in enumerate(
                    (("a", "music"), ("b", "noise"), ("c", "speech")),
                    start=1,
                )
            ]
            noise_inventory_sha256 = canonical_sha256(
                [
                    {
                        "noise_id": row["noise_id"],
                        "audio_sha256": row["audio_sha256"],
                        "noise_type": row["noise_type"],
                    }
                    for row in registry_rows
                ]
            )
            lock = {
                "protocol_version": FINAL_BENCHMARK_VERSION,
                "status": "LOCKED",
                "selection_eligible": False,
                "final_test_eligible": True,
                "split_lock_sha256": _h("1"),
                "source_test_manifest_sha256": _h("3"),
                "noise_split_lock_sha256": _h("6"),
                "builder": {
                    "params": builder,
                    "params_sha256": canonical_sha256(builder),
                },
                "schema": list(FINAL_BENCHMARK_COLUMNS),
                "output": {
                    "manifest": "benchmark/final.jsonl",
                    "manifest_sha256": _h("7"),
                    "row_count": 2300,
                    "clean_row_count": 460,
                    "noisy_row_count": 1840,
                    "audio_dir": "derived/final_audio",
                    "audio_hashes_recorded": True,
                    "audio_inventory_sha256": _h("8"),
                },
                "source_test": {
                    "manifest": "manifests/test_locked.jsonl",
                    "manifest_sha256": _h("3"),
                    "utterance_count": 460,
                    "audio_text_inventory_sha256": _h("9"),
                },
                "noise": {
                    "split_lock": "protocol/noise.json",
                    "split_lock_sha256": _h("6"),
                    "registry_manifest_sha256": _h("a"),
                    "test_manifest": "manifests/musan_test.jsonl",
                    "test_manifest_sha256": _h("b"),
                    "partition": "test",
                    "file_count": 3,
                    "audio_inventory_sha256": noise_inventory_sha256,
                },
                "audit": {
                    "path": "protocol/final_audit.csv",
                    "sha256": sha256_file(audit_path),
                    "checks": 1,
                    "failed_checks": 0,
                },
            }
            lock_path = root / "protocol/final.json"
            lock_path.write_text(
                json.dumps(lock, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            noise = {
                "lock": {
                    "registry": {"manifest_sha256": _h("a")},
                    "splits": {
                        "test": {
                            "manifest": "manifests/musan_test.jsonl",
                            "manifest_sha256": _h("b"),
                            "file_count": 3,
                        }
                    },
                },
                "registry_rows": registry_rows,
            }
            kwargs = {
                "expected_lock_sha256": sha256_file(lock_path),
                "expected_manifest": root / "benchmark/final.jsonl",
                "expected_manifest_sha256": _h("7"),
                "expected_rows": 2300,
                "split_lock_sha256": _h("1"),
                "source_test_manifest_sha256": _h("3"),
                "noise_split_lock_sha256": _h("6"),
                "noise_integrity": noise,
            }
            # No final manifest is created: metadata authorization must not open it.
            with patch.object(final_module, "ROOT", root):
                evidence = verify_final_benchmark_lock(lock_path, **kwargs)
                self.assertEqual(evidence["row_count"], 2300)
                self.assertFalse((root / "benchmark/final.jsonl").exists())
                with self.assertRaisesRegex(FinalBenchmarkError, "another split_lock"):
                    verify_final_benchmark_lock(
                        lock_path, **{**kwargs, "split_lock_sha256": _h("d")}
                    )
                with self.assertRaisesRegex(FinalBenchmarkError, "has changed"):
                    verify_final_benchmark_lock(
                        lock_path,
                        **{**kwargs, "expected_lock_sha256": _h("e")},
                    )


if __name__ == "__main__":
    unittest.main()
