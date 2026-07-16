from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.vitonesr.phat.evaluation import (
    ABLATION_RESULT_COLUMNS,
    PREDICTION_COLUMNS,
    PREDICTION_PROVENANCE_VERSION,
    PREDICTION_RECOVERY_VERSION,
    _atomic_write_prediction,
    _prediction_csv_bytes,
    _prepare_prediction_output,
    _publish_prediction_progress,
    load_benchmark_rows,
    prediction_provenance_path,
    prediction_recovery_path,
    prediction_resume_path,
    write_ablation_results,
)
from src.vitonesr.phat.protocol import (
    canonical_sha256,
    selected_rows_sha256,
    sha256_file,
)
def _manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "utt_id": "dev_1_clean",
            "dataset": "vivos",
            "audio_path": "unused-1.wav",
            "snr": "clean",
            "noise_type": "clean",
            "ref": "xin chào",
            "evaluation_split": "dev",
        },
        {
            "utt_id": "dev_1_snr0",
            "dataset": "vivos",
            "audio_path": "unused-2.wav",
            "snr": "0",
            "noise_type": "speech",
            "ref": "xin chào",
            "evaluation_split": "dev",
        },
    ]


def _prediction_rows() -> list[dict[str, str]]:
    rows = []
    for index, manifest in enumerate(_manifest_rows(), start=1):
        rows.append(
            {
                "utt_id": manifest["utt_id"],
                "dataset": manifest["dataset"],
                "model": "phowhisper",
                "model_size": "base",
                "train_type": "tone_aware_lora",
                "lambda": "0.05",
                "seed": "42",
                "snr": manifest["snr"],
                "noise_type": manifest["noise_type"],
                "ref": manifest["ref"],
                "hyp": f"hypothesis {index}",
            }
        )
    return rows


def _identity() -> dict[str, str]:
    return {
        "resume_contract_sha256": "1" * 64,
        "config_identity_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "selected_rows_sha256": "4" * 64,
        "checkpoint_sha256": "5" * 64,
        "resolved_config_sha256": "6" * 64,
        "training_contract_sha256": "7" * 64,
    }


class EvaluationResumeTests(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, Path, Path]:
        prediction = root / "prediction.csv"
        return (
            prediction,
            prediction_provenance_path(prediction),
            prediction_resume_path(prediction),
            prediction_recovery_path(prediction),
        )

    def test_partial_progress_resumes_only_with_exact_identity_and_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            prediction, provenance, state, receipt = paths
            _publish_prediction_progress(
                prediction_path=prediction,
                resume_path=state,
                recovery_path=receipt,
                rows=_prediction_rows()[:1],
                previous_rows=[],
                identity=_identity(),
            )
            self.assertTrue(state.is_file())
            self.assertFalse(receipt.exists())
            prefix, completed = _prepare_prediction_output(
                prediction_path=prediction,
                provenance_path=provenance,
                resume_state_path=state,
                recovery_path=receipt,
                manifest_rows=_manifest_rows(),
                train_type="tone_aware_lora",
                lambda_tone=0.05,
                seed=42,
                identity=_identity(),
                resume=True,
                overwrite=False,
            )
            self.assertEqual(len(prefix), 1)
            self.assertFalse(completed)

            changed = dict(_identity())
            changed["config_identity_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "config_identity_sha256"):
                _prepare_prediction_output(
                    prediction_path=prediction,
                    provenance_path=provenance,
                    resume_state_path=state,
                    recovery_path=receipt,
                    manifest_rows=_manifest_rows(),
                    train_type="tone_aware_lora",
                    lambda_tone=0.05,
                    seed=42,
                    identity=changed,
                    resume=True,
                    overwrite=False,
                )

    def test_csv_before_state_crash_recovers_from_receipt_and_rejects_tamper(self) -> None:
        for tamper in (False, True):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                prediction, provenance, state, receipt = self._paths(Path(tmp))
                rows = _prediction_rows()
                target_sha = sha256_file_bytes(_prediction_csv_bytes(rows))
                receipt.write_text(
                    json.dumps(
                        {
                            "recovery_version": PREDICTION_RECOVERY_VERSION,
                            **_identity(),
                            "completed_rows": len(rows),
                            "prediction_sha256": target_sha,
                            "previous_completed_rows": 0,
                            "previous_prediction_sha256": "",
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _atomic_write_prediction(prediction, rows)
                if tamper:
                    with prediction.open("r", encoding="utf-8", newline="") as handle:
                        changed = list(csv.DictReader(handle))
                    changed[-1]["hyp"] = "tampered"
                    _atomic_write_prediction(prediction, changed)
                    with self.assertRaisesRegex(ValueError, "possible tamper"):
                        _prepare_prediction_output(
                            prediction_path=prediction,
                            provenance_path=provenance,
                            resume_state_path=state,
                            recovery_path=receipt,
                            manifest_rows=_manifest_rows(),
                            train_type="tone_aware_lora",
                            lambda_tone=0.05,
                            seed=42,
                            identity=_identity(),
                            resume=True,
                            overwrite=False,
                        )
                else:
                    prefix, completed = _prepare_prediction_output(
                        prediction_path=prediction,
                        provenance_path=provenance,
                        resume_state_path=state,
                        recovery_path=receipt,
                        manifest_rows=_manifest_rows(),
                        train_type="tone_aware_lora",
                        lambda_tone=0.05,
                        seed=42,
                        identity=_identity(),
                        resume=True,
                        overwrite=False,
                    )
                    self.assertEqual(len(prefix), 2)
                    self.assertFalse(completed)
                    self.assertTrue(state.is_file())
                    self.assertFalse(receipt.exists())

    def test_orphan_csv_and_no_resume_never_silently_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prediction, provenance, state, receipt = self._paths(Path(tmp))
            _atomic_write_prediction(prediction, _prediction_rows()[:1])
            with self.assertRaisesRegex(ValueError, "neither provenance"):
                _prepare_prediction_output(
                    prediction_path=prediction,
                    provenance_path=provenance,
                    resume_state_path=state,
                    recovery_path=receipt,
                    manifest_rows=_manifest_rows(),
                    train_type="tone_aware_lora",
                    lambda_tone=0.05,
                    seed=42,
                    identity=_identity(),
                    resume=True,
                    overwrite=False,
                )
            with self.assertRaises(FileExistsError):
                _prepare_prediction_output(
                    prediction_path=prediction,
                    provenance_path=provenance,
                    resume_state_path=state,
                    recovery_path=receipt,
                    manifest_rows=_manifest_rows(),
                    train_type="tone_aware_lora",
                    lambda_tone=0.05,
                    seed=42,
                    identity=_identity(),
                    resume=False,
                    overwrite=False,
                )

    def test_completed_v4_is_reused_after_exact_provenance_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction, provenance, state, receipt = self._paths(root)
            audio_paths = [root / "audio-1.wav", root / "audio-2.wav"]
            for index, audio in enumerate(audio_paths):
                audio.write_bytes(f"not-decoded-{index}".encode("ascii"))
            manifest = root / "external.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps(
                        {
                            "utt_id": row["utt_id"],
                            "dataset": row["dataset"],
                            "split": "external",
                            "audio_path": str(audio_paths[index]),
                            "transcript": row["ref"],
                            "snr": row["snr"],
                            "noise_type": row["noise_type"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for index, row in enumerate(_manifest_rows())
                ),
                encoding="utf-8",
            )
            selected_manifest = load_benchmark_rows(manifest)
            rows = _prediction_rows()
            _atomic_write_prediction(prediction, rows)
            identity = dict(_identity())
            identity["manifest_sha256"] = sha256_file(manifest)
            identity["selected_rows_sha256"] = selected_rows_sha256(
                selected_manifest
            )
            evaluation_contract = {
                "evaluation": {
                    "benchmark_protocol": "locked_vivos",
                    "batch_size": 1,
                    "inference_precision": "fp32",
                }
            }
            provenance.write_text(
                json.dumps(
                    {
                        "provenance_version": PREDICTION_PROVENANCE_VERSION,
                        **identity,
                        "config_path": str(root / "unused.yaml"),
                        "config_file_sha256": "8" * 64,
                        "evaluation_split": "external",
                        "manifest": str(manifest),
                        "manifest_sha256": sha256_file(manifest),
                        "manifest_num_rows": 2,
                        "audio_hashes_verified": True,
                        "evaluation_scope": "full_manifest",
                        "training_scope": "smoke",
                        "evaluation_contract": evaluation_contract,
                        "evaluation_contract_sha256": canonical_sha256(
                            evaluation_contract
                        ),
                        "runtime_environment": {
                            "batch_size": 1,
                            "device_type": "cpu",
                            "dtype": "torch.float32",
                            "torch_version": "test",
                            "transformers_version": "test",
                        },
                        "filters": {
                            "subset": "all",
                            "snrs": [],
                            "noise_types": [],
                            "limit": None,
                        },
                        "checkpoint": str(root / "checkpoint"),
                        "prediction_sha256": sha256_file(prediction),
                        "num_rows": len(rows),
                        "metric_version": "aligned_v1",
                        "method_lock_sha256": "",
                        "method_identity_sha256": "",
                        "environment_artifact_sha256": "",
                        "environment_identity_sha256": "",
                        "source_tree_sha256": "",
                        "benchmark_protocol": "locked_vivos",
                        "split_lock_sha256": "",
                        "noise_split_lock_sha256": "",
                        "noisy_dev_lock_sha256": "",
                        "decision_lock_sha256": "",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            prefix, completed = _prepare_prediction_output(
                prediction_path=prediction,
                provenance_path=provenance,
                resume_state_path=state,
                recovery_path=receipt,
                manifest_rows=selected_manifest,
                train_type="tone_aware_lora",
                lambda_tone=0.05,
                seed=42,
                identity=identity,
                resume=True,
                overwrite=False,
            )
            self.assertEqual(len(prefix), 2)
            self.assertTrue(completed)
            changed = _prediction_rows()
            changed[-1]["hyp"] = "tampered after completion"
            _atomic_write_prediction(prediction, changed)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _prepare_prediction_output(
                    prediction_path=prediction,
                    provenance_path=provenance,
                    resume_state_path=state,
                    recovery_path=receipt,
                    manifest_rows=selected_manifest,
                    train_type="tone_aware_lora",
                    lambda_tone=0.05,
                    seed=42,
                    identity=identity,
                    resume=True,
                    overwrite=False,
                )

    def test_ablation_resume_reuses_only_exact_aggregate(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lambda_ablation_results.csv"
            row = {column: "" for column in ABLATION_RESULT_COLUMNS}
            row.update(
                {
                    "seed": "42",
                    "evaluation_split": "dev",
                    "manifest_sha256": "3" * 64,
                    "evaluation_scope": "full_manifest",
                    "selected_rows_sha256": "4" * 64,
                    "training_scope": "formal",
                    "evaluation_contract_sha256": "5" * 64,
                    "metric_version": "aligned_v1",
                    "lambda": "0.05",
                }
            )
            with patch(
                "src.vitonesr.phat.evaluation.aggregate_prediction_file",
                return_value=[row],
            ):
                write_ablation_results(
                    [("checkpoint", "prediction")],
                    output,
                    require_lambdas=None,
                )
                reused = write_ablation_results(
                    [("checkpoint", "prediction")],
                    output,
                    require_lambdas=None,
                    resume=True,
                )
                self.assertEqual(reused, output)
                output.write_text("tampered\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    write_ablation_results(
                        [("checkpoint", "prediction")],
                        output,
                        require_lambdas=None,
                        resume=True,
                    )


def sha256_file_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    unittest.main()
