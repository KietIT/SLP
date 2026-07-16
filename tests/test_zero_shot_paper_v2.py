from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.vitonesr.zero_shot_paper_v2 import (
    AuthorizationEvidence,
    LoadedZeroShotModel,
    PREDICTION_COLUMNS,
    RECOVERY_VERSION,
    REPOSITORY_ROOT,
    ZeroShotProtocolError,
    authorize_final_benchmark,
    load_authorized_benchmark,
    load_huggingface_model,
    provenance_path,
    recovery_path,
    resume_path,
    run_zero_shot_suite,
    sha256_file,
    validate_suite_config,
)
from src.vitonesr.phat.protocol import canonical_sha256


REVISION = "a" * 40
HASHES = {
    "split": "1" * 64,
    "decision": "2" * 64,
    "benchmark": "3" * 64,
    "manifest": "4" * 64,
    "source": "5" * 64,
    "snapshot": "6" * 64,
    "model": "7" * 64,
    "processor": "8" * 64,
}


def _repo_ref(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _config(root: Path) -> dict:
    return {
        "protocol": {
            "formal": True,
            "final_test_unlocked": True,
            "split_lock": _repo_ref(root / "split_lock.json"),
            "expected_split_lock_sha256": HASHES["split"],
            "decision_lock": _repo_ref(root / "decision_lock.json"),
            "expected_decision_lock_sha256": HASHES["decision"],
        },
        "benchmark": {
            "lock_protocol_version": "paper_v2_final_benchmark_v1",
            "lock": _repo_ref(root / "benchmark_lock.json"),
            "expected_lock_sha256": HASHES["benchmark"],
            "manifest": _repo_ref(root / "benchmark.jsonl"),
            "expected_manifest_sha256": HASHES["manifest"],
            "expected_rows": 2,
            "dataset": "vivos",
            "verify_audio_sha256": True,
        },
        "output_dir": _repo_ref(root / "predictions"),
        "seed": 42,
        "decoding": {
            "language": "vi",
            "task": "transcribe",
            "sample_rate": 16000,
            "max_audio_seconds": 15.0,
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
        },
        "runtime": {
            "batch_size": 1,
            "device": "cpu",
            "precision": "fp32",
            "local_files_only": True,
        },
        "models": {
            "whisper_tiny": {
                "repo_id": "openai/whisper-tiny",
                "revision": REVISION,
                "model": "whisper",
                "model_size": "tiny",
                "filename": "pred_whisper_tiny.csv",
            }
        },
    }


def _write_config(root: Path, config: dict) -> Path:
    path = root / "suite.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return Path(_repo_ref(path))


def _evidence() -> AuthorizationEvidence:
    return AuthorizationEvidence(
        split_lock_sha256=HASHES["split"],
        decision_lock_sha256=HASHES["decision"],
        benchmark_lock_sha256=HASHES["benchmark"],
        manifest_sha256=HASHES["manifest"],
        manifest_num_rows=2,
        source_test_manifest_sha256=HASHES["source"],
        benchmark_lock_protocol_version="paper_v2_final_benchmark_v1",
    )


def _rows() -> list[dict[str, str]]:
    return [
        {
            "utt_id": "u1_clean",
            "source_utt_id": "u1",
            "dataset": "vivos",
            "audio_path": "unused-1.wav",
            "audio_sha256": "9" * 64,
            "snr": "clean",
            "noise_type": "clean",
            "ref": "xin chào",
        },
        {
            "utt_id": "u1_snr0",
            "source_utt_id": "u1",
            "dataset": "vivos",
            "audio_path": "unused-2.wav",
            "audio_sha256": "a" * 64,
            "snr": "0",
            "noise_type": "speech",
            "ref": "xin chào",
        },
    ]


def _loaded(*, snapshot_path: str | None = None) -> LoadedZeroShotModel:
    return LoadedZeroShotModel(
        processor=object(),
        model=object(),
        device="cpu",
        dtype_name="float32",
        torch_module=None,
        snapshot_path=snapshot_path or "cache/snapshots/" + REVISION,
        snapshot_sha256=HASHES["snapshot"],
        model_fingerprint_sha256=HASHES["model"],
        processor_fingerprint_sha256=HASHES["processor"],
        runtime_environment={
            "device_type": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
        },
    )


class ZeroShotPaperV2Tests(unittest.TestCase):
    def test_authorization_precedes_manifest_and_model_and_sidecar_binds_hashes(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            root = Path(tmp)
            config_path = _write_config(root, _config(root))
            events: list[str] = []
            local_snapshot_path = str(
                (root / "private-hf-cache" / "snapshots" / REVISION).resolve()
            )

            def authorizer(_config):
                events.append("authorize")
                return _evidence()

            def manifest_loader(_config, _evidence_value):
                events.append("manifest")
                return _rows()

            def model_loader(spec, _config):
                events.append("model:" + str(spec["revision"]))
                return _loaded(snapshot_path=local_snapshot_path)

            def decoder(_loaded_value, rows, _config):
                events.append("decode")
                return ["giả thuyết"] * len(rows)

            result = run_zero_shot_suite(
                config_path,
                authorizer=authorizer,
                manifest_loader=manifest_loader,
                model_loader=model_loader,
                decoder=decoder,
            )
            self.assertEqual(events[:3], ["authorize", "manifest", "model:" + REVISION])
            prediction = root / "predictions" / "pred_whisper_tiny.csv"
            sidecar = provenance_path(prediction)
            self.assertTrue(prediction.is_file())
            self.assertTrue(sidecar.is_file())
            self.assertFalse(resume_path(prediction).exists())
            self.assertFalse(recovery_path(prediction).exists())
            with prediction.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, PREDICTION_COLUMNS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["train_type"], "zero_shot")
            self.assertEqual(rows[0]["lambda"], "")
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(provenance["prediction_sha256"], sha256_file(prediction))
            self.assertEqual(provenance["manifest_sha256"], HASHES["manifest"])
            self.assertEqual(provenance["benchmark_lock_sha256"], HASHES["benchmark"])
            self.assertEqual(provenance["model_revision"], REVISION)
            self.assertEqual(provenance["snapshot_sha256"], HASHES["snapshot"])
            self.assertNotIn("snapshot_path", provenance)
            for field in ("prediction", "manifest", "benchmark_lock", "suite_config"):
                self.assertFalse(Path(provenance[field]).is_absolute())
                self.assertNotIn("\\", provenance[field])

            def all_strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from all_strings(item)
                elif isinstance(value, list):
                    for item in value:
                        yield from all_strings(item)

            self.assertFalse(
                any(local_snapshot_path in value for value in all_strings(provenance)),
                "formal provenance must not expose the local Hugging Face snapshot path",
            )
            self.assertEqual(result["models"][0]["status"], "complete")

            events.clear()

            def forbidden_model(*_args):
                raise AssertionError("a verified completed artifact must not reload a model")

            rerun = run_zero_shot_suite(
                config_path,
                authorizer=authorizer,
                manifest_loader=manifest_loader,
                model_loader=forbidden_model,
                decoder=decoder,
            )
            self.assertEqual(events, ["authorize", "manifest"])
            self.assertEqual(rerun["models"][0]["status"], "verified_existing")

    def test_formal_paths_reject_absolute_or_nonportable_references(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            root = Path(tmp)
            config = _config(root)
            for section, field in (
                ("protocol", "split_lock"),
                ("protocol", "decision_lock"),
                ("benchmark", "lock"),
                ("benchmark", "manifest"),
            ):
                candidate = json.loads(json.dumps(config))
                candidate[section][field] = str((root / field).resolve())
                with self.subTest(field=f"{section}.{field}"):
                    with self.assertRaisesRegex(
                        ZeroShotProtocolError, "repository-relative"
                    ):
                        validate_suite_config(candidate)
            candidate = json.loads(json.dumps(config))
            candidate["output_dir"] = "outputs\\nonportable"
            with self.assertRaisesRegex(ZeroShotProtocolError, "repository-relative"):
                validate_suite_config(candidate)

            config_path = _write_config(root, config)
            with self.assertRaisesRegex(ZeroShotProtocolError, "config_path"):
                run_zero_shot_suite(
                    (REPOSITORY_ROOT / config_path).resolve(),
                    authorizer=lambda _config: (_ for _ in ()).throw(
                        AssertionError("absolute config must fail before authorization")
                    ),
                )

    def test_orphan_csv_recovers_only_from_exact_write_ahead_receipt(self) -> None:
        for tamper in (False, True):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory(
                dir=REPOSITORY_ROOT
            ) as tmp:
                root = Path(tmp)
                config_path = _write_config(root, _config(root))
                calls = 0

                def interrupted_decoder(_loaded_value, rows, _config):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("simulated interruption")
                    return ["ok-one"] * len(rows)

                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_zero_shot_suite(
                        config_path,
                        authorizer=lambda _config: _evidence(),
                        manifest_loader=lambda *_args: _rows(),
                        model_loader=lambda *_args: _loaded(),
                        decoder=interrupted_decoder,
                    )
                prediction = root / "predictions" / "pred_whisper_tiny.csv"
                progress = resume_path(prediction)
                receipt = recovery_path(prediction)
                state = json.loads(progress.read_text(encoding="utf-8"))
                previous_sha = sha256_file(prediction)
                with prediction.open("r", encoding="utf-8", newline="") as handle:
                    existing = list(csv.DictReader(handle))
                benchmark = _rows()[1]
                existing.append(
                    {
                        "utt_id": benchmark["utt_id"],
                        "dataset": benchmark["dataset"],
                        "model": "whisper",
                        "model_size": "tiny",
                        "train_type": "zero_shot",
                        "lambda": "",
                        "seed": "42",
                        "snr": benchmark["snr"],
                        "noise_type": benchmark["noise_type"],
                        "ref": benchmark["ref"],
                        "hyp": "ok-two",
                    }
                )
                intended = root / "intended.csv"
                with intended.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(existing)
                intended_sha = sha256_file(intended)
                receipt.write_text(
                    json.dumps(
                        {
                            "recovery_version": RECOVERY_VERSION,
                            "run_contract_sha256": state["run_contract_sha256"],
                            "selected_rows_sha256": state["selected_rows_sha256"],
                            "manifest_sha256": state["manifest_sha256"],
                            "benchmark_lock_sha256": state[
                                "benchmark_lock_sha256"
                            ],
                            "decision_lock_sha256": state["decision_lock_sha256"],
                            "completed_rows": 2,
                            "prediction_sha256": intended_sha,
                            "previous_completed_rows": 1,
                            "previous_prediction_sha256": previous_sha,
                            "snapshot": state["snapshot"],
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                intended.replace(prediction)
                progress.unlink()  # exact crash window: CSV published, state absent
                if tamper:
                    with prediction.open("r", encoding="utf-8", newline="") as handle:
                        altered = list(csv.DictReader(handle))
                    altered[-1]["hyp"] = "tampered hypothesis"
                    with prediction.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n"
                        )
                        writer.writeheader()
                        writer.writerows(altered)

                    def forbidden_model(*_args):
                        raise AssertionError("tamper must fail before model access")

                    with self.assertRaisesRegex(
                        ZeroShotProtocolError, "possible tamper"
                    ):
                        run_zero_shot_suite(
                            config_path,
                            resume=True,
                            authorizer=lambda _config: _evidence(),
                            manifest_loader=lambda *_args: _rows(),
                            model_loader=forbidden_model,
                        )
                else:
                    result = run_zero_shot_suite(
                        config_path,
                        resume=True,
                        authorizer=lambda _config: _evidence(),
                        manifest_loader=lambda *_args: _rows(),
                        model_loader=lambda *_args: _loaded(),
                        decoder=lambda *_args: (_ for _ in ()).throw(
                            AssertionError("complete recovered CSV must not decode")
                        ),
                    )
                    self.assertEqual(result["models"][0]["status"], "complete")
                    self.assertFalse(receipt.exists())
                    self.assertFalse(progress.exists())
                    self.assertTrue(provenance_path(prediction).is_file())

    def test_failed_authorization_cannot_touch_manifest_or_model(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            root = Path(tmp)
            config_path = _write_config(root, _config(root))
            events: list[str] = []

            def deny(_config):
                events.append("authorize")
                raise ZeroShotProtocolError("locked")

            def forbidden(*_args):
                events.append("forbidden")
                raise AssertionError("must not be called")

            with self.assertRaisesRegex(ZeroShotProtocolError, "locked"):
                run_zero_shot_suite(
                    config_path,
                    authorizer=deny,
                    manifest_loader=forbidden,
                    model_loader=forbidden,
                    decoder=forbidden,
                )
            self.assertEqual(events, ["authorize"])

    def test_resume_rejects_changed_decode_contract_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            root = Path(tmp)
            config = _config(root)
            config_path = _write_config(root, config)
            calls = 0

            def decoder(_loaded_value, rows, _config):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return ["ok"] * len(rows)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_zero_shot_suite(
                    config_path,
                    authorizer=lambda _config: _evidence(),
                    manifest_loader=lambda *_args: _rows(),
                    model_loader=lambda *_args: _loaded(),
                    decoder=decoder,
                )
            prediction = root / "predictions" / "pred_whisper_tiny.csv"
            self.assertTrue(prediction.is_file())
            self.assertTrue(resume_path(prediction).is_file())
            config["decoding"]["max_new_tokens"] = 64
            _write_config(root, config)

            def forbidden_model(*_args):
                raise AssertionError("resume contract must fail before model load")

            with self.assertRaisesRegex(ZeroShotProtocolError, "run_contract_sha256"):
                run_zero_shot_suite(
                    config_path,
                    resume=True,
                    authorizer=lambda _config: _evidence(),
                    manifest_loader=lambda *_args: _rows(),
                    model_loader=forbidden_model,
                    decoder=decoder,
                )

    def test_mutable_or_placeholder_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            config = _config(Path(tmp))
            config["models"]["whisper_tiny"]["revision"] = "REQUIRED_IMMUTABLE_REVISION"
            with self.assertRaisesRegex(ZeroShotProtocolError, "immutable"):
                validate_suite_config(config)

    def test_real_authorizer_uses_full_transitive_verifiers_without_manifest_access(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            root = Path(tmp)
            split = root / "split_lock.json"
            decision = root / "decision_lock.json"
            split.write_text("{}\n", encoding="utf-8")
            split_hash = sha256_file(split)
            noise_lock = root / "noise_split_lock.json"
            noise_lock.write_text("{}\n", encoding="utf-8")
            noise_hash = sha256_file(noise_lock)
            method_lock = root / "method_lock.json"
            method = {
                "schema_version": "paper_v2_method_contract_v1",
                "status": "LOCKED",
                "mode": "formal",
                "artifacts": {
                    "noise_split_lock": {
                        "path": _repo_ref(noise_lock),
                        "sha256": noise_hash,
                    }
                },
            }
            method["identity_sha256"] = canonical_sha256(method)
            method_lock.write_text(
                json.dumps(method, sort_keys=True) + "\n", encoding="utf-8"
            )
            method_hash = sha256_file(method_lock)
            decision_payload = {
                "method_lock": _repo_ref(method_lock),
                "method_lock_sha256": method_hash,
                "method_identity_sha256": method["identity_sha256"],
            }
            decision.write_text(
                json.dumps(decision_payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            decision_hash = sha256_file(decision)
            manifest = root / "not-created.jsonl"
            benchmark_lock = root / "benchmark_lock.json"
            benchmark_lock.write_text("{}\n", encoding="utf-8")
            config = _config(root)
            config["protocol"]["expected_split_lock_sha256"] = split_hash
            config["protocol"]["expected_decision_lock_sha256"] = decision_hash
            config["benchmark"]["expected_lock_sha256"] = sha256_file(benchmark_lock)
            config["benchmark"]["manifest"] = _repo_ref(manifest)

            def verified(**kwargs):
                self.assertIs(kwargs.get("verify_checkpoints"), False)
                return {
                    "split_lock_sha256": split_hash,
                    "decision_lock_sha256": decision_hash,
                    "test_manifest_sha256": HASHES["source"],
                    "method_lock_sha256": method_hash,
                    "method_identity_sha256": method["identity_sha256"],
                }

            def verified_noise(path, **kwargs):
                self.assertEqual(path.resolve(), noise_lock.resolve())
                self.assertIs(kwargs.get("verify_audio"), False)
                return {"lock_sha256": noise_hash, "lock": {}}

            def verified_method(path, **kwargs):
                self.assertEqual(path.resolve(), method_lock.resolve())
                self.assertEqual(kwargs.get("repo_root"), REPOSITORY_ROOT)
                self.assertIs(kwargs.get("formal"), True)
                return {
                    "method_lock_sha256": method_hash,
                    "method_identity_sha256": method["identity_sha256"],
                    "mode": "formal",
                    "artifacts": {
                        "noise_split_lock": {
                            "path": _repo_ref(noise_lock),
                            "sha256": noise_hash,
                        }
                    },
                }

            def verified_benchmark(path, **kwargs):
                self.assertEqual(path.resolve(), benchmark_lock.resolve())
                self.assertEqual(kwargs["method_lock_sha256"], method_hash)
                self.assertEqual(
                    kwargs["method_identity_sha256"], method["identity_sha256"]
                )
                self.assertEqual(kwargs["noise_split_lock_sha256"], noise_hash)
                self.assertEqual(kwargs["source_test_manifest_sha256"], HASHES["source"])
                return {
                    "lock_sha256": sha256_file(benchmark_lock),
                    "protocol_version": "paper_v2_final_benchmark_v1",
                    "manifest_sha256": HASHES["manifest"],
                    "row_count": 2,
                }

            evidence = authorize_final_benchmark(
                config,
                decision_verifier=verified,
                method_artifact_verifier=verified_method,
                noise_verifier=verified_noise,
                benchmark_verifier=verified_benchmark,
            )
            self.assertEqual(evidence.manifest_sha256, HASHES["manifest"])
            self.assertFalse(manifest.exists(), "authorization must not read/create manifest")

            def rejected_benchmark(*_args, **_kwargs):
                raise ValueError("builder contract is invalid")

            with self.assertRaisesRegex(ZeroShotProtocolError, "builder contract"):
                authorize_final_benchmark(
                    config,
                    decision_verifier=verified,
                    method_artifact_verifier=verified_method,
                    noise_verifier=verified_noise,
                    benchmark_verifier=rejected_benchmark,
                )

    def test_default_huggingface_loader_passes_revision_to_every_resolver(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            calls: list[tuple[str, dict]] = []

            class Config:
                _commit_hash = REVISION
                use_cache = False

                def to_dict(self):
                    return {"model_type": "whisper"}

            class Model:
                def __init__(self):
                    self.config = Config()

                def to(self, **_kwargs):
                    return self

                def eval(self):
                    return self

            class ModelFactory:
                @classmethod
                def from_pretrained(cls, repo_id, **kwargs):
                    calls.append(("model:" + repo_id, kwargs))
                    return Model()

            class FeatureExtractor:
                def to_dict(self):
                    return {"sampling_rate": 16000}

            class Tokenizer:
                init_kwargs = {"language": "vi"}

            class Processor:
                feature_extractor = FeatureExtractor()
                tokenizer = Tokenizer()

            class ProcessorFactory:
                @classmethod
                def from_pretrained(cls, repo_id, **kwargs):
                    calls.append(("processor:" + repo_id, kwargs))
                    return Processor()

            def resolve(**kwargs):
                calls.append(("snapshot", kwargs))
                return str(snapshot)

            class Device:
                type = "cpu"

            class Cuda:
                @staticmethod
                def is_available():
                    return False

                @staticmethod
                def manual_seed_all(_seed):
                    return None

            class Cudnn:
                benchmark = True
                deterministic = False

            class Backends:
                cudnn = Cudnn()

            class Version:
                cuda = None

            class FakeTorch:
                __version__ = "test"
                cuda = Cuda()
                backends = Backends()
                version = Version()
                float16 = "float16"
                float32 = "float32"
                _deterministic = False

                @staticmethod
                def manual_seed(_seed):
                    return None

                @classmethod
                def use_deterministic_algorithms(cls, enabled):
                    cls._deterministic = enabled

                @classmethod
                def are_deterministic_algorithms_enabled(cls):
                    return cls._deterministic

                @staticmethod
                def device(_name):
                    return Device()

            config = _config(Path(tmp))
            loaded = load_huggingface_model(
                {
                    "repo_id": "openai/whisper-tiny",
                    "revision": REVISION,
                    "model": "whisper",
                    "model_size": "tiny",
                },
                config,
                torch_module=FakeTorch,
                processor_class=ProcessorFactory,
                model_class=ModelFactory,
                snapshot_resolver=resolve,
            )
            self.assertEqual([kwargs["revision"] for _, kwargs in calls], [REVISION] * 3)
            self.assertTrue(all(kwargs["local_files_only"] for _, kwargs in calls))
            self.assertEqual(len(loaded.snapshot_sha256), 64)
            self.assertEqual(len(loaded.model_fingerprint_sha256), 64)
            self.assertEqual(len(loaded.processor_fingerprint_sha256), 64)

    def test_authorized_manifest_verifies_manifest_and_audio_hashes(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as tmp:
            root = Path(tmp)
            clean = root / "clean.wav"
            noisy = root / "noisy.wav"
            clean.write_bytes(b"clean-audio")
            noisy.write_bytes(b"noisy-audio")
            manifest = root / "benchmark.jsonl"
            records = [
                {
                    "utt_id": "u1_clean",
                    "source_utt_id": "u1",
                    "dataset": "vivos",
                    "split": "test",
                    "audio_path": _repo_ref(clean),
                    "audio_sha256": sha256_file(clean),
                    "transcript": "xin chào",
                    "snr": "clean",
                    "noise_type": "clean",
                },
                {
                    "utt_id": "u1_snr0",
                    "source_utt_id": "u1",
                    "dataset": "vivos",
                    "split": "test",
                    "audio_path": _repo_ref(noisy),
                    "audio_sha256": sha256_file(noisy),
                    "transcript": "xin chào",
                    "snr": 0,
                    "noise_type": "speech",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            config = _config(root)
            config["benchmark"]["manifest"] = _repo_ref(manifest)
            evidence = AuthorizationEvidence(
                **{
                    **_evidence().__dict__,
                    "manifest_sha256": sha256_file(manifest),
                }
            )
            rows = load_authorized_benchmark(config, evidence)
            self.assertEqual([row["utt_id"] for row in rows], ["u1_clean", "u1_snr0"])
            noisy.write_bytes(b"tampered")
            with self.assertRaisesRegex(ZeroShotProtocolError, "audio SHA-256 mismatch"):
                load_authorized_benchmark(config, evidence)


if __name__ == "__main__":
    unittest.main()
