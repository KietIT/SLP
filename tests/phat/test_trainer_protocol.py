from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from src.vitonesr.phat.trainer import (
    TrainerProgress,
    _append_dev_metrics,
    _assert_resume_compatible,
    _assert_training_gate,
    _build_dataset,
    _evaluate_dev,
    _is_strictly_better,
    _merge_saved_best,
    _reached_max_train_steps,
    _resume_needs_dev_evaluation,
    _save_checkpoint,
    _write_dev_metrics_header,
)
from src.vitonesr.phat.training_data import DeterministicASRTrainingDataset


class _EvalModel:
    def __init__(self) -> None:
        self.training = True
        self.lambda_tone = 1.0
        self.call_states: list[tuple[bool, bool]] = []

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def __call__(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self.call_states.append((self.training, torch.is_grad_enabled()))
        if batch["input_features"].size(0) == 1:
            values = (1.0, 2.0, 3.0)
        else:
            values = (3.0, 4.0, 7.0)
        return {
            "asr_loss": torch.tensor(values[0]),
            "tone_loss": torch.tensor(values[1]),
            "total_loss": torch.tensor(values[2]),
        }


class _SavedAsrModel:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")


class _SavedModel:
    def __init__(self) -> None:
        self.asr_model = _SavedAsrModel()
        self.tone_head = None


class _SavedProcessor:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "processor.json").write_text("{}", encoding="utf-8")


class _Scheduler:
    def state_dict(self) -> dict[str, int]:
        return {"step": 1}


def _trainer_config() -> dict[str, object]:
    return {
        "seed": 42,
        "experiment": {
            "method_id": "tone_lora_lambda005",
            "train_type": "tone_aware_lora",
        },
        "model": {
            "name_or_path": "vinai/PhoWhisper-base",
            "revision": "a" * 40,
            "language": "vi",
            "task": "transcribe",
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        "training": {
            "lambda_tone": 0.05,
            "checkpoint_metric": "dev_asr_loss",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-4,
            "tone_label_policy": "last_subtoken",
        },
        "data": {
            "train_manifest": "data/manifests/paper_v2/vivos_train.jsonl",
            "valid_manifest": "data/manifests/paper_v2/vivos_dev.jsonl",
            "sample_rate": 16000,
        },
        "noise": {
            "enable_train_noise": True,
            "noise_manifest": "data/manifests/noise/musan_noise.jsonl",
            "snr_choices": [20, 10, 5, 0],
            "prob": 0.7,
        },
        "_runtime_protocol": {
            "split_lock_sha256": "1" * 64,
            "train_manifest_sha256": "2" * 64,
            "dev_manifest_sha256": "3" * 64,
        },
    }


def _write_complete_best_checkpoint(
    root: Path,
    config: dict[str, object],
    *,
    state_overrides: dict[str, object] | None = None,
) -> Path:
    best = root / "best"
    (best / "adapter").mkdir(parents=True)
    (best / "adapter" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (best / "processor").mkdir()
    (best / "processor" / "processor.json").write_text("{}", encoding="utf-8")
    (best / "tone_head.pt").write_bytes(b"tone-head")

    saved_config = copy.deepcopy(config)
    runtime_protocol = saved_config.pop("_runtime_protocol")
    saved_config["runtime_protocol"] = runtime_protocol
    (best / "resolved_config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=True),
        encoding="utf-8",
    )

    state: dict[str, object] = {
        "global_step": 20,
        "best_eval_asr_loss": 0.75,
        "best_global_step": 20,
        "last_eval_step": 20,
    }
    if state_overrides:
        state.update(state_overrides)
    (best / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    return best


class TrainerProtocolTests(unittest.TestCase):
    def test_formal_training_gate_fails_closed_until_gates_two_and_three(self) -> None:
        config = {
            "training": {"run_scope": "formal"},
            "protocol": {"formal_training_unlocked": False},
        }
        for spelling in ("formal", "Formal", " formal ", "\tFORMAL\n"):
            with self.subTest(run_scope=spelling):
                config["training"]["run_scope"] = spelling
                with self.assertRaisesRegex(
                    RuntimeError, "Formal paper-v2 training is locked"
                ):
                    _assert_training_gate(config)  # type: ignore[arg-type]
        config["training"]["run_scope"] = "smoke"
        _assert_training_gate(config)  # type: ignore[arg-type]
        config["training"]["run_scope"] = "formal"
        config["protocol"]["formal_training_unlocked"] = True
        _assert_training_gate(config)  # type: ignore[arg-type]

    def test_dev_evaluation_is_weighted_inference_only_and_restores_mode(self) -> None:
        model = _EvalModel()
        loader = [
            {
                "input_features": torch.zeros((1, 2)),
                "labels": torch.tensor([[1, 2, -100]]),
                "tone_labels": torch.tensor([[0, -100, -100]]),
            },
            {
                "input_features": torch.zeros((2, 2)),
                "labels": torch.tensor([[1, -100], [2, -100]]),
                "tone_labels": torch.tensor([[0, -100], [1, -100]]),
            },
        ]
        metrics = _evaluate_dev(
            model,  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            mixed_precision="no",
        )
        self.assertTrue(model.training)
        self.assertEqual(model.call_states, [(False, False), (False, False)])
        self.assertEqual(metrics.num_samples, 3)
        self.assertEqual(metrics.asr_tokens, 4)
        self.assertEqual(metrics.tone_targets, 3)
        self.assertAlmostEqual(metrics.asr_loss, 2.0)
        self.assertAlmostEqual(metrics.tone_loss, 10.0 / 3.0)
        self.assertAlmostEqual(metrics.total_loss, 16.0 / 3.0)

    def test_best_metric_is_strict_and_finite(self) -> None:
        self.assertTrue(_is_strictly_better(1.0, None))
        self.assertTrue(_is_strictly_better(0.9, 1.0))
        self.assertFalse(_is_strictly_better(1.0, 1.0))
        self.assertFalse(_is_strictly_better(1.1, 1.0))
        with self.assertRaises(FloatingPointError):
            _is_strictly_better(math.nan, 1.0)

    def test_manifest_split_guard_rejects_test_before_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "audio": str(audio),
                        "text": "xin chào",
                        "utt_id": "utt-1",
                        "split": "test",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "split='dev'"):
                DeterministicASRTrainingDataset(
                    str(manifest),
                    object(),
                    seed=42,
                    expected_split="dev",
                )

    def test_dev_dataset_disables_train_noise_and_requires_dev_split(self) -> None:
        config = {
            "seed": 42,
            "training": {
                "max_audio_seconds": 15,
                "max_label_length": 100,
                "tone_label_policy": "last_subtoken",
            },
            "data": {
                "train_manifest": "train.jsonl",
                "valid_manifest": "dev.jsonl",
                "sample_rate": 16000,
            },
            "noise": {
                "enable_train_noise": True,
                "noise_manifest": "noise.jsonl",
                "prob": 0.7,
            },
        }
        with patch(
            "src.vitonesr.phat.trainer.DeterministicASRTrainingDataset"
        ) as dataset_class:
            _build_dataset(config, object(), split="dev")  # type: ignore[arg-type]
        _, kwargs = dataset_class.call_args
        self.assertEqual(dataset_class.call_args.args[0], "dev.jsonl")
        self.assertIsNone(kwargs["noise_manifest"])
        self.assertEqual(kwargs["noise_prob"], 0.0)
        self.assertEqual(kwargs["expected_split"], "dev")

    def test_saved_best_state_is_merged_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _trainer_config()
            best = _write_complete_best_checkpoint(
                Path(temporary_directory), config
            )
            progress = TrainerProgress(
                global_step=30,
                best_eval_asr_loss=None,
                last_eval_step=10,
            )
            _merge_saved_best(progress, best, config)
            self.assertEqual(progress.best_eval_asr_loss, 0.75)
            self.assertEqual(progress.best_global_step, 20)
            self.assertEqual(progress.last_eval_step, 20)

    def test_saved_best_rejects_stale_training_and_runtime_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _trainer_config()
            best = _write_complete_best_checkpoint(
                Path(temporary_directory), config
            )
            progress = TrainerProgress(global_step=30)

            stale_training = copy.deepcopy(config)
            stale_training["training"]["learning_rate"] = 2.0e-4  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "training contract"):
                _merge_saved_best(progress, best, stale_training)

            stale_runtime = copy.deepcopy(config)
            stale_runtime["_runtime_protocol"]["dev_manifest_sha256"] = "4" * 64  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "runtime protocol mismatch"):
                _merge_saved_best(progress, best, stale_runtime)

    def test_saved_best_requires_state_adapter_and_processor(self) -> None:
        required_paths = {
            r"trainer_state\.json": Path("trainer_state.json"),
            "adapter": Path("adapter") / "adapter_config.json",
            "processor": Path("processor"),
        }
        for expected_message, relative_path in required_paths.items():
            with self.subTest(required=expected_message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    config = _trainer_config()
                    best = _write_complete_best_checkpoint(
                        Path(temporary_directory), config
                    )
                    target = best / relative_path
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    with self.assertRaisesRegex(
                        FileNotFoundError, expected_message
                    ):
                        _merge_saved_best(
                            TrainerProgress(global_step=30), best, config
                        )

    def test_saved_best_requires_best_global_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _trainer_config()
            best = _write_complete_best_checkpoint(
                Path(temporary_directory),
                config,
                state_overrides={"best_global_step": None},
            )
            with self.assertRaisesRegex(ValueError, "no best_global_step"):
                _merge_saved_best(
                    TrainerProgress(global_step=30), best, config
                )

    def test_resume_rejects_model_affecting_config_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _trainer_config()
            checkpoint = _write_complete_best_checkpoint(
                Path(temporary_directory), config
            )
            changed = copy.deepcopy(config)
            changed["model"]["lora"]["target_modules"] = ["q_proj"]  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "training contract mismatch"):
                _assert_resume_compatible(changed, checkpoint)

    def test_resume_rejects_future_best_and_recovers_pending_boundary_eval(self) -> None:
        boundary = TrainerProgress(
            epoch=2,
            next_batch_index=0,
            global_step=200,
            last_eval_step=100,
        )
        self.assertTrue(
            _resume_needs_dev_evaluation(
                boundary,
                is_resume=True,
                reached_max_steps=False,
            )
        )
        at_limit = TrainerProgress(
            epoch=1,
            next_batch_index=7,
            global_step=50,
            last_eval_step=40,
        )
        self.assertTrue(_reached_max_train_steps(at_limit, 50))
        self.assertTrue(
            _resume_needs_dev_evaluation(
                at_limit,
                is_resume=True,
                reached_max_steps=True,
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _trainer_config()
            best = _write_complete_best_checkpoint(
                Path(temporary_directory),
                config,
                state_overrides={
                    "global_step": 60,
                    "best_eval_asr_loss": 0.5,
                    "best_global_step": 60,
                    "last_eval_step": 60,
                },
            )
            with self.assertRaisesRegex(ValueError, "future best checkpoint"):
                _merge_saved_best(at_limit, best, config)

    def test_best_checkpoint_overwrite_leaves_no_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "best"
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = torch.optim.AdamW([parameter])
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            common = {
                "target": target,
                "model": _SavedModel(),
                "processor": _SavedProcessor(),
                "optimizer": optimizer,
                "scheduler": _Scheduler(),
                "scaler": scaler,
                "config": {"training": {"lambda_tone": 0.05}},
            }
            _save_checkpoint(
                **common,  # type: ignore[arg-type]
                progress=TrainerProgress(best_eval_asr_loss=1.0),
            )
            _save_checkpoint(
                **common,  # type: ignore[arg-type]
                progress=TrainerProgress(best_eval_asr_loss=0.5),
                overwrite=True,
            )
            state = json.loads((target / "trainer_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["best_eval_asr_loss"], 0.5)
            self.assertFalse(target.with_name(".best.tmp").exists())
            self.assertFalse(target.with_name(".best.backup").exists())

    def test_resumed_dev_metric_append_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dev.csv"
            _write_dev_metrics_header(path, append=False)
            row = {
                "epoch": 0,
                "global_step": 10,
                "num_samples": 2,
                "asr_tokens": 4,
                "tone_targets": 2,
                "dev_asr_loss": 1.0,
                "dev_tone_loss": 0.5,
                "dev_total_loss": 1.05,
                "is_best": "true",
            }
            _append_dev_metrics(path, row)
            _append_dev_metrics(path, row)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                _append_dev_metrics(path, {**row, "dev_asr_loss": 2.0})


if __name__ == "__main__":
    unittest.main()
