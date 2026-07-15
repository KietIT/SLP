from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.run_external_fleurs import (
    DEFAULT_MAX_NEW_TOKENS,
    MAX_CHUNK_SECONDS,
    RESULT_COLUMNS,
    RUN_TEMPLATES,
    ExternalFleursError,
    ExternalRun,
    WhisperAdapterTranscriber,
    _partial_path,
    _prediction_row,
    _load_processor_with_fallback,
    build_external_results,
    build_external_runs,
    join_chunk_hypotheses,
    load_fleurs_manifest,
    main,
    parse_args,
    run_external_prediction,
    run_external_suite,
    split_waveform,
)
from src.vitonesr.analysis import CANONICAL_PREDICTION_COLUMNS, METRIC_VERSION
from src.vitonesr.prediction import atomic_write_csv


ROOT = Path(__file__).resolve().parents[1]


def _manifest(path: Path, audio_paths: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, audio in enumerate(audio_paths):
            handle.write(
                json.dumps(
                    {
                        "utt_id": f"fleurs-{index}",
                        "dataset": "fleurs",
                        "split": "test",
                        "audio_path": audio.as_posix(),
                        "transcript": "xin chào" if index == 0 else "tôi là trung",
                        "snr": "clean",
                        "noise_type": "clean",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class FakeTranscriber:
    def __init__(self, hypothesis: str, calls: list[int]) -> None:
        self.hypothesis = hypothesis
        self.calls = calls
        self.closed = False

    def transcribe_chunk(self, waveform: Any) -> str:
        self.calls.append(len(waveform))
        return self.hypothesis

    def close(self) -> None:
        self.closed = True


class ExternalFleursTests(unittest.TestCase):
    def test_fixed_three_run_contract_matches_local_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = build_external_runs(
                config_dir=ROOT / "configs" / "phat",
                checkpoint_root=Path(temporary),
            )
        self.assertEqual(
            [(run.train_type, run.lambda_value) for run in runs],
            [
                ("ordinary_lora", "0"),
                ("tone_aware_lora", "0.05"),
                ("tone_aware_lora", "0.1"),
            ],
        )
        self.assertEqual(len(runs), 3)

    def test_incompatible_checkpoint_processor_falls_back_to_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "final"
            (checkpoint / "processor").mkdir(parents=True)
            run = ExternalRun(
                train_type="tone_aware_lora",
                lambda_value="0.05",
                seed="42",
                model_name_or_path="vinai/PhoWhisper-base",
                language="vi",
                task="transcribe",
                checkpoint=checkpoint,
                prediction_name="prediction.csv",
            )
            calls: list[tuple[str, str, str]] = []

            class ProcessorStub:
                @classmethod
                def from_pretrained(
                    cls, source: str, *, language: str, task: str
                ) -> object:
                    calls.append((source, language, task))
                    if Path(source) == checkpoint / "processor":
                        raise AttributeError("'list' object has no attribute 'keys'")
                    return {"source": source}

            with self.assertWarnsRegex(RuntimeWarning, "falling back"):
                processor = _load_processor_with_fallback(ProcessorStub, run)

            self.assertEqual(processor, {"source": "vinai/PhoWhisper-base"})
        self.assertEqual(
            calls,
                [
                    (str(checkpoint / "processor"), "vi", "transcribe"),
                    ("vinai/PhoWhisper-base", "vi", "transcribe"),
            ],
        )

    def test_invalid_byte_bpe_falls_back_to_same_vocab_slow_decode(self) -> None:
        vocab = {"token": 1}

        class FastTokenizer:
            def get_vocab(self) -> dict[str, int]:
                return vocab

        class ProcessorStub:
            tokenizer = FastTokenizer()

            @staticmethod
            def batch_decode(_generated: object, *, skip_special_tokens: bool) -> list[str]:
                self.assertTrue(skip_special_tokens)
                return ["c\ufffdối"]

        class SlowTokenizerStub:
            errors = "strict"

            @classmethod
            def from_pretrained(cls, *_args: object, **kwargs: object) -> "SlowTokenizerStub":
                self.assertEqual(kwargs["errors"], "strict")
                return cls()

            def get_vocab(self) -> dict[str, int]:
                return vocab

            def decode(self, _ids: object, *, skip_special_tokens: bool) -> str:
                if not skip_special_tokens:
                    raise AssertionError("special tokens must be skipped")
                if self.errors == "strict":
                    raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
                return "cối"

        transcriber = object.__new__(WhisperAdapterTranscriber)
        transcriber.processor = ProcessorStub()
        transcriber.slow_tokenizer_class = SlowTokenizerStub
        transcriber.slow_tokenizer = None
        transcriber.model_name_or_path = "vinai/PhoWhisper-base"
        transcriber.language = "vi"
        transcriber.task = "transcribe"

        with self.assertWarnsRegex(UnicodeWarning, "invalid byte-BPE"):
            decoded = transcriber._decode_generated([[1, 2, 3]])
        self.assertEqual(decoded, "cối")
        self.assertEqual(transcriber.slow_tokenizer.errors, "strict")

    def test_manifest_accepts_external_size_instead_of_vivos_1500(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = [root / "a.wav", root / "b.wav"]
            for path in audio:
                path.touch()
            manifest = root / "test.jsonl"
            _manifest(manifest, audio)

            rows = load_fleurs_manifest(manifest, expected_rows=None)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["snr"] == "clean" for row in rows))
            with self.assertRaisesRegex(ExternalFleursError, "expected 857"):
                load_fleurs_manifest(manifest)

    def test_manifest_rejects_nonclean_or_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio.wav"
            audio.touch()
            manifest = root / "test.jsonl"
            records = [
                {
                    "utt_id": "same",
                    "dataset": "fleurs",
                    "split": "test",
                    "audio_path": audio.as_posix(),
                    "transcript": "một",
                    "snr": "clean",
                    "noise_type": "clean",
                },
                {
                    "utt_id": "same",
                    "dataset": "fleurs",
                    "split": "test",
                    "audio_path": audio.as_posix(),
                    "transcript": "hai",
                    "snr": "clean",
                    "noise_type": "clean",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ExternalFleursError, "duplicate utt_id"):
                load_fleurs_manifest(manifest, expected_rows=None)

            records[1]["utt_id"] = "different"
            records[1]["snr"] = "5"
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ExternalFleursError, "must be clean"):
                load_fleurs_manifest(manifest, expected_rows=None)

    def test_smoke_limit_requires_separate_output_directory(self) -> None:
        with self.assertRaisesRegex(ExternalFleursError, "separate --output-dir"):
            main(["--limit", "3"])

    def test_waveform_chunking_is_nonoverlapping_complete_and_at_most_30s(self) -> None:
        sample_rate = 2
        waveform = list(range(125))
        chunks = split_waveform(waveform, sample_rate=sample_rate)
        self.assertEqual([len(chunk) for chunk in chunks], [42, 42, 41])
        self.assertEqual([value for chunk in chunks for value in chunk], waveform)
        self.assertLessEqual(max(map(len, chunks)), sample_rate * MAX_CHUNK_SECONDS)
        self.assertLessEqual(max(map(len, chunks)) - min(map(len, chunks)), 1)
        self.assertEqual(join_chunk_hypotheses(["  xin ", "", " chào  "]), "xin chào")

    def test_cli_default_leaves_whisper_decoder_token_headroom(self) -> None:
        self.assertEqual(DEFAULT_MAX_NEW_TOKENS, 440)
        self.assertEqual(parse_args([]).max_new_tokens, DEFAULT_MAX_NEW_TOKENS)

    def test_suite_writes_three_canonical_predictions_and_aligned_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_paths = [root / "one.wav", root / "two.wav"]
            for path in audio_paths:
                path.touch()
            manifest = root / "test.jsonl"
            _manifest(manifest, audio_paths)
            calls: dict[str, list[int]] = {}

            def factory(run: ExternalRun, _device: str, _tokens: int) -> FakeTranscriber:
                calls[run.lambda_value] = []
                hypotheses = {
                    "0": "xin",
                    "0.05": "xin chào",
                    "0.1": "xin chao",
                }
                return FakeTranscriber(hypotheses[run.lambda_value], calls[run.lambda_value])

            predictions, results = run_external_suite(
                manifest,
                output_dir=root / "external",
                config_dir=ROOT / "configs" / "phat",
                checkpoint_root=root / "checkpoints",
                expected_rows=None,
                transcriber_factory=factory,
                audio_loader=lambda _path, _sample_rate: [0.0] * 16,
                checkpoint_every=1,
            )

            self.assertEqual(len(predictions), 3)
            self.assertEqual(set(calls), {"0", "0.05", "0.1"})
            self.assertTrue(all(lengths == [16, 16] for lengths in calls.values()))
            for path, template in zip(predictions, RUN_TEMPLATES):
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(tuple(reader.fieldnames or ()), CANONICAL_PREDICTION_COLUMNS)
                    rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["train_type"], template.train_type)
                self.assertEqual(rows[0]["lambda"], template.lambda_value)
                self.assertTrue(all(row["snr"] == "clean" for row in rows))
                self.assertFalse(_partial_path(path).exists())

            with results.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), RESULT_COLUMNS)
                result_rows = list(reader)
            self.assertEqual(len(result_rows), 3)
            self.assertTrue(all(row["n"] == "2" for row in result_rows))
            self.assertTrue(
                all(row["metric_version"] == METRIC_VERSION for row in result_rows)
            )
            self.assertEqual([row["lambda"] for row in result_rows], ["0", "0.05", "0.1"])

            with self.assertRaisesRegex(FileExistsError, "External result already exists"):
                run_external_suite(
                    manifest,
                    output_dir=root / "external",
                    config_dir=ROOT / "configs" / "phat",
                    checkpoint_root=root / "checkpoints",
                    expected_rows=None,
                    transcriber_factory=factory,
                    audio_loader=lambda _path, _sample_rate: [0.0],
                )

    def test_suite_preflights_all_outputs_before_loading_any_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "one.wav"
            audio.touch()
            manifest = root / "test.jsonl"
            _manifest(manifest, [audio])
            occupied = (
                root
                / "external"
                / "predictions"
                / RUN_TEMPLATES[1].prediction_name
            )
            occupied.parent.mkdir(parents=True)
            occupied.write_text("do not overwrite", encoding="utf-8")
            factory_calls: list[str] = []

            def factory(run: ExternalRun, _device: str, _tokens: int) -> FakeTranscriber:
                factory_calls.append(run.lambda_value)
                return FakeTranscriber("unused", [])

            with self.assertRaisesRegex(FileExistsError, "Prediction file already exists"):
                run_external_suite(
                    manifest,
                    output_dir=root / "external",
                    config_dir=ROOT / "configs" / "phat",
                    checkpoint_root=root / "checkpoints",
                    expected_rows=None,
                    transcriber_factory=factory,
                    audio_loader=lambda _path, _sample_rate: [0.0],
                )
            self.assertEqual(factory_calls, [])

    def test_resume_uses_valid_prefix_and_only_transcribes_remaining_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = build_external_runs(
                config_dir=ROOT / "configs" / "phat", checkpoint_root=root
            )[0]
            manifest_rows = [
                {
                    "utt_id": "one",
                    "dataset": "fleurs",
                    "audio_path": str(root / "one.wav"),
                    "ref": "xin chào",
                    "snr": "clean",
                    "noise_type": "clean",
                },
                {
                    "utt_id": "two",
                    "dataset": "fleurs",
                    "audio_path": str(root / "two.wav"),
                    "ref": "tôi là trung",
                    "snr": "clean",
                    "noise_type": "clean",
                },
            ]
            output = root / "prediction.csv"
            partial = _partial_path(output)
            atomic_write_csv(
                partial,
                [_prediction_row(manifest_rows[0], run, "xin chào")],
                CANONICAL_PREDICTION_COLUMNS,
            )
            calls: list[int] = []
            created: list[FakeTranscriber] = []

            def factory(_run: ExternalRun, _device: str, _tokens: int) -> FakeTranscriber:
                transcriber = FakeTranscriber("tôi là trung", calls)
                created.append(transcriber)
                return transcriber

            run_external_prediction(
                run,
                manifest_rows,
                output,
                resume=True,
                transcriber_factory=factory,
                audio_loader=lambda _path, _sample_rate: [0.0] * 7,
            )
            self.assertEqual(calls, [7])
            self.assertTrue(created[0].closed)
            self.assertFalse(partial.exists())
            with output.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

            # A completed resume is validation-only and must not reload a model.
            run_external_prediction(
                run,
                manifest_rows,
                output,
                resume=True,
                transcriber_factory=lambda *_args: self.fail("model should not load"),
            )

    def test_resume_rejects_wrong_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = build_external_runs(
                config_dir=ROOT / "configs" / "phat", checkpoint_root=root
            )
            manifest_rows = [
                {
                    "utt_id": "one",
                    "dataset": "fleurs",
                    "audio_path": str(root / "one.wav"),
                    "ref": "một",
                    "snr": "clean",
                    "noise_type": "clean",
                }
            ]
            output = root / "prediction.csv"
            atomic_write_csv(
                _partial_path(output),
                [_prediction_row(manifest_rows[0], runs[1], "một")],
                CANONICAL_PREDICTION_COLUMNS,
            )
            with self.assertRaisesRegex(ExternalFleursError, "conflicts"):
                run_external_prediction(runs[0], manifest_rows, output, resume=True)

    def test_results_require_exact_paired_utterance_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = build_external_runs(
                config_dir=ROOT / "configs" / "phat", checkpoint_root=root
            )
            artifacts: list[tuple[ExternalRun, Path]] = []
            for index, run in enumerate(runs):
                path = root / run.prediction_name
                manifest_row = {
                    "utt_id": "same",
                    "dataset": "fleurs",
                    "audio_path": "unused.wav",
                    "ref": "khác" if index == 2 else "một",
                    "snr": "clean",
                    "noise_type": "clean",
                }
                atomic_write_csv(
                    path,
                    [_prediction_row(manifest_row, run, "một")],
                    CANONICAL_PREDICTION_COLUMNS,
                )
                artifacts.append((run, path))
            with self.assertRaisesRegex(ExternalFleursError, "paired FLEURS run"):
                build_external_results(artifacts)


if __name__ == "__main__":
    unittest.main()
