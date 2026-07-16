from __future__ import annotations

import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Sequence

import scripts.aggregate_results as aggregate_results
import scripts.build_error_artifacts as error_artifacts
import scripts.build_error_breakdowns as error_breakdowns
import scripts.download_fleurs as download_fleurs
import scripts.error_analysis as error_analysis
import scripts.make_vivos_manifest as vivos_manifest
import scripts.run_external_fleurs as external_fleurs
import src.vitonesr.comparison as comparison
import src.vitonesr.final_benchmark as final_benchmark
import src.vitonesr.noise_protocol as noise_protocol
import src.vitonesr.phat.evaluation as phat_evaluation
import src.vitonesr.phat.final_evaluation as final_lora
import src.vitonesr.phat.trainer as phat_trainer
import src.vitonesr.zero_shot_paper_v2 as zero_shot
from src.vitonesr.analysis import CANONICAL_PREDICTION_COLUMNS
from src.vitonesr.phat.protocol import sha256_file
from src.vitonesr.prediction import atomic_write_csv


REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(columns: Sequence[str]) -> dict[str, str]:
    return {column: f"giá trị {index}" for index, column in enumerate(columns)}


class CanonicalCsvLfContractTests(unittest.TestCase):
    def assert_lf_csv(self, payload: bytes, *, label: str) -> None:
        self.assertTrue(payload.endswith(b"\n"), label)
        self.assertNotIn(b"\r\n", payload, label)

    def test_formal_csv_renderers_emit_lf_only(self) -> None:
        renderers: list[tuple[str, Callable[[], bytes]]] = [
            (
                "lambda evaluation",
                lambda: phat_evaluation._prediction_csv_bytes(
                    [_row(phat_evaluation.PREDICTION_COLUMNS)]
                ),
            ),
            (
                "zero shot",
                lambda: zero_shot._prediction_csv_bytes(
                    [_row(zero_shot.PREDICTION_COLUMNS)]
                ),
            ),
            (
                "final LoRA prediction",
                lambda: final_lora._csv_bytes(
                    [_row(final_lora.PREDICTION_COLUMNS)],
                    final_lora.PREDICTION_COLUMNS,
                ),
            ),
            (
                "FLEURS prediction",
                lambda: external_fleurs._prediction_csv_bytes(
                    [_row(CANONICAL_PREDICTION_COLUMNS)]
                ),
            ),
            (
                "FLEURS results",
                lambda: external_fleurs._canonical_result_csv_bytes(
                    [_row(external_fleurs.RESULT_COLUMNS)]
                ),
            ),
            (
                "aggregate",
                lambda: aggregate_results._csv_bytes(
                    [_row(aggregate_results.RESULTS_BY_SNR_COLUMNS)],
                    aggregate_results.RESULTS_BY_SNR_COLUMNS,
                ),
            ),
            (
                "error analysis",
                lambda: error_analysis._csv_bytes(
                    [_row(error_analysis.EVENT_COLUMNS)], error_analysis.EVENT_COLUMNS
                ),
            ),
            (
                "error artifacts",
                lambda: error_artifacts._render_csv_bytes(
                    Path("unused.csv"),
                    [_row(error_artifacts.TONE_MATRIX_COLUMNS)],
                    columns=error_artifacts.TONE_MATRIX_COLUMNS,
                    resume=True,
                ),
            ),
            (
                "error breakdowns",
                lambda: error_breakdowns._render_csv_bytes(
                    [_row(error_breakdowns.WER_COLUMNS)],
                    error_breakdowns.WER_COLUMNS,
                ),
            ),
            (
                "comparison",
                lambda: comparison._csv_bytes(
                    [_row(comparison.COMPARISON_COLUMNS)]
                ),
            ),
            (
                "noise protocol",
                lambda: noise_protocol._csv_bytes(
                    [_row(noise_protocol.NOISE_AUDIT_COLUMNS)],
                    noise_protocol.NOISE_AUDIT_COLUMNS,
                ),
            ),
            (
                "final benchmark protocol",
                lambda: final_benchmark._csv_bytes(
                    [_row(final_benchmark.FINAL_AUDIT_COLUMNS)],
                    final_benchmark.FINAL_AUDIT_COLUMNS,
                ),
            ),
            (
                "VIVOS protocol",
                lambda: vivos_manifest._csv_bytes(
                    [_row(vivos_manifest.AUDIT_COLUMNS)]
                ),
            ),
            (
                "FLEURS protocol",
                lambda: download_fleurs._audit_bytes(
                    [_row(download_fleurs.AUDIT_FIELDS)]
                ),
            ),
        ]
        for label, render in renderers:
            with self.subTest(label=label):
                self.assert_lf_csv(render(), label=label)

    def test_prediction_hashes_bind_exact_published_lf_bytes(self) -> None:
        rows = [_row(phat_evaluation.PREDICTION_COLUMNS)]
        expected = phat_evaluation._prediction_csv_bytes(rows)
        self.assert_lf_csv(expected, label="prediction payload")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "prediction.csv"
            published_hash = phat_evaluation._atomic_write_prediction(
                prediction, rows
            )
            self.assertEqual(prediction.read_bytes(), expected)
            self.assertEqual(
                published_hash,
                hashlib.sha256(prediction.read_bytes()).hexdigest(),
            )
            self.assertEqual(published_hash, sha256_file(prediction))

            zero_shot_rows = [_row(zero_shot.PREDICTION_COLUMNS)]
            zero_shot_path = root / "zero-shot.csv"
            zero_shot_hash = zero_shot._atomic_write_csv(
                zero_shot_path, zero_shot_rows
            )
            self.assert_lf_csv(
                zero_shot_path.read_bytes(), label="zero-shot prediction"
            )
            self.assertEqual(zero_shot_hash, sha256_file(zero_shot_path))

            fleurs_rows = [_row(CANONICAL_PREDICTION_COLUMNS)]
            fleurs_path = root / "fleurs.csv"
            atomic_write_csv(
                fleurs_path,
                fleurs_rows,
                CANONICAL_PREDICTION_COLUMNS,
            )
            expected_fleurs = external_fleurs._prediction_csv_bytes(fleurs_rows)
            self.assertEqual(fleurs_path.read_bytes(), expected_fleurs)
            self.assert_lf_csv(expected_fleurs, label="FLEURS prediction")
            self.assertEqual(
                hashlib.sha256(expected_fleurs).hexdigest(),
                sha256_file(fleurs_path),
            )

    def test_shared_and_training_csv_writers_emit_lf_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared.csv"
            atomic_write_csv(
                shared,
                [{"utt_id": "a", "ref": "xin chào"}],
                ["utt_id", "ref"],
            )

            training = root / "training.csv"
            phat_trainer._write_metrics_header(training, append=False)
            phat_trainer._append_metrics(
                training,
                {
                    "epoch": 1,
                    "global_step": 2,
                    "learning_rate": 0.001,
                    "asr_loss": 1.0,
                    "tone_loss": 0.5,
                    "total_loss": 1.5,
                },
            )

            dev = root / "dev.csv"
            phat_trainer._write_dev_metrics_header(dev, append=False)
            phat_trainer._append_dev_metrics(
                dev,
                {
                    "epoch": 1,
                    "global_step": 2,
                    "num_samples": 3,
                    "asr_tokens": 4,
                    "tone_targets": 5,
                    "dev_asr_loss": 1.0,
                    "dev_tone_loss": 0.5,
                    "dev_total_loss": 1.5,
                    "is_best": True,
                },
            )
            for path in (shared, training, dev):
                with self.subTest(path=path.name):
                    self.assert_lf_csv(path.read_bytes(), label=path.name)

    def test_every_production_dict_writer_pins_lf(self) -> None:
        missing: list[str] = []
        invalid: list[str] = []
        for source_root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
            for path in source_root.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    function = node.func
                    if not (
                        isinstance(function, ast.Attribute)
                        and isinstance(function.value, ast.Name)
                        and function.value.id == "csv"
                        and function.attr in {"DictWriter", "writer"}
                    ):
                        continue
                    keyword = next(
                        (item for item in node.keywords if item.arg == "lineterminator"),
                        None,
                    )
                    location = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                    if keyword is None:
                        missing.append(location)
                    elif not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value == "\n"
                    ):
                        invalid.append(location)
        self.assertEqual(missing, [], f"CSV writers missing lineterminator: {missing}")
        self.assertEqual(invalid, [], f"CSV writers not pinned to LF: {invalid}")

    def test_gitattributes_prevents_checkout_rewriting_of_paper_csvs(self) -> None:
        attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("outputs/paper_v2/**/*.csv text eol=lf\n", attributes)


if __name__ == "__main__":
    unittest.main()
