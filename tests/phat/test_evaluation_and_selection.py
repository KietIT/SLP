from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.vitonesr.phat.evaluation import PREDICTION_COLUMNS, validate_prediction_schema
from src.vitonesr.phat.selection import select_best_lambda_from_rows
from src.vitonesr.prediction import atomic_write_csv


class EvaluationAndSelectionTests(unittest.TestCase):
    def test_prediction_csv_has_exact_schema(self) -> None:
        row = {
            "utt_id": "utt-1",
            "dataset": "vivos",
            "model": "phowhisper",
            "model_size": "base",
            "train_type": "ordinary_lora",
            "lambda": "0",
            "seed": "42",
            "snr": "clean",
            "noise_type": "clean",
            "ref": "xin chào",
            "hyp": "xin chào",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.csv"
            atomic_write_csv(path, [row], PREDICTION_COLUMNS)
            validated = validate_prediction_schema(path)
            self.assertEqual(validated[0]["utt_id"], "utt-1")
            with path.open("r", encoding="utf-8", newline="") as prediction_file:
                self.assertEqual(list(csv.DictReader(prediction_file).fieldnames or []), PREDICTION_COLUMNS)

    def test_best_lambda_prioritizes_low_snr_with_guards(self) -> None:
        rows: list[dict[str, str]] = []
        values = {
            0.0: (0.30, 0.18, 0.24, 0.20),
            0.05: (0.31, 0.18, 0.20, 0.17),
            0.1: (0.32, 0.19, 0.18, 0.15),
            0.3: (0.38, 0.25, 0.16, 0.14),
            0.5: (0.34, 0.20, 0.19, 0.16),
        }
        for lambda_value, (wer, cer, ter, der) in values.items():
            train_type = "ordinary_lora" if lambda_value == 0 else "tone_aware_lora"
            common = {
                "model": "phowhisper",
                "model_size": "base",
                "train_type": train_type,
                "lambda": str(lambda_value),
                "seed": "42",
                "noise_type": "all",
                "num_samples": "100",
                "wer": str(wer),
                "cer": str(cer),
                "ter": str(ter),
                "der": str(der),
                "fcer": "0.1",
                "swdr": "0.1",
                "checkpoint_path": "checkpoint",
                "prediction_path": "prediction",
            }
            rows.append({**common, "split": "all", "snr": "all"})
            rows.append(
                {
                    **common,
                    "split": "clean",
                    "snr": "clean",
                    "noise_type": "clean",
                    "num_samples": "50",
                }
            )
            rows.append({**common, "split": "noisy", "snr": "0", "num_samples": "50"})
            rows.append({**common, "split": "noisy", "snr": "5", "num_samples": "50"})
        result = select_best_lambda_from_rows(
            rows,
            {
                "low_snr": [0, 5],
                "ter_weight": 0.5,
                "der_weight": 0.5,
                "max_wer_absolute_increase": 0.05,
                "max_cer_absolute_increase": 0.03,
                "guard_split": "all",
                "guard_snr": "all",
                "allow_lambda_zero": True,
            },
        )
        self.assertEqual(result.selected_lambda, 0.1)
        lambda_03 = next(summary for summary in result.summaries if summary.lambda_value == 0.3)
        self.assertFalse(lambda_03.eligible)


if __name__ == "__main__":
    unittest.main()
