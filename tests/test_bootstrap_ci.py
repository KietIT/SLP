from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import bootstrap_ci as bootstrap


PREDICTION_COLUMNS = [
    "utt_id",
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "snr",
    "noise_type",
    "ref",
    "hyp",
]


class BootstrapCITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_run(
        self,
        filename: str,
        *,
        train_type: str,
        lambda_value: str,
        hypotheses: list[str],
        refs: list[str] | None = None,
        ids: list[str] | None = None,
    ) -> Path:
        refs = refs or ["má"] * len(hypotheses)
        ids = ids or [f"utt_{index}" for index in range(len(hypotheses))]
        path = self.root / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for utt_id, ref, hyp in zip(ids, refs, hypotheses):
                writer.writerow(
                    {
                        "utt_id": utt_id,
                        "dataset": "fleurs",
                        "model": "phowhisper",
                        "model_size": "base",
                        "train_type": train_type,
                        "lambda": lambda_value,
                        "seed": "42",
                        "snr": "clean",
                        "noise_type": "clean",
                        "ref": ref,
                        "hyp": hyp,
                    }
                )
        return path

    def _three_runs(
        self,
        ordinary_hyp: list[str],
        lambda_005_hyp: list[str],
        lambda_01_hyp: list[str],
        *,
        refs: list[str] | None = None,
    ) -> tuple[Path, Path, Path]:
        return (
            self._write_run(
                "ordinary.csv",
                train_type="ordinary_lora",
                lambda_value="0",
                hypotheses=ordinary_hyp,
                refs=refs,
            ),
            self._write_run(
                "lambda_005.csv",
                train_type="tone_aware_lora",
                lambda_value="0.05",
                hypotheses=lambda_005_hyp,
                refs=refs,
            ),
            self._write_run(
                "lambda_01.csv",
                train_type="tone_aware_lora",
                lambda_value="0.1",
                hypotheses=lambda_01_hyp,
                refs=refs,
            ),
        )

    def test_builds_twelve_rows_with_b_minus_a_sign(self) -> None:
        paths = self._three_runs(["ma"], ["má"], ["mà"])
        runs, utt_ids = bootstrap.load_paired_runs(*paths, expected_rows=1)
        rows = bootstrap.build_bootstrap_rows(
            runs, utt_ids, n_bootstrap=25, bootstrap_seed=7
        )

        self.assertEqual(len(rows), 12)
        self.assertEqual(list(rows[0]), bootstrap.OUTPUT_COLUMNS)
        keys = {(row["pair_id"], row["metric"]) for row in rows}
        self.assertEqual(len(keys), 12)
        wer = next(
            row
            for row in rows
            if row["pair_id"] == "ordinary_vs_lambda_005"
            and row["metric"] == "wer"
        )
        self.assertEqual(wer["delta_b_minus_a"], "-1.000000000000")
        self.assertEqual(wer["ci_lower"], "-1.000000000000")
        self.assertEqual(wer["ci_upper"], "-1.000000000000")
        self.assertEqual(wer["ci_excludes_zero"], "true")

    def test_point_estimate_is_ratio_of_totals_not_mean_of_rates(self) -> None:
        refs = ["một", "một hai ba bốn"]
        paths = self._three_runs(
            ["", "một hai ba bốn"],
            ["một", "một hai ba"],
            ["một", "một hai ba bốn"],
            refs=refs,
        )
        runs, utt_ids = bootstrap.load_paired_runs(*paths, expected_rows=2)
        rows = bootstrap.build_bootstrap_rows(runs, utt_ids, n_bootstrap=10)
        wer = next(
            row
            for row in rows
            if row["pair_id"] == "ordinary_vs_lambda_005"
            and row["metric"] == "wer"
        )
        self.assertEqual(wer["numerator_a"], 1)
        self.assertEqual(wer["denominator_a"], 5)
        self.assertEqual(wer["estimate_a"], "0.200000000000")

    def test_identical_hypotheses_have_zero_delta_and_interval(self) -> None:
        paths = self._three_runs(["ma", "má"], ["ma", "má"], ["ma", "má"])
        runs, utt_ids = bootstrap.load_paired_runs(*paths, expected_rows=2)
        rows = bootstrap.build_bootstrap_rows(
            runs, utt_ids, n_bootstrap=50, bootstrap_seed=9
        )
        self.assertTrue(
            all(row["delta_b_minus_a"] == "0.000000000000" for row in rows)
        )
        self.assertTrue(all(row["ci_lower"] == "0.000000000000" for row in rows))
        self.assertTrue(all(row["ci_upper"] == "0.000000000000" for row in rows))

    def test_rejects_mismatched_ids_and_references(self) -> None:
        ordinary, candidate, other = self._three_runs(["má"], ["má"], ["má"])
        self._write_run(
            candidate.name,
            train_type="tone_aware_lora",
            lambda_value="0.05",
            hypotheses=["má"],
            ids=["different"],
        )
        with self.assertRaisesRegex(bootstrap.BootstrapError, "utt_id set differs"):
            bootstrap.load_paired_runs(
                ordinary, candidate, other, expected_rows=1
            )

        self._write_run(
            candidate.name,
            train_type="tone_aware_lora",
            lambda_value="0.05",
            hypotheses=["má"],
            refs=["khác"],
        )
        with self.assertRaisesRegex(bootstrap.BootstrapError, "ref differs"):
            bootstrap.load_paired_runs(
                ordinary, candidate, other, expected_rows=1
            )

    def test_rejects_wrong_run_contract_and_conditions(self) -> None:
        ordinary, candidate, other = self._three_runs(["má"], ["má"], ["má"])
        self._write_run(
            candidate.name,
            train_type="tone_aware_lora",
            lambda_value="0.3",
            hypotheses=["má"],
        )
        with self.assertRaisesRegex(bootstrap.BootstrapError, "requires lambda"):
            bootstrap.load_paired_runs(
                ordinary, candidate, other, expected_rows=1
            )

    def test_output_is_deterministic_atomic_and_no_overwrite(self) -> None:
        paths = self._three_runs(["ma", "má"], ["má", "má"], ["mà", "má"])
        first = self.root / "first.csv"
        second = self.root / "second.csv"
        bootstrap.run_bootstrap(
            *paths,
            first,
            expected_rows=2,
            n_bootstrap=50,
            bootstrap_seed=123,
        )
        bootstrap.run_bootstrap(
            *paths,
            second,
            expected_rows=2,
            n_bootstrap=50,
            bootstrap_seed=123,
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with self.assertRaisesRegex(bootstrap.BootstrapError, "already exists"):
            bootstrap.run_bootstrap(
                *paths,
                first,
                expected_rows=2,
                n_bootstrap=10,
            )
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_rejects_invalid_bootstrap_parameters(self) -> None:
        paths = self._three_runs(["má"], ["má"], ["má"])
        runs, utt_ids = bootstrap.load_paired_runs(*paths, expected_rows=1)
        for kwargs in (
            {"n_bootstrap": 0},
            {"ci_level": 0.0},
            {"ci_level": 1.0},
            {"bootstrap_seed": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(bootstrap.BootstrapError):
                    bootstrap.build_bootstrap_rows(runs, utt_ids, **kwargs)


if __name__ == "__main__":
    unittest.main()
