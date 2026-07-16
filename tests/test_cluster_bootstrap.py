from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.vitonesr import statistics


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
BENCHMARK_COLUMNS = [
    "utt_id",
    "source_utt_id",
    "dataset",
    "split",
    "snr",
    "noise_type",
    "transcript",
]


class ClusterBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.observations = [
            {
                "utt_id": "s1_clean",
                "source_utt_id": "s1",
                "dataset": "vivos",
                "split": "test",
                "snr": "clean",
                "noise_type": "clean",
                "transcript": "m\u00e1",
            },
            {
                "utt_id": "s1_0db",
                "source_utt_id": "s1",
                "dataset": "vivos",
                "split": "test",
                "snr": "0",
                "noise_type": "speech",
                "transcript": "m\u00e1",
            },
            {
                "utt_id": "s2_clean",
                "source_utt_id": "s2",
                "dataset": "vivos",
                "split": "test",
                "snr": "clean",
                "noise_type": "clean",
                "transcript": "b\u00e0",
            },
            {
                "utt_id": "s2_0db",
                "source_utt_id": "s2",
                "dataset": "vivos",
                "split": "test",
                "snr": "0",
                "noise_type": "music",
                "transcript": "b\u00e0",
            },
        ]
        self.benchmark = self.root / "benchmark.csv"
        self._write_benchmark()
        self.decision = self.root / "decision.json"
        self.decision_object = self._decision_object()
        self._write_decision()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _configuration(
        configuration_id: str,
        role: str,
        method_id: str,
        train_type: str,
        lambda_value: float,
        seed: int = 42,
    ) -> dict[str, object]:
        return {
            "configuration_id": configuration_id,
            "role": role,
            "method_id": method_id,
            "train_type": train_type,
            "lambda": lambda_value,
            "seed": seed,
        }

    def _decision_object(self) -> dict[str, object]:
        return {
            "status": "LOCKED",
            "test_unlocked": True,
            "selected_method_id": "tone_objective_v2",
            "selected_lambda": 0.3,
            "locked_configurations": [
                self._configuration(
                    "ordinary_seed42",
                    "ordinary_baseline",
                    "ordinary_lora",
                    "ordinary_lora",
                    0.0,
                ),
                self._configuration(
                    "tone_selected_seed42",
                    "selected_method",
                    "tone_objective_v2",
                    "tone_aware_lora",
                    0.3,
                ),
                self._configuration(
                    "tone_control_seed42",
                    "locked_control",
                    "tone_objective_v2",
                    "tone_aware_lora",
                    0.1,
                ),
            ],
        }

    def _write_decision(self) -> None:
        self.decision.write_text(
            json.dumps(self.decision_object, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _write_benchmark(self) -> None:
        with self.benchmark.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BENCHMARK_COLUMNS)
            writer.writeheader()
            writer.writerows(self.observations)

    def _write_prediction(
        self,
        filename: str,
        *,
        train_type: str,
        lambda_value: str,
        hypotheses: list[str],
        seed: str = "42",
        observations: list[dict[str, str]] | None = None,
    ) -> Path:
        source_rows = observations or self.observations
        path = self.root / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for observation, hyp in zip(source_rows, hypotheses):
                writer.writerow(
                    {
                        "utt_id": observation["utt_id"],
                        "dataset": observation["dataset"],
                        "model": "phowhisper",
                        "model_size": "base",
                        "train_type": train_type,
                        "lambda": lambda_value,
                        "seed": seed,
                        "snr": observation["snr"],
                        "noise_type": observation["noise_type"],
                        "ref": observation["transcript"],
                        "hyp": hyp,
                    }
                )
        return path

    def _prediction_paths(
        self,
        *,
        ordinary: list[str] | None = None,
        selected: list[str] | None = None,
        control: list[str] | None = None,
    ) -> dict[str, Path]:
        refs = [row["transcript"] for row in self.observations]
        return {
            "ordinary_seed42": self._write_prediction(
                "ordinary.csv",
                train_type="ordinary_lora",
                lambda_value="0",
                hypotheses=ordinary or refs,
            ),
            "tone_selected_seed42": self._write_prediction(
                "selected.csv",
                train_type="tone_aware_lora",
                lambda_value="0.3",
                hypotheses=selected or refs,
            ),
            "tone_control_seed42": self._write_prediction(
                "control.csv",
                train_type="tone_aware_lora",
                lambda_value="0.1",
                hypotheses=control or refs,
            ),
        }

    def _load(
        self, paths: dict[str, Path] | None = None, **kwargs: object
    ) -> statistics.ClusterBootstrapInputs:
        return statistics.load_cluster_bootstrap_inputs(
            self.decision,
            self.benchmark,
            paths or self._prediction_paths(),
            **kwargs,
        )

    def test_dynamic_selected_lambda_roles_and_three_pairs(self) -> None:
        inputs = self._load()
        rows = statistics.build_cluster_bootstrap_rows(
            inputs, n_bootstrap=25, bootstrap_seed=7
        )

        self.assertEqual(len(rows), 12)
        self.assertEqual(tuple(rows[0]), statistics.OUTPUT_COLUMNS)
        self.assertEqual(
            {row["pair_id"] for row in rows},
            {pair[0] for pair in statistics.PAIR_SPECS},
        )
        selected_rows = [
            row
            for row in rows
            if row["role_a"] == "selected_method"
            or row["role_b"] == "selected_method"
        ]
        self.assertTrue(selected_rows)
        for row in selected_rows:
            suffix = "a" if row["role_a"] == "selected_method" else "b"
            self.assertEqual(row[f"lambda_{suffix}"], "0.3")
            self.assertEqual(row[f"configuration_id_{suffix}"], "tone_selected_seed42")
        self.assertTrue(all(row["n_bootstrap"] == 25 for row in rows))
        self.assertTrue(all(row["bootstrap_unit"] == "source_utt_id" for row in rows))

    def test_jsonl_final_manifest_preserves_source_clusters(self) -> None:
        jsonl = self.root / "benchmark.jsonl"
        jsonl.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in self.observations
            ),
            encoding="utf-8",
        )
        inputs = statistics.load_cluster_bootstrap_inputs(
            self.decision,
            jsonl,
            self._prediction_paths(),
        )
        rows = statistics.build_cluster_bootstrap_rows(inputs, n_bootstrap=10)
        self.assertEqual(inputs.bootstrap_unit, "source_utt_id")
        self.assertTrue(all(row["n_source_clusters"] == 2 for row in rows))
        self.assertTrue(all(row["bootstrap_unit"] == "source_utt_id" for row in rows))

    def test_external_jsonl_requires_explicit_singleton_policy(self) -> None:
        external = [
            {
                "utt_id": "fleurs-1",
                "dataset": "fleurs",
                "split": "test",
                "snr": "clean",
                "noise_type": "clean",
                "ref": "má",
            },
            {
                "utt_id": "fleurs-2",
                "dataset": "fleurs",
                "split": "test",
                "snr": "clean",
                "noise_type": "clean",
                "ref": "bà",
            },
        ]
        manifest = self.root / "fleurs.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in external
            ),
            encoding="utf-8",
        )
        role_specs = (
            ("ordinary_seed42", "ordinary_lora", "0"),
            ("tone_selected_seed42", "tone_aware_lora", "0.3"),
            ("tone_control_seed42", "tone_aware_lora", "0.1"),
        )
        paths = {
            configuration_id: self._write_prediction(
                f"{configuration_id}_fleurs.csv",
                train_type=train_type,
                lambda_value=lambda_value,
                hypotheses=["má", "bà"],
                observations=[
                    {**row, "transcript": row["ref"]} for row in external
                ],
            )
            for configuration_id, train_type, lambda_value in role_specs
        }
        with self.assertRaisesRegex(
            statistics.ClusterBootstrapError, "empty paired field"
        ):
            statistics.load_cluster_bootstrap_inputs(
                self.decision, manifest, paths
            )

        inputs = statistics.load_cluster_bootstrap_inputs(
            self.decision,
            manifest,
            paths,
            cluster_unit="utt_id_singleton_external",
        )
        rows = statistics.build_cluster_bootstrap_rows(inputs, n_bootstrap=10)
        self.assertEqual(inputs.bootstrap_unit, "utt_id_singleton_external")
        self.assertTrue(all(row["n_source_clusters"] == 2 for row in rows))
        self.assertTrue(
            all(row["bootstrap_unit"] == "utt_id_singleton_external" for row in rows)
        )

    def test_vivos_cannot_opt_out_of_source_cluster_bootstrap(self) -> None:
        with self.assertRaisesRegex(
            statistics.ClusterBootstrapError,
            "VIVOS final evaluation requires source_utt_id",
        ):
            self._load(cluster_unit="utt_id_singleton_external")

    def test_cluster_sampling_keeps_clean_and_noisy_replicas_together(self) -> None:
        paths = self._prediction_paths(
            selected=["ma", "ma", "b\u00e0", "b\u00e0"]
        )
        inputs = self._load(paths)
        rows = statistics.build_cluster_bootstrap_rows(
            inputs,
            n_bootstrap=200,
            bootstrap_seed=11,
            chunk_size=13,
        )
        wer = next(
            row
            for row in rows
            if row["pair_id"] == "ordinary_baseline_vs_selected_method"
            and row["metric"] == "wer"
        )

        rng = np.random.default_rng(11)
        sampled = rng.integers(0, 2, size=(200, 2), dtype=np.int64)
        expected_deltas = (sampled == 0).sum(axis=1) / 2.0
        expected_low, expected_high = np.quantile(
            expected_deltas, [0.025, 0.975], method="linear"
        )
        self.assertEqual(wer["n_source_clusters"], 2)
        self.assertEqual(wer["n_paired_conditions"], 4)
        self.assertEqual(wer["numerator_b"], 2)
        self.assertEqual(wer["denominator_b"], 4)
        self.assertEqual(wer["estimate_b"], "0.500000000000")
        self.assertEqual(wer["ci_lower"], f"{expected_low:.12f}")
        self.assertEqual(wer["ci_upper"], f"{expected_high:.12f}")

    def test_ratio_of_totals_and_metric_coverage_are_auditable(self) -> None:
        # One selected lexical substitution is ineligible for TER/DER while the
        # other three observations remain eligible.
        paths = self._prediction_paths(selected=["xe", "m\u00e1", "b\u00e0", "b\u00e0"])
        inputs = self._load(paths)
        rows = statistics.build_cluster_bootstrap_rows(inputs, n_bootstrap=40)
        der = next(
            row
            for row in rows
            if row["pair_id"] == "ordinary_baseline_vs_selected_method"
            and row["metric"] == "der"
        )
        self.assertEqual(der["coverage_numerator_b"], 3)
        self.assertEqual(der["coverage_denominator_b"], 4)
        self.assertEqual(der["coverage_b"], "0.750000000000")
        self.assertEqual(der["denominator_b"], 3)
        self.assertEqual(der["n_valid_bootstrap"], 40)

    def test_rejects_ambiguous_multiseed_without_explicit_comparison_set(self) -> None:
        self.decision_object["locked_configurations"].append(
            self._configuration(
                "ordinary_seed7",
                "ordinary_baseline",
                "ordinary_lora",
                "ordinary_lora",
                0.0,
                seed=7,
            )
        )
        self._write_decision()
        paths = self._prediction_paths()
        with self.assertRaisesRegex(statistics.ClusterBootstrapError, "ambiguous"):
            self._load(paths)

        comparison_set = {
            "ordinary_baseline": "ordinary_seed42",
            "selected_method": "tone_selected_seed42",
            "locked_control": "tone_control_seed42",
        }
        inputs = self._load(paths, comparison_set=comparison_set)
        self.assertEqual(
            inputs.runs["ordinary_baseline"].configuration.configuration_id,
            "ordinary_seed42",
        )

    def test_rejects_reference_condition_and_id_mismatch(self) -> None:
        paths = self._prediction_paths()
        changed = [dict(row) for row in self.observations]
        changed[0]["transcript"] = "kh\u00e1c"
        paths["tone_selected_seed42"] = self._write_prediction(
            "selected_bad_ref.csv",
            train_type="tone_aware_lora",
            lambda_value="0.3",
            hypotheses=[row["transcript"] for row in changed],
            observations=changed,
        )
        with self.assertRaisesRegex(statistics.ClusterBootstrapError, "ref differs"):
            self._load(paths)

        paths = self._prediction_paths()
        changed = [dict(row) for row in self.observations]
        changed[1]["noise_type"] = "babble"
        paths["tone_selected_seed42"] = self._write_prediction(
            "selected_bad_condition.csv",
            train_type="tone_aware_lora",
            lambda_value="0.3",
            hypotheses=[row["transcript"] for row in changed],
            observations=changed,
        )
        with self.assertRaisesRegex(statistics.ClusterBootstrapError, "condition differs"):
            self._load(paths)

        paths = self._prediction_paths()
        short_observations = self.observations[:-1]
        paths["tone_selected_seed42"] = self._write_prediction(
            "selected_missing_id.csv",
            train_type="tone_aware_lora",
            lambda_value="0.3",
            hypotheses=[row["transcript"] for row in short_observations],
            observations=short_observations,
        )
        with self.assertRaisesRegex(statistics.ClusterBootstrapError, "utt_id set differs"):
            self._load(paths)

    def test_default_1000_is_deterministic_across_chunk_sizes(self) -> None:
        inputs = self._load(
            self._prediction_paths(selected=["ma", "ma", "b\u00e0", "b\u00e0"])
        )
        first = statistics.build_cluster_bootstrap_rows(
            inputs, bootstrap_seed=42, chunk_size=1
        )
        second = statistics.build_cluster_bootstrap_rows(
            inputs, bootstrap_seed=42, chunk_size=127
        )
        self.assertEqual(first, second)
        self.assertTrue(all(row["n_bootstrap"] == 1000 for row in first))
        self.assertTrue(all(row["n_valid_bootstrap"] == 1000 for row in first))

    def test_provenance_atomic_write_and_no_overwrite(self) -> None:
        paths = self._prediction_paths()
        inputs = self._load(paths)
        rows = statistics.build_cluster_bootstrap_rows(inputs, n_bootstrap=10)
        first = rows[0]
        self.assertEqual(first["decision_sha256"], statistics.sha256_file(self.decision))
        self.assertEqual(first["benchmark_sha256"], statistics.sha256_file(self.benchmark))
        self.assertEqual(len(first["comparison_set_sha256"]), 64)
        self.assertEqual(
            first["prediction_sha256_a"],
            statistics.sha256_file(
                inputs.runs[first["role_a"]].prediction_path
            ),
        )

        output = self.root / "results" / "bootstrap.csv"
        statistics.write_cluster_bootstrap_csv(
            output,
            rows,
            protected_inputs=[self.decision, self.benchmark, *paths.values()],
        )
        content = output.read_bytes()
        with self.assertRaisesRegex(statistics.ClusterBootstrapError, "already exists"):
            statistics.write_cluster_bootstrap_csv(output, rows)
        self.assertEqual(output.read_bytes(), content)
        self.assertFalse(list(output.parent.glob("*.tmp")))
        with self.assertRaisesRegex(statistics.ClusterBootstrapError, "input artifact"):
            statistics.write_cluster_bootstrap_csv(
                self.decision,
                rows,
                overwrite=True,
                protected_inputs=[self.decision],
            )

    def test_rejects_invalid_parameters(self) -> None:
        inputs = self._load()
        for kwargs in (
            {"n_bootstrap": 0},
            {"ci_level": 0.0},
            {"ci_level": 1.0},
            {"bootstrap_seed": -1},
            {"chunk_size": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(statistics.ClusterBootstrapError):
                    statistics.build_cluster_bootstrap_rows(inputs, **kwargs)


if __name__ == "__main__":
    unittest.main()
