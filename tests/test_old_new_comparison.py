from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.vitonesr.comparison as comparison_module
from src.vitonesr.comparison import (
    COMPARISON_COLUMNS,
    ComparisonError,
    ComparisonInputs,
    build_comparison,
    sha256_file,
    write_comparison,
)


METRICS = ("wer", "cer", "ter", "der", "fcer", "swdr")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _metric_values(base: float) -> dict[str, float]:
    return {metric: base + index / 1000 for index, metric in enumerate(METRICS)}


class ComparisonFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.old_runs = [
            ("whisper", "tiny", "zero_shot", "", "42"),
            ("whisper", "small", "zero_shot", "", "42"),
            ("whisper", "base", "zero_shot", "", "42"),
            ("phowhisper", "tiny", "zero_shot", "", "42"),
            ("phowhisper", "small", "zero_shot", "", "42"),
            ("phowhisper", "base", "zero_shot", "", "42"),
            ("phowhisper", "base", "ordinary_lora", "0", "42"),
            ("phowhisper", "base", "tone_aware_lora", "0.05", "42"),
            ("phowhisper", "base", "tone_aware_lora", "0.1", "42"),
            ("phowhisper", "base", "tone_aware_lora", "0.3", "42"),
            ("phowhisper", "base", "tone_aware_lora", "0.5", "42"),
        ]
        self.new_runs = self.old_runs[:7] + [self.old_runs[9], self.old_runs[8]]
        self.roles = {
            "ordinary_baseline": self.new_runs[6],
            "selected_method": self.new_runs[7],
            "locked_control": self.new_runs[8],
        }

    def _aggregate(self, runs: list[tuple[str, str, str, str, str]], groups: list[str], group_column: str, n: int, base: float) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for run_index, (model, size, train_type, lambda_value, seed) in enumerate(runs):
            for group_index, group in enumerate(groups):
                count = n
                if group == "noisy_all":
                    count = n * 4
                elif group == "all":
                    count = n * 5
                row: dict[str, object] = {
                    "dataset": "vivos", "model": model, "model_size": size,
                    "train_type": train_type, "lambda": lambda_value, "seed": seed,
                    group_column: group, "n": count,
                    **_metric_values(base + run_index / 100 + group_index / 10000),
                    "metric_version": "aligned_v1",
                }
                rows.append(row)
        return rows

    def _fleurs_results(self, *, new: bool) -> list[dict[str, object]]:
        rows = []
        for index, run in enumerate(self.roles.values()):
            model, size, train_type, lambda_value, seed = run
            base = (0.10 if new else 0.12) + index / 100
            rows.append({
                "dataset": "fleurs", "model": model, "model_size": size,
                "train_type": train_type, "lambda": lambda_value, "seed": seed,
                "n": 857, **_metric_values(base), "metric_version": "aligned_v1",
            })
        return rows

    def _predictions(self, directory: Path, *, prefix: str) -> list[tuple[str, str, Path]]:
        output = []
        for role, run in self.roles.items():
            model, size, train_type, lambda_value, seed = run
            path = directory / f"pred_{prefix}_{role}.csv"
            rows = [
                {
                    "utt_id": f"fleurs_{index:04d}", "dataset": "fleurs",
                    "model": model, "model_size": size, "train_type": train_type,
                    "lambda": lambda_value, "seed": seed, "snr": "clean",
                    "noise_type": "clean", "ref": f"câu tham chiếu {index}",
                    "hyp": f"câu giả thuyết {index}",
                }
                for index in range(857)
            ]
            _write_csv(path, rows)
            output.append((role, f"{train_type}_{lambda_value or 'blank'}", path))
        return output

    def _bootstrap(self, *, new: bool, final: bool = False, decision_sha256: str = "d" * 64, benchmark_sha256: str = "b" * 64) -> list[dict[str, object]]:
        pairs = (("ordinary_baseline", "selected_method"), ("ordinary_baseline", "locked_control"), ("selected_method", "locked_control"))
        rows = []
        for pair_index, (role_a, role_b) in enumerate(pairs):
            a = self.roles[role_a]
            b = self.roles[role_b]
            for metric in ("wer", "cer", "ter", "der"):
                common = {
                    "metric_version": "aligned_v1", "dataset": "vivos" if final else "fleurs",
                    "split": "test", "pair_id": f"{role_a}_vs_{role_b}",
                    "train_type_a": a[2], "lambda_a": a[3], "seed_a": a[4],
                    "train_type_b": b[2], "lambda_b": b[3], "seed_b": b[4],
                    "metric": metric, "delta_b_minus_a": 0.01 + pair_index / 100,
                    "n_bootstrap": 1000, "ci_level": 0.95, "ci_lower": -0.01,
                    "ci_upper": 0.02, "ci_excludes_zero": "false", "bootstrap_seed": 42,
                }
                if new:
                    rows.append({
                        **common, "decision_sha256": decision_sha256, "benchmark_sha256": benchmark_sha256,
                        "comparison_set_sha256": "c" * 64, "role_a": role_a,
                        "configuration_id_a": f"cfg_{role_a}", "method_id_a": a[2],
                        "prediction_sha256_a": "a" * 64, "role_b": role_b,
                        "configuration_id_b": f"cfg_{role_b}", "method_id_b": b[2],
                        "prediction_sha256_b": "e" * 64, "n_source_clusters": 460 if final else 857,
                        "n_paired_conditions": 2300 if final else 857, "numerator_a": 1,
                        "denominator_a": 10, "coverage_numerator_a": 10,
                        "coverage_denominator_a": 10, "coverage_a": 1, "estimate_a": 0.1,
                        "numerator_b": 2, "denominator_b": 10, "coverage_numerator_b": 10,
                        "coverage_denominator_b": 10, "coverage_b": 1, "estimate_b": 0.2,
                        "n_valid_bootstrap": 1000,
                        "bootstrap_unit": "source_utt_id" if final else "utt_id_singleton_external",
                        "ci_method": "paired_cluster_percentile_ratio_of_totals",
                    })
                else:
                    rows.append({
                        **common, "model_a": a[0], "model_size_a": a[1],
                        "model_b": b[0], "model_size_b": b[1], "n_paired": 857,
                        "numerator_a": 1, "denominator_a": 10, "estimate_a": 0.1,
                        "numerator_b": 2, "denominator_b": 10, "estimate_b": 0.2,
                        "bootstrap_unit": "utt_id", "ci_method": "paired_percentile",
                    })
        return rows

    def materialize(self) -> ComparisonInputs:
        old_snr = self.root / "old/results_by_snr.csv"
        old_noise = self.root / "old/results_by_noise_type.csv"
        new_snr = self.root / "new/results_by_snr.csv"
        new_noise = self.root / "new/results_by_noise_type.csv"
        _write_csv(old_snr, self._aggregate(self.old_runs, ["clean", "20", "10", "5", "0", "noisy_all", "all"], "snr", 300, 0.2))
        _write_csv(new_snr, self._aggregate(self.new_runs, ["clean", "20", "10", "5", "0", "noisy_all", "all"], "snr", 460, 0.1))
        _write_csv(old_noise, self._aggregate(self.old_runs, ["clean", "music", "noise", "speech"], "noise_type", 300, 0.2))
        _write_csv(new_noise, self._aggregate(self.new_runs, ["clean", "music", "noise", "speech"], "noise_type", 460, 0.1))

        old_benchmark = self.root / "old/benchmark.csv"
        _write_csv(old_benchmark, [{"utt_id": f"old_{index}"} for index in range(1500)])
        new_benchmark = self.root / "new/benchmark.jsonl"
        new_benchmark.parent.mkdir(parents=True, exist_ok=True)
        new_benchmark.write_text("".join(json.dumps({"utt_id": f"new_{index}"}) + "\n" for index in range(2300)), encoding="utf-8")

        fleurs_manifest = self.root / "fleurs.jsonl"
        fleurs_manifest.write_text("".join(json.dumps({"utt_id": f"fleurs_{index:04d}", "transcript": f"câu tham chiếu {index}"}, ensure_ascii=False) + "\n" for index in range(857)), encoding="utf-8")
        old_fleurs = self.root / "old/fleurs_results.csv"
        new_fleurs = self.root / "new/fleurs_results.csv"
        _write_csv(old_fleurs, self._fleurs_results(new=False))
        _write_csv(new_fleurs, self._fleurs_results(new=True))
        old_predictions = self._predictions(self.root / "old/predictions", prefix="old")
        new_predictions = self._predictions(self.root / "new/predictions", prefix="new")

        decision = self.root / "protocol/decision.json"
        decision_object = {
            "status": "LOCKED", "selection_complete": True, "test_unlocked": True,
            "selected_lambda": 0.3,
            "locked_control_lambda": 0.1,
            "locked_configurations": [
                {"role": role, "configuration_id": f"cfg_{role}", "method_id": run[2],
                 "train_type": run[2], "lambda": run[3], "seed": int(run[4])}
                for role, run in self.roles.items()
            ],
        }
        identity_payload = json.dumps(decision_object, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        decision_object["identity_sha256"] = hashlib.sha256(identity_payload).hexdigest()
        _write_json(decision, decision_object)

        provenance = self.root / "new/fleurs_results.csv.provenance.json"
        _write_json(provenance, {
            "evaluation_domain": "legacy_exposed_external_replication",
            "evaluation_scope": "full_fleurs_857", "results_sha256": sha256_file(new_fleurs),
            "manifest_sha256": sha256_file(fleurs_manifest), "decision_lock_sha256": sha256_file(decision),
            "runs": [
                {"role": role, "prediction": str(path), "prediction_sha256": sha256_file(path)}
                for role, _, path in new_predictions
            ],
        })

        old_bootstrap = self.root / "old/bootstrap.csv"
        new_bootstrap = self.root / "new/bootstrap_fleurs.csv"
        final_bootstrap = self.root / "new/bootstrap_final.csv"
        _write_csv(old_bootstrap, self._bootstrap(new=False))
        _write_csv(new_bootstrap, self._bootstrap(new=True, decision_sha256=sha256_file(decision), benchmark_sha256=sha256_file(fleurs_manifest)))
        _write_csv(final_bootstrap, self._bootstrap(new=True, final=True, decision_sha256=sha256_file(decision), benchmark_sha256=sha256_file(new_benchmark)))

        split = self.root / "protocol/split.json"
        _write_json(split, {
            "official_test": {
                "legacy_exposed_utterance_count": 300, "unseen_locked_utterance_count": 460,
                "exposure_evidence": {"benchmark_manifest_sha256": sha256_file(old_benchmark)},
            }
        })
        noisy_dev = self.root / "protocol/noisy_dev.json"
        _write_json(noisy_dev, {"output": {"row_count": 14125}})
        final_lock = self.root / "protocol/final.json"
        _write_json(final_lock, {"output": {"manifest_sha256": sha256_file(new_benchmark)}})
        fleurs_lock = self.root / "protocol/fleurs.json"
        _write_json(fleurs_lock, {
            "status": "LOCKED",
            "output": {"manifest_sha256": sha256_file(fleurs_manifest), "row_count": 857},
        })
        other_locks = []
        for name in ("noise", "environment", "method"):
            path = self.root / f"protocol/{name}.json"
            _write_json(path, {"status": "LOCKED"})
            other_locks.append(path)

        return ComparisonInputs(
            repo_root=self.root, old_by_snr=old_snr, old_by_noise_type=old_noise,
            old_fleurs_results=old_fleurs, old_fleurs_bootstrap=old_bootstrap,
            old_benchmark_manifest=old_benchmark,
            old_fleurs_predictions_dir=self.root / "old/predictions",
            fleurs_manifest=fleurs_manifest, fleurs_preparation_lock=fleurs_lock,
            new_by_snr=new_snr,
            new_by_noise_type=new_noise, new_fleurs_results=new_fleurs,
            new_fleurs_provenance=provenance, new_fleurs_bootstrap=new_bootstrap,
            new_final_bootstrap=final_bootstrap, new_benchmark_manifest=new_benchmark,
            decision_lock=decision, split_lock=split, noise_split_lock=other_locks[0],
            noisy_dev_lock=noisy_dev, environment_lock=other_locks[1],
            method_lock=other_locks[2], final_benchmark_lock=final_lock,
        )


class OldNewComparisonTests(unittest.TestCase):
    def test_formal_comparison_is_dynamic_and_scope_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = ComparisonFixture(Path(temporary)).materialize()
            bundle = build_comparison(inputs)
            self.assertEqual(bundle.provenance["selected_lambda"], "0.3")
            self.assertEqual(bundle.provenance["locked_control_lambda"], "0.1")
            self.assertTrue(bundle.provenance["fleurs_row_identity_verified"])
            selected = [row for row in bundle.rows if row["section"] == "fleurs_metrics" and "role=selected_method" in row["run_identity"] and row["metric"] == "wer"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(float(selected[0]["delta_new_minus_old"]), -0.02)
            vivos = [row for row in bundle.rows if row["section"] == "vivos_aggregate"]
            self.assertTrue(vivos)
            self.assertTrue(all(not row["delta_new_minus_old"] for row in vivos))
            self.assertTrue(all(row["comparability"].startswith("not_comparable") for row in vivos))

    def test_formal_fails_closed_but_diagnostic_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = ComparisonFixture(Path(temporary)).materialize()
            inputs.new_by_snr.unlink()
            with self.assertRaisesRegex(ComparisonError, "formal comparison is incomplete"):
                build_comparison(inputs)
            bundle = build_comparison(inputs, diagnostic_allow_partial=True)
            missing = [row for row in bundle.rows if row["section"] == "diagnostic_missing_artifacts"]
            self.assertEqual([row["artifact"] for row in missing], ["new_by_snr"])
            self.assertEqual(bundle.provenance["mode"], "diagnostic_allow_partial")

    def test_fleurs_identity_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = ComparisonFixture(Path(temporary)).materialize()
            prediction = next((inputs.old_fleurs_predictions_dir).glob("pred_*.csv"))
            with prediction.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["ref"] = "khác"
            _write_csv(prediction, rows)
            with self.assertRaisesRegex(ComparisonError, "identity differs from manifest"):
                build_comparison(inputs)

    def test_atomic_outputs_are_hashed_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_comparison(ComparisonFixture(root).materialize())
            csv_path = root / "report/comparison.csv"
            md_path = root / "report/comparison.md"
            provenance_path = root / "report/comparison.provenance.json"
            write_comparison(bundle, csv_path=csv_path, markdown_path=md_path, provenance_path=provenance_path)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(tuple(csv.DictReader(handle).fieldnames or ()), COMPARISON_COLUMNS)
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(provenance["comparison_csv_sha256"], sha256_file(csv_path))
            self.assertEqual(provenance["comparison_markdown_sha256"], sha256_file(md_path))
            with self.assertRaises(FileExistsError):
                write_comparison(bundle, csv_path=csv_path, markdown_path=md_path, provenance_path=provenance_path)

    @staticmethod
    def _output_paths(root: Path) -> tuple[Path, Path, Path]:
        return (
            root / "report/comparison.csv",
            root / "report/comparison.md",
            root / "report/comparison.provenance.json",
        )

    def test_interrupted_comparison_resumes_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_comparison(ComparisonFixture(root).materialize())
            outputs = self._output_paths(root)
            original = comparison_module._promote_comparison_staged_file
            promotions = 0

            def crash_on_second(staged: Path, destination: Path) -> None:
                nonlocal promotions
                promotions += 1
                if promotions == 2:
                    raise OSError("simulated comparison crash")
                original(staged, destination)

            with mock.patch.object(
                comparison_module,
                "_promote_comparison_staged_file",
                side_effect=crash_on_second,
            ), self.assertRaisesRegex(OSError, "simulated comparison crash"):
                write_comparison(
                    bundle,
                    csv_path=outputs[0],
                    markdown_path=outputs[1],
                    provenance_path=outputs[2],
                )
            report_dir = outputs[0].parent
            self.assertEqual(len(list(report_dir.glob(".*.bundle.transaction.json"))), 1)
            self.assertEqual(sum(path.is_file() for path in outputs), 1)
            write_comparison(
                bundle,
                csv_path=outputs[0],
                markdown_path=outputs[1],
                provenance_path=outputs[2],
                resume=True,
            )
            marker = report_dir / "comparison.bundle.commit.json"
            self.assertTrue(marker.is_file())
            before = tuple(path.read_bytes() for path in outputs)
            write_comparison(
                bundle,
                csv_path=outputs[0],
                markdown_path=outputs[1],
                provenance_path=outputs[2],
                resume=True,
            )
            self.assertEqual(before, tuple(path.read_bytes() for path in outputs))

    def test_comparison_orphan_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_comparison(ComparisonFixture(root).materialize())
            outputs = self._output_paths(root)
            with mock.patch.object(
                comparison_module,
                "_promote_comparison_staged_file",
                side_effect=OSError("simulated crash"),
            ), self.assertRaises(OSError):
                write_comparison(
                    bundle,
                    csv_path=outputs[0],
                    markdown_path=outputs[1],
                    provenance_path=outputs[2],
                )
            journal = next(outputs[0].parent.glob(".*.bundle.transaction.json"))
            journal.unlink()
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                write_comparison(
                    bundle,
                    csv_path=outputs[0],
                    markdown_path=outputs[1],
                    provenance_path=outputs[2],
                )
            write_comparison(
                bundle,
                csv_path=outputs[0],
                markdown_path=outputs[1],
                provenance_path=outputs[2],
                resume=True,
            )
            self.assertTrue(
                (outputs[0].parent / "comparison.bundle.commit.json").is_file()
            )

    def test_comparison_resume_rejects_stage_and_canonical_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_comparison(ComparisonFixture(root).materialize())
            outputs = self._output_paths(root)
            with mock.patch.object(
                comparison_module,
                "_promote_comparison_staged_file",
                side_effect=OSError("simulated crash"),
            ), self.assertRaises(OSError):
                write_comparison(
                    bundle,
                    csv_path=outputs[0],
                    markdown_path=outputs[1],
                    provenance_path=outputs[2],
                )
            stage = next(outputs[0].parent.glob(".*.bundle.stage.*"))
            next(stage.iterdir()).write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError, "staged output.*tampered"):
                write_comparison(
                    bundle,
                    csv_path=outputs[0],
                    markdown_path=outputs[1],
                    provenance_path=outputs[2],
                    resume=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_comparison(ComparisonFixture(root).materialize())
            outputs = self._output_paths(root)
            write_comparison(
                bundle,
                csv_path=outputs[0],
                markdown_path=outputs[1],
                provenance_path=outputs[2],
            )
            outputs[0].write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError, "tampered"):
                write_comparison(
                    bundle,
                    csv_path=outputs[0],
                    markdown_path=outputs[1],
                    provenance_path=outputs[2],
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
