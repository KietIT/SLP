from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.vitonesr.analysis import (
    CANONICAL_PREDICTION_COLUMNS,
    METRIC_VERSION,
    PredictionValidationError,
    analyze_error_events,
    analyze_prediction_rows,
    compute_aligned_metric_result,
    compute_aligned_metrics,
    final_coda,
    load_prediction_csv,
    serialize_alignment_events,
    validate_prediction_rows,
)


def canonical_row(**overrides: str) -> dict[str, str]:
    row = {
        "utt_id": "utt-1",
        "dataset": "vivos",
        "model": "phowhisper",
        "model_size": "base",
        "train_type": "tone_aware_lora",
        "lambda": "0.05",
        "seed": "42",
        "snr": "5",
        "noise_type": "music",
        "ref": "má là",
        "hyp": "ma là",
    }
    row.update(overrides)
    return row


class CanonicalPredictionTests(unittest.TestCase):
    def test_loader_validates_exact_schema_and_preserves_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pred.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(CANONICAL_PREDICTION_COLUMNS)
                )
                writer.writeheader()
                writer.writerow(
                    canonical_row(
                        **{
                            "lambda": "0.0500",
                            "ref": "Giữ nguyên văn bản",
                            "hyp": " giữ nguyên khoảng trắng ",
                        }
                    )
                )

            rows = load_prediction_csv(path)

        self.assertEqual(rows[0]["lambda"], "0.05")
        self.assertEqual(rows[0]["ref"], "Giữ nguyên văn bản")
        self.assertEqual(rows[0]["hyp"], " giữ nguyên khoảng trắng ")

    def test_loader_rejects_wrong_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["utt_id", "ref", "hyp"])
                writer.writeheader()
                writer.writerow({"utt_id": "x", "ref": "a", "hyp": "a"})
            with self.assertRaisesRegex(PredictionValidationError, "expected columns"):
                load_prediction_csv(path)

    def test_loader_distinguishes_missing_cell_from_empty_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-cell.csv"
            path.write_text(
                ",".join(CANONICAL_PREDICTION_COLUMNS)
                + "\n"
                + "u1,test,whisper,tiny,zero_shot,,42,clean,clean,tham chiếu\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PredictionValidationError, "missing CSV cells"):
                load_prediction_csv(path)

            path.write_text(
                ",".join(CANONICAL_PREDICTION_COLUMNS)
                + "\n"
                + "u1,test,whisper,tiny,zero_shot,,42,clean,clean,tham chiếu,\n",
                encoding="utf-8",
            )
            self.assertEqual(load_prediction_csv(path)[0]["hyp"], "")

    def test_validator_rejects_duplicate_and_run_metadata_conflict(self) -> None:
        duplicate = canonical_row()
        with self.assertRaisesRegex(PredictionValidationError, "duplicate utt_id"):
            validate_prediction_rows([duplicate, duplicate])

        conflicting = canonical_row(utt_id="utt-2", seed="7")
        with self.assertRaisesRegex(PredictionValidationError, "run metadata"):
            validate_prediction_rows([canonical_row(), conflicting])

    def test_validator_enforces_zero_shot_lambda_and_clean_condition(self) -> None:
        bad_lambda = canonical_row(train_type="zero_shot", **{"lambda": "0"})
        with self.assertRaisesRegex(PredictionValidationError, "empty lambda"):
            validate_prediction_rows([bad_lambda])

        bad_clean = canonical_row(snr="clean", noise_type="music")
        with self.assertRaisesRegex(PredictionValidationError, "noise_type='clean'"):
            validate_prediction_rows([bad_clean])


class AlignmentEventTests(unittest.TestCase):
    def test_alignment_is_indexed_and_prefers_same_tone_stripped_base(self) -> None:
        events = analyze_error_events("má là", "xin ma là", utt_id="sample")
        summary = [
            (event.operation, event.ref_index, event.hyp_index)
            for event in events
        ]
        self.assertEqual(
            summary,
            [
                ("insertion", None, 0),
                ("substitution", 0, 1),
                ("match", 1, 2),
            ],
        )
        self.assertTrue(events[1].tone_eligible)
        self.assertTrue(events[1].tone_error)
        self.assertEqual(events[1].ref_tone, "sac")
        self.assertEqual(events[1].hyp_tone, "ngang")
        self.assertEqual(events, analyze_error_events("má là", "xin ma là", utt_id="sample"))

    def test_tone_and_letter_diacritic_errors_are_separate(self) -> None:
        tone = compute_aligned_metric_result(["má"], ["mà"])
        self.assertEqual((tone.tone_errors, tone.tone_reference_units), (1, 1))
        self.assertEqual((tone.diacritic_errors, tone.diacritic_reference_units), (0, 1))
        self.assertEqual(tone.ter, 1.0)
        self.assertEqual(tone.der, 0.0)

        quality = compute_aligned_metric_result(["mà"], ["mằ"])
        self.assertEqual((quality.tone_errors, quality.tone_reference_units), (0, 0))
        self.assertEqual(
            (quality.diacritic_errors, quality.diacritic_reference_units),
            (1, 1),
        )
        self.assertEqual(quality.ter, 0.0)
        self.assertEqual(quality.der, 1.0)

        letter_d = analyze_error_events("đã", "dã")[0]
        self.assertTrue(letter_d.diacritic_error)
        self.assertFalse(letter_d.tone_eligible)

    def test_deletions_drive_ter_fcer_and_short_word_deletion_rate(self) -> None:
        result = compute_aligned_metric_result(["đã có là một và"], [""])
        self.assertEqual(result.short_word_deletions, 5)
        self.assertEqual(result.short_word_reference_units, 5)
        self.assertEqual(result.swdr, 1.0)
        self.assertEqual(result.ter, 1.0)
        self.assertEqual(
            (result.final_consonant_errors, result.final_consonant_reference_units),
            (1, 1),
        )

    def test_final_coda_uses_longest_labels_and_counts_changes(self) -> None:
        self.assertEqual(final_coda("bạch"), "ch")
        self.assertEqual(final_coda("bang"), "ng")
        self.assertEqual(final_coda("bánh"), "nh")
        self.assertEqual(final_coda("ban"), "n")

        result = compute_aligned_metric_result(["ban bánh"], ["bang bán"])
        self.assertEqual(
            (result.final_consonant_errors, result.final_consonant_reference_units),
            (2, 2),
        )
        self.assertEqual(result.fcer, 1.0)

    def test_event_serialization_carries_run_metadata(self) -> None:
        rows = [canonical_row()]
        events = analyze_prediction_rows(rows, source="unit-test")
        serialized = serialize_alignment_events(events)
        first = serialized[0]
        self.assertEqual(first["metric_version"], METRIC_VERSION)
        self.assertEqual(first["utt_id"], "utt-1")
        self.assertEqual(first["lambda"], "0.05")
        self.assertEqual(first["snr"], "5")
        self.assertIn(first["operation"], {"match", "substitution"})


class CorpusMetricTests(unittest.TestCase):
    def test_corpus_alignment_never_crosses_utterance_boundaries(self) -> None:
        result = compute_aligned_metric_result(["a", "b"], ["", "a b"])
        self.assertEqual((result.word_errors, result.word_reference_units), (2, 2))
        self.assertEqual(result.wer, 1.0)
        self.assertEqual(
            (result.character_errors, result.character_reference_units),
            (3, 2),
        )
        self.assertEqual(result.cer, 1.5)

    def test_scalar_api_has_stable_versioned_keys(self) -> None:
        metrics = compute_aligned_metrics(["xin chào"], ["xin chào"])
        self.assertEqual(
            set(metrics),
            {"metric_version", "wer", "cer", "ter", "der", "fcer", "swdr"},
        )
        self.assertEqual(metrics["metric_version"], "aligned_v1")
        for name in ("wer", "cer", "ter", "der", "fcer", "swdr"):
            self.assertEqual(metrics[name], 0.0)

        counts = compute_aligned_metric_result(["xin chào"], ["xin chào"]).to_dict(
            include_counts=True
        )
        self.assertEqual(counts["wer_numerator"], 0)
        self.assertEqual(counts["wer_denominator"], 2)

    def test_length_mismatch_and_empty_corpus_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            compute_aligned_metrics(["a"], [])
        with self.assertRaisesRegex(ValueError, "empty corpus"):
            compute_aligned_metrics([], [])


if __name__ == "__main__":
    unittest.main()
