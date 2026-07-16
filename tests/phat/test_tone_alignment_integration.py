from __future__ import annotations

import os
import unicodedata
import unittest
from types import SimpleNamespace
from typing import Any

import torch

from scripts.audit_tone_alignment import (
    DEFAULT_TOKENIZER_REVISION,
    TranscriptRecord,
    audit_records,
    build_parser,
)
from src.vitonesr.data import (
    DataCollatorSpeechSeq2SeqWithTone,
    align_tone_labels_to_token_ids,
)
from src.vitonesr.tone import (
    IGNORE_INDEX,
    TONE_TO_ID,
    ToneAlignmentError,
    build_token_tone_alignment,
)


class _TokenResult:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class _BoundaryTokenizer:
    """Small compositional tokenizer whose first-word boundary changes its IDs."""

    all_special_ids = [90, 91]

    def __init__(self) -> None:
        self.mapping = {
            "nghiêng": [11, 12],
            " nghiêng": [19],
            " má": [13],
            "nghiêng má": [11, 12, 32],
            "xin": [21],
            " gpu": [22, 23],
            "xin gpu": [21, 22, 23],
            " machine": [24, 25],
            "xin machine": [21, 24, 25],
            "tối": [41],
            "tối gpu": [41, 22, 23],
            "ma": [31],
            " má": [32],
            " mà": [33],
            " mả": [34],
            " mã": [35],
            " mạ": [36],
            "ma má mà mả mã mạ": [31, 32, 33, 34, 35, 36],
        }

    def __call__(self, text: str, *, add_special_tokens: bool) -> _TokenResult:
        if text not in self.mapping:
            raise AssertionError(f"Unexpected boundary-tokenizer input: {text!r}")
        ids = list(self.mapping[text])
        if add_special_tokens:
            ids = [90, *ids, 91]
        return _TokenResult(ids)


class _NonCompositionalTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> _TokenResult:
        del add_special_tokens
        mapping = {
            "xin": [1],
            " chào": [2],
            "xin chào": [1, 3],
        }
        return _TokenResult(mapping[text])


class ToneAlignmentContractTests(unittest.TestCase):
    def test_first_word_has_no_prefix_and_multi_piece_target_is_exact(self) -> None:
        alignment = build_token_tone_alignment(
            "nghiêng má",
            _BoundaryTokenizer(),
            policy="last_subtoken",
        )
        self.assertEqual(list(alignment.token_ids), [11, 12, 32])
        self.assertEqual(
            list(alignment.tone_labels),
            [IGNORE_INDEX, TONE_TO_ID["ngang"], TONE_TO_ID["sac"]],
        )
        self.assertEqual(alignment.words[0].piece_text, "nghiêng")
        self.assertEqual(alignment.words[1].piece_text, " má")

    def test_non_compositional_tokenizer_fails_instead_of_zipping(self) -> None:
        with self.assertRaisesRegex(ToneAlignmentError, "do not match"):
            build_token_tone_alignment("xin chào", _NonCompositionalTokenizer())

    def test_alignment_rejects_piece_id_and_label_mismatches(self) -> None:
        with self.assertRaisesRegex(ToneAlignmentError, "differ"):
            align_tone_labels_to_token_ids(
                [90, 11, 12, 91],
                [11, 12],
                [11, 99],
                [IGNORE_INDEX, TONE_TO_ID["ngang"]],
                [90, 91],
            )
        with self.assertRaisesRegex(ToneAlignmentError, "Non-special"):
            align_tone_labels_to_token_ids(
                [90, 77, 11, 12, 91],
                [11, 12],
                [11, 12],
                [IGNORE_INDEX, TONE_TO_ID["ngang"]],
                [90, 91],
            )
        with self.assertRaisesRegex(ToneAlignmentError, "length mismatch"):
            align_tone_labels_to_token_ids(
                [90, 11, 12, 91],
                [11, 12],
                [11, 12],
                [TONE_TO_ID["ngang"]],
                [90, 91],
            )

    def test_special_tokens_are_ignored_after_exact_mapping(self) -> None:
        alignment = build_token_tone_alignment("nghiêng má", _BoundaryTokenizer())
        mapped = align_tone_labels_to_token_ids(
            [90, *alignment.token_ids, 91],
            alignment.token_ids,
            alignment.token_ids,
            alignment.tone_labels,
            [90, 91],
        )
        self.assertEqual(mapped[0], IGNORE_INDEX)
        self.assertEqual(mapped[-1], IGNORE_INDEX)
        self.assertEqual(mapped[1:-1], list(alignment.tone_labels))

    def test_nfc_nfd_and_truncation_preserve_identical_prefixes(self) -> None:
        tokenizer = _BoundaryTokenizer()
        nfc = "nghiêng má"
        nfd = unicodedata.normalize("NFD", nfc)
        full = build_token_tone_alignment(nfc, tokenizer)
        decomposed = build_token_tone_alignment(nfd, tokenizer)
        truncated = build_token_tone_alignment(nfc, tokenizer, max_length=2)
        self.assertEqual(full.token_ids, decomposed.token_ids)
        self.assertEqual(full.tone_labels, decomposed.tone_labels)
        self.assertEqual(truncated.token_ids, full.token_ids[:2])
        self.assertEqual(truncated.tone_labels, full.tone_labels[:2])
        self.assertEqual(truncated.words[-1].token_end, len(truncated.token_ids))
        self.assertEqual(
            truncated.words[-1].token_ids,
            truncated.token_ids[
                truncated.words[-1].token_start : truncated.words[-1].token_end
            ],
        )

    def test_acronym_is_masked_and_foreign_word_is_review_candidate(self) -> None:
        tokenizer = _BoundaryTokenizer()
        acronym = build_token_tone_alignment(
            "xin gpu",
            tokenizer,
            source_text="xin GPU",
        )
        self.assertEqual(acronym.words[1].status, "masked_acronym")
        self.assertTrue(
            all(
                label == IGNORE_INDEX
                for label in acronym.tone_labels[
                    acronym.words[1].token_start : acronym.words[1].token_end
                ]
            )
        )
        foreign = build_token_tone_alignment("xin machine", tokenizer)
        self.assertEqual(foreign.words[1].status, "unmarked_or_foreign_candidate")

    def test_all_caps_context_is_explicit_without_masking_vietnamese(self) -> None:
        alignment = build_token_tone_alignment(
            "tối gpu",
            _BoundaryTokenizer(),
            source_text="TỐI GPU",
        )
        self.assertTrue(alignment.words[0].is_valid)
        self.assertEqual(alignment.words[0].status, "marked_tone")
        self.assertTrue(alignment.words[1].is_valid)
        self.assertEqual(
            alignment.words[1].status,
            "all_caps_unmarked_or_acronym_candidate",
        )

    def test_mixed_case_ascii_acronym_is_masked(self) -> None:
        uppercase_vietnamese = build_token_tone_alignment(
            "tối gpu",
            _BoundaryTokenizer(),
            source_text="TỐI gpu",
        )
        self.assertTrue(uppercase_vietnamese.words[0].is_valid)
        self.assertEqual(uppercase_vietnamese.words[0].status, "marked_tone")
        mixed = build_token_tone_alignment(
            "tối gpu",
            _BoundaryTokenizer(),
            source_text="tối GPU",
        )
        self.assertFalse(mixed.words[1].is_valid)
        self.assertEqual(mixed.words[1].status, "masked_acronym")

    def test_all_six_tones_land_on_last_subtokens(self) -> None:
        alignment = build_token_tone_alignment(
            "ma má mà mả mã mạ",
            _BoundaryTokenizer(),
        )
        self.assertEqual(
            list(alignment.tone_labels),
            [
                TONE_TO_ID["ngang"],
                TONE_TO_ID["sac"],
                TONE_TO_ID["huyen"],
                TONE_TO_ID["hoi"],
                TONE_TO_ID["nga"],
                TONE_TO_ID["nang"],
            ],
        )

    def test_all_subtokens_policy_targets_each_piece(self) -> None:
        alignment = build_token_tone_alignment(
            "nghiêng má",
            _BoundaryTokenizer(),
            policy="all_subtokens",
        )
        self.assertEqual(
            list(alignment.tone_labels),
            [TONE_TO_ID["ngang"], TONE_TO_ID["ngang"], TONE_TO_ID["sac"]],
        )

    def test_collator_rejects_mismatched_label_lengths(self) -> None:
        class Batch(dict):
            def __getattr__(self, name: str) -> Any:
                return self[name]

        class FeatureExtractor:
            def pad(self, features: list[dict[str, object]], return_tensors: str) -> dict:
                del features, return_tensors
                return {"input_features": torch.zeros((1, 2))}

        class Tokenizer:
            def pad(self, features: list[dict[str, list[int]]], return_tensors: str) -> Batch:
                del return_tensors
                values = torch.tensor([features[0]["input_ids"]])
                return Batch(input_ids=values, attention_mask=torch.ones_like(values))

        processor = SimpleNamespace(
            feature_extractor=FeatureExtractor(),
            tokenizer=Tokenizer(),
        )
        collator = DataCollatorSpeechSeq2SeqWithTone(processor, decoder_start_token_id=90)
        with self.assertRaisesRegex(ToneAlignmentError, "mismatched"):
            collator(
                [
                    {
                        "input_features": [0.0, 0.0],
                        "labels": [90, 11, 91],
                        "tone_labels": [IGNORE_INDEX, TONE_TO_ID["ngang"]],
                    }
                ]
            )

    def test_audit_deduplicates_and_reports_strict_failures(self) -> None:
        records = [
            TranscriptRecord("fixture.csv", "0" * 64, 2, "utt-a", "nghiêng má"),
            TranscriptRecord("fixture.csv", "0" * 64, 3, "utt-b", "NGHIÊNG MÁ"),
        ]
        passed = audit_records(records, _BoundaryTokenizer())
        self.assertTrue(passed.passed)
        self.assertEqual(passed.selected_transcripts, 1)
        self.assertEqual(passed.duplicate_transcripts_skipped, 1)
        self.assertEqual(passed.targeted_token_count, 2)

        failed = audit_records(
            [TranscriptRecord("fixture.csv", "0" * 64, 2, "utt-c", "xin chào")],
            _NonCompositionalTokenizer(),
        )
        self.assertFalse(failed.passed)
        self.assertEqual(failed.mismatched_transcripts, 1)
        self.assertEqual(failed.rows[0]["status"], "alignment_error")

    def test_invalid_policy_and_empty_normalized_text_fail(self) -> None:
        with self.assertRaises(ValueError):
            build_token_tone_alignment("xin", _BoundaryTokenizer(), policy="invalid")
        with self.assertRaises(ToneAlignmentError):
            build_token_tone_alignment("...", _BoundaryTokenizer())


class RealPhoWhisperTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from transformers import AutoTokenizer

            cls.tokenizer: Any | None = AutoTokenizer.from_pretrained(
                "vinai/PhoWhisper-base",
                revision=DEFAULT_TOKENIZER_REVISION,
                local_files_only=True,
            )
            cls.load_error = ""
        except Exception as exc:  # pragma: no cover - depends on local model cache
            cls.tokenizer = None
            cls.load_error = str(exc)

    def setUp(self) -> None:
        if self.tokenizer is None:
            if os.environ.get("REQUIRE_PHOWHISPER_INTEGRATION") == "1":
                self.fail(f"PhoWhisper tokenizer is required but unavailable: {self.load_error}")
            self.skipTest(f"PhoWhisper tokenizer is not cached: {self.load_error}")

    def test_real_tokenizer_word_reconstruction_and_tone_positions(self) -> None:
        fixtures = [
            "nghiêng má",
            "tớ đi kiếm nhà trọ",
            "sự giúp đỡ",
            "những nhược điểm",
            "ma má mà mả mã mạ",
        ]
        for text in fixtures:
            with self.subTest(text=text):
                alignment = build_token_tone_alignment(text, self.tokenizer)
                expected = self.tokenizer(text, add_special_tokens=False).input_ids
                self.assertEqual(list(alignment.token_ids), list(expected))
                for word in alignment.words:
                    if word.is_valid:
                        self.assertEqual(
                            alignment.tone_labels[word.token_end - 1],
                            word.tone_id,
                        )

    def test_real_tokenizer_exercises_leading_space_regression(self) -> None:
        unspaced = self.tokenizer("nghiêng", add_special_tokens=False).input_ids
        spaced = self.tokenizer(" nghiêng", add_special_tokens=False).input_ids
        self.assertNotEqual(list(unspaced), list(spaced))

    def test_real_tokenizer_nfc_nfd_equivalence(self) -> None:
        nfc = "tớ đi kiếm nhà trọ"
        nfd = unicodedata.normalize("NFD", nfc)
        first = build_token_tone_alignment(nfc, self.tokenizer)
        second = build_token_tone_alignment(nfd, self.tokenizer)
        self.assertEqual(first.token_ids, second.token_ids)
        self.assertEqual(first.tone_labels, second.tone_labels)


class ToneAuditCliTests(unittest.TestCase):
    def test_default_tokenizer_revision_is_immutable(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.tokenizer_revision, DEFAULT_TOKENIZER_REVISION)
        self.assertEqual(len(args.tokenizer_revision), 40)


if __name__ == "__main__":
    unittest.main()
