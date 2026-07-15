from __future__ import annotations

import unittest

import torch

from src.vitonesr.phat.losses import combine_asr_tone_losses, safe_tone_cross_entropy
from src.vitonesr.tone import IGNORE_INDEX, TONE_TO_ID, build_token_tone_labels, extract_tone


class _TokenResult:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str, *, add_special_tokens: bool) -> _TokenResult:
        del add_special_tokens
        self.calls.append(text)
        mapping = {
            "nghiêng": [11, 12],
            " má": [13],
            "nghiêng má": [11, 12, 13],
        }
        if text not in mapping:
            raise AssertionError(f"Unexpected fake-tokenizer input: {text!r}")
        return _TokenResult(mapping[text])


class ToneAndLossTests(unittest.TestCase):
    def test_six_vietnamese_tone_classes(self) -> None:
        examples = {
            "ma": "ngang",
            "má": "sac",
            "mà": "huyen",
            "mả": "hoi",
            "mã": "nga",
            "mạ": "nang",
        }
        for word, tone_name in examples.items():
            with self.subTest(word=word):
                tone_id, is_valid = extract_tone(word)
                self.assertTrue(is_valid)
                self.assertEqual(tone_id, TONE_TO_ID[tone_name])

    def test_non_tone_tokens_are_ignored(self) -> None:
        for token in ("123", "GPU", "...", "brrr"):
            with self.subTest(token=token):
                tone_id, is_valid = extract_tone(token)
                self.assertFalse(is_valid)
                self.assertEqual(tone_id, IGNORE_INDEX)

    def test_token_tone_labels_respect_last_subtoken_policy(self) -> None:
        tokenizer = _FakeTokenizer()
        labels = build_token_tone_labels("nghiêng má", tokenizer, policy="last_subtoken")
        self.assertEqual(labels, [IGNORE_INDEX, TONE_TO_ID["ngang"], TONE_TO_ID["sac"]])
        self.assertEqual(tokenizer.calls[:2], ["nghiêng", " má"])

    def test_all_ignored_tone_loss_is_finite_zero(self) -> None:
        logits = torch.randn(2, 4, 6, requires_grad=True)
        labels = torch.full((2, 4), IGNORE_INDEX, dtype=torch.long)
        loss = safe_tone_cross_entropy(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_ignore_index_excludes_padding_position(self) -> None:
        first_logits = torch.tensor([[[4.0, 0, 0, 0, 0, 0], [100.0, 0, 0, 0, 0, 0]]])
        second_logits = torch.tensor([[[4.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 100.0]]])
        labels = torch.tensor([[0, IGNORE_INDEX]])
        first_loss = safe_tone_cross_entropy(first_logits, labels)
        second_loss = safe_tone_cross_entropy(second_logits, labels)
        self.assertTrue(torch.equal(first_loss, second_loss))

    def test_tone_loss_depends_on_labels_and_logits(self) -> None:
        logits = torch.tensor([[[5.0, 0, 0, 0, 0, 0], [0, 5.0, 0, 0, 0, 0]]])
        correct = safe_tone_cross_entropy(logits, torch.tensor([[0, 1]]))
        incorrect = safe_tone_cross_entropy(logits, torch.tensor([[1, 0]]))
        self.assertLess(float(correct), float(incorrect))

    def test_lambda_zero_total_loss_equals_asr_loss(self) -> None:
        asr_loss = torch.tensor(2.5, requires_grad=True)
        tone_loss = torch.tensor(9.0, requires_grad=True)
        total_loss = combine_asr_tone_losses(asr_loss, tone_loss, 0.0)
        self.assertIs(total_loss, asr_loss)


if __name__ == "__main__":
    unittest.main()
