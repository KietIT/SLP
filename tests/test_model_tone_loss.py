from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.vitonesr.model import WhisperToneMTL


class _FakeWhisper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=4)

    def forward(self, *, input_features=None, labels=None, **kwargs):
        del input_features, labels, kwargs
        hidden = torch.zeros((1, 2, 4), dtype=torch.float32, requires_grad=True)
        return SimpleNamespace(
            loss=hidden.sum() * 0.0 + 1.0,
            decoder_hidden_states=(hidden,),
            logits=torch.zeros((1, 2, 8), dtype=torch.float32),
        )


class WhisperToneMTLTests(unittest.TestCase):
    def test_rejects_shiftable_tone_label_length_mismatch(self) -> None:
        model = WhisperToneMTL(_FakeWhisper(), lambda_tone=0.1)
        with self.assertRaisesRegex(ValueError, "sequence mismatch"):
            model(
                input_features=torch.zeros((1, 80, 4)),
                labels=torch.tensor([[1, 2]]),
                tone_labels=torch.tensor([[0, 1, 2]]),
            )

    def test_all_ignored_tone_targets_keep_finite_loss(self) -> None:
        model = WhisperToneMTL(_FakeWhisper(), lambda_tone=0.1)
        result = model(
            input_features=torch.zeros((1, 80, 4)),
            labels=torch.tensor([[1, 2]]),
            tone_labels=torch.full((1, 2), -100),
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(float(result["tone_loss"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
