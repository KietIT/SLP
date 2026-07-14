from __future__ import annotations

import torch
import torch.nn.functional as F

from src.vitonesr.tone import IGNORE_INDEX


def safe_tone_cross_entropy(
    tone_logits: torch.Tensor,
    tone_labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute tone CE only on valid positions and return differentiable zero otherwise."""
    if tone_logits.ndim != 3:
        raise ValueError("tone_logits must have shape [batch, sequence, classes]")
    if tone_labels.ndim != 2:
        raise ValueError("tone_labels must have shape [batch, sequence]")
    sequence_length = min(tone_logits.size(1), tone_labels.size(1))
    logits = tone_logits[:, :sequence_length, :]
    labels = tone_labels[:, :sequence_length]
    valid_mask = labels.ne(ignore_index)
    if not torch.any(valid_mask):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid_mask], labels[valid_mask])


def combine_asr_tone_losses(
    asr_loss: torch.Tensor,
    tone_loss: torch.Tensor | None,
    lambda_tone: float,
) -> torch.Tensor:
    if lambda_tone < 0.0:
        raise ValueError("lambda_tone must be non-negative")
    if lambda_tone == 0.0 or tone_loss is None:
        return asr_loss
    return asr_loss + float(lambda_tone) * tone_loss
