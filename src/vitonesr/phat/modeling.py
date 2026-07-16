from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration

from .losses import combine_asr_tone_losses, safe_tone_cross_entropy


@dataclass(frozen=True)
class ParameterStats:
    total: int
    trainable: int

    @property
    def trainable_ratio(self) -> float:
        return self.trainable / self.total if self.total else 0.0


class PhoWhisperLoRA(nn.Module):
    """PEFT PhoWhisper with an optional decoder-side six-tone classifier."""

    def __init__(
        self,
        asr_model: nn.Module,
        *,
        lambda_tone: float,
        num_tones: int = 6,
    ) -> None:
        super().__init__()
        self.asr_model = asr_model
        self.lambda_tone = float(lambda_tone)
        self.tone_head: nn.Module | None = None
        if self.lambda_tone > 0.0:
            d_model = int(asr_model.config.d_model)
            self.tone_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_tones))

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor,
        tone_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        use_tone = self.tone_head is not None and tone_labels is not None
        outputs = self.asr_model(
            input_features=input_features,
            labels=labels,
            output_hidden_states=use_tone,
            return_dict=True,
            **kwargs,
        )
        asr_loss = outputs.loss
        tone_logits: torch.Tensor | None = None
        tone_loss: torch.Tensor | None = None
        if use_tone:
            if outputs.decoder_hidden_states is None:
                raise RuntimeError("Decoder hidden states are required for tone-aware training")
            tone_logits = self.tone_head(outputs.decoder_hidden_states[-1])
            tone_loss = safe_tone_cross_entropy(tone_logits, tone_labels)
        total_loss = combine_asr_tone_losses(asr_loss, tone_loss, self.lambda_tone)
        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "asr_loss": asr_loss,
            "tone_loss": tone_loss,
            "logits": outputs.logits,
            "tone_logits": tone_logits,
        }

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.asr_model.generate(*args, **kwargs)


def _new_base_model(
    model_name_or_path: str,
    *,
    revision: str,
) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        revision=revision,
    )
    model.config.use_cache = False
    return model


def build_trainable_model(
    *,
    model_name_or_path: str,
    revision: str,
    lambda_tone: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    adapter_checkpoint: str | Path | None = None,
) -> PhoWhisperLoRA:
    base = _new_base_model(model_name_or_path, revision=revision)
    if adapter_checkpoint is None:
        lora_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(target_modules),
            bias="none",
        )
        asr_model = get_peft_model(base, lora_config)
    else:
        adapter_path = Path(adapter_checkpoint)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter checkpoint does not exist: {adapter_path}")
        asr_model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    return PhoWhisperLoRA(asr_model, lambda_tone=lambda_tone)


def parameter_stats(model: nn.Module) -> ParameterStats:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return ParameterStats(total=total, trainable=trainable)


def assert_only_lora_and_tone_trainable(model: PhoWhisperLoRA) -> None:
    unexpected = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" not in name and not name.startswith("tone_head."):
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"Unexpected trainable backbone parameters: {unexpected[:10]}")
