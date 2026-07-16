from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .phat.losses import safe_tone_cross_entropy


class WhisperToneMTL(nn.Module):
    """Whisper/PhoWhisper with an auxiliary decoder-side tone head.

    Forward returns the base ASR loss plus lambda_tone * tone CE loss.
    tone_labels must be aligned to decoder token positions and use -100 for ignored tokens.
    """
    def __init__(self, base_model: WhisperForConditionalGeneration, num_tones: int = 6, lambda_tone: float = 0.1):
        super().__init__()
        self.base_model = base_model
        d_model = base_model.config.d_model
        self.tone_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_tones),
        )
        self.lambda_tone = lambda_tone

    def forward(self, input_features=None, labels=None, tone_labels=None, **kwargs):
        outputs = self.base_model(
            input_features=input_features,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        loss = outputs.loss
        tone_loss = None
        tone_logits = None
        if tone_labels is not None:
            hidden = outputs.decoder_hidden_states[-1]
            tone_logits = self.tone_head(hidden)
            # A length mismatch is a label-alignment validity error.  Silently
            # truncating either side can train the tone head against shifted BPE
            # positions while still producing a finite loss.
            tone_loss = safe_tone_cross_entropy(tone_logits, tone_labels)
            loss = loss + self.lambda_tone * tone_loss
        return {
            "loss": loss,
            "asr_loss": outputs.loss,
            "tone_loss": tone_loss,
            "logits": outputs.logits,
            "tone_logits": tone_logits,
        }

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


def build_model(model_name_or_path: str, use_lora: bool = True, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, target_modules=("q_proj", "v_proj"), lambda_tone: float = 0.1):
    base = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    base.config.use_cache = False
    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        base = get_peft_model(base, lora_cfg)
        base.print_trainable_parameters()
    model = WhisperToneMTL(base, num_tones=6, lambda_tone=lambda_tone)
    return model
