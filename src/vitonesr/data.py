from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .noise import choose_and_mix, read_audio
from .text_norm import normalize_vi_text
from .tone import build_token_tone_labels, IGNORE_INDEX


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> Optional[int]:
    if not needle or len(needle) > len(values):
        return None
    n = len(needle)
    for i in range(len(values) - n + 1):
        if list(values[i:i + n]) == list(needle):
            return i
    return None


def align_tone_labels_to_token_ids(
    full_token_ids: Sequence[int],
    text_token_ids: Sequence[int],
    tone_piece_labels: Sequence[int],
    special_token_ids: Sequence[int],
) -> List[int]:
    """Align syllable-derived tone labels to tokenizer output with special tokens ignored.

    Whisper labels may include decoder/language/task/eos special tokens. Tone labels are
    produced from text-only BPE pieces, so we fill only text-token positions and keep all
    special tokens at IGNORE_INDEX.
    """
    aligned = [IGNORE_INDEX] * len(full_token_ids)
    start = _find_subsequence(full_token_ids, text_token_ids)
    if start is not None:
        positions = list(range(start, min(start + len(text_token_ids), len(full_token_ids))))
    else:
        specials = set(special_token_ids)
        positions = [i for i, token_id in enumerate(full_token_ids) if token_id not in specials]
    for pos, tone_id in zip(positions, tone_piece_labels):
        aligned[pos] = tone_id
    return aligned


class ASRManifestDataset(Dataset):
    def __init__(self, manifest_path: str, processor, sample_rate: int = 16000, max_audio_seconds: float = 15.0, max_label_length: Optional[int] = None, noise_manifest: Optional[str] = None, noise_prob: float = 0.0, snr_choices=(20, 10, 5, 0), tone_label_policy: str = "last_subtoken"):
        self.items = read_jsonl(manifest_path)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_len = int(max_audio_seconds * sample_rate)
        self.max_label_length = max_label_length
        self.noise_prob = noise_prob
        self.snr_choices = snr_choices
        self.tone_label_policy = tone_label_policy
        self.noise_paths = []
        if noise_manifest:
            self.noise_paths = [x["audio"] for x in read_jsonl(noise_manifest)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        wav = read_audio(item["audio"], sr=self.sample_rate)
        if len(wav) > self.max_len:
            wav = wav[:self.max_len]
        if self.noise_paths and random.random() < self.noise_prob:
            wav, noise_meta = choose_and_mix(wav, self.noise_paths, snr_choices=self.snr_choices, sr=self.sample_rate, seed=random.randint(0, 10**9))
        text = normalize_vi_text(item["text"])
        features = self.processor.feature_extractor(wav, sampling_rate=self.sample_rate).input_features[0]
        tokenizer = self.processor.tokenizer
        labels = tokenizer(text, add_special_tokens=True).input_ids
        text_token_ids = tokenizer(text, add_special_tokens=False).input_ids
        tone_piece_labels = build_token_tone_labels(text, tokenizer, policy=self.tone_label_policy)
        tone_labels = align_tone_labels_to_token_ids(
            labels,
            text_token_ids,
            tone_piece_labels,
            getattr(tokenizer, "all_special_ids", []),
        )
        if self.max_label_length is not None:
            labels = labels[:self.max_label_length]
            tone_labels = tone_labels[:self.max_label_length]
        return {"input_features": features, "labels": labels, "tone_labels": tone_labels, "text": text}


class DataCollatorSpeechSeq2SeqWithTone:
    def __init__(self, processor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        remove_decoder_start = (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        if remove_decoder_start:
            labels = labels[:, 1:]
        tone_max = labels.size(1)
        tone = torch.full_like(labels, fill_value=-100)
        for i, f in enumerate(features):
            ids = torch.tensor(f["tone_labels"], dtype=torch.long)
            if remove_decoder_start and ids.numel() > 0:
                ids = ids[1:]
            n = min(tone_max, ids.numel())
            if n > 0:
                tone[i, :n] = ids[:n]
        batch["labels"] = labels
        batch["tone_labels"] = tone
        return batch
