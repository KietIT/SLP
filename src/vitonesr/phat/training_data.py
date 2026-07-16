from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import Dataset

from src.vitonesr.data import align_tone_labels_to_token_ids, read_jsonl
from src.vitonesr.noise import choose_and_mix, read_audio
from src.vitonesr.text_norm import normalize_vi_text
from src.vitonesr.tone import build_token_tone_alignment


def _stable_item_seed(seed: int, epoch: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}|{epoch}|{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def validate_jsonl_audio_manifest(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    rows = read_jsonl(str(manifest_path))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    selected = rows if max_rows is None else rows[:max_rows]
    for index, row in enumerate(selected):
        if not row.get("audio") or not row.get("text"):
            raise ValueError(f"Manifest row {index} must contain audio and text: {manifest_path}")
        if not Path(str(row["audio"])).exists():
            raise FileNotFoundError(f"Audio file does not exist at row {index}: {row['audio']}")
    return rows


def validate_manifest_declared_split(
    path: str | Path,
    *,
    expected_split: str,
) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    rows = read_jsonl(str(manifest_path))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    expected = str(expected_split).strip().casefold()
    observed = {
        str(row.get("split", "")).strip().casefold()
        for row in rows
    }
    if observed != {expected}:
        raise ValueError(
            f"Manifest {manifest_path} must contain only split={expected!r}; "
            f"observed={sorted(observed)}"
        )
    return rows


class DeterministicASRTrainingDataset(Dataset):
    """Manifest dataset with augmentation fixed by seed, epoch, and item index."""

    def __init__(
        self,
        manifest_path: str,
        processor: Any,
        *,
        seed: int,
        sample_rate: int = 16000,
        max_audio_seconds: float = 15.0,
        max_label_length: int | None = None,
        noise_manifest: str | None = None,
        noise_prob: float = 0.0,
        snr_choices: Sequence[float] = (20, 10, 5, 0),
        tone_label_policy: str = "last_subtoken",
        max_samples: int | None = None,
        expected_split: str | None = None,
    ) -> None:
        rows = validate_jsonl_audio_manifest(manifest_path)
        if expected_split is not None:
            expected = str(expected_split).strip().casefold()
            observed = {
                str(row.get("split", "")).strip().casefold()
                for row in rows
            }
            if observed != {expected}:
                raise ValueError(
                    f"Manifest {manifest_path} must contain only split={expected!r}; "
                    f"observed={sorted(observed)}"
                )
        self.items = rows if max_samples is None else rows[:max_samples]
        if max_samples is not None and len(self.items) < max_samples:
            raise ValueError(f"Requested {max_samples} samples but manifest only has {len(rows)} rows")
        self.processor = processor
        self.seed = int(seed)
        self.epoch = 0
        self.sample_rate = int(sample_rate)
        self.max_length = int(max_audio_seconds * sample_rate)
        self.max_label_length = max_label_length
        self.noise_prob = float(noise_prob)
        self.snr_choices = tuple(float(value) for value in snr_choices)
        self.tone_label_policy = tone_label_policy
        self.noise_paths: list[str] = []
        if noise_manifest:
            noise_rows = validate_jsonl_audio_manifest_fields(noise_manifest, required=("audio",))
            self.noise_paths = [str(row["audio"]) for row in noise_rows]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        waveform = read_audio(str(item["audio"]), sr=self.sample_rate)
        if len(waveform) > self.max_length:
            waveform = waveform[: self.max_length]
        item_seed = _stable_item_seed(self.seed, self.epoch, index)
        rng = random.Random(item_seed)
        if self.noise_paths and rng.random() < self.noise_prob:
            waveform, _ = choose_and_mix(
                waveform,
                self.noise_paths,
                snr_choices=self.snr_choices,
                sr=self.sample_rate,
                seed=item_seed,
            )
        source_text = str(item["text"])
        text = normalize_vi_text(source_text)
        features = self.processor.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
        ).input_features[0]
        tokenizer = self.processor.tokenizer
        labels = tokenizer(text, add_special_tokens=True).input_ids
        text_token_ids = tokenizer(text, add_special_tokens=False).input_ids
        tone_alignment = build_token_tone_alignment(
            text,
            tokenizer,
            policy=self.tone_label_policy,
            source_text=source_text,
        )
        tone_labels = align_tone_labels_to_token_ids(
            labels,
            text_token_ids,
            tone_alignment.token_ids,
            tone_alignment.tone_labels,
            getattr(tokenizer, "all_special_ids", []),
        )
        if self.max_label_length is not None:
            labels = labels[: self.max_label_length]
            tone_labels = tone_labels[: self.max_label_length]
        return {
            "input_features": features,
            "labels": labels,
            "tone_labels": tone_labels,
            "text": text,
        }


def validate_jsonl_audio_manifest_fields(
    path: str | Path,
    *,
    required: Sequence[str],
) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    rows = read_jsonl(str(manifest_path))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    for index, row in enumerate(rows):
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ValueError(f"Manifest row {index} is missing {missing}: {manifest_path}")
        if "audio" in required and not Path(str(row["audio"])).exists():
            raise FileNotFoundError(f"Audio file does not exist at row {index}: {row['audio']}")
    return rows
