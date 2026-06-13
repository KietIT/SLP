from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import soundfile as sf
import librosa


def read_audio(path: str, sr: int = 16000) -> np.ndarray:
    wav, in_sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if in_sr != sr:
        wav = librosa.resample(wav, orig_sr=in_sr, target_sr=sr)
    return wav


def _fit_noise(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(noise) < length:
        reps = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, reps)
    start = 0 if len(noise) == length else rng.randint(0, len(noise) - length)
    return noise[start:start + length].astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, eps: float = 1e-8) -> np.ndarray:
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)
    clean_power = np.mean(clean ** 2) + eps
    noise_power = np.mean(noise ** 2) + eps
    scale = np.sqrt(clean_power / (10 ** (snr_db / 10.0) * noise_power))
    mixed = clean + scale * noise
    peak = max(float(np.max(np.abs(mixed))), 1.0)
    return (mixed / peak).astype(np.float32)


def choose_and_mix(clean: np.ndarray, noise_paths: Sequence[str], snr_choices=(20, 10, 5, 0), sr: int = 16000, seed: int | None = None) -> Tuple[np.ndarray, dict]:
    rng = random.Random(seed)
    npath = rng.choice(list(noise_paths))
    snr = rng.choice(list(snr_choices))
    noise = read_audio(npath, sr=sr)
    noise = _fit_noise(noise, len(clean), rng)
    return mix_at_snr(clean, noise, snr), {"noise_path": npath, "snr": snr}
