from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from .noise import fit_noise, read_audio
from .prediction import write_jsonl


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MUSAN_TYPES = ("music", "noise", "speech")


def _is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS


def _noise_subtype(type_root: Path, path: Path) -> str:
    rel = path.relative_to(type_root)
    if rel.parent == Path("."):
        return ""
    return rel.parts[0]


def _selected_type_roots(
    musan_root: Path,
    include_music: bool,
    include_noise: bool,
    include_speech: bool,
) -> list[tuple[str, Path]]:
    include = {
        "music": include_music,
        "noise": include_noise,
        "speech": include_speech,
    }
    roots: list[tuple[str, Path]] = []
    has_typed_children = any((musan_root / name).is_dir() for name in MUSAN_TYPES)
    if has_typed_children:
        for name in MUSAN_TYPES:
            type_root = musan_root / name
            if include[name] and type_root.is_dir():
                roots.append((name, type_root))
        return roots

    root_type = musan_root.name if musan_root.name in MUSAN_TYPES else "noise"
    if include.get(root_type, True):
        roots.append((root_type, musan_root))
    return roots


def build_musan_noise_manifest(
    musan_root: str | Path,
    out_path: str | Path,
    include_music: bool = True,
    include_noise: bool = True,
    include_speech: bool = True,
    seed: int = 42,
) -> list[dict]:
    root = Path(musan_root)
    if not root.exists():
        raise FileNotFoundError(f"MUSAN root does not exist: {root}")

    rows: list[dict] = []
    for noise_type, type_root in _selected_type_roots(root, include_music, include_noise, include_speech):
        paths = sorted(path for path in type_root.rglob("*") if _is_audio(path))
        for path in paths:
            rows.append({
                "audio": str(path),
                "noise_type": noise_type,
                "noise_subtype": _noise_subtype(type_root, path),
            })

    rng = random.Random(seed)
    rng.shuffle(rows)
    write_noise_manifest(out_path, rows)
    return rows


def write_noise_manifest(path: str | Path, rows: Iterable[dict]) -> None:
    write_jsonl(path, rows)


def generate_babble_noise(
    speech_items: list[dict],
    out_dir: str | Path,
    num_files: int = 200,
    min_speakers: int = 3,
    max_speakers: int = 6,
    duration_seconds: float = 15.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> list[dict]:
    if not speech_items:
        raise ValueError("Cannot generate babble because no speech noise items are available.")
    if min_speakers < 1 or max_speakers < min_speakers:
        raise ValueError("Speaker count bounds are invalid.")
    if num_files < 1:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_len = int(duration_seconds * sample_rate)
    rng = random.Random(seed)
    rows: list[dict] = []
    import numpy as np

    for index in range(num_files):
        speaker_count = rng.randint(min_speakers, max_speakers)
        if len(speech_items) >= speaker_count:
            selected = rng.sample(speech_items, speaker_count)
        else:
            selected = [rng.choice(speech_items) for _ in range(speaker_count)]

        mixed = np.zeros(target_len, dtype=np.float32)
        item_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(item_seed)
        for item in selected:
            wav = read_audio(item["audio"], sr=sample_rate)
            fitted = fit_noise(wav, target_len, item_rng)
            rms = float(np.sqrt(np.mean(fitted ** 2))) if len(fitted) else 0.0
            if rms > 1e-8:
                fitted = fitted / rms
            mixed += fitted.astype(np.float32)

        mixed = mixed / max(float(speaker_count), 1.0)
        peak = max(float(np.max(np.abs(mixed))), 1.0)
        mixed = (mixed / peak).astype(np.float32)

        out_path = out_dir / f"babble_{index:05d}.wav"
        import soundfile as sf

        sf.write(out_path, mixed, sample_rate)
        rows.append({
            "audio": str(out_path),
            "noise_type": "babble",
            "noise_subtype": "synthetic_musan_speech",
            "source_count": speaker_count,
            "seed": item_seed,
        })

    return rows
