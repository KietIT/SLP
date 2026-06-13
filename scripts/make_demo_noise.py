from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def normalize(wav: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(wav)))
    if peak > 0:
        wav = wav / peak
    return (0.5 * wav).astype(np.float32)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(x, kernel, mode="same")


def write_noise(path: Path, wav: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, normalize(wav), sr)


def main() -> None:
    p = argparse.ArgumentParser(description="Create small controlled noise WAV files for fast midterm demos.")
    p.add_argument("--out_dir", default="data/raw/demo_noise")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    sr = args.sample_rate
    n = int(args.seconds * sr)
    t = np.arange(n, dtype=np.float32) / float(sr)
    out_dir = Path(args.out_dir)

    white = rng.normal(0, 1, n).astype(np.float32)
    write_noise(out_dir / "white" / "white_01.wav", white, sr)

    fan = 0.6 * np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 240 * t)
    fan += 0.3 * moving_average(rng.normal(0, 1, n).astype(np.float32), 400)
    write_noise(out_dir / "fan" / "fan_01.wav", fan.astype(np.float32), sr)

    rain = rng.normal(0, 1, n).astype(np.float32)
    rain = rain - moving_average(rain, 80)
    rain += 0.25 * rng.normal(0, 1, n).astype(np.float32)
    write_noise(out_dir / "rain" / "rain_01.wav", rain, sr)

    traffic = moving_average(rng.normal(0, 1, n).astype(np.float32), 1200)
    traffic += 0.4 * np.sin(2 * np.pi * 55 * t)
    for center in rng.integers(low=sr, high=max(sr + 1, n - sr), size=18):
        width = rng.integers(sr // 10, sr // 2)
        lo = max(0, center - width)
        hi = min(n, center + width)
        traffic[lo:hi] += np.hanning(hi - lo).astype(np.float32) * rng.uniform(0.5, 1.2)
    write_noise(out_dir / "traffic" / "traffic_01.wav", traffic.astype(np.float32), sr)

    cafe = 0.45 * rng.normal(0, 1, n).astype(np.float32)
    cafe += 0.25 * moving_average(rng.normal(0, 1, n).astype(np.float32), 300)
    for center in rng.integers(low=0, high=n, size=80):
        width = rng.integers(80, 700)
        lo = max(0, center - width)
        hi = min(n, center + width)
        cafe[lo:hi] += np.hanning(hi - lo).astype(np.float32) * rng.uniform(-0.8, 0.8)
    write_noise(out_dir / "cafe" / "cafe_01.wav", cafe.astype(np.float32), sr)

    print(f"wrote demo noise files under {out_dir}")


if __name__ == "__main__":
    main()
