from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.noise import _fit_noise, mix_at_snr, read_audio


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:120] or "utt"


def main() -> None:
    p = argparse.ArgumentParser(description="Create a fixed noisy ASR test manifest from clean speech and a noise manifest.")
    p.add_argument("--manifest", required=True, help="Clean JSONL manifest with audio/text fields.")
    p.add_argument("--noise_manifest", required=True, help="JSONL manifest created by make_noise_manifest.py.")
    p.add_argument("--out_dir", default="data/noisy/test", help="Directory for generated noisy wav files.")
    p.add_argument("--out_manifest", default="data/manifests/vivos/test_noisy.jsonl")
    p.add_argument("--snrs", type=float, nargs="+", default=[20, 10, 5, 0])
    p.add_argument("--limit", type=int, default=50, help="Maximum clean utterances to use. Use <=50 for a one-day midterm run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    args = p.parse_args()

    rng = random.Random(args.seed)
    items = read_jsonl(Path(args.manifest))
    noises = read_jsonl(Path(args.noise_manifest))
    if not noises:
        raise ValueError("Noise manifest is empty.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_manifest.open("w", encoding="utf-8") as f:
        for item in items[: args.limit]:
            clean = read_audio(item["audio"], sr=args.sample_rate)
            utt_id = str(item.get("utt_id") or Path(item["audio"]).stem)
            for snr in args.snrs:
                noise_item = rng.choice(noises)
                noise = read_audio(noise_item["audio"], sr=args.sample_rate)
                fitted = _fit_noise(noise, len(clean), rng)
                mixed = mix_at_snr(clean, fitted, snr)
                snr_label = str(int(snr)) if float(snr).is_integer() else str(snr).replace(".", "p")
                rel_name = f"{safe_stem(utt_id)}_snr{snr_label}.wav"
                audio_out = out_dir / rel_name
                sf.write(audio_out, mixed, args.sample_rate)
                row = {
                    **item,
                    "audio": str(audio_out),
                    "source_audio": item["audio"],
                    "utt_id": f"{utt_id}_snr{snr_label}",
                    "snr": snr,
                    "noise_path": noise_item["audio"],
                    "noise_type": noise_item.get("noise_type", Path(noise_item["audio"]).parent.name),
                    "seed": args.seed,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"wrote {written} noisy utterances to {out_manifest}")


if __name__ == "__main__":
    main()
