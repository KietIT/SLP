"""Create a small deterministic subset manifest from VietSuperSpeech for cloud experiments.

Use this only after checking dataset license/terms and storage constraints.
"""
import argparse
import json
from pathlib import Path
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=30.0)
    p.add_argument("--out", default="data/manifests/vietsuperspeech/train_30h.jsonl")
    p.add_argument("--cache_dir", default="data/raw/hf_cache")
    args = p.parse_args()
    ds = load_dataset("thanhnew2001/VietSuperSpeech", split="train", cache_dir=args.cache_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds.shuffle(seed=42):
            audio = row.get("audio")
            text = row.get("text") or row.get("transcription") or row.get("sentence")
            if not audio or not text:
                continue
            duration = row.get("duration") or 0.0
            if isinstance(audio, dict) and not duration and audio.get("array") is not None:
                duration = len(audio["array"]) / audio["sampling_rate"]
            item = {
                "audio": audio["path"] if isinstance(audio, dict) else audio,
                "text": text,
                "dataset": "vietsuperspeech",
                "split": "train_subset",
                "snr": "clean",
                "duration": duration,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += float(duration)
            n += 1
            if total / 3600 >= args.hours:
                break
    print(f"wrote {n} utterances, {total/3600:.2f} hours to {out}")


if __name__ == "__main__":
    main()
