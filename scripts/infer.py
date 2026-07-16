from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.noise import read_audio


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Run Whisper/PhoWhisper inference over a JSONL manifest and write predictions CSV.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", default="vinai/PhoWhisper-base")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_audio_seconds", type=float, default=15.0)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--language", default="vi")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = choose_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    generate_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.language:
        generate_kwargs["language"] = args.language
    if args.task:
        generate_kwargs["task"] = args.task

    rows = read_jsonl(Path(args.manifest))
    if args.limit:
        rows = rows[: args.limit]
    max_len = int(args.max_audio_seconds * args.sample_rate)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utt_id", "audio", "text", "prediction", "snr", "noise_type", "dataset"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in tqdm(rows):
            wav = read_audio(row["audio"], sr=args.sample_rate)
            if len(wav) > max_len:
                wav = wav[:max_len]
            features = processor.feature_extractor(wav, sampling_rate=args.sample_rate, return_tensors="pt").input_features
            features = features.to(device=device, dtype=dtype)
            try:
                pred_ids = model.generate(features, **generate_kwargs)
            except TypeError:
                fallback_kwargs = {"max_new_tokens": args.max_new_tokens}
                fallback_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
                pred_ids = model.generate(features, **fallback_kwargs)
            prediction = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            writer.writerow({
                "utt_id": row.get("utt_id", Path(row["audio"]).stem),
                "audio": row["audio"],
                "text": row["text"],
                "prediction": prediction,
                "snr": row.get("snr", "clean"),
                "noise_type": row.get("noise_type", ""),
                "dataset": row.get("dataset", ""),
            })

    print(f"wrote {len(rows)} predictions to {out_path} on {device}")


if __name__ == "__main__":
    main()
