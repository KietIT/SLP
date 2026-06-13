from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from tqdm.auto import tqdm

from src.vitonesr.data import ASRManifestDataset, DataCollatorSpeechSeq2SeqWithTone
from src.vitonesr.model import build_model
from src.vitonesr.metrics import compute_all


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None, help="Optional CSV path for predictions.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ckpt = args.checkpoint or cfg["model"]["name_or_path"]
    language = cfg["model"].get("language", "vi")
    task = cfg["model"].get("task", "transcribe")
    processor = WhisperProcessor.from_pretrained(ckpt, language=language, task=task)
    model = build_model(ckpt, use_lora=False, lambda_tone=0.0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model.eval().to(device)
    manifest = cfg["data"].get(f"{args.split}_manifest", cfg["data"].get("test_manifest"))
    ds = ASRManifestDataset(
        manifest,
        processor,
        sample_rate=cfg["data"].get("sample_rate", 16000),
        max_audio_seconds=cfg["training"].get("max_audio_seconds", 15.0),
        max_label_length=cfg["training"].get("max_label_length"),
    )
    if args.limit:
        ds.items = ds.items[:args.limit]
    refs, hyps = [], []
    rows = []
    for raw, item in tqdm(zip(ds.items, ds), total=len(ds)):
        feats = torch.tensor(item["input_features"], dtype=torch.float32).unsqueeze(0).to(device)
        try:
            ids = model.generate(feats, max_new_tokens=args.max_new_tokens, language=language, task=task)
        except TypeError:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
            ids = model.generate(feats, max_new_tokens=args.max_new_tokens, forced_decoder_ids=forced_decoder_ids)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        refs.append(item["text"])
        hyps.append(hyp)
        rows.append({
            "utt_id": raw.get("utt_id", Path(raw["audio"]).stem),
            "audio": raw["audio"],
            "text": item["text"],
            "prediction": hyp,
            "snr": raw.get("snr", "clean"),
            "noise_type": raw.get("noise_type", ""),
            "dataset": raw.get("dataset", ""),
        })
    metrics = compute_all(refs, hyps)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["utt_id", "audio", "text", "prediction", "snr", "noise_type", "dataset"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote predictions to {out}")

if __name__ == "__main__":
    main()
