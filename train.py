from __future__ import annotations

import argparse
import math
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.vitonesr.data import ASRManifestDataset, DataCollatorSpeechSeq2SeqWithTone
from src.vitonesr.model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    accelerator = Accelerator(mixed_precision="fp16" if cfg["training"].get("fp16") else "no")
    processor = WhisperProcessor.from_pretrained(cfg["model"]["name_or_path"], language=cfg["model"].get("language", "vi"), task=cfg["model"].get("task", "transcribe"))

    mcfg = cfg["model"]
    lcfg = mcfg.get("lora", {})
    model = build_model(
        mcfg["name_or_path"],
        use_lora=mcfg.get("use_lora", True),
        lora_r=lcfg.get("r", 8),
        lora_alpha=lcfg.get("alpha", 16),
        lora_dropout=lcfg.get("dropout", 0.05),
        target_modules=lcfg.get("target_modules", ["q_proj", "v_proj"]),
        lambda_tone=cfg["training"].get("lambda_tone", 0.1),
    )
    if cfg["training"].get("gradient_checkpointing", False):
        model.base_model.gradient_checkpointing_enable()

    train_ds = ASRManifestDataset(
        cfg["data"]["train_manifest"], processor,
        sample_rate=cfg["data"].get("sample_rate", 16000),
        max_audio_seconds=cfg["training"].get("max_audio_seconds", 15.0),
        max_label_length=cfg["training"].get("max_label_length"),
        noise_manifest=cfg.get("noise", {}).get("noise_manifest"),
        noise_prob=cfg.get("noise", {}).get("prob", 0.0) if cfg.get("noise", {}).get("enable_train_noise", False) else 0.0,
        snr_choices=cfg.get("noise", {}).get("snr_choices", [20, 10, 5, 0]),
        tone_label_policy=cfg["training"].get("tone_label_policy", "last_subtoken"),
    )
    collator = DataCollatorSpeechSeq2SeqWithTone(processor, model.base_model.config.decoder_start_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"].get("per_device_train_batch_size", 1), shuffle=True, collate_fn=collator, num_workers=2)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["training"].get("learning_rate", 1e-4)))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg["training"].get("gradient_accumulation_steps", 1)))
    total_steps = steps_per_epoch * cfg["training"].get("num_train_epochs", 3)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg["training"].get("warmup_steps", 100), num_training_steps=total_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    grad_acc = cfg["training"].get("gradient_accumulation_steps", 1)
    global_step = 0
    model.train()
    for epoch in range(cfg["training"].get("num_train_epochs", 3)):
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            out = model(**batch)
            loss = out["loss"] / grad_acc
            accelerator.backward(loss)
            should_update = (step + 1) % grad_acc == 0 or (step + 1) == len(train_loader)
            if should_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_description(f"epoch={epoch} step={global_step} loss={loss.item()*grad_acc:.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        out_dir = cfg["training"].get("output_dir", "experiments/run")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        unwrapped.base_model.save_pretrained(out_dir)
        torch.save(unwrapped.tone_head.state_dict(), f"{out_dir}/tone_head.pt")
        processor.save_pretrained(out_dir)
        print(f"saved to {out_dir}")

if __name__ == "__main__":
    main()
