# Phat PhoWhisper LoRA Ablation Guide

## Scope

This guide covers only Nguyen Thanh Phat's deliverables:

- ordinary PhoWhisper-base LoRA with `lambda = 0`
- tone-aware LoRA with `L_total = L_ASR + lambda * L_tone`
- lambda values `0`, `0.05`, `0.1`, `0.3`, and `0.5`
- checkpoint/resume, prediction export, metric aggregation, and best-lambda selection

The benchmark builder, zero-shot baselines, team-wide error analysis, and multi-seed experiments remain owned by the other assigned members.

## Required environment

Use the existing `slp` conda environment. Do not create another environment and do not use the default Python interpreter.

```powershell
conda activate slp
& 'C:\Users\phath\.conda\envs\slp\python.exe' --version
& 'C:\Users\phath\.conda\envs\slp\python.exe' -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
& 'C:\Users\phath\.conda\envs\slp\python.exe' -m pip install -r requirements.txt
```

The verified development machine used Python 3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 4060 Laptop GPU.

## File structure

```text
configs/phat/
  base.yaml
  lambda_0.yaml
  lambda_005.yaml
  lambda_01.yaml
  lambda_03.yaml
  lambda_05.yaml
  phat_pipeline.yaml

src/vitonesr/phat/
  config.py
  evaluation.py
  losses.py
  modeling.py
  reproducibility.py
  selection.py
  trainer.py
  training_data.py

scripts/
  train_phat_lora.py
  train_all_lambdas.py
  evaluate_phat_checkpoint.py
  evaluate_all_lambdas.py
  select_best_lambda.py
  run_phat_pipeline.py

tests/phat/
```

Generated artifacts are written under `outputs/phat/` and are ignored by Git.

## Dataset contracts

Training uses the existing JSONL manifests. Each row must contain real `audio` and `text` values, and every audio path is validated before training.

```text
data/manifests/vivos/train.jsonl
data/manifests/vivos/dev.jsonl
data/manifests/noise/musan_noise.jsonl
```

Evaluation defaults to the team benchmark:

```text
outputs/benchmark/benchmark_manifest.csv
```

The evaluator reads `audio_path` and `transcript` from the benchmark schema and also supports the older JSONL `audio`/`text` fields. It never hard-codes audio filenames. A full official run expects 1,500 rows: 300 clean and 300 at each of 20, 10, 5, and 0 dB.

## Tone labels and tone-aware loss

The shared `src/vitonesr/tone.py` implementation defines six classes:

| ID | Tone |
| ---: | --- |
| 0 | ngang |
| 1 | sac |
| 2 | huyen |
| 3 | hoi |
| 4 | nga |
| 5 | nang |

Transcripts are Unicode-normalized and split into syllable-like whitespace tokens. Tone marks are extracted from Unicode combining marks. Digits, punctuation, uppercase acronyms, and tokens without a Vietnamese vowel use `ignore_index = -100`.

Each syllable is tokenized with the same Whisper tokenizer. The default `last_subtoken` policy attaches the tone to the final BPE piece and ignores preceding pieces. Special tokens and padding remain `-100`.

The decoder's final hidden state feeds a trainable `LayerNorm + Linear(512, 6)` tone head. Cross-entropy is computed only on valid labels. If a batch contains no valid tone positions, the function returns differentiable zero instead of `NaN`.

For `lambda = 0`, no tone head is created and the total loss is exactly the ASR loss. For positive lambda values, only PEFT LoRA weights and the tone head are trainable; the backbone remains frozen.

## Train one lambda

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\train_phat_lora.py `
  --config configs\phat\lambda_01.yaml `
  --device cuda
```

Useful overrides:

```text
--lambda-value
--seed
--manifest
--output-dir
--device
--overwrite
--max-train-samples
--max-train-steps
```

`--max-train-samples` and `--max-train-steps` are for real-data smoke validation. Outputs from limited runs must use a separate path containing `smoke` and must not be reported as the official ablation.

## Train all five lambdas

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\train_all_lambdas.py `
  --config configs\phat\phat_pipeline.yaml `
  --device cuda
```

Default checkpoint roots:

```text
outputs/phat/checkpoints/ckpt_lora_ordinary_lambda0/
outputs/phat/checkpoints/ckpt_tone_lora_lambda_005/
outputs/phat/checkpoints/ckpt_tone_lora_lambda_01/
outputs/phat/checkpoints/ckpt_tone_lora_lambda_03/
outputs/phat/checkpoints/ckpt_tone_lora_lambda_05/
```

Every final checkpoint contains:

```text
final/
  adapter/
  processor/
  tone_head.pt          # positive lambda only
  optimizer.pt
  scheduler.pt
  scaler.pt
  rng_state.pt
  trainer_state.json
  resolved_config.yaml
```

Periodic `checkpoint_step_XXXXXX/` directories use the same layout.

## Resume training

Resume from a periodic checkpoint with the same model, lambda, and LoRA rank:

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\train_phat_lora.py `
  --config configs\phat\lambda_01.yaml `
  --resume `
  --checkpoint outputs\phat\checkpoints\ckpt_tone_lora_lambda_01\checkpoint_step_000500 `
  --device cuda
```

The adapter, tone head, optimizer, scheduler, gradient scaler, Python/NumPy/PyTorch RNG state, epoch, batch position, and global step are restored. Existing output directories are rejected unless `--resume` or `--overwrite` is explicit.

## Evaluate one checkpoint

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\evaluate_phat_checkpoint.py `
  --config configs\phat\lambda_01.yaml `
  --checkpoint outputs\phat\checkpoints\ckpt_tone_lora_lambda_01\final `
  --manifest outputs\benchmark\benchmark_manifest.csv `
  --device cuda
```

Filters are available through `--subset`, `--snr`, and `--noise-type`.

Every prediction CSV uses exactly:

```text
utt_id,dataset,model,model_size,train_type,lambda,seed,snr,noise_type,ref,hyp
```

## Evaluate and aggregate all lambdas

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\evaluate_all_lambdas.py `
  --config configs\phat\phat_pipeline.yaml `
  --device cuda
```

This command requires real final checkpoints, evaluates all 1,500 benchmark rows for every lambda, exports five prediction files, and writes:

```text
outputs/phat/reports/lambda_ablation_results.csv
```

The result contains WER, CER, TER, DER, FCER, and SWDR when the shared metric implementation provides them. It includes aggregate rows for all data, clean, noisy-all, each SNR, each noise type, and each SNR/noise-type combination.

Limited evaluation is isolated explicitly:

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\evaluate_all_lambdas.py `
  --config configs\phat\phat_pipeline.yaml `
  --checkpoint-root outputs\phat\smoke\checkpoints `
  --output-dir outputs\phat\smoke\predictions `
  --results-path outputs\phat\smoke\reports\lambda_ablation_results.csv `
  --limit 5 `
  --allow-partial `
  --device cuda
```

## Select the best lambda

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\select_best_lambda.py `
  --config configs\phat\phat_pipeline.yaml
```

The selector:

1. computes sample-weighted TER and DER at 0 and 5 dB;
2. compares aggregate WER/CER against ordinary LoRA (`lambda = 0`);
3. rejects candidates exceeding the configured absolute WER/CER increases;
4. chooses the lowest weighted low-SNR TER/DER score;
5. breaks ties by WER, CER, then smaller lambda.

All thresholds, SNR values, and weights are in `configs/phat/base.yaml`. The report is written to:

```text
outputs/phat/reports/best_lambda_report.md
```

The default selector refuses to produce an official report if any lambda, clean result, low-SNR result, or expected benchmark sample count is missing. `--allow-partial` is only for clearly labeled diagnostics.

## End-to-end command

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' scripts\run_phat_pipeline.py `
  --config configs\phat\phat_pipeline.yaml `
  --device cuda
```

## Tests

The tests do not download a model or dataset. They use small tensors, a fake tokenizer, temporary CSV files, and in-memory selection rows.

```powershell
& 'C:\Users\phath\.conda\envs\slp\python.exe' -m unittest discover -s tests\phat -v
```

Covered behavior:

- six Vietnamese tone classes and invalid-token handling
- token-level tone labels and ignored positions
- finite zero loss for all-ignored batches
- tone loss depends on logits and labels
- exact `lambda = 0` loss identity
- exact prediction schema
- best-lambda selection and WER/CER guards
- loading all five configs
- deterministic Python, NumPy, and PyTorch seeds

## Current validation boundary

The implementation was first validated with unit tests, real PhoWhisper-base loading, resume checks, and limited GPU smoke runs. The official experiment was then completed in the existing `slp` environment on CUDA:

- all five lambdas (`0`, `0.05`, `0.1`, `0.3`, and `0.5`) completed three epochs and `2,187` optimizer steps
- every final checkpoint contains the adapter, processor, optimizer, scheduler, scaler, RNG state, resolved config, and trainer state; positive-lambda checkpoints also contain `tone_head.pt`
- every lambda was evaluated on all `1,500` benchmark rows: `300` clean and `300` for each SNR in `20`, `10`, `5`, and `0` dB
- all five prediction files contain the exact 11-column shared schema with no empty reference or hypothesis
- `outputs/phat/reports/lambda_ablation_results.csv` contains the real WER, CER, TER, DER, FCER, and SWDR aggregates
- the strict selector chose `lambda = 0.05`; see `outputs/phat/reports/best_lambda_report.md` for the measured comparison and selection rationale

These runtime artifacts are local and intentionally ignored by Git through the repository's `outputs/` rule. Re-run the commands above when the artifacts need to be reproduced on another machine. Smoke outputs under `outputs/phat/smoke/` must never replace or be mixed with the official artifacts.

## Troubleshooting

- `Output directory is not empty`: use a new path, `--resume` with a periodic checkpoint, or explicit `--overwrite`.
- `Missing PEFT adapter`: pass the experiment root containing `final/adapter/`, the `final/` checkpoint, or the adapter directory itself.
- `Resume config mismatch`: keep the same backbone, lambda, and LoRA rank as the saved checkpoint.
- CUDA out of memory: reduce the configured batch size from 16 and increase gradient accumulation so the effective batch remains 16. Record identical changes for all lambdas before comparing them.
- Missing audio path: regenerate/fix the manifest. The pipeline intentionally does not fabricate replacement data.
- Incomplete best-lambda report: complete all five prediction/evaluation runs with 300 samples for clean, 0 dB, and 5 dB.
- NumPy/PyTorch DLL `Access is denied` under a restricted shell: run the same exact `slp` interpreter in a normal authorized terminal; do not switch to default Python or create a new environment.
