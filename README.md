# Tone-aware Vietnamese ASR Starter Kit

This repository contains the code scaffold for a Vietnamese noisy-ASR project:
controlled VIVOS + MUSAN benchmark construction, Whisper/PhoWhisper inference,
WER/CER scoring, prototype Vietnamese tone/diacritic metrics, and a prepared
LoRA + tone-aware multi-task learning training path.

The current reported numbers are **baseline results only**. No fine-tuned model
has been trained yet. The next recommended step is to fine-tune PhoWhisper with
LoRA on the VIVOS manifests and compare ordinary noisy LoRA against the
tone-aware MTL variant.

## What is included

```text
configs/                 Training/evaluation config files
docs/                    Midterm runbook and experiment notes
latex/                   Paper skeleton
notebooks/               Colab demo notebook
scripts/                 Download, manifest, noise, inference, scoring scripts
src/vitonesr/            Dataset, model, tone, noise, and metric utilities
evaluate.py              Evaluation entrypoint scaffold
train.py                 LoRA + tone-aware MTL training entrypoint
requirements.txt         Python dependencies
```

Large files are intentionally excluded from Git:

```text
data/                    Downloaded VIVOS/MUSAN and generated noisy audio
outputs/                 Generated CSV/report/slide artifacts
experiments/             Fine-tuned checkpoints and training outputs
```

## Dataset links

Download datasets through the provided scripts or the official pages below.
Do not commit these files to GitHub.

| Dataset | Purpose | Official page | Direct download used by script |
| --- | --- | --- | --- |
| VIVOS | Vietnamese clean ASR speech | https://zenodo.org/records/7068130 | https://zenodo.org/records/7068130/files/vivos.tar.gz?download=1 |
| MUSAN | Noise for controlled SNR mixing | https://www.openslr.org/17/ | https://www.openslr.org/resources/17/musan.tar.gz |
| FLEURS Vietnamese | Optional external eval | https://huggingface.co/datasets/google/fleurs | Use `scripts/download_fleurs.py` |
| PhoWhisper-base | Recommended fine-tuning base model | https://huggingface.co/vinai/PhoWhisper-base | Hugging Face model hub |

Expected local dataset layout after download:

```text
data/raw/vivos/
data/raw/musan/musan/noise/
data/manifests/noise/musan_noise.jsonl
data/manifests/vivos/train.jsonl
data/manifests/vivos/dev.jsonl
data/manifests/vivos/test.jsonl
data/manifests/vivos/test_noisy.jsonl
```

## Environment setup

Recommended for training: Colab GPU or another CUDA machine. Local CPU is
acceptable for manifest generation and small inference smoke tests.

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

On Colab, clone the repository, enable GPU runtime, then run the same commands:

```bash
git clone git@github.com:KietIT/SLP.git
cd SLP
python -m pip install -r requirements.txt
```

If SSH is not configured in Colab, use the HTTPS clone URL instead.

## Prepare VIVOS + MUSAN

```bash
bash scripts/download_vivos.sh
bash scripts/download_musan.sh

python scripts/make_vivos_manifest.py \
  --vivos_root data/raw/vivos \
  --out_dir data/manifests/vivos

python scripts/make_noisy_test.py \
  --manifest data/manifests/vivos/test.jsonl \
  --noise_manifest data/manifests/noise/musan_noise.jsonl \
  --out_manifest data/manifests/vivos/test_noisy.jsonl \
  --limit 50 \
  --snrs 20 10 5 0 \
  --seed 42

python scripts/dataset_stats.py \
  data/manifests/vivos/test.jsonl \
  data/manifests/vivos/test_noisy.jsonl \
  --out outputs/dataset_stats.csv
```

Dataset statistics from the current midterm run:

| Manifest | SNR | Utterances | Hours | Avg seconds |
| --- | ---: | ---: | ---: | ---: |
| `data/manifests/vivos/test.jsonl` | clean | 760 | 0.746 | 3.53 |
| `data/manifests/vivos/test_noisy.jsonl` | 20 dB | 50 | 0.046 | 3.31 |
| `data/manifests/vivos/test_noisy.jsonl` | 10 dB | 50 | 0.046 | 3.31 |
| `data/manifests/vivos/test_noisy.jsonl` | 5 dB | 50 | 0.046 | 3.31 |
| `data/manifests/vivos/test_noisy.jsonl` | 0 dB | 50 | 0.046 | 3.31 |

## Current baseline inference

The current baseline uses `openai/whisper-base` with Vietnamese decoding forced
in `scripts/infer.py` via `language=vi` and `task=transcribe`.

Clean subset:

```bash
python scripts/infer.py \
  --manifest data/manifests/vivos/test.jsonl \
  --model openai/whisper-base \
  --out outputs/whisper_clean_forced.csv \
  --limit 30 \
  --language vi \
  --task transcribe

python scripts/score_predictions.py \
  --pred outputs/whisper_clean_forced.csv \
  --out outputs/metrics_whisper_clean_forced.csv
```

Noisy subset:

```bash
python scripts/infer.py \
  --manifest data/manifests/vivos/test_noisy.jsonl \
  --model openai/whisper-base \
  --out outputs/whisper_noisy_forced.csv \
  --limit 120 \
  --language vi \
  --task transcribe

python scripts/score_predictions.py \
  --pred outputs/whisper_noisy_forced.csv \
  --out outputs/metrics_whisper_noisy_forced_by_snr.csv \
  --group_by snr
```

### Baseline result table

All values are error rates, so lower is better. These results are from a small
midterm subset, not a final benchmark.

| Condition | N | WER | CER | TER simple | DER simple |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean, forced VI | 30 | 43.06% | 19.61% | 23.13% | 12.22% |
| Noisy all, forced VI | 120 | 47.95% | 23.55% | 25.98% | 10.82% |

Noisy results by SNR:

| SNR | N | WER | CER | TER simple | DER simple |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 dB | 30 | 43.42% | 20.02% | 21.00% | 10.17% |
| 10 dB | 30 | 49.82% | 24.02% | 25.62% | 9.62% |
| 5 dB | 30 | 45.55% | 22.39% | 25.62% | 11.24% |
| 0 dB | 30 | 53.02% | 27.78% | 31.67% | 12.41% |

Before/after decoder control:

| Set | Metric | Before | Forced VI | Delta |
| --- | --- | ---: | ---: | ---: |
| Clean | WER | 43.42% | 43.06% | -0.36 pp |
| Clean | CER | 20.26% | 19.61% | -0.65 pp |
| Noisy all | WER | 48.31% | 47.95% | -0.36 pp |
| Noisy all | CER | 24.08% | 23.55% | -0.53 pp |

Interpretation: forcing Vietnamese decoding removes occasional non-Vietnamese
outputs, but the aggregate improvement is small. The main remaining issue is
model adaptation, not just decoder configuration.

## Fast midterm reproduction

For a quick run with synthetic demo noise:

```bash
bash scripts/run_midterm_demo.sh
```

For real MUSAN noise:

```bash
USE_MUSAN=1 bash scripts/run_midterm_demo.sh
```

To change the baseline model:

```bash
MODEL=vinai/PhoWhisper-base USE_MUSAN=1 bash scripts/run_midterm_demo.sh
```

After running, inspect:

```text
outputs/dataset_stats.csv
outputs/*_clean.csv
outputs/*_noisy.csv
outputs/metrics_*_clean.csv
outputs/metrics_*_noisy_by_snr.csv
outputs/midterm_summary.md
```

`outputs/` is ignored by Git because it contains generated artifacts.

## Fine-tuning configuration

The main training config is:

```text
configs/phowhisper_base_lora.yaml
```

Important fields:

```yaml
model:
  name_or_path: vinai/PhoWhisper-base
  language: vi
  task: transcribe
  use_lora: true
  lora:
    r: 8
    alpha: 16
    dropout: 0.05
    target_modules: [q_proj, v_proj]

training:
  output_dir: experiments/phowhisper_base_lora_mtl
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
  fp16: true
  gradient_checkpointing: true
  lambda_tone: 0.10

data:
  train_manifest: data/manifests/vivos/train.jsonl
  valid_manifest: data/manifests/vivos/dev.jsonl
  test_manifest: data/manifests/vivos/test.jsonl
  test_noisy_manifest: data/manifests/vivos/test_noisy.jsonl

noise:
  enable_train_noise: true
  noise_manifest: data/manifests/noise/musan_noise.jsonl
  snr_choices: [20, 10, 5, 0]
  prob: 0.70
```

Run fine-tuning on Colab/GPU:

```bash
python train.py --config configs/phowhisper_base_lora.yaml
```

Evaluate a trained checkpoint:

```bash
python evaluate.py \
  --config configs/phowhisper_base_lora.yaml \
  --checkpoint experiments/phowhisper_base_lora_mtl \
  --split test_noisy \
  --out outputs/phowhisper_mtl_noisy.csv
```

Score the predictions by SNR:

```bash
python scripts/score_predictions.py \
  --pred outputs/phowhisper_mtl_noisy.csv \
  --out outputs/metrics_phowhisper_mtl_noisy_by_snr.csv \
  --group_by snr
```

## Recommended experiment order

1. Reproduce the current `openai/whisper-base` baseline.
2. Run `vinai/PhoWhisper-base` zero-shot on the same clean/noisy subset.
3. Fine-tune PhoWhisper with clean LoRA.
4. Fine-tune PhoWhisper with noisy LoRA using MUSAN augmentation.
5. Fine-tune tone-aware MTL with `lambda_tone` values such as `0.05`, `0.10`,
   and `0.30`.
6. Compare WER, CER, TER simple, and DER simple across clean/noisy/SNR splits.
7. Replace simple TER/DER with edit-aligned syllable metrics after the core
   LoRA-vs-MTL comparison is stable.

## Current project claim

At midterm, the correct claim is:

> The project has a reproducible Vietnamese noisy-ASR pipeline, real VIVOS +
> MUSAN baseline results, forced Vietnamese decoding, and prepared LoRA +
> tone-aware MTL training code.

Do **not** claim that tone-aware MTL improves results until a fine-tuned model
has been trained and evaluated on the same clean/noisy manifests.
