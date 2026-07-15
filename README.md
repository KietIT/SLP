# Tone-aware Vietnamese ASR Starter Kit

This repository contains the code scaffold for a Vietnamese noisy-ASR project:
controlled VIVOS + MUSAN benchmark construction, Whisper/PhoWhisper inference,
WER/CER scoring, Vietnamese diagnostic metrics, and a PhoWhisper LoRA +
tone-aware multi-task learning training path.

The current midterm run compares three systems on the same clean and noisy
VIVOS manifests:

1. `openai/whisper-base` zero-shot
2. `vinai/PhoWhisper-base` zero-shot
3. `vinai/PhoWhisper-base` tone-aware LoRA trained on VIVOS with MUSAN noise
   augmentation

All reported results use six metrics: `WER`, `CER`, `TER`, `DER`, `FCER`, and
`SWDR`.

## What is included

```text
configs/                 Training/evaluation config files
docs/                    Midterm runbook and experiment notes
latex/                   Paper skeleton
notebooks/               Result analysis notebook
scripts/                 Manifest, noise, inference, scoring, pipeline scripts
src/vitonesr/            Dataset, model, tone, noise, and metric utilities
evaluate.py              Evaluation entrypoint scaffold
train.py                 LoRA + tone-aware MTL training entrypoint
requirements.txt         Python dependencies
```

Large generated files are intentionally excluded from Git by default:

```text
data/                    Downloaded VIVOS/MUSAN and generated noisy audio
outputs/                 Generated CSV/report/slide artifacts
experiments/             Fine-tuned checkpoints and training outputs
```

For coursework submission, keep `data/` out of Git. If the team wants to share
the generated result CSVs/checkpoints, add selected `outputs/` and
`experiments/` files explicitly.

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
data/manifests/fleurs/test.jsonl
data/manifests/fleurs/audio/test/*.wav
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

python scripts/make_noise_manifest.py \
  --noise_root data/raw/musan/musan/noise \
  --out data/manifests/noise/musan_noise.jsonl

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

## Full midterm pipeline

Run the complete experiment pipeline with the active Python environment:

```powershell
conda activate slp
python scripts/run_full_midterm_pipeline.py
```

The pipeline performs these steps:

1. Create VIVOS clean manifests.
2. Create the MUSAN noise manifest.
3. Create a fixed noisy VIVOS test set with SNR `20/10/5/0`.
4. Run Whisper-base and PhoWhisper-base zero-shot inference on clean/noisy sets.
5. Train PhoWhisper tone-aware LoRA with `configs/phowhisper_base_lora.yaml`.
6. Evaluate the LoRA checkpoint on the same clean/noisy sets.
7. Score all predictions with six metrics.
8. Build the final comparison table.

Main outputs:

```text
outputs/metrics_whisper_clean.csv
outputs/metrics_whisper_noisy_by_snr.csv
outputs/metrics_phowhisper_clean.csv
outputs/metrics_phowhisper_noisy_by_snr.csv
outputs/metrics_phowhisper_lora_clean.csv
outputs/metrics_phowhisper_lora_noisy_by_snr.csv
outputs/model_comparison_6metrics.csv
notebooks/midterm_results_analysis.ipynb
```

## Robust benchmark and zero-shot baseline

The larger benchmark pipeline keeps the midterm outputs untouched and writes new
artifacts under `outputs/benchmark/`, `outputs/zero_shot/`, and
`data/noisy_eval/`.

Build a typed MUSAN manifest:

```bash
python scripts/make_musan_noise_manifest_typed.py \
  --musan_root data/raw/musan/musan \
  --out data/manifests/noise/musan_noise_typed.jsonl \
  --seed 42
```

Build the fixed benchmark:

```bash
python scripts/build_robust_benchmark.py \
  --vivos_manifest data/manifests/vivos/test.jsonl \
  --noise_manifest data/manifests/noise/musan_noise_typed.jsonl \
  --out_manifest outputs/benchmark/benchmark_manifest.csv \
  --pool_manifest outputs/benchmark/benchmark_pool_manifest.csv \
  --report_out outputs/benchmark/benchmark_report.md \
  --out_noisy_dir data/noisy_eval \
  --pool_size 500 \
  --eval_size 300 \
  --snrs 20 10 5 0 \
  --seed 42 \
  --sample_rate 16000
```

Run one zero-shot model:

```bash
python scripts/infer_zero_shot.py \
  --benchmark_manifest outputs/benchmark/benchmark_manifest.csv \
  --model_name_or_path openai/whisper-tiny \
  --model whisper \
  --model_size tiny \
  --out outputs/zero_shot/pred_whisper_tiny.csv \
  --batch_size 4 \
  --device auto \
  --resume
```

Run the full zero-shot pipeline:

```bash
python scripts/run_zero_shot_pipeline.py \
  --vivos_root data/raw/vivos \
  --musan_root data/raw/musan/musan \
  --seed 42 \
  --pool_size 500 \
  --eval_size 300 \
  --batch_size 4 \
  --device auto
```

Smoke test with a small benchmark and one model:

```bash
python scripts/run_zero_shot_pipeline.py \
  --vivos_root data/raw/vivos \
  --musan_root data/raw/musan/musan \
  --seed 42 \
  --pool_size 10 \
  --eval_size 3 \
  --snrs 20 0 \
  --models whisper_tiny \
  --batch_size 1 \
  --device auto \
  --smoke_test
```

Aggregate and validate after predictions are available:

```bash
python scripts/aggregate_zero_shot.py \
  --pred_dir outputs/zero_shot \
  --out_by_snr outputs/zero_shot/zero_shot_results_by_snr.csv \
  --out_by_noise_type outputs/zero_shot/zero_shot_results_by_noise_type.csv

python scripts/validate_robust_benchmark.py \
  --benchmark_manifest outputs/benchmark/benchmark_manifest.csv \
  --pool_manifest outputs/benchmark/benchmark_pool_manifest.csv \
  --pred_dir outputs/zero_shot \
  --expected_eval_size 300 \
  --expected_pool_size 500 \
  --snrs 20 10 5 0
```

The benchmark manifest has 1500 rows for the full run: 300 clean utterances and
300 noisy utterances at each SNR level. Prediction CSV files use the schema
`utt_id,dataset,model,model_size,snr,noise_type,ref,hyp`.

## Metrics

All metrics are error rates, so lower is better.

| Metric | Meaning |
| --- | --- |
| WER | Word Error Rate |
| CER | Character Error Rate |
| TER simple | Simple Vietnamese tone error rate |
| DER simple | Simple Vietnamese diacritic error rate |
| FCER simple | Final consonant error rate for Vietnamese final consonants |
| SWDR simple | Short word deletion rate over reference occurrences in the fixed lexicon: `đã`, `có`, `là`, `một`, `và` |

`TER`, `DER`, `FCER`, and `SWDR` are diagnostic metrics for this midterm
project. They should be reported as simple/prototype Vietnamese ASR error
indicators, not as external benchmark-standard metrics.

## Current midterm results

These results are from the current small midterm subset:

| Model | Condition | N | WER | CER | TER | DER | FCER | SWDR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whisper-base zero-shot | Clean | 30 | 43.06% | 19.61% | 23.13% | 12.22% | 22.78% | 4.35% |
| Whisper-base zero-shot | Noisy all | 120 | 48.04% | 23.69% | 26.07% | 10.56% | 30.54% | 3.26% |
| PhoWhisper-base zero-shot | Clean | 30 | 8.19% | 4.66% | 2.49% | 0.39% | 4.43% | 0.00% |
| PhoWhisper-base zero-shot | Noisy all | 120 | 11.83% | 6.41% | 3.29% | 0.71% | 6.01% | 0.00% |
| PhoWhisper tone-aware LoRA | Clean | 30 | 9.25% | 4.49% | 1.42% | 0.00% | 5.06% | 0.00% |
| PhoWhisper tone-aware LoRA | Noisy all | 120 | 11.39% | 5.82% | 2.76% | 0.70% | 6.01% | 0.00% |

Noisy WER by SNR:

| Model | 20 dB | 10 dB | 5 dB | 0 dB |
| --- | ---: | ---: | ---: | ---: |
| Whisper-base zero-shot | 43.42% | 49.82% | 45.91% | 53.02% |
| PhoWhisper-base zero-shot | 8.90% | 9.25% | 11.74% | 17.44% |
| PhoWhisper tone-aware LoRA | 9.61% | 9.96% | 11.03% | 14.95% |

Interpretation:

- PhoWhisper-base strongly outperforms Whisper-base for Vietnamese ASR.
- Tone-aware LoRA slightly improves noisy overall WER compared with PhoWhisper
  zero-shot.
- The clearest LoRA gain appears at the hardest `0 dB` noisy condition.
- LoRA does not improve every condition; clean WER is slightly worse than the
  PhoWhisper zero-shot baseline.

## Fine-tuning configuration

### Phat: five-lambda LoRA ablation pipeline

The complete ordinary/tone-aware PhoWhisper-base LoRA workflow is documented in
[`docs/PHAT_LORA_GUIDE.md`](docs/PHAT_LORA_GUIDE.md). It adds five separate
lambda configs, deterministic training, checkpoint/resume state, exact shared
prediction schema export, full benchmark aggregation, guarded best-lambda
selection, and small unit tests that do not load a large model.

Use the existing `slp` conda environment and its exact interpreter for this
workspace. Limited smoke outputs must stay under `outputs/phat/smoke/` and must
not be reported as full ablation results.

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
  --out outputs/phowhisper_lora_noisy.csv
```

Score the predictions by SNR:

```bash
python scripts/score_predictions.py \
  --pred outputs/phowhisper_lora_noisy.csv \
  --out outputs/metrics_phowhisper_lora_noisy_by_snr.csv \
  --group_by snr
```

## FLEURS Vietnamese external evaluation

The external evaluation uses all 857 utterances in the FLEURS Vietnamese test
split and the same `aligned_v1` metric implementation as the internal
benchmark. It compares ordinary LoRA, tone-aware LoRA with lambda 0.05, and
tone-aware LoRA with lambda 0.1 on exactly the same utterance IDs and
references. Audio longer than 30 seconds is split into balanced, non-overlapping
chunks to avoid sub-second tails, and each chunk has a 440-token decode limit
within Whisper's 448-position target window. If the fast tokenizer encounters
an invalid byte-BPE sequence, the runner validates the slow tokenizer has the
same vocabulary and decodes only that sequence with invalid bytes ignored.

Published-run provenance: FLEURS manifest SHA-256
`4BEF10B833B7AE8B39D0202F2849564ED4299562B8C546C8828DB7900DC1EA22`,
PhoWhisper-base revision `7ebdb9e88f5cc5271fb88f4d642c82ff9388650e`, and adapter
SHA-256 values `821B91821DBE30029C044DD106692CA85B92307B9589799A2115D708A22A79F6`
(ordinary), `A62F81FFB31BF2C72F01B405ED322CDEBE30C5D2ECD54F426FE43131D641DDB3`
(lambda 0.05), and
`28BFBE3D3C1BEDF63A4CF92DAA5446BD6F536F220EA02293BD5071D420606C3B`
(lambda 0.1).

```bash
python scripts/download_fleurs.py
python scripts/run_external_fleurs.py --device cuda
python scripts/error_analysis.py --pred-glob "outputs/external/fleurs/predictions/pred_*.csv" --out-dir outputs/external/fleurs/error_analysis
python scripts/build_error_breakdowns.py --events outputs/external/fleurs/error_analysis/error_events.csv --out-dir outputs/external/fleurs/error_analysis
python scripts/bootstrap_ci.py --ordinary outputs/external/fleurs/predictions/pred_lora_ordinary_lambda0.csv --lambda-005 outputs/external/fleurs/predictions/pred_tone_lora_lambda_005.csv --lambda-01 outputs/external/fleurs/predictions/pred_tone_lora_lambda_01.csv --output outputs/external/fleurs/bootstrap_ci_results.csv
```

Use `scripts/build_error_artifacts.py` with `--overall-only` and one
`--focus-run` for each of `ordinary_lora:0`, `tone_aware_lora:0.05`, and
`tone_aware_lora:0.1` to build the tone, final-coda, and short-word artifacts.
`orthographic_breakdown.csv` additionally reports the requested missing-mark,
wrong-tone-mark, and wrong-vowel-mark diagnostics. These three feature labels
are explicitly nonexclusive; the primary TER and DER tables remain additive.

Current full-test results (lower is better):

| Configuration | N | WER | CER | TER | DER | FCER | SWDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary LoRA | 857 | 16.6384% | 9.8534% | 1.1220% | 0.5373% | 8.3826% | 0.4915% |
| Tone-aware LoRA, lambda 0.05 | 857 | 16.9235% | 10.0067% | 1.1661% | 0.5295% | 8.5887% | 0.6554% |
| Tone-aware LoRA, lambda 0.1 | 857 | 17.1123% | 10.0137% | 1.0684% | 0.5593% | 8.5189% | 0.4915% |

Paired percentile bootstrap uses 1,000 shared utterance-level resamples, seed
42, and reports delta as configuration B minus configuration A. The 95% CI
excludes zero only for WER and CER in ordinary LoRA versus lambda 0.05, and for
WER in ordinary LoRA versus lambda 0.1. All three significant deltas are
positive, so they favor ordinary LoRA on this FLEURS test set. The complete 12
intervals are in `outputs/external/fleurs/bootstrap_ci_results.csv`.

## Completed experiment order

1. Reproduce the `openai/whisper-base` zero-shot baseline.
2. Run `vinai/PhoWhisper-base` zero-shot on the same clean/noisy subset.
3. Fine-tune PhoWhisper tone-aware LoRA using VIVOS train and MUSAN
   augmentation.
4. Evaluate all three systems on the same clean and noisy manifests.
5. Compare WER, CER, TER, DER, FCER, and SWDR across clean/noisy/SNR splits.
6. Build the final table at `outputs/model_comparison_6metrics.csv`.

## Current project claim

At midterm, the correct claim is:

> The project has a reproducible Vietnamese noisy-ASR pipeline, real VIVOS +
> MUSAN evaluation data, Whisper/PhoWhisper zero-shot baselines, a trained
> PhoWhisper tone-aware LoRA checkpoint, and six-metric comparison results.

The strongest safe conclusion is that PhoWhisper is much better than
Whisper-base for Vietnamese ASR, while tone-aware LoRA gives a small noisy-set
gain and clearer improvement at severe `0 dB` noise, but does not improve every
condition.
