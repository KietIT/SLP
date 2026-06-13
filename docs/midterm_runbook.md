# Midterm Runbook: From Dataset to Report Tables

## Goal

Produce enough real artifacts for the midterm slide/report:

- clean/noisy ASR manifests
- dataset statistics
- zero-shot baseline prediction CSV
- WER/CER/TER/DER metric CSV
- 2-3 qualitative error examples under noise
- `outputs/midterm_summary.md` for fast report/slide drafting

## Dataset Priority

Use this order:

1. VIVOS: required clean Vietnamese ASR dataset for the midterm prototype.
2. MUSAN noise: required if download time/storage is acceptable.
3. FLEURS Vietnamese: optional external evaluation after the midterm.
4. VietSuperSpeech: optional larger/cloud training data after the midterm.

Do not wait for FLEURS or VietSuperSpeech before producing the midterm results.

## Official Sources

- VIVOS: https://zenodo.org/records/7068130
- MUSAN: https://www.openslr.org/17/
- FLEURS: https://huggingface.co/datasets/google/fleurs
- PhoWhisper-base: https://huggingface.co/vinai/PhoWhisper-base

## Recommended Runtime

Use Colab GPU for inference/training if local hardware is slow. Use local only for editing, manifest inspection, and small smoke tests.

Colab runtime:

- Python 3.10+
- GPU runtime enabled
- repo folder mounted from Google Drive or uploaded as zip

Notebook option:

- Open `notebooks/midterm_colab_demo.ipynb` in Colab.
- Adjust the `%cd /content/tone_asr_starter` cell if your folder is in Google Drive.
- Run cells top to bottom.

## Environment

```bash
cd tone_asr_starter
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Fast Path

Use this when you need report/slide CSV files quickly:

```bash
bash scripts/run_midterm_demo.sh
```

This path downloads VIVOS, generates a small controlled noise set locally, creates clean/noisy manifests, runs `openai/whisper-base`, and writes all required CSV/Markdown files under `outputs/`.
The script also runs `scripts/validate_midterm_outputs.py`; success means the required report/slide artifacts exist.

If you have time/storage for real MUSAN noise:

```bash
USE_MUSAN=1 bash scripts/run_midterm_demo.sh
```

To run PhoWhisper instead of Whisper:

```bash
MODEL=vinai/PhoWhisper-base bash scripts/run_midterm_demo.sh
```

## Step 1: Download Data

```bash
bash scripts/download_vivos.sh
bash scripts/download_musan.sh
```

Expected important paths:

```text
data/raw/vivos/
data/raw/musan/musan/noise/
data/manifests/noise/musan_noise.jsonl
```

If MUSAN is too large or too slow, use a small folder of public/recorded noise WAV files and run:

```bash
python scripts/make_noise_manifest.py --noise_root data/raw/my_noise --out data/manifests/noise/my_noise.jsonl
```

Then replace `musan_noise.jsonl` with `my_noise.jsonl` in the later commands. Label this honestly in slides as a small controlled noise subset.

## Step 2: Create Manifests

```bash
python scripts/make_vivos_manifest.py --vivos_root data/raw/vivos --out_dir data/manifests/vivos
python scripts/make_noisy_test.py --manifest data/manifests/vivos/test.jsonl --noise_manifest data/manifests/noise/musan_noise.jsonl --out_manifest data/manifests/vivos/test_noisy.jsonl --limit 50 --snrs 20 10 5 0 --seed 42
python scripts/dataset_stats.py data/manifests/vivos/test.jsonl data/manifests/vivos/test_noisy.jsonl --out outputs/dataset_stats.csv
```

Expected outputs:

```text
data/manifests/vivos/train.jsonl
data/manifests/vivos/dev.jsonl
data/manifests/vivos/test.jsonl
data/manifests/vivos/test_noisy.jsonl
outputs/dataset_stats.csv
```

## Step 3: Run Baseline Inference

Start with Whisper-base because it is a stable baseline:

```bash
python scripts/infer.py --manifest data/manifests/vivos/test.jsonl --model openai/whisper-base --out outputs/whisper_clean.csv --limit 30
python scripts/infer.py --manifest data/manifests/vivos/test_noisy.jsonl --model openai/whisper-base --out outputs/whisper_noisy.csv --limit 120
```

If time/GPU allows, run PhoWhisper too:

```bash
python scripts/infer.py --manifest data/manifests/vivos/test.jsonl --model vinai/PhoWhisper-base --out outputs/phowhisper_clean.csv --limit 30
python scripts/infer.py --manifest data/manifests/vivos/test_noisy.jsonl --model vinai/PhoWhisper-base --out outputs/phowhisper_noisy.csv --limit 120
```

## Step 4: Score Results

```bash
python scripts/score_predictions.py --pred outputs/whisper_clean.csv --out outputs/metrics_whisper_clean.csv
python scripts/score_predictions.py --pred outputs/whisper_noisy.csv --out outputs/metrics_whisper_noisy_by_snr.csv --group_by snr
```

For PhoWhisper, if available:

```bash
python scripts/score_predictions.py --pred outputs/phowhisper_clean.csv --out outputs/metrics_phowhisper_clean.csv
python scripts/score_predictions.py --pred outputs/phowhisper_noisy.csv --out outputs/metrics_phowhisper_noisy_by_snr.csv --group_by snr
```

## Step 5: What to Put in Slides

Use these files:

- `outputs/dataset_stats.csv`: dataset/noise benchmark slide
- `outputs/metrics_whisper_noisy_by_snr.csv`: preliminary result table
- `outputs/whisper_noisy.csv`: qualitative examples
- `outputs/midterm_summary.md`: ready-to-copy summary tables and examples
- `docs/experiment_matrix.md`: after-midterm experiment matrix
- `configs/phowhisper_base_lora.yaml`: planned LoRA/MTL setup

Recommended claim:

```text
At midterm, we have completed a reproducible Vietnamese noisy-ASR pipeline,
including manifest construction, controlled SNR noise injection, zero-shot
baseline inference, and Vietnamese-specific metric prototypes. LoRA and
tone-aware MTL training are prepared for Colab execution after midterm.
```

## Step 6: If Time Remains

Run a short Colab training smoke test:

```bash
python train.py --config configs/phowhisper_base_lora.yaml
python evaluate.py --config configs/phowhisper_base_lora.yaml --split test_noisy --out outputs/phowhisper_mtl_noisy.csv
```

Only include trained-model numbers in the report if the run finishes cleanly and the result file exists.
