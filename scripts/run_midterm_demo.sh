#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openai/whisper-base}"
CLEAN_LIMIT="${CLEAN_LIMIT:-30}"
NOISY_CLEAN_LIMIT="${NOISY_CLEAN_LIMIT:-50}"
NOISY_INFER_LIMIT="${NOISY_INFER_LIMIT:-120}"
USE_MUSAN="${USE_MUSAN:-0}"

mkdir -p data/raw data/manifests outputs

echo "[1/6] Downloading/preparing VIVOS"
bash scripts/download_vivos.sh
python scripts/make_vivos_manifest.py --vivos_root data/raw/vivos --out_dir data/manifests/vivos

if [[ "${USE_MUSAN}" == "1" ]]; then
  echo "[2/6] Downloading/preparing MUSAN noise"
  bash scripts/download_musan.sh
  NOISE_MANIFEST="data/manifests/noise/musan_noise.jsonl"
else
  echo "[2/6] Creating small demo noise set"
  python scripts/make_demo_noise.py --out_dir data/raw/demo_noise --seconds 60 --seed 42
  python scripts/make_noise_manifest.py --noise_root data/raw/demo_noise --out data/manifests/noise/demo_noise.jsonl
  NOISE_MANIFEST="data/manifests/noise/demo_noise.jsonl"
fi

echo "[3/6] Creating fixed noisy test subset"
python scripts/make_noisy_test.py \
  --manifest data/manifests/vivos/test.jsonl \
  --noise_manifest "${NOISE_MANIFEST}" \
  --out_manifest data/manifests/vivos/test_noisy.jsonl \
  --limit "${NOISY_CLEAN_LIMIT}" \
  --snrs 20 10 5 0 \
  --seed 42

echo "[4/6] Writing dataset stats"
python scripts/dataset_stats.py \
  data/manifests/vivos/test.jsonl \
  data/manifests/vivos/test_noisy.jsonl \
  --out outputs/dataset_stats.csv

MODEL_TAG="$(echo "${MODEL}" | tr '/:' '__')"

echo "[5/6] Running baseline inference with ${MODEL}"
python scripts/infer.py \
  --manifest data/manifests/vivos/test.jsonl \
  --model "${MODEL}" \
  --out "outputs/${MODEL_TAG}_clean.csv" \
  --limit "${CLEAN_LIMIT}"
python scripts/infer.py \
  --manifest data/manifests/vivos/test_noisy.jsonl \
  --model "${MODEL}" \
  --out "outputs/${MODEL_TAG}_noisy.csv" \
  --limit "${NOISY_INFER_LIMIT}"

echo "[6/6] Scoring predictions"
python scripts/score_predictions.py \
  --pred "outputs/${MODEL_TAG}_clean.csv" \
  --out "outputs/metrics_${MODEL_TAG}_clean.csv"
python scripts/score_predictions.py \
  --pred "outputs/${MODEL_TAG}_noisy.csv" \
  --out "outputs/metrics_${MODEL_TAG}_noisy_by_snr.csv" \
  --group_by snr

python scripts/build_midterm_summary.py \
  --outputs_dir outputs \
  --model_tag "${MODEL_TAG}" \
  --out outputs/midterm_summary.md

python scripts/validate_midterm_outputs.py \
  --outputs_dir outputs \
  --model_tag "${MODEL_TAG}" \
  --min_clean_predictions 1 \
  --min_noisy_predictions 1

echo "Done. Use these files in the report/slides:"
echo "  outputs/dataset_stats.csv"
echo "  outputs/${MODEL_TAG}_clean.csv"
echo "  outputs/${MODEL_TAG}_noisy.csv"
echo "  outputs/metrics_${MODEL_TAG}_clean.csv"
echo "  outputs/metrics_${MODEL_TAG}_noisy_by_snr.csv"
echo "  outputs/midterm_summary.md"
