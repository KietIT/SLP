#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw/musan
cd data/raw/musan
if [ ! -f musan.tar.gz ]; then
  wget -O musan.tar.gz "https://www.openslr.org/resources/17/musan.tar.gz"
fi
if [ ! -d musan ]; then
  tar -xzf musan.tar.gz
fi
cd ../../..
python scripts/make_noise_manifest.py --noise_root data/raw/musan/musan/noise --out data/manifests/noise/musan_noise.jsonl
