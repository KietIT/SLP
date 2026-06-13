#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw/vivos
cd data/raw/vivos
if [ ! -f vivos.tar.gz ]; then
  wget -O vivos.tar.gz "https://zenodo.org/records/7068130/files/vivos.tar.gz?download=1"
fi
if [ ! -d vivos ]; then
  tar -xzf vivos.tar.gz
fi
echo "VIVOS ready at data/raw/vivos"
