#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/predict.sh IMAGE_OR_DIR [OUTPUT_DIR] [UM_PER_PIXEL]" >&2
  exit 2
fi
input="$1"
output="${2:-results}"
if [[ $# -ge 3 ]]; then
  facetquant predict --input "$input" --weights weights/facetquant_fp16.pth --out "$output" --max-size 768 --um-per-pixel "$3"
else
  facetquant predict --input "$input" --weights weights/facetquant_fp16.pth --out "$output" --max-size 768
fi
