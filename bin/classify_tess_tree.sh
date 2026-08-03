#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

SECTOR=$(printf "%04d" "$1")

SPOC_URL="https://archive.stsci.edu/hlsps/tess-spoc/download_scripts/hlsp_tess-spoc_tess_phot_s${SECTOR}_tess_v1_dl-lc.sh"

FITS_URLS=$(curl "$SPOC_URL" | awk '{print$8}' | sed "s/'//g")

echo "$FITS_URLS" | poetry run python "$ROOT_DIR/bin/classify.py" -o "$ROOT_DIR/data/tess_spoc_${SECTOR}_xgboost.csv" xgboost -
