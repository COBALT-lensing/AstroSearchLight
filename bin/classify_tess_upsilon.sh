#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_DIR="${ASTROSEARCHLIGHT_DATA_DIR:-$HOME/.astrosearchlight}"

SECTOR=$(printf "%04d" "$1")

SPOC_URL="https://archive.stsci.edu/hlsps/tess-spoc/download_scripts/hlsp_tess-spoc_tess_phot_s${SECTOR}_tess_v1_dl-lc.sh"

FITS_URLS=$(curl "$SPOC_URL" | awk '{print$8}' | sed "s/'//g")

echo "$FITS_URLS" | uv run python "$ROOT_DIR/bin/classify.py" -o "$DATA_DIR/tess_spoc_${SECTOR}_upsilon.csv" upsilon -
