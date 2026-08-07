#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_DIR="${ASTROSEARCHLIGHT_DATA_DIR:-$HOME/.astrosearchlight}"

SECTORS=$(seq 1 82)

for SECTOR in $SECTORS
do
    if [[ -f "$DATA_DIR/tess_spoc_$(printf "%04d" "$SECTOR")_resnet.csv" ]]
    then
        echo "*** SKIPPING sector ${SECTOR} (completed)"
        continue
    fi
    echo "*** STARTING sector ${SECTOR}"
    until "$SCRIPT_DIR/classify_tess_cnn.sh" "$SECTOR"
    do
        echo -n "*** Sector {$SECTOR} failed..."
        sleep 60
        echo "Retrying"
    done
    echo "*** FINISHED sector ${SECTOR}"
    rm -rf "$DATA_DIR/cache"/*
done
