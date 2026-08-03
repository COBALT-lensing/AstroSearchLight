#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

SECTORS=$(seq 1 82)

for SECTOR in $SECTORS
do
    if [[ -f "$ROOT_DIR/data/tess_spoc_$(printf "%04d" "$SECTOR")_resnet.csv" ]]
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
    rm -rf "$ROOT_DIR/data/cache"/*
done
