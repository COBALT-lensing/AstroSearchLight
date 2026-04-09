#!/bin/bash -e

SECTORS=$(seq 1 82)

for SECTOR in $SECTORS
do
    if [[ -f "data/tess_spoc_$(printf "%04d" "$SECTOR")_resnet.csv" ]]
    then
        echo "*** SKIPPING sector ${SECTOR} (completed)"
        continue
    fi
    echo "*** STARTING sector ${SECTOR}"
    until ./classify_tess_cnn.sh $SECTOR
    do
        echo -n "*** Sector {$SECTOR} failed..."
        sleep 60
        echo "Retrying"
    done
    echo "*** FINISHED sector ${SECTOR}"
    rm -rf data/cache/*
done