#!/usr/bin/env bash
# Submits the whole local-level comparison from the LOGIN node:
#   6 GPU jobs = {church, bedroom} x {sdedit, boomerang, rf_inverse}
#   on the stroke paintings in data_stroke/<dataset>, each job covering BOTH
#   its baseline arm and its +JIVE arm.
#
# Prereqs (CPU, done once):
#   sbatch scripts/download_stroke2image_data.sh
#
# Usage:
#   bash scripts/submit_stroke2image.sh <partition>
set -euo pipefail

PARTITION="${1:?Usage: bash scripts/submit_stroke2image.sh <partition>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$HERE")"   # local_level/, where data_stroke/ lives

DATASETS=(church bedroom)
METHODS=(sdedit boomerang rf_inverse)

for DATASET in "${DATASETS[@]}"; do
    if [[ ! -d "data_stroke/${DATASET}/sources" || ! -e "data_stroke/${DATASET}/reference" ]]; then
        echo "ERROR: data_stroke/${DATASET} is incomplete. Run:" >&2
        echo "  sbatch scripts/download_stroke2image_data.sh" >&2
        exit 1
    fi
done

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        JID=$(sbatch --parsable --partition="${PARTITION}" \
              --export=ALL,METHOD="${METHOD}",DATASET="${DATASET}" \
              "${HERE}/run_stroke2image.sh")
        echo "Submitted stroke2image ${METHOD} (${DATASET}) as job ${JID}"
    done
done

echo
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results land in out_stroke/<dataset>/<method>/results.json"
echo "Aggregate with:  python aggregate_results.py --out_root out_stroke"
