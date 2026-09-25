#!/usr/bin/env bash
#SBATCH --job-name=stroke2img
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/stroke2img_%j.log
#SBATCH --error=logs/stroke2img_%j.log

set -euo pipefail

METHOD="${METHOD:?Set METHOD=sdedit, boomerang or rf_inverse}"
DATASET="${DATASET:-classroom}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${PARTITION:-}" ]]; then
    echo "Choose a partition, e.g.:"
    echo "  PARTITION=<partition> METHOD=${METHOD} bash $0"
    echo "  sbatch --partition=<partition> --export=ALL,METHOD=${METHOD} $0"
    exit 1
  fi
  echo "Submitting to partition: ${PARTITION} (method: ${METHOD}, dataset: ${DATASET})"
  exec sbatch --partition="${PARTITION}" --export=ALL,METHOD="${METHOD}",DATASET="${DATASET}" "$0" "$@"
fi

cd "${JIVE_LOCAL_LEVEL:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(cd .. && pwd)"
mkdir -p logs

if [[ -n "${JIVE_VENV:-}" ]]; then
  source "${JIVE_VENV}/bin/activate"
fi
export FLUX_MODEL="${FLUX_MODEL:-black-forest-labs/FLUX.1-schnell}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Host: $(hostname) | Partition: ${SLURM_JOB_PARTITION:-unknown} | Method: ${METHOD} | Dataset: ${DATASET}"
echo "FLUX_MODEL=${FLUX_MODEL}"
nvidia-smi -L || true

SHARED_ARGS=(
    --dataset "${DATASET}"
    --data_root data_stroke
    --out_dir "out_stroke/${DATASET}/${METHOD}"
    --n_sources 32
    --n 16
    --steps 4
    --seed 32
    --inject_norms 4 8 12
    --inject_n 4
    --height 256 --width 256
    --batch_size 16
    --fwd_chunk 4
    --save_sources 8
    --save_samples_per_source 8
)

case "${METHOD}" in
  sdedit|boomerang)
    python run_flux_arm.py \
        --method "${METHOD}" \
        --strengths 0.5 0.7 \
        "${SHARED_ARGS[@]}"
    ;;
  rf_inverse)
    python run_rf_inverse.py \
        --etas 0.9 0.5 \
        --gamma 0.5 \
        --stop_timestep 0.25 \
        "${SHARED_ARGS[@]}"
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 1
    ;;
esac

echo "Stroke2Image ${METHOD} (${DATASET}) done."
