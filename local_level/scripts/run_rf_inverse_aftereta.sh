#!/usr/bin/env bash
#SBATCH --job-name=rfinv_aftereta
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rf_inverse_aftereta_%j.log
#SBATCH --error=logs/rf_inverse_aftereta_%j.log
set -euo pipefail

DATASET="${DATASET:?Set DATASET=classroom|kitchen|conference_room|dining_room|restaurant}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${PARTITION:-}" ]]; then
    echo "Choose a partition, e.g.:"
    echo "  PARTITION=<partition> DATASET=${DATASET} bash $0"
    echo "  sbatch --partition=<partition> --export=ALL,DATASET=${DATASET} $0"
    exit 1
  fi
  echo "Submitting to partition: ${PARTITION} (dataset: ${DATASET})"
  exec sbatch --partition="${PARTITION}" --export=ALL,DATASET="${DATASET}" "$0" "$@"
fi

cd "${JIVE_LOCAL_LEVEL:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(cd .. && pwd)"
mkdir -p logs

if [[ -n "${JIVE_VENV:-}" ]]; then
  source "${JIVE_VENV}/bin/activate"
fi
export FLUX_MODEL="${FLUX_MODEL:-black-forest-labs/FLUX.1-dev}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Host: $(hostname)  dataset=${DATASET}  eta=0.5  baseline+JIVE  inject_after_eta  norms=8,12"
nvidia-smi -L || true

python run_rf_inverse.py \
    --dataset "${DATASET}" \
    --data_root data_stroke \
    --model "${FLUX_MODEL}" \
    --out_dir "out_stroke_aftereta_eta05/${DATASET}/rf_inverse" \
    --n_sources 32 \
    --n 16 \
    --steps 30 \
    --seed 32 \
    --guidance_scale 3.5 \
    --height 256 --width 256 \
    --gamma 0.5 \
    --etas 0.5 \
    --start_timestep 0.0 \
    --stop_timestep 0.25 \
    --inject_norms 8.0 12.0 \
    --jive_iter_mode jtj \
    --perturb_mode additive \
    --inject_after_eta \
    --vjp_chunk 2 \
    --inject_n 4 \
    --power_iters 10 \
    --fd_eps 0.1 \
    --batch_size 16 \
    --fwd_chunk 4 \
    --save_sources 32 \
    --save_samples_per_source 16 \
    --metric_resolution 256

echo "RF-Inverse ${DATASET} eta=0.5 baseline+after-eta JIVE norms=8,12 done."
