#!/usr/bin/env bash
#SBATCH --job-name=stroke2img
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/stroke2img_%j.log
#SBATCH --error=logs/stroke2img_%j.log

# Stroke2Image: baseline + +JIVE for ONE method arm on FLUX.
#
#   input  = coarse stroke painting of an LSUN scene (data_stroke/<dataset>,
#            built by make_stroke_sources.py)
#   output = realistic photos, guided by "a photo of a <scene>"
#   KID reference = REAL photos (data_stroke/<dataset>/reference is a symlink
#            to data/<dataset>/reference)
#
#   METHOD=sdedit     noise painting to sigma(t0), deterministic Euler ODE
#                     back (the official SDEdit procedure adapted to rectified
#                     flow; diversity from the forward draw). Two sweep points:
#                     t0=0.5 (the paper's stroke operating point, t=500/1000)
#                     and t0=0.7 (more noise => freer, more diverse).
#   METHOD=boomerang  same forward, stochastic SDE reverse; same two t0.
#   METHOD=rf_inverse invert the painting once, SDE denoise; two sweep points
#                     at the eta extremes: 0.9 (faithful) and 0.5 (diverse).
#
#   sbatch --partition=<partition> --export=ALL,METHOD=sdedit scripts/run_stroke2image.sh
#   PARTITION=<partition> METHOD=sdedit bash scripts/run_stroke2image.sh  (auto-sbatch)
#
# Or submit all three at once with scripts/submit_stroke2image.sh.
#
# NOTE: single-GPU job on purpose. --n 16 --batch_size 16 produces ONE chunk
# per sampling call, which always lands on pipes[0]; a second GPU would idle
# at 0% and trip the cluster's <50% average-utilisation auto-cancel. Only
# raise --gres to 2 together with --batch_size 8 so chunks fan out.
#
# Safe to resubmit: each arm skips if its results.json exists.
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
# Local checkpoint dir or HF repo id; the arms share one backbone so that
# KID / Vendi stay comparable across methods.
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
