#!/usr/bin/env bash
#SBATCH --job-name=jive_parti_jtj_cads
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-10%6
#SBATCH --output=logs/jive_parti_jtj_cads_%A_%a.log
#SBATCH --error=logs/jive_parti_jtj_cads_%A_%a.log

# JIVE(J^T J)+CADS: run_parti_jive_cads.sh with the subspace estimated by the
# J^T J iteration instead of J. Same norm (4), same CADS schedule
# (tau1=0.6, tau2=1.2, s=0.15, psi=1.0), so the pair of +CADS rows isolates
# the estimator exactly as the pair of JIVE-alone rows does.
#
# --gres=gpu:2 for the VJP backward's memory peak; see run_parti_jive_jtj.sh.
# Output tree: outputs_jive_cads_jtj_n4/.
#
# Submit all 11 tasks:  sbatch scripts/run_parti_jive_jtj_cads.sh
# Aggregate afterwards: python analysis/aggregate_parti.py --src outputs_jive_cads_jtj_n4

TAG="jive_cads_jtj_n4"
# Under sbatch $0 is a spool copy, so fall back to the submission dir.
SCRIPTS_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
[[ -f "${SCRIPTS_DIR}/_common.sh" ]] || SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-$PWD}/scripts"
source "${SCRIPTS_DIR}/_common.sh"

python -m jive_flux \
  --model schnell \
  --spec "${SPEC_ONE}" \
  --method "${METHOD}" \
  --outputs-subdir jive_cads_jtj_n4 \
  --arms deterministic jive jive_cads \
  --height 512 --width 512 \
  --steps 4 \
  --n-images 16 --keep-images-per-arm 16 \
  --G 16 --seeds 42 \
  --inject-norms 4.0 --inject-n 4 \
  --jive-iters 10 \
  --jive-iter-mode jtj \
  --fwd-chunk "${FWD_CHUNK}" \
  --jive-vjp-chunk "${VJP_CHUNK}" \
  --cads-s 0.15 --cads-tau1 0.6 --cads-tau2 1.2 --cads-psi 1.0 \
  --device-transformer "${DEV_TR}" \
  --device-vae "${DEV_VAE}" \
  --device-clip "${DEV_CLIP}" \
  --quality-metrics clip_score clip_iqa hpsv2 \
  --kid \
  --kid-subsets 50 \
  --out-root "${ROOT}" \
  --enable-vae-tiling \
  --skip-existing

rm -f "${SPEC_ONE}"
echo "Done: parti/${CHALLENGE}  $(date -Is)"
