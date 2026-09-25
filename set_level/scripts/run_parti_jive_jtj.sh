#!/usr/bin/env bash
#SBATCH --job-name=jive_parti_jtj
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-10%6
#SBATCH --output=logs/jive_parti_jtj_%A_%a.log
#SBATCH --error=logs/jive_parti_jtj_%A_%a.log


TAG="jive_jtj_n12"
SCRIPTS_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
[[ -f "${SCRIPTS_DIR}/_common.sh" ]] || SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-$PWD}/scripts"
source "${SCRIPTS_DIR}/_common.sh"

python -m jive_flux \
  --model schnell \
  --spec "${SPEC_ONE}" \
  --method "${METHOD}" \
  --outputs-subdir jive_jtj_n12 \
  --arms deterministic jive \
  --height 512 --width 512 \
  --steps 4 \
  --n-images 16 --keep-images-per-arm 16 \
  --G 16 --seeds 42 \
  --inject-norms 12.0 --inject-n 4 \
  --jive-iters 10 \
  --jive-iter-mode jtj \
  --fwd-chunk "${FWD_CHUNK}" \
  --jive-vjp-chunk "${VJP_CHUNK}" \
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
