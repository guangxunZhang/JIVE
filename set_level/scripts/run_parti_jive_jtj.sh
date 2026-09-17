#!/usr/bin/env bash
#SBATCH --job-name=jive_parti_jtj
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-10%6
#SBATCH --output=logs/jive_parti_jtj_%A_%a.log
#SBATCH --error=logs/jive_parti_jtj_%A_%a.log

# JIVE(J^T J) at inject-norm 12: the same arm as run_parti_base_oscar_jive.sh
# with ONE change,
#
#   iteration operator     J  ->  J^T J     (--jive-iter-mode jtj)
#
# so it isolates the estimator at the norm the main table already reports.
#
# Why J^T J: Q <- QR(J Q) is a block POWER method, which converges to the
# dominant INVARIANT subspace (eigenvectors of J). That equals the singular
# subspace only when J is symmetric, and the endpoint Jacobian
# J = I - sigma*dv/dz is not. Q <- QR(J^T J Q) is the block power method on the
# symmetric PSD J^T J, whose eigenvectors ARE J's right singular vectors and
# whose eigenvalues are the SQUARED singular values -- so it targets the true
# singular subspace and separates the spectrum faster at equal --jive-iters.
# Each iteration costs one batched FD matvec block (W = J Q, no grad) plus one
# batched autograd VJP block (J^T W = W - sigma*(dv/dz)^T W). See
# jive_flux/jive_subspace.py: top_singular_subspace_jtj.
#
# The VJP backward is the memory peak, hence --gres=gpu:2: _common.sh then
# gives the transformer a GPU to itself and raises FWD_CHUNK / VJP_CHUNK, which
# is where the J^T J speedup actually shows.
#
# The deterministic arm is regenerated here so this tree carries its own KID
# reference. Output tree: outputs_jive_jtj_n12/.
#
# Submit all 11 tasks:  sbatch scripts/run_parti_jive_jtj.sh
# Aggregate afterwards: python analysis/aggregate_parti.py --src outputs_jive_jtj_n12

TAG="jive_jtj_n12"
# Under sbatch $0 is a spool copy, so fall back to the submission dir.
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
