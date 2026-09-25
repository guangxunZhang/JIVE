#!/usr/bin/env bash
#SBATCH --job-name=jive_parti_base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-10%5
#SBATCH --output=logs/jive_parti_base_%A_%a.log
#SBATCH --error=logs/jive_parti_base_%A_%a.log

# Rows 1-3 of the set-level table: UNMODIFIED SAMPLER, OSCAR and JIVE(J), all
# three from identical starting latents in one pass so the comparison is exact
# and the KID reference is the run's own deterministic arm.
#
# JIVE here is the published configuration: inject-norm 12, rank 4, 10 subspace
# iterations, subspace found by the J iteration (--jive-iter-mode j, the
# default). The J^T J variant is a separate launcher.
#
# OSCAR perturbs at ALL denoising steps (--t-gate 0.0,1.0 --sched-shape const);
# its default 0.85,0.95 gate never triggers on schnell's 4-step grid.
#
# One array task per PartiPrompts challenge aspect (11 tasks, 1632 prompts
# total), 16 images per prompt per arm. Output tree: outputs/.
# --skip-existing resumes after a QoS cancel without redoing finished prompts.
#
# Submit all 11 tasks:  sbatch scripts/run_parti_base_oscar_jive.sh
# Submit one aspect:    sbatch --array=0 scripts/run_parti_base_oscar_jive.sh
# Aggregate afterwards: python analysis/aggregate_parti.py

TAG="base_oscar_jive"
# Under sbatch $0 is a spool copy, so fall back to the submission dir.
SCRIPTS_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
[[ -f "${SCRIPTS_DIR}/_common.sh" ]] || SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-$PWD}/scripts"
source "${SCRIPTS_DIR}/_common.sh"

python -m jive_flux \
  --model schnell \
  --spec "${SPEC_ONE}" \
  --method "${METHOD}" \
  --arms deterministic oscar jive \
  --height 512 --width 512 \
  --steps 4 \
  --n-images 16 --keep-images-per-arm 16 \
  --G 16 --seeds 42 \
  --inject-norms 12.0 --inject-n 4 \
  --jive-iters 10 \
  --t-gate 0.0,1.0 --sched-shape const \
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
