#!/usr/bin/env bash
#SBATCH --job-name=cads_parti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-10%5
#SBATCH --output=logs/cads_parti_%A_%a.log
#SBATCH --error=logs/cads_parti_%A_%a.log

# The standalone CADS baseline (Sadat et al., ICLR 2024) via
# baselines/cads.py, run under the SAME setup as the other arms so its row
# sits beside them: same prompts (all 11 challenge aspects, 1632 prompts),
# same base seed 42, 16 images per prompt, 512x512, 4-step schnell,
# guidance 0.0, same local FLUX.1-schnell weights, same quality metrics.
#
# Two schedules per task (tau1=0.6, psi=1.0, s=0.15 throughout):
#   tau2=1.2  ->  arm cads_t2-1.2_n0.15   (gamma at 4 steps: .33/.58/.91/1)
#   tau2=1.8  ->  arm cads_t2-1.8_n0.15   (gamma at 4 steps: .67/.79/.96/1)
# Both keep gamma(1) > 0, so unlike the paper default (tau2=0.9) NO denoising
# step is fully unconditional on the 4-step grid.
#
# SEEDS: cads.py derives per-image seeds as seed + crc32(uid) + i (its own
# pairing convention), NOT jive_flux's seed + i, so individual images are not
# noise-paired with the other arms; the base seed and every other setting
# match. The unmodified-sampler row is not regenerated here -- the
# deterministic arm of run_parti_base_oscar_jive.sh already covers it.
#
# Output (consumable by the fdeval scoring harness):
#   outputs_cads/schnell_parti_seed42/  images|features / cads_t2-*/ <uid>
# All 11 tasks share this one run dir; uids are unique per aspect and cads.py
# skips groups whose images+npz already exist, so re-submitting resumes.
#
# In-process scoring needs the external fdeval harness ($FDEVAL_ROOT);
# without it this still generates the images, to be scored separately.
#
# Submit all 11 tasks (from set_level/):
#   sbatch --partition=<part> --account=<acct> scripts/run_parti_cads.sh

TAG="cads"
# Under sbatch $0 is a spool copy, so fall back to the submission dir.
SCRIPTS_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
[[ -f "${SCRIPTS_DIR}/_common.sh" ]] || SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-$PWD}/scripts"
source "${SCRIPTS_DIR}/_common.sh"
rm -f "${SPEC_ONE}"   # cads.py takes its own prompt-set format, built below

REPO_ROOT="$(dirname "${ROOT}")"
SCHNELL="${FLUX_ROOT:+${FLUX_ROOT}/FLUX.1-schnell}"
SCHNELL="${SCHNELL:-black-forest-labs/FLUX.1-schnell}"

# cads.py --score imports fdeval's ScorerBank (its only dependency on that
# harness). Without FDEVAL_ROOT, generate here and score separately.
SCORE_ARGS=()
if [[ -n "${FDEVAL_ROOT:-}" ]]; then
  export PYTHONPATH="${FDEVAL_ROOT}:${PYTHONPATH:-}"
  SCORE_ARGS=(--score --metrics clip,clipiqa,hpsv2
              --cache_dir "${FDEVAL_ROOT}/eval_cache"
              --skip_unavailable_metrics)
else
  echo "FDEVAL_ROOT unset -> generating images only, no in-process scoring"
fi

OUT_DIR="${ROOT}/outputs_cads/schnell_parti_seed42"

# cads.py takes an fdeval-style prompt_set ([{"uid":..., "text":...}]), so
# convert this task's challenge slice of the spec. UIDs carry the slugified
# challenge name, which is what lets all 11 tasks share one OUT_DIR safely.
PROMPT_SET="${ROOT}/logs/cads_prompts_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}.json"
python - <<PY
import json, re
challenge = "${CHALLENGE}"
slug = re.sub(r"[^A-Za-z0-9]+", "_", challenge).strip("_")
with open("${SPEC_FILE}") as f:
    spec = json.load(f)
prompts = [{"uid": f"parti_{slug}_{i:03d}", "text": t}
           for i, t in enumerate(spec[challenge])]
with open("${PROMPT_SET}", "w") as f:
    json.dump(prompts, f, indent=2)
print(f"{challenge}: {len(prompts)} prompts")
PY

for TAU2 in 1.2 1.8; do
  echo "=== CADS tau2=${TAU2} s=0.15 (tau1=0.6 psi=1.0) ===  $(date -Is)"
  python "${REPO_ROOT}/baselines/cads.py" \
    --prompt_set "${PROMPT_SET}" \
    --out_dir "${OUT_DIR}" \
    --model_id "${SCHNELL}" \
    --height 512 --width 512 --steps 4 \
    --num_images 16 --save_images_per_group 16 \
    --seed 42 --guidance_scale 0.0 \
    --cads_tau2 ${TAU2} --cads_s 0.15 \
    --device cuda:0 --dtype bf16 \
    "${SCORE_ARGS[@]}"
done

rm -f "${PROMPT_SET}"
echo "Done: parti/${CHALLENGE}  $(date -Is)"
