#!/usr/bin/env bash
#SBATCH --job-name=cads_parti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-10%5
#SBATCH --output=logs/cads_parti_%A_%a.log
#SBATCH --error=logs/cads_parti_%A_%a.log


TAG="cads"
SCRIPTS_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
[[ -f "${SCRIPTS_DIR}/_common.sh" ]] || SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-$PWD}/scripts"
source "${SCRIPTS_DIR}/_common.sh"
rm -f "${SPEC_ONE}"

REPO_ROOT="$(dirname "${ROOT}")"
SCHNELL="${FLUX_ROOT:+${FLUX_ROOT}/FLUX.1-schnell}"
SCHNELL="${SCHNELL:-black-forest-labs/FLUX.1-schnell}"

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
