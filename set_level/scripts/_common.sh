#!/usr/bin/env bash
set -euo pipefail

ROOT="${JIVE_SET_LEVEL:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}"
mkdir -p logs

if [[ -n "${JIVE_VENV:-}" ]]; then
  source "${JIVE_VENV}/bin/activate"
fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPEC_FILE="${ROOT}/specs/parti_prompts.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

METHOD="jive_flux_schnell_parti"

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [[ "${NGPU}" -ge 3 ]]; then
  DEV_TR="cuda:0"; DEV_VAE="cuda:1"; DEV_CLIP="cuda:2"
  FWD_CHUNK=8; VJP_CHUNK=4
elif [[ "${NGPU}" -ge 2 ]]; then
  DEV_TR="cuda:0"; DEV_VAE="cuda:1"; DEV_CLIP="cuda:1"
  FWD_CHUNK=8; VJP_CHUNK=4
else
  DEV_TR="cuda:0"; DEV_VAE="cuda:0"; DEV_CLIP="cuda:0"
  FWD_CHUNK=4; VJP_CHUNK=2
fi

CHALLENGES=(
  "Basic"
  "Simple Detail"
  "Fine-grained Detail"
  "Style & Format"
  "Imagination"
  "Complex"
  "Quantity"
  "Properties & Positioning"
  "Perspective"
  "Linguistic Structures"
  "Writing & Symbols"
)
CHALLENGE="${CHALLENGES[TASK_ID]}"

SPEC_ONE="${ROOT}/logs/spec_${TAG}_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}.json"
python - <<PY
import json
with open("${SPEC_FILE}") as f:
    spec = json.load(f)
with open("${SPEC_ONE}", "w") as f:
    json.dump({"${CHALLENGE}": spec["${CHALLENGE}"]}, f, indent=2)
PY

echo "Host: $(hostname)  GPUs: ${NGPU}"
echo "Task ${TASK_ID}: challenge='${CHALLENGE}' method=${METHOD}"
nvidia-smi -L || true
