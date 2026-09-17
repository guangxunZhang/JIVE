#!/usr/bin/env bash
# Shared prelude for every set-level launcher in this directory. Sourced, not
# executed. Keeping it in one place is what guarantees the five table rows are
# generated under identical conditions: same prompts, same per-task challenge
# slicing, same environment.
#
# Sets up, in order:
#   ROOT            set_level/
#   the python env, HF cache and allocator flags
#   DEV_TR/DEV_VAE/DEV_CLIP + FWD_CHUNK/VJP_CHUNK from the GPUs Slurm granted
#   CHALLENGE       this array task's PartiPrompts aspect
#   SPEC_ONE        a one-aspect spec JSON, written to logs/ and removed by
#                   the caller when it finishes
#
# The caller supplies TAG (used in the temp spec filename) before sourcing.
#
# Environment knobs (all optional):
#   JIVE_SET_LEVEL  set_level/ path. Defaults to the submission directory, so
#                   submitting from set_level/ needs nothing.
#   JIVE_VENV       python virtualenv to activate.
#   FLUX_ROOT       directory holding FLUX.1-dev/ and FLUX.1-schnell/. Unset
#                   means pull the weights from HF by repo id.
#   FDEVAL_ROOT     fdeval scoring harness, needed for feature Vendi.
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

# Every run writes under this one method name; the trees are kept apart by
# --outputs-subdir instead. analysis/aggregate_parti.py globs for this prefix.
METHOD="jive_flux_schnell_parti"

# Place the three hot modules on separate GPUs when Slurm granted enough; fall
# back gracefully so the script still runs interactively on a single GPU. The
# J^T J VJP backward is the memory peak, which is why giving the transformer a
# GPU to itself is what lets FWD_CHUNK/VJP_CHUNK go up.
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [[ "${NGPU}" -ge 3 ]]; then
  # cuda:0 = transformer (+ VJP peak); cuda:1 = VAE; cuda:2 = CLIP/metrics.
  # Extra GPUs (3+) sit idle -- the pipeline only has three device slots.
  DEV_TR="cuda:0"; DEV_VAE="cuda:1"; DEV_CLIP="cuda:2"
  FWD_CHUNK=8; VJP_CHUNK=4
elif [[ "${NGPU}" -ge 2 ]]; then
  DEV_TR="cuda:0"; DEV_VAE="cuda:1"; DEV_CLIP="cuda:1"
  FWD_CHUNK=8; VJP_CHUNK=4
else
  DEV_TR="cuda:0"; DEV_VAE="cuda:0"; DEV_CLIP="cuda:0"
  FWD_CHUNK=4; VJP_CHUNK=2
fi

# Ordered challenge groups -- must match the insertion order of the spec JSON
# (difficulty order: Standard, Intermediate, then the 7 Challenging aspects).
# Array index = one aspect; prompt counts in parentheses, 1632 total.
CHALLENGES=(
  "Basic"                     #   0  272
  "Simple Detail"             #   1  232
  "Fine-grained Detail"       #   2  312
  "Style & Format"            #   3  206
  "Imagination"               #   4  150
  "Complex"                   #   5  113
  "Quantity"                  #   6   90
  "Properties & Positioning"  #   7   35
  "Perspective"               #   8   70
  "Linguistic Structures"     #   9   61
  "Writing & Symbols"         #  10   91
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
