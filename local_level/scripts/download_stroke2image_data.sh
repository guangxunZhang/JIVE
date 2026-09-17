#!/usr/bin/env bash
#SBATCH --job-name=dl_stroke2img
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/dl_stroke2img_%j.log
#SBATCH --error=logs/dl_stroke2img_%j.log

# Downloads / builds EVERYTHING the Stroke2Image comparison needs, on a CPU
# node (no GPU wasted):
#   1. LSUN photos   : data/<dataset>/{reference,sources} via HF streaming
#                      (2000 real reference + 32 source photos, 256x256)
#   2. Stroke inputs : data_stroke/<dataset>/sources  (dab paintings of the
#                      32 sources, make_stroke_sources.py)
#   3. KID reference : data_stroke/<dataset>/reference -> symlink to the real
#                      photos in data/<dataset>/reference
#   4. Metric models : DINO / CLIP / torchvision Inception-v3 / LPIPS cached
#                      into $HF_HOME so GPU jobs never touch the network
#
# The FLUX.1-dev checkpoint is NOT downloaded here — point $FLUX_MODEL at it.
#
# Usage (cluster):   sbatch scripts/download_stroke2image_data.sh
# Usage (one only):   DATASET=church bash scripts/download_stroke2image_data.sh
#                    DATASET=bedroom bash scripts/download_stroke2image_data.sh
# Idempotent: every step skips work that is already done.
set -euo pipefail

N_SOURCES=32
N_REFERENCE=2000
RESOLUTION=256

if [[ -n "${DATASET:-}" ]]; then
    DATASETS=("${DATASET}")
else
    DATASETS=(church bedroom)
fi

# sbatch copies the script into /opt/slurm/...; $0 is not the project dir.
cd "${JIVE_LOCAL_LEVEL:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs

# Env: prefer the cluster env when present; otherwise assume the caller's
# python already has torch/diffusers/transformers/datasets/pillow/scikit-image.
if [[ -n "${JIVE_VENV:-}" ]]; then
    source "${JIVE_VENV}/bin/activate"
fi
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Datasets: ${DATASETS[*]}"

prepare_dataset() {
    local dataset="$1"
    echo
    echo "======== ${dataset} ========"

    # NOTE: the `datasets` streaming client can crash on interpreter shutdown
    # (PyGILState_Release on a dead thread) AFTER all exports completed. The
    # export writes its files before that point, so tolerate a post-export crash
    # and verify counts below.
    python common/prepare_data.py \
        --dataset "${dataset}" \
        --out_root data \
        --n_sources "${N_SOURCES}" \
        --n_reference "${N_REFERENCE}" \
        --resolution "${RESOLUTION}" \
        || echo "WARNING: prepare_data exited non-zero (may be a shutdown crash)"

    python make_stroke_sources.py --dataset "${dataset}" --style dabs

    mkdir -p "data_stroke/${dataset}"
    ln -sfn "$(pwd)/data/${dataset}/reference" "data_stroke/${dataset}/reference"
    echo "Linked data_stroke/${dataset}/reference -> data/${dataset}/reference"

    local n_src n_ref n_stroke
    n_src=$(ls "data/${dataset}/sources" 2>/dev/null | wc -l)
    n_ref=$(ls "data/${dataset}/reference" 2>/dev/null | wc -l)
    n_stroke=$(ls "data_stroke/${dataset}/sources" 2>/dev/null | wc -l)
    echo "${dataset}: ${n_src} photos, ${n_ref} reference, ${n_stroke} paintings"

    local ok=1
    [[ "${n_src}" -ge "${N_SOURCES}" ]] || { echo "ERROR: incomplete sources"; ok=0; }
    [[ "${n_ref}" -ge "${N_REFERENCE}" ]] || { echo "ERROR: incomplete reference"; ok=0; }
    [[ "${n_stroke}" -ge "${N_SOURCES}" ]] || { echo "ERROR: incomplete paintings"; ok=0; }
    [[ -e "data_stroke/${dataset}/reference" ]] || { echo "ERROR: missing reference symlink"; ok=0; }
    [[ "${ok}" -eq 1 ]] || return 1
}

for dataset in "${DATASETS[@]}"; do
    prepare_dataset "${dataset}"
done

# -------- metric models (once, shared across datasets) --------------------
python - <<'PY'
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
AutoImageProcessor.from_pretrained("facebook/dino-vits16")
AutoModel.from_pretrained("facebook/dino-vits16")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
from torchvision.models import inception_v3, Inception_V3_Weights
inception_v3(weights=Inception_V3_Weights.DEFAULT)
try:
    import lpips
    lpips.LPIPS(net="alex")
    print("LPIPS cached.")
except ImportError:
    print("lpips not installed - lpips_div will be skipped "
          "(pip install lpips to enable).")
try:
    import skimage  # noqa: F401
    print(f"scikit-image {skimage.__version__} present (ssim_div enabled).")
except ImportError:
    print("scikit-image not installed - ssim_div will be skipped "
          "(pip install scikit-image to enable).")
print("Metric models cached.")
PY

echo
echo "Done. Next: bash submit_stroke2image.sh <partition>"
