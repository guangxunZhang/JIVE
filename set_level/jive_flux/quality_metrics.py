"""Quality scorers for the FLUX comparison package.

Reuses the SD3.5 builders where they work; overrides clip_iqa with a
standalone open_clip implementation that follows the CLIP-IQA algorithm
(Wang et al., AAAI 2023) using the same multi-prompt pairs as pyiqa, but
without importing pyiqa (which currently breaks on NumPy 2.x via imgaug).
"""
import torch
import torch.nn.functional as F

from core.quality_metrics import (
    BUILDERS as _BASE_BUILDERS,
    PROMPT_AWARE_METRICS,
    build_clip_scorer,
    build_brisque_scorer,
    build_image_reward_scorer,
)

_CLIP_IQA_PROMPTS = [
    "Good image", "bad image",
    "Sharp image", "blurry image",
    "sharp edges", "blurry edges",
    "High resolution image", "low resolution image",
    "Noise-free image", "noisy image",
]


def build_clip_iqa_scorer(device="cpu", batch_size=32):
    """score_fn(images NCHW in [0,1]) -> mean CLIP-IQA (0-1, higher=better).

    Implements the standard CLIP-IQA recipe: for each good/bad prompt pair,
    softmax the two CLIP logits and take the 'good' class probability; return
    the mean over all pairs. Uses open_clip ViT-B/32 (openai weights).

    The logits MUST be scaled by CLIP's learned temperature
    (logit_scale.exp(), ~100 for ViT-B/32) exactly as CLIP's forward() does:
    raw cosine similarities are ~0.2 with good/bad differences of ~0.01, so
    without the scale every pair's softmax sits at ~0.5 for every image and
    the metric carries no signal (both pyiqa's CLIPIQA and the official
    IceClear/CLIP-IQA repo score through the full model forward, which
    applies this scale).
    """
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model = model.to(device).eval()
    except Exception as exc:
        print(f"  [clip_iqa] open_clip unavailable ({type(exc).__name__}: {exc}); skipping")
        return None

    logit_scale = model.logit_scale.exp().item()

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        text_tokens = tokenizer(_CLIP_IQA_PROMPTS).to(device)
        text_feats = model.encode_text(text_tokens).float()
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True).clamp_min(1e-12)

    n_pairs = len(_CLIP_IQA_PROMPTS) // 2

    @torch.no_grad()
    def score(images):
        pair_probs = []
        for chunk in images.split(batch_size):
            x = chunk.to(device).float()
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(224, 224), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
            ifeat = model.encode_image((x - mean) / std).float()
            ifeat = ifeat / ifeat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            logits = logit_scale * (ifeat @ text_feats.T)
            probs = logits.reshape(logits.size(0), n_pairs, 2).softmax(dim=-1)
            pair_probs.append(probs[..., 0].mean(dim=1).cpu())
        return float(torch.cat(pair_probs).mean().item())

    print("  CLIP-IQA: open_clip ViT-B/32 (standalone, CLIP-IQA prompt pairs)")
    return score


def build_hpsv2_scorer(device="cpu", batch_size=32):
    """score_fn(images NCHW in [0,1], prompt str) -> mean HPSv2.1 score
    (human-preference score v2.1, Wu et al. 2023; unbounded-ish, typical
    range ~0.2-0.35, HIGHER is better). A more modern cross-check for
    image_reward (which is 2023-era and somewhat OOD for FLUX outputs).
    Requires the `hpsv2` package; downloads the ~1.6GB HPS_v2.1_compressed
    checkpoint from HF hub (xswu/HPSv2) on first use. `batch_size` is
    accepted for builder-signature uniformity; the package scores images
    one at a time internally."""
    try:
        import hpsv2
        from hpsv2 import img_score as _hps_img_score
        from torchvision.transforms.functional import to_pil_image
        _hps_img_score.device = str(device)
    except Exception as exc:
        print(f"  [hpsv2] unavailable ({type(exc).__name__}: {exc}); skipping")
        return None

    @torch.no_grad()
    def score(images, prompt):
        pil_images = [to_pil_image(img.float().clamp(0, 1)) for img in images]
        rewards = hpsv2.score(pil_images, prompt, hps_version="v2.1")
        score.last_per_image = [float(r) for r in rewards]
        return float(sum(rewards) / len(rewards))

    score.last_per_image = None

    print("  HPSv2.1: xswu/HPSv2 ViT-H-14 (human preference v2.1, higher=better)")
    return score


BUILDERS = {
    **_BASE_BUILDERS,
    "clip_iqa": build_clip_iqa_scorer,
    "hpsv2": build_hpsv2_scorer,
}

__all__ = ["BUILDERS", "PROMPT_AWARE_METRICS"]
