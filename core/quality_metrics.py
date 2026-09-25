"""Fidelity / no-reference quality scorers, reported per arm alongside the
diversity metrics so "more diverse" is never read without a quality axis.

Each builder loads its model once, resident on `device` for the whole run
(same pattern as vendi.build_image_embedder), and returns a callable -- or
None if the backing package isn't installed, in which case the metric is
simply omitted from the results rather than aborting the run.

Memory note: these run on the SAME device as the Vendi feature embedder and
(for the OSCAR arm) the volume objective's CLIP, all resident for the whole
run alongside the SD3.5 pipeline. If GPU memory is tight, trim
--quality-metrics to just what you need (clip_score is cheapest; image_reward
is the heaviest at ~1.5GB) or point --device-clip at a second GPU / cpu.
"""
import torch
import torch.nn.functional as F

PROMPT_AWARE_METRICS = {"clip_score", "image_reward", "hpsv2"}


def build_clip_scorer(device="cpu", batch_size=32):
    """score_fn(images NCHW in [0,1], prompt str) -> mean CLIPScore (100 x
    cosine similarity between image and prompt embeddings). Prompt fidelity:
    catches diversity gains that come from drifting off-prompt."""
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model = model.to(device).eval()
    except Exception as exc:
        print(f"  [clip_score] open_clip unavailable ({type(exc).__name__}: {exc}); skipping")
        return None

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def score(images, prompt):
        tfeat = model.encode_text(tokenizer([prompt]).to(device)).float()
        tfeat = tfeat / tfeat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sims = []
        for chunk in images.split(batch_size):
            x = chunk.to(device).float()
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(224, 224), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
            ifeat = model.encode_image((x - mean) / std).float()
            ifeat = ifeat / ifeat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            sims.append((ifeat @ tfeat.T).squeeze(1).cpu())
        return float(torch.cat(sims).mean().item() * 100.0)

    print("  CLIPScore: open_clip ViT-B/32 (openai)")
    return score


def build_brisque_scorer(device="cpu", batch_size=32):
    """score_fn(images NCHW in [0,1]) -> mean BRISQUE (no-reference natural
    image quality; LOWER is better, independent of the prompt). Requires
    `pyiqa` (pip install pyiqa)."""
    try:
        import pyiqa
        metric = pyiqa.create_metric("brisque", device=torch.device(device))
    except Exception as exc:
        print(f"  [brisque] pyiqa unavailable ({type(exc).__name__}: {exc}); skipping")
        return None

    @torch.no_grad()
    def score(images):
        vals = [metric(chunk.to(device).float().clamp(0, 1)).float().reshape(-1).cpu()
                for chunk in images.split(batch_size)]
        return float(torch.cat(vals).mean().item())

    print("  BRISQUE: pyiqa (no-reference quality, lower=better)")
    return score


def build_clip_iqa_scorer(device="cpu", batch_size=32):
    """score_fn(images NCHW in [0,1]) -> mean CLIP-IQA (no-reference
    perceived quality; 0-1, HIGHER is better, independent of the prompt).
    Requires `pyiqa`."""
    try:
        import pyiqa
        metric = pyiqa.create_metric("clipiqa", device=torch.device(device))
    except Exception as exc:
        print(f"  [clip_iqa] pyiqa unavailable ({type(exc).__name__}: {exc}); skipping")
        return None

    @torch.no_grad()
    def score(images):
        vals = [metric(chunk.to(device).float().clamp(0, 1)).float().reshape(-1).cpu()
                for chunk in images.split(batch_size)]
        return float(torch.cat(vals).mean().item())

    print("  CLIP-IQA: pyiqa (no-reference quality, higher=better)")
    return score


def _patch_transformers_for_image_reward():
    """image-reward's vendored BLIP (ImageReward/models/BLIP/med.py) targets
    transformers ~4.15 and imports apply_chunking_to_forward,
    find_pruneable_heads_and_indices and prune_linear_layer from
    transformers.modeling_utils; transformers >=4.49 moved all three to
    transformers.pytorch_utils, so the import fails there. Re-export the
    moved names before importing ImageReward. No-op where they already exist."""
    try:
        import transformers.modeling_utils as mu
        import transformers.pytorch_utils as pu
        for name in ("apply_chunking_to_forward",
                     "find_pruneable_heads_and_indices",
                     "prune_linear_layer"):
            if not hasattr(mu, name) and hasattr(pu, name):
                setattr(mu, name, getattr(pu, name))
    except Exception:
        pass


def build_image_reward_scorer(device="cpu", batch_size=32):
    """score_fn(images NCHW in [0,1], prompt str) -> mean ImageReward score
    (human-preference-aligned prompt fidelity + quality; unbounded, HIGHER is
    better). Requires the `image-reward` package and downloads a ~1.5GB
    checkpoint on first use."""
    try:
        _patch_transformers_for_image_reward()
        import ImageReward as RM
        from torchvision.transforms.functional import to_pil_image
        model = RM.load("ImageReward-v1.0", device=str(device))
    except Exception as exc:
        print(f"  [image_reward] unavailable ({type(exc).__name__}: {exc}); skipping")
        return None

    @torch.no_grad()
    def score(images, prompt):
        pil_images = [to_pil_image(img.float().clamp(0, 1)) for img in images]
        rewards = model.score(prompt, pil_images)
        if isinstance(rewards, (int, float)):
            rewards = [rewards]
        score.last_per_image = [float(r) for r in rewards]
        return float(sum(rewards) / len(rewards))

    score.last_per_image = None

    print("  ImageReward: THUDM ImageReward-v1.0 (prompt fidelity + quality, higher=better)")
    return score


BUILDERS = {
    "clip_score": build_clip_scorer,
    "brisque": build_brisque_scorer,
    "clip_iqa": build_clip_iqa_scorer,
    "image_reward": build_image_reward_scorer,
}
