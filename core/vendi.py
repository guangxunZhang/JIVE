"""Vendi diversity scores (pixel and pretrained-feature) and the feature
embedder used to compute the latter.

Feature Vendi matches fdeval / Friedman & Dieng: cosine kernel on
L2-normalised rows, eigenvalues of K/n, exp of Renyi entropy (q=1 Shannon
by default). The default DINOv2 frontend is the same as
``fdeval.scorers.DinoScorer``: HuggingFace ``facebook/dinov2-base``,
AutoImageProcessor (resize short side 256, center-crop 224), CLS token.

Pixel Vendi is unchanged (mean-centred cosine, rescaled to [0, 1]).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

FDEVAL = os.environ.get("FDEVAL_ROOT", "")
VENDI_ORDERS = (0.5, 1.0, 2.0, np.inf)


def _ensure_fdeval():
    if not FDEVAL:
        raise RuntimeError(
            "Feature Vendi needs the fdeval scoring harness; set FDEVAL_ROOT "
            "to its checkout, or pass --vendi-feature pixel."
        )
    if FDEVAL not in sys.path:
        sys.path.insert(0, FDEVAL)
    from fdeval.metrics_core import (  # noqa: WPS433
        vendi_all_orders,
        vendi_from_eigenvalues,
        vendi_score,
        gram_eigenvalues,
    )
    return vendi_score, vendi_all_orders, gram_eigenvalues, vendi_from_eigenvalues


def _as_numpy(feats) -> np.ndarray:
    if torch.is_tensor(feats):
        return feats.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(feats, dtype=np.float64)


@torch.no_grad()
def vendi_score_pixel(images, kernel="cosine", rbf_gamma=None, eps=1e-12):
    """Vendi Score from raw pixels; images (N,C,H,W) in [0,1]."""
    n = images.shape[0]
    if n <= 1:
        return 1.0

    x = images.float().reshape(n, -1)
    if kernel == "cosine":
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(eps)
        sim = x @ x.T
        sim = (sim + 1.0) * 0.5
        sim.fill_diagonal_(1.0)
    elif kernel == "rbf":
        dist2 = torch.cdist(x, x, p=2).pow(2)
        if rbf_gamma is None:
            nonzero = dist2[dist2 > 0]
            median_dist2 = nonzero.median().clamp_min(eps) if nonzero.numel() else torch.tensor(1.0)
            rbf_gamma = 1.0 / median_dist2.item()
        sim = torch.exp(-rbf_gamma * dist2)
    else:
        raise ValueError(f"Unknown pixel Vendi kernel: {kernel}")

    sim = 0.5 * (sim + sim.T)
    eigvals = torch.linalg.eigvalsh(sim).clamp_min(0)
    probs = eigvals / eigvals.sum().clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum()
    return entropy.exp().item()


def vendi_from_features(feats, q: float = 1.0, kernel: str = "cosine") -> float:
    """Official / fdeval feature Vendi (cosine Shannon at q=1 by default)."""
    X = _as_numpy(feats)
    if X.shape[0] <= 1:
        return 1.0
    vendi_score, _, _, _ = _ensure_fdeval()
    return float(vendi_score(X, q=q, kernel=kernel))


def vendi_from_normalized_feats(feats, eps=1e-12, q: float = 1.0):
    """Back-compat wrapper: fdeval re-normalises rows, so pre-norm is optional."""
    del eps
    return vendi_from_features(feats, q=q, kernel="cosine")


def vendi_orders_from_features(feats, prefix: str = "vendi_dino") -> dict:
    """fdeval-style multi-order block (q=0.5, 1, 2, inf) plus frac and n."""
    X = _as_numpy(feats)
    if X.shape[0] <= 1:
        out = {f"{prefix}_q0.5": 1.0, f"{prefix}_q1": 1.0,
               f"{prefix}_q2": 1.0, f"{prefix}_qinf": 1.0}
        out[f"{prefix}_frac"] = 1.0
        out[f"{prefix}_n"] = int(X.shape[0])
        return out
    _, vendi_all_orders, _, _ = _ensure_fdeval()
    raw = vendi_all_orders(X, orders=VENDI_ORDERS, kernel="cosine")
    out = {}
    for q in VENDI_ORDERS:
        qk = "inf" if np.isinf(q) else f"{q:g}"
        out[f"{prefix}_q{qk}"] = float(raw[f"vendi_q{qk}"])
    out[f"{prefix}_frac"] = out[f"{prefix}_q1"] / X.shape[0]
    out[f"{prefix}_n"] = int(X.shape[0])
    return out


def vendi_score_features(images, embed_fn, eps=1e-12, q: float = 1.0):
    """Vendi Score from pretrained image features (fdeval cosine kernel)."""
    del eps
    n = images.shape[0]
    if n <= 1:
        return 1.0
    return vendi_from_features(embed_fn(images).float(), q=q)


def _tensors_to_pils(images):
    """NCHW float [0, 1] -> list of RGB PILs (DINOv2 processor input)."""
    from torchvision.transforms.functional import to_pil_image

    pils = []
    for im in images:
        x = im.detach().cpu().float().clamp(0, 1)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        pils.append(to_pil_image(x))
    return pils


def build_image_embedder(kind="auto", device="cpu", batch_size=32):
    """Returns embed_fn(images NCHW in [0, 1]) -> feature tensor, or None.
    The embed functions process images in chunks of `batch_size` so pooled
    metric sets of hundreds of images do not OOM the GPU.

    The returned callable has a ``kind`` attribute (``dinov2`` / ``inception``
    / ``clip``) so reporting can emit fdeval ``vendi_dino_*`` keys.
    """
    import torch.nn.functional as F

    kind = kind.lower()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def prep(images, size):
        x = images.to(device).float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(
            x, size=(size, size), mode="bicubic",
            align_corners=False, antialias=True,
        ).clamp(0, 1)
        return (x - mean) / std

    def chunked(fn, images, size):
        outs = []
        for chunk in images.split(batch_size):
            outs.append(fn(prep(chunk, size)).float().cpu())
        return torch.cat(outs, dim=0)

    def tag(fn, name):
        fn.kind = name
        return fn

    order = ["dinov2", "inception", "clip"] if kind == "auto" else [kind]
    for k in order:
        try:
            if k == "dinov2":
                from transformers import AutoImageProcessor, AutoModel

                model_id = "facebook/dinov2-base"
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id).to(device).eval()

                @torch.no_grad()
                def embed(images, _m=model, _p=processor):
                    outs = []
                    for chunk in images.split(batch_size):
                        inputs = _p(images=_tensors_to_pils(chunk), return_tensors="pt")
                        inputs = {key: val.to(device) for key, val in inputs.items()}
                        o = _m(**inputs)
                        h = getattr(o, "last_hidden_state", None)
                        if not torch.is_tensor(h):
                            h = o[0]
                        outs.append(h[:, 0].float().cpu())
                    return torch.cat(outs, dim=0)

                print("  Vendi feature extractor: HuggingFace DINOv2-base CLS "
                      "(fdeval facebook/dinov2-base, AutoImageProcessor crop)")
                return tag(embed, "dinov2")

            if k == "inception":
                import torchvision

                weights = torchvision.models.Inception_V3_Weights.DEFAULT
                model = torchvision.models.inception_v3(weights=weights, aux_logits=True)
                model.fc = torch.nn.Identity()
                model = model.to(device).eval()

                @torch.no_grad()
                def embed(images, _m=model):
                    return chunked(_m, images, 299)

                print("  Vendi feature extractor: InceptionV3 pool3 (2048-d)")
                return tag(embed, "inception")

            if k == "clip":
                import open_clip

                model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai")
                model = model.to(device).eval()

                @torch.no_grad()
                def embed(images, _m=model):
                    return chunked(_m.encode_image, images, 224)

                print("  Vendi feature extractor: CLIP ViT-B/32 (512-d)")
                return tag(embed, "clip")
        except Exception as exc:
            print(f"  [vendi] '{k}' embedder unavailable ({type(exc).__name__}: {exc}); trying next")

    print("  [vendi] no feature extractor available; falling back to pixel Vendi only.")
    return None
