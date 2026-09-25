"""Metrics for the local-level comparison.

Per (source image, sweep point) we generate n samples and report:

  Faithfulness / locality
    l2          : mean Euclidean distance ||guide - sample||_2 over [0,1]
                  images at 256x256 (SDEdit Table 1's "L2" faithfulness score).
    dino_faith  : mean DINO CLS cosine similarity to the source (same metric
                  as v2 Metrics.semantic_preservation).

  Realism
    kid         : Kernel Inception Distance between the POOLED generated set
                  (all sources x samples of one arm/sweep point) and a real
                  reference set from the same LSUN category. Unbiased MMD^2
                  with the standard polynomial kernel k(x,y) = (x.y/d + 1)^3
                  on Inception-v3 pool features, averaged over random subsets
                  (Binkowski et al., 2018 — the metric SDEdit reports).
                  NOTE: features come from torchvision's Inception-v3, not the
                  TF-ported one in torch-fidelity, so compare KID numbers only
                  WITHIN this experiment, not against published tables.

  Diversity (local: computed per source over its n samples, then averaged)
    vendi_clip  : Vendi score of CLIP image embeddings (Friedman & Dieng 2022).
    vendi_pixel : Vendi score of normalized raw pixels.
    vendi_dino  : fdeval vendi_dino_q1 — Shannon Vendi (q=1, cosine) of
                  HuggingFace DINOv2-base CLS (AutoImageProcessor crop).
                  DINO_f / DINO_d below still use DINO v1 (dino-vits16).
    clip_div    : mean pairwise CLIP cosine DISTANCE (1 - similarity).
    dino_div    : mean pairwise DINO cosine DISTANCE (1 - similarity).
    ssim_div    : mean pairwise 1 - SSIM (structural perceptual spread).
    lpips_div   : mean pairwise LPIPS distance (optional; requires `pip
                  install lpips`, silently reported as None if unavailable).

  User Pref (%) needs humans by definition — make_user_study.py builds the
  pairwise A/B study pages and tallies the votes CSV.
"""
import itertools
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import os

_FDEVAL = os.environ.get("FDEVAL_ROOT", "")


class LocalLevelMetrics:
    def __init__(self, device, batch_size=32, with_lpips=True):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        from torchvision.models import inception_v3, Inception_V3_Weights

        self.device = device
        self.batch_size = batch_size

        print("Loading DINO v1 (facebook/dino-vits16) for faith/div...")
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vits16")
        self.dino_model = AutoModel.from_pretrained("facebook/dino-vits16").to(device).eval()

        print("Loading DINOv2-base for Vendi (fdeval vendi_dino_q1)...")
        self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

        print("Loading CLIP (openai/clip-vit-base-patch32)...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

        print("Loading Inception-v3 (torchvision) for KID features...")
        inc = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        inc.fc = torch.nn.Identity()
        self.inception = inc.to(device).eval()
        self._inc_norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.lpips_model = None
        if with_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
                print("Loaded LPIPS (alex).")
            except Exception as e:  # noqa: BLE001
                print(f"LPIPS unavailable ({e}); lpips_div will be None. "
                      f"`pip install lpips` to enable.")


    @torch.no_grad()
    def dino_embed(self, images_01):
        embeds = []
        for chunk in images_01.split(self.batch_size):
            pil = [T.ToPILImage()(im.cpu()) for im in chunk]
            inputs = self.dino_processor(images=pil, return_tensors="pt").to(self.device)
            out = self.dino_model(**inputs).last_hidden_state[:, 0]
            embeds.append(F.normalize(out, dim=-1))
        return torch.cat(embeds, dim=0)

    @staticmethod
    def _unwrap_clip_projection(out, embeds_attr, method_name):
        """Same transformers-version compatibility shim as v2's Metrics."""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        if hasattr(out, embeds_attr):
            return getattr(out, embeds_attr)
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        raise TypeError(f"Unexpected output type from {method_name}: {type(out)}")

    @torch.no_grad()
    def clip_embed(self, images_01):
        embeds = []
        for chunk in images_01.split(self.batch_size):
            pil = [T.ToPILImage()(im.cpu()) for im in chunk]
            inputs = self.clip_processor(images=pil, return_tensors="pt").to(self.device)
            out = self.clip_model.get_image_features(**inputs)
            emb = self._unwrap_clip_projection(out, "image_embeds", "get_image_features")
            embeds.append(F.normalize(emb, dim=-1))
        return torch.cat(embeds, dim=0)

    @torch.no_grad()
    def inception_feats(self, images_01):
        """2048-d pool features for KID. images_01: (B,3,H,W) in [0,1]."""
        feats = []
        for chunk in images_01.split(self.batch_size):
            x = F.interpolate(chunk.to(self.device), size=(299, 299),
                              mode="bilinear", align_corners=False)
            x = self._inc_norm(x)
            feats.append(self.inception(x).float().cpu())
        return torch.cat(feats, dim=0)


    @torch.no_grad()
    def l2_to_guide(self, images_01, guide_01):
        """Mean ||guide - sample||_2 over flattened [0,1] images."""
        g = guide_01.reshape(1, -1).to(images_01.device)
        x = images_01.reshape(images_01.shape[0], -1)
        return (x - g).norm(dim=1).mean().item()

    @torch.no_grad()
    def dino_faithfulness(self, images_01, guide_01):
        ref = self.dino_embed(guide_01.unsqueeze(0) if guide_01.dim() == 3 else guide_01)
        gen = self.dino_embed(images_01)
        return (gen @ ref.T).squeeze(-1).mean().item()


    @staticmethod
    def _vendi_from_embeds(emb):
        """exp(entropy of eigenvalues of E E^T / n) on L2-normalized rows."""
        n = emb.shape[0]
        if n < 2:
            return 0.0
        K = (emb @ emb.T) / n
        eig = torch.linalg.eigvalsh(K).clamp(min=0)
        eig = eig / eig.sum().clamp_min(1e-12)
        entropy = -(eig * torch.log(eig.clamp(min=1e-12))).sum()
        return torch.exp(entropy).item()

    @torch.no_grad()
    def vendi_clip(self, images_01):
        return self._vendi_from_embeds(self.clip_embed(images_01))

    @torch.no_grad()
    def vendi_pixel(self, images_01):
        flat = images_01.reshape(images_01.shape[0], -1)
        return self._vendi_from_embeds(F.normalize(flat, dim=-1))

    @torch.no_grad()
    def dinov2_cls(self, images_01):
        """Unnormalized DINOv2-base CLS, same as fdeval.scorers.DinoScorer."""
        out = []
        for chunk in images_01.split(self.batch_size):
            pil = [T.ToPILImage()(im.cpu()) for im in chunk]
            inputs = self.dinov2_processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            h = self.dinov2_model(**inputs).last_hidden_state[:, 0]
            out.append(h.float().cpu())
        return torch.cat(out, dim=0).numpy()

    @torch.no_grad()
    def vendi_dino(self, images_01):
        """fdeval vendi_dino_q1: cosine Shannon Vendi on DINOv2-base CLS."""
        if images_01.shape[0] < 2:
            return 0.0
        if not _FDEVAL:
            raise RuntimeError("vendi_dino needs FDEVAL_ROOT set to the "
                               "fdeval scoring harness")
        if _FDEVAL not in sys.path:
            sys.path.insert(0, _FDEVAL)
        from fdeval.metrics_core import vendi_score
        return float(vendi_score(self.dinov2_cls(images_01), q=1.0, kernel="cosine"))

    @torch.no_grad()
    def clip_pairwise_diversity(self, images_01):
        emb = self.clip_embed(images_01)
        n = emb.shape[0]
        if n < 2:
            return 0.0
        sims = emb @ emb.T
        off = sims[~torch.eye(n, dtype=torch.bool, device=sims.device)]
        return (1.0 - off).mean().item()

    @staticmethod
    @torch.no_grad()
    def ssim_pairwise_diversity(images_01, max_pairs=256):
        """Mean pairwise 1 - SSIM over samples for one source.
        images_01: (n,3,H,W) in [0,1]. Requires scikit-image; None if missing."""
        try:
            from skimage.metrics import structural_similarity as ssim_fn
        except ImportError:
            return None
        imgs = (images_01.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        n = imgs.shape[0]
        if n < 2:
            return 0.0
        pairs = list(itertools.combinations(range(n), 2))
        if len(pairs) > max_pairs:
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(pairs), generator=g)[:max_pairs]
            pairs = [pairs[i] for i in idx]
        dists = [1.0 - ssim_fn(imgs[i], imgs[j], channel_axis=2, data_range=255)
                 for i, j in pairs]
        return float(sum(dists) / len(dists))

    @torch.no_grad()
    def dino_pairwise_diversity(self, images_01):
        emb = self.dino_embed(images_01)
        n = emb.shape[0]
        if n < 2:
            return 0.0
        sims = emb @ emb.T
        off = sims[~torch.eye(n, dtype=torch.bool, device=sims.device)]
        return (1.0 - off).mean().item()

    @torch.no_grad()
    def lpips_pairwise_diversity(self, images_01, max_pairs=256):
        if self.lpips_model is None:
            return None
        n = images_01.shape[0]
        if n < 2:
            return 0.0
        pairs = list(itertools.combinations(range(n), 2))
        if len(pairs) > max_pairs:
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(pairs), generator=g)[:max_pairs]
            pairs = [pairs[i] for i in idx]
        x = images_01.to(self.device) * 2.0 - 1.0
        dists = []
        for s in range(0, len(pairs), self.batch_size):
            chunk = pairs[s:s + self.batch_size]
            a = torch.stack([x[i] for i, _ in chunk])
            b = torch.stack([x[j] for _, j in chunk])
            dists.append(self.lpips_model(a, b).reshape(-1).cpu())
        return torch.cat(dists).mean().item()


    @staticmethod
    def kid(feats_gen, feats_ref, subset_size=1000, n_subsets=100, seed=0):
        """Unbiased-MMD^2 KID with polynomial kernel (x.y/d + 1)^3, averaged
        over `n_subsets` random subsets of size min(subset_size, n_gen, n_ref).
        Returns the raw KID (multiply by 1e3 for the usual reporting scale).
        """
        x = feats_gen.double()
        y = feats_ref.double()
        m = min(subset_size, x.shape[0], y.shape[0])
        d = x.shape[1]
        g = torch.Generator().manual_seed(seed)

        def poly(a, b):
            return (a @ b.T / d + 1.0) ** 3

        vals = []
        for _ in range(n_subsets):
            xi = x[torch.randperm(x.shape[0], generator=g)[:m]]
            yi = y[torch.randperm(y.shape[0], generator=g)[:m]]
            k_xx = poly(xi, xi)
            k_yy = poly(yi, yi)
            k_xy = poly(xi, yi)
            sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
            sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
            sum_xy = k_xy.mean()
            vals.append((sum_xx + sum_yy - 2 * sum_xy).item())
        return float(sum(vals) / len(vals))
