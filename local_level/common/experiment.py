"""Shared experiment plumbing for all six arms: source loading, the KID
reference-feature cache, per-sweep-point metric accumulation, and sample
saving (individual PNGs for the user study + a preview grid)."""
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image


def load_sources(data_root, dataset, n_sources, resolution=256):
    """Returns list of (name, tensor(3,H,W) in [0,1])."""
    src_dir = os.path.join(data_root, dataset, "sources")
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
    if len(files) < n_sources:
        raise RuntimeError(
            f"{src_dir} has {len(files)} sources < requested {n_sources}; "
            f"run common/prepare_data.py (see download_data.sh)."
        )
    out = []
    for f in files[:n_sources]:
        img = Image.open(os.path.join(src_dir, f)).convert("RGB")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)
        out.append((os.path.splitext(f)[0], T.ToTensor()(img)))
    return out


def reference_inception_feats(metrics, data_root, dataset, resolution=256):
    """Inception features of the real reference set, cached next to the data."""
    ref_dir = os.path.join(data_root, dataset, "reference")
    cache = os.path.join(data_root, dataset, f"reference_inception_{resolution}.pt")
    if os.path.isfile(cache):
        print(f"Loading cached reference Inception features: {cache}")
        return torch.load(cache, map_location="cpu")
    files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    if not files:
        raise RuntimeError(f"No reference images in {ref_dir}; run prepare_data.")
    print(f"Computing Inception features for {len(files)} reference images...")
    feats = []
    chunk = 64
    for s in range(0, len(files), chunk):
        imgs = []
        for f in files[s:s + chunk]:
            img = Image.open(os.path.join(ref_dir, f)).convert("RGB")
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.LANCZOS)
            imgs.append(T.ToTensor()(img))
        feats.append(metrics.inception_feats(torch.stack(imgs)))
    feats = torch.cat(feats, dim=0)
    torch.save(feats, cache)
    return feats


class PointAccumulator:
    """Accumulates per-source metrics for ONE (arm, sweep point); KID is
    computed at finalize() over the pooled generated set."""

    def __init__(self, metrics, guide_res=256):
        self.metrics = metrics
        self.guide_res = guide_res
        self.per_source = {k: [] for k in
                           ["l2", "dino_faith", "vendi_clip", "vendi_pixel",
                            "vendi_dino", "vendi_dino_q1", "clip_div", "dino_div",
                            "ssim_div", "lpips_div"]}
        self.gen_feats = []

    def add_source(self, images_01, guide_01):
        """images_01: (n,3,H,W) generated for one source; guide_01: (3,h,w).
        Images are resized to the guide resolution for L2/KID consistency."""
        if images_01.shape[-1] != self.guide_res:
            images_01 = torch.nn.functional.interpolate(
                images_01, size=(self.guide_res, self.guide_res),
                mode="bilinear", align_corners=False)
        m = self.metrics
        self.per_source["l2"].append(m.l2_to_guide(images_01, guide_01))
        self.per_source["dino_faith"].append(m.dino_faithfulness(images_01, guide_01))
        self.per_source["vendi_clip"].append(m.vendi_clip(images_01))
        self.per_source["vendi_pixel"].append(m.vendi_pixel(images_01))
        vd = m.vendi_dino(images_01)
        self.per_source["vendi_dino"].append(vd)
        self.per_source["vendi_dino_q1"].append(vd)
        self.per_source["clip_div"].append(m.clip_pairwise_diversity(images_01))
        self.per_source["dino_div"].append(m.dino_pairwise_diversity(images_01))
        sd = m.ssim_pairwise_diversity(images_01)
        if sd is not None:
            self.per_source["ssim_div"].append(sd)
        lp = m.lpips_pairwise_diversity(images_01)
        if lp is not None:
            self.per_source["lpips_div"].append(lp)
        self.gen_feats.append(m.inception_feats(images_01))

    def finalize(self, ref_feats, kid_subset_size=1000, kid_subsets=100):
        out = {}
        for k, v in self.per_source.items():
            if v:
                out[k] = float(np.mean(v))
                out[k + "_std"] = float(np.std(v))
            else:
                out[k] = None
        gen = torch.cat(self.gen_feats, dim=0)
        kid = self.metrics.kid(gen, ref_feats,
                               subset_size=kid_subset_size,
                               n_subsets=kid_subsets)
        out["kid"] = kid
        out["kid_x1000"] = kid * 1000.0
        out["n_pooled"] = int(gen.shape[0])
        return out

    def state_dict(self):
        """Plain-tensor snapshot for checkpointing (metrics/guide_res are
        reconstructed by the caller, not saved)."""
        return {"per_source": self.per_source, "gen_feats": self.gen_feats}

    def load_state_dict(self, state):
        self.per_source = state["per_source"]
        self.gen_feats = state["gen_feats"]


def save_source_samples(images_01, out_dir, src_name, arm_label, n_save=8):
    """samples/<src>/<arm_label>/s00.png ... (for make_user_study.py) plus a
    one-row preview grid."""
    d = os.path.join(out_dir, "samples", src_name, arm_label)
    os.makedirs(d, exist_ok=True)
    n = min(n_save, images_01.shape[0])
    for j in range(n):
        save_image(images_01[j], os.path.join(d, f"s{j:02d}.png"))
    save_image(images_01[:n], os.path.join(d, "grid.png"), nrow=n)
