"""Per-arm metric computation and the run's summary/grid/json/plot outputs."""
import os
import json
from typing import Any, Dict, List, Tuple

import torch

from .vendi import (
    vendi_score_pixel,
    vendi_from_features,
    vendi_orders_from_features,
)
from .quality_metrics import PROMPT_AWARE_METRICS


def mean_pairwise_l2(imgs_cpu: torch.Tensor) -> float:
    """Mean L2 distance over all unique (i<j) image pairs -- a simple,
    interpretable diversity measure to sanity-check the Vendi scores against."""
    x = imgs_cpu.float().flatten(1)
    n = x.size(0)
    if n < 2:
        return 0.0
    d = torch.cdist(x, x)
    iu = torch.triu_indices(n, n, offset=1)  # upper triangle, excluding the diagonal
    return float(d[iu[0], iu[1]].mean().item())


def compute_arm_metrics(imgs_cpu: torch.Tensor, prompt_text: str, args,
                        embedder, scorers: Dict[str, Any]) -> Dict[str, Any]:
    """scorers: dict of metric_name -> callable, built once per run from the
    --quality-metrics selection. Callables in PROMPT_AWARE_METRICS take
    (images, prompt); the rest take (images) only.

    group_mean metrics are averaged over the generation batches of --G
    images: that is the group OSCAR's volume objective couples, so it gets a
    metric aligned with what it optimizes next to the pooled ones.
    """
    # Pooled pixel Vendi over ALL n images, plus the per-batch (--G) mean --
    # the latter matches the group size OSCAR's volume objective actually
    # optimizes, so it's directly comparable to what ARM B is targeting.
    grp_pix = [vendi_score_pixel(gr, args.vendi_kernel, args.vendi_rbf_gamma)
               for gr in imgs_cpu.split(args.G) if gr.size(0) > 1]
    m: Dict[str, Any] = {
        "pixel_vendi": vendi_score_pixel(imgs_cpu, args.vendi_kernel, args.vendi_rbf_gamma),
        "pixel_vendi_group_mean": (sum(grp_pix) / len(grp_pix)) if grp_pix else None,
        "mean_pairwise_l2": mean_pairwise_l2(imgs_cpu),
    }
    if embedder is not None:
        # fdeval / Friedman–Dieng feature Vendi: cosine kernel on L2-normalised
        # rows, eigenvalues of K/n. DINOv2 uses the same HF processor+CLS as
        # fdeval.scorers.DinoScorer; feature_vendi is then vendi_dino_q1.
        feats = embedder(imgs_cpu).float()
        m["feature_vendi"] = vendi_from_features(feats, q=1.0)
        grp_feat = [vendi_from_features(fg, q=1.0)
                    for fg in feats.split(args.G) if fg.size(0) > 1]
        m["feature_vendi_group_mean"] = (sum(grp_feat) / len(grp_feat)) if grp_feat else None
        if getattr(embedder, "kind", None) == "dinov2":
            m.update(vendi_orders_from_features(feats, prefix="vendi_dino"))
    # Fidelity/no-reference quality metrics from --quality-metrics, so
    # diversity numbers above are never read without a quality axis.
    for name, fn in scorers.items():
        m[name] = fn(imgs_cpu, prompt_text) if name in PROMPT_AWARE_METRICS else fn(imgs_cpu)
    # Scorers that expose per-image scores (attribute `last_per_image` set on
    # the most recent call) get them recorded under a private key, so the
    # distribution -- not just the mean -- lands in results.json and
    # score-sorted worst/best grids can be drawn from them.
    per_image = {}
    for name, fn in scorers.items():
        li = getattr(fn, "last_per_image", None)
        if li is not None and len(li) == imgs_cpu.size(0):
            per_image[name] = li
    if per_image:
        m["_per_image_scores"] = per_image
    return m


def _fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "n/a"


def print_summary(prompt_text, sd, g, steps, n_total, results: "Dict[str, Dict[str, Any]]"):
    print("\n===== SUMMARY (same starting points) =====")
    print(f"prompt='{prompt_text}' seed={sd} guidance={g} steps={steps} n_images={n_total}")
    for name, m in results.items():
        print(f"  {name:<16} feature_vendi={_fmt(m.get('feature_vendi'))} "
              f"(group {_fmt(m.get('feature_vendi_group_mean'))})  "
              f"pixel_vendi={_fmt(m.get('pixel_vendi'))} "
              f"(group {_fmt(m.get('pixel_vendi_group_mean'))})  "
              f"mean_pairwise_L2={_fmt(m.get('mean_pairwise_l2'))}  "
              f"clip_score={_fmt(m.get('clip_score'), '.2f')}  "
              f"brisque={_fmt(m.get('brisque'), '.2f')}  "
              f"clip_iqa={_fmt(m.get('clip_iqa'), '.3f')}  "
              f"image_reward={_fmt(m.get('image_reward'), '.3f')}  "
              f"hpsv2={_fmt(m.get('hpsv2'), '.4f')}  "
              f"kid_vs_det={_fmt(m.get('kid_vs_deterministic_mean'), '.4f')}"
              f"±{_fmt(m.get('kid_vs_deterministic_std'), '.4f')}  "
              f"budget_L2={_fmt(m.get('applied_budget_l2_mean'), '.2f')}")
        if m.get('cost_wall_time_s') is not None:
            print(f"  {'':<16} cost: time={_fmt(m.get('cost_wall_time_s'), '.1f')}s  "
                  f"TFLOPs={_fmt(m.get('cost_tflops_est'), '.1f')}  "
                  f"tf_fwds={m.get('cost_tf_forwards')}  "
                  f"vae_decodes={m.get('cost_vae_decodes')}  "
                  f"peak_mem={_fmt(m.get('cost_peak_gpu_mem_gb'), '.1f')}GB")
    print("==========================================\n")


def save_comparison_grid(grid_rows: List[Tuple[str, torch.Tensor]], run_dir: str, keep_n: int = 8):
    """One row per arm (first `keep_n` images only). Returns the path, or
    None if there were no rows to draw."""
    from torchvision.utils import save_image, make_grid

    if not grid_rows:
        return None
    # Use the smallest available count across arms so every row has the
    # same number of columns (needed for make_grid's fixed nrow layout).
    n = min(min(r.size(0) for _, r in grid_rows), keep_n)
    grid = make_grid(
        torch.cat([r[:n].float() for _, r in grid_rows], dim=0),
        nrow=n, padding=4,
    )
    grid_path = os.path.join(run_dir, "compare_" + "_".join(nm for nm, _ in grid_rows) + ".png")
    save_image(grid, grid_path)
    return grid_path


def save_score_sorted_grids(imgs_cpu: torch.Tensor, per_image_scores: Dict[str, List[float]],
                            out_dir: str, arm_name: str, keep: int = 16, nrow: int = 8) -> List[str]:
    """For each metric with per-image scores, saves a worst-`keep` and a
    best-`keep` grid (images sorted ascending/descending by score) named
    `<arm>_<worst|best><keep>_by_<metric>.png` in `out_dir`. These make the
    low-reward tail directly inspectable, which the mean alone hides."""
    from torchvision.utils import save_image, make_grid

    paths: List[str] = []
    for metric, scores in per_image_scores.items():
        s = torch.tensor(scores, dtype=torch.float32)
        n = min(int(keep), s.numel())
        if s.numel() != imgs_cpu.size(0) or n == 0:
            continue
        order = torch.argsort(s)
        for tag, idx in (("worst", order[:n]), ("best", order[-n:].flip(0))):
            grid = make_grid(imgs_cpu[idx].float(), nrow=min(nrow, n), padding=4)
            p = os.path.join(out_dir, f"{arm_name}_{tag}{n}_by_{metric}.png")
            save_image(grid, p)
            paths.append(p)
    return paths


def write_results_json(run_dir, prompt_text, sd, g, args, results) -> str:
    """Serializes the full run config (so results.json is self-describing
    even without the original command line) plus every arm's metrics."""
    payload = {
        "prompt": prompt_text,
        "seed": int(sd),
        "guidance": float(g),
        "steps": int(args.steps),
        "n_images": int(args.n_images),
        "G": int(args.G),
        "height": int(args.height),
        "width": int(args.width),
        "inject": {
            "norms": [float(x) for x in args.inject_norms],
            "at": "pure_noise_start (t=1, sigma=1)",
            "rank": int(args.inject_n),
            "jive_iters": int(args.jive_iters),
            "fd_eps": float(args.fd_eps),
        },
        "oscar": {
            "noise_mode": args.oscar_noise_mode,
            "t_gate": args.t_gate,
            "gamma0": float(args.gamma0),
            "gamma_max_ratio": float(args.gamma_max_ratio),
            "rho": float(args.rho),
            "eta_sde": float(args.eta_sde),
            "group_size_G": int(args.G),
        },
        "quality_metrics": list(args.quality_metrics),
        "arms": results,
    }
    path = os.path.join(run_dir, "results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_metric_plot(run_dir, sd, g, n_total, results):
    """Returns the plot path, or None if no metric was shared by every arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    metric_keys = ["feature_vendi", "feature_vendi_group_mean", "pixel_vendi",
                   "mean_pairwise_l2", "clip_score", "brisque", "clip_iqa", "image_reward",
                   "hpsv2", "kid_vs_deterministic_mean",
                   "cost_tflops_est", "cost_wall_time_s"]
    # Only plot metrics every arm actually has (e.g. skip clip_score entirely
    # if it wasn't in --quality-metrics, rather than plotting a partial bar).
    metric_keys = [mk for mk in metric_keys
                   if all(results[nm].get(mk) is not None for nm in names)]
    if not metric_keys:
        return None

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4.5 * len(metric_keys), 4))
    if len(metric_keys) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    for ax, mk in zip(axes, metric_keys):
        vals = [results[nm][mk] for nm in names]
        ax.bar(range(len(names)), vals, color=[colors[q % 10] for q in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(mk)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"seed={sd} g={g} n={n_total} | same random starts")
    fig.tight_layout()
    plot_path = os.path.join(run_dir, "compare_metrics.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path
