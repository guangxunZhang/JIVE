import json
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch

from baselines.oscar.utils import (
    log as _log,
    parse_concepts_spec as _parse_concepts_spec,
    build_root_out as _build_root_out,
    prompt_run_dir as _prompt_run_dir,
)

from .args import parse_args
from .pipeline_setup import build_pipeline_context
from .runner import FluxArmRunner
from .oscar_arm import OscarArmFlux
from .jive_arm import JiveArmFlux
from .jive_cads_arm import JiveCadsArmFlux
from core.kid import kid_mmd2
from core.reporting import (
    compute_arm_metrics, print_summary, save_comparison_grid,
    write_results_json, save_metric_plot, save_score_sorted_grids,
)


def _save_arm_images(imgs_cpu: torch.Tensor, out_dir: str, keep_n: int) -> None:
    from torchvision.utils import save_image
    os.makedirs(out_dir, exist_ok=True)
    keep = min(int(keep_n), imgs_cpu.size(0))
    for k in range(keep):
        save_image(imgs_cpu[k].float(), os.path.join(out_dir, f"{k:03d}.png"))


def _run_one(args, ctx, imgs_root, prompt_text, g, sd):
    run_dir = _prompt_run_dir(imgs_root, prompt_text, int(sd), float(g), int(args.steps))
    results_path = os.path.join(run_dir, "results.json")
    if getattr(args, "skip_existing", False) and os.path.isfile(results_path):
        _log(f"[SKIP] results.json exists -> {run_dir}", True)
        return
    os.makedirs(run_dir, exist_ok=True)
    _log(f"[RUN] prompt='{prompt_text}' | seed={sd} | guidance={g} | steps={args.steps} -> {run_dir}", True)

    n_total = int(args.n_images)
    runner = FluxArmRunner(ctx.pipe, args, ctx.dev_tr, ctx.dev_vae, ctx.dtype,
                           prompt_text, g, sd)
    seq_len = (runner.lat_h // 2) * (runner.lat_w // 2)
    _log(f"[INIT] {n_total} shared per-image latent seeds (sd+0 .. sd+{n_total-1}), "
         f"packed latent shape=[{seq_len}, {runner.num_ch * 4}] "
         f"(unpacked {runner.num_ch}x{runner.lat_h}x{runner.lat_w}), batches of {args.G}", True)

    results: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    grid_rows: List[Tuple[str, torch.Tensor]] = []
    kid_ref_feats = None

    def _metrics(imgs):
        return compute_arm_metrics(imgs, prompt_text, args, ctx.embedder, ctx.scorers)

    def _kid(arm_name, m, imgs, is_reference=False):
        nonlocal kid_ref_feats
        if ctx.kid_featurizer is None:
            return
        if is_reference:
            kid_ref_feats = ctx.kid_featurizer(imgs).float()
            m["kid_vs_deterministic_mean"] = 0.0
            m["kid_vs_deterministic_std"] = 0.0
            return
        if kid_ref_feats is None:
            _log(f"[KID] {arm_name}: deterministic arm not in this run; skipping KID", True)
            return
        feats = ctx.kid_featurizer(imgs).float()
        mu, sigma = kid_mmd2(feats, kid_ref_feats, args.kid_subsets, args.kid_subset_size)
        m["kid_vs_deterministic_mean"], m["kid_vs_deterministic_std"] = mu, sigma

    def _sorted_grids(arm_name, m, imgs):
        per_img = m.pop("_per_image_scores", None)
        if not per_img:
            return None
        for p in save_score_sorted_grids(imgs, per_img, run_dir, arm_name):
            _log(f"[GRID] {arm_name} score-sorted -> {p}", True)
        return per_img

    if 'deterministic' in args.arms:
        _log(f"[ARM A] deterministic flow matching, {n_total} images from shared random starts ...", True)
        imgs_A = runner.run_arm("ARM A")
        _save_arm_images(imgs_A, os.path.join(run_dir, "A_deterministic"), args.keep_images_per_arm)
        results["deterministic"] = _metrics(imgs_A)
        results["deterministic"]["applied_budget_l2_mean"] = 0.0
        _kid("deterministic", results["deterministic"], imgs_A, is_reference=True)
        per_img_A = _sorted_grids("deterministic", results["deterministic"], imgs_A)
        _log(f"[ARM A] {results['deterministic']}", True)
        if per_img_A is not None:
            results["deterministic"]["_per_image_scores"] = per_img_A
        grid_rows.append(("deterministic", imgs_A[:args.keep_images_per_arm].clone()))
        del imgs_A
        if ctx.dev_tr.type == 'cuda': torch.cuda.empty_cache()

    if 'oscar' in args.arms:
        oscar = OscarArmFlux(ctx.pipe, ctx.vol, ctx.cfg, args, ctx.dev_vae, ctx.dev_clip)
        _log(f"[ARM B] OSCAR per-step perturbation, {n_total} images from the SAME starts "
             f"(volume group size = {args.G}, noise mode = {args.oscar_noise_mode}) ...", True)
        imgs_B = runner.run_arm("ARM B", callback_factory=oscar.callback_factory)
        oscar.finalize()
        _save_arm_images(imgs_B, os.path.join(run_dir, "B_oscar"), args.keep_images_per_arm)
        results["oscar"] = _metrics(imgs_B)
        results["oscar"]["applied_budget_l2_mean"] = oscar.mean_budget
        _kid("oscar", results["oscar"], imgs_B)
        per_img_B = _sorted_grids("oscar", results["oscar"], imgs_B)
        _log(f"[ARM B] {results['oscar']}", True)
        if per_img_B is not None:
            results["oscar"]["_per_image_scores"] = per_img_B
        grid_rows.append(("oscar", imgs_B[:args.keep_images_per_arm].clone()))
        del imgs_B
        if ctx.dev_tr.type == 'cuda': torch.cuda.empty_cache()

    if 'jive' in args.arms:
        jive = JiveArmFlux(ctx.pipe, args, ctx.dev_tr, prompt_text, g)
        _log(f"[ARM C] injection point resolved from the scheduler: "
             f"t0={float(jive.t_init.item()):.2f}, sigma0={jive.sigma_init:.4f}", True)
        for inj_norm in args.inject_norms:
            arm_name = f"jive_norm{inj_norm:g}" if len(args.inject_norms) > 1 else "jive"
            _log(f"[ARM C] per-image JIVE projection of the SAME pure-noise starts "
                 f"(sigma=1), {n_total} images (inject_norm={inj_norm}, rank={args.inject_n}, "
                 f"iters={args.jive_iters}, iter_mode={args.jive_iter_mode}) ...", True)
            transform = jive.make_start_transform(inj_norm, int(sd))
            imgs_C = runner.run_arm(f"ARM C:{arm_name}", latents_transform=transform)
            _save_arm_images(imgs_C, os.path.join(run_dir, f"C_{arm_name}"), args.keep_images_per_arm)
            results[arm_name] = _metrics(imgs_C)
            results[arm_name]["applied_budget_l2_mean"] = float(inj_norm)
            results[arm_name]["jive"] = {
                "rank": int(args.inject_n), "iters": int(args.jive_iters),
                "fd_eps": float(args.fd_eps), "iter_mode": args.jive_iter_mode,
            }
            _kid(arm_name, results[arm_name], imgs_C)
            per_img_C = _sorted_grids(arm_name, results[arm_name], imgs_C)
            _log(f"[ARM C:{arm_name}] {results[arm_name]}", True)
            if per_img_C is not None:
                results[arm_name]["_per_image_scores"] = per_img_C
            grid_rows.append((arm_name, imgs_C[:args.keep_images_per_arm].clone()))
            del imgs_C
            if ctx.dev_tr.type == 'cuda': torch.cuda.empty_cache()
        jive.close()

    if 'jive_cads' in args.arms:
        jive_cads = JiveCadsArmFlux(ctx.pipe, args, ctx.dev_tr, prompt_text, g)
        _log(f"[ARM D] CADS schedule: s={args.cads_s} tau1={args.cads_tau1} "
             f"tau2={args.cads_tau2} psi={args.cads_psi} (per-image noise, "
             f"seed-offset stream, pooled corrupted at step 0 only)", True)
        for inj_norm in args.inject_norms:
            arm_name = f"jive_cads_norm{inj_norm:g}" if len(args.inject_norms) > 1 else "jive_cads"
            _log(f"[ARM D] per-image JIVE projection of the SAME pure-noise starts "
                 f"+ per-step CADS conditioning corruption, {n_total} images "
                 f"(inject_norm={inj_norm}, rank={args.inject_n}, iters={args.jive_iters}, "
                 f"iter_mode={args.jive_iter_mode}) ...", True)
            transform = jive_cads.make_start_transform(inj_norm, int(sd))
            imgs_D = runner.run_arm(
                f"ARM D:{arm_name}",
                latents_transform=transform,
                callback_factory=jive_cads.make_cads_callback,
                embeds_factory=jive_cads.make_step0_embeds,
                callback_tensor_inputs=["latents", "prompt_embeds"],
            )
            _save_arm_images(imgs_D, os.path.join(run_dir, f"D_{arm_name}"), args.keep_images_per_arm)
            results[arm_name] = _metrics(imgs_D)
            results[arm_name]["applied_budget_l2_mean"] = float(inj_norm)
            results[arm_name]["cads"] = {
                "s": float(args.cads_s), "tau1": float(args.cads_tau1),
                "tau2": float(args.cads_tau2), "psi": float(args.cads_psi),
                "corrupt_pooled": "step0_only",
            }
            results[arm_name]["jive"] = {
                "rank": int(args.inject_n), "iters": int(args.jive_iters),
                "fd_eps": float(args.fd_eps), "iter_mode": args.jive_iter_mode,
            }
            _kid(arm_name, results[arm_name], imgs_D)
            per_img_D = _sorted_grids(arm_name, results[arm_name], imgs_D)
            _log(f"[ARM D:{arm_name}] {results[arm_name]}", True)
            if per_img_D is not None:
                results[arm_name]["_per_image_scores"] = per_img_D
            grid_rows.append((arm_name, imgs_D[:args.keep_images_per_arm].clone()))
            del imgs_D
            if ctx.dev_tr.type == 'cuda': torch.cuda.empty_cache()
        jive_cads.close()

    print_summary(prompt_text, sd, g, args.steps, n_total, results)

    grid_path = save_comparison_grid(grid_rows, run_dir, keep_n=8)
    if grid_path:
        _log(f"[GRID] rows={[nm for nm, _ in grid_rows]} -> {grid_path}", True)

    json_path = write_results_json(run_dir, prompt_text, sd, g, args, results)
    _log(f"[JSON] -> {json_path}", True)

    try:
        plot_path = save_metric_plot(run_dir, sd, g, n_total, results)
        if plot_path:
            _log(f"[PLOT] -> {plot_path}", True)
    except Exception as e:
        _log(f"metric plot failed: {e}", True)

    del results, grid_rows
    if ctx.dev_tr.type == 'cuda': torch.cuda.empty_cache()


def main():
    args = parse_args()

    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = args.out_root or pkg_parent

    if args.spec:
        spec_path = args.spec
        if not os.path.isabs(spec_path):
            spec_path = spec_path if os.path.isfile(spec_path) else os.path.join(pkg_parent, spec_path)
        with open(spec_path, "r", encoding="utf-8") as fp:
            spec_obj = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec_obj)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide --spec (JSON with {concept:[prompts...]}) or --prompt")

    guidances = args.guidances if args.guidances else [args.guidance]
    seeds = args.seeds if args.seeds else [args.seed]

    try:
        _log(f"[CFG] model={args.model} dir={args.model_dir} steps={args.steps} "
             f"guidances={guidances} method={args.method}", True)
        ctx = build_pipeline_context(args)

        sub = getattr(args, "outputs_subdir", None)
        sweep_root = os.path.join(out_root, f"outputs_{sub}") if sub else out_root
        for concept, prompt_list in concept_to_prompts.items():
            base_dir, imgs_root, eval_dir = _build_root_out(sweep_root, args.method, concept)
            _log(f"[OUT] base={base_dir}", True)
            for prompt_text in prompt_list:
                for g in guidances:
                    for sd in seeds:
                        _run_one(args, ctx, imgs_root, prompt_text, g, sd)

        _log("Done.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
