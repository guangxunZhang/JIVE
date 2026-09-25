import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "baselines", "rf_inversion"))

from diffusers import FluxPipeline
from diffusers.training_utils import set_seed
from diffusers.pipelines.flux.pipeline_flux import (
    calculate_shift, retrieve_timesteps,
)
from pipeline_rf_inversion_sde import RFInversionFluxPipelineSDE

from rf_inversion_sampling import (
    DEFAULT_FLUX_MODEL, build_deltas, make_flux_velocity_fn,
    rf_inversion_sample,
)
from common.experiment import (
    PointAccumulator, load_sources, reference_inception_feats,
    save_source_samples,
)
from common.metrics import LocalLevelMetrics
from common.prepare_data import DATASETS
from common.volume_expansion import top_subspace, top_subspace_jtj

DEFAULT_PROMPTS = {
    "classroom": "a photo of a classroom",
    "kitchen": "a photo of a kitchen",
    "conference_room": "a photo of a conference room",
    "dining_room": "a photo of a dining room",
    "restaurant": "a photo of a restaurant",
}


def make_flux_vjp_fn(transformer, prompt_embeds, pooled_prompt_embeds,
                     text_ids, latent_image_ids, guidance_scale, vjp_chunk,
                     device, sigma, t_val, z0, latent_shape):
    use_guidance = bool(getattr(transformer.config, "guidance_embeds", False))
    weight_dtype = transformer.x_embedder.weight.dtype
    z0_f32 = z0.reshape(latent_shape).to(device=device, dtype=torch.float32)
    D = z0_f32.numel()

    def vjp_fn(W):
        k = W.shape[1]
        req = [p.requires_grad for p in transformer.parameters()]
        transformer.requires_grad_(False)
        grads = []
        try:
            for s in range(0, k, vjp_chunk):
                r = min(vjp_chunk, k - s)
                z_r = z0_f32.expand(r, -1, -1).clone().requires_grad_(True)
                tt = torch.full((r,), t_val / 1000.0, device=device,
                                dtype=weight_dtype)
                g = (torch.full((r,), float(guidance_scale), device=device,
                                dtype=torch.float32) if use_guidance else None)
                with torch.enable_grad():
                    v = transformer(
                        hidden_states=z_r.to(weight_dtype),
                        timestep=tt,
                        guidance=g,
                        pooled_projections=pooled_prompt_embeds.expand(r, -1).to(
                            dtype=weight_dtype),
                        encoder_hidden_states=prompt_embeds.expand(
                            r, -1, -1).to(dtype=weight_dtype),
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0].float().reshape(r, D)
                loss = (v * W[:, s:s + r].T).sum()
                gz = torch.autograd.grad(loss, z_r)[0]
                grads.append(gz.reshape(r, D).float())
        finally:
            for p, flag in zip(transformer.parameters(), req):
                p.requires_grad_(flag)
        dvT_W = torch.cat(grads, dim=0).T
        return W - float(sigma) * dvT_W

    return vjp_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--prompt", default=None)
    p.add_argument("--model", default=DEFAULT_FLUX_MODEL)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_sources", type=int, default=16,
                   help="FLUX inversion+sampling is expensive; fewer sources "
                        "than the DDPM arms by default")
    p.add_argument("--n", type=int, default=16, help="Samples per source")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--etas", type=float, nargs="+", default=[0.9, 0.7, 0.6, 0.5])
    p.add_argument("--start_timestep", type=float, default=0.0)
    p.add_argument("--stop_timestep", type=float, default=0.25)
    p.add_argument("--inject_norms", type=float, nargs="+",
                   default=[4.0, 8.0, 12.0, 16.0])
    p.add_argument("--skip_baseline", action="store_true",
                   help="Only run +JIVE; skip RF-Inversion baseline sampling")
    p.add_argument("--results_name", default="results.json",
                   help="Metrics JSON filename under out_dir. Use a distinct "
                        "name to add a new inject_norm without skipping or "
                        "overwriting an existing results.json.")
    p.add_argument("--inject_n", type=int, default=4)
    p.add_argument("--power_iters", type=int, default=10)
    p.add_argument("--fd_eps", type=float, default=1e-1)
    p.add_argument("--jive_iter_mode", choices=["j", "jtj"], default="j",
                   help="j: Q <- QR(J Q) (invariant subspace of J). "
                        "jtj: Q <- QR(J^T J Q) (right singular vectors of J).")
    p.add_argument("--vjp_chunk", type=int, default=2,
                   help="VJP batch cap for --jive_iter_mode jtj (memory peak).")
    p.add_argument("--perturb_mode", choices=["additive", "boundary"],
                   default="additive",
                   help="additive: set-level z += projected Gaussian at exact "
                        "L2 norm. boundary: local sphere rotation (old).")
    p.add_argument("--inject_after_eta", action="store_true",
                   help="Add JIVE at the first reverse step with eta_t=0 "
                        "(after stop_timestep), not at t=0.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--fwd_chunk", type=int, default=2)
    p.add_argument("--metric_batch_size", type=int, default=32)
    p.add_argument("--kid_subset_size", type=int, default=250)
    p.add_argument("--kid_subsets", type=int, default=100)
    p.add_argument("--save_sources", type=int, default=8)
    p.add_argument("--save_samples_per_source", type=int, default=8)
    p.add_argument("--metric_resolution", type=int, default=256,
                   help="L2/KID/Vendi are computed at this size so numbers "
                        "are comparable with the 256px DDPM arms")
    args = p.parse_args()

    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS[args.dataset]
    if args.out_dir is None:
        args.out_dir = os.path.join("out", args.dataset, "rf_inverse")
    os.makedirs(args.out_dir, exist_ok=True)

    results_path = os.path.join(args.out_dir, args.results_name)
    if os.path.isfile(results_path):
        print(f"SKIP: {results_path} already exists.")
        return

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = "cuda:0" if n_gpus > 0 else "cpu"
    model_dtype = torch.bfloat16 if n_gpus > 0 else torch.float32
    local_only = os.path.isdir(args.model)
    print(f"dataset={args.dataset} prompt='{args.prompt}' device={device} "
          f"n_gpus={n_gpus} model={args.model} "
          f"jive_iter_mode={args.jive_iter_mode} "
          f"perturb_mode={args.perturb_mode} "
          f"inject_after_eta={args.inject_after_eta} "
          f"skip_baseline={args.skip_baseline} inject_norms={args.inject_norms}")

    def _load_pipe_rf(dev):
        pipe = FluxPipeline.from_pretrained(
            args.model, torch_dtype=model_dtype, local_files_only=local_only)
        pipe.to(device=dev, dtype=model_dtype)
        return RFInversionFluxPipelineSDE.from_pipe(pipe, torch_dtype=model_dtype)

    pipes = [_load_pipe_rf(f"cuda:{i}" if n_gpus > 0 else "cpu")
             for i in range(max(1, n_gpus))]
    pipe_rf = pipes[0]
    print(f"Loaded {len(pipes)} RF-Inversion pipeline replica(s)")

    sources = load_sources(args.data_root, args.dataset, args.n_sources,
                           args.metric_resolution)

    print(f"Encoding prompt: \"{args.prompt}\"")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe_rf.encode_prompt(
        prompt=args.prompt, prompt_2=None, device=device,
        num_images_per_prompt=1, max_sequence_length=512,
    )

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as ex:
        metrics_future = ex.submit(LocalLevelMetrics, device, args.metric_batch_size)
        metrics = metrics_future.result()
    ref_feats = reference_inception_feats(metrics, args.data_root,
                                          args.dataset, args.metric_resolution)

    acc_base = ({eta: PointAccumulator(metrics, args.metric_resolution)
                 for eta in args.etas} if not args.skip_baseline else {})
    acc_ours = {(eta, nrm): PointAccumulator(metrics, args.metric_resolution)
                for eta in args.etas for nrm in args.inject_norms}

    ckpt_stem = ("checkpoint" if args.results_name == "results.json"
                 else os.path.splitext(args.results_name)[0] + ".checkpoint")
    ckpt_path = os.path.join(args.out_dir, ckpt_stem + ".pt")
    resume_from = 0
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if (ckpt.get("etas") == args.etas
                and ckpt.get("inject_norms") == args.inject_norms
                and ckpt.get("skip_baseline", False) == args.skip_baseline
                and ckpt.get("jive_iter_mode", "j") == args.jive_iter_mode
                and ckpt.get("perturb_mode", "boundary") == args.perturb_mode
                and ckpt.get("inject_after_eta", False) == args.inject_after_eta):
            if not args.skip_baseline:
                for eta in args.etas:
                    acc_base[eta].load_state_dict(ckpt["acc_base"][eta])
            for eta in args.etas:
                for nrm in args.inject_norms:
                    acc_ours[(eta, nrm)].load_state_dict(ckpt["acc_ours"][(eta, nrm)])
            resume_from = ckpt["next_source_idx"]
            print(f"Resuming from checkpoint: {resume_from}/{len(sources)} "
                  f"sources already done.")
        else:
            print("Checkpoint found but etas/inject_norms differ from this "
                  "run's args; ignoring it and starting over.")

    def save_checkpoint(next_idx):
        torch.save({
            "next_source_idx": next_idx,
            "etas": args.etas, "inject_norms": args.inject_norms,
            "skip_baseline": args.skip_baseline,
            "jive_iter_mode": args.jive_iter_mode,
            "perturb_mode": args.perturb_mode,
            "inject_after_eta": args.inject_after_eta,
            "acc_base": {eta: acc_base[eta].state_dict() for eta in acc_base},
            "acc_ours": {k: v.state_dict() for k, v in acc_ours.items()},
        }, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    for src_idx, (src_name, src_01) in enumerate(sources):
        if src_idx < resume_from:
            continue
        print(f"\n=== source {src_idx + 1}/{len(sources)}: {src_name} ===")
        img = T.ToPILImage()(src_01).resize((args.width, args.height),
                                            Image.LANCZOS)

        set_seed(args.seed)
        inverted_latents, image_latents, latent_image_ids = pipe_rf.invert(
            image=img, num_inversion_steps=args.steps, gamma=args.gamma,
            height=args.height, width=args.width,
        )
        latent_shape = (1,) + tuple(inverted_latents.shape[1:])

        sigmas_lin = np.linspace(1.0, 1.0 / args.steps, args.steps)
        mu = calculate_shift(
            inverted_latents.shape[1],
            pipe_rf.scheduler.config.get("base_image_seq_len", 256),
            pipe_rf.scheduler.config.get("max_image_seq_len", 4096),
            pipe_rf.scheduler.config.get("base_shift", 0.5),
            pipe_rf.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipe_rf.scheduler, args.steps, device, None, sigmas_lin, mu=mu)
        t0 = float(timesteps[0].item())
        sigma0 = float(pipe_rf.scheduler.sigmas[0].item())

        velocity = make_flux_velocity_fn(
            pipe_rf.transformer, prompt_embeds, pooled_prompt_embeds,
            text_ids, latent_image_ids, args.guidance_scale, args.fwd_chunk,
            device, model_dtype,
        )
        z_lin = inverted_latents.reshape(latent_shape).float()
        if args.jive_iter_mode == "jtj":
            def endpoint_fn(z_batch, _t=t0, _sig=sigma0):
                return z_batch - _sig * velocity(z_batch, _t)

            vjp_fn = make_flux_vjp_fn(
                pipe_rf.transformer, prompt_embeds, pooled_prompt_embeds,
                text_ids, latent_image_ids, args.guidance_scale, args.vjp_chunk,
                device, sigma0, t0, z_lin, latent_shape,
            )
            print(f"  JIVE(J^T J) subspace, k={args.inject_n}, "
                  f"iters={args.power_iters}")
            U, _S = top_subspace_jtj(
                endpoint_fn, vjp_fn, z_lin, args.inject_n, latent_shape,
                n_iters=args.power_iters, fd_eps=args.fd_eps, device=device,
            )
        else:
            def endpoint(z_batch):
                return z_batch - sigma0 * velocity(z_batch, t0)

            U, _S = top_subspace(
                endpoint, z_lin, args.inject_n, latent_shape,
                n_iters=args.power_iters, fd_eps=args.fd_eps, device=device,
            )

        z_ref = inverted_latents.reshape(latent_shape).float()
        for eta in args.etas:
            if not args.skip_baseline:
                imgs = rf_inversion_sample(
                    pipes, inverted_latents, image_latents, latent_image_ids,
                    prompt_embeds, pooled_prompt_embeds, args, eta,
                    deltas=None, enable_sde=True,
                )
                acc_base[eta].add_source(imgs, src_01)
                if src_idx < args.save_sources:
                    save_source_samples(imgs, args.out_dir, src_name,
                                        f"baseline_eta_{eta:.2f}",
                                        args.save_samples_per_source)
            for nrm in args.inject_norms:
                deltas = build_deltas(U, z_ref, nrm, args, latent_shape, device)
                imgs = rf_inversion_sample(
                    pipes, inverted_latents, image_latents, latent_image_ids,
                    prompt_embeds, pooled_prompt_embeds, args, eta,
                    deltas=deltas, enable_sde=True,
                )
                acc_ours[(eta, nrm)].add_source(imgs, src_01)
                if src_idx < args.save_sources:
                    save_source_samples(
                        imgs, args.out_dir, src_name,
                        f"ours_eta_{eta:.2f}_norm_{nrm:.1f}",
                        args.save_samples_per_source)
                del deltas
        del U

        save_checkpoint(src_idx + 1)
        progress_name = ("progress.json" if args.results_name == "results.json"
                         else os.path.splitext(args.results_name)[0] + ".progress.json")
        with open(os.path.join(args.out_dir, progress_name), "w") as f:
            json.dump({"sources_done": src_idx + 1, "of": len(sources)}, f)

    baseline_results, proposed_results = [], []
    for eta in args.etas:
        if not args.skip_baseline:
            row = {"strength": eta, "eta": eta,
                   **acc_base[eta].finalize(ref_feats, args.kid_subset_size,
                                            args.kid_subsets)}
            print(f"[baseline eta={eta}] {json.dumps({k: v for k, v in row.items() if not k.endswith('_std')})}")
            baseline_results.append(row)
        for nrm in args.inject_norms:
            row = {"strength": eta, "eta": eta, "inject_norm": nrm,
                   **acc_ours[(eta, nrm)].finalize(
                       ref_feats, args.kid_subset_size, args.kid_subsets)}
            print(f"[ours eta={eta} norm={nrm}] {json.dumps({k: v for k, v in row.items() if not k.endswith('_std')})}")
            proposed_results.append(row)

    with open(results_path, "w") as f:
        json.dump({"dataset": args.dataset, "method": "rf_inverse",
                   "sweep_param": "eta", "args": vars(args),
                   "baseline": baseline_results,
                   "proposed": proposed_results}, f, indent=2)
    print(f"\nWrote {results_path}")

    canonical = os.path.join(args.out_dir, "results.json")
    if (os.path.abspath(results_path) != os.path.abspath(canonical)
            and os.path.isfile(canonical)):
        with open(canonical) as f:
            existing = json.load(f)
        existing_keys = {
            (r.get("eta"), r.get("inject_norm"))
            for r in existing.get("proposed", [])
        }
        appended = 0
        for row in proposed_results:
            key = (row.get("eta"), row.get("inject_norm"))
            if key not in existing_keys:
                existing.setdefault("proposed", []).append(row)
                appended += 1
        inj = list(existing.get("args", {}).get("inject_norms", []))
        for nrm in args.inject_norms:
            if nrm not in inj:
                inj.append(nrm)
        existing.setdefault("args", {})["inject_norms"] = inj
        with open(canonical, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Merged {appended} proposed row(s) into {canonical}")
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
