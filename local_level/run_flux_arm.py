"""SDEdit and Boomerang arms (baseline and +JIVE) on FLUX.

ALL SIX ARMS in this folder share the FLUX.1-dev backbone so cross-method
numbers (KID, Vendi, ...) are directly comparable. This script covers the
noise-and-denoise arms; run_rf_inverse.py covers the inversion-based arm.

Both papers' procedures adapted to rectified flow (x_sigma = (1-sigma) x0 +
sigma eps, sigma: 1 -> 0):

  SDEdit    (Meng et al., ICLR 2022): draw fresh forward noise per sample,
            z_start = (1-sigma_t0) z0 + sigma_t0 eps, then integrate the
            deterministic Euler ODE from sigma_t0 to 0 — the standard SDEdit /
            img2img adaptation for flow models (diversity comes from the
            forward draw). Sweep knob: strength t0 in [0.3, 0.6] (paper Fig. 3).
  Boomerang (Luzi et al., TMLR 2023): same closed-form forward noising, but
            the reverse process is STOCHASTIC at every step (paper Alg. 1 adds
            noise each ancestral step). We use the flow-matching SDE update
            from baselines/rf_inversion/scheduling_flow_match_euler_discrete_sde.py:
                drift = 2 v + z / (1 - sigma)
                diff  = sqrt(2 sigma / (1 - sigma) * (sigma - sigma_next))
                z    <- z + (sigma_next - sigma) drift + diff * noise
            Sweep knob: tBoom/T in {0.2 ... 0.5} (paper Sec. 4.1 picks 25-40%).

On the same backbone this reverse-process difference (ODE vs SDE) is exactly
what separates the two papers' samplers; everything else is shared.

+JIVE: per source, top-k subspace of the flow endpoint Jacobian
D(z) = z - sigma_t0 * v(z, t0) at the noised latent, then rotate each
per-sample noised latent inside that subspace with
projected_noise_boundary. Forward-noise and reverse-SDE seeds are shared with
the baseline, so inject_norm -> 0 reproduces the baseline exactly. (In the
results JSON and the sample directories this arm is still keyed "proposed" /
"ours_*"; see local_level/README.md.)

Run on the cluster via scripts/run_stroke2image.sh — not locally.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # JIVE/
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "baselines", "rf_inversion"))

from diffusers import FluxPipeline  # noqa: E402
from diffusers.training_utils import set_seed  # noqa: E402
from diffusers.pipelines.flux.pipeline_flux import (  # noqa: E402
    calculate_shift, retrieve_timesteps,
)
from pipeline_rf_inversion_sde import RFInversionFluxPipelineSDE  # noqa: E402

from rf_inversion_sampling import (  # noqa: E402
    DEFAULT_FLUX_MODEL, PERTURB_SEED_OFFSET, make_flux_velocity_fn,
)
from common.experiment import (  # noqa: E402
    PointAccumulator, load_sources, reference_inception_feats,
    save_source_samples,
)
from common.metrics import LocalLevelMetrics  # noqa: E402
from common.volume_expansion import (  # noqa: E402
    projected_noise_boundary, top_subspace,
)

DEFAULT_PROMPTS = {
    "bedroom": "a photo of a bedroom",
    "church": "a photo of a church",
}

# Seed offsets keep the independent noise roles from colliding; the
# perturbation offset is shared with the RF-Inversion arm via
# rf_inversion_sampling.PERTURB_SEED_OFFSET.
FWD_SEED_OFFSET = 10_000        # forward (interpolation) noise, per source
REV_SEED_OFFSET = 50_000        # reverse-SDE noise, per (source, chunk)
SUBSPACE_SEED_OFFSET = 200_000  # eps_ref of the subspace linearization point


def sweep_defaults(method):
    if method == "sdedit":
        return [0.3, 0.4, 0.5, 0.6]
    return [0.2, 0.3, 0.4, 0.5]


def _pipe_device(pipe):
    return next(pipe.transformer.parameters()).device


# ── schedule ──────────────────────────────────────────────────────────────────

def build_schedule(pipe, steps, image_seq_len, device):
    """FLUX shifted sigma schedule. Returns (timesteps (steps,), sigmas
    (steps+1,) ending at 0) as CPU floats."""
    sigmas_lin = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, None,
                                      sigmas_lin, mu=mu)
    sigmas = pipe.scheduler.sigmas.detach().float().cpu()
    return timesteps.detach().float().cpu(), sigmas


def start_index(steps, strength):
    """diffusers img2img convention: use the last round(steps*strength) steps."""
    init = min(int(round(steps * strength)), steps)
    return max(steps - init, 0)


# ── denoising (chunked across GPUs, mirrors rf_inversion_sample's pattern) ────

@torch.no_grad()
def _denoise_chunk(pipe, velocity, z_chunk_cpu, timesteps, sigmas, i0, method,
                   height, width, rev_seed):
    """Denoise one chunk from sigma_{i0} to 0 on pipe's device and decode.
    Reverse noise (Boomerang/SDE only) uses a generator seeded with rev_seed,
    which is identical for the baseline and +JIVE arms of the same chunk."""
    device = _pipe_device(pipe)
    dtype = next(pipe.transformer.parameters()).dtype
    z = z_chunk_cpu.to(device=device, dtype=torch.float32)
    gen = torch.Generator(device=device)
    gen.manual_seed(rev_seed)

    n_steps = len(timesteps)
    for i in range(i0, n_steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        v = velocity(z, float(timesteps[i]))
        if method == "sdedit" or sigma > 0.999:
            # deterministic Euler ODE step
            z = z + (sigma_next - sigma) * v
        else:
            # flow-matching SDE step (scheduling_flow_match_euler_discrete_sde)
            drift = 2.0 * v + z / (1.0 - sigma)
            diff = (2.0 * sigma / (1.0 - sigma) * (sigma - sigma_next)) ** 0.5
            noise = torch.randn(z.shape, device=device, dtype=z.dtype,
                                generator=gen)
            z = z + (sigma_next - sigma) * drift + diff * noise

    lat = pipe._unpack_latents(z.to(dtype), height, width,
                               pipe.vae_scale_factor)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    img = pipe.image_processor.postprocess(img, output_type="pt")
    return img.float().cpu().clamp(0, 1)


@torch.no_grad()
def denoise_batch(pipes, velocities, z_start_cpu, timesteps, sigmas, i0,
                  method, args, src_idx):
    """Fans chunks of the (n, seq, ch) start latents out across pipes."""
    starts = list(range(0, z_start_cpu.shape[0], args.batch_size))
    results = {}
    n_workers = len(pipes)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for wave in range(0, len(starts), n_workers):
            futs = []
            for i, start in enumerate(starts[wave:wave + n_workers]):
                chunk = z_start_cpu[start:start + args.batch_size]
                rev_seed = args.seed + REV_SEED_OFFSET + src_idx * 1000 + start
                futs.append((start, ex.submit(
                    _denoise_chunk, pipes[i], velocities[i], chunk, timesteps,
                    sigmas, i0, method, args.height, args.width, rev_seed)))
            for start, fut in futs:
                results[start] = fut.result()
    return torch.cat([results[s] for s in starts], dim=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["sdedit", "boomerang"], required=True)
    p.add_argument("--dataset", choices=["bedroom", "church"], required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--prompt", default=None)
    p.add_argument("--model", default=DEFAULT_FLUX_MODEL)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_sources", type=int, default=16)
    p.add_argument("--n", type=int, default=16, help="Samples per source")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--strengths", type=float, nargs="+", default=None,
                   help="t0 (SDEdit) / tBoom/T (Boomerang) sweep; defaults "
                        "to each paper's recommended range")
    # +JIVE (same latent space as the RF-Inversion arm -> same norm scale)
    p.add_argument("--inject_norms", type=float, nargs="+",
                   default=[4.0, 8.0, 12.0, 16.0])
    p.add_argument("--inject_n", type=int, default=4)
    p.add_argument("--power_iters", type=int, default=10)
    p.add_argument("--fd_eps", type=float, default=4.0,
                   help="Finite-difference step for the JVPs. FLUX runs in bf16 (ULP ~0.008 near 1), so smaller steps round away; keep in sync with set_level --fd-eps.")
    # perf / eval
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--fwd_chunk", type=int, default=2)
    p.add_argument("--metric_batch_size", type=int, default=32)
    p.add_argument("--kid_subset_size", type=int, default=250)
    p.add_argument("--kid_subsets", type=int, default=100)
    p.add_argument("--save_sources", type=int, default=8)
    p.add_argument("--save_samples_per_source", type=int, default=8)
    p.add_argument("--metric_resolution", type=int, default=256)
    args = p.parse_args()

    if args.strengths is None:
        args.strengths = sweep_defaults(args.method)
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS[args.dataset]
    if args.out_dir is None:
        args.out_dir = os.path.join("out", args.dataset, args.method)
    os.makedirs(args.out_dir, exist_ok=True)

    results_path = os.path.join(args.out_dir, "results.json")
    if os.path.isfile(results_path):
        print(f"SKIP: {results_path} already exists.")
        return

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = "cuda:0" if n_gpus > 0 else "cpu"
    model_dtype = torch.bfloat16 if n_gpus > 0 else torch.float32
    local_only = os.path.isdir(args.model)
    print(f"method={args.method} dataset={args.dataset} "
          f"prompt='{args.prompt}' device={device} n_gpus={n_gpus}")

    def _load_pipe(dev):
        pipe = FluxPipeline.from_pretrained(
            args.model, torch_dtype=model_dtype, local_files_only=local_only)
        pipe.to(device=dev, dtype=model_dtype)
        # diffusers' from_pipe defaults torch_dtype=float32 and finishes with
        # new_pipeline.to(dtype=that), silently upcasting everything to fp32
        # (fp32 VAE vs the bf16 tensors we feed it -> encode_image crashes).
        return RFInversionFluxPipelineSDE.from_pipe(pipe, torch_dtype=model_dtype)

    pipes = [_load_pipe(f"cuda:{i}" if n_gpus > 0 else "cpu")
             for i in range(max(1, n_gpus))]
    pipe0 = pipes[0]
    print(f"Loaded {len(pipes)} FLUX pipeline replica(s)")

    metrics = LocalLevelMetrics(device, batch_size=args.metric_batch_size)
    sources = load_sources(args.data_root, args.dataset, args.n_sources,
                           args.metric_resolution)
    ref_feats = reference_inception_feats(metrics, args.data_root,
                                          args.dataset, args.metric_resolution)

    print(f"Encoding prompt: \"{args.prompt}\"")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe0.encode_prompt(
        prompt=args.prompt, prompt_2=None, device=device,
        num_images_per_prompt=1, max_sequence_length=512,
    )

    acc_base = {s: PointAccumulator(metrics, args.metric_resolution)
                for s in args.strengths}
    acc_ours = {(s, nrm): PointAccumulator(metrics, args.metric_resolution)
                for s in args.strengths for nrm in args.inject_norms}

    velocities = None  # built after the first source (needs latent_image_ids)

    for src_idx, (src_name, src_01) in enumerate(sources):
        print(f"\n=== source {src_idx + 1}/{len(sources)}: {src_name} ===")
        img = T.ToPILImage()(src_01).resize((args.width, args.height),
                                            Image.LANCZOS)

        # clean packed latent z0 (deterministic given the seed) + ids
        set_seed(args.seed)
        image_latents, _ = pipe0.encode_image(img, dtype=model_dtype,
                                              height=args.height,
                                              width=args.width)
        z0, latent_image_ids = pipe0.prepare_latents_inversion(
            1, pipe0.transformer.config.in_channels // 4, args.height,
            args.width, model_dtype, device, image_latents,
        )
        z0 = z0.float()
        latent_shape = (1,) + tuple(z0.shape[1:])

        if velocities is None:
            # one velocity closure per pipe; latent_image_ids/text embeds are
            # positional and identical for every source at fixed resolution
            velocities = []
            for pipe in pipes:
                dev = _pipe_device(pipe)
                velocities.append(make_flux_velocity_fn(
                    pipe.transformer,
                    prompt_embeds.to(dev), pooled_prompt_embeds.to(dev),
                    text_ids.to(dev), latent_image_ids.to(dev),
                    args.guidance_scale, args.fwd_chunk, dev, model_dtype))
            timesteps, sigmas = build_schedule(pipe0, args.steps,
                                              z0.shape[1], device)

        for strength in args.strengths:
            i0 = start_index(args.steps, strength)
            sigma0 = float(sigmas[i0])
            t0_val = float(timesteps[i0])

            # per-sample forward noise, shared between baseline and +JIVE
            fwd_gen = torch.Generator(device=device)
            fwd_gen.manual_seed(args.seed + FWD_SEED_OFFSET + src_idx)
            eps = torch.randn((args.n, *z0.shape[1:]), device=device,
                              generator=fwd_gen)
            z_start = (1.0 - sigma0) * z0 + sigma0 * eps  # (n, seq, ch)

            # ── baseline ────────────────────────────────────────────────────
            imgs = denoise_batch(pipes, velocities, z_start.cpu(), timesteps,
                                 sigmas, i0, args.method, args, src_idx)
            acc_base[strength].add_source(imgs, src_01)
            if src_idx < args.save_sources:
                save_source_samples(imgs, args.out_dir, src_name,
                                    f"baseline_t0_{strength:.2f}",
                                    args.save_samples_per_source)

            # ── +JIVE: subspace once per (source, strength) ─────────────────
            sub_gen = torch.Generator(device=device)
            sub_gen.manual_seed(args.seed + SUBSPACE_SEED_OFFSET + src_idx)
            eps_ref = torch.randn(z0.shape, device=device, generator=sub_gen)
            z_ref = (1.0 - sigma0) * z0 + sigma0 * eps_ref

            def endpoint(z_batch):
                return z_batch - sigma0 * velocities[0](z_batch, t0_val)

            U, _S = top_subspace(endpoint, z_ref, args.inject_n, latent_shape,
                                 n_iters=args.power_iters, fd_eps=args.fd_eps,
                                 device=device, verbose=(src_idx == 0))

            for nrm in args.inject_norms:
                deltas = torch.cat([
                    projected_noise_boundary(
                        U, z_start[i:i + 1], nrm, latent_shape, device,
                        seed=args.seed + PERTURB_SEED_OFFSET + i)
                    for i in range(args.n)
                ], dim=0)
                imgs = denoise_batch(pipes, velocities,
                                     (z_start + deltas).cpu(), timesteps,
                                     sigmas, i0, args.method, args, src_idx)
                acc_ours[(strength, nrm)].add_source(imgs, src_01)
                if src_idx < args.save_sources:
                    save_source_samples(
                        imgs, args.out_dir, src_name,
                        f"ours_t0_{strength:.2f}_norm_{nrm:.1f}",
                        args.save_samples_per_source)
                del deltas
            del U
            torch.cuda.empty_cache()

        with open(os.path.join(args.out_dir, "progress.json"), "w") as f:
            json.dump({"sources_done": src_idx + 1, "of": len(sources)}, f)

    baseline_results, proposed_results = [], []
    for strength in args.strengths:
        row = {"strength": strength, "t0": strength,
               **acc_base[strength].finalize(ref_feats, args.kid_subset_size,
                                             args.kid_subsets)}
        print(f"[baseline t0={strength}] "
              f"{json.dumps({k: v for k, v in row.items() if not k.endswith('_std')})}")
        baseline_results.append(row)
        for nrm in args.inject_norms:
            row = {"strength": strength, "t0": strength, "inject_norm": nrm,
                   **acc_ours[(strength, nrm)].finalize(
                       ref_feats, args.kid_subset_size, args.kid_subsets)}
            print(f"[ours t0={strength} norm={nrm}] "
                  f"{json.dumps({k: v for k, v in row.items() if not k.endswith('_std')})}")
            proposed_results.append(row)

    with open(results_path, "w") as f:
        json.dump({"dataset": args.dataset, "method": args.method,
                   "sweep_param": "t0", "args": vars(args),
                   "baseline": baseline_results,
                   "proposed": proposed_results}, f, indent=2)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
