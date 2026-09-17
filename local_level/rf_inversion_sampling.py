"""Shared FLUX sampling machinery for the local-level arms: the velocity
closure every arm linearizes, the batched RF-Inversion SDE sampler, and the
per-sample JIVE perturbation draw.

Split out of the drivers because run_flux_arm.py (SDEdit, Boomerang) and
run_rf_inverse.py (RF-Inversion) must agree on all three: the velocity
convention fixes what "the Jacobian" means, and the seed offsets are what make
inject_norm -> 0 reproduce the baseline sample-for-sample.

The subspace estimator itself is in common/volume_expansion.py (shared with
the SDEdit/Boomerang arms, which linearize a different endpoint).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from common.volume_expansion import projected_noise_boundary

# Local checkpoint directory or HF repo id; override with $FLUX_MODEL.
DEFAULT_FLUX_MODEL = os.environ.get(
    "FLUX_MODEL", "black-forest-labs/FLUX.1-dev",
)

# Seed offsets keep the independent noise roles from colliding.
SDE_SEED_OFFSET = 50_000       # reverse-SDE noise, per chunk
PERTURB_SEED_OFFSET = 100_000  # JIVE perturbation directions, per sample


# ── FLUX velocity closure ─────────────────────────────────────────────────────

def make_flux_velocity_fn(transformer, prompt_embeds, pooled_prompt_embeds,
                          text_ids, latent_image_ids, guidance_scale,
                          fwd_chunk, device, model_dtype):
    """Returns velocity(z_batch, t_val) -> raw transformer output (noise_pred),
    chunked over the batch to cap VRAM. z_batch is a PACKED FLUX latent
    (B, seq_len, channels) in float32; t_val is the raw scheduler timestep
    (the pipeline feeds the transformer t/1000).

    RF-Inversion denoises with z_next = z + (sigma_i - sigma_{i+1}) * (-noise_pred),
    so the flow endpoint is D_t(z) = z - sigma_t * noise_pred — the convention
    the subspace estimator assumes.
    """
    use_guidance = transformer.config.guidance_embeds
    # After pipe.to(device), some FLUX submodules can remain float32 while
    # others are bf16. Match the x_embedder weight dtype so Linear does not
    # see bf16 activations against fp32 weights (or vice versa).
    weight_dtype = transformer.x_embedder.weight.dtype

    @torch.no_grad()
    def velocity(z_batch, t_val):
        outs = []
        for s in range(0, z_batch.shape[0], fwd_chunk):
            zc = z_batch[s:s + fwd_chunk].to(dtype=weight_dtype)
            b = zc.shape[0]
            timestep = torch.full((b,), t_val / 1000.0, device=device, dtype=zc.dtype)
            guidance = (torch.full((b,), guidance_scale, device=device, dtype=torch.float32)
                        if use_guidance else None)
            out = transformer(
                hidden_states=zc,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds.expand(b, -1).to(dtype=weight_dtype),
                encoder_hidden_states=prompt_embeds.expand(b, -1, -1).to(dtype=weight_dtype),
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            outs.append(out.float())
        return torch.cat(outs, dim=0)

    return velocity


# ── batched RF-Inversion sampling ─────────────────────────────────────────────

def _pipe_device(pipe_rf):
    return next(pipe_rf.transformer.parameters()).device


@torch.no_grad()
def _rf_inversion_chunk(pipe_rf, z_inv_cpu, image_latents_cpu, latent_image_ids_cpu,
                        prompt_embeds_cpu, pooled_prompt_embeds_cpu, args, eta,
                        start, b, delta_chunk, enable_sde):
    """One batch chunk on the device owned by pipe_rf. SDE noise uses a
    per-chunk torch.Generator so multi-GPU threads do not race the global RNG."""
    device = _pipe_device(pipe_rf)
    dtype = next(pipe_rf.transformer.parameters()).dtype

    lat = z_inv_cpu.expand(b, -1, -1).clone()
    if delta_chunk is not None:
        lat = lat + delta_chunk.float()
    lat = lat.to(device=device, dtype=dtype)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + SDE_SEED_OFFSET + start)

    out = pipe_rf(
        prompt_embeds=prompt_embeds_cpu.expand(b, -1, -1).to(device=device, dtype=dtype),
        pooled_prompt_embeds=pooled_prompt_embeds_cpu.expand(b, -1).to(
            device=device, dtype=dtype),
        inverted_latents=lat,
        image_latents=image_latents_cpu.to(device=device, dtype=dtype),
        latent_image_ids=latent_image_ids_cpu.to(device=device),
        start_timestep=args.start_timestep,
        stop_timestep=args.stop_timestep,
        num_inference_steps=args.steps,
        eta=eta,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        enable_sde=enable_sde,
        generator=gen,
        output_type="pt",
    ).images
    return start, out.float().cpu()


@torch.no_grad()
def rf_inversion_sample(pipes, inverted_latents, image_latents, latent_image_ids,
                        prompt_embeds, pooled_prompt_embeds, args, eta,
                        deltas=None, enable_sde=True):
    """Generates args.n samples from the (optionally perturbed) inverted latent
    by chunked batched calls to the RF-Inversion SDE pipeline.

    pipes: list of RFInversionFluxPipelineSDE replicas (one per GPU). Chunks are
    farmed out across them in parallel when len(pipes) > 1.

    deltas: optional (n, seq_len, ch) float32 per-sample perturbations added to
    the inverted latent before denoising.

    SDE noise is seeded per chunk from (args.seed + SDE_SEED_OFFSET + chunk
    start), identically for baseline and JIVE runs, so with deltas=None vs.
    deltas=0 the two produce bit-identical samples.

    Returns images as a float tensor (n, 3, H, W) in [0, 1] on CPU.
    """
    z_inv_cpu = inverted_latents.detach().float().cpu()
    image_latents_cpu = image_latents.detach().cpu()
    latent_image_ids_cpu = latent_image_ids.detach().cpu()
    prompt_embeds_cpu = prompt_embeds.detach().cpu()
    pooled_prompt_embeds_cpu = pooled_prompt_embeds.detach().cpu()
    deltas_cpu = deltas.detach().float().cpu() if deltas is not None else None

    starts = list(range(0, args.n, args.batch_size))
    results = {}

    def _submit(ex, pipe, start):
        b = min(args.batch_size, args.n - start)
        delta_chunk = (deltas_cpu[start:start + b] if deltas_cpu is not None else None)
        return ex.submit(
            _rf_inversion_chunk, pipe, z_inv_cpu, image_latents_cpu,
            latent_image_ids_cpu, prompt_embeds_cpu, pooled_prompt_embeds_cpu,
            args, eta, start, b, delta_chunk, enable_sde,
        )

    n_workers = len(pipes)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for wave in range(0, len(starts), n_workers):
            wave_starts = starts[wave:wave + n_workers]
            futs = [
                _submit(ex, pipes[i], start)
                for i, start in enumerate(wave_starts)
            ]
            for fut in futs:
                start, imgs = fut.result()
                results[start] = imgs
            for pipe in pipes:
                torch.cuda.empty_cache()

    return torch.cat([results[s] for s in starts], dim=0).clamp(0, 1)


def build_deltas(U, z_ref, inject_norm, args, latent_shape, device):
    """Per-sample JIVE perturbations of the shared inverted latent: a
    norm-preserving rotation along span(U) (projected_noise_boundary), one
    independent direction seed per sample."""
    perturb_seeds = [args.seed + PERTURB_SEED_OFFSET + i for i in range(args.n)]
    deltas = [
        projected_noise_boundary(U, z_ref, inject_norm, latent_shape, device, seed=s)
        for s in perturb_seeds
    ]
    return torch.cat(deltas, dim=0)
