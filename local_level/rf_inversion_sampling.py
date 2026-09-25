import os
from concurrent.futures import ThreadPoolExecutor

import torch

from common.volume_expansion import projected_noise_boundary, projected_noise_like

DEFAULT_FLUX_MODEL = os.environ.get(
    "FLUX_MODEL", "black-forest-labs/FLUX.1-dev",
)

SDE_SEED_OFFSET = 50_000
PERTURB_SEED_OFFSET = 100_000



def make_flux_velocity_fn(transformer, prompt_embeds, pooled_prompt_embeds,
                          text_ids, latent_image_ids, guidance_scale,
                          fwd_chunk, device, model_dtype):
    use_guidance = transformer.config.guidance_embeds
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



def _pipe_device(pipe_rf):
    return next(pipe_rf.transformer.parameters()).device


@torch.no_grad()
def _rf_inversion_chunk(pipe_rf, z_inv_cpu, image_latents_cpu, latent_image_ids_cpu,
                        prompt_embeds_cpu, pooled_prompt_embeds_cpu, args, eta,
                        start, b, delta_chunk, enable_sde):
    device = _pipe_device(pipe_rf)
    dtype = next(pipe_rf.transformer.parameters()).dtype

    lat = z_inv_cpu.expand(b, -1, -1).clone()
    jive_delta = None
    jive_after_eta = bool(getattr(args, "inject_after_eta", False))
    if delta_chunk is not None:
        if jive_after_eta:
            jive_delta = delta_chunk.float().to(device=device, dtype=dtype)
        else:
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
        jive_delta=jive_delta,
        jive_after_eta=jive_after_eta,
    ).images
    return start, out.float().cpu()


@torch.no_grad()
def rf_inversion_sample(pipes, inverted_latents, image_latents, latent_image_ids,
                        prompt_embeds, pooled_prompt_embeds, args, eta,
                        deltas=None, enable_sde=True):
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
    seeds = [args.seed + PERTURB_SEED_OFFSET + i for i in range(args.n)]
    if getattr(args, "perturb_mode", "additive") == "additive":
        deltas = [
            projected_noise_like(U, inject_norm, latent_shape, device, seed=s)
            for s in seeds
        ]
    else:
        deltas = [
            projected_noise_boundary(
                U, z_ref, inject_norm, latent_shape, device, seed=s)
            for s in seeds
        ]
    return torch.cat(deltas, dim=0)
