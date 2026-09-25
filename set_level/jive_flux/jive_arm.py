import numpy as np
import torch

from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from .jive_subspace import (
    top_singular_subspace_j, top_singular_subspace_jtj,
    projected_noise_like, PERTURB_SEED_OFFSET,
)


class JiveArmFlux:

    def __init__(self, pipe, args, dev_tr, prompt_text, guidance_scale):
        self.pipe = pipe
        self.args = args
        self.guidance_scale = float(guidance_scale)

        with torch.no_grad():
            self.pe, self.pooled, self.text_ids = pipe.encode_prompt(
                prompt=prompt_text, prompt_2=None,
                device=dev_tr, num_images_per_prompt=1,
                max_sequence_length=512,
            )

        vae_sf = getattr(pipe, "vae_scale_factor", 8)
        lat_h = 2 * (int(args.height) // (vae_sf * 2))
        lat_w = 2 * (int(args.width) // (vae_sf * 2))
        self.img_ids = pipe._prepare_latent_image_ids(
            1, lat_h // 2, lat_w // 2, dev_tr, self.pe.dtype,
        )

        image_seq_len = (lat_h // 2) * (lat_w // 2)
        sigmas_lin = np.linspace(1.0, 1.0 / int(args.steps), int(args.steps))
        if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
            sigmas_lin = None
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, int(args.steps), dev_tr, sigmas=sigmas_lin, mu=mu,
        )
        self.t_init = timesteps[0].float()
        self.sigma_init = float(pipe.scheduler.sigmas[0].item())

    def make_start_transform(self, inject_norm, seed_val):
        def transform(latents, img_lo, _norm=float(inject_norm)):
            return self._make_projected_start(latents, _norm, seed_val, img_lo)
        return transform

    def _make_projected_start(self, latents, inject_norm, seed_val, img_lo):
        args = self.args
        lat_new = latents.clone()
        for j in range(latents.size(0)):
            if getattr(args, "jive_iter_mode", "j") == "jtj":
                U, S = top_singular_subspace_jtj(
                    self.pipe.transformer, latents[j], self.t_init, self.sigma_init,
                    self.pe, self.pooled, self.text_ids, self.img_ids,
                    self.guidance_scale,
                    n_vectors=args.inject_n, n_iters=args.jive_iters,
                    fd_eps=args.fd_eps, fwd_chunk=args.fwd_chunk,
                    vjp_chunk=getattr(args, "jive_vjp_chunk", 2),
                    dtype=latents.dtype,
                    joint_attention_kwargs=None,
                    debug=args.debug,
                )
            else:
                U, S = top_singular_subspace_j(
                    self.pipe.transformer, latents[j], self.t_init, self.sigma_init,
                    self.pe, self.pooled, self.text_ids, self.img_ids,
                    self.guidance_scale,
                    n_vectors=args.inject_n, n_iters=args.jive_iters,
                    fd_eps=args.fd_eps, fwd_chunk=args.fwd_chunk,
                    dtype=latents.dtype,
                    joint_attention_kwargs=None,
                    debug=args.debug,
                )
            delta = projected_noise_like(
                U, inject_norm, latents[j:j+1].shape, latents.device,
                seed=seed_val + PERTURB_SEED_OFFSET + img_lo + j,
            )
            lat_new[j:j+1] = (lat_new[j:j+1].float() + delta).to(latents.dtype)
            del U, S, delta
            if latents.device.type == 'cuda':
                torch.cuda.empty_cache()
        return lat_new

    def close(self):
        del self.pe, self.pooled, self.text_ids, self.img_ids
