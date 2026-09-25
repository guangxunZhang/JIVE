"""Generic per-arm image-generation loop for FLUX.1-dev: regenerates the
shared per-image starting latents for every arm (so batch composition never
changes them), optionally transforms them before denoising (ARM C) or
perturbs them during denoising via a callback (ARM B), then decodes to CPU
images.

FLUX difference vs the SD3.5 runner: the pipeline works on PACKED latents
(B, seq_len, 64) with distilled guidance. Latents are sampled per image in
the UNPACKED layout (1, 16, lat_h, lat_w) -- matching what
FluxPipeline.prepare_latents would draw before packing -- and packed with
FluxPipeline._pack_latents right before the denoise call. Decoding reverses
this: unpack, then (z / scaling_factor) + shift_factor through the VAE.
"""
import torch

from baselines.oscar.utils import log as _log
from core.schedulers import brownian_std_from_scheduler  # noqa: F401


class FluxArmRunner:
    """Regenerates the shared pure-noise starting latents for a single
    (prompt, guidance, seed) run and drives any arm's generation loop."""

    def __init__(self, pipe, args, dev_tr, dev_vae, dtype, prompt_text, guidance, seed):
        self.pipe = pipe
        self.args = args
        self.dev_tr = dev_tr
        self.dev_vae = dev_vae
        self.dtype = dtype
        self.prompt_text = prompt_text
        self.guidance = guidance
        self.seed = seed

        vae_sf = getattr(pipe, "vae_scale_factor", 8)
        self.num_ch = pipe.transformer.config.in_channels // 4
        self.lat_h = 2 * (int(args.height) // (vae_sf * 2))
        self.lat_w = 2 * (int(args.width) // (vae_sf * 2))
        self.n_total = int(args.n_images)

    def _new_generator(self, seed_val):
        """A fresh, deterministically-seeded generator on the transformer's
        device (CUDA generators are per-device, so this must match dev_tr)."""
        gg = torch.Generator(device=self.dev_tr) if self.dev_tr.type == 'cuda' else torch.Generator()
        gg.manual_seed(int(seed_val))
        return gg

    def make_batch_latents(self, img_lo, img_hi):
        """One latent PER IMAGE, seeded with (self.seed + image_index), so
        every arm regenerates the identical starting points regardless of
        batch composition. Returned in the PACKED layout the FLUX pipeline
        expects: (B, (lat_h/2)*(lat_w/2), num_ch*4)."""
        with torch.no_grad():
            outs = []
            for img_idx in range(img_lo, img_hi):
                outs.append(torch.randn(
                    1, self.num_ch, self.lat_h, self.lat_w,
                    device=self.dev_tr, dtype=self.dtype,
                    generator=self._new_generator(self.seed + img_idx),
                ))
            lat = torch.cat(outs, dim=0)
            return self.pipe._pack_latents(lat, lat.size(0), self.num_ch, self.lat_h, self.lat_w)

    def decode(self, latents_out):
        """Unpack + VAE-decode a batch of final PACKED latents -> CPU fp16
        images in [0,1]."""
        unpacked = self.pipe._unpack_latents(
            latents_out, self.args.height, self.args.width,
            getattr(self.pipe, "vae_scale_factor", 8),
        )
        sf = getattr(self.pipe.vae.config, "scaling_factor", 1.0)
        shift = getattr(self.pipe.vae.config, "shift_factor", None) or 0.0
        latents_final = unpacked.to(self.dev_vae, non_blocking=True)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
            imgs = (self.pipe.vae.decode(latents_final / sf + shift, return_dict=False)[0]
                    .float().clamp(-1, 1) + 1.0) / 2.0
        return imgs.detach().to("cpu", torch.float16)

    def _denoise(self, callback, latents_in, prompt_embeds=None,
                 pooled_prompt_embeds=None, callback_tensor_inputs=None):
        """Runs the FLUX pipeline's own denoising loop with output_type
        'latent' (no VAE decode here); `callback` is diffusers'
        callback_on_step_end hook, invoked after every scheduler step so ARM
        B/D can inject its per-step perturbation. latents_in must be PACKED.

        When prompt_embeds/pooled_prompt_embeds are given they are fed to the
        pipeline INSTEAD of the raw prompt text (one batch element per image,
        so num_images_per_prompt is 1 and batch_size == latents batch) -- ARM
        D needs this to hand CADS-corrupted conditioning to step 0, and pairs
        it with callback_tensor_inputs containing "prompt_embeds" so the
        callback can re-corrupt the conditioning for the later steps."""
        true_cfg = float(getattr(self.args, "true_cfg_scale", 1.0)) > 1.0
        use_embeds = prompt_embeds is not None
        kw = dict(
            negative_prompt=(self.args.negative if (true_cfg and self.args.negative) else None),
            true_cfg_scale=float(getattr(self.args, "true_cfg_scale", 1.0)),
            height=self.args.height, width=self.args.width,
            num_images_per_prompt=(1 if use_embeds else latents_in.shape[0]),
            num_inference_steps=self.args.steps,
            guidance_scale=float(self.guidance),
            latents=latents_in,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=(callback_tensor_inputs or ["latents"]),
            output_type="latent",
            return_dict=False,
        )
        if use_embeds:
            kw["prompt_embeds"] = prompt_embeds
            kw["pooled_prompt_embeds"] = pooled_prompt_embeds
        else:
            kw["prompt"] = self.prompt_text
        return self.pipe(**kw)[0]

    def run_arm(self, arm_label, callback_factory=None, latents_transform=None,
                embeds_factory=None, callback_tensor_inputs=None):
        """Generates n_total images in batches of --G, sharing the per-image
        starting latents across arms.

        latents_transform(latents, img_lo) -> latents: applied to the
        pure-noise starting latents BEFORE denoising (ARM C's one-shot
        subspace projection); latents are PACKED.
        callback_factory(img_lo) -> per-batch callback applied DURING
        denoising (ARM B's per-step OSCAR perturbation, ARM D's CADS
        conditioning corruption).
        embeds_factory(img_lo, batch_size) -> (prompt_embeds,
        pooled_prompt_embeds): when given, the batch is denoised from these
        embeddings instead of the raw prompt text (ARM D's CADS-corrupted
        step-0 conditioning).
        Everything else is plain denoising + decode.
        """
        all_imgs = []
        for img_lo in range(0, self.n_total, self.args.G):
            img_hi = min(self.n_total, img_lo + self.args.G)
            lat_in = self.make_batch_latents(img_lo, img_hi)
            if latents_transform is not None:
                lat_in = latents_transform(lat_in, img_lo)
            cb = callback_factory(img_lo) if callback_factory is not None else None
            pe = ppe = None
            if embeds_factory is not None:
                pe, ppe = embeds_factory(img_lo, img_hi - img_lo)
            lat = self._denoise(cb, lat_in, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                                callback_tensor_inputs=callback_tensor_inputs)
            all_imgs.append(self.decode(lat))
            del lat, lat_in
            if self.dev_tr.type == 'cuda':
                torch.cuda.empty_cache()
            _log(f"[{arm_label}] images {img_hi}/{self.n_total} done", True)
        return torch.cat(all_imgs, dim=0)
