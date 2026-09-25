"""ARM D for FLUX: JIVE COMBINED with CADS.

Composition order (each mechanism does what it does in its standalone arm):

  1. "Perturb at beginning" -- exactly ARM C (jive_arm.JiveArmFlux): at the
     pure-noise starting point (sigma=1, before any denoising), each PACKED
     latent is pushed along the top singular directions of the Jacobian of
     the flow-matching endpoint predictor D_t(z) = z - sigma_t * v_theta(z, t),
     computed against the CLEAN prompt embeddings (JIVE's math is untouched:
     the Jacobian linearization is part of our method, so it is evaluated at
     the un-corrupted condition).

  2. "Then CADS" -- the CADS conditioning corruption of baselines/cads.py
     (Sadat et al., ICLR 2024, Eq. 1-4) is applied ON TOP during the same
     denoising trajectory: at every step the CLEAN T5/CLIP-pooled embeddings
     are re-corrupted with fresh annealed noise,

         y_hat = sqrt(gamma(t)) * y + s * sqrt(1 - gamma(t)) * n

     with gamma(t) the tau1/tau2 ramp and the psi rescale, imported from
     baselines/cads.py so the corruption math is byte-identical to the
     standalone CADS baseline and the two rows of the table cannot drift.

Where ARM C injects diversity once through the latent start and CADS
injects it per-step through the conditioning, ARM D does both, so the two
diversity sources add: different high-volume starting points, each denoised
under a differently-corrupted condition.

Batched-generation differences vs the standalone baseline (which generates
one image at a time): the runner denoises --G images per batch, so the
embeddings are EXPANDED to the batch and corrupted PER IMAGE (batch element
j of the batch starting at img_lo uses global image index img_lo + j),
giving every image its own conditioning-noise stream -- strictly more
diverse than sharing one corruption across the batch, and matching the
standalone baseline's per-image semantics.

Reproducibility: the CADS noise for (image i, step k) comes from a generator
seeded seed + CADS_SEED_OFFSET + i * CADS_STEP_STRIDE + k, so (a) it can
never consume draws from the latent seeds (seed + i) or the perturbation
seeds (seed + PERTURB_SEED_OFFSET + i), and (b) results are independent of
how images are chunked into batches -- the same invariant the runner's
per-image latent seeds guarantee.

diffusers-version note (mirrors the standalone baseline's situation on this
install): callback_on_step_end may only rewrite tensors in
FluxPipeline._callback_tensor_inputs = ["latents", "prompt_embeds"], so the
pooled CLIP projection is corrupted ONCE for step 0 (passed in corrupted)
while the T5 sequence is re-corrupted at every step through the callback.
That is exactly what baselines/cads.py's --cads-no-pooled flag calls a
"judgement call"; here it is the only thing the installed pipeline exposes,
and it is the same effective behaviour the standalone CADS baseline gets.
"""
import torch

from baselines.cads import cads_gamma, cads_corrupt  # noqa: F401

from .jive_arm import JiveArmFlux

CADS_SEED_OFFSET = 200_000
CADS_STEP_STRIDE = 64


class JiveCadsArmFlux(JiveArmFlux):
    """ARM C (JIVE start projection) + CADS (per-step conditioning noise).

    The start transform is inherited verbatim from JiveArmFlux; this subclass
    adds the CADS side: the step-0 corrupted embeddings handed to the
    pipeline call, and the per-step callback that re-corrupts the T5
    sequence for the NEXT step's sigma (the baseline's exact stepping scheme:
    callback_on_step_end fires at the END of a step, so it can only prepare
    the next one; step 0's conditioning is corrupted BEFORE the call at
    t = sigma_0 = 1.0).
    """

    def __init__(self, pipe, args, dev_tr, prompt_text, guidance_scale):
        super().__init__(pipe, args, dev_tr, prompt_text, guidance_scale)
        self.cads_s = float(args.cads_s)
        self.cads_tau1 = float(args.cads_tau1)
        self.cads_tau2 = float(args.cads_tau2)
        self.cads_psi = float(args.cads_psi)
        self.base_seed = None


    def _corrupt(self, y_clean, t, seed_val):
        """One CADS corruption of a SINGLE image's conditioning (y_clean is
        that image's (1, ...) slice), with the noise drawn from a generator
        seeded by seed_val -- per (image, step), see module docstring."""
        gen = torch.Generator(device=y_clean.device)
        gen.manual_seed(int(seed_val))
        return cads_corrupt(y_clean, t, self.cads_s, self.cads_tau1,
                            self.cads_tau2, self.cads_psi, gen)

    def _corrupt_pe(self, t, img_lo, batch_size, step_idx):
        """The (B, L, D) T5 embeddings for the batch at img_lo, corrupted at
        sigma t from the CLEAN embeddings, per image."""
        pe = self.pe.expand(batch_size, -1, -1)
        outs = [
            self._corrupt(pe[j:j + 1], t,
                          self.base_seed + CADS_SEED_OFFSET
                          + (img_lo + j) * CADS_STEP_STRIDE + step_idx)
            for j in range(batch_size)
        ]
        return torch.cat(outs, dim=0)

    def make_step0_embeds(self, img_lo, batch_size):
        """(prompt_embeds, pooled_prompt_embeds) for the batch at img_lo,
        corrupted at t = sigma_0 (= 1.0 at the pure-noise start), i.e. the
        conditioning the FIRST denoising step sees -- the pre-call corruption
        the standalone baseline applies because the callback can only
        prepare later steps. Step index 0 matches the callback's
        j = step + 1 numbering."""
        pe0 = self._corrupt_pe(self.sigma_init, img_lo, batch_size, 0)
        pooled = self.pooled.expand(batch_size, -1)
        ppe0 = torch.cat([
            self._corrupt(pooled[j:j + 1], self.sigma_init,
                          self.base_seed + 2 * CADS_SEED_OFFSET
                          + (img_lo + j) * CADS_STEP_STRIDE + 0)
            for j in range(batch_size)
        ], dim=0)
        return pe0, ppe0

    def make_cads_callback(self, img_lo):
        """callback_factory(img_lo) matching FluxArmRunner.run_arm: returns
        the per-batch diffusers callback that re-corrupts the T5 sequence
        for the next step's sigma (read off the live scheduler, exactly as
        baselines/cads.py does). The batch size is taken from the callback kwargs, so
        a short final batch needs no special casing."""
        def _cb(ppl, step, timestep, cb_kwargs):
            j = step + 1
            sig = ppl.scheduler.sigmas
            if j >= len(sig):
                return cb_kwargs
            t_next = float(sig[j])
            b = cb_kwargs["prompt_embeds"].shape[0]
            cb_kwargs["prompt_embeds"] = self._corrupt_pe(t_next, img_lo, b, j)
            return cb_kwargs
        return _cb

    def make_start_transform(self, inject_norm, seed_val):
        """Binds the run's base seed for the CADS stream, then defers to the
        inherited JIVE start transform (the 'perturb at beginning' half)."""
        self.base_seed = int(seed_val)
        return super().make_start_transform(inject_norm, seed_val)
