import torch

from baselines.cads import cads_gamma, cads_corrupt

from .jive_arm import JiveArmFlux

CADS_SEED_OFFSET = 200_000
CADS_STEP_STRIDE = 64


class JiveCadsArmFlux(JiveArmFlux):

    def __init__(self, pipe, args, dev_tr, prompt_text, guidance_scale):
        super().__init__(pipe, args, dev_tr, prompt_text, guidance_scale)
        self.cads_s = float(args.cads_s)
        self.cads_tau1 = float(args.cads_tau1)
        self.cads_tau2 = float(args.cads_tau2)
        self.cads_psi = float(args.cads_psi)
        self.base_seed = None


    def _corrupt(self, y_clean, t, seed_val):
        gen = torch.Generator(device=y_clean.device)
        gen.manual_seed(int(seed_val))
        return cads_corrupt(y_clean, t, self.cads_s, self.cads_tau1,
                            self.cads_tau2, self.cads_psi, gen)

    def _corrupt_pe(self, t, img_lo, batch_size, step_idx):
        pe = self.pe.expand(batch_size, -1, -1)
        outs = [
            self._corrupt(pe[j:j + 1], t,
                          self.base_seed + CADS_SEED_OFFSET
                          + (img_lo + j) * CADS_STEP_STRIDE + step_idx)
            for j in range(batch_size)
        ]
        return torch.cat(outs, dim=0)

    def make_step0_embeds(self, img_lo, batch_size):
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
        self.base_seed = int(seed_val)
        return super().make_start_transform(inject_norm, seed_val)
