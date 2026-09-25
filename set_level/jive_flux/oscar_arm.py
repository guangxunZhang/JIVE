"""ARM B for FLUX: full per-step OSCAR volume perturbation, applied
batch-by-batch so the volume objective couples the images within each
generation batch of --G, exactly as OSCAR normally runs. The objective
itself is the authors' own code, vendored in baselines/oscar/.

FLUX-specifics: the pipeline's denoising loop (and therefore
kw["latents"]) works on PACKED latents (B, seq_len, 64), while the VAE and
the gradient math below act on the unpacked layout (B, 16, lat_h, lat_w).
The callback unpacks on entry and re-packs on exit; both are pure
permutations, so every L2 norm, trust region and budget number is identical
in either layout. State (previous latents / applied perturbation) is kept in
the UNPACKED layout. The VAE decode also applies FLUX's shift_factor.
"""
import math
import torch

from baselines.oscar.utils import project_partial_orth, batched_norm as _bn, sched_factor as time_sched_factor
from .runner import brownian_std_from_scheduler


class OscarArmFlux:
    """Builds a fresh per-batch OSCAR callback (state is reset per generation
    batch, since the volume objective couples only the images within a
    batch) and tracks the mean applied perturbation budget ||delta||_2 per
    image across the whole arm."""

    def __init__(self, pipe, vol, cfg, args, dev_vae, dev_clip):
        self.pipe = pipe
        self.vol = vol
        self.cfg = cfg
        self.args = args
        self.dev_vae = dev_vae
        self.dev_clip = dev_clip
        self.state = {}
        self.budget_parts = []

        vae_sf = getattr(pipe, "vae_scale_factor", 8)
        self.vae_sf = vae_sf
        self.num_ch = pipe.transformer.config.in_channels // 4  # 16
        self.lat_h = 2 * (int(args.height) // (vae_sf * 2))
        self.lat_w = 2 * (int(args.width) // (vae_sf * 2))

    def _reset_state(self):
        """Per-batch state carried between consecutive denoising steps of
        the SAME batch (never across batches, since each batch's latents
        are unrelated): the previous step's latents/perturbation (used to
        estimate the local velocity `v_est` and its timestep spacing), the
        running logdet used for the anti-oscillation gamma halving, and the
        per-image accumulated perturbation budget."""
        self.state.clear()
        self.state.update({
            "prev_latents_vae_cpu": None,
            "prev_ctrl_vae_cpu": None,
            "prev_dt_unit": None,
            "prev_prev_latents_vae_cpu": None,
            "last_logdet": None,
            "gamma_auto_done": False,
        })

    def _harvest_budget(self):
        # per-image sum of applied ||delta||_2 over a batch's gated steps;
        # collected before every state reset, plus once more at the end
        acc = self.state.get("budget_acc", None)
        if acc is not None:
            self.budget_parts.append(acc.clone())

    def callback_factory(self, img_lo):
        """Fresh OSCAR state for every generation batch; the volume
        objective couples the images within a batch."""
        self._harvest_budget()
        self._reset_state()
        return self._callback

    def finalize(self):
        """Call once after the arm's last batch to harvest its budget."""
        self._harvest_budget()

    @property
    def mean_budget(self):
        if not self.budget_parts:
            return 0.0
        return float(torch.cat(self.budget_parts).mean().item())

    def _callback(self, ppl, i, t, kw):
        """diffusers callback_on_step_end hook: runs after scheduler step i,
        perturbing kw["latents"] in place (returned via kw) before the next
        denoising step. Two phases per call: (1) a batch-wide volume-loss
        image-gradient pass, (2) a per-chunk VAE vector-Jacobian-product
        (VJP) pass that turns that image-space gradient into a latent-space
        displacement, adds trust-region-capped diversity noise, and applies
        both to the latents."""
        args, cfg, pipe, vol = self.args, self.cfg, self.pipe, self.vol
        dev_vae, dev_clip = self.dev_vae, self.dev_clip
        state = self.state

        # Normalize the current step's position in [0, 1] (t_norm) and its
        # spacing to the next step (dt_unit), both against the scheduler's
        # own timestep range so they are independent of --steps.
        ts = ppl.scheduler.timesteps
        t_cur  = float(ts[i].item())
        t_next = float(ts[i + 1].item()) if i + 1 < len(ts) else float(ts[-1].item())
        t_max, t_min = float(ts[0].item()), float(ts[-1].item())
        t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
        dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

        lat_packed = kw.get("latents")
        if lat_packed is None:
            return kw

        # Unpack (B, seq_len, 64) -> (B, 16, lat_h, lat_w); a pure
        # permutation, so norms are preserved exactly.
        lat = pipe._unpack_latents(lat_packed, args.height, args.width, self.vae_sf)

        # gamma_sched: this step's perturbation strength, gated to only fire
        # inside cfg.t_gate's [t0, t1] window (see baselines/oscar/utils.py).
        gamma_sched = cfg.gamma0 * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
        if args.debug:
            print(f"[DBG-GATE] i={i} t_norm={t_norm:.4f} gamma_sched={gamma_sched:.6f} "
                  f"lat_dtype={lat.dtype}")
        if gamma_sched <= 0:
            # Outside the gate window: no perturbation this step, just carry
            # the latents forward so v_est can still be estimated once we
            # re-enter the gate.
            state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
            state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
            state["prev_dt_unit"] = dt_unit
            return kw

        # accumulate in float32: bf16 ULP at ||lat||~480 is ~2, which
        # would round away the (relatively small) diversity perturbation.
        lat_new = lat.float().clone()

        lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
        B = lat_vae_full.size(0)
        chunk = max(1, min(int(args.vae_grad_chunk), B))

        prev_cpu = state.get("prev_latents_vae_cpu", None)
        sf_vae = getattr(pipe.vae.config, "scaling_factor", 1.0)
        shift_vae = getattr(pipe.vae.config, "shift_factor", None) or 0.0

        # ---- Phase 1: volume image-gradient over the FULL batch ----
        # Decode every image in the batch (no grad; VJP happens per-chunk in
        # phase 2) so vol.volume_loss_and_grad sees the WHOLE group at once
        # -- its log-det diversity objective couples all images in the batch.
        imgs_list = []
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
            for s in range(0, B, chunk):
                e = min(B, s + chunk)
                dec = (pipe.vae.decode(lat_vae_full[s:e] / sf_vae + shift_vae, return_dict=False)[0].float().clamp(-1, 1) + 1.0) / 2.0
                imgs_list.append(dec.to(dev_clip, non_blocking=True))
        imgs_all = torch.cat(imgs_list, dim=0)
        del imgs_list

        _loss, grad_img_all, _logs = vol.volume_loss_and_grad(imgs_all)
        # Anti-oscillation safeguard: if the batch's log-det diversity score
        # dropped since the last gated step, the previous step's move likely
        # overshot, so halve this step's strength before applying it.
        current_logdet = float(_logs.get("logdet", 0.0))
        last_logdet = state.get("last_logdet", None)
        if (last_logdet is not None) and (current_logdet < last_logdet):
            gamma_sched = 0.5 * gamma_sched
        state["last_logdet"] = current_logdet
        del imgs_all

        # ---- Phase 2: per-chunk VAE VJP + per-sample perturbation ----
        # Re-decode each chunk WITH grad enabled so autograd can pull
        # grad_img_all (an image-space gradient) back through the VAE
        # decoder into a latent-space gradient grad_lat -- i.e. compute the
        # vector-Jacobian product (VJP) of the decoder at this chunk.
        for s in range(0, B, chunk):
            e = min(B, s + chunk)
            z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                imgs_chunk = (pipe.vae.decode(z / sf_vae + shift_vae, return_dict=False)[0].float().clamp(-1, 1) + 1.0) / 2.0

            grad_img_vae = grad_img_all[s:e].to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

            grad_lat = torch.autograd.grad(
                outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
                retain_graph=False, create_graph=False, allow_unused=False
            )[0]

            # v_est: an estimate of the local flow-matching velocity from the
            # PREVIOUS step's total latent displacement, with that step's own
            # applied perturbation (prev_ctrl) subtracted out first so v_est
            # reflects the underlying denoising velocity, not our own noise.
            # Falls back to the raw (unadjusted) displacement on the first
            # gated step, when there is no prev_ctrl yet.
            v_est = None
            if prev_cpu is not None:
                total_diff = z - prev_cpu[s:e].to(dev_vae, non_blocking=True)
                prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
                prev_dt   = state.get("prev_dt_unit", None)
                if (prev_ctrl is not None) and (prev_dt is not None):
                    ctrl_prev = prev_ctrl[s:e].to(dev_vae, non_blocking=True)
                    base_move_prev = total_diff - ctrl_prev
                    v_est = base_move_prev / max(prev_dt, 1e-8)
                else:
                    v_est = total_diff / max(dt_unit, 1e-8)

            # Project the diversity gradient partially orthogonal to v_est so
            # the perturbation pushes images apart WITHOUT fighting the
            # denoising trajectory itself (partial_ortho in [0,1] trades off
            # how strictly orthogonal vs. how much raw gradient is kept).
            g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
            div_disp = g_proj * dt_unit  # gradient step scaled to this step's time increment

            brown_std = brownian_std_from_scheduler(ppl.scheduler, i)
            eta = float(args.eta_sde)
            # 'sde': a true Brownian increment has per-coordinate std
            # brown_std, i.e. total norm brown_std*sqrt(D); 'norm' keeps
            # the legacy target (total norm = brown_std). --rho
            # still caps the noise at rho*||base_disp|| per step.
            if args.oscar_noise_mode == 'sde':
                base_brown = eta * brown_std * math.sqrt(z[0].numel())
            else:
                base_brown = eta * brown_std

            rho_t = float(args.rho)

            # base_disp: the "natural" denoising displacement this step
            # would take without any perturbation, used as the reference
            # scale for both the noise target and the trust-region cap below.
            base_disp = (v_est * dt_unit) if v_est is not None else z
            base_norm = _bn(base_disp)

            # Per-image noise target norm = min(fixed Brownian target, an
            # SNR-relative cap of rho_t * ||base_disp||) -- whichever is
            # smaller, so noise never dominates the underlying denoising step.
            target_brown = torch.full_like(base_norm, fill_value=max(base_brown, 0.0))
            target_snr   = torch.clamp(base_norm * max(rho_t, 0.0), min=0.0)
            target = torch.minimum(target_brown, target_snr)

            # Random exploration noise, also partially-orthogonalized against
            # v_est (skipped if v_est is unreliable / near-zero, i.e. below
            # --vnorm-threshold, in which case raw isotropic noise is used).
            xi = torch.randn_like(g_proj)
            vnorm = _bn(v_est) if v_est is not None else None
            if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                xi_eff = xi
            else:
                xi_eff = project_partial_orth(xi, v_est, float(args.partial_ortho))

            xi_norm = _bn(xi_eff)
            noise_disp = xi_eff / (xi_norm.view(-1, 1, 1, 1) + 1e-12) * target.view(-1, 1, 1, 1)

            # Trust-region NORMALIZATION: rescale the diversity gradient step
            # so its norm never exceeds gamma_max_ratio * ||base_disp||,
            # keeping the perturbation a bounded fraction of the natural
            # denoising step regardless of how large the raw gradient is.
            disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
            div_raw   = _bn(div_disp)
            scale     = disp_cap / (div_raw + 1e-12)

            # Final per-step latent perturbation = scaled, gated diversity
            # gradient step + trust-region-capped exploration noise.
            delta_chunk = (gamma_sched * scale.view(-1, 1, 1, 1)) * div_disp + noise_disp
            delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
            lat_new[s:e] = lat_new[s:e] + delta_tr

            # Cached so the NEXT gated step can subtract it out of v_est
            # (see above) and so the total applied budget can be tallied.
            if "ctrl_cache" not in state:
                state["ctrl_cache"] = []
            state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

            # Per-image running sum of ||delta||_2 across every gated step in
            # this batch; averaged into `mean_budget` once the arm finishes.
            if "budget_acc" not in state:
                state["budget_acc"] = torch.zeros(B)
            state["budget_acc"][s:e] += _bn(delta_chunk).detach().flatten().float().cpu()

            if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
            if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
            del imgs_chunk, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

        del grad_img_all
        # Back to the pipeline's PACKED layout (the inverse permutation of
        # the unpack at entry).
        lat_new_packed = pipe._pack_latents(
            lat_new.to(lat_packed.dtype).contiguous(), B, self.num_ch, self.lat_h, self.lat_w,
        )
        kw["latents"] = lat_new_packed

        # Roll state forward for the next step of this same batch (kept in
        # the UNPACKED layout on CPU).
        state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
        state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")

        if "ctrl_cache" in state:
            state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
            del state["ctrl_cache"]
        state["prev_dt_unit"] = dt_unit
        return kw
