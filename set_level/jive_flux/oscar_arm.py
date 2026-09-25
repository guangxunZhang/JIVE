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
        self.num_ch = pipe.transformer.config.in_channels // 4
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

        ts = ppl.scheduler.timesteps
        t_cur  = float(ts[i].item())
        t_next = float(ts[i + 1].item()) if i + 1 < len(ts) else float(ts[-1].item())
        t_max, t_min = float(ts[0].item()), float(ts[-1].item())
        t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
        dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

        lat_packed = kw.get("latents")
        if lat_packed is None:
            return kw

        lat = pipe._unpack_latents(lat_packed, args.height, args.width, self.vae_sf)

        gamma_sched = cfg.gamma0 * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
        if args.debug:
            print(f"[DBG-GATE] i={i} t_norm={t_norm:.4f} gamma_sched={gamma_sched:.6f} "
                  f"lat_dtype={lat.dtype}")
        if gamma_sched <= 0:
            state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
            state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
            state["prev_dt_unit"] = dt_unit
            return kw

        lat_new = lat.float().clone()

        lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
        B = lat_vae_full.size(0)
        chunk = max(1, min(int(args.vae_grad_chunk), B))

        prev_cpu = state.get("prev_latents_vae_cpu", None)
        sf_vae = getattr(pipe.vae.config, "scaling_factor", 1.0)
        shift_vae = getattr(pipe.vae.config, "shift_factor", None) or 0.0

        imgs_list = []
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
            for s in range(0, B, chunk):
                e = min(B, s + chunk)
                dec = (pipe.vae.decode(lat_vae_full[s:e] / sf_vae + shift_vae, return_dict=False)[0].float().clamp(-1, 1) + 1.0) / 2.0
                imgs_list.append(dec.to(dev_clip, non_blocking=True))
        imgs_all = torch.cat(imgs_list, dim=0)
        del imgs_list

        _loss, grad_img_all, _logs = vol.volume_loss_and_grad(imgs_all)
        current_logdet = float(_logs.get("logdet", 0.0))
        last_logdet = state.get("last_logdet", None)
        if (last_logdet is not None) and (current_logdet < last_logdet):
            gamma_sched = 0.5 * gamma_sched
        state["last_logdet"] = current_logdet
        del imgs_all

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

            g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
            div_disp = g_proj * dt_unit

            brown_std = brownian_std_from_scheduler(ppl.scheduler, i)
            eta = float(args.eta_sde)
            if args.oscar_noise_mode == 'sde':
                base_brown = eta * brown_std * math.sqrt(z[0].numel())
            else:
                base_brown = eta * brown_std

            rho_t = float(args.rho)

            base_disp = (v_est * dt_unit) if v_est is not None else z
            base_norm = _bn(base_disp)

            target_brown = torch.full_like(base_norm, fill_value=max(base_brown, 0.0))
            target_snr   = torch.clamp(base_norm * max(rho_t, 0.0), min=0.0)
            target = torch.minimum(target_brown, target_snr)

            xi = torch.randn_like(g_proj)
            vnorm = _bn(v_est) if v_est is not None else None
            if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                xi_eff = xi
            else:
                xi_eff = project_partial_orth(xi, v_est, float(args.partial_ortho))

            xi_norm = _bn(xi_eff)
            noise_disp = xi_eff / (xi_norm.view(-1, 1, 1, 1) + 1e-12) * target.view(-1, 1, 1, 1)

            disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
            div_raw   = _bn(div_disp)
            scale     = disp_cap / (div_raw + 1e-12)

            delta_chunk = (gamma_sched * scale.view(-1, 1, 1, 1)) * div_disp + noise_disp
            delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
            lat_new[s:e] = lat_new[s:e] + delta_tr

            if "ctrl_cache" not in state:
                state["ctrl_cache"] = []
            state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

            if "budget_acc" not in state:
                state["budget_acc"] = torch.zeros(B)
            state["budget_acc"][s:e] += _bn(delta_chunk).detach().flatten().float().cpu()

            if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
            if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
            del imgs_chunk, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

        del grad_img_all
        lat_new_packed = pipe._pack_latents(
            lat_new.to(lat_packed.dtype).contiguous(), B, self.num_ch, self.lat_h, self.lat_w,
        )
        kw["latents"] = lat_new_packed

        state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
        state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")

        if "ctrl_cache" in state:
            state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
            del state["ctrl_cache"]
        state["prev_dt_unit"] = dt_unit
        return kw
