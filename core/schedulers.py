"""Scheduler-derived noise scale used by the OSCAR arm: the per-coordinate
std of the Brownian increment the flow-matching SDE would inject between two
consecutive denoising steps.

Read off the scheduler's own schedule rather than hard-coded, so the same
function serves the FLUX flow-matching sigmas and DDPM-style alphas_cumprod.
"""


def brownian_std_from_scheduler(scheduler, i):
    """Per-coordinate std of the Brownian increment the flow-matching SDE
    would inject between step i and i+1, derived from the scheduler's own
    sigma (or alpha) schedule; falls back to a timestep-spacing heuristic if
    neither is exposed."""
    sch = scheduler
    try:
        if hasattr(sch, "sigmas"):
            s = sch.sigmas.float()
            cur = s[i].item()
            nxt = s[i + 1].item() if i + 1 < len(s) else s[i].item()
            var = max(cur**2 - nxt**2, 0.0)
            return float(var**0.5)
        elif hasattr(sch, "alphas_cumprod"):
            ac = sch.alphas_cumprod.float()
            cur = ac[i].item()
            nxt = ac[i + 1].item() if i + 1 < len(ac) else ac[i].item()
            sig2_cur = max(1.0 - cur, 0.0)
            sig2_nxt = max(1.0 - nxt, 0.0)
            var = max(sig2_cur - sig2_nxt, 0.0)
            return float(var**0.5)
    except Exception:
        pass
    ts = sch.timesteps
    t_cur = float(ts[i].item())
    t_next = float(ts[i + 1].item()) if i + 1 < len(ts) else float(ts[-1].item())
    t_max, t_min = float(ts[0].item()), float(ts[-1].item())
    dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)
    return float(dt_unit**0.5)
