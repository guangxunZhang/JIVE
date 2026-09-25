"""The perturbation JIVE actually injects, once the high-volume subspace is
known: Gaussian noise projected into span(U) and rescaled to an exact target
L2 norm.

Shape- and device-agnostic, so the same draw serves the FLUX packed latents
of set_level and the 4D latents of local_level. The subspace estimators
themselves live next to the backbone they linearize
(set_level/jive_flux/jive_subspace.py for FLUX, local_level/common/
volume_expansion.py for the local-level arms).
"""
import torch

PERTURB_SEED_OFFSET = 100_000


def projected_noise_like(U, target_norm, shape, device, seed):
    """Gaussian noise projected into span(U), rescaled to target_norm.
    Uses an explicit torch.Generator rather than the global RNG, so a draw
    depends only on `seed` and never on how many images preceded it."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    z = torch.randn(U.shape[0], device=device, dtype=torch.float32, generator=gen)
    z_proj = U @ (U.t() @ z)
    z_proj = z_proj / z_proj.norm().clamp_min(1e-8) * float(target_norm)
    return z_proj.reshape(shape)
