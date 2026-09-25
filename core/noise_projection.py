import torch

PERTURB_SEED_OFFSET = 100_000


def projected_noise_like(U, target_norm, shape, device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    z = torch.randn(U.shape[0], device=device, dtype=torch.float32, generator=gen)
    z_proj = U @ (U.t() @ z)
    z_proj = z_proj / z_proj.norm().clamp_min(1e-8) * float(target_norm)
    return z_proj.reshape(shape)
