import torch


@torch.no_grad()
def top_subspace(endpoint_fn, z_ref, n_vectors, latent_shape, n_iters=10,
                 fd_eps=1e-1, device="cpu", verbose=True):
    D = z_ref.numel()
    chan_shape = latent_shape[1:]
    z0 = z_ref.reshape(latent_shape).to(device=device, dtype=torch.float32)

    curr_endpoint = endpoint_fn(z0).reshape(D)

    def jvp_block(Q_cols):
        k = Q_cols.shape[1]
        om = Q_cols.T.reshape(k, *chan_shape)
        om_norm = (om.reshape(k, -1).norm(dim=1).clamp(min=1e-12)
                   .reshape(k, *([1] * len(chan_shape))))
        om_hat = om / om_norm
        d_plus = endpoint_fn(z0 + fd_eps * om_hat).reshape(k, D)
        Y = (d_plus - curr_endpoint.unsqueeze(0)) / fd_eps
        return Y.T

    Q = torch.randn(D, n_vectors, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q, mode="reduced")

    for _ in range(n_iters):
        Y = jvp_block(Q)
        Q, _ = torch.linalg.qr(Y, mode="reduced")

    M = Q.T @ jvp_block(Q)
    M = 0.5 * (M + M.T)
    evals, evecs = torch.linalg.eigh(M)

    order = torch.argsort(evals.abs(), descending=True)
    evals = evals[order]
    U = Q @ evecs[:, order]
    S = evals.abs()

    if verbose:
        n_above_one = (evals.abs() > 1).sum().item()
        print(f"  eigvals (signed): {[f'{e:+.2f}' for e in evals.tolist()]}")
        print(f"  singular values : {[f'{s:.2f}' for s in S.tolist()]}")
        print(f"  |eigval| > 1     : {n_above_one}/{len(evals)}")
    return U, S


def top_subspace_jtj(endpoint_fn, vjp_fn, z_ref, n_vectors, latent_shape,
                     n_iters=10, fd_eps=1e-1, device="cpu", verbose=True):
    D = z_ref.numel()
    chan_shape = latent_shape[1:]
    z0 = z_ref.reshape(latent_shape).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        curr_endpoint = endpoint_fn(z0).reshape(D)

    def jvp_block(Q_cols):
        k = Q_cols.shape[1]
        om = Q_cols.T.reshape(k, *chan_shape)
        om_norm = (om.reshape(k, -1).norm(dim=1).clamp(min=1e-12)
                   .reshape(k, *([1] * len(chan_shape))))
        om_hat = om / om_norm
        with torch.no_grad():
            d_plus = endpoint_fn(z0 + fd_eps * om_hat).reshape(k, D)
            Y = (d_plus - curr_endpoint.unsqueeze(0)) / fd_eps
        return Y.T

    Q = torch.randn(D, n_vectors, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q, mode="reduced")

    for _ in range(n_iters):
        W = jvp_block(Q)
        Y = vjp_fn(W)
        Q, _ = torch.linalg.qr(Y, mode="reduced")

    W = jvp_block(Q)
    Y = vjp_fn(W)
    M = Q.T @ Y
    M = 0.5 * (M + M.T)
    evals, evecs = torch.linalg.eigh(M)

    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    U = Q @ evecs[:, order]
    S = evals.clamp_min(0).sqrt()

    if verbose:
        print(f"  [jtj] eigvals of J^T J : {[f'{e:.2f}' for e in evals.tolist()]}")
        print(f"  [jtj] singular values  : {[f'{s:.2f}' for s in S.tolist()]}")
    return U, S


def projected_noise_like(U, target_norm, shape, device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    z = torch.randn(U.shape[0], device=device, dtype=torch.float32, generator=gen)
    z_proj = U @ (U.t() @ z)
    z_proj = z_proj / z_proj.norm().clamp_min(1e-8) * float(target_norm)
    return z_proj.reshape(shape)


def projected_noise_boundary(U, z_ref, norm, latent_shape, device, seed=0,
                             S=None, power=0.0):
    torch.manual_seed(seed)
    D = U.shape[0]
    z0 = z_ref.reshape(D).to(U.dtype)
    R = z0.norm().clamp_min(1e-8)
    z_hat = z0 / R

    k = U.shape[1]
    coeffs = torch.randn(k, device=device, dtype=U.dtype)
    if S is not None and power != 0.0:
        coeffs = coeffs * (S / (S.max() + 1e-12)) ** power
    d = U @ coeffs

    d = d - (d @ z_hat) * z_hat
    d_hat = d / d.norm().clamp_min(1e-8)

    ratio = torch.clamp(torch.as_tensor(norm, device=device, dtype=U.dtype)
                        / (2.0 * R), max=1.0)
    theta = 2.0 * torch.asin(ratio)

    z_new = R * (torch.cos(theta) * z_hat + torch.sin(theta) * d_hat)
    return (z_new - z0).reshape(latent_shape)
