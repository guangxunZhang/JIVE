"""JIVE's subspace estimator and perturbation draw for the local-level arms,
written backbone-agnostically so all three arms share one implementation:

  * DDPM (SDEdit / Boomerang arms): endpoint is the x0-prediction
        D_t(x) = (x - sqrt(1 - abar_t) * eps_theta(x, t)) / sqrt(abar_t)
  * FLUX rectified flow (RF-Inversion arm): endpoint is
        D_t(z) = z - sigma_t * v_theta(z, t)

The caller supplies `endpoint_fn(z_batch) -> endpoint_batch` and nothing else
differs: the orthogonal iteration with forward finite differences plus
Rayleigh-Ritz below is the same for every arm.

Unlike the set-level estimator (set_level/jive_flux/jive_subspace.py), which
adds noise projected into the subspace at a fixed L2 norm, the local-level
arms need the injection to be NORM-PRESERVING -- the start latent encodes the
source image, so its magnitude must not drift. projected_noise_boundary
therefore rotates along span(U) on the sphere ||z|| = ||z_ref|| instead.
"""
import torch


@torch.no_grad()
def top_subspace(endpoint_fn, z_ref, n_vectors, latent_shape, n_iters=10,
                 fd_eps=4.0, device="cpu", verbose=True):
    """Top-k eigenvector subspace of the endpoint Jacobian J_D at z_ref.

    Parameters
    ----------
    endpoint_fn  : closure D(z_batch) -> endpoint batch (same shape as input).
                   Must accept a batch whose leading dim is k (the number of
                   probe directions) and be differentiable in the finite-
                   difference sense (chunk internally to cap VRAM).
    z_ref        : reference point, any shape reshapeable to latent_shape.
    n_vectors    : subspace dimension k.
    latent_shape : (1, ...) full latent shape (4D pixel/latent tensors and
                   packed (1, seq, ch) FLUX latents both work).

    Returns
    -------
    U : (D, k) orthonormal basis, sorted by |eigenvalue| descending
    S : (k,)   |eigenvalues| (singular values), sorted descending
    """
    D = z_ref.numel()
    chan_shape = latent_shape[1:]
    z0 = z_ref.reshape(latent_shape).to(device=device, dtype=torch.float32)

    curr_endpoint = endpoint_fn(z0).reshape(D)

    def jvp_block(Q_cols):
        """Forward finite-difference J_D applied to each column of Q_cols."""
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

    # Rayleigh-Ritz on the converged subspace
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


def projected_noise_boundary(U, z_ref, norm, latent_shape, device, seed=0,
                             S=None, power=0.0):
    """Norm-preserving injection: rotate z_ref ALONG span(U), staying on the
    sphere ||z|| = ||z_ref||. Returns a *delta* such that (z_ref + delta) lies
    on that sphere and ||delta|| == norm (chord length).
    """
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
