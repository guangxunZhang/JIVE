"""JIVE's subspace estimator for FLUX: the top singular subspace of the
endpoint Jacobian, found by subspace iteration with forward
finite-difference matvecs on the flow-matching endpoint predictor
D_t(z) = z - sigma_t * v_theta(z, t).

Two operators are available, selected by --jive-iter-mode:
  j     Q <- QR(J Q)       block power method on J itself
  jtj   Q <- QR(J^T J Q)   block power method on the symmetric PSD J^T J,
                           whose eigenvectors ARE J's right singular vectors
See the two functions below for why the second is the one that actually
targets the singular subspace.

FLUX-specifics: z lives in the PACKED latent layout (1, seq_len, 64) -- the
reshape logic is shape-agnostic, so nothing else changes. The transformer
call takes distilled guidance (a per-batch tensor), the T5/CLIP text
embeddings plus their rotary ids (txt_ids), and the packed latent rotary ids
(img_ids); there is no classifier-free negative branch (FLUX.1-dev folds
guidance into the distilled embedding), so the endpoint is single-branch.
The k finite-difference probes of one iteration are BATCHED through the
transformer (chunked by fwd_chunk to cap VRAM); each probe is an independent
forward, so this is numerically identical to probing one column at a time.

The projected-noise draw applied once the subspace is known lives in
core/noise_projection.py (it is backbone-independent).
"""
import torch

from core.noise_projection import (  # noqa: F401  (re-exported for jive_arm)
    projected_noise_like, PERTURB_SEED_OFFSET,
)


@torch.no_grad()
def top_singular_subspace_j(transformer, z_ref, t_val, sigma,
                            prompt_embeds, pooled_embeds, text_ids, img_ids,
                            guidance_scale, n_vectors=4, n_iters=10, fd_eps=4.0,
                            fwd_chunk=4, dtype=torch.bfloat16,
                            joint_attention_kwargs=None, debug=False):
    """Top-n left singular vectors of the Jacobian of the FLUX flow-matching
    endpoint predictor D_t(z) = z - sigma_t * v_theta(z, t), evaluated at the
    PACKED latent z_ref, via forward finite-difference subspace iteration.

    Parameters
    ----------
    z_ref         : (seq_len, ch) or (1, seq_len, ch) PACKED latent to
                    linearize around
    t_val         : scalar timestep (scheduler units, e.g. 1000 at sigma=1);
                    the transformer is fed t_val/1000, matching the pipeline
    sigma         : float flow-matching sigma at that timestep (1.0 at the
                    pure-noise start)
    prompt_embeds : (1, L, D_txt) T5 embeddings (expanded to the probe batch
                    inside)
    pooled_embeds : (1, D_pool) CLIP pooled embeddings
    text_ids      : (L, 3) rotary ids for the text tokens
    img_ids       : (seq_len, 3) rotary ids for the packed latent tokens
    guidance_scale: distilled (embedded) guidance value

    Returns Q (D, n_vectors) fp32 orthonormal, S (n_vectors,) singular-value
    estimates from the last iteration.
    """
    z0 = z_ref.detach().unsqueeze(0) if z_ref.dim() == 2 else z_ref.detach()
    device = z0.device
    D = z0.numel()
    t_scalar = t_val if torch.is_tensor(t_val) else torch.tensor(float(t_val), device=device)
    t_scalar = t_scalar.to(device)
    sigma = float(sigma)
    z0_f32 = z0.float()
    chan_shape = z0.shape[1:]

    use_guidance = bool(getattr(transformer.config, "guidance_embeds", False))
    # After pipe.to(device), some FLUX submodules can remain float32 while
    # others are bf16. Match the x_embedder weight dtype so Linear does not
    # see bf16 activations against fp32 weights (or vice versa).
    weight_dtype = transformer.x_embedder.weight.dtype

    def endpoint_batch(z_batch_f32):
        """D_t(z) = z - sigma_t * v_theta(z, t) for a BATCH of packed
        latents, run in chunks of fwd_chunk. Accumulates in fp32 so the
        finite-difference subtraction below stays numerically stable."""
        outs = []
        for s in range(0, z_batch_f32.shape[0], fwd_chunk):
            zc = z_batch_f32[s:s + fwd_chunk].to(weight_dtype)
            b = zc.shape[0]
            # the pipeline feeds the transformer timestep/1000
            tt = (t_scalar.float() / 1000.0).to(zc.dtype).expand(b)
            g = (torch.full((b,), float(guidance_scale), device=device, dtype=torch.float32)
                 if use_guidance else None)
            v = transformer(hidden_states=zc, timestep=tt,
                            guidance=g,
                            pooled_projections=pooled_embeds.expand(b, -1).to(weight_dtype),
                            encoder_hidden_states=prompt_embeds.expand(b, -1, -1).to(weight_dtype),
                            txt_ids=text_ids, img_ids=img_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False)[0]
            outs.append(v.float())
        v_all = torch.cat(outs, dim=0)
        return z_batch_f32 - sigma * v_all

    base = endpoint_batch(z0_f32).reshape(D)

    def jvp_fd_block(Q_cols):
        """Forward finite-difference approximation of J @ Q_cols, one column
        per probe, with all k probes batched through the transformer (each
        probe is an independent forward, so this equals the SD3.5 per-column
        loop). fd_eps must be large enough to survive bf16 rounding (see
        --fd-eps help in args.py)."""
        k = Q_cols.shape[1]
        om = Q_cols.T.reshape(k, *chan_shape)
        om_norm = om.reshape(k, -1).norm(dim=1).clamp_min(1e-12)
        om_hat = om / om_norm.reshape(k, *([1] * len(chan_shape)))
        d_plus = endpoint_batch(z0_f32 + fd_eps * om_hat).reshape(k, D)
        return ((d_plus - base.unsqueeze(0)) / fd_eps).T

    # Subspace iteration (block power method): repeatedly apply J to an
    # orthonormal basis Q and re-orthonormalize, converging Q's column space
    # toward the top-n_vectors left-singular subspace of J.
    Q = torch.randn(D, n_vectors, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    S = None
    for it in range(n_iters):
        Y = jvp_fd_block(Q)
        if it == n_iters - 1:
            # Column norms of J@Q right before the final re-orthonormalization
            # approximate the top singular values.
            S = Y.norm(dim=0)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        if debug:
            print(f"    [jive-j] iter {it+1}/{n_iters} colnorms="
                  f"{[f'{float(n):.3f}' for n in Y.norm(dim=0).tolist()]}")
    return Q, S


def top_singular_subspace_jtj(transformer, z_ref, t_val, sigma,
                              prompt_embeds, pooled_embeds, text_ids, img_ids,
                              guidance_scale, n_vectors=4, n_iters=10, fd_eps=4.0,
                              fwd_chunk=4, vjp_chunk=2, dtype=torch.bfloat16,
                              joint_attention_kwargs=None, debug=False):
    """Top-n RIGHT singular vectors of the SAME endpoint Jacobian as
    top_singular_subspace_j, but estimated with subspace iteration on
    J^T J instead of on J alone.

    Why this variant exists
    -----------------------
    The plain iteration Q <- QR(J Q) is a block POWER method: it converges to
    the dominant INVARIANT subspace (eigenvectors of J), which coincides with
    the singular subspace only when J is symmetric. The FLUX endpoint Jacobian
    J = I - sigma * dv/dz is not symmetric, so that iteration answers "which
    directions does J amplify under repeated application", not "which
    directions does J stretch most in one application". Iterating

        Q <- QR(J^T J Q)

    is the block power method on the symmetric PSD matrix J^T J, whose
    eigenvectors ARE the right singular vectors of J and whose eigenvalues are
    the SQUARED singular values. So this variant (a) targets the true singular
    subspace and (b) converges on the squared spectrum, i.e. the gap ratio per
    iteration is (s_i/s_j)^2 rather than |lambda_i/lambda_j| -- at least as
    fast at equal n_iters.

    Here J is square (latent -> latent), which is exactly why the cheaper
    J-only iteration above was possible in the first place -- and why it was
    approximate.

    Each iteration costs one batched FD matvec block (W = J Q, no grad) plus
    one batched autograd VJP block (Y = J^T W). Since D = z - sigma*v(z),

        J^T w = w - sigma * (dv/dz)^T w

    and the (dv/dz)^T w term is one backward w.r.t. the latent input only
    (transformer parameters are temporarily frozen, so no parameter gradients
    are allocated). The identity term is kept explicitly -- unlike the
    conditioning arm, where dz_ref/dy = 0 makes it vanish.

    Parameters mirror top_singular_subspace_j, plus:
    vjp_chunk     : batch cap for the VJP backward. Backward activations are
                    the memory peak (the FD forwards are no_grad), so this
                    defaults lower than fwd_chunk.

    Returns Q (D, n_vectors) fp32 orthonormal, S (n_vectors,) singular-value
    estimates ||J q_i|| of the converged basis.
    """
    z0 = z_ref.detach().unsqueeze(0) if z_ref.dim() == 2 else z_ref.detach()
    device = z0.device
    D = z0.numel()
    t_scalar = t_val if torch.is_tensor(t_val) else torch.tensor(float(t_val), device=device)
    t_scalar = t_scalar.to(device)
    sigma = float(sigma)
    z0_f32 = z0.float()
    chan_shape = z0.shape[1:]

    use_guidance = bool(getattr(transformer.config, "guidance_embeds", False))
    weight_dtype = transformer.x_embedder.weight.dtype

    def _call_transformer(z_batch, grad: bool):
        """v_theta(z_batch, t) at the fixed clean conditioning. grad=False
        runs under no_grad (FD probes); grad=True builds the graph w.r.t.
        z_batch (explicit enable_grad, so an outer no_grad cannot disable
        the VJP)."""
        b = z_batch.shape[0]
        tt = (t_scalar.float() / 1000.0).to(weight_dtype).expand(b)
        g = (torch.full((b,), float(guidance_scale), device=device, dtype=torch.float32)
             if use_guidance else None)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            v = transformer(hidden_states=z_batch.to(weight_dtype), timestep=tt,
                            guidance=g,
                            pooled_projections=pooled_embeds.expand(b, -1).to(weight_dtype),
                            encoder_hidden_states=prompt_embeds.expand(b, -1, -1).to(weight_dtype),
                            txt_ids=text_ids, img_ids=img_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False)[0]
        return v.float()

    def endpoint_batch(z_batch_f32):
        """D_t(z) = z - sigma_t * v_theta(z, t), chunked by fwd_chunk, fp32
        accumulation for FD stability."""
        outs = []
        for s in range(0, z_batch_f32.shape[0], fwd_chunk):
            outs.append(_call_transformer(z_batch_f32[s:s + fwd_chunk], grad=False))
        return z_batch_f32 - sigma * torch.cat(outs, dim=0)

    with torch.no_grad():
        base = endpoint_batch(z0_f32).reshape(D)

    def jvp_fd_block(Q_cols):
        """W = J @ Q_cols by forward FD, k probes batched (identical math to
        top_singular_subspace_j's block)."""
        k = Q_cols.shape[1]
        om = Q_cols.T.reshape(k, *chan_shape)
        om_norm = om.reshape(k, -1).norm(dim=1).clamp_min(1e-12)
        om_hat = om / om_norm.reshape(k, *([1] * len(chan_shape)))
        with torch.no_grad():
            d_plus = endpoint_batch(z0_f32 + fd_eps * om_hat).reshape(k, D)
        return ((d_plus - base.unsqueeze(0)) / fd_eps).T          # (D, k)

    def vjp_block(W):
        """J^T @ W = W - sigma * (dv/dz)^T @ W by autograd. loss_r =
        <v_theta(z_r), w_r> is row-independent, so one backward on the
        batched loss yields all k VJPs at once (chunked by vjp_chunk)."""
        k = W.shape[1]
        req = [p.requires_grad for p in transformer.parameters()]
        transformer.requires_grad_(False)
        grads = []
        try:
            for s in range(0, k, vjp_chunk):
                r = min(vjp_chunk, k - s)
                z_r = z0_f32.expand(r, -1, -1).clone().requires_grad_(True)
                v = _call_transformer(z_r, grad=True).reshape(r, D)
                loss = (v * W[:, s:s + r].T).sum()
                g = torch.autograd.grad(loss, z_r)[0]
                grads.append(g.reshape(r, D).float())
        finally:
            for p, flag in zip(transformer.parameters(), req):
                p.requires_grad_(flag)
        dvT_W = torch.cat(grads, dim=0).T                          # (D, k)
        return W - sigma * dvT_W                                   # J^T W

    # Subspace iteration on J^T J (block power method on a symmetric PSD
    # operator): Q converges to the top-n right singular subspace of J.
    Q = torch.randn(D, n_vectors, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    for it in range(n_iters):
        W = jvp_fd_block(Q)          # J Q
        Y = vjp_block(W)             # J^T (J Q)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        if debug:
            print(f"    [jive-jtj] iter {it+1}/{n_iters} colnorms="
                  f"{[f'{float(n):.3f}' for n in Y.norm(dim=0).tolist()]}")
    # Singular-value estimates of the converged basis: ||J q_i||.
    S = jvp_fd_block(Q).norm(dim=0)
    return Q, S
