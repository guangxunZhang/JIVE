import torch

from core.noise_projection import (
    projected_noise_like, PERTURB_SEED_OFFSET,
)


@torch.no_grad()
def top_singular_subspace_j(transformer, z_ref, t_val, sigma,
                            prompt_embeds, pooled_embeds, text_ids, img_ids,
                            guidance_scale, n_vectors=4, n_iters=10, fd_eps=4.0,
                            fwd_chunk=4, dtype=torch.bfloat16,
                            joint_attention_kwargs=None, debug=False):
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

    def endpoint_batch(z_batch_f32):
        outs = []
        for s in range(0, z_batch_f32.shape[0], fwd_chunk):
            zc = z_batch_f32[s:s + fwd_chunk].to(weight_dtype)
            b = zc.shape[0]
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
        k = Q_cols.shape[1]
        om = Q_cols.T.reshape(k, *chan_shape)
        om_norm = om.reshape(k, -1).norm(dim=1).clamp_min(1e-12)
        om_hat = om / om_norm.reshape(k, *([1] * len(chan_shape)))
        d_plus = endpoint_batch(z0_f32 + fd_eps * om_hat).reshape(k, D)
        return ((d_plus - base.unsqueeze(0)) / fd_eps).T

    Q = torch.randn(D, n_vectors, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    S = None
    for it in range(n_iters):
        Y = jvp_fd_block(Q)
        if it == n_iters - 1:
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
        outs = []
        for s in range(0, z_batch_f32.shape[0], fwd_chunk):
            outs.append(_call_transformer(z_batch_f32[s:s + fwd_chunk], grad=False))
        return z_batch_f32 - sigma * torch.cat(outs, dim=0)

    with torch.no_grad():
        base = endpoint_batch(z0_f32).reshape(D)

    def jvp_fd_block(Q_cols):
        k = Q_cols.shape[1]
        om = Q_cols.T.reshape(k, *chan_shape)
        om_norm = om.reshape(k, -1).norm(dim=1).clamp_min(1e-12)
        om_hat = om / om_norm.reshape(k, *([1] * len(chan_shape)))
        with torch.no_grad():
            d_plus = endpoint_batch(z0_f32 + fd_eps * om_hat).reshape(k, D)
        return ((d_plus - base.unsqueeze(0)) / fd_eps).T

    def vjp_block(W):
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
        dvT_W = torch.cat(grads, dim=0).T
        return W - sigma * dvT_W

    Q = torch.randn(D, n_vectors, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    for it in range(n_iters):
        W = jvp_fd_block(Q)
        Y = vjp_block(W)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        if debug:
            print(f"    [jive-jtj] iter {it+1}/{n_iters} colnorms="
                  f"{[f'{float(n):.3f}' for n in Y.norm(dim=0).tolist()]}")
    S = jvp_fd_block(Q).norm(dim=0)
    return Q, S
