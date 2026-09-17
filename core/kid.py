"""KID (Kernel Inception Distance): squared MMD between the InceptionV3
pool3 feature distributions of two image sets, using the cubic polynomial
kernel k(x, y) = (x.y / d + 1)^3 (Binkowski et al., ICLR 2018,
"Demystifying MMD GANs").

In this comparison the reference set is the DETERMINISTIC arm of the same
(prompt, guidance, seed) run: every arm shares identical pure-noise starts,
so KID(arm || deterministic) isolates the distribution shift each
perturbation method introduces -- the set-level fidelity counterpart to the
within-set diversity metrics (Vendi, mean pairwise L2). Lower = closer to
the base sampler's distribution; the unbiased estimator is ~0 (and can go
slightly negative) when the two distributions match.

Following the standard protocol, the estimate is reported as mean +/- std
over random subsets, so the std reflects estimator noise at the given
sample count rather than a per-image quantity.
"""
import torch

_KERNEL_DEGREE = 3
_KERNEL_COEF0 = 1.0


@torch.no_grad()
def _poly_gram(X, Y):
    """Gram matrix of the KID polynomial kernel k(x, y) = (x.y/d + 1)^3,
    computed in float64: the MMD is a small difference of O(n^2) sums of
    cubed dot products, so the extra precision keeps it stable."""
    return ((X @ Y.T) / X.shape[1] + _KERNEL_COEF0) ** _KERNEL_DEGREE


def _unbiased_mmd2(Ktt, Krr, Ktr):
    """Unbiased squared-MMD estimate from precomputed Gram blocks: the
    within-set terms exclude the diagonal; the cross term is a plain mean."""
    m, n = Ktt.shape[0], Krr.shape[0]
    tt = (Ktt.sum() - Ktt.diagonal().sum()) / (m * (m - 1))
    rr = (Krr.sum() - Krr.diagonal().sum()) / (n * (n - 1))
    return (tt + rr - 2.0 * Ktr.mean()).item()


def kid_mmd2(feats_test, feats_ref, n_subsets=100, subset_size=100, seed=0):
    """KID estimate between two feature sets -> (mean, std) over subsets.

    feats_*: (n, d) tensors of RAW Inception pool3 activations (NOT
    L2-normalized -- the polynomial kernel expects activations). Each subset
    draws `subset_size` rows (clamped to the smaller set; without replacement
    within a subset) independently from each set with a seeded generator, so
    the estimate is reproducible across arms of the same run.
    """
    X, Y = feats_test.double(), feats_ref.double()
    n_t, n_r = X.shape[0], Y.shape[0]
    if min(n_t, n_r) < 2:
        raise ValueError(f"KID needs >= 2 images per set, got {n_t} and {n_r}")
    m = min(int(subset_size), n_t, n_r)
    Ktt, Krr, Ktr = _poly_gram(X, X), _poly_gram(Y, Y), _poly_gram(X, Y)
    g = torch.Generator().manual_seed(int(seed))
    vals = []
    for _ in range(max(1, int(n_subsets))):
        it = torch.randperm(n_t, generator=g)[:m]
        ir = torch.randperm(n_r, generator=g)[:m]
        vals.append(_unbiased_mmd2(Ktt[it][:, it], Krr[ir][:, ir], Ktr[it][:, ir]))
    v = torch.tensor(vals)
    return float(v.mean()), (float(v.std(unbiased=True)) if len(vals) > 1 else 0.0)


def build_kid_featurizer(device="cpu", batch_size=32):
    """featurize(images NCHW in [0,1]) -> (n, 2048) CPU float32 InceptionV3
    pool3 activations (raw, unnormalized -- KID's kernel expects activations).
    Same torchvision weights + 299px ImageNet prep as the Vendi inception
    embedder. Returns None when the weights cannot be loaded (e.g. an
    offline node), in which case KID is skipped rather than aborting the run."""
    try:
        import torchvision
        weights = torchvision.models.Inception_V3_Weights.DEFAULT
        model = torchvision.models.inception_v3(weights=weights, aux_logits=True)
        model.fc = torch.nn.Identity()  # pool3 (avgpool) 2048-d output
        model = model.to(device).eval()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  [kid] InceptionV3 unavailable ({type(exc).__name__}: {exc}); KID disabled")
        return None

    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def featurize(images):
        outs = []
        for chunk in images.split(batch_size):
            x = chunk.to(device).float()
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(299, 299), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
            outs.append(model((x - mean) / std).float().cpu())
        return torch.cat(outs, dim=0)

    print("  KID: InceptionV3 pool3 (2048-d), cubic-kernel MMD^2 vs the deterministic arm")
    return featurize
