"""Command-line arguments for the set-level comparison (FLUX.1-dev / FLUX.1-schnell).

Model-specific defaults are selected via --model (dev: 28 steps + distilled
guidance 3.5; schnell: 4 steps + guidance 0).
"""
import os
import argparse

_FLUX_ROOT = os.environ.get("FLUX_ROOT", "")

MODEL_PRESETS = {
    "dev": {
        "model_dir": os.path.join(_FLUX_ROOT, "FLUX.1-dev") if _FLUX_ROOT
                     else "black-forest-labs/FLUX.1-dev",
        "steps": 28,
        "guidance": 3.5,
        "guidances": [3.5],
        "method": "jive_flux_dev",
    },
    "schnell": {
        "model_dir": os.path.join(_FLUX_ROOT, "FLUX.1-schnell") if _FLUX_ROOT
                     else "black-forest-labs/FLUX.1-schnell",
        "steps": 4,
        "guidance": 0.0,
        "guidances": [0.0],
        "method": "jive_flux_schnell",
    },
}


def _apply_model_preset(args):
    preset = MODEL_PRESETS[args.model]
    if args.model_dir is None:
        args.model_dir = preset["model_dir"]
    if args.steps is None:
        args.steps = preset["steps"]
    if args.guidance is None:
        args.guidance = preset["guidance"]
    if args.guidances is None:
        args.guidances = list(preset["guidances"])
    if args.method == "jive_flux":
        args.method = preset["method"]


def parse_args():
    ap = argparse.ArgumentParser(
        description='FLUX set-level comparison from the same random starts: '
                    'the unmodified sampler vs OSCAR vs JIVE vs JIVE+CADS'
    )

    ap.add_argument('--model', type=str, default='dev', choices=['dev', 'schnell'],
                    help='FLUX variant preset. dev: FLUX.1-dev, 28 steps, guidance 3.5. '
                         'schnell: FLUX.1-schnell, 4 steps, guidance 0 (no guidance embeds).')

    ap.add_argument('--spec', type=str, default=None, help='Path to JSON: {concept:[prompts...]}')
    ap.add_argument('--prompt', type=str, default=None, help='Single prompt if --spec not provided')
    ap.add_argument('--negative', type=str, default='',
                    help='Negative prompt. Only used when --true-cfg-scale > 1 (FLUX.1-dev '
                         'normally uses distilled guidance, which has no negative branch).')

    ap.add_argument('--n-images', type=int, default=200,
                    help='Total number of images to generate PER ARM for evaluation. '
                         'Generated in batches of --G; metrics are pooled over all of them.')
    ap.add_argument('--G', type=int, default=4,
                    help='Generation batch size. For the OSCAR arm this is also the '
                         'group size its volume objective acts on.')
    ap.add_argument('--keep-images-per-arm', type=int, default=16,
                    help='How many PNGs to write to disk per arm (metrics still use '
                         'all --n-images samples).')
    ap.add_argument('--metric-batch-size', type=int, default=32,
                    help='Feature-extractor / scorer batch size for the pooled metrics.')
    ap.add_argument('--height', type=int, default=512,
                    help='Image height. FLUX is trained at 1024; 512 keeps the '
                         'large-sample comparison affordable.')
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=None,
                    help='Denoising steps. Default: 28 (dev) or 4 (schnell), from --model.')

    ap.add_argument('--guidances', type=float, nargs='+', default=None)
    ap.add_argument('--guidance', type=float, default=None)
    ap.add_argument('--true-cfg-scale', type=float, default=1.0,
                    help='True classifier-free guidance scale for the pipeline denoise '
                         '(>1 enables the negative-prompt branch). JIVE linearizes '
                         'the distilled-guidance velocity only; keep at 1.0 for a fair '
                         'comparison with ARM C.')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1111])
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--model-dir', type=str, default=None,
                    help='Path to local FLUX weights. Default from --model preset.')
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--clip-impl', type=str, default='auto', choices=['auto', 'openai_clip', 'open_clip'])
    ap.add_argument('--clip-arch', type=str, default=None,
                    help='open_clip model arch. Defaults to ViT-B-32-quickgelu for pretrained=openai, '
                         'ViT-B-32 otherwise. Ignored when --clip-impl openai_clip.')
    ap.add_argument('--clip-checkpoint', type=str, default=None)
    ap.add_argument('--clip-pretrained', type=str, default='openai')

    ap.add_argument('--method', type=str, default='jive_flux')
    ap.add_argument('--out-root', type=str, default=None,
                    help='Root under which outputs/<method>_<concept>/ run directories are '
                         'created. Defaults to the directory containing this package.')
    ap.add_argument('--outputs-subdir', type=str, default=None,
                    help="Optional suffix: write under outputs_<suffix>/ instead of the "
                         "default outputs/ tree (e.g. 'larger_gamma_OSCAR' -> "
                         "outputs_larger_gamma_OSCAR/) so a variant run does not clobber "
                         "the default run's results.")
    ap.add_argument('--arms', type=str, nargs='+',
                    default=['deterministic', 'oscar', 'jive'],
                    choices=['deterministic', 'oscar', 'jive', 'jive_cads'],
                    help="Which arms to run (default: the unmodified sampler, OSCAR "
                         "and JIVE). 'jive_cads' is ARM D: JIVE's start projection "
                         "with CADS conditioning corruption layered on top during "
                         "denoising. The standalone CADS baseline is a separate "
                         "script (baselines/cads.py), not an arm here.")
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip a (prompt, guidance, seed) run if its results.json already exists.')

    ap.add_argument('--inject-norms', type=float, nargs='+', default=[8.0],
                    help='L2 norm(s) of the projected perturbation. One ARM C run per value. '
                         'The FLUX packed latent at H x W has dim 64*(H/16)*(W/16) '
                         '(e.g. 65536 at 512x512, latent norm ~256), so useful norms fall '
                         'in the 4-32 range at that resolution.')
    ap.add_argument('--inject-n', type=int, default=4, help='Rank of the singular subspace.')
    ap.add_argument('--jive-iters', type=int, default=10,
                    help='Subspace-iteration steps for the top singular vectors.')
    ap.add_argument('--fd-eps', type=float, default=4.0,
                    help='Finite-difference step for the Jacobian-vector products. NOTE: the '
                         'transformer runs in bf16, whose ULP near 1.0 is ~0.008; with a '
                         'unit-norm direction spread over ~65k dims, per-entry probes below '
                         '~4/sqrt(D) would round away entirely. Hence the default is much '
                         'larger than the fp32 SD1.5 value (0.1).')
    ap.add_argument('--fwd-chunk', type=int, default=4,
                    help='Transformer batch cap for the k finite-difference probes of one '
                         'subspace iteration (batched together for speed; math is identical '
                         'to one probe at a time). Lower it if ARM C OOMs.')
    ap.add_argument('--jive-iter-mode', type=str, default='j', choices=['j', 'jtj'],
                    help="Operator iterated to find ARM C/D's subspace, reported in the "
                         "paper as JIVE(J) and JIVE(J^T J). 'j': Q <- QR(J Q), a block "
                         "POWER method that converges to the dominant INVARIANT subspace "
                         "-- equal to the singular subspace only for symmetric J, which "
                         "the endpoint Jacobian I - sigma*dv/dz is not. 'jtj': "
                         "Q <- QR(J^T J Q), the block power method on the symmetric PSD "
                         "J^T J, whose eigenvectors ARE J's right singular vectors and "
                         "whose eigenvalues are the SQUARED singular values (so it also "
                         "separates the spectrum faster at equal --jive-iters). Costs one "
                         "extra autograd VJP block per iteration.")
    ap.add_argument('--jive-vjp-chunk', type=int, default=2,
                    help="Batch cap for the autograd VJP block of --jive-iter-mode jtj. "
                         'Backward activations are the memory peak (the FD forwards run '
                         'under no_grad), so this defaults lower than --fwd-chunk. '
                         'Ignored when --jive-iter-mode j.')

    ap.add_argument('--cads-s', type=float, default=0.15,
                    help='ARM D CADS noise scale s (Eq. 1).')
    ap.add_argument('--cads-tau1', type=float, default=0.6,
                    help='ARM D CADS annealing schedule lower threshold (gamma = 1 below it).')
    ap.add_argument('--cads-tau2', type=float, default=1.2,
                    help='ARM D CADS annealing schedule upper threshold (gamma = 0 at/above '
                         'it). tau2 > 1 keeps gamma(1) > 0 so the FIRST step is only '
                         'partially corrupted instead of fully unconditional.')
    ap.add_argument('--cads-psi', type=float, default=1.0,
                    help='ARM D CADS rescale mixing factor (Eq. 3-4); 1 restores the '
                         'conditioning mean/std fully, as the paper recommends.')

    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.95)
    ap.add_argument('--t-gate', type=str, default='0.85,0.95')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2', 't1mt', 'const'],
                    help='Perturbation schedule inside --t-gate. sin2/t1mt taper to zero at '
                         'the gate edges (so the first/last normalized step is never perturbed); '
                         'const applies full gamma0 at every step inside the gate.')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)
    ap.add_argument('--eta-sde', type=float, default=1.0)
    ap.add_argument('--rho', type=float, default=0.25)
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4)
    ap.add_argument('--oscar-noise-mode', type=str, default='sde', choices=['sde', 'norm'],
                    help="Scale of OSCAR's per-step noise target: 'sde' targets a true "
                         "Brownian increment (brown_std*sqrt(D), still capped by --rho); "
                         "'norm' is the legacy behavior (total norm = brown_std, "
                         "~sqrt(D)x smaller).")

    ap.add_argument('--device-transformer', type=str, default='cuda:0')
    ap.add_argument('--device-vae', type=str, default='cuda:0')
    ap.add_argument('--device-clip', type=str, default='cuda:0')

    ap.add_argument('--vae-grad-chunk', type=int, default=1,
                    help='Sub-batch size for the VAE forward/backward pass inside '
                         "ARM B's per-step callback (oscar_arm.py); lower it if that "
                         'step OOMs on --device-vae.')
    ap.add_argument('--enable-model-cpu-offload', action='store_true')
    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true', help='Verbose per-step logging.')

    ap.add_argument('--vendi-kernel', choices=('cosine', 'rbf'), default='cosine')
    ap.add_argument('--vendi-feature', choices=('auto', 'dinov2', 'inception', 'clip', 'pixel'),
                    default='auto',
                    help='Embedding backbone for feature Vendi. "auto"/"dinov2" use '
                         'HuggingFace facebook/dinov2-base + AutoImageProcessor (fdeval '
                         'vendi_dino_q1). "pixel" disables feature Vendi.')
    ap.add_argument('--vendi-rbf-gamma', type=float, default=None)

    ap.add_argument('--quality-metrics', type=str, nargs='+',
                    default=['clip_score', 'brisque', 'clip_iqa', 'image_reward', 'hpsv2'],
                    choices=['clip_score', 'brisque', 'clip_iqa', 'image_reward', 'hpsv2'],
                    help='Which fidelity/no-reference quality metrics to compute per arm. '
                         'clip_score and image_reward need the prompt; brisque and clip_iqa '
                         'are no-reference (prompt-independent). Each is skipped automatically '
                         'if its package (open_clip / pyiqa / image-reward) is not installed. '
                         'Trim this list if you are tight on --device-clip memory -- '
                         'image_reward is the heaviest (~1.5GB checkpoint).')

    ap.add_argument('--kid', action='store_true',
                    help='Compute KID (Kernel Inception Distance: unbiased squared MMD with '
                         'the cubic polynomial kernel on InceptionV3 pool3 features) of each '
                         'perturbed arm AGAINST the deterministic arm of the same run. All '
                         'arms share identical starts, so this isolates the distribution '
                         'shift each method introduces (lower = closer to the base sampler; '
                         'the deterministic arm itself is the reference and gets exactly 0). '
                         'No-op when the deterministic arm is not in --arms.')
    ap.add_argument('--kid-subsets', type=int, default=100,
                    help='Random subsets for the KID mean/std estimate.')
    ap.add_argument('--kid-subset-size', type=int, default=100,
                    help='Images per KID subset (clamped to the smaller set).')
    args = ap.parse_args()
    _apply_model_preset(args)
    return args
