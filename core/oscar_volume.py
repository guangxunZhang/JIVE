"""Builder for the OSCAR baseline's CLIP wrapper + volume objective.

Kept out of the per-backbone pipeline_setup modules because it is
backbone-independent: OSCAR's objective acts on decoded images through CLIP,
so both the FLUX arm and any other backbone construct it identically.
"""
import os

import torch

from baselines.oscar.utils import log as _log


def build_oscar_volume_objective(args, dev_clip):
    """Builds the CLIP wrapper + VolumeObjective + DiversityConfig used by the
    OSCAR arm's per-step perturbation (jive_flux/oscar_arm.py). Only called
    when 'oscar' is one of the requested --arms."""
    from baselines.oscar.clip_wrapper import CLIPWrapper
    from baselines.oscar.volume_objective import VolumeObjective
    from baselines.oscar.config import DiversityConfig

    _log("Loading CLIP ...", args.debug)
    clip_jit = os.path.expanduser(args.clip_jit) if args.clip_jit else None
    clip_checkpoint = os.path.expanduser(args.clip_checkpoint) if args.clip_checkpoint else None
    clip_impl = args.clip_impl
    # 'auto': prefer the JIT checkpoint if it's actually present on disk,
    # otherwise fall back to open_clip (which can download weights).
    if clip_impl == "auto":
        clip_impl = "openai_clip" if clip_jit and os.path.isfile(clip_jit) else "open_clip"
    if clip_impl == "openai_clip" and (not clip_jit or not os.path.isfile(clip_jit)):
        raise FileNotFoundError(f"CLIP JIT not found at {clip_jit}; use --clip-impl open_clip or provide --clip-jit")
    if clip_impl == "open_clip":
        # ViT-B-32-quickgelu matches the openai pretrained weights
        if args.clip_arch is not None:
            clip_arch = args.clip_arch
        elif args.clip_pretrained == "openai":
            clip_arch = "ViT-B-32-quickgelu"
        else:
            clip_arch = "ViT-B-32"
    else:
        clip_arch = "ViT-B-32"
    clip = CLIPWrapper(
        impl=clip_impl, arch=clip_arch,
        jit_path=clip_jit, checkpoint_path=clip_checkpoint,
        pretrained=args.clip_pretrained if clip_impl == "open_clip" and clip_checkpoint is None else None,
        device=dev_clip if dev_clip.type == 'cuda' else torch.device("cpu"),
    )
    _log("CLIP ready.", args.debug)

    # t_gate is passed as "t0,t1" on the CLI; DiversityConfig wants a tuple.
    t0, t1 = args.t_gate.split(',')
    cfg = DiversityConfig(
        num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
        gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
        partial_ortho=args.partial_ortho, t_gate=(float(t0), float(t1)),
        sched_shape=args.sched_shape, clip_image_size=224,
        leverage_alpha=0.5,
    )
    vol = VolumeObjective(clip, cfg)
    _log("Volume objective ready.", args.debug)
    return vol, cfg
