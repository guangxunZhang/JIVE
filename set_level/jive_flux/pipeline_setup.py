"""Loads the FLUX.1-dev pipeline and the optional per-run auxiliary models:
the CLIP + VolumeObjective needed by ARM B (OSCAR), the Vendi feature
embedder, and the quality scorers selected via --quality-metrics. The
auxiliary builders are model-agnostic and live in core/.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from baselines.oscar.utils import print_mem_all, inspect_pipe_devices, assert_on, log as _log, resolve_model_dir
from core.oscar_volume import build_oscar_volume_objective
from core.vendi import build_image_embedder
from core.kid import build_kid_featurizer
from .quality_metrics import BUILDERS as _QUALITY_BUILDERS


@dataclass
class PipelineContext:
    """Everything loaded once per process and shared read-only across the
    whole prompt/guidance/seed sweep in main.py, so heavy models (FLUX,
    CLIP, Vendi embedder, quality scorers) are never reloaded per run."""
    pipe: Any                # the FluxPipeline
    dev_tr: torch.device     # transformer (denoiser) device
    dev_vae: torch.device    # VAE encode/decode device
    dev_clip: torch.device   # CLIP / Vendi embedder / quality scorer device
    dtype: torch.dtype
    vol: Optional[Any] = None       # VolumeObjective, only built if ARM B (oscar) is requested
    cfg: Optional[Any] = None       # matching DiversityConfig for `vol`
    embedder: Optional[Callable] = None   # Vendi feature embedder (None => pixel Vendi only)
    scorers: Dict[str, Callable] = field(default_factory=dict)  # name -> quality metric fn
    kid_featurizer: Optional[Callable] = None   # InceptionV3 pool3 for KID (None unless --kid)


def build_pipeline_context(args) -> PipelineContext:
    """Loads FLUX.1-dev plus every auxiliary model needed by the requested arms
    and metrics, and returns them bundled in a PipelineContext. Called once
    per process, before the prompt/guidance/seed sweep in main()."""
    from diffusers import FluxPipeline

    dev_tr   = torch.device(args.device_transformer)
    dev_vae  = torch.device(args.device_vae)
    dev_clip = torch.device(args.device_clip)
    # bf16 on GPU for speed/memory; fp32 on CPU since bf16 CPU kernels are
    # slow/unsupported for many ops.
    dtype    = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

    _log(f"Devices: transformer={dev_tr}, vae={dev_vae}, clip={dev_clip}", args.debug)
    print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_clip])

    # 1) load on CPU then move to devices
    model_dir = resolve_model_dir(args.model_dir)
    _log("Loading FLUX.1-dev (CPU) ...", args.debug)
    pipe = FluxPipeline.from_pretrained(
        model_dir, torch_dtype=dtype, local_files_only=True,
    )
    pipe.set_progress_bar_config(leave=True)
    pipe = pipe.to("cpu")
    print("scheduler:", pipe.scheduler.__class__.__name__)

    if args.enable_model_cpu_offload:
        # diffusers-managed offloading: submodules are moved to GPU only
        # while running and swapped back to CPU otherwise. Mutually
        # exclusive with the explicit per-submodule placement below.
        pipe.enable_model_cpu_offload()
    else:
        # Explicit placement: transformer + both text encoders (CLIP ViT-L
        # and T5-XXL) share dev_tr, the VAE gets its own (possibly different)
        # device. FLUX has no third text encoder.
        if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
        if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

    if args.enable_vae_tiling:
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            _log(f"enable_xformers failed: {e}", args.debug)

    inspect_pipe_devices(pipe)
    if not args.enable_model_cpu_offload:
        if hasattr(pipe, "transformer"): assert_on(pipe.transformer, dev_tr)
        if hasattr(pipe, "vae"):         assert_on(pipe.vae, dev_vae)

    # 2) CLIP & Volume objective (only needed for the OSCAR arm) -- shared
    # with the SD3.5 comparison; it is model-agnostic.
    vol = cfg = None
    if 'oscar' in args.arms:
        vol, cfg = build_oscar_volume_objective(args, dev_clip)

    # 3) Vendi feature embedder
    embedder = None
    if args.vendi_feature != "pixel":
        embedder = build_image_embedder(args.vendi_feature, dev_clip, batch_size=args.metric_batch_size)

    # 4) fidelity / no-reference quality scorers
    scorers: Dict[str, Callable] = {}
    for name in args.quality_metrics:
        fn = _QUALITY_BUILDERS[name](dev_clip, args.metric_batch_size)
        if fn is not None:
            scorers[name] = fn

    # 5) KID feature extractor (InceptionV3 pool3) -- only with --kid; KID
    # compares each perturbed arm against the deterministic one in main.py.
    kid_featurizer = build_kid_featurizer(dev_clip, args.metric_batch_size) if args.kid else None

    return PipelineContext(pipe=pipe, dev_tr=dev_tr, dev_vae=dev_vae, dev_clip=dev_clip,
                           dtype=dtype, vol=vol, cfg=cfg, embedder=embedder, scorers=scorers,
                           kid_featurizer=kid_featurizer)
