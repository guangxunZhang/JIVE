#!/usr/bin/env python3
"""
Standalone external-baseline generator for FLUX.1-schnell.

Deliberately independent of fdeval: it imports nothing from it. The only two
things it borrows are CONVENTIONS, both replicated below with the reason they
must match --

  * the per-image seed formula, so every image is paired with the
    corresponding image of an fdeval run (identical initial noise), and
  * the on-disk run layout, so `rescore_from_images.py` and `analyze.py`
    can consume the output with no changes.

It writes images plus a metadata-only .npz stub per group; scoring is then done
by the existing tooling, so the baseline is measured by exactly the same scorer
stack as the method and nothing here can drift from it.

This is the CADS row of the set-level table; see set_level/README.md for the
launcher that runs it over PartiPrompts under the same prompts/seed/steps as
the other arms. cads_gamma and cads_corrupt below are ALSO imported by the
JIVE+CADS arm (set_level/jive_flux/jive_cads_arm.py), so the corruption math
of the two rows is literally the same code.

    # 1. generate
    python baselines/cads.py --prompt_set eval_cache/parti1632.json \\
        --out_dir runs/cads --cads_s 0.05 0.10 0.15 0.25 --num_images 4

    # 2. score with fdeval's own scorers, 3. analyse
    python rescore_from_images.py --run_dir runs/cads \\
        --metrics clip,dino,inception,lpips,pickscore,hpsv2,imagereward,clipiqa
    python analyze.py --run_dir runs/cads --prompt_set eval_cache/parti1632.json

Steps 2-3 (and the in-process --score path) live in the separate fdeval
scoring harness, which must be on PYTHONPATH; generation itself needs
nothing but diffusers.

CADS (Sadat et al., ICLR 2024), Eq. 1-2:

    y_hat = sqrt(gamma(t)) * y + s * sqrt(1 - gamma(t)) * n,     n ~ N(0, I)

    gamma(t) = 1                          t <= tau1
               (tau2 - t) / (tau2 - tau1)   tau1 < t < tau2
               0                          t >= tau2

followed by the optional rescale of Eq. 3-4 with mixing factor psi. Defaults
here are the paper's Table 13 *Stable Diffusion* row -- the only text-to-image
entry -- tau1=0.6, tau2=0.9, s=0.25, psi=1.

THE STEP-COUNT TRAP, because it silently produces a meaningless baseline:
sampling runs backward from t=1, so gamma(1)=0 under every published tau2
(0.6 ... 1.0) and the FIRST step is deliberately UNCONDITIONAL, with later
steps reintroducing the prompt. At --steps 1 there are no later steps: the
sample is unconditional and CLIP collapses. This script REFUSES a 1-step run
unless tau2 > 1 puts gamma(1) above zero. At 4 steps gamma is 0 / 0.169 /
0.825 / 1.0 -- one unconditional step, i.e. 25% of the trajectory against ~10%
at the 100-step schedule the paper tuned on, so if alignment suffers, lower s
before reaching for tau2 > 1.

Costs no extra forward passes: the corruption is applied to text embeddings
that were computed anyway, so NFE per image equals the plain baseline's.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import zlib
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Conventions shared with fdeval. Replicated, not imported, so this file has no
# dependency on that tree -- but they MUST stay in step, so each is stated with
# the invariant it protects.
# ---------------------------------------------------------------------------

def group_seed(base_seed: int, prompt_uid: str, image_idx: int) -> int:
    """
    Must match fdeval's `pipeline.group_seed` EXACTLY.

    Image i of prompt p then starts from the same initial noise here as in an
    fdeval run, which is what makes a CADS row comparable to a jacobian row
    prompt-by-prompt rather than only in aggregate. Keyed on the uid (not the
    prompt's position in the list) so sharding and ordering cannot change it.
    """
    h = zlib.crc32(prompt_uid.encode("utf-8"))
    return int((base_seed + h + image_idx) % (2 ** 31 - 1))


def group_image_dir(out_dir: str, cond: str, uid: str) -> str:
    return os.path.join(out_dir, "images", cond, uid)


def group_feature_path(out_dir: str, cond: str, uid: str) -> str:
    return os.path.join(out_dir, "features", cond, f"{uid}.npz")


def write_stub_npz(path: str, meta: dict) -> None:
    """
    A metadata-only .npz, which is what makes `rescore_from_images.py` see this
    group at all: it enumerates FEATURE files and then looks for the images
    beside them, so images alone would be invisible to it. Written tmp+rename
    for the same reason fdeval does -- a truncated archive read back later is
    indistinguishable from a corrupt one.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, _meta=buf)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CADS
# ---------------------------------------------------------------------------

def cads_gamma(t: float, tau1: float, tau2: float) -> float:
    """Annealing schedule, Eq. 2. `t` is diffusion time in [0,1] = sigma."""
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    return (tau2 - t) / (tau2 - tau1)


def cads_corrupt(y, t: float, s: float, tau1: float, tau2: float, psi: float,
                 generator=None, noise=None):
    """
    One application of Eq. 1 followed by the Eq. 3-4 rescale.

    The rescale exists because adding noise shifts the conditioning vector's
    mean and standard deviation, which destabilises sampling at larger s; psi=1
    restores them fully and is what the paper recommends. Noise is drawn FRESH
    per call, which is the point of the method -- "the model processes a
    different input at each step".

    `noise` (optional) substitutes a caller-supplied perturbation for the
    isotropic Gaussian draw, so a variant can shape the corruption
    direction while keeping Eq. 1 + the rescale byte-identical to this
    baseline. Must be fp32 and broadcastable to y.shape.
    """
    import torch

    g = cads_gamma(t, tau1, tau2)
    if g >= 1.0:
        return y
    if noise is None:
        n = torch.randn(y.shape, generator=generator, device=y.device, dtype=torch.float32)
    else:
        n = noise.to(device=y.device, dtype=torch.float32)
    y32 = y.float()
    y_hat = math.sqrt(g) * y32 + s * math.sqrt(1.0 - g) * n
    if psi > 0.0:
        mu_in, sd_in = y32.mean(), y32.std()
        y_res = (y_hat - y_hat.mean()) / (y_hat.std() + 1e-12) * sd_in + mu_in
        y_hat = psi * y_res + (1.0 - psi) * y_hat
    return y_hat.to(y.dtype)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt_set", required=True,
                   help="prompt_set.json from `python -m fdeval.prompts` (read as "
                        "plain JSON; only `uid` and `text` are used)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--save_images_per_group", type=int, default=0,
                   help="how many images per (prompt, arm) to keep on disk. 0 or "
                        ">= --num_images keeps all. WHAT THIS COSTS DEPENDS ON "
                        "--score. WITH --score the whole group is scored in "
                        "memory first, so this only controls disk and the metrics "
                        "still cover all --num_images -- same meaning as fdeval's "
                        "flag of this name. WITHOUT --score nothing is scored "
                        "until `rescore_from_images.py` reads the PNGs, so an "
                        "unsaved image can never be scored and is not generated "
                        "either: the GROUP shrinks, and Vendi, LPIPS and every "
                        "pairwise statistic are n-dependent, so an arm kept at 2 "
                        "cannot sit in a table beside one scored over 4.")
    p.add_argument("--seed", type=int, default=0,
                   help="must equal the fdeval run's --seed for the pairing to hold")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--offload_device", default=None,
                   help="put the text encoders on this device instead of "
                        "--device. Required for SD3.5-Large on a 24 GB card: "
                        "the full pipeline is 27.4 GB (16.1 transformer + 9.5 "
                        "T5 + 1.6 CLIP), so it cannot be resident on one GPU. "
                        "The VAE deliberately STAYS on --device -- it is only "
                        "0.17 GB and the pipeline decodes there.")
    p.add_argument("--backend", choices=["flux", "sd3"], default=None,
                   help="model family; inferred from --model_id when omitted. "
                        "Must match the fdeval run you are pairing against.")
    p.add_argument("--sd3_shift", type=float, default=3.0,
                   help="SD3 only, and only for the --dry_run schedule preview: "
                        "the scheduler's static shift (SD3/3.5 ship 3.0). The "
                        "real run reads sigmas off the loaded scheduler and "
                        "prints them, so this never affects generation.")
    p.add_argument("--sd3_max_seq_len", type=int, default=256,
                   help="SD3 only: T5 tokens kept. MUST equal the fdeval run's "
                        "--sd3_max_seq_len, or the two are not comparable: a "
                        "different padded length changes the token count the "
                        "transformer sees and hence every image.")
    p.add_argument("--guidance_scale", type=float, default=0.0,
                   help="0.0 for schnell and for SD3.5-Large-Turbo (both are "
                        "distilled and guidance-free)")

    g = p.add_argument_group("arms")
    g.add_argument("--cads_s", type=float, nargs="*", default=[0.25],
                   help="CADS noise scale(s); one arm per value, named cads_n<s>. "
                        "Pass with no values to drop the CADS arm. The paper's "
                        "text-to-image setting is 0.25, tuned at 100 steps -- at 4 "
                        "steps a quarter of the trajectory is unconditional rather "
                        "than a tenth, so sweep downward from it.")
    g.add_argument("--cads_tau1", type=float, default=0.6)
    g.add_argument("--cads_tau2", type=float, default=0.9)
    g.add_argument("--cads_psi", type=float, default=1.0)
    g.add_argument("--cads_no_pooled", action="store_true",
                   help="corrupt only the T5 sequence, leaving the pooled CLIP "
                        "projection clean. Both are 'the condition' on FLUX and the "
                        "paper's single-encoder models had no such split, so this is "
                        "a judgement call rather than a settled detail.")
    g.add_argument("--baseline", action="store_true",
                   help="ALSO generate an unperturbed `baseline_plain` arm. Off by "
                        "default: if you already have one from an fdeval run at the "
                        "same seed, steps and resolution, it is bit-identical and "
                        "regenerating it is wasted compute -- merge the folders "
                        "instead. Turn it on for a self-contained run.")
    g.add_argument("--baseline_only", action="store_true",
                   help="generate only the baseline arm and no CADS arms")
    sc = p.add_argument_group("scoring (optional)")
    sc.add_argument("--score", action="store_true",
                    help="score each group in-process, exactly as fdeval does at "
                         "generation time, instead of leaving it to "
                         "`rescore_from_images.py`. Turning this on is what lets "
                         "--save_images_per_group be a pure disk knob. It imports "
                         "fdeval's ScorerBank -- the ONLY dependency this file has "
                         "on that tree, and a deliberate one: the alternative is a "
                         "second implementation of nine scorers that would drift "
                         "from the one your method is measured with.")
    sc.add_argument("--metrics",
                    default="clip,dino,inception,lpips,pickscore,hpsv2,imagereward",
                    help="comma-separated, same names as run_eval.py --metrics")
    sc.add_argument("--scorer_device", default=None,
                    help="GPU for the scorers. Defaults to --device, i.e. FLUX and "
                         "every scorer on one card -- fine at 80GB, tight at 48 "
                         "once HPSv2/ImageReward/Aesthetic are in the list. Point "
                         "it at a second card (--device cuda:0 --scorer_device "
                         "cuda:1) to split them; both live in this one process, "
                         "and images cross as numpy so there is no device "
                         "mismatch to get wrong.")
    sc.add_argument("--match_scorer_from", default=None, metavar="RUN_DIR",
                    help="read that run's manifest.json and adopt its "
                         "scorer-critical settings. STRONGLY RECOMMENDED: the four "
                         "settings below bake into the cached arrays, so a "
                         "baseline scored under different ones is not on a common "
                         "scale with your method and every mean, CI and paired "
                         "test mixing them is wrong while looking healthy. "
                         "fdeval refuses such a resume; this arm lands in a "
                         "separate folder, so nothing would catch it but this.")
    sc.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    sc.add_argument("--dino_model_id", default="facebook/dinov2-base")
    sc.add_argument("--clip_score_w", type=float, default=100.0)
    sc.add_argument("--lpips_resize", type=int, default=0)
    sc.add_argument("--lpips_net", default="alex")
    sc.add_argument("--cache_dir", default="./eval_cache")
    sc.add_argument("--batch_size", type=int, default=32)
    sc.add_argument("--hps_checkpoint", default=None)
    sc.add_argument("--clipiqa_pairs", default="quality")
    sc.add_argument("--clipiqa_full_res", action="store_true")
    sc.add_argument("--skip_unavailable_metrics", action="store_true")

    p.add_argument("--overwrite", action="store_true",
                   help="regenerate groups that already have images on disk")
    p.add_argument("--limit", type=int, default=0,
                   help="only the first N prompts, for a smoke test")
    p.add_argument("--dry_run", action="store_true",
                   help="print the arms, the per-step gamma schedule and the image "
                        "count, then exit without loading the model")
    return p.parse_args()


def _family(a) -> str:
    """Model family for this run: explicit --backend, else inferred."""
    if getattr(a, "backend", None):
        return a.backend
    mid = (a.model_id or "").lower()
    if "flux" in mid:
        return "flux"
    if "stable-diffusion-3" in mid or "/sd3" in mid or mid.endswith("sd3"):
        return "sd3"
    raise SystemExit(
        f"cannot infer a model family from --model_id {a.model_id!r}; "
        f"pass --backend flux|sd3.")


def schedule_preview(steps: int, h: int, w: int, tau1: float, tau2: float,
                     s: float, family: str = "flux",
                     shift: float = 3.0) -> List[tuple]:
    """
    The gamma schedule this run will apply. Printed by --dry_run so the
    unconditional-first-step behaviour is visible BEFORE a GPU is touched.

    DERIVED, NOT READ. --dry_run deliberately touches no weights, so the sigmas
    here are recomputed from the family's published formula rather than taken
    from the model's scheduler config. The real run prints the scheduler's own
    sigmas once the pipe is up; if the two ever disagree, believe the run.

      flux -- dynamic shifting: mu from the image sequence length, then
              sigma = e^mu / (e^mu + 1/t - 1).
      sd3  -- static shift (SD3/3.5 ship use_dynamic_shifting=false, shift=3):
              sigma = shift*t / (1 + (shift-1)*t).
    """
    if steps == 1:
        sig = [1.0]
    else:
        lin = [1.0 - i * (1.0 - 1.0 / steps) / (steps - 1) for i in range(steps)]
        if family == "sd3":
            sig = [shift * t / (1.0 + (shift - 1.0) * t) for t in lin]
        else:
            seq = (2 * (h // 16) // 2) * (2 * (w // 16) // 2)
            m = (1.15 - 0.5) / (4096 - 256)
            mu = seq * m + 0.5 - 256 * m
            sig = [math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0)) for t in lin]
    out = []
    for i, t in enumerate(sig):
        g = cads_gamma(t, tau1, tau2)
        out.append((i, t, g, math.sqrt(g), s * math.sqrt(1.0 - g)))
    return out


def main() -> int:
    a = parse_args()

    # Arm names encode every schedule parameter that is off the paper's
    # default, so two runs that differ only in tau2 or psi cannot land on the
    # same condition directory and silently overwrite one another. The swept
    # value stays last as `_nS` because `analysis.parse_condition_name` splits
    # on the final "_n" -- so `cads_t2-1.2_n0.25` parses as method
    # `cads_t2-1.2` at 0.25 and gets its own curve, which is what you want when
    # comparing schedules.
    tag = ""
    if abs(a.cads_tau1 - 0.6) > 1e-9:
        tag += f"_t1-{a.cads_tau1:g}"
    if abs(a.cads_tau2 - 0.9) > 1e-9:
        tag += f"_t2-{a.cads_tau2:g}"
    if abs(a.cads_psi - 1.0) > 1e-9:
        tag += f"_psi-{a.cads_psi:g}"
    if a.cads_no_pooled:
        tag += "_nopooled"

    arms: List[Dict] = []
    if not a.baseline_only:
        for s in (a.cads_s or []):
            arms.append({"name": f"cads{tag}_n{s:g}", "kind": "cads",
                         "s": float(s)})
    if a.baseline or a.baseline_only:
        arms.insert(0, {"name": "baseline_plain", "kind": "plain"})
    if not arms:
        raise SystemExit(
            "no arms selected: pass --cads_s with at least one value, or "
            "--baseline / --baseline_only.")

    # Refuse the configuration that silently yields unconditional samples.
    if a.steps == 1 and any(x["kind"] == "cads" for x in arms):
        if cads_gamma(1.0, a.cads_tau1, a.cads_tau2) <= 0.0:
            raise SystemExit(
                f"CADS at --steps 1 with tau2={a.cads_tau2} gives gamma(1)=0: the "
                f"single step would be FULLY UNCONDITIONAL and the arm would "
                f"measure prompt-free generation, not CADS. Use --steps >= 2, or "
                f"set --cads_tau2 > 1 (e.g. 1.2 -> gamma(1)="
                f"{cads_gamma(1.0, a.cads_tau1, 1.2):.3f}) and report it as an "
                f"adaptation, since no published tau2 exceeds 1.0.")

    # How many images per group actually get generated AND kept. They are the
    # same number here by construction: nothing scores an image that was not
    # written, so generating one to throw away would be pure waste.
    # n_img: images GENERATED and scored.  n_save: images kept on disk.
    # They diverge only under --score, where the group is scored in memory
    # before anything is written.
    n_img = a.num_images
    n_save = (a.num_images if a.save_images_per_group <= 0
              else min(a.save_images_per_group, a.num_images))
    if n_save < a.num_images and not a.score:
        n_img = n_save
        print(f"[baselines] !! --save_images_per_group {n_save} < --num_images "
              f"{a.num_images} without --score: nothing scores an image that was "
              f"not written, so each group will BE {n_save} images and every "
              f"diversity metric is n-dependent. NOT comparable with anything "
              f"scored over {a.num_images}. Add --score to keep the full group "
              f"and still write only {n_save} images.")
    elif n_save < a.num_images:
        print(f"[baselines] scoring all {n_img} images per group, writing "
              f"{n_save} to disk")

    if a.match_scorer_from:
        mpath = os.path.join(a.match_scorer_from, "manifest.json")
        try:
            prev = json.load(open(mpath)).get("args", {})
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"--match_scorer_from: cannot read {mpath}: {e}")
        for k in ("clip_model_id", "dino_model_id", "clip_score_w",
                  "lpips_resize", "lpips_net"):
            if k in prev and getattr(a, k, None) != prev[k]:
                print(f"[baselines] scorer setting {k}: {getattr(a, k)!r} -> "
                      f"{prev[k]!r} (from {a.match_scorer_from})")
                setattr(a, k, type(getattr(a, k))(prev[k]))

    with open(a.prompt_set) as f:
        prompts = [{"uid": d["uid"], "text": d["text"]} for d in json.load(f)]
    if a.limit:
        prompts = prompts[:a.limit]

    print(f"[baselines] {len(arms)} arm(s): {[x['name'] for x in arms]}")
    print(f"[baselines] {len(prompts)} prompts x {n_img} images x "
          f"{len(arms)} arms = {len(prompts) * n_img * len(arms):,} images")
    if any(x["kind"] == "cads" for x in arms):
        s0 = next(x["s"] for x in arms if x["kind"] == "cads")
        print(f"\n[cads] tau1={a.cads_tau1} tau2={a.cads_tau2} psi={a.cads_psi}, "
              f"schedule at s={s0:g}:")
        print(f"  {'step':>4} {'t=sigma':>8} {'gamma':>7} {'signal':>7} {'noise':>7}")
        for i, t, g, sw, nw in schedule_preview(
                a.steps, a.height, a.width, a.cads_tau1, a.cads_tau2, s0,
                family=_family(a), shift=a.sd3_shift):
            tag = "  <-- UNCONDITIONAL" if g <= 0 else ""
            print(f"  {i:>4} {t:>8.4f} {g:>7.4f} {sw:>7.3f} {nw:>7.3f}{tag}")
    if a.dry_run:
        print("\n[baselines] --dry_run: stopping before model load.")
        return 0

    import torch
    from PIL import Image

    if _family(a) == "sd3":
        from diffusers import StableDiffusion3Pipeline as _Pipe
    else:
        from diffusers import FluxPipeline as _Pipe

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    print(f"\n[baselines] loading {a.model_id} ({a.dtype}) on {a.device} "
          f"[backend={_family(a)}]")
    pipe = _Pipe.from_pretrained(a.model_id, torch_dtype=dt)
    if a.offload_device:
        # Piecewise, because `.to(device)` moves EVERY component at once and
        # 27.4 GB does not fit on a 24 GB card. The VAE stays with the
        # transformer: `pipe()` decodes on its execution device, and at 0.17 GB
        # it is far cheaper to co-locate than to work around.
        pipe.transformer.to(a.device)
        pipe.vae.to(a.device)
        moved = []
        for _n in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            _m = getattr(pipe, _n, None)
            if _m is not None and hasattr(_m, "to"):
                _m.to(a.offload_device)
                moved.append(_n)
        # `pipe()` places latents on `_execution_device`, which resolves to the
        # FIRST component's device -- a text encoder, i.e. the offload card --
        # while the transformer is on --device. Pin it, or every call dies in
        # pos_embed with a device mismatch. Set for the process; nothing here
        # needs the original.
        type(pipe)._execution_device = property(
            lambda self, _d=torch.device(a.device): _d)
        print(f"[baselines] transformer + vae -> {a.device}; "
              f"{', '.join(moved)} -> {a.offload_device}")
    else:
        pipe = pipe.to(a.device)
    pipe.set_progress_bar_config(disable=True)

    # The scheduler's OWN sigmas, now that there is a real one. schedule_preview
    # above is a derivation; this is the thing that will actually run, and CADS
    # is a function of sigma, so a mismatch here changes gamma at every step.
    try:
        pipe.scheduler.set_timesteps(a.steps, device=a.device)
        _sig = [float(x) for x in pipe.scheduler.sigmas[:a.steps]]
        print(f"[baselines] scheduler sigmas: "
              f"{', '.join(f'{x:.4f}' for x in _sig)}")
        print(f"[baselines] cads gamma:       "
              f"{', '.join(f'{cads_gamma(x, a.cads_tau1, a.cads_tau2):.4f}' for x in _sig)}")
    except Exception as e:  # pragma: no cover
        print(f"[baselines] [!] could not read scheduler sigmas: {e!r}")

    # The callback may only rewrite tensors the pipeline declares. FLUX declares
    # latents and prompt_embeds but not the pooled projection, so it is added
    # here -- otherwise the pooled half of the condition would stay clean while
    # the T5 half is annealed, which is neither the paper's method nor a
    # coherent ablation of it.
    want_pooled = not a.cads_no_pooled
    cb_inputs = list(getattr(pipe, "_callback_tensor_inputs", ["latents", "prompt_embeds"]))
    if want_pooled and "pooled_prompt_embeds" not in cb_inputs:
        cb_inputs.append("pooled_prompt_embeds")
        pipe._callback_tensor_inputs = cb_inputs
    if want_pooled and "pooled_prompt_embeds" not in pipe._callback_tensor_inputs:
        print("[cads] this diffusers version will not expose pooled_prompt_embeds "
              "to the callback; corrupting the T5 sequence only.")
        want_pooled = False

    bank = None
    if a.score:
        # The single fdeval import in this file, and only on this path. Using
        # their ScorerBank rather than a second implementation is the whole
        # point: the baseline is then measured by byte-identical code to the
        # method, and cannot silently drift onto a different CLIP backbone,
        # LPIPS resolution or CLIPScore multiplier.
        from fdeval.scorers import ScorerBank
        from fdeval.scorers_config import METRIC_KEYS, ScorerConfig

        scfg = ScorerConfig(
            metrics=a.metrics, clip_model_id=a.clip_model_id,
            dino_model_id=a.dino_model_id, lpips_net=a.lpips_net,
            lpips_resize=a.lpips_resize, clip_score_w=a.clip_score_w,
            hps_checkpoint=a.hps_checkpoint, cache_dir=a.cache_dir,
            batch_size=a.batch_size, skip_unavailable=a.skip_unavailable_metrics,
            clipiqa_pairs=a.clipiqa_pairs, clipiqa_full_res=a.clipiqa_full_res,
        )
        sdev = torch.device(a.scorer_device or a.device)
        print(f"[baselines] building scorers on {sdev}: {a.metrics}")
        bank = ScorerBank(scfg, sdev)
        # Catches a scorer that raises inside the loop before a multi-hour run
        # burns itself producing groups that all failed the same way.
        bank.self_test()

    device = torch.device(a.device)
    n_done = n_skip = 0

    for arm in arms:
        for pr in prompts:
            uid, text = pr["uid"], pr["text"]
            idir = group_image_dir(a.out_dir, arm["name"], uid)
            npz = group_feature_path(a.out_dir, arm["name"], uid)
            if (not a.overwrite) and os.path.isdir(idir) and os.path.exists(npz):
                if len([f for f in os.listdir(idir)
                        if f.lower().endswith(".png")]) >= min(n_save, n_img):
                    n_skip += 1
                    continue
            os.makedirs(idir, exist_ok=True)

            # Seeds 0..n_img-1: seed i does not depend on the group size, so a
            # smaller group is a PREFIX of the full one and every image still
            # pairs with the same image of an fdeval run.
            seeds = [group_seed(a.seed, uid, i) for i in range(n_img)]
            group_imgs = []
            for i, sd in enumerate(seeds):
                gen = torch.Generator(device="cpu").manual_seed(sd)
                kw = dict(prompt=text, height=a.height, width=a.width,
                          num_inference_steps=a.steps,
                          guidance_scale=a.guidance_scale, generator=gen,
                          output_type="pil")

                # With the encoders on another card, `pipe()` cannot call them
                # itself (it would pass them inputs on --device), so EVERY arm
                # has to be fed embeddings -- not just CADS, which needs them
                # anyway. Encode on the encoders' card, then move the (small)
                # result to the transformer's.
                if a.offload_device and arm["kind"] != "cads":
                    _ed = torch.device(a.offload_device)
                    if _family(a) == "sd3":
                        _pe, _, _ppe, _ = pipe.encode_prompt(
                            prompt=text, prompt_2=text, prompt_3=text,
                            device=_ed, num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            max_sequence_length=a.sd3_max_seq_len)
                        kw["max_sequence_length"] = a.sd3_max_seq_len
                    else:
                        _pe, _ppe, _ = pipe.encode_prompt(
                            prompt=text, prompt_2=None, device=_ed,
                            num_images_per_prompt=1, max_sequence_length=512)
                    kw["prompt_embeds"] = _pe.to(device)
                    kw["pooled_prompt_embeds"] = _ppe.to(device)
                    kw.pop("prompt")

                if arm["kind"] == "cads":
                    # Noise for the conditioning is drawn from its own stream so
                    # it can never consume draws from `gen` and shift the initial
                    # latent -- that would break the pairing with the fdeval run.
                    cgen = torch.Generator(device=device).manual_seed(sd + 777_777)
                    s, t1, t2, psi = arm["s"], a.cads_tau1, a.cads_tau2, a.cads_psi

                    enc_dev = (torch.device(a.offload_device)
                               if a.offload_device else device)
                    if _family(a) == "sd3":
                        pe, _, ppe, _ = pipe.encode_prompt(
                            prompt=text, prompt_2=text, prompt_3=text,
                            device=enc_dev, num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            max_sequence_length=a.sd3_max_seq_len)
                        kw["max_sequence_length"] = a.sd3_max_seq_len
                    else:
                        pe, ppe, _ = pipe.encode_prompt(
                            prompt=text, prompt_2=None, device=enc_dev,
                            num_images_per_prompt=1, max_sequence_length=512)
                    # The CADS corruption and the callback both run on the
                    # transformer's card, so move the embeddings once here
                    # rather than per step.
                    pe, ppe = pe.to(device), ppe.to(device)
                    # Step 0's conditioning must be corrupted BEFORE the pipeline
                    # runs: callback_on_step_end fires at the END of a step, so it
                    # can only ever prepare the NEXT one. sigma_0 is 1.0 exactly
                    # under the schnell schedule (time_shift fixes 1.0).
                    kw["prompt_embeds"] = cads_corrupt(pe, 1.0, s, t1, t2, psi, cgen)
                    kw["pooled_prompt_embeds"] = (
                        cads_corrupt(ppe, 1.0, s, t1, t2, psi, cgen)
                        if want_pooled else ppe)
                    kw.pop("prompt")

                    def _cb(pl, step, timestep, cbk, _pe=pe, _ppe=ppe,
                            _s=s, _t1=t1, _t2=t2, _psi=psi, _g=cgen):
                        j = step + 1
                        sig = pl.scheduler.sigmas
                        if j >= len(sig):
                            return cbk
                        t_next = float(sig[j])
                        cbk["prompt_embeds"] = cads_corrupt(
                            _pe, t_next, _s, _t1, _t2, _psi, _g)
                        if want_pooled and "pooled_prompt_embeds" in cbk:
                            cbk["pooled_prompt_embeds"] = cads_corrupt(
                                _ppe, t_next, _s, _t1, _t2, _psi, _g)
                        return cbk

                    kw["callback_on_step_end"] = _cb
                    kw["callback_on_step_end_tensor_inputs"] = list(
                        pipe._callback_tensor_inputs)

                img = pipe(**kw).images[0]
                if i < n_save:
                    img.save(os.path.join(idir, f"img_{i:03d}.png"))
                if bank is not None:
                    group_imgs.append(np.asarray(img, dtype=np.uint8))

            meta = {
                "prompt_uid": uid, "prompt_text": text, "condition": arm["name"],
                # What rescore_from_images checks the PNG count against, so it
                # records the images that EXIST, not the count that was asked
                # for -- otherwise every group would look truncated.
                "num_images": len(seeds), "num_images_requested": a.num_images,
                "seeds": seeds,
                "num_inference_steps": a.steps,
                "height": a.height, "width": a.width,
                "nfe_per_image": a.steps,   # CADS adds no forward passes
                "source": "baselines/cads.py",
                "cads": ({"s": arm["s"], "tau1": a.cads_tau1, "tau2": a.cads_tau2,
                          "psi": a.cads_psi, "corrupt_pooled": bool(want_pooled)}
                         if arm["kind"] == "cads" else None),
                # Names the scorers that have NOT been computed, which is how
                # rescore_from_images decides there is work to do here.
                "metrics_unavailable": ["all"],
            }
            if bank is None:
                write_stub_npz(npz, meta)
            else:
                feats = bank.score_group(group_imgs, text)
                # Requested-but-absent means the model would not load. Recorded
                # the same way fdeval records it so `group_is_complete` does not
                # treat these groups as perpetually unfinished.
                produced = set(feats)
                meta["metrics_unavailable"] = sorted(
                    m for m in a.metrics.split(",")
                    if m.strip() and not set(METRIC_KEYS.get(m.strip(), ())) & produced)
                meta["scorer_provenance"] = bank.provenance()
                meta["images_saved"] = int(min(n_save, n_img))
                payload = dict(feats)
                payload["_meta"] = np.frombuffer(
                    json.dumps(meta).encode("utf-8"), dtype=np.uint8)
                os.makedirs(os.path.dirname(npz), exist_ok=True)
                tmp = npz + ".tmp.npz"
                np.savez_compressed(tmp, **payload)
                os.replace(tmp, npz)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {arm['name']}: {n_done} groups written")

    print(f"\n[baselines] wrote {n_done} group(s), skipped {n_skip} already present")
    if bank is None:
        print(f"[baselines] next:\n"
              f"  python rescore_from_images.py --run_dir {a.out_dir} "
              f"--metrics <list>\n"
              f"  python analyze.py --run_dir {a.out_dir} "
              f"--prompt_set {a.prompt_set}")
    else:
        print(f"[baselines] scored in-process; next:\n"
              f"  python analyze.py --run_dir {a.out_dir} "
              f"--prompt_set {a.prompt_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())