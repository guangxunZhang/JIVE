# JIVE

JIVE diversifies a flow-matching sampler by moving each starting latent along the top singular directions of the endpoint Jacobian

```
D_t(z) = z − σ_t v_θ(z, t)
```

then running the **same** deterministic denoising as the unmodified sampler. The only difference from the base sampler is that one-shot start projection.

This folder is a self-contained copy of the comparison code: the five set-level methods (unmodified sampler, OSCAR, JIVE, CADS, JIVE+CADS) and the local-level (stroke-to-image) comparison on SDEdit, Boomerang, and RF-Inversion.

## Layout

```
JIVE/
├── core/                 shared metrics, noise projection, OSCAR volume builder
├── baselines/
│   ├── oscar/            OSCAR volume objective (authors' package)
│   ├── cads.py           CADS (Sadat et al., ICLR 2024)
│   └── rf_inversion/     RF-Inversion SDE pipeline (Rout et al.)
├── set_level/            text-to-image comparison on FLUX + PartiPrompts
│   ├── jive_flux/        unmodified / OSCAR / JIVE / JIVE+CADS arms
│   ├── specs/            PartiPrompts (1,632 prompts, 11 challenge aspects)
│   ├── scripts/          SLURM launchers, one per table row
│   └── analysis/         aggregation + LaTeX tables
└── local_level/          stroke-to-image comparison on FLUX
    ├── run_flux_arm.py   SDEdit + Boomerang (baseline and +JIVE)
    ├── run_rf_inverse.py RF-Inversion (baseline and +JIVE)
    ├── common/           metrics, data, subspace estimator
    └── scripts/
```

## Environment

Python with `torch`, `diffusers`, `transformers`, `open_clip` (or OpenAI CLIP), and the usual image stack (`pillow`, `torchvision`). Optional quality scorers: `pyiqa` (CLIP-IQA), `hpsv2`.

Checkpoints and scoring live behind environment variables so nothing in this tree hard-codes a user or machine:

| Variable | Used for |
|---|---|
| `FLUX_ROOT` | directory containing `FLUX.1-dev/` and `FLUX.1-schnell/` |
| `FLUX_MODEL` | local-level backbone (path or HuggingFace id) |
| `JIVE_VENV` | virtualenv to activate in the launchers |
| `FDEVAL_ROOT` | optional [fdeval](https://github.com) scoring harness (feature Vendi / CADS `--score`) |

Without `FLUX_ROOT`, set-level falls back to the HuggingFace repo ids. Without `FDEVAL_ROOT`, pass `--vendi-feature pixel` or score CADS images separately.

## Set-level (text-to-image)

Five methods, identical starting latents, FLUX.1-schnell, 512×512, 4 steps, 16 images per prompt, PartiPrompts (1,632 prompts).

| Method | How it is generated |
|---|---|
| Unmodified sampler | ARM A in `jive_flux` (`--arms deterministic`) |
| OSCAR | ARM B (`--arms oscar`) |
| JIVE | ARM C (`--arms jive`). `--jive-iter-mode j` or `jtj` |
| CADS | standalone `baselines/cads.py` |
| JIVE+CADS | ARM D (`--arms jive_cads`): JIVE start + CADS during denoising |

`--jive-iter-mode j` iterates `Q ← QR(JQ)` (block power method on `J`). `--jive-iter-mode jtj` iterates `Q ← QR(JᵀJQ)`, whose eigenvectors are the right singular vectors of `J`.

### Single prompt

From `set_level/`:

```bash
python -m jive_flux \
  --model schnell \
  --prompt "A steaming cup of coffee next to an open book on a rainy day" \
  --height 512 --width 512 --n-images 16 --keep-images-per-arm 16 \
  --G 4 --seeds 42 --inject-norms 12.0 --inject-n 4 \
  --arms deterministic oscar jive \
  --enable-vae-tiling
```

### PartiPrompts table (cluster)

Launchers live in `set_level/scripts/`. Submit from `set_level/` so logs and outputs land next to the code. Each array job is one PartiPrompts challenge aspect (11 tasks).

```bash
cd set_level
sbatch --partition=<partition> --account=<account> scripts/run_parti_base_oscar_jive.sh  # unmodified, OSCAR, JIVE(J)
sbatch --partition=<partition> --account=<account> scripts/run_parti_jive_jtj.sh         # JIVE(JᵀJ)
sbatch --partition=<partition> --account=<account> scripts/run_parti_cads.sh             # CADS τ₂ = 1.2 and 1.8
sbatch --partition=<partition> --account=<account> scripts/run_parti_jive_cads.sh        # JIVE(J)+CADS
sbatch --partition=<partition> --account=<account> scripts/run_parti_jive_jtj_cads.sh    # JIVE(JᵀJ)+CADS
```

Then:

```bash
python analysis/aggregate_parti.py
python analysis/aggregate_parti.py --src outputs_jive_jtj_n12
python analysis/jive_bands_table.py   # eight-row paper table
python analysis/make_main_table.py    # three-arm summary from one tree
```

JIVE-alone uses inject-norm 12. JIVE+CADS uses inject-norm 4 and CADS `(τ₁, τ₂, s, ψ) = (0.6, 1.2, 0.15, 1.0)`. `τ₂ > 1` keeps every step of schnell's 4-step grid at `γ > 0`; the paper default `τ₂ = 0.9` would make the first step fully unconditional.

## Local-level (stroke-to-image)

Same FLUX backbone for every method so KID and Vendi are comparable. Each job runs the **baseline** and **+JIVE** on the same sources and seeds; `inject_norm → 0` reproduces the baseline.

| Method | Driver | Sweep |
|---|---|---|
| SDEdit (Meng et al.) | `run_flux_arm.py --method sdedit` | `t₀ ∈ {0.5, 0.7}` |
| Boomerang (Luzi et al.) | `run_flux_arm.py --method boomerang` | same `t₀` |
| RF-Inversion (Rout et al.) | `run_rf_inverse.py` | `η ∈ {0.9, 0.5}` |

+JIVE injects into the top-4 Jacobian subspace (`inject_norm ∈ {4, 8, 12}`). RF-Inversion defaults to the set-level additive draw (`--perturb_mode additive`); `--jive_iter_mode jtj` iterates `Q ← QR(JᵀJQ)`. Scenes are LSUN classroom / kitchen / conference_room / dining_room / restaurant.

```bash
cd local_level
sbatch --partition=<partition> --account=<account> scripts/download_stroke2image_data.sh
bash scripts/submit_stroke2image.sh <partition>
python aggregate_results.py --out_root out_stroke
```

`matched_pairs.md` pairs every +JIVE point with the baseline point of closest L2 (faithfulness). Positive ΔVendi at matched L2 is a diversity gain that does not cost fidelity to the painting.

## Algorithm files

| What | Where |
|---|---|
| JIVE subspace (FLUX, set-level) | `set_level/jive_flux/jive_subspace.py` |
| JIVE arm / JIVE+CADS arm | `jive_arm.py`, `jive_cads_arm.py` |
| JIVE subspace (local-level, any endpoint) | `local_level/common/volume_expansion.py` |
| Projected Gaussian draw (set-level) | `core/noise_projection.py` |
| OSCAR | `baselines/oscar/`, wired in `jive_flux/oscar_arm.py` |
| CADS | `baselines/cads.py` (`cads_gamma` / `cads_corrupt` reused by JIVE+CADS) |
| RF-Inversion | `baselines/rf_inversion/` |
