# Set-level comparison

Unmodified sampler, OSCAR, JIVE, CADS, and JIVE+CADS on FLUX.1-schnell over PartiPrompts. All `jive_flux` arms share per-image starting latents (`seed + i`); CADS is a separate generator (`../baselines/cads.py`) under the same prompts, seed, and step count.

See the top-level [README](../README.md) for environment variables and the paper-table launchers.

```bash
# one prompt
python -m jive_flux --model schnell --prompt "..." --arms deterministic oscar jive

# PartiPrompts, one challenge aspect (array task 0 = Basic)
sbatch --array=0 --partition=<partition> --account=<account> scripts/run_parti_base_oscar_jive.sh

python analysis/aggregate_parti.py
python analysis/jive_bands_table.py
```
