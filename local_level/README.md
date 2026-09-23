# Local-level comparison

Stroke-to-image on FLUX: SDEdit, Boomerang, and RF-Inversion, each as a baseline and with +JIVE. Inputs are dab paintings of LSUN classroom, kitchen, conference room, dining room, and restaurant photos.

JIVE on RF-Inversion supports `--jive_iter_mode {j,jtj}` (block power on `J` vs `J^T J`) and `--perturb_mode {additive,boundary}` (exact L2-norm projected noise vs sphere rotation). `--inject_after_eta` adds the offset at the first reverse step with `eta_t=0`.

See the top-level [README](../README.md) for the metric table and cluster commands.

```bash
sbatch --partition=<partition> --account=<account> scripts/download_stroke2image_data.sh
bash scripts/submit_stroke2image.sh <partition>
python aggregate_results.py --out_root out_stroke

# RF-Inversion only: JIVE(J^T J), additive norm, inject after the eta window
PARTITION=<partition> DATASET=classroom bash scripts/run_rf_inverse_aftereta.sh
```
