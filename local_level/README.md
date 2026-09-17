# Local-level comparison

Stroke-to-image on FLUX: SDEdit, Boomerang, and RF-Inversion, each as a baseline and with +JIVE. Inputs are dab paintings of LSUN church / bedroom photos.

See the top-level [README](../README.md) for the metric table and cluster commands.

```bash
sbatch --partition=<partition> --account=<account> scripts/download_stroke2image_data.sh
bash scripts/submit_stroke2image.sh <partition>
python aggregate_results.py --out_root out_stroke
```
