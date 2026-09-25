"""Batch-convert clean LSUN sources into coarse stroke paintings.

Turns data/<dataset>/sources/*.png (real photos) into the Stroke2Image
conditional input: a flat color-stroke painting that keeps only the coarse
color layout (sky / building mass / ground), exactly as in the SDEdit paper's
stroke-based image generation (Meng et al., ICLR 2022 — the model must recover
a realistic photo from color blobs, guided here by the text prompt).

Two painting styles (both from stroke_painting.py, imported unchanged):
    --style dabs    discrete brush dabs (default) — mimics the hand-painted
                    stroke inputs shipped with the official SDEdit repo
    --style mosaic  flat quantized color blobs (the original version)

Output layout mirrors what common/experiment.load_sources expects:
    <out_root>/<dataset>/sources/src_000.png ...

The KID reference set is NOT produced here: Stroke2Image is scored against
REAL photos, so point <out_root>/<dataset>/reference at the existing
data/<dataset>/reference directory (symlink) — see run_stroke2image.sh.
"""
import argparse
import os
import sys

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from stroke_painting import stroke_dabs, stroke_paint  # noqa: E402

from common.prepare_data import DATASETS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--data_root", default="data",
                   help="Root holding the clean sources")
    p.add_argument("--out_root", default="data_stroke",
                   help="Root for the painted sources")
    p.add_argument("--style", choices=["dabs", "mosaic"], default="dabs")
    p.add_argument("--grid", type=int, default=32,
                   help="Coarse color-field grid (smaller = more abstract)")
    p.add_argument("--colors", type=int, default=16, help="Palette size")
    p.add_argument("--n_dabs", type=int, default=1400,
                   help="Number of brush dabs (dabs style only)")
    p.add_argument("--blur", type=float, default=1.0,
                   help="Final Gaussian blur radius")
    p.add_argument("--smear", type=int, default=0, help="mosaic style only")
    p.add_argument("--smear_angle", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; per-source seed = seed + index")
    args = p.parse_args()

    src_dir = os.path.join(args.data_root, args.dataset, "sources")
    out_dir = os.path.join(args.out_root, args.dataset, "sources")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
    if not files:
        raise RuntimeError(f"No sources in {src_dir}; run prepare_data first.")

    for i, f in enumerate(files):
        out_path = os.path.join(out_dir, f)
        if os.path.isfile(out_path):
            continue
        img = Image.open(os.path.join(src_dir, f)).convert("RGB")
        if args.style == "dabs":
            painted = stroke_dabs(img, grid=args.grid, n_colors=args.colors,
                                  n_dabs=args.n_dabs, blur=args.blur,
                                  seed=args.seed + i)
        else:
            painted = stroke_paint(img, args.grid, args.colors, args.blur,
                                   args.smear, args.smear_angle)
        painted.save(out_path)
    print(f"Painted {len(files)} sources -> {out_dir} "
          f"(style={args.style}, grid={args.grid}, colors={args.colors}, "
          f"blur={args.blur})")


if __name__ == "__main__":
    main()
