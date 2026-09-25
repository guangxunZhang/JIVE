"""Turn a real photo into a coarse color-stroke painting (Stroke2Image input).

The output keeps the scene's coarse color layout (sky on top, building mass
in the middle, ground below) but destroys all photographic detail, mimicking
the abstract stroke paintings used as the conditional input in Stroke2Image:
the model must recover a realistic photo from these color blobs + the prompt.

Pipeline:
  1. downsample to a tiny grid (grid x grid)         -> coarse color regions
  2. median-cut quantization to n_colors             -> flat "paint" palette
  3. upscale back, optional directional smear        -> brushed look
  4. light Gaussian blur                             -> soft stroke edges

Example:
  python stroke_painting.py \
      --image data/classroom/sources/src_000.png \
      --out images/classroom_stroke_src_000.png --grid 24 --colors 12 --blur 2
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def directional_smear(img, length, angle_deg):
    """Cheap directional smear: average copies of the image shifted along one
    direction, giving a brushed-stroke feel. length <= 1 is a no-op."""
    if length <= 1:
        return img
    rad = np.deg2rad(angle_deg)
    dx, dy = float(np.cos(rad)), float(np.sin(rad))
    acc = np.asarray(img).astype(np.float32)
    count = 1
    for i in range(1, length):
        shifted = img.transform(
            img.size, Image.AFFINE, (1, 0, -dx * i, 0, 1, -dy * i),
            resample=Image.BILINEAR,
        )
        acc += np.asarray(shifted).astype(np.float32)
        count += 1
    return Image.fromarray(np.clip(acc / count, 0, 255).astype(np.uint8))


def stroke_paint(img, grid, n_colors, blur, smear, smear_angle):
    small = img.resize((grid, grid), Image.BICUBIC)
    q = small.quantize(colors=n_colors, method=Image.MEDIANCUT,
                       dither=Image.Dither.NONE)
    out = q.convert("RGB").resize(img.size, Image.BICUBIC)
    out = directional_smear(out, smear, smear_angle)
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def stroke_dabs(img, grid=32, n_colors=16, n_dabs=1400, dab_width=(7, 12),
                dab_ratio=(2.0, 3.5), jitter=8.0, alpha=235, blur=1.0,
                seed=0):
    """Render `img` as discrete brush dabs, mimicking the hand-painted stroke
    inputs shipped with the official SDEdit repo (visible dabs, per-dab color
    jitter, slight canvas show-through).

    1. coarse color field: downsample to grid x grid, median-cut quantize to
       n_colors, upsample smooth -> paint-like palette with regional variation
    2. smooth random orientation field (low-frequency angle map)
    3. scatter short rounded dabs colored from the field (+ jitter) in random
       order over a blurred base of the field, so thin gaps read as canvas
    """
    rng = np.random.default_rng(seed)
    w, h = img.size

    small = img.resize((grid, grid), Image.BICUBIC)
    q = small.quantize(colors=n_colors, method=Image.MEDIANCUT,
                       dither=Image.Dither.NONE)
    field = np.asarray(
        q.convert("RGB").resize((w, h), Image.BICUBIC)).astype(np.float32)

    base = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))
    base = base.filter(ImageFilter.GaussianBlur(max(blur * 2.0, 1.0)))
    canvas = base.convert("RGBA")

    coarse_angles = rng.uniform(0.0, np.pi, size=(4, 4))
    ang_img = Image.fromarray((coarse_angles * 255 / np.pi).astype(np.uint8))
    ang_img = ang_img.resize((w, h), Image.BICUBIC)
    angles = np.asarray(ang_img).astype(np.float32) * np.pi / 255.0

    draw = ImageDraw.Draw(canvas, "RGBA")
    ys = rng.uniform(0, h, n_dabs)
    xs = rng.uniform(0, w, n_dabs)
    for y, x in zip(ys, xs):
        dab_w = rng.uniform(*dab_width)
        half = 0.5 * dab_w * rng.uniform(*dab_ratio)
        a = angles[int(min(y, h - 1)), int(min(x, w - 1))] \
            + rng.normal(0.0, 0.25)
        dx, dy = half * np.cos(a), half * np.sin(a)
        r, g, b = field[int(min(y, h - 1)), int(min(x, w - 1))]
        j = rng.normal(0.0, jitter, 3)
        color = (int(np.clip(r + j[0], 0, 255)),
                 int(np.clip(g + j[1], 0, 255)),
                 int(np.clip(b + j[2], 0, 255)), alpha)
        x0, y0, x1, y1 = x - dx, y - dy, x + dx, y + dy
        draw.line([(x0, y0), (x1, y1)], fill=color, width=int(dab_w))
        rad = dab_w / 2.0
        draw.ellipse([x0 - rad, y0 - rad, x0 + rad, y0 + rad], fill=color)
        draw.ellipse([x1 - rad, y1 - rad, x1 + rad, y1 + rad], fill=color)

    out = canvas.convert("RGB")
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input photo")
    p.add_argument("--out", required=True, help="Output stroke painting path")
    p.add_argument("--grid", type=int, default=24,
                   help="Coarse grid size (smaller = more abstract)")
    p.add_argument("--colors", type=int, default=12,
                   help="Palette size after quantization")
    p.add_argument("--blur", type=float, default=2.0,
                   help="Gaussian blur radius on the upscaled painting")
    p.add_argument("--smear", type=int, default=0,
                   help="Directional smear length in px (0 = off)")
    p.add_argument("--smear_angle", type=float, default=0.0)
    p.add_argument("--size", type=int, default=512,
                   help="Output resolution (square)")
    args = p.parse_args()

    img = Image.open(args.image).convert("RGB")
    s = min(img.size)
    img = img.crop(((img.width - s) // 2, (img.height - s) // 2,
                    (img.width + s) // 2, (img.height + s) // 2))
    img = img.resize((args.size, args.size), Image.LANCZOS)

    out = stroke_paint(img, args.grid, args.colors, args.blur,
                       args.smear, args.smear_angle)
    out.save(args.out)
    print(f"Saved stroke painting -> {args.out}  "
          f"(grid={args.grid}, colors={args.colors}, blur={args.blur}, "
          f"smear={args.smear}@{args.smear_angle}deg)")


if __name__ == "__main__":
    main()
