"""Download & export LSUN Bedroom / Church images.

Produces, under --out_root/<dataset>/:
  sources/src_000.png ...    guides the six arms locally sample around
  reference/ref_00000.png ...real images for the KID reference set

Primary path: Hugging Face mirrors of LSUN (the official dl.yf.io server is
frequently down and the full bedroom train LMDB is 43 GB):
    bedroom : pcuenq/lsun-bedrooms
    church  : tglcourse/lsun_church_train
Streaming is used, so only the exported images touch disk. Sources are taken
AFTER the reference block (disjoint by construction).

Fallback path (--from_lmdb ZIP_OR_DIR): export from an official LSUN LMDB
(e.g. bedroom_val_lmdb.zip from http://dl.yf.io/lsun/scenes/) — needs the
`lmdb` package.

All images are center-cropped and resized to --resolution (default 256, the
LSUN DDPM checkpoint resolution).
"""
import argparse
import io
import os
import zipfile

from PIL import Image

HF_REPOS = {
    "bedroom": [("pcuenq/lsun-bedrooms", "train")],
    "church": [("tglcourse/lsun_church_train", "train")],
}


def center_crop_resize(img, res):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((res, res), Image.LANCZOS)


def _first_image(example):
    for v in example.values():
        if isinstance(v, Image.Image):
            return v
    raise ValueError(f"No PIL image found in example keys: {list(example)}")


def export_from_hf(dataset, out_dir, n_reference, n_sources, res):
    from datasets import load_dataset

    last_err = None
    for repo, split in HF_REPOS[dataset]:
        try:
            print(f"Streaming {repo} [{split}] ...")
            ds = load_dataset(repo, split=split, streaming=True)
            it = iter(ds)
            ref_dir = os.path.join(out_dir, "reference")
            src_dir = os.path.join(out_dir, "sources")
            os.makedirs(ref_dir, exist_ok=True)
            os.makedirs(src_dir, exist_ok=True)
            for i in range(n_reference):
                img = center_crop_resize(_first_image(next(it)), res)
                img.save(os.path.join(ref_dir, f"ref_{i:05d}.png"))
                if (i + 1) % 500 == 0:
                    print(f"  reference {i + 1}/{n_reference}")
            for i in range(n_sources):
                img = center_crop_resize(_first_image(next(it)), res)
                img.save(os.path.join(src_dir, f"src_{i:03d}.png"))
            print(f"Exported {n_reference} reference + {n_sources} source "
                  f"images to {out_dir}")
            return
        except Exception as e:  # noqa: BLE001 - try next mirror
            print(f"  FAILED on {repo}: {e}")
            last_err = e
    raise RuntimeError(
        f"All HF mirrors failed for {dataset}; retry or use --from_lmdb "
        f"with an official LSUN LMDB zip."
    ) from last_err


def export_from_lmdb(path, out_dir, n_reference, n_sources, res):
    import lmdb

    if path.endswith(".zip"):
        extract_dir = path[:-4]
        if not os.path.isdir(extract_dir):
            print(f"Unzipping {path} ...")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(os.path.dirname(path) or ".")
        # official zips contain a single <name>_lmdb directory
        cands = [os.path.join(extract_dir)] + [
            os.path.join(os.path.dirname(path), d)
            for d in os.listdir(os.path.dirname(path) or ".") if d.endswith("_lmdb")
        ]
        path = next(d for d in cands if os.path.isdir(d))

    ref_dir = os.path.join(out_dir, "reference")
    src_dir = os.path.join(out_dir, "sources")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    env = lmdb.open(path, map_size=1099511627776, max_readers=100,
                    readonly=True)
    n_total = n_reference + n_sources
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        i = 0
        for _key, val in cursor:
            img = center_crop_resize(Image.open(io.BytesIO(val)), res)
            if i < n_reference:
                img.save(os.path.join(ref_dir, f"ref_{i:05d}.png"))
            else:
                img.save(os.path.join(src_dir, f"src_{i - n_reference:03d}.png"))
            i += 1
            if i % 500 == 0:
                print(f"  exported {i}/{n_total}")
            if i >= n_total:
                break
    if i < n_total:
        print(f"WARNING: LMDB only had {i} images (< {n_total} requested); "
              f"KID reference set will be smaller than planned.")
    print(f"Exported to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["bedroom", "church"], required=True)
    p.add_argument("--out_root", default="data")
    p.add_argument("--n_sources", type=int, default=32)
    p.add_argument("--n_reference", type=int, default=2000)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--from_lmdb", default=None,
                   help="Path to an official LSUN LMDB dir or .zip (fallback "
                        "when the HF mirrors are unreachable)")
    args = p.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    src_dir = os.path.join(out_dir, "sources")
    if (os.path.isdir(src_dir)
            and len(os.listdir(src_dir)) >= args.n_sources):
        print(f"SKIP: {src_dir} already has >= {args.n_sources} images.")
        return

    if args.from_lmdb:
        export_from_lmdb(args.from_lmdb, out_dir, args.n_reference,
                         args.n_sources, args.resolution)
    else:
        export_from_hf(args.dataset, out_dir, args.n_reference,
                       args.n_sources, args.resolution)


if __name__ == "__main__":
    main()
