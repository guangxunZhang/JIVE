"""Download & export LSUN scene images.

Produces, under --out_root/<dataset>/:
  sources/src_000.png ...    guides the six arms locally sample around
  reference/ref_00000.png ...real images for the KID reference set

The official LSUN LMDBs, mirrored on Hugging Face as RichardErkhov/LSUN
(the original dl.yf.io server is frequently down). Used for:
    classroom, kitchen, conference_room, dining_room, restaurant
The train zip is downloaded, unpacked, read until enough images have been
exported, then deleted again unless --keep_lmdb is passed. Needs the `lmdb`
package.

--from_lmdb ZIP_OR_DIR overrides that and exports from a local LMDB.

All images are center-cropped and resized to --resolution (default 256, the
LSUN DDPM checkpoint resolution).
"""
import argparse
import io
import os
import shutil
import zipfile

from PIL import Image

LMDB_MIRROR_REPO = "RichardErkhov/LSUN"
LMDB_MIRROR_FILES = {
    "classroom": "scenes/classroom_train_lmdb.zip",
    "kitchen": "scenes/kitchen_train_lmdb.zip",
    "conference_room": "scenes/conference_room_train_lmdb.zip",
    "dining_room": "scenes/dining_room_train_lmdb.zip",
    "restaurant": "scenes/restaurant_train_lmdb.zip",
}

DATASETS = sorted(LMDB_MIRROR_FILES)


def center_crop_resize(img, res):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((res, res), Image.LANCZOS)


def unzip_lmdb(path):
    """Unpack an official LSUN <name>_lmdb.zip; returns the LMDB directory."""
    parent = os.path.dirname(path) or "."
    extract_dir = path[:-4]
    if not os.path.isdir(extract_dir):
        print(f"Unzipping {path} ...")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(parent)
    cands = [extract_dir] + [
        os.path.join(parent, d)
        for d in os.listdir(parent) if d.endswith("_lmdb")
    ]
    return next(d for d in cands if os.path.isdir(d))


def fetch_lmdb_from_hf(dataset, cache_dir):
    """Download + unpack the mirrored LSUN train LMDB; returns (dir, zip)."""
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {LMDB_MIRROR_FILES[dataset]} from "
          f"{LMDB_MIRROR_REPO} into {cache_dir} (tens of GB) ...")
    zip_path = hf_hub_download(
        LMDB_MIRROR_REPO, LMDB_MIRROR_FILES[dataset],
        repo_type="dataset", local_dir=cache_dir,
    )
    return unzip_lmdb(zip_path), zip_path


def export_from_lmdb(path, out_dir, n_reference, n_sources, res):
    import lmdb

    if path.endswith(".zip"):
        path = unzip_lmdb(path)

    ref_dir = os.path.join(out_dir, "reference")
    src_dir = os.path.join(out_dir, "sources")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    env = lmdb.open(path, map_size=1099511627776, max_readers=100,
                    readonly=True, lock=False)
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
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--out_root", default="data")
    p.add_argument("--n_sources", type=int, default=32)
    p.add_argument("--n_reference", type=int, default=2000)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--from_lmdb", default=None,
                   help="Path to an official LSUN LMDB dir or .zip (fallback "
                        "when the HF mirrors are unreachable)")
    p.add_argument("--lmdb_cache", default="lsun_lmdb",
                   help="Scratch space for the mirrored LSUN train LMDBs; "
                        "needs tens of GB while a category is being exported")
    p.add_argument("--keep_lmdb", action="store_true",
                   help="Keep the downloaded zip and unpacked LMDB instead of "
                        "deleting them once the images are exported")
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
        lmdb_dir, zip_path = fetch_lmdb_from_hf(args.dataset, args.lmdb_cache)
        export_from_lmdb(lmdb_dir, out_dir, args.n_reference,
                         args.n_sources, args.resolution)
        if not args.keep_lmdb:
            shutil.rmtree(lmdb_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(args.lmdb_cache, ".cache"),
                          ignore_errors=True)
            if os.path.isfile(zip_path):
                os.remove(zip_path)
            print(f"Removed {zip_path} and {lmdb_dir}")


if __name__ == "__main__":
    main()
