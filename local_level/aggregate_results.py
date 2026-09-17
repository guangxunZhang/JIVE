"""Aggregate all six arms' results.json into one CSV + markdown table.

Usage (login node is fine, no GPU needed):
    python aggregate_results.py --out_root out

Emits:
  out/summary.csv            every (dataset, method, arm, sweep point) row
  out/summary.md             the same as a readable markdown table
  out/matched_pairs.md       for each method: every +JIVE point next to the
                             baseline point with the CLOSEST L2 (faithfulness-
                             matched comparison — diversity/KID deltas at equal
                             locality are the honest headline numbers)
"""
import argparse
import csv
import glob
import json
import os

METRIC_COLS = ["l2", "kid_x1000", "vendi_clip", "vendi_dino", "vendi_dino_q1", "vendi_pixel",
               "clip_div", "dino_div", "ssim_div", "lpips_div", "dino_faith"]


def load_rows(out_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_root, "*", "*", "results.json"))):
        with open(path) as f:
            data = json.load(f)
        ds, method = data["dataset"], data["method"]
        sweep = data.get("sweep_param", "strength")
        for arm_key, arm_label in [("baseline", "baseline"), ("proposed", "+JIVE")]:
            for r in data.get(arm_key, []):
                rows.append({
                    "dataset": ds, "method": method, "arm": arm_label,
                    "sweep_param": sweep, "sweep_value": r.get("strength"),
                    "inject_norm": r.get("inject_norm"),
                    **{k: r.get(k) for k in METRIC_COLS},
                })
    return rows


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_csv(rows, path):
    cols = ["dataset", "method", "arm", "sweep_param", "sweep_value",
            "inject_norm"] + METRIC_COLS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows, path):
    cols = ["dataset", "method", "arm", "sweep_value", "inject_norm"] + METRIC_COLS
    with open(path, "w") as f:
        f.write("# Local-level comparison — all sweep points\n\n")
        f.write("KID uses torchvision Inception features: compare only within "
                "this experiment. `l2` lower = more faithful; `vendi_*`, "
                "`*_div` higher = more diverse; `kid_x1000` lower = more "
                "realistic.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[c]) for c in cols) + " |\n")


def write_matched_pairs(rows, path):
    """Pair each +JIVE point with the L2-closest baseline point of the same
    (dataset, method), then report metric deltas at ~equal faithfulness."""
    with open(path, "w") as f:
        f.write("# Faithfulness-matched comparison (per method)\n\n")
        f.write("Each +JIVE sweep point vs. the baseline point with the "
                "closest L2. Positive d_vendi / negative d_kid favour JIVE.\n\n")
        keys = sorted({(r["dataset"], r["method"]) for r in rows})
        for ds, method in keys:
            base = [r for r in rows if r["dataset"] == ds
                    and r["method"] == method and r["arm"] == "baseline"
                    and r["l2"] is not None]
            ours = [r for r in rows if r["dataset"] == ds
                    and r["method"] == method and r["arm"] == "+JIVE"
                    and r["l2"] is not None]
            if not base or not ours:
                continue
            f.write(f"## {ds} / {method}\n\n")
            f.write("| JIVE sweep | norm | l2 (JIVE) | l2 (base) | "
                    "d_vendi_clip | d_dino_div | d_lpips_div | d_kid_x1000 |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for o in ours:
                b = min(base, key=lambda r: abs(r["l2"] - o["l2"]))
                def delta(k):
                    if o[k] is None or b[k] is None:
                        return "-"
                    return f"{o[k] - b[k]:+.4g}"
                f.write(
                    f"| {fmt(o['sweep_value'])} | {fmt(o['inject_norm'])} "
                    f"| {fmt(o['l2'])} | {fmt(b['l2'])} "
                    f"| {delta('vendi_clip')} | {delta('dino_div')} "
                    f"| {delta('lpips_div')} | {delta('kid_x1000')} |\n")
            f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="out")
    args = p.parse_args()

    rows = load_rows(args.out_root)
    if not rows:
        print(f"No results.json found under {args.out_root}/*/*/")
        return
    write_csv(rows, os.path.join(args.out_root, "summary.csv"))
    write_md(rows, os.path.join(args.out_root, "summary.md"))
    write_matched_pairs(rows, os.path.join(args.out_root, "matched_pairs.md"))
    print(f"Wrote {len(rows)} rows to {args.out_root}/summary.{{csv,md}} and "
          f"matched_pairs.md")


if __name__ == "__main__":
    main()
