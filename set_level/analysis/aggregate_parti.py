"""Aggregate every PartiPrompts results.json of one run tree.

Reads run dirs produced by the generation launchers in scripts/:
  <src>/jive_flux_schnell_parti_<CHALLENGE>/imgs/<prompt_slug>.../results.json

Writes:
  outputs_parti/parti_all_results.json          - nested: challenge -> category -> prompt -> run
  outputs_parti/parti_summary_by_run.csv        - one row per (challenge, prompt, arm)
  outputs_parti/parti_summary_by_challenge.csv  - mean +/- std per (challenge, arm) over prompts
  outputs_parti/parti_summary_overall.csv       - mean +/- std per arm, pooled + by difficulty band
                                                  (Standard / Intermediate / Challenging)

Run:  python analysis/aggregate_parti.py                     # outputs/
      python analysis/aggregate_parti.py --src outputs_jive_jtj_n12
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import OrderedDict
from statistics import mean, stdev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # set_level/
# Jobs write under <run tree>/jive_flux_schnell_parti_<CHALLENGE>/ (via
# --out-root); --src points at a variant tree (e.g. outputs_jive_jtj_n12) so
# each configuration can be rolled up separately. Summary CSVs/JSON land in
# outputs_parti/ so they stay out of the run trees.
SRC = os.path.join(ROOT, "outputs")
OUT = os.path.join(ROOT, "outputs_parti")
SPEC = os.path.join(ROOT, "specs", "parti_prompts.json")
METHOD_PREFIX = "jive_flux_schnell_parti_"

METRICS = [
    "feature_vendi", "feature_vendi_group_mean",
    "vendi_dino_q1", "vendi_dino_q0.5", "vendi_dino_q2", "vendi_dino_frac",
    "pixel_vendi", "pixel_vendi_group_mean",
    "mean_pairwise_l2",
    "clip_score", "clip_iqa", "image_reward", "hpsv2",
    "kid_vs_deterministic_mean", "kid_vs_deterministic_std",
    "applied_budget_l2_mean",
    "cost_tflops_est", "cost_wall_time_s", "cost_tf_forwards",
]

# Official PartiPrompts difficulty bands: the 11 challenge aspects rolled up
# into Standard / Intermediate / Challenging.
BANDS = OrderedDict([
    ("Standard", ["Basic", "Simple Detail"]),
    ("Intermediate", ["Fine-grained Detail", "Style & Format"]),
    ("Challenging", ["Imagination", "Quantity", "Complex", "Linguistic Structures",
                     "Writing & Symbols", "Properties & Positioning", "Perspective"]),
])
BAND_OF = {ch: band for band, chs in BANDS.items() for ch in chs}


def load_spec():
    with open(SPEC) as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def parse_challenge(folder, known=None):
    # folder = <METHOD_PREFIX><CHALLENGE_SLUG>; the slugifier in
    # baselines/oscar/utils.py strips '&' ("Style & Format" -> "Style_Format").
    # Recover the official aspect name by matching against the spec keys: strip
    # non-alphanumerics from both sides and compare.
    raw = folder.replace(METHOD_PREFIX, "").replace("_", " ")
    for ch in (known or []):
        norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
        if norm(raw) == norm(ch):
            return ch
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC,
                    help="Run tree to aggregate. Relative paths are taken from "
                         "set_level/. Default: outputs/")
    ap.add_argument("--out", default=OUT,
                    help="Where the summary CSVs/JSON go. Default: outputs_parti/")
    a = ap.parse_args()
    src = a.src if os.path.isabs(a.src) else os.path.join(ROOT, a.src)
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    spec = load_spec()
    # Map prompt text -> challenge aspect, so every row also carries the band.
    prompt_challenge = {p.strip(): ch for ch, prompts in spec.items() for p in prompts}

    rows, nested = [], {}
    pattern = os.path.join(src, METHOD_PREFIX + "*", "imgs", "*", "results.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as fp:
            d = json.load(fp)
        folder = os.path.relpath(path, src).split(os.sep)[0]
        challenge = prompt_challenge.get(d["prompt"].strip(),
                                         parse_challenge(folder, known=spec))
        band = BAND_OF.get(challenge, "?")
        nested.setdefault(challenge, {}).setdefault(folder, {})[d["prompt"]] = d
        for arm, m in d["arms"].items():
            row = {
                "band": band, "challenge": challenge,
                "prompt": d["prompt"], "seed": d["seed"], "arm": arm,
            }
            for k in METRICS:
                row[k] = m.get(k)
            if m.get("vendi_dino_q1") is not None:
                row["feature_vendi"] = m["vendi_dino_q1"]
            rows.append(row)

    if not rows:
        print(f"no results found under {src} (pattern {pattern}); run the generation launcher first")
        return

    os.makedirs(out, exist_ok=True)
    json_path = os.path.join(out, "parti_all_results.json")
    with open(json_path, "w") as f:
        json.dump(nested, f, indent=2)

    def write_csv(path, table, fieldnames):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table)

    # per-run csv
    cols = ["band", "challenge", "prompt", "seed", "arm"] + METRICS
    write_csv(os.path.join(out, "parti_summary_by_run.csv"), rows, cols)

    def grouped(key_fields, table_rows):
        groups = {}
        for r in table_rows:
            groups.setdefault(tuple(r[k] for k in key_fields), []).append(r)
        out = []
        for key, rs in sorted(groups.items()):
            rec = dict(zip(key_fields, key))
            rec["n_prompts"] = len({r["prompt"] for r in rs})
            for k in METRICS:
                vals = [r[k] for r in rs if r.get(k) is not None]
                rec[k + "_mean"] = mean(vals) if vals else None
                rec[k + "_std"] = stdev(vals) if len(vals) > 1 else 0.0
            out.append(rec)
        return out

    # per-challenge means
    ccols = ["challenge", "arm", "n_prompts"] + [k + s for k in METRICS for s in ("_mean", "_std")]
    write_csv(os.path.join(out, "parti_summary_by_challenge.csv"),
              grouped(["challenge", "arm"], rows), ccols)

    # overall: per band + pooled
    bcols = ["band", "arm", "n_prompts"] + [k + s for k in METRICS for s in ("_mean", "_std")]
    by_band = grouped(["band", "arm"], rows)
    pooled = grouped(["arm"], rows)
    for r in pooled:
        r["band"] = "ALL"
    write_csv(os.path.join(out, "parti_summary_overall.csv"), by_band + pooled, bcols)

    n_runs = len({(r["challenge"], r["prompt"]) for r in rows})
    print(f"runs: {n_runs} prompts | rows: {len(rows)}")
    for p in ("parti_all_results.json", "parti_summary_by_run.csv",
              "parti_summary_by_challenge.csv", "parti_summary_overall.csv"):
        print("wrote", os.path.join(out, p))


if __name__ == "__main__":
    main()
