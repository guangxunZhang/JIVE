"""Emit the eight-row JIVE PartiPrompts band table (mean ± standard error,
HPSv2 × 100), the main set-level table of the paper.

Each row comes from its own run tree, produced by the matching launcher in
scripts/:
  Unmodified sampler   }
  OSCAR                } run_parti_base_oscar_jive.sh  -> outputs/
  JIVE(J)              }   (inject-norm 12, J iteration)
  JIVE(J^T J)          run_parti_jive_jtj.sh           -> outputs_jive_jtj_n12/
  CADS tau2=1.8 / 1.2  run_parti_cads.sh               -> outputs_cads/
  JIVE(J)+CADS         run_parti_jive_cads.sh          -> outputs_jive_cads_n4/
  JIVE(J^T J)+CADS     run_parti_jive_jtj_cads.sh      -> outputs_jive_cads_jtj_n4/

The first three are read from the by-run CSV that aggregate_parti.py writes;
the rest are read straight out of their run trees.

Run:  python analysis/jive_bands_table.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import OrderedDict, defaultdict
from statistics import mean, stdev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # set_level/
SPEC = os.path.join(ROOT, "specs", "parti_prompts.json")
BY_RUN = os.path.join(ROOT, "outputs_parti", "parti_summary_by_run.csv")
JIVE_CADS_SRC = os.path.join(ROOT, "outputs_jive_cads_n4", "outputs")
JTJ_SRC = os.path.join(ROOT, "outputs_jive_jtj_n12", "outputs")
JTJ_CADS_SRC = os.path.join(ROOT, "outputs_jive_cads_jtj_n4", "outputs")
CADS_DIR = os.path.join(ROOT, "outputs_cads", "schnell_parti_seed42")
OUT_TEX = os.path.join(ROOT, "outputs_parti", "parti_bands_jive.tex")

BANDS = OrderedDict([
    ("Standard", ["Basic", "Simple Detail"]),
    ("Intermediate", ["Fine-grained Detail", "Style & Format"]),
    ("Challenging", ["Imagination", "Quantity", "Complex",
                     "Linguistic Structures", "Writing & Symbols",
                     "Properties & Positioning", "Perspective"]),
])
BAND_OF = {ch: band for band, chs in BANDS.items() for ch in chs}
BAND_N = {"Standard": 504, "Intermediate": 518, "Challenging": 610, "All": 1632}

FIELDS = ["feature_vendi", "pixel_vendi", "mean_pairwise_l2",
          "clip_score", "clip_iqa", "hpsv2"]

ARMS = [
    ("det", r"Unmodified sampler"),
    ("oscar", r"OSCAR"),
    ("jive_j", r"JIVE($J$)"),
    ("jive_jtj", r"JIVE($J^\top J$)"),
    ("cads18", r"CADS ($\tau_2{=}1.8$)"),
    ("cads12", r"CADS ($\tau_2{=}1.2$)"),
    ("jive_j_cads", r"JIVE($J$)+CADS"),
    ("jive_jtj_cads", r"JIVE($J^\top J$)+CADS"),
]
CSV_ARM = {"deterministic": "det", "oscar": "oscar", "jive": "jive_j"}


def load_spec():
    with open(SPEC) as f:
        spec = json.load(f, object_pairs_hook=OrderedDict)
    prompt_challenge = {p.strip(): ch for ch, ps in spec.items() for p in ps}
    return prompt_challenge


def _arm_metrics(m):
    return {
        "feature_vendi": m.get("vendi_dino_q1", m.get("feature_vendi")),
        "pixel_vendi": m.get("pixel_vendi"),
        "mean_pairwise_l2": m.get("mean_pairwise_l2"),
        "clip_score": m.get("clip_score"),
        "clip_iqa": m.get("clip_iqa"),
        "hpsv2": m.get("hpsv2"),
    }


def _mean_std(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def summarize(rows_by_band):
    """rows_by_band: {band: [metric dict, ...]} -> {band: {k: (mean, std)}}."""
    out = {}
    all_rows = []
    for band in BANDS:
        all_rows.extend(rows_by_band.get(band, []))
        out[band] = {k: _mean_std(r[k] for r in rows_by_band.get(band, []))
                     for k in FIELDS}
    out["All"] = {k: _mean_std(r[k] for r in all_rows) for k in FIELDS}
    return out


def from_by_run_csv():
    buckets = {akey: defaultdict(list) for akey in CSV_ARM.values()}
    with open(BY_RUN) as f:
        for r in csv.DictReader(f):
            akey = CSV_ARM.get(r["arm"])
            if akey is None:
                continue
            rec = {
                "feature_vendi": float(r["vendi_dino_q1"] or r["feature_vendi"]),
                "pixel_vendi": float(r["pixel_vendi"]),
                "mean_pairwise_l2": float(r["mean_pairwise_l2"]),
                "clip_score": float(r["clip_score"]),
                "clip_iqa": float(r["clip_iqa"]),
                "hpsv2": float(r["hpsv2"]),
            }
            buckets[akey][r["band"]].append(rec)
    return {akey: summarize(bands) for akey, bands in buckets.items()}


def from_tree(src, arm_key, prompt_challenge):
    buckets = defaultdict(list)
    pat = os.path.join(src, "*", "imgs", "*", "results.json")
    n = 0
    for path in glob.glob(pat):
        with open(path) as f:
            d = json.load(f)
        ch = prompt_challenge.get(str(d.get("prompt", "")).strip())
        if ch is None or arm_key not in d.get("arms", {}):
            continue
        rec = _arm_metrics(d["arms"][arm_key])
        if rec["feature_vendi"] is None:
            continue
        buckets[BAND_OF[ch]].append(rec)
        n += 1
    return summarize(buckets), n


def from_cads(arm_csv, condition):
    vendi = {}
    with open(os.path.join(CADS_DIR, "analysis", "per_group.csv")) as f:
        for r in csv.DictReader(f):
            if r["condition"] == condition:
                vendi[r["prompt_uid"]] = float(r["vendi_dino_q1"])
    buckets = defaultdict(list)
    with open(os.path.join(CADS_DIR, "analysis", arm_csv)) as f:
        for r in csv.DictReader(f):
            buckets[r["band"]].append({
                "feature_vendi": vendi[r["uid"]],
                "pixel_vendi": float(r["pixel_vendi"]),
                "mean_pairwise_l2": float(r["mean_pairwise_l2"]),
                "clip_score": float(r["clip_score"]),
                "clip_iqa": float(r["clip_iqa"]),
                "hpsv2": float(r["hpsv2"]),
            })
    return summarize(buckets)


# Mean precision matches the published table; HPSv2 is 100x with two decimals
# ("last two digits" of the original 0.xxx score).
FMT = {
    "feature_vendi": (3, 1.0),
    "pixel_vendi": (3, 1.0),
    "mean_pairwise_l2": (2, 1.0),
    "clip_score": (2, 1.0),
    "clip_iqa": (3, 1.0),
    "hpsv2": (2, 100.0),
}


def fmt_pm(mu, sd, k, n):
    """Report mean ± standard error of the mean (sd / sqrt(n_prompts))."""
    nd, scale = FMT[k]
    se = sd / (n ** 0.5)
    return f"{mu * scale:.{nd}f}$\\pm${se * scale:.{nd}f}"


def cell(mu, sd, k, n, best, bold_keys):
    s = fmt_pm(mu, sd, k, n)
    if k in bold_keys and abs(mu - best[k]) < 1e-9:
        return f"\\textbf{{{s}}}"
    return s


def write_tex(table):
    bold_keys = ("feature_vendi", "pixel_vendi", "mean_pairwise_l2")
    best = {}
    for band in list(BANDS) + ["All"]:
        best[band] = {k: max(table[a][band][k][0] for a, _ in ARMS)
                      for k in bold_keys}

    lines = [
        "% Auto-generated by jive_bands_table.py",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Diversity and plug-in composition on PartiPrompts.} "
        r"Four-step FLUX.1-schnell, 16 images per prompt. Values are "
        r"mean$\pm$standard error over prompts. JIVE uses perturbation norm 12 alone "
        r"and norm 4 with CADS ($\tau_2=1.2$). F-Vendi and P-Vendi measure "
        r"feature- and pixel-space diversity. HPSv2 is reported as "
        r"$100\times$ the raw score. All metrics are higher-is-better; "
        r"bold marks the highest diversity score within each band.}",
        r"\label{tab:parti-bands}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\footnotesize",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrr@{}}",
        r"\toprule",
        r"Band & Method & F-Vendi$\uparrow$ & P-Vendi$\uparrow$ & "
        r"\makecell[r]{Pairwise\\$L_2\uparrow$} & CLIP$\uparrow$ & "
        r"CLIP-IQA$\uparrow$ & HPSv2$\times$100$\uparrow$ \\",
        r"\midrule",
    ]
    for band in list(BANDS) + ["All"]:
        n = BAND_N[band]
        lines.append(
            rf"\multirow{{8}}{{*}}{{\makecell[l]{{\textbf{{{band}}}\\($n={n}$)}}}}"
        )
        for akey, label in ARMS:
            cells = " & ".join(
                cell(table[akey][band][k][0], table[akey][band][k][1],
                     k, n, best[band], bold_keys)
                for k in FIELDS)
            lines.append(f" & {label} & {cells} \\\\")
        if band != "All":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    text = "\n".join(lines)
    with open(OUT_TEX, "w") as f:
        f.write(text)
    return text


def main():
    prompt_challenge = load_spec()
    print("loading det / OSCAR / JIVE(J) from by-run CSV ...")
    table = from_by_run_csv()

    print("loading JIVE(J^T J) n12 ...")
    table["jive_jtj"], n_jtj = from_tree(JTJ_SRC, "jive", prompt_challenge)
    print(f"  {n_jtj} prompts")

    print("loading JIVE(J)+CADS n4 t212 ...")
    table["jive_j_cads"], n_jc = from_tree(
        JIVE_CADS_SRC, "jive_cads", prompt_challenge)
    print(f"  {n_jc} prompts")

    print("loading JIVE(J^T J)+CADS n4 ...")
    table["jive_jtj_cads"], n_jtc = from_tree(
        JTJ_CADS_SRC, "jive_cads", prompt_challenge)
    print(f"  {n_jtc} prompts")

    print("loading CADS ...")
    table["cads18"] = from_cads(
        "cads_table_metrics_cads_t2-1.8_n0.15.csv", "cads_t2-1.8_n0.15")
    table["cads12"] = from_cads(
        "cads_table_metrics_cads_t2-1.2_n0.15.csv", "cads_t2-1.2_n0.15")

    print("\n=== All-row means (sanity vs published table) ===")
    for a, lab in ARMS:
        mu = {k: table[a]["All"][k][0] for k in FIELDS}
        print(f"  {lab:28s} FV={mu['feature_vendi']:.3f}  "
              f"pix={mu['pixel_vendi']:.3f}  L2={mu['mean_pairwise_l2']:.2f}  "
              f"CLIP={mu['clip_score']:.2f}  IQA={mu['clip_iqa']:.3f}  "
              f"HPS={100 * mu['hpsv2']:.2f}")

    tex = write_tex(table)
    print("\nwrote", OUT_TEX)
    print("\n" + tex)


if __name__ == "__main__":
    main()
