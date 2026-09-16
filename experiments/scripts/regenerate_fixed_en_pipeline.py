#!/usr/bin/env python3
"""Compute fixed-E/N metrics and regenerate affected paper figures with suffix.

Phase A: computes metrics from cmm_result_fixed_en.csv (FMM unchanged,
         fmm_result.csv) for sigma sweep, sample rate, degraded, mismatch.
Phase B: regenerates the four affected figures with `_fixed_en` suffix by
         exec-ing string-patched copies of the regenerate scripts.

Outputs:
  metrics: experiments/output/3_full_matching/{sigma_sweep,sample_rate}_full_fixed_en.json
           experiments/output/5_degraded/degraded_full_fixed_en.json
           experiments/output/4_sigma_mismatch/mismatch_summary_fixed_en.csv
  figures: docs/.../figs/{sigma_sweep,sample_rate_sensitivity,
           degraded_comparison,mismatch_analysis}_fixed_en.png
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments/scripts"))

import exp3_full_matching as exp3
import exp5_degraded_conditions as exp5
import exp4_sigma_mismatch as exp4

SIGMA_DIRS = ["sigma_01", "sigma_05", "sigma_10", "sigma_15",
              "sigma_20", "sigma_25", "sigma_30"]
SIM = BASE / "data/simulation"
OUT3 = BASE / "experiments/output/3_full_matching"
OUT4 = BASE / "experiments/output/4_sigma_mismatch"
OUT5 = BASE / "experiments/output/5_degraded"
FIGS = BASE / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
for d in (OUT3, OUT4, OUT5, FIGS):
    d.mkdir(parents=True, exist_ok=True)

FIXED_SUFFIX = "cmm_result_fixed_en.csv"
FMM_NAME = "fmm_result.csv"


def compute_pair(data_dir: Path, label: str, exp_module):
    """Return (cmm_metrics, fmm_metrics) for one dataset using the fixed CMM file."""
    gt_pts_seq, gt_edg_seq, gt_pts_ts, gt_seq_ts = exp_module.load_ground_truth(data_dir)
    cmm = exp_module.compute_metrics(
        data_dir, data_dir / FIXED_SUFFIX,
        gt_pts_seq, gt_edg_seq, gt_pts_ts, gt_seq_ts, label)
    fmm = exp_module.compute_metrics(
        data_dir, data_dir / FMM_NAME,
        gt_pts_seq, gt_edg_seq, gt_pts_ts, gt_seq_ts, label)
    return cmm, fmm


def write_full_json(path: Path, cmm_list, fmm_list):
    def ser(ml):
        out = []
        for m in ml:
            d = dict(m)
            for k in ("fpr", "tpr"):
                if k in d and d[k] is not None:
                    d[k] = d[k].tolist() if hasattr(d[k], "tolist") else list(d[k])
            out.append(d)
        return out
    with open(path, "w") as f:
        json.dump({"cmm": ser(cmm_list), "fmm": ser(fmm_list)}, f, indent=2)
    print(f"  wrote {path}")


# ═══════════════════════════════════════════════════════════════════
# Phase A1: sigma sweep
# ═══════════════════════════════════════════════════════════════════
print("=== Sigma sweep metrics (fixed) ===")
cmm_all, fmm_all = [], []
for sd in SIGMA_DIRS:
    d = SIM / sd / "no_occlusion" / "no_fault"
    cmm, fmm = compute_pair(d, sd, exp3)
    cmm_all.append(cmm); fmm_all.append(fmm)
    print(f"  {sd}: CMM acc={cmm.get('seg_accuracy',0):.4f} ECE={cmm.get('ece_tw',1):.4f} "
          f"AUC={cmm.get('roc_auc',0.5):.3f} | FMM acc={fmm.get('seg_accuracy',0):.4f} ECE={fmm.get('ece_tw',1):.4f}")
write_full_json(OUT3 / "sigma_sweep_full_fixed_en.json", cmm_all, fmm_all)

# ═══════════════════════════════════════════════════════════════════
# Phase A2: sample rate
# ═══════════════════════════════════════════════════════════════════
print("\n=== Sample-rate metrics (fixed) ===")
sr_metrics = {}
SAMPLE_INTERVALS = [1, 2, 5, 10]
SR_SIGMA_LABELS = ["sigma_05", "sigma_15", "sigma_25"]
for sigma_key in SR_SIGMA_LABELS:
    sr_metrics[sigma_key] = []
    for interval in SAMPLE_INTERVALS:
        if interval == 1:
            d = SIM / sigma_key / "no_occlusion" / "no_fault"
        else:
            d = SIM / sigma_key / "no_occlusion" / "no_fault" / f"subsample_{interval}s"
        label = f"{sigma_key}_sr{interval}"
        cmm, fmm = compute_pair(d, label, exp3)
        cmm["algorithm"] = "CMM"; cmm["sample_interval"] = interval
        fmm["algorithm"] = "FMM"; fmm["sample_interval"] = interval
        sr_metrics[sigma_key].append(cmm)
        sr_metrics[sigma_key].append(fmm)
        print(f"  {label}: CMM acc={cmm.get('seg_accuracy',0):.4f} | FMM acc={fmm.get('seg_accuracy',0):.4f}")
# Flatten to the JSON shape exp3's regenerate_sample_rate expects:
# {"cmm": [...], "fmm": [...]} per the original write_summary call (line 826)
sr_cmm = [m for key in SR_SIGMA_LABELS for m in sr_metrics[key] if m.get("algorithm") == "CMM"]
sr_fmm = [m for key in SR_SIGMA_LABELS for m in sr_metrics[key] if m.get("algorithm") == "FMM"]
write_full_json(OUT3 / "sample_rate_full_fixed_en.json", sr_cmm, sr_fmm)

# ═══════════════════════════════════════════════════════════════════
# Phase A3: degraded
# ═══════════════════════════════════════════════════════════════════
print("\n=== Degraded metrics (fixed) ===")
cond_order = ["clean", "fault", "occlusion", "both"]
deg_cmm, deg_fmm = [], []
for c in cond_order:
    sub = exp5.CONDITIONS[c]
    d = SIM / "sigma_30" / sub[0] / sub[1]
    cmm, fmm = compute_pair(d, c, exp5)
    deg_cmm.append(cmm); deg_fmm.append(fmm)
    print(f"  {c}: CMM acc={cmm.get('seg_accuracy',0):.4f} ECE={cmm.get('ece_tw',1):.4f} "
          f"| FMM acc={fmm.get('seg_accuracy',0):.4f}")
write_full_json(OUT5 / "degraded_full_fixed_en.json", deg_cmm, deg_fmm)

# ═══════════════════════════════════════════════════════════════════
# Phase A4: sigma mismatch
# ═══════════════════════════════════════════════════════════════════
print("\n=== Mismatch metrics (fixed) ===")
mis_cmm, mis_fmm = [], []
for pr in ["pr10", "pr15", "pr20", "pr25", "pr30"]:
    d = SIM / "sigma_mismatch" / f"{pr}_wls20"
    cmm, fmm = compute_pair(d, pr, exp4)
    mis_cmm.append(cmm); mis_fmm.append(fmm)
    print(f"  {pr}: CMM acc={cmm.get('seg_accuracy',0):.4f} ECE={cmm.get('ece_tw',1):.4f} "
          f"| FMM acc={fmm.get('seg_accuracy',0):.4f}")

# exp4's mismatch_summary.csv format (rows with algorithm column)
rows = []
for m in mis_cmm:
    rows.append({"algorithm": "CMM", **{k: v for k, v in m.items()
                 if k not in ("fpr", "tpr", "ece_tw_bins", "corr_trust_mean", "mis_trust_mean")}})
for m in mis_fmm:
    rows.append({"algorithm": "FMM", **{k: v for k, v in m.items()
                 if k not in ("fpr", "tpr", "ece_tw_bins", "corr_trust_mean", "mis_trust_mean")}})
import csv
out = OUT4 / "mismatch_summary_fixed_en.csv"
fieldnames = ["algorithm", "label", "n", "point_error_mean", "point_error_median",
              "point_error_rmse", "point_error_p95", "seg_accuracy",
              "ece_tw", "mce_tw", "roc_auc", "trust_separation"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"  wrote {out}")

# ═══════════════════════════════════════════════════════════════════
# Phase B: regenerate figures with _fixed_en suffix via patched exec
# ═══════════════════════════════════════════════════════════════════
print("\n=== Regenerating figures (fixed) ===")

def run_patched(script_name: str, replacements):
    src = (BASE / "experiments/scripts" / script_name).read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in src:
            print(f"  WARNING: pattern not found in {script_name}: {old[:60]}")
        src = src.replace(old, new)
    exec(compile(src, f"<patched:{script_name}>", "exec"), {"__name__": "__main__", "__file__": str(BASE / "experiments/scripts" / script_name)})

# Fig.7 sigma sweep
run_patched("regenerate_sigma_sweep.py", [
    ('FULL_JSON = OUTPUT_DIR / "sigma_sweep_full.json"',
     'FULL_JSON = OUTPUT_DIR / "sigma_sweep_full_fixed_en.json"'),
    ('SUMMARY_CSV = OUTPUT_DIR / "sigma_sweep_table.csv"',
     'SUMMARY_CSV = OUTPUT_DIR / "sigma_sweep_table_fixed_en.csv"'),
    ('mr_name = "cmm_result.csv" if algo == "cmm" else "fmm_result.csv"',
     'mr_name = "cmm_result_fixed_en.csv" if algo == "cmm" else "fmm_result.csv"'),
    ('out = FIGS_DIR / "sigma_sweep.png"', 'out = FIGS_DIR / "sigma_sweep_fixed_en.png"'),
    ('out_svg = FIGS_DIR / "sigma_sweep.svg"', 'out_svg = FIGS_DIR / "sigma_sweep_fixed_en.svg"'),
])

# Fig.8 sample rate
run_patched("regenerate_sample_rate.py", [
    ('JSON_PATH = PROJECT / "experiments/output/3_full_matching/sample_rate_full.json"',
     'JSON_PATH = PROJECT / "experiments/output/3_full_matching/sample_rate_full_fixed_en.json"'),
    ('out = FIGS_DIR / "sample_rate_sensitivity.png"',
     'out = FIGS_DIR / "sample_rate_sensitivity_fixed_en.png"'),
])

# Fig.9 degraded
run_patched("regenerate_degraded.py", [
    ('METRICS_JSON = PROJECT / "experiments/output/5_degraded/degraded_full.json"',
     'METRICS_JSON = PROJECT / "experiments/output/5_degraded/degraded_full_fixed_en.json"'),
    ('out = FIGS_DIR / "degraded_comparison.png"',
     'out = FIGS_DIR / "degraded_comparison_fixed_en.png"'),
])

# Fig.19 mismatch
run_patched("regenerate_mismatch.py", [
    ('CSV_PATH = PROJECT / "experiments/output/4_sigma_mismatch/mismatch_summary.csv"',
     'CSV_PATH = PROJECT / "experiments/output/4_sigma_mismatch/mismatch_summary_fixed_en.csv"'),
    ('out = FIGS_DIR / "mismatch_analysis.png"',
     'out = FIGS_DIR / "mismatch_analysis_fixed_en.png"'),
])

print("\nDone.")
