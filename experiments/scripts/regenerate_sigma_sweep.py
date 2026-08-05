#!/usr/bin/env python3
"""Regenerate sigma_sweep.png (Figure 6) from available data sources.

Data sources:
  - summary_table.csv: summary metrics for all sigma levels (panels a-d)
  - sample_rate_full.json: detailed bins/ROC data for panels (e-f)
    Uses the closest available sigma level with full data.

Changes from previous version:
  - FMM → HMM in all legend labels
  - Uses CSV fallback when JSON is empty
  - Handles n=0 entries gracefully
"""
import csv, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "experiments/output/3_full_matching"
SUMMARY_CSV = OUTPUT_DIR / "summary_table.csv"
SAMPLE_RATE_JSON = OUTPUT_DIR / "sample_rate_full.json"
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_CMM = "#2166ac"
COLOR_HMM = "#b2182b"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# ── Load summary data from CSV ──
if not SUMMARY_CSV.exists():
    print(f"ERROR: {SUMMARY_CSV} not found.")
    sys.exit(1)

cmm_data = {}  # label -> {metric: value}
hmm_data = {}

with open(SUMMARY_CSV, newline="") as f:
    for row in csv.DictReader(f):
        algo = row["algorithm"].strip()
        label = row["label"].strip()
        target = cmm_data if algo == "CMM" else hmm_data
        target[label] = {
            "n": int(row["n"]) if row["n"] else 0,
            "point_error_mean": float(row["point_error_mean"]) if row["point_error_mean"] else None,
            "point_error_median": float(row["point_error_median"]) if row["point_error_median"] else None,
            "point_error_rmse": float(row["point_error_rmse"]) if row["point_error_rmse"] else None,
            "point_error_p95": float(row["point_error_p95"]) if row["point_error_p95"] else None,
            "seg_accuracy": float(row["seg_accuracy"]) if row["seg_accuracy"] else None,
            "ece_tw": float(row["ece_tw"]) if row["ece_tw"] else None,
            "ece_ep": float(row["ece_ep"]) if row["ece_ep"] else None,
            "roc_auc": float(row["roc_auc"]) if row["roc_auc"] else None,
        }

print(f"Loaded CSV: {len(cmm_data)} CMM, {len(hmm_data)} HMM entries")

# Determine sigma levels (sorted by numeric value, filter out sr variants)
sigma_labels_cmm = sorted(
    [k for k in cmm_data.keys() if not "_sr" in k and k.startswith("sigma_")],
    key=lambda s: int(s.replace("sigma_", ""))
)
sigma_labels_hmm = sorted(
    [k for k in hmm_data.keys() if not "_sr" in k and k.startswith("sigma_")],
    key=lambda s: int(s.replace("sigma_", ""))
)
sigma_labels = sorted(set(sigma_labels_cmm) | set(sigma_labels_hmm),
                      key=lambda s: int(s.replace("sigma_", "")))
sigma_vals = [int(s.replace("sigma_", "")) for s in sigma_labels]
print(f"Sigma levels: {sigma_labels}")

# ── Load detailed data from sample_rate_full.json for panels (e)-(f) ──
detail_cmm = {}  # label -> full metrics dict
detail_hmm = {}
if SAMPLE_RATE_JSON.exists():
    with open(SAMPLE_RATE_JSON) as f:
        sr_data = json.load(f)
    for m in sr_data.get("cmm", []):
        detail_cmm[m["label"]] = m
    for m in sr_data.get("fmm", []):
        detail_hmm[m["label"]] = m
    print(f"Loaded detailed JSON: {len(detail_cmm)} CMM, {len(detail_hmm)} HMM entries")
else:
    print(f"WARNING: {SAMPLE_RATE_JSON} not found — panels (e)-(f) will use fallback")

# Helper: get a metric from CSV data
def csv_get(data_dict, sigma_label, key, default=np.nan):
    entry = data_dict.get(sigma_label, {})
    val = entry.get(key)
    return val if val is not None else default

# ── Build figure ──
fig, axes = plt.subplots(3, 2, figsize=(7.5, 10.5))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

# (a) Point error
ax1.plot(sigma_vals, [csv_get(cmm_data, s, "point_error_mean") for s in sigma_labels],
         "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CaMM")
ax1.plot(sigma_vals, [csv_get(hmm_data, s, "point_error_mean") for s in sigma_labels],
         "s--", color=COLOR_HMM, lw=1.2, ms=5, label="HMM")
ax1.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax1.set_ylabel("Mean error (m)")
ax1.set_title("(a) Point Error")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 32)

# (b) Segment accuracy
ax2.plot(sigma_vals, [csv_get(cmm_data, s, "seg_accuracy", 0) * 100 for s in sigma_labels],
         "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CaMM")
ax2.plot(sigma_vals, [csv_get(hmm_data, s, "seg_accuracy", 0) * 100 for s in sigma_labels],
         "s--", color=COLOR_HMM, lw=1.2, ms=5, label="HMM")
ax2.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("(b) Segment Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 32)
ax2.set_ylim(0, 105)

# (c) ECE
ax3.plot(sigma_vals, [csv_get(cmm_data, s, "ece_tw") for s in sigma_labels],
         "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CaMM")
ax3.plot(sigma_vals, [csv_get(hmm_data, s, "ece_tw") for s in sigma_labels],
         "s--", color=COLOR_HMM, lw=1.2, ms=5, label="HMM")
ax3.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax3.set_ylabel("ECE")
ax3.set_title("(c) ECE (Trustworthiness)")
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 32)
ax3.set_ylim(0, 0.6)

# (d) ROC AUC
ax4.plot(sigma_vals, [csv_get(cmm_data, s, "roc_auc") for s in sigma_labels],
         "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CaMM")
ax4.plot(sigma_vals, [csv_get(hmm_data, s, "roc_auc") for s in sigma_labels],
         "s--", color=COLOR_HMM, lw=1.2, ms=5, label="HMM")
ax4.axhline(0.5, color="gray", lw=0.8, ls="--")
ax4.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax4.set_ylabel("AUC")
ax4.set_title("(d) ROC AUC")
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 32)
ax4.set_ylim(0.4, 1.0)

# (e) Reliability diagram — use sigma_10 if available in detailed JSON,
#     fall back to sigma_05_sr1 or sigma_15_sr1 from sample_rate_full.json
#     then fall back to CSV bins if present
RELIABILITY_SIGMA = "sigma_10"
def _find_detail_data(detail_dict, preferred_label):
    """Find detailed data, trying preferred label first, then fallbacks."""
    if preferred_label in detail_dict:
        return detail_dict[preferred_label]
    # Try sr1 variants
    for fallback in [f"{preferred_label}_sr1", "sigma_05_sr1", "sigma_15_sr1", "sigma_25_sr1"]:
        if fallback in detail_dict:
            return detail_dict[fallback]
    # Try any key starting with preferred prefix
    for k, v in detail_dict.items():
        if k.startswith(preferred_label):
            return v
    return None

ax5.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
# Try detailed JSON first, then CSV fallback
cmm_detail = _find_detail_data(detail_cmm, RELIABILITY_SIGMA)
hmm_detail = _find_detail_data(detail_hmm, RELIABILITY_SIGMA)

reliability_plotted = False
for label, detail, color in [("CaMM", cmm_detail, COLOR_CMM), ("HMM", hmm_detail, COLOR_HMM)]:
    if detail is None:
        continue
    bins = detail.get("ece_tw_bins", [])
    if bins:
        confs = [b["mean_conf"] for b in bins if b.get("n", 0) > 0]
        accs = [b["accuracy"] for b in bins if b.get("n", 0) > 0]
        ns = [b["n"] for b in bins if b.get("n", 0) > 0]
        if confs:
            ax5.scatter(confs, accs, s=[max(n * 0.5, 10) for n in ns],
                       color=color, label=label, edgecolors="white", lw=0.5, zorder=3, alpha=0.8)
            ax5.plot(confs, accs, "-", color=color, lw=1.0, alpha=0.4)
            reliability_plotted = True

# Fallback: if no detailed bins, show note
if not reliability_plotted:
    ax5.text(0.5, 0.5, "No detailed bin data available", ha="center", va="center",
             transform=ax5.transAxes, fontsize=10, color="gray")

# Determine which sigma was actually used for the panel title
used_sigma_label = RELIABILITY_SIGMA
if cmm_detail:
    used_sigma_label = cmm_detail.get("label", RELIABILITY_SIGMA)
ax5.set_xlabel("Confidence")
ax5.set_ylabel("Accuracy")
ax5.set_title(f"(e) Reliability ({used_sigma_label})")
ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

# (f) ROC curves
ax6.plot([0, 1], [0, 1], "k--", lw=0.8)
roc_plotted = False
for label, detail, color in [("CaMM", cmm_detail, COLOR_CMM), ("HMM", hmm_detail, COLOR_HMM)]:
    if detail is None:
        continue
    fpr = detail.get("fpr")
    tpr = detail.get("tpr")
    auc = detail.get("roc_auc", np.nan)
    if fpr and tpr:
        ax6.plot(fpr, tpr, color=color, lw=1.2, label=f"{label} (AUC={auc:.3f})")
        roc_plotted = True
    elif auc is not None and not np.isnan(auc):
        # At least note the AUC even without the curve
        pass

if not roc_plotted:
    ax6.text(0.5, 0.5, "No ROC curve data available", ha="center", va="center",
             transform=ax6.transAxes, fontsize=10, color="gray")
ax6.set_xlabel("FPR")
ax6.set_ylabel("TPR")
ax6.set_title(f"(f) ROC ({used_sigma_label})")
ax6.legend(fontsize=7)
ax6.grid(alpha=0.3)

fig.suptitle("CaMM vs HMM: Sigma Sensitivity ($k=16$)", fontsize=13, fontweight="bold")
fig.tight_layout()
out = FIGS_DIR / "sigma_sweep.png"
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"Saved {out}")
print(f"  File size: {out.stat().st_size // 1024} KB")
