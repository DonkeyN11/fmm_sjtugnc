#!/usr/bin/env python3
"""Regenerate mismatch_analysis.png from mismatch_summary.csv with CaMM/HMM labels."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT / "experiments/output/4_sigma_mismatch/mismatch_summary.csv"
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_CaMM = "#2166ac"
COLOR_HMM = "#b2182b"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

if not CSV_PATH.exists():
    print(f"ERROR: {CSV_PATH} not found.")
    sys.exit(1)

# Load CSV
cmm_data = {}  # pr_value -> metrics dict
hmm_data = {}
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        algo = row["algorithm"].strip()
        label = row["label"].strip()  # e.g., "pr10_wls20"
        pr_val = int(label.replace("pr", "").split("_")[0])
        target = cmm_data if algo == "CMM" else hmm_data
        target[pr_val] = {
            "n": int(row["n"]),
            "point_error_mean": float(row["point_error_mean"]),
            "seg_accuracy": float(row["seg_accuracy"]),
            "ece_tw": float(row["ece_tw"]),
            "roc_auc": float(row["roc_auc"]),
            "trust_separation": float(row["trust_separation"]) if row["trust_separation"] else None,
        }

sigma_pr = sorted(cmm_data.keys())
mismatch = [s - 20 for s in sigma_pr]  # sigma_true - sigma_wls (wls=20)

def get_cmm(key):
    return [cmm_data[s].get(key, np.nan) for s in sigma_pr]

def get_hmm(key):
    return [hmm_data[s].get(key, np.nan) for s in sigma_pr]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

# (a) Point error vs sigma_pseudorange
ax1.plot(sigma_pr, get_cmm("point_error_mean"), "o-", color=COLOR_CaMM, lw=1.5, ms=7, label="CaMM")
ax1.plot(sigma_pr, get_hmm("point_error_mean"), "s--", color=COLOR_HMM, lw=1.5, ms=7, label="HMM")
ax1.axvline(20, color="gray", ls=":", lw=1, alpha=0.5, label=r"$\sigma_{wls}=20$m")
ax1.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax1.set_ylabel("Mean error (m)")
ax1.set_title("(a) Point Error vs True Noise"); ax1.legend(); ax1.grid(alpha=0.3)

# (b) Segment accuracy
ax2.plot(sigma_pr, [v * 100 for v in get_cmm("seg_accuracy")], "o-", color=COLOR_CaMM, lw=1.5, ms=7, label="CaMM")
ax2.plot(sigma_pr, [v * 100 for v in get_hmm("seg_accuracy")], "s--", color=COLOR_HMM, lw=1.5, ms=7, label="HMM")
ax2.axvline(20, color="gray", ls=":", lw=1, alpha=0.5)
ax2.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("(b) Segment Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

# (c) ECE — KEY PLOT
ax3.plot(sigma_pr, get_cmm("ece_tw"), "o-", color=COLOR_CaMM, lw=1.5, ms=7, label="CaMM")
ax3.plot(sigma_pr, get_hmm("ece_tw"), "s--", color=COLOR_HMM, lw=1.5, ms=7, label="HMM")
ax3.axvline(20, color="gray", ls=":", lw=1, alpha=0.5)
ax3.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax3.set_ylabel("ECE")
ax3.set_title("(c) Calibration Error (ECE) — Key Metric"); ax3.legend(); ax3.grid(alpha=0.3)

# (d) Trustworthiness separation
ax4.plot(sigma_pr, get_cmm("trust_separation"), "o-", color=COLOR_CaMM, lw=1.5, ms=7, label="CaMM")
ax4.plot(sigma_pr, get_hmm("trust_separation"), "s--", color=COLOR_HMM, lw=1.5, ms=7, label="HMM")
ax4.axhline(0, color="gray", lw=0.8, ls="--")
ax4.axvline(20, color="gray", ls=":", lw=1, alpha=0.5)
ax4.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax4.set_ylabel(r"$\Delta$Trust (correct $-$ wrong)")
ax4.set_title("(d) Trust Separation"); ax4.legend(); ax4.grid(alpha=0.3)

# (e) ECE vs true sigma for both methods (simplified reliability substitute)
ax5.plot(sigma_pr, get_cmm("ece_tw"), "o-", color=COLOR_CaMM, lw=1.5, ms=7, label="CaMM")
ax5.plot(sigma_pr, get_hmm("ece_tw"), "s--", color=COLOR_HMM, lw=1.5, ms=7, label="HMM")
ax5.axvline(20, color="gray", ls=":", lw=1, alpha=0.5)
ax5.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax5.set_ylabel("ECE")
ax5.set_title("(e) ECE vs True Noise — CaMM vs HMM"); ax5.legend(); ax5.grid(alpha=0.3)

# (f) ECE vs mismatch
ax6.plot(mismatch, get_cmm("ece_tw"), "o-", color=COLOR_CaMM, lw=1.5, ms=7, label="CaMM")
ax6.plot(mismatch, get_hmm("ece_tw"), "s--", color=COLOR_HMM, lw=1.5, ms=7, label="HMM")
ax6.axvline(0, color="gray", ls=":", lw=1, alpha=0.5)
ax6.set_xlabel(r"$\sigma_{true} - \sigma_{assumed}$ (m)")
ax6.set_ylabel("ECE")
ax6.set_title(r"(f) ECE vs Mismatch ($\sigma_{true}-\sigma_{wls}$)")
ax6.legend(); ax6.grid(alpha=0.3)
y_lim = ax6.get_ylim()
ax6.annotate("Over-\nconservative", xy=(-5, y_lim[1]*0.9), fontsize=6, ha="center", color=COLOR_CaMM)
ax6.annotate("Over-\nconfident", xy=(5, y_lim[1]*0.9), fontsize=6, ha="center", color=COLOR_HMM)

fig.suptitle("Effect of Wrong Emission Model ($\\sigma$ mismatch): CaMM vs HMM", fontsize=10, fontweight="bold")
fig.tight_layout()
out = FIGS_DIR / "mismatch_analysis.png"
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"Saved {out}  ({out.stat().st_size // 1024} KB)")
