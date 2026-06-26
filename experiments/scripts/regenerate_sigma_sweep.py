#!/usr/bin/env python3
"""Regenerate sigma_sweep.png from full metrics JSON with 3×2 layout.
First run: python experiments/scripts/exp3_full_matching.py --skip-match
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
METRICS_JSON = PROJECT / "experiments/output/3_full_matching/sigma_sweep_full.json"
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_CMM = "#2166ac"
COLOR_FMM = "#b2182b"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

if not METRICS_JSON.exists():
    print(f"ERROR: {METRICS_JSON} not found. Run exp3_full_matching.py --skip-match first.")
    sys.exit(1)

with open(METRICS_JSON) as f:
    data = json.load(f)

cmm_all = data["cmm"]
fmm_all = data["fmm"]

sigmas = sorted(
    set(m["label"] for m in cmm_all if "fault" not in m["label"] and "occ" not in m["label"] and "_sr" not in m["label"]),
    key=lambda s: int(s.replace("sigma_", ""))
)

def get(m_list, s, key, default=0):
    for m in m_list:
        if m["label"] == s: return m.get(key, default)
    return default

fig, axes = plt.subplots(3, 2, figsize=(7.5, 10.5))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat
sigma_vals = [int(s.replace("sigma_", "")) for s in sigmas]

# (a) Point error
ax1.plot(sigma_vals, [get(cmm_all, s, "point_error_mean") for s in sigmas], "o-", color=COLOR_CMM, lw=1.2, ms=5, label="TMM")
ax1.plot(sigma_vals, [get(fmm_all, s, "point_error_mean") for s in sigmas], "s-", color=COLOR_FMM, lw=1.2, ms=5, label="FMM")
ax1.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax1.set_ylabel("Mean error (m)")
ax1.set_title("(a) Point Error"); ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xlim(0, 32)

# (b) Segment accuracy
ax2.plot(sigma_vals, [(get(cmm_all, s, "seg_accuracy", 0) or 0) * 100 for s in sigmas], "o-", color=COLOR_CMM, lw=1.2, ms=5)
ax2.plot(sigma_vals, [(get(fmm_all, s, "seg_accuracy", 0) or 0) * 100 for s in sigmas], "s-", color=COLOR_FMM, lw=1.2, ms=5)
ax2.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("(b) Segment Accuracy"); ax2.grid(alpha=0.3); ax2.set_xlim(0, 32); ax2.set_ylim(0, 105)

# (c) ECE
ax3.plot(sigma_vals, [get(cmm_all, s, "ece_tw") for s in sigmas], "o-", color=COLOR_CMM, lw=1.2, ms=5, label="TMM")
ax3.plot(sigma_vals, [get(fmm_all, s, "ece_tw") for s in sigmas], "s-", color=COLOR_FMM, lw=1.2, ms=5, label="FMM")
ax3.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax3.set_ylabel("ECE")
ax3.set_title("(c) ECE (Trustworthiness)"); ax3.legend(); ax3.grid(alpha=0.3)
ax3.set_xlim(0, 32); ax3.set_ylim(0, 0.6)

# (d) ROC AUC
ax4.plot(sigma_vals, [get(cmm_all, s, "roc_auc") for s in sigmas], "o-", color=COLOR_CMM, lw=1.2, ms=5)
ax4.plot(sigma_vals, [get(fmm_all, s, "roc_auc") for s in sigmas], "s-", color=COLOR_FMM, lw=1.2, ms=5)
ax4.axhline(0.5, color="gray", lw=0.8, ls="--")
ax4.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax4.set_ylabel("AUC")
ax4.set_title("(d) ROC AUC"); ax4.grid(alpha=0.3); ax4.set_xlim(0, 32); ax4.set_ylim(0.4, 1.0)

# (e) Reliability diagram for sigma=10
mid_sigma = "sigma_10"
for label, metrics, color in [("TMM", cmm_all, COLOR_CMM), ("FMM", fmm_all, COLOR_FMM)]:
    bins = get(metrics, mid_sigma, "ece_tw_bins", [])
    if not bins:
        for s in sigmas:
            bins = get(metrics, s, "ece_tw_bins", [])
            if bins: break
    if not bins: continue
    confs = [b["mean_conf"] for b in bins if b.get("n", 0) > 0]
    accs = [b["accuracy"] for b in bins if b.get("n", 0) > 0]
    if confs:
        ax5.scatter(confs, accs, s=25, color=color, label=label, edgecolors="white", lw=0.5)
ax5.plot([0, 1], [0, 1], "k--", lw=0.8)
ax5.set_xlabel("Confidence"); ax5.set_ylabel("Accuracy")
ax5.set_title("(e) Reliability (σ=10m)"); ax5.legend(); ax5.grid(alpha=0.3)

# (f) ROC curves for sigma=10
ax6.plot([0, 1], [0, 1], "k--", lw=0.8)
for label, metrics, color in [("TMM", cmm_all, COLOR_CMM), ("FMM", fmm_all, COLOR_FMM)]:
    fpr = get(metrics, mid_sigma, "fpr")
    tpr = get(metrics, mid_sigma, "tpr")
    auc = get(metrics, mid_sigma, "roc_auc")
    if not fpr:
        for s in sigmas:
            fpr = get(metrics, s, "fpr"); tpr = get(metrics, s, "tpr")
            auc = get(metrics, s, "roc_auc")
            if fpr: break
    if fpr and tpr:
        ax6.plot(fpr, tpr, color=color, lw=1.2, label=f"{label} (AUC={auc:.3f})")
ax6.set_xlabel("FPR"); ax6.set_ylabel("TPR")
ax6.set_title("(f) ROC (σ=10m)"); ax6.legend(); ax6.grid(alpha=0.3)

fig.suptitle("TMM vs FMM: Matching Performance (k=16)", fontsize=13, fontweight="bold")
fig.tight_layout()
out = FIGS_DIR / "sigma_sweep.png"
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"Saved {out}")
