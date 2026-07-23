#!/usr/bin/env python3
"""Regenerate degraded_comparison.png from full metrics JSON with 3×2 layout.
First run: python experiments/scripts/exp5_degraded_conditions.py --skip-match
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
METRICS_JSON = PROJECT / "experiments/output/5_degraded/degraded_full.json"
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_CMM = "#2166ac"
COLOR_FMM = "#b2182b"
CONDITION_COLORS = {"clean": "#2166ac", "fault": "#d73027", "occlusion": "#f46d43", "both": "#a6d96a"}
CONDITION_LABELS = {"clean": "Clean", "fault": "Step Fault", "occlusion": "Occlusion", "both": "Fault+Occlusion"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

if not METRICS_JSON.exists():
    print(f"ERROR: {METRICS_JSON} not found. Run exp5_degraded_conditions.py --skip-match first.")
    sys.exit(1)

with open(METRICS_JSON) as f:
    data = json.load(f)

cmm_all = data["cmm"]
fmm_all = data["fmm"]

cond_order = ["clean", "fault", "occlusion", "both"]
x = np.arange(len(cond_order))
x_labels = [CONDITION_LABELS[c] for c in cond_order]
bar_w = 0.35

def _get(mlist, cond, key, default=0):
    for m in mlist:
        if m["label"] == cond: return m.get(key, default) or default
    return default

fig, axes = plt.subplots(3, 2, figsize=(7.5, 10.5))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

# (a) Point error
ax1.bar(x - bar_w/2, [_get(cmm_all, c, "point_error_mean") for c in cond_order],
        bar_w, color=COLOR_CMM, label="CaMM", edgecolor="white", lw=0.5)
ax1.bar(x + bar_w/2, [_get(fmm_all, c, "point_error_mean") for c in cond_order],
        bar_w, color=COLOR_FMM, label="FMM", edgecolor="white", lw=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(x_labels, rotation=15, ha="right")
ax1.set_ylabel("Mean error (m)"); ax1.set_title("(a) Point Error")
ax1.legend(); ax1.grid(alpha=0.3, axis="y")

# (b) Segment accuracy
ax2.bar(x - bar_w/2, [(_get(cmm_all, c, "seg_accuracy", 0))*100 for c in cond_order],
        bar_w, color=COLOR_CMM, edgecolor="white", lw=0.5)
ax2.bar(x + bar_w/2, [(_get(fmm_all, c, "seg_accuracy", 0))*100 for c in cond_order],
        bar_w, color=COLOR_FMM, edgecolor="white", lw=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(x_labels, rotation=15, ha="right")
ax2.set_ylabel("Accuracy (%)"); ax2.set_title("(b) Segment Accuracy")
ax2.set_ylim(0, 105); ax2.grid(alpha=0.3, axis="y")

# (c) ECE
ax3.bar(x - bar_w/2, [_get(cmm_all, c, "ece_tw") for c in cond_order],
        bar_w, color=COLOR_CMM, edgecolor="white", lw=0.5)
ax3.bar(x + bar_w/2, [_get(fmm_all, c, "ece_tw") for c in cond_order],
        bar_w, color=COLOR_FMM, edgecolor="white", lw=0.5)
ax3.set_xticks(x); ax3.set_xticklabels(x_labels, rotation=15, ha="right")
ax3.set_ylabel("ECE"); ax3.set_title("(c) Expected Calibration Error")
ax3.grid(alpha=0.3, axis="y")

# (d) ROC AUC
ax4.bar(x - bar_w/2, [_get(cmm_all, c, "roc_auc") for c in cond_order],
        bar_w, color=COLOR_CMM, edgecolor="white", lw=0.5)
ax4.bar(x + bar_w/2, [_get(fmm_all, c, "roc_auc") for c in cond_order],
        bar_w, color=COLOR_FMM, edgecolor="white", lw=0.5)
ax4.axhline(0.5, color="gray", ls=":", lw=0.8)
ax4.set_xticks(x); ax4.set_xticklabels(x_labels, rotation=15, ha="right")
ax4.set_ylabel("AUC"); ax4.set_title("(d) ROC AUC")
ax4.set_ylim(0.3, 1.0); ax4.grid(alpha=0.3, axis="y")

# (e) Reliability diagram — CaMM only, all 4 conditions overlaid
ax5.plot([0, 1], [0, 1], "k--", lw=0.6, label="Perfect")
for cond in cond_order:
    m = next((m for m in cmm_all if m["label"] == cond), None)
    if m is None: continue
    bins = m.get("ece_tw_bins", [])
    if not bins: continue
    confs = [b["mean_conf"] for b in bins if b.get("n", 0) > 0]
    accs = [b["accuracy"] for b in bins if b.get("n", 0) > 0]
    if not confs: continue
    ece = m.get("ece_tw", 0)
    ax5.plot(confs, accs, "o-", color=CONDITION_COLORS.get(cond, "gray"),
             lw=1.0, ms=5, label=f"{CONDITION_LABELS[cond]} (ECE={ece:.3f})")
ax5.set_xlabel("Confidence"); ax5.set_ylabel("Accuracy")
ax5.set_title("(e) Reliability Diagram — CaMM"); ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# (f) ROC curves — CaMM only
ax6.plot([0, 1], [0, 1], "k--", lw=0.6)
for cond in cond_order:
    m = next((m for m in cmm_all if m["label"] == cond), None)
    if m is None: continue
    fpr = m.get("fpr"); tpr = m.get("tpr"); auc = m.get("roc_auc", 0.5)
    if fpr and tpr:
        ax6.plot(fpr, tpr, lw=1.2, color=CONDITION_COLORS.get(cond, "gray"),
                 label=f"{CONDITION_LABELS[cond]} (AUC={auc:.3f})")
ax6.set_xlabel("FPR"); ax6.set_ylabel("TPR")
ax6.set_title("(f) ROC Curves — CaMM"); ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

fig.suptitle("CaMM vs FMM Under Degraded Conditions (σ=30m)", fontsize=13, fontweight="bold")
fig.tight_layout()
out = FIGS_DIR / "degraded_comparison.png"
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"Saved {out}")
