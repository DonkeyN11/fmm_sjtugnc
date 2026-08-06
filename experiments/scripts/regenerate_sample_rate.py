#!/usr/bin/env python3
"""Regenerate sample_rate_sensitivity.png from sample_rate_full.json with CaMM/HMM labels."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
JSON_PATH = PROJECT / "experiments/output/3_full_matching/sample_rate_full.json"
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_CaMM = "#2166ac"
COLOR_HMM = "#b2182b"
SIGMA_STYLES = {
    "sigma_05": {"color": "#2166ac", "marker": "o", "label": r"$\sigma_\rho=5$m"},
    "sigma_15": {"color": "#d95f02", "marker": "s", "label": r"$\sigma_\rho=15$m"},
    "sigma_25": {"color": "#7570b3", "marker": "D", "label": r"$\sigma_\rho=25$m"},
}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

if not JSON_PATH.exists():
    print(f"ERROR: {JSON_PATH} not found.")
    sys.exit(1)

with open(JSON_PATH) as f:
    data = json.load(f)

cmm_data = {m["label"]: m for m in data["cmm"]}
hmm_data = {m["label"]: m for m in data["fmm"]}

sr_intervals = [1, 2, 5, 10]
sigma_levels = ["sigma_05", "sigma_15", "sigma_25"]

def get_vals(data_dict, sigma, intervals, key):
    vals = []
    for iv in intervals:
        label = f"{sigma}_sr{iv}"
        m = data_dict.get(label, {})
        v = m.get(key)
        vals.append(v if v is not None else np.nan)
    return vals

fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))
ax1, ax2, ax3, ax4 = axes.flat

for sigma in sigma_levels:
    s = SIGMA_STYLES[sigma]
    sr = np.array(sr_intervals)

    # (a) Segment accuracy
    cmm_acc = [v * 100 if v else np.nan for v in get_vals(cmm_data, sigma, sr_intervals, "seg_accuracy")]
    hmm_acc = [v * 100 if v else np.nan for v in get_vals(hmm_data, sigma, sr_intervals, "seg_accuracy")]
    ax1.plot(sr, cmm_acc, "-o", color=s["color"], lw=1.2, ms=5, label=f"CaMM {s['label']}")
    ax1.plot(sr, hmm_acc, "--s", color=s["color"], lw=1.2, ms=5, alpha=0.6, label=f"HMM {s['label']}")

    # (b) ECE
    cmm_ece = get_vals(cmm_data, sigma, sr_intervals, "ece_tw")
    hmm_ece = get_vals(hmm_data, sigma, sr_intervals, "ece_tw")
    ax2.plot(sr, cmm_ece, "-o", color=s["color"], lw=1.2, ms=5)
    ax2.plot(sr, hmm_ece, "--s", color=s["color"], lw=1.2, ms=5, alpha=0.6)

    # (c) ROC AUC
    cmm_auc = get_vals(cmm_data, sigma, sr_intervals, "roc_auc")
    hmm_auc = get_vals(hmm_data, sigma, sr_intervals, "roc_auc")
    ax3.plot(sr, cmm_auc, "-o", color=s["color"], lw=1.2, ms=5)
    ax3.plot(sr, hmm_auc, "--s", color=s["color"], lw=1.2, ms=5, alpha=0.6)

    # (d) Point error
    cmm_pe = get_vals(cmm_data, sigma, sr_intervals, "point_error_mean")
    hmm_pe = get_vals(hmm_data, sigma, sr_intervals, "point_error_mean")
    ax4.plot(sr, cmm_pe, "-o", color=s["color"], lw=1.2, ms=5)
    ax4.plot(sr, hmm_pe, "--s", color=s["color"], lw=1.2, ms=5, alpha=0.6)

ax1.set_xlabel("Sample Interval (s)"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("(a) Segment Accuracy"); ax1.legend(fontsize=5.5, ncol=2); ax1.grid(alpha=0.3)

ax2.set_xlabel("Sample Interval (s)"); ax2.set_ylabel("ECE")
ax2.set_title("(b) ECE (Trustworthiness)"); ax2.grid(alpha=0.3)

ax3.set_xlabel("Sample Interval (s)"); ax3.set_ylabel("AUC")
ax3.axhline(0.5, color="gray", lw=0.8, ls="--")
ax3.set_title("(c) ROC AUC"); ax3.grid(alpha=0.3)

ax4.set_xlabel("Sample Interval (s)"); ax4.set_ylabel("Mean Error (m)")
ax4.set_title("(d) Point Error"); ax4.grid(alpha=0.3)

# Legend showing CaMM (solid) vs HMM (dashed)
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color="black", lw=1.5, linestyle="-", marker="o", ms=5, label="CaMM (solid)"),
    Line2D([0], [0], color="black", lw=1.5, linestyle="--", marker="s", ms=5, label="HMM (dashed)"),
] + [Line2D([0], [0], color=SIGMA_STYLES[s]["color"], lw=1.5, label=SIGMA_STYLES[s]["label"]) for s in sigma_levels]
fig.legend(handles=custom_lines, loc="lower center", ncol=5, fontsize=7, frameon=False)

fig.suptitle("Sample Rate Sensitivity: CaMM vs HMM ($k=16$)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0.08, 1, 0.95])
out = FIGS_DIR / "sample_rate_sensitivity.png"
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"Saved {out}  ({out.stat().st_size // 1024} KB)")
