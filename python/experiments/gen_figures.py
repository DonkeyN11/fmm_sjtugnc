#!/usr/bin/env python3
"""Generate publication-quality figures for the CaMM IEEE T-ITS article.

Produces:
  1. reliability_diagram.png — 10-bin calibration curve, CaMM vs HMM at 5m threshold
  2. ece_ablation.png        — ECE ablation bar chart (HMM → CaMM aniso+norm → CaMM full+reverse)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
MR_DIR = PROJECT / "data/real_vehicle" / "mr"
FIGS_DIR = PROJECT / "docs" / "Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Reliability Diagram — CaMM vs HMM at 5m threshold
# ══════════════════════════════════════════════════════════════════════════════

def fig_reliability_diagram():
    json_path = MR_DIR / "exp1_reliability.json"
    if not json_path.exists():
        print(f"SKIP: {json_path} not found — run exp1_reliability_diagram.py first")
        return

    with open(json_path) as f:
        data = json.load(f)

    key = "5"
    if key not in data.get("per_threshold", {}):
        print(f"SKIP: threshold {key} not in reliability JSON")
        return

    cmm_bins = data["per_threshold"][key]["cmm"]["per_bin"]
    fmm_bins = data["per_threshold"][key]["fmm"]["per_bin"]

    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")

    for label, bins, color, marker in [
        ("CaMM (anisotropic Mahalanobis)", cmm_bins, "#2166ac", "o"),
        ("HMM (isotropic Euclidean)", fmm_bins, "#b2182b", "s"),
    ]:
        confs = [b["mean_conf"] for b in bins]
        accs  = [b["accuracy"] for b in bins]
        w = max(1, int(max(len(bins) * 3, 6)))  # marker size proportional
        ax.scatter(confs, accs, s=w * 4, c=color, marker=marker,
                   edgecolors="white", linewidth=0.5, label=label, zorder=5)

    ax.set_xlabel("Mean Trustworthiness (confidence)")
    ax.set_ylabel("Observed Accuracy")
    ax.set_title("Reliability Diagram — 5 m error threshold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out = FIGS_DIR / "reliability_diagram.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: ECE Ablation Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_ece_ablation():
    """Results from paper ablation study (real data, 5m threshold, 13,052 epochs).
    Three configs: HMM baseline → CaMM aniso+norm HMM → CaMM full + reverse guard."""
    configs = [
        ("HMM\n(isotropic\nbaseline)",  0.107),
        ("CaMM\n(anisotropic\n+ norm. HMM)", 0.072),
        ("CaMM full\n(+ reverse\nguard 3%)", 0.069),
    ]
    ece_decompose = [
        ("HMM baseline\n(isotropic)", 0.107),
        ("+ anisotropic\n+ norm. HMM", 0.072),
        ("+ cumulative\nreverse guard", 0.069),
    ]

    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    colors = ["#b2182b", "#2166ac", "#1b7837"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Left: grouped bars
    x = np.arange(len(labels))
    bars = ax1.bar(x, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("ECE (trustworthiness)", fontsize=10)
    ax1.set_title("Absolute ECE", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 0.13)
    ax1.grid(axis="y", alpha=0.3)

    # Right: ECE decomposition (stepwise delta)
    step_labels = [s[0] for s in ece_decompose]
    step_values = [s[1] for s in ece_decompose]
    deltas = [0.0]
    for i in range(1, len(step_values)):
        deltas.append(step_values[i] - step_values[i - 1])

    x_s = np.arange(len(step_labels))
    colors_s = ["#b2182b", "#2166ac", "#1b7837"]
    bars_s = ax2.bar(x_s, step_values, color=colors_s, width=0.55, edgecolor="white", linewidth=0.8)
    # Annotate deltas on bars
    for i, (bar, val, d) in enumerate(zip(bars_s, step_values, deltas)):
        lbl = f"{val:.3f}"
        if i > 0:
            lbl += f"\n({d:+.3f})"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 lbl, ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x_s)
    ax2.set_xticklabels(step_labels, rotation=20, ha="right", fontsize=7.5)
    ax2.set_title("ECE Decomposition", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.13)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("ECE Ablation Study (Real Data, 5 m Threshold, 13,052 epochs)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = FIGS_DIR / "ece_ablation.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating CaMM article figures...")
    fig_reliability_diagram()
    fig_ece_ablation()
    print("Done.")
