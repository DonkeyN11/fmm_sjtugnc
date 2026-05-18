#!/usr/bin/env python3
"""Generate publication-quality figures for the CMM IEEE T-ITS article.

Produces:
  1. reliability_diagram.png — 10-bin calibration curve, CMM vs FMM at 5m threshold
  2. ece_ablation.png        — ECE ablation bar chart (FMM→CMM→CMM+lag→CMM+PHMI)
  3. lag_sweep_multitraj.png — ECE vs lag_steps for all 7 trajectories
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
MR_DIR = PROJECT / "dataset-hainan-06" / "mr"
FIGS_DIR = PROJECT / "docs" / "Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Reliability Diagram — CMM vs FMM at 5m threshold
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
        ("CMM (anisotropic Mahalanobis)", cmm_bins, "#2166ac", "o"),
        ("FMM (isotropic Euclidean)", fmm_bins, "#b2182b", "s"),
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
    """Hard-coded results from exp4_ablation_ece.py (real data, 5m threshold)."""
    configs = [
        ("FMM\n(isotropic)",  0.978),
        ("CMM\n(aniso, L=0)", 0.121),
        ("CMM\n(aniso, L=20)", 0.261),
        ("CMM\n(aniso, L=20,\nPHMI)", 0.261),
    ]
    ece_decompose = [
        ("FMM baseline", 0.978),
        ("+ anisotropic Mahalanobis", 0.121),
        ("+ fixed-lag smoothing", 0.261),
        ("+ PHMI integrity", 0.261),
    ]

    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    colors = ["#b2182b", "#2166ac", "#4393c3", "#92c5de"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Left: grouped bars
    x = np.arange(len(labels))
    bars = ax1.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylabel("ECE (trustworthiness)")
    ax1.set_title("Absolute ECE")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Right: ECE decomposition (stepwise delta)
    step_labels = [s[0] for s in ece_decompose]
    step_values = [s[1] for s in ece_decompose]
    deltas = [0.0]
    for i in range(1, len(step_values)):
        deltas.append(step_values[i] - step_values[i - 1])

    x_s = np.arange(len(step_labels))
    colors_s = ["#757575", "#d73027", "#fee090", "#4575b4"]
    bars_s = ax2.bar(x_s, step_values, color=colors_s, width=0.5, edgecolor="white", linewidth=0.5)
    # Annotate deltas on bars
    for i, (bar, val, d) in enumerate(zip(bars_s, step_values, deltas)):
        lbl = f"{val:.3f}"
        if i > 0:
            lbl += f"\n({d:+.3f})"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 lbl, ha="center", va="bottom", fontsize=6)

    ax2.set_xticks(x_s)
    ax2.set_xticklabels(step_labels, rotation=25, ha="right", fontsize=6)
    ax2.set_title("ECE Decomposition")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("ECE Ablation Study (Real Data, 5 m Threshold)", fontsize=10, fontweight="bold")
    fig.tight_layout()

    out = FIGS_DIR / "ece_ablation.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Lag Sweep ECE Curves — all 7 trajectories
# ══════════════════════════════════════════════════════════════════════════════

def fig_lag_sweep():
    sweep_dir = MR_DIR / "multitraj_sweep"
    if not sweep_dir.is_dir():
        print(f"SKIP: {sweep_dir} not found — run exp1_multitraj_lag_sweep.py first")
        return

    import csv

    # Lag values are embedded in filenames: cmm_all_lag000.csv, cmm_all_lag005.csv, ...
    lags = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    traj_ids = [11, 12, 13, 14, 21, 22, 23]
    ece_tw = {tid: [] for tid in traj_ids}

    for lag in lags:
        fname = sweep_dir / f"cmm_all_lag{lag:03d}.csv"
        if not fname.exists():
            continue
        per_traj = {tid: {"confs": [], "corrects": []} for tid in traj_ids}
        with open(fname, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    tid = int(row.get("id", -1))
                    tw = float(row.get("trustworthiness", 0))
                    eu_dist = float(row.get("eu_dist", -1))
                    if eu_dist < 0:
                        # Try computing from ogeom and pgeom if present
                        ogeom = row.get("ogeom", "")
                        pgeom = row.get("pgeom", "")
                        if ogeom and pgeom and "POINT" in ogeom:
                            import re
                            m1 = re.findall(r"[-+]?\d*\.\d+", ogeom)
                            m2 = re.findall(r"[-+]?\d*\.\d+", pgeom)
                            if len(m1) >= 2 and len(m2) >= 2:
                                dx = float(m1[0]) - float(m2[0])
                                dy = float(m1[1]) - float(m2[1])
                                eu_dist = np.sqrt(dx**2 + dy**2) * 111320.0
                except (ValueError, KeyError):
                    continue
                if tid not in per_traj:
                    continue
                per_traj[tid]["confs"].append(tw)
                per_traj[tid]["corrects"].append(1.0 if eu_dist <= 5.0 else 0.0)

        for tid in traj_ids:
            d = per_traj[tid]
            if not d["confs"]:
                ece_tw[tid].append(np.nan)
                continue
            confs = np.array(d["confs"])
            corrects = np.array(d["corrects"])
            # 10-bin ECE
            n_bins = 10
            ece = 0.0
            total = len(confs)
            if total == 0:
                ece_tw[tid].append(np.nan)
                continue
            for b in range(n_bins):
                lo = b / n_bins
                hi = (b + 1) / n_bins
                mask = (confs >= lo) & (confs < hi)
                if b == n_bins - 1:
                    mask = (confs >= lo) & (confs <= hi)
                n_b = mask.sum()
                if n_b == 0:
                    continue
                acc_b = corrects[mask].mean()
                conf_b = confs[mask].mean()
                ece += (n_b / total) * abs(acc_b - conf_b)
            ece_tw[tid].append(ece)

    # Also compute mean ECE across all trajectories per lag
    mean_ece = []
    for li in range(len(lags)):
        vals = [ece_tw[tid][li] for tid in traj_ids
                if li < len(ece_tw[tid]) and not np.isnan(ece_tw[tid][li])]
        mean_ece.append(np.mean(vals) if vals else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a",
              "#66a61e", "#e6ab02", "#a6761d"]
    markers = ["o", "s", "D", "^", "v", "<", ">"]

    for idx, tid in enumerate(traj_ids):
        values = ece_tw[tid]
        valid_lags = [l for l, v in zip(lags, values) if not np.isnan(v)]
        valid_vals = [v for v in values if not np.isnan(v)]
        ax1.plot(valid_lags, valid_vals,
                 color=colors[idx], marker=markers[idx],
                 markersize=4, linewidth=1.2, label=f"Traj {tid}")

    ax1.set_xlabel("lag_steps")
    ax1.set_ylabel("ECE (trustworthiness)")
    ax1.set_title("Per-Trajectory ECE vs Lag")
    ax1.legend(fontsize=5.5, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Mean ECE across trajectories
    clean_lags = [l for l, v in zip(lags, mean_ece) if not np.isnan(v)]
    clean_mean = [v for v in mean_ece if not np.isnan(v)]
    ax2.plot(clean_lags, clean_mean, "k-o", linewidth=1.5, markersize=5)
    ax2.fill_between(clean_lags,
                     [v - 0.02 for v in clean_mean],
                     [v + 0.02 for v in clean_mean],
                     alpha=0.15, color="black")
    ax2.set_xlabel("lag_steps")
    ax2.set_ylabel("Mean ECE (trustworthiness)")
    ax2.set_title("Cross-Trajectory Mean ECE vs Lag")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Fixed-Lag Smoothing Calibration Trade-off", fontsize=10, fontweight="bold")
    fig.tight_layout()

    out = FIGS_DIR / "lag_sweep_multitraj.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating CMM article figures...")
    fig_reliability_diagram()
    fig_ece_ablation()
    fig_lag_sweep()
    print("Done.")
