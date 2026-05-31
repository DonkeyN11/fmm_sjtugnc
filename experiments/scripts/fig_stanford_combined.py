#!/usr/bin/env python3
"""
Combined Stanford plot: all 4 sigma=30 conditions in one 2×2 figure.

Copied to docs/.../figs/stanford_combined.png for the paper.
"""

import csv, math, sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300
plt.rcParams.update({"font.size":7,"axes.labelsize":8,"axes.titlesize":9,
    "legend.fontsize":6,"xtick.labelsize":6,"ytick.labelsize":6,
    "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "experiments/data/sigma_30"
PAPER_FIGS = PROJECT_ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"

CONDITIONS = [
    ("no_occlusion/no_fault",        "Clean (σ=30m)",           "#4dac26"),
    ("no_occlusion/with_fault",      "Step Fault (U(100,500)m)", "#e66101"),
    ("with_occlusion/no_fault",      "Cross-Road Occlusion",     "#5e3c99"),
    ("with_occlusion/with_fault",    "Occlusion + Fault",        "#b2182b"),
]

def haversine_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1+lat2)/2)
    dx = (lon1-lon2)*111320*math.cos(mlat)
    dy = (lat1-lat2)*111320
    return math.sqrt(dx*dx+dy*dy)

def load_errors(data_dir: Path):
    """Load horizontal errors and PL values."""
    obs_file = data_dir / "observations.csv"
    gt_file = data_dir / "ground_truth_points.csv"
    if not obs_file.exists() or not gt_file.exists():
        return [], []

    # Load GT points keyed by (id, seq)
    gt = {}
    with open(gt_file, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            gt[(row["id"].strip(), int(row["seq"]))] = (float(row["x"]), float(row["y"]))

    errors = []
    pls = []
    with open(obs_file, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            key = (row["id"].strip(), int(row["seq"]))
            if key not in gt: continue
            gt_x, gt_y = gt[key]
            obs_x, obs_y = float(row["x"]), float(row["y"])
            err = haversine_m(obs_x, obs_y, gt_x, gt_y)
            pl = float(row["protection_level"])
            # Convert PL from degrees to meters
            mlat = math.radians((obs_y+gt_y)/2)
            pl_m = pl * 111320.0 * math.cos(mlat)
            errors.append(err)
            pls.append(pl_m)
    return np.array(errors), np.array(pls)

fig, axes = plt.subplots(2, 2, figsize=(9, 8))

for idx, (subdir, title, color) in enumerate(CONDITIONS):
    ax = axes.flat[idx]
    data_dir = DATA_ROOT / subdir
    errors, pls = load_errors(data_dir)
    if len(errors) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        continue

    # Count HMI
    hmi = errors > pls
    n_hmi = np.sum(hmi)
    n_total = len(errors)
    p_md = n_hmi / n_total

    # Scatter: normal in blue, HMI in red
    ok_mask = ~hmi
    ax.scatter(pls[ok_mask], errors[ok_mask], s=1, alpha=0.3, color=color, label=f"OK ({n_total-n_hmi})")
    ax.scatter(pls[hmi], errors[hmi], s=6, alpha=0.8, color="red", marker="x", label=f"HMI ({n_hmi})")

    # Diagonal
    mx = max(np.max(pls), np.max(errors)) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5)
    ax.fill_between([0, mx], [0, mx], mx, alpha=0.08, color="red", label="HMI zone")

    # Stats text
    ax.text(0.05, 0.95,
            f"n={n_total}  P_md={p_md:.4f}\n"
            f"PL mean={np.mean(pls):.0f}m  median={np.median(pls):.0f}m\n"
            f"Err mean={np.mean(errors):.0f}m  P95={np.percentile(errors,95):.0f}m",
            transform=ax.transAxes, fontsize=6, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Protection Level (m)"); ax.set_ylabel("Horizontal Error (m)")
    ax.set_title(title); ax.legend(fontsize=5, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)

fig.suptitle("Stanford Diagram: RAIM Protection Level vs Horizontal Error (σ=30m)",
             fontsize=10, fontweight="bold")
fig.tight_layout()

out = PAPER_FIGS / "stanford_combined.png"
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"Saved: {out}")
