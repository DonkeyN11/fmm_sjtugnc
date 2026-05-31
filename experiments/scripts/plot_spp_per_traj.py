#!/usr/bin/env python3
"""Generate per-trajectory SPP error time-series figures from NMEA SPP data."""
import csv, math
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
plt.rcParams.update({"font.size":8,"axes.labelsize":9,"axes.titlesize":10,
    "legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
    "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "experiments/output/spp_error"
OUT_DIR = IN_DIR

# Load all data per trajectory
by_traj = defaultdict(list)
with open(IN_DIR / "spp_vs_rtk_all.csv", newline="") as f:
    for row in csv.DictReader(f):
        by_traj[int(row["traj"])].append(row)

stats = {}
with open(IN_DIR / "spp_vs_rtk_stats.csv", newline="") as f:
    for row in csv.DictReader(f):
        stats[int(row["traj"])] = row

order = [11,12,13,14,21,22,23]

for tid in order:
    rows = by_traj[tid]
    errs = np.array([float(r["err_m"]) for r in rows])
    sdes = np.array([float(r["sde_m"]) for r in rows])
    sdns = np.array([float(r["sdn_m"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Error time series
    ax = axes[0]
    ax.plot(range(len(errs)), errs, lw=0.3, color="#2166ac", alpha=0.8)
    ax.axhline(float(stats[tid].get("mean",0)), color="red", lw=0.8, ls="--",
               label=f"mean={np.mean(errs):.1f}m")
    ax.axhline(float(stats[tid].get("p95",0)), color="orange", lw=0.8, ls=":",
               label=f"P95={np.percentile(errs,95):.1f}m")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Horiz. error (m)")
    ax.set_title(f"Traj {tid}: SPP Error Time Series"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    ymax = min(max(np.percentile(errs,95)*2, 20), 50)
    ax.set_ylim(0, ymax)

    # (b) Error histogram
    ax = axes[1]
    ax.hist(errs, bins=40, density=True, alpha=0.6, color="#2166ac",
            edgecolor="white", lw=0.3)
    for val, c, ls, lbl in [(np.median(errs),"red","--",f"med={np.median(errs):.1f}"),
                              (np.percentile(errs,95),"orange",":",f"P95={np.percentile(errs,95):.1f}")]:
        ax.axvline(val, color=c, lw=1, ls=ls, label=lbl)
    ax.set_xlabel("Error (m)"); ax.set_ylabel("Density")
    ax.set_title(f"Traj {tid}: Error Distribution (n={len(errs)})"); ax.legend(fontsize=6)
    ax.grid(alpha=0.3); ax.set_xlim(0, np.percentile(errs,99)*1.1)

    # (c) Error vs formal sigma
    ax = axes[2]
    sigma_h = np.sqrt(sdes**2 + sdns**2)
    ax.scatter(sigma_h, errs, s=2, alpha=0.3, color="#2166ac")
    mx = max(np.max(sigma_h), np.max(errs)) * 1.1
    ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5, label="y=x (ideal)")
    ax.set_xlabel("SPP formal σ_H (m)"); ax.set_ylabel("Horiz. error (m)")
    ax.set_title(f"Traj {tid}: Error vs Formal σ"); ax.legend(fontsize=6)
    ax.grid(alpha=0.3); ax.set_xlim(0, np.percentile(sigma_h, 99)*1.1)

    fig.suptitle(f"SPP vs RTK: Trajectory {tid}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"spp_error_traj{tid}.png", dpi=DPI)
    plt.close(fig)

# ── Combined overlay time series ──
fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(order)))
for idx, tid in enumerate(order):
    errs = np.array([float(r["err_m"]) for r in by_traj[tid]])
    ax.plot(range(len(errs)), errs, lw=0.2, color=colors[idx], alpha=0.7, label=f"Traj {tid}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Horiz. error (m)")
ax.set_title("SPP Error: All Trajectories Overlay"); ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3); ax.set_ylim(0, 12)
fig.tight_layout(); fig.savefig(OUT_DIR / "spp_error_all_traj_overlay.png", dpi=DPI); plt.close()

print(f"Done: {OUT_DIR}/spp_error_traj*.png ({len(order)} files)")
