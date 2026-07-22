#!/usr/bin/env python3
"""Plot SPP vs RTK error analysis figures."""
import csv, math, sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR = PROJECT_ROOT / "experiments/output/spp_error"
OUT_DIR = IN_DIR

# Load data
by_traj = defaultdict(list)
all_data = []
with open(IN_DIR / "spp_vs_rtk_all.csv", newline="") as f:
    for row in csv.DictReader(f):
        tid = int(row["traj"]); err = float(row["err_m"])
        all_data.append(row)
        by_traj[tid].append(row)

all_errs = np.array([float(r["err_m"]) for r in all_data])
order = [11,12,13,14,21,22,23]

# ── Figure 1: Per-trajectory subplots ──
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, tid in enumerate(order):
    ax = axes.flat[idx]
    rows = by_traj[tid]
    errs = np.array([float(r["err_m"]) for r in rows])
    ax.hist(errs, bins=40, density=True, alpha=0.6, color="#2166ac",
            edgecolor="white", lw=0.3)
    for val, c, ls, lbl in [(np.median(errs),"red","--","median"),
                              (np.percentile(errs,95),"orange",":","P95")]:
        ax.axvline(val, color=c, lw=1, ls=ls, label=f"{lbl}={val:.1f}m")
    ax.set_title(f"Traj {tid} (n={len(rows)})"); ax.legend(fontsize=5)
    ax.grid(alpha=0.3); ax.set_xlim(0, np.percentile(errs, 98)*1.1)
axes.flat[7].axis("off")
fig.suptitle("SPP vs RTK: Per-Trajectory Error Distribution", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT_DIR/"spp_error_per_traj.png", dpi=DPI); plt.close()

# ── Figure 2: Combined summary ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# (a) Histogram
ax = axes[0,0]
ax.hist(all_errs, bins=60, density=True, alpha=0.6, color="#2166ac",
        edgecolor="white", lw=0.3)
for val, c, ls, lbl in [(np.median(all_errs),"red","--",f"median={np.median(all_errs):.1f}m"),
                         (np.percentile(all_errs,95),"orange",":",f"P95={np.percentile(all_errs,95):.1f}m")]:
    ax.axvline(val, color=c, lw=1.5, ls=ls, label=lbl)
ax.set_xlabel("Horizontal error (m)"); ax.set_ylabel("Density")
ax.set_title(f"(a) Error Distribution (n={len(all_errs)})"); ax.legend(); ax.grid(alpha=0.3)

# (b) CDF
ax = axes[0,1]
se = np.sort(all_errs); cdf = np.arange(1,len(se)+1)/len(se)
ax.plot(se, cdf, lw=1.5, color="#2166ac")
ax.axhline(0.5, color="gray", lw=0.8, ls="--")
ax.axhline(0.68, color="gray", lw=0.8, ls="--")
ax.axhline(0.95, color="gray", lw=0.8, ls="--")
for pct, c in [(50,0.5),(68,0.68),(95,0.95)]:
    v = np.percentile(all_errs, pct)
    ax.axvline(v, color="red", lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("Error (m)"); ax.set_ylabel("CDF")
ax.set_title("(b) Cumulative Distribution"); ax.grid(alpha=0.3)
ax.set_xlim(0, np.percentile(all_errs, 99))

# (c) Box plot per traj
ax = axes[0,2]
data_boxes = [np.array([float(r["err_m"]) for r in by_traj[t]]) for t in order]
bp = ax.boxplot(data_boxes, labels=[str(t) for t in order],
                patch_artist=True, showfliers=False, widths=0.6)
for patch in bp["boxes"]: patch.set_facecolor("#92c5de"); patch.set_alpha(0.7)
ax.set_xlabel("Trajectory"); ax.set_ylabel("Error (m)")
ax.set_title("(c) Per-Trajectory Distribution"); ax.grid(alpha=0.3, axis="y")

# (d) Error vs SPP sigma
ax = axes[1,0]
sdes = np.array([float(r["sde_m"]) for r in all_data])
ax.scatter(sdes, all_errs, s=1, alpha=0.3, color="#2166ac")
ax.plot([0,max(sdes)],[0,max(sdes)],"k--",lw=0.8,alpha=0.5)
ax.set_xlabel("SPP σ_E (m)"); ax.set_ylabel("Horiz. error (m)")
ax.set_title("(d) Error vs SPP Formal Error"); ax.grid(alpha=0.3)

# (e) Mean+median bar chart
ax = axes[1,1]
x = np.arange(len(order)); w = 0.35
stats_csv = IN_DIR / "spp_vs_rtk_stats.csv"
means = {}; medians = {}
with open(stats_csv, newline="") as f:
    for row in csv.DictReader(f):
        tid = int(row["traj"]); means[tid]=float(row["mean"]); medians[tid]=float(row["median"])
ax.bar(x-w/2, [means[t] for t in order], w, color="#2166ac", label="Mean", edgecolor="white",lw=0.5)
ax.bar(x+w/2, [medians[t] for t in order], w, color="#b2182b", label="Median", edgecolor="white",lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in order])
ax.set_ylabel("Error (m)"); ax.set_title("(e) Mean & Median"); ax.legend(); ax.grid(alpha=0.3,axis="y")

# (f) Text summary
ax = axes[1,2]; ax.axis("off")
lines = [f"SPP vs RTK Error Summary",f"","n = {len(all_errs)} epochs",
    f"Mean  = {np.mean(all_errs):.3f} m",f"Median = {np.median(all_errs):.3f} m",
    f"RMSE  = {np.sqrt(np.mean(all_errs**2)):.3f} m",
    f"Std   = {np.std(all_errs):.3f} m",
    f"P68   = {np.percentile(all_errs,68):.3f} m",
    f"P95   = {np.percentile(all_errs,95):.3f} m",
    f"Max   = {np.max(all_errs):.3f} m",
    f"","Traj  Mean  Med  P95",
    *(f" {t}   {means[t]:.2f}  {medians[t]:.2f}  {np.percentile([float(r['err_m']) for r in by_traj[t]],95):.2f}"
      for t in order)]
for i, line in enumerate(lines):
    ax.text(0.05, 1.0 - i*0.055, line, fontfamily="monospace", fontsize=8,
            transform=ax.transAxes)

fig.suptitle("SPP Positioning Error vs RTK Ground Truth", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT_DIR/"spp_error_summary.png", dpi=DPI); plt.close()
print(f"Figures saved to {OUT_DIR}/")
