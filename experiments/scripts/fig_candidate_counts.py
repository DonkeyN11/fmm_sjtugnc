#!/usr/bin/env python3
"""Candidate count comparison: TMM vs HMM."""
import csv, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
SIM_DATA = PROJECT / "experiments/data"
REAL_DATA = PROJECT / "data/real_vehicle/processed"
OUT_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
DPI, C_TMM, C_HMM = 300, "#2166ac", "#b2182b"
plt.rcParams.update({"font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
    "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

def count_candidates(path):
    counts = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            cand = row.get("candidates","")
            if cand: counts.append(cand.count("),(") + 1)
    return np.array(counts) if counts else np.array([0])

# Simulation figure
SIGMA_DIRS = sorted(SIM_DATA.glob("sigma_*/no_occlusion/no_fault"),
    key=lambda d: int(d.parent.parent.name.replace("sigma_","")))
labels, means = [], []
for d in SIGMA_DIRS:
    cf = d / "cmm_result.csv"
    if not cf.exists(): continue
    sv = int(d.parent.parent.name.replace("sigma_",""))
    labels.append(f"$\sigma$={sv}")
    c = count_candidates(cf)
    means.append(float(np.mean(c)) if len(c) else 0)
    print(f"  sigma={sv}: {means[-1]:.1f}")

fig, ax = plt.subplots(figsize=(7,4))
x = np.arange(len(labels))
ax.bar(x-0.175, means, 0.35, color=C_TMM, label="TMM (HPL-adaptive)", edgecolor="white",lw=0.5)
ax.bar(x+0.175, [16]*len(labels), 0.35, color=C_HMM, label="HMM (fixed r=0.03, k=16)", edgecolor="white",lw=0.5)
for i,v in enumerate(means): ax.text(i-0.175, v+0.3, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold", color=C_TMM)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Mean candidates per epoch")
ax.set_title("Candidate Count: TMM (HPL-adaptive) vs HMM (fixed-radius)")
ax.legend(); ax.grid(alpha=0.3,axis="y"); ax.set_ylim(0,18)
fig.tight_layout(); out = OUT_DIR / "candidate_count_simulation.png"
fig.savefig(out, dpi=DPI); plt.close(fig); print(f"Saved {out}")

# Real-vehicle figure
cmm_file = REAL_DATA / "cmm_result.csv"
if cmm_file.exists():
    rc = count_candidates(cmm_file)
    tavg = float(np.mean(rc))
    print(f"Real: {len(rc)} epochs, mu={tavg:.1f}")
    fig, ax = plt.subplots(figsize=(5.5,4))
    ax.bar(0, tavg, 0.45, color=C_TMM, label="TMM (HPL-adaptive)", edgecolor="white",lw=0.5)
    ax.bar(1, 16, 0.45, color=C_HMM, label="HMM (fixed r=0.03, k=16)", edgecolor="white",lw=0.5)
    ax.text(0, tavg+0.3, f"{tavg:.1f}", ha="center", fontsize=10, fontweight="bold", color=C_TMM)
    ax.text(1, 16.3, "16", ha="center", fontsize=10, fontweight="bold", color=C_HMM)
    ax.set_xticks([0,1]); ax.set_xticklabels(["TMM","HMM"])
    ax.set_ylabel("Mean candidates per epoch")
    ax.set_title("Real-Vehicle Candidate Count (16,155 epochs)")
    ax.legend(); ax.grid(alpha=0.3,axis="y"); ax.set_ylim(0,18)
    fig.tight_layout(); out = OUT_DIR / "candidate_count_real.png"
    fig.savefig(out, dpi=DPI); plt.close(fig); print(f"Saved {out}")
