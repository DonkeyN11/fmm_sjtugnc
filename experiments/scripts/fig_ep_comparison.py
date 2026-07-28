#!/usr/bin/env python3
"""EP comparison: CaMM (Mahalanobis) vs HMM (isotropic) across all sigma levels."""
import csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data/simulation"
OUT = ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs/ep_comparison.png"
DPI = 300
plt.rcParams.update({"font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "legend.fontsize":8,"figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

sigmas = sorted([d.name for d in SIM.glob("sigma_*") if d.is_dir() and "mismatch" not in d.name],
                key=lambda s: int(s.replace("sigma_","")))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, s in enumerate(sigmas):
    cmm_f = SIM / s / "no_occlusion/no_fault/cmm_result.csv"
    fmm_f = SIM / s / "no_occlusion/no_fault/fmm_result.csv"
    if not cmm_f.exists() or not fmm_f.exists():
        continue

    eps_c, eps_f = [], []
    with open(cmm_f) as f:
        for row in csv.DictReader(f, delimiter=";"):
            try: eps_c.append(float(row["ep"]))
            except: pass
    with open(fmm_f) as f:
        for row in csv.DictReader(f, delimiter=";"):
            try: eps_f.append(float(row["ep"]))
            except: pass

    eps_c, eps_f = np.array(eps_c), np.array(eps_f)

    ax = axes[idx]
    bins = np.linspace(0, 1, 51)
    ax.hist(eps_c, bins=bins, alpha=0.6, color="#2166ac", label=f"CaMM (μ={np.mean(eps_c):.3f})", density=True)
    ax.hist(eps_f, bins=bins, alpha=0.6, color="#b2182b", label=f"HMM (μ={np.mean(eps_f):.3f})", density=True)
    ax.set_title(f"σ={s.replace('sigma_','')} m")
    ax.set_xlabel("EP"); ax.set_ylabel("Density")
    ax.legend(fontsize=6)
    ax.set_xlim(0, 1)

# Hide empty subplot
if len(sigmas) < 8:
    for idx in range(len(sigmas), 8):
        axes[idx].set_visible(False)

fig.suptitle("Emission Probability Distribution: CaMM (Mahalanobis) vs HMM (Isotropic)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=DPI)
plt.close(fig)
print(f"Saved {OUT}")
