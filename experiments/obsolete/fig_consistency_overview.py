#!/usr/bin/env python3
"""DO NOT RUN — DO NOT RE-USE. This script fabricates its own figure.

It was written to *illustrate* what a covariance-model consistency check looks
like, and it does so by construction rather than by measurement: it reads no
data file at all, draws `np.random.seed(42)` samples, adds a deliberate
`0.15 * chi2(3)` misspecification term, and prints the literal string
`"KS p = 0.12"` that no test ever produced (the real KS p-value of its own
sample against chi2(2) is 2.1e-39, because the misspecification is deliberate).

That was acceptable as a mock-up and became a defect when the figure it wrote,
`figs/consistency_overview.png`, was placed in the manuscript under a caption
describing the WLS covariance validation over 10,000 simulated epochs. Archived
on 2026-09-22, when the paper was re-pointed at the genuine measurement:

    experiments/output/1_covariance_validation/validation_sigma_10.png
    produced by experiments/scripts/exp1_covariance_validation.py
    (n = 10,000 epochs, KS 0.0084 against 0.1326 for an isotropic model,
     E[D^2] = 2.03 against the theoretical 2)

See the row for this file in README.md, and section "Covariance Model
Validation" in the manuscript's appendix.

--- original docstring ---
Fig.9: Covariance Model Validation at sigma_rho=10m — 4-panel statistical check."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
FIGS = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 300; C = {"blue":"#2471A3","red":"#C0392B","gray":"#95A5A6"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9.5,
    "axes.labelsize":8,"legend.fontsize":7,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

np.random.seed(42)
n = 2000
# Generate truly chi2(2) distributed data
true_chi2 = np.random.chisquare(2, n)
# Add some model-misspecification component
mismatch = 0.15 * np.random.chisquare(3, n)
observed = true_chi2 + mismatch
# Whitened errors for isotropic check
angles = np.random.uniform(0, 2*np.pi, n)
radius = np.sqrt(np.random.chisquare(2, n))
wx, wy = radius*np.cos(angles), radius*np.sin(angles)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

# ── (a) Mahalanobis distance histogram vs chi2(2) ──
ax = axes[0,0]
ax.hist(observed, bins=40, density=True, color=C["blue"], alpha=0.4, edgecolor="white", lw=0.3, label="Observed $d_M^2$")
x = np.linspace(0, max(observed), 200)
ax.plot(x, stats.chi2.pdf(x, 2), "-", color=C["red"], lw=2, label="$\\chi^2(2)$ theoretical")
ax.set_xlabel("Squared Mahalanobis distance $d_M^2$"); ax.set_ylabel("Density")
ax.set_title("(a) $d_M^2$ Distribution vs. $\\chi^2(2)$", pad=8, fontweight="bold")
ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
ax.text(0.95, 0.95, f"$n={n}$", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=C["gray"])

# ── (b) P-P plot ──
ax = axes[0,1]
sorted_obs = np.sort(observed)
theoretical = stats.chi2.ppf((np.arange(1,n+1)-0.5)/n, 2)
ax.plot(theoretical, sorted_obs, ".", color=C["blue"], ms=2, alpha=0.5)
lim = max(theoretical.max(), sorted_obs.max())
ax.plot([0,lim],[0,lim],"r-",lw=1.5)
ax.set_xlabel("Theoretical $\\chi^2(2)$ quantiles"); ax.set_ylabel("Observed $d_M^2$ quantiles")
ax.set_title("(b) P-P Plot", pad=8, fontweight="bold")
ax.text(0.95, 0.05, "KS $p$ = 0.12", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=C["gray"])

# ── (c) Whitened errors scatter ──
ax = axes[1,0]
ax.set_aspect("equal")
ax.plot(wx, wy, ".", color=C["blue"], ms=1.5, alpha=0.3)
t = np.linspace(0,2*np.pi,200)
for r, ls in [(1,"-"),(2,"--")]:
    ax.plot(r*np.cos(t), r*np.sin(t), color=C["red"], lw=1.5, ls=ls, alpha=0.7)
ax.set_xlabel("Whitened East"); ax.set_ylabel("Whitened North")
ax.set_xlim(-5,5); ax.set_ylim(-5,5)
ax.set_title("(c) Whitened Errors  $\\tilde{\\mathbf{e}} = \\mathbf{L}^{-1}\\mathbf{e}$", pad=8, fontweight="bold")
ax.text(0.95, 0.95, f"$n={n}$\n$1\\sigma$ / $2\\sigma$", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=C["gray"])

# ── (d) Radial error CDF vs Rayleigh ──
ax = axes[1,1]
r_err = np.sqrt(wx**2 + wy**2)
r_sorted = np.sort(r_err)
ecdf = np.arange(1,n+1)/n
cdf_rayleigh = stats.rayleigh.cdf(r_sorted, scale=1.0)
ax.plot(r_sorted, ecdf, "-", color=C["blue"], lw=2, label="Empirical CDF")
ax.plot(r_sorted, cdf_rayleigh, "--", color=C["red"], lw=2, label="Rayleigh(1)")
ax.set_xlabel("Radial error $\\|\\tilde{\\mathbf{e}}\\|$"); ax.set_ylabel("CDF")
ax.set_title("(d) Radial Error CDF vs. Rayleigh", pad=8, fontweight="bold")
ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

fig.suptitle("Covariance Model Consistency Validation  ($\\sigma_{\\rho}\\!=\\!10$ m, synthetic, $n\\!=\\!2000$ epochs)",
             fontsize=10.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=(0,0,1,0.96))

for fmt in ["svg","png"]: fig.savefig(FIGS/f"consistency_overview.{fmt}",dpi=DPI,format=fmt)
plt.close(fig); print("Saved consistency_overview.svg/.png")
