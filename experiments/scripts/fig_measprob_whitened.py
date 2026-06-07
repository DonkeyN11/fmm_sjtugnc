#!/usr/bin/env python3
"""Fig. 8: Emission Probability — bivariate Gaussian in original & whitened space.
Output: SVG to docs/.../figs/measprob.svg
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_BLUE = "#2166ac"
COLOR_RED = "#e63946"
COLOR_GREEN = "#4dac26"
COLOR_GRAY = "#666666"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9.5, "axes.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "text.usetex": False,
})


def main():
    fig = plt.figure(figsize=(7.5, 4.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.7, 0.3],
                          width_ratios=[1.0, 1.0],
                          left=0.05, right=0.97, top=0.92, bottom=0.06,
                          hspace=0.35, wspace=0.30)

    # ── Setup synthetic geometry ──
    z_i = np.array([0.0, 0.0])
    road_start = np.array([-2.5, 0.8])
    road_end = np.array([2.5, -0.4])
    road_vec = road_end - road_start
    road_dir = road_vec / np.linalg.norm(road_vec)
    perp = np.array([-road_dir[1], road_dir[0]])

    # Covariance: elongated cross-track
    cov = (0.3 ** 2) * np.outer(road_dir, road_dir) + (1.5 ** 2) * np.outer(perp, perp)
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)

    # Candidate point via Mahalanobis projection
    z_w = L_inv @ z_i
    rs_w = L_inv @ road_start
    re_w = L_inv @ road_end
    rv_w = re_w - rs_w
    rl_w = np.linalg.norm(rv_w)
    t = max(0, min(1, np.dot(z_w - rs_w, rv_w) / (rl_w ** 2 + 1e-9)))
    x_w = rs_w + t * rv_w
    x_ij = L @ x_w
    d_vec = z_i - x_ij
    d_m = np.sqrt((d_vec @ L_inv.T) @ (L_inv @ d_vec))

    # ═══════════════════════════════════════════
    # TOP-LEFT: Original Space
    # ═══════════════════════════════════════════
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.set_aspect("equal")
    ax_orig.set_xlim(-3.5, 3.5)
    ax_orig.set_ylim(-2.5, 2.5)
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    ax_orig.set_title("(a) Original Coordinate Frame", fontsize=9.5)

    # Road
    ax_orig.plot([road_start[0], road_end[0]], [road_start[1], road_end[1]],
                 color=COLOR_GRAY, linewidth=3, zorder=2)

    # Covariance ellipses
    for s, a in [(3, 0.05), (2, 0.10), (1, 0.18)]:
        eigvals, eigvecs = np.linalg.eigh(cov)
        w, h = 2 * s * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(z_i, w, h, angle=angle, fill=True,
                      facecolor=COLOR_BLUE, alpha=a, edgecolor="none")
        ax_orig.add_patch(ell)

    # GNSS observation
    ax_orig.plot(z_i[0], z_i[1], "o", color=COLOR_RED, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax_orig.annotate("$z_i$", (z_i[0], z_i[1]), textcoords="offset points",
                     xytext=(5, -12), fontsize=9, color=COLOR_RED, fontweight="bold")

    # Candidate
    ax_orig.plot(x_ij[0], x_ij[1], "D", color=COLOR_GREEN, markersize=8,
                 markeredgecolor="darkgreen", markeredgewidth=1, zorder=5)
    ax_orig.annotate("$x_{i,j}$", (x_ij[0], x_ij[1]), textcoords="offset points",
                     xytext=(5, 5), fontsize=9, color="darkgreen", fontweight="bold")

    # Residual vector
    ax_orig.annotate("", xy=x_ij, xytext=z_i,
                     arrowprops=dict(arrowstyle="<->", color=COLOR_RED, lw=1.5, linestyle="dashed"))
    mid_d = (z_i + x_ij) / 2
    ax_orig.annotate("$\\mathbf{d}_{i,j}$", mid_d, textcoords="offset points",
                     xytext=(-15, 8), fontsize=8, color=COLOR_RED)

    # Formula box
    ax_orig.text(0.02, 0.02,
                 "$p(z_i \\mid x_{i,j}) = \\frac{1}{2\\pi\\sqrt{|\\mathbf{\\Sigma}_i|}}"
                 "\\exp\\!\\left[-\\frac{1}{2}\\mathbf{d}_{i,j}^{\\mathsf{T}}\\mathbf{\\Sigma}_i^{-1}\\mathbf{d}_{i,j}\\right]$",
                 transform=ax_orig.transAxes, fontsize=7.5, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRAY, alpha=0.85))

    # ═══════════════════════════════════════════
    # TOP-RIGHT: Whitened Space
    # ═══════════════════════════════════════════
    ax_white = fig.add_subplot(gs[0, 1])
    ax_white.set_aspect("equal")
    ax_white.set_xlim(-3.5, 3.5)
    ax_white.set_ylim(-2.5, 2.5)
    ax_white.set_xticks([])
    ax_white.set_yticks([])
    ax_white.set_title("(b) Whitened Frame  $(\\tilde{z} = \\mathbf{L}^{-1}z)$", fontsize=9.5)

    # Whitened road
    ax_white.plot([rs_w[0], re_w[0]], [rs_w[1], re_w[1]],
                  color=COLOR_GRAY, linewidth=3, zorder=2)

    # Unit circles
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, a, ls in [(3, 0.05, "dotted"), (2, 0.10, "dashed"), (1, 0.18, "solid")]:
        ax_white.fill(z_w[0] + r * np.cos(theta), z_w[1] + r * np.sin(theta),
                      facecolor=COLOR_BLUE, alpha=a, edgecolor=COLOR_BLUE, linewidth=0.8, linestyle=ls)

    # Whitened GNSS
    ax_white.plot(z_w[0], z_w[1], "o", color=COLOR_RED, markersize=8,
                  markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax_white.annotate("$\\tilde{z}_i$", (z_w[0], z_w[1]), textcoords="offset points",
                      xytext=(5, -12), fontsize=9, color=COLOR_RED, fontweight="bold")

    # Whitened candidate
    ax_white.plot(x_w[0], x_w[1], "D", color=COLOR_GREEN, markersize=8,
                  markeredgecolor="darkgreen", markeredgewidth=1, zorder=5)
    ax_white.annotate("$\\tilde{x}_{i,j}$", (x_w[0], x_w[1]), textcoords="offset points",
                      xytext=(5, 5), fontsize=9, color="darkgreen", fontweight="bold")

    # Euclidean distance in whitened = Mahalanobis
    ax_white.annotate("", xy=x_w, xytext=z_w,
                      arrowprops=dict(arrowstyle="<->", color=COLOR_GREEN, lw=1.5))
    mid_w = (z_w + x_w) / 2
    ax_white.annotate(f"$d_M = {d_m:.2f}$", mid_w, textcoords="offset points",
                      xytext=(-18, 8), fontsize=8, color="darkgreen", fontweight="bold")

    # Formula box
    ax_white.text(0.02, 0.02,
                 "$p(\\tilde{z}_i \\mid \\tilde{x}_{i,j}) = \\frac{1}{2\\pi}"
                 "\\exp\\!\\left[-\\frac{1}{2}\\|\\tilde{z}_i - \\tilde{x}_{i,j}\\|^2\\right]$\n"
                 "$\\mathbf{L}^{-1}\\mathbf{\\Sigma}_i(\\mathbf{L}^{-1})^{\\mathsf{T}} = \\mathbf{I}$",
                 transform=ax_white.transAxes, fontsize=7, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRAY, alpha=0.85))

    # ═══════════════════════════════════════════
    # BOTTOM: Explanation text
    # ═══════════════════════════════════════════
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    explanation = (
        "The Mahalanobis distance in the original anisotropic space (left) is "
        "mathematically equivalent to the Euclidean distance in the whitened space (right).\n"
        "Whitening via Cholesky decomposition $\\mathbf{\\Sigma}_i = \\mathbf{L}\\mathbf{L}^{\\mathsf{T}}$ "
        "transforms the error ellipse into a unit circle, making the emission probability "
        "a standard isotropic Gaussian in the transformed coordinates.\n"
        "This equivalence ensures the emission model is statistically consistent with the "
        "GNSS stochastic error model at every epoch."
    )
    ax_text.text(0.5, 0.55, explanation, ha="center", va="center", fontsize=8,
                 transform=ax_text.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor=COLOR_GRAY, alpha=0.7))

    fig.suptitle("Emission Probability: Anisotropic Gaussian via Whitening Transformation",
                 fontsize=10, fontweight="bold", y=0.98)

    out = FIGS_DIR / "measprob.svg"
    fig.savefig(out, dpi=DPI, format="svg")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
