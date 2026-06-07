#!/usr/bin/env python3
"""Fig. 7: Mahalanobis vs Euclidean Projection — original space + whitened space.
Output: SVG to docs/.../figs/maha_vs_eu.svg
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_BLUE = "#2166ac"
COLOR_RED = "#e63946"
COLOR_YELLOW = "#f4a582"
COLOR_GREEN = "#4dac26"
COLOR_GRAY = "#666666"
COLOR_ROAD = "#555555"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9.5, "axes.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "text.usetex": False,
})


def ellipse_params(cov):
    """Return (width, height, angle_deg) for 1-sigma ellipse."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    w = 2 * np.sqrt(max(eigvals[0], 1e-9))
    h = 2 * np.sqrt(max(eigvals[1], 1e-9))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return w, h, angle


def add_ellipse(ax, center, cov, scale=1.0, **kwargs):
    w, h, angle = ellipse_params(cov)
    ell = Ellipse(center, w * scale, h * scale, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


def main():
    fig = plt.figure(figsize=(7.5, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.08, 1.0],
                          left=0.04, right=0.98, top=0.90, bottom=0.06)

    # ── Common geometry ──
    # Road segment defined by two points
    road_start = np.array([-2.5, 1.5])
    road_end = np.array([2.5, -0.5])
    road_vec = road_end - road_start
    road_len = np.linalg.norm(road_vec)
    road_dir = road_vec / road_len

    # GNSS observation
    z_i = np.array([0.0, 0.0])

    # Covariance matrix (elongated perpendicular to road)
    # Make it elongated roughly perpendicular to road direction
    perp = np.array([-road_dir[1], road_dir[0]])
    cov = (0.2 ** 2) * np.outer(road_dir, road_dir) + (1.8 ** 2) * np.outer(perp, perp)

    # Cholesky: L @ L.T = cov
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)

    # ── Euclidean projection ──
    # Orthogonal projection onto road segment
    t_euc = np.dot(z_i - road_start, road_dir) / road_len
    t_euc = max(0, min(1, t_euc))
    x_euc = road_start + t_euc * road_dir
    d_euc = np.linalg.norm(z_i - x_euc)

    # ── Mahalanobis projection ──
    # Minimize (z_i - x)^T cov^{-1} (z_i - x) subject to x on segment
    # In whitened space: minimize ||z̃_i - x̃|| where z̃ = L^{-1} z_i
    z_w = L_inv @ z_i
    r_w_start = L_inv @ road_start
    r_w_end = L_inv @ road_end
    r_w_vec = r_w_end - r_w_start
    r_w_len = np.linalg.norm(r_w_vec)

    t_maha = np.dot(z_w - r_w_start, r_w_vec) / (r_w_len ** 2 + 1e-9)
    t_maha = max(0, min(1, t_maha))
    x_w = r_w_start + t_maha * r_w_vec
    x_maha = L @ x_w
    d_maha = np.linalg.norm(z_w - x_w)  # = Mahalanobis distance

    # ═════════════════════════════════════════════════
    # PANEL (a): Original Space
    # ═════════════════════════════════════════════════
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.set_aspect("equal")
    ax_orig.set_xlim(-3.5, 3.5)
    ax_orig.set_ylim(-3.0, 3.0)
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])

    # Road segment
    ax_orig.plot([road_start[0], road_end[0]], [road_start[1], road_end[1]],
                 color=COLOR_ROAD, linewidth=3.5, zorder=2, label="Road segment")
    # Extend slightly for visual clarity
    ext_start = road_start - 0.3 * road_dir
    ext_end = road_end + 0.3 * road_dir
    ax_orig.plot([ext_start[0], ext_end[0]], [ext_start[1], ext_end[1]],
                 color=COLOR_ROAD, linewidth=3.5, alpha=0.3, zorder=1)

    # Covariance ellipses (1σ, 2σ, 3σ)
    for s, a in [(3, 0.06), (2, 0.10), (1, 0.18)]:
        add_ellipse(ax_orig, z_i, cov, scale=s, fill=True,
                    facecolor=COLOR_BLUE, alpha=a, edgecolor="none")

    # GNSS observation
    ax_orig.plot(z_i[0], z_i[1], "o", color=COLOR_RED, markersize=8,
                 markeredgecolor="white", markeredgewidth=1, zorder=5)

    # Euclidean projection
    ax_orig.plot(x_euc[0], x_euc[1], "s", color=COLOR_YELLOW, markersize=9,
                 markeredgecolor="#8B6914", markeredgewidth=1, zorder=5)
    ax_orig.plot([z_i[0], x_euc[0]], [z_i[1], x_euc[1]], "--",
                 color=COLOR_YELLOW, linewidth=1.5, zorder=4)

    # Mahalanobis projection
    ax_orig.plot(x_maha[0], x_maha[1], "D", color=COLOR_GREEN, markersize=8,
                 markeredgecolor="darkgreen", markeredgewidth=1, zorder=5)
    ax_orig.plot([z_i[0], x_maha[0]], [z_i[1], x_maha[1]], "-",
                 color=COLOR_GREEN, linewidth=1.5, zorder=4)

    # Distance annotations
    mid_euc = (z_i + x_euc) / 2
    mid_maha = (z_i + x_maha) / 2
    ax_orig.annotate(f"$d_E = {d_euc:.1f}$", xy=mid_euc,
                     xytext=(mid_euc[0] - 1.0, mid_euc[1] + 0.3),
                     fontsize=7.5, color="#8B6914", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#8B6914", lw=1.0))
    ax_orig.annotate(f"$d_M = {d_maha:.1f}$", xy=mid_maha,
                     xytext=(mid_maha[0] + 0.3, mid_maha[1] - 0.4),
                     fontsize=7.5, color="darkgreen", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.0))

    ax_orig.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax_orig.set_title("(a) Original Coordinate Space", fontsize=9.5, pad=8)

    # Annotation box
    ax_orig.text(0.02, 0.98,
                 "Euclidean: geometrically closest\nMahalanobis: statistically most plausible",
                 transform=ax_orig.transAxes, fontsize=7, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRAY, alpha=0.85))

    # ═════════════════════════════════════════════════
    # PANEL (b): Whitened Space
    # ═════════════════════════════════════════════════
    ax_white = fig.add_subplot(gs[0, 2])
    ax_white.set_aspect("equal")
    ax_white.set_xlim(-3.5, 3.5)
    ax_white.set_ylim(-3.0, 3.0)
    ax_white.set_xticks([])
    ax_white.set_yticks([])

    # Whitened road segment (may be curved/transformed)
    # For a straight line under linear transform: still straight
    ax_white.plot([r_w_start[0], r_w_end[0]], [r_w_start[1], r_w_end[1]],
                  color=COLOR_ROAD, linewidth=3.5, zorder=2)
    ext_w_start = r_w_start - 0.3 * r_w_vec / r_w_len
    ext_w_end = r_w_end + 0.3 * r_w_vec / r_w_len
    ax_white.plot([ext_w_start[0], ext_w_end[0]], [ext_w_start[1], ext_w_end[1]],
                  color=COLOR_ROAD, linewidth=3.5, alpha=0.3, zorder=1)

    # Unit circles (whitened space: covariance = I)
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, alpha, ls in [(3, 0.06, "dotted"), (2, 0.10, "dashed"), (1, 0.18, "solid")]:
        ax_white.fill(z_w[0] + r * np.cos(theta), z_w[1] + r * np.sin(theta),
                      facecolor=COLOR_BLUE, alpha=alpha, edgecolor=COLOR_BLUE, linewidth=0.8, linestyle=ls)

    # Whitened GNSS
    ax_white.plot(z_w[0], z_w[1], "o", color=COLOR_RED, markersize=8,
                  markeredgecolor="white", markeredgewidth=1, zorder=5)

    # Whitened projection (Euclidean in whitened = Mahalanobis in original)
    ax_white.plot(x_w[0], x_w[1], "D", color=COLOR_GREEN, markersize=8,
                  markeredgecolor="darkgreen", markeredgewidth=1, zorder=5)
    ax_white.plot([z_w[0], x_w[0]], [z_w[1], x_w[1]], "-",
                  color=COLOR_GREEN, linewidth=1.5, zorder=4)

    ax_white.set_title("(b) Whitened Space  $(\\mathbf{L}^{-1})$", fontsize=9.5, pad=8)

    # Annotation
    ax_white.text(0.02, 0.98,
                  "$\\tilde{z} = \\mathbf{L}^{-1}z$\n$\\tilde{x} = \\mathbf{L}^{-1}x$\n"
                  "$d_M^2 = \\|\\tilde{z} - \\tilde{x}\\|^2$",
                  transform=ax_white.transAxes, fontsize=7, va="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRAY, alpha=0.85))

    # ═════════════════════════════════════════════════
    # MIDDLE PANEL: Transformation arrow
    # ═════════════════════════════════════════════════
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_mid.axis("off")
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                    arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=2.5, connectionstyle="arc3,rad=0"))
    ax_mid.text(0.5, 0.62, "$\\mathbf{L}^{-1}$", ha="center", fontsize=11,
                fontweight="bold", color=COLOR_BLUE)
    ax_mid.text(0.5, 0.38, "whitening", ha="center", fontsize=7.5, color=COLOR_GRAY)
    ax_mid.text(0.5, 0.20,
                "$\\mathbf{\\Sigma} = \\mathbf{L}\\mathbf{L}^{\\mathsf{T}}$\n"
                "$\\mathbf{L}^{-1}\\mathbf{\\Sigma}(\\mathbf{L}^{-1})^{\\mathsf{T}} = \\mathbf{I}$",
                ha="center", fontsize=7, color=COLOR_GRAY)

    # Equation at bottom
    fig.text(0.5, 0.01,
             "$\\min_{x \\in s} (z_i - x)^{T}\\mathbf{\\Sigma}_i^{-1}(z_i - x)"
             " = \\min_{\\tilde{x} \\in \\tilde{s}} \\|\\tilde{z}_i - \\tilde{x}\\|^2$"
             "  —  Euclidean projection in whitened space",
             ha="center", va="bottom", fontsize=8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor=COLOR_GRAY, alpha=0.7))

    fig.suptitle("Mahalanobis vs. Euclidean Projection Under Anisotropic GNSS Uncertainty",
                 fontsize=10, fontweight="bold", y=0.98)

    out = FIGS_DIR / "maha_vs_eu.svg"
    fig.savefig(out, dpi=DPI, format="svg")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
