#!/usr/bin/env python3
"""Fig. 5: GNSS Covariance Formation — 3-panel: skyplot, pseudorange noise, error ellipse.
Output: SVG to docs/.../figs/covariance.svg
Avoids mathtext-incompatible LaTeX (bmatrix, etc.).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge, Circle
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_BLUE = "#2166ac"
COLOR_RED = "#b2182b"
COLOR_GRAY = "#777777"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "text.usetex": False,
})


def skyplot_panel(ax):
    """Skyplot as concentric circles (azimuth=angle, elevation=radius)."""
    rng = np.random.default_rng(99)
    n_sats = 8
    azimuths = rng.uniform(0, 360, n_sats)
    elevations = rng.uniform(15, 85, n_sats)

    # Draw concentric elevation rings
    theta = np.linspace(0, 2 * np.pi, 300)
    for el in [30, 60, 90]:
        r = el
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=COLOR_GRAY, linewidth=0.4, alpha=0.5)
        if el < 90:
            ax.text(r, 5, f"{el:d}$^\\circ$", fontsize=5.5, color=COLOR_GRAY, ha="center")

    # Cardinal direction lines
    for angle_deg in [0, 90, 180, 270]:
        rad = np.radians(angle_deg)
        ax.plot([0, 95 * np.cos(rad)], [0, 95 * np.sin(rad)], color=COLOR_GRAY, linewidth=0.3, alpha=0.4)

    # Direction labels
    ax.text(0, 98, "N", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(98, 0, "E", ha="left", va="center", fontsize=7)
    ax.text(0, -98, "S", ha="center", va="top", fontsize=7)
    ax.text(-98, 0, "W", ha="right", va="center", fontsize=7)

    # Plot satellites
    az_rad = np.radians(azimuths)
    xs = elevations * np.sin(az_rad)
    ys = elevations * np.cos(az_rad)
    ax.scatter(xs, ys, s=40, c=COLOR_BLUE, edgecolors="white", linewidth=0.6, zorder=4)

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(f"$s_{{{i+1}}}$", (x, y), textcoords="offset points",
                    xytext=(3, 3), fontsize=6, color=COLOR_BLUE)

    ax.set_aspect("equal")
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Satellite Geometry $\\mathbf{H}$", pad=8, fontsize=9.5)

    ax.text(0.5, -0.12, f"$m = {n_sats}$ satellites",
            ha="center", va="top", transform=ax.transAxes, fontsize=7, color=COLOR_GRAY)


def noise_matrix_panel(ax):
    """Diagonal weight matrix visualization."""
    ax.set_title("(b) Pseudorange Noise $\\mathbf{W}$", pad=8, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = 8
    sigma_rho = [1.2, 0.9, 1.5, 1.1, 0.8, 1.3, 1.0, 1.4]
    max_s = max(sigma_rho)

    for i in range(n):
        y = 0.85 - i * 0.1
        alpha_val = 0.3 + 0.7 * (sigma_rho[i] / max_s)
        rect = plt.Rectangle((0.2, y - 0.04), 0.2, 0.08, fill=True,
                              facecolor=COLOR_BLUE, alpha=alpha_val,
                              edgecolor=COLOR_BLUE, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(0.10, y, f"$\\sigma_{{\\rho,{i+1}}}$", ha="center", va="center", fontsize=6.5)
        ax.text(0.31, y, f"{sigma_rho[i]:.1f} m", ha="left", va="center", fontsize=6.5, color=COLOR_GRAY)

    ax.text(0.5, 0.06,
            "$\\mathbf{W} = \\mathrm{diag}(\\sigma_{\\rho,1}^{-2}, \\ldots, \\sigma_{\\rho,m}^{-2})$",
            ha="center", va="top", fontsize=7, transform=ax.transAxes)
    ax.text(0.5, -0.01,
            "$\\mathbf{\\Sigma}_{\\rho} = \\mathbf{W}^{-1}$",
            ha="center", va="top", fontsize=7, color=COLOR_GRAY, transform=ax.transAxes)


def error_ellipse_panel(ax):
    """Resulting error ellipse with annotations."""
    ax.set_title("(c) Position Covariance $\\mathbf{\\Sigma}_i$", pad=8, fontsize=9.5)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Road segment for context
    ax.plot([-3.5, 3.5], [1.0, -0.5], color=COLOR_GRAY, linewidth=1.5, alpha=0.5, label="Road segment")

    # GNSS point at center
    ax.plot(0, 0, "o", color=COLOR_RED, markersize=6, zorder=4, label="GNSS estimate $z_i$")

    # Covariance ellipse
    ell = Ellipse((0, 0), width=4.0, height=1.5, angle=25, fill=True,
                  facecolor=COLOR_BLUE, alpha=0.15, edgecolor=COLOR_BLUE, linewidth=1.8, zorder=2)
    ax.add_patch(ell)

    # Semi-major and semi-minor axes
    angle_rad = np.radians(25)
    maj_x = 2.0 * np.cos(angle_rad)
    maj_y = 2.0 * np.sin(angle_rad)
    min_x = -0.75 * np.sin(angle_rad)
    min_y = 0.75 * np.cos(angle_rad)

    ax.annotate("", xy=(maj_x, maj_y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=COLOR_BLUE, lw=1.5))
    ax.text(1.0, 1.1, "$\\sigma_{\\mathrm{major}}$",
            fontsize=7, color=COLOR_BLUE, fontweight="bold")
    ax.annotate("", xy=(min_x, min_y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=COLOR_BLUE, lw=1.0, linestyle="dashed"))
    ax.text(-1.5, -0.2, "$\\sigma_{\\mathrm{minor}}$",
            fontsize=7, color=COLOR_BLUE)

    # Covariance matrix annotation — simple rendered text
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRAY, alpha=0.85)
    ax.text(2.5, -2.2, "$\\mathbf{\\Sigma}_i$ = horizontal covariance", fontsize=7.5, ha="center",
            bbox=bbox_props)
    ax.text(2.5, -2.8, "[ $\\sigma_E^2$ , $\\sigma_{EN}$ ; $\\sigma_{EN}$ , $\\sigma_N^2$ ]",
            fontsize=7, ha="center")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.8)


def main():
    fig = plt.figure(figsize=(7.5, 3.2))

    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.7, 1.1], wspace=0.35,
                          left=0.04, right=0.98, top=0.88, bottom=0.08)

    ax_sky = fig.add_subplot(gs[0, 0])
    skyplot_panel(ax_sky)

    ax_noise = fig.add_subplot(gs[0, 1])
    noise_matrix_panel(ax_noise)

    ax_ellipse = fig.add_subplot(gs[0, 2])
    error_ellipse_panel(ax_ellipse)

    # Arrows between panels
    fig.text(0.347, 0.5, "->", ha="center", va="center", fontsize=16,
             color=COLOR_GRAY, fontweight="bold", fontfamily="monospace")
    fig.text(0.622, 0.5, "->", ha="center", va="center", fontsize=16,
             color=COLOR_GRAY, fontweight="bold", fontfamily="monospace")

    # Bottom equation
    fig.text(0.5, 0.01,
             r"$\mathbf{\Sigma}_x = (\mathbf{H}^T\mathbf{W}\mathbf{H})^{-1}$"
             "  $\longrightarrow$  extract $2\times 2$ horizontal block $\mathbf{\Sigma}_i$",
             ha="center", va="bottom", fontsize=8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor=COLOR_GRAY, alpha=0.7))

    fig.suptitle("GNSS Positioning Covariance: From Satellite Geometry to Error Ellipse",
                 fontsize=10, fontweight="bold", y=0.97)

    out = FIGS_DIR / "covariance.svg"
    fig.savefig(out, dpi=DPI, format="svg")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
