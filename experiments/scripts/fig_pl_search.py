#!/usr/bin/env python3
"""Fig. 6: HPL-Based Candidate Search — adaptive search radius illustration.
Output: SVG to docs/.../figs/PLsearch.svg
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_BLUE = "#2166ac"
COLOR_RED = "#b2182b"
COLOR_GREEN = "#4dac26"
COLOR_GRAY = "#777777"
COLOR_LIGHT_GRAY = "#cccccc"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "text.usetex": False,
})


def make_road_network():
    """Synthetic road network — simple grid + diagonals."""
    roads = [
        # Horizontal
        [(-3, 2), (3, 2)], [(-3.5, 0), (3.5, 0)], [(-3, -2), (3, -2)],
        [(-2, 4), (-2, -4)], [(0, 4.5), (0, -4.5)], [(2, 4), (2, -4)],
        # Diagonal
        [(-3, -1.5), (-1, 1)], [(1, -1), (3, 1.2)],
    ]
    return roads


def plot_candidates(ax, center, hpl, roads, label, show_ellipse=False, cov=None):
    """Plot a candidate search scenario."""
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)

    # Draw all roads
    for r in roads:
        xs, ys = zip(*r)
        ax.plot(xs, ys, color=COLOR_LIGHT_GRAY, linewidth=1.0, zorder=1)

    # Draw HPL circle
    circle = plt.Circle(center, hpl, fill=True, facecolor=COLOR_BLUE, alpha=0.08,
                        edgecolor=COLOR_BLUE, linewidth=1.8, linestyle="--", zorder=2)
    ax.add_patch(circle)

    # Identify intersecting roads
    intersect_roads = []
    for r in roads:
        (x1, y1), (x2, y2) = r
        # Simple check: closest point on segment to center
        dx, dy = x2 - x1, y2 - y1
        t = max(0, min(1, ((center[0] - x1) * dx + (center[1] - y1) * dy) / (dx * dx + dy * dy + 1e-9)))
        px, py = x1 + t * dx, y1 + t * dy
        if (px - center[0]) ** 2 + (py - center[1]) ** 2 <= hpl ** 2:
            intersect_roads.append(r)

    # Highlight intersecting roads
    for r in intersect_roads:
        xs, ys = zip(*r)
        ax.plot(xs, ys, color=COLOR_BLUE, linewidth=2.2, zorder=3)

    # GNSS point
    if show_ellipse and cov is not None:
        from matplotlib.patches import Ellipse
        eigvals, eigvecs = np.linalg.eigh(cov)
        w = 2 * np.sqrt(eigvals[0])
        h = 2 * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(center, w, h, angle=angle, fill=True,
                      facecolor=COLOR_RED, alpha=0.15, edgecolor=COLOR_RED, linewidth=1.2)
        ax.add_patch(ell)

    ax.plot(center[0], center[1], "o", color=COLOR_RED, markersize=7, zorder=5, markeredgecolor="white", markeredgewidth=0.8)

    # Candidate points at road intersections
    n_cands = len(intersect_roads)
    candidate_positions = []
    for r in intersect_roads:
        (x1, y1), (x2, y2) = r
        dx, dy = x2 - x1, y2 - y1
        t = max(0, min(1, ((center[0] - x1) * dx + (center[1] - y1) * dy) / (dx * dx + dy * dy + 1e-9)))
        px, py = x1 + t * dx, y1 + t * dy
        candidate_positions.append((px, py))

    for px, py in candidate_positions:
        ax.plot(px, py, "s", color=COLOR_GREEN, markersize=5, zorder=4,
                markeredgecolor="white", markeredgewidth=0.5)

    # Labels
    ax.text(0.98, 0.96, f"HPL = {hpl:.0f} m\n{n_cands} candidates",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRAY, alpha=0.85))

    ax.set_title(label, fontsize=9.5, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    leg = ax.legend(
        [mpatches.Patch(color=COLOR_BLUE, alpha=0.3), plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_RED, markersize=7), plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_GREEN, markersize=5)],
        ["HPL search region", "GNSS observation $z_i$", "Candidate points $x_{i,j}$"],
        loc="lower left", fontsize=6.5, framealpha=0.8, ncol=1)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8))

    roads = make_road_network()
    center = (0.0, 0.0)

    # Panel (a): Good geometry, small HPL
    cov_good = np.array([[0.3, 0.1], [0.1, 0.25]])
    plot_candidates(ax1, center, hpl=1.2, roads=roads,
                    label="(a) Good Geometry — Tight Search",
                    show_ellipse=True, cov=cov_good)

    # Panel (b): Degraded geometry, large HPL
    cov_bad = np.array([[2.5, 1.8], [1.8, 2.0]])
    plot_candidates(ax2, center, hpl=3.0, roads=roads,
                    label="(b) Degraded Geometry — Wide Search",
                    show_ellipse=True, cov=cov_bad)

    fig.suptitle("HPL-Based Adaptive Candidate Search: $\\mathrm{HPL} = K \\cdot \\sigma_{\\mathrm{major}}$",
                 fontsize=10, fontweight="bold", y=0.995)

    # Bottom annotation
    fig.text(0.5, 0.01,
             "$\\Omega_i = \\{x \\in \\mathbb{R}^2 \\mid \\|x - z_i\\|_2 \\leq \\mathrm{HPL}_i\\}$"
             "  —  $\\mathcal{C}_{\\mathrm{seg},i} = \\{s \\in \\mathcal{G} \\mid s \\cap \\Omega_i \\neq \\varnothing\\}$",
             ha="center", va="bottom", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor=COLOR_GRAY, alpha=0.7))

    out = FIGS_DIR / "PLsearch.svg"
    fig.savefig(out, dpi=DPI, format="svg")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
