#!/usr/bin/env python3
"""Fig. 1: Abstract graphical overview — Traditional HMM vs Proposed CMM framework.
Output: SVG to docs/.../figs/abstract_figure.svg
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FONT_FAMILY = "DejaVu Sans"
COLOR_TRAD_BG = "#e8e8e8"
COLOR_TRAD_BORDER = "#999999"
COLOR_TRAD_TEXT = "#555555"
COLOR_CMM_BG = "#d9e6f5"
COLOR_CMM_BORDER = "#2166ac"
COLOR_CMM_TEXT = "#2166ac"
COLOR_CMM_ACCENT = "#2166ac"
COLOR_RED = "#b2182b"
COLOR_GREEN = "#4dac26"

plt.rcParams.update({
    "font.family": FONT_FAMILY, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "text.usetex": False,
})


def draw_rounded_box(ax, xy, width, height, facecolor, edgecolor, text="", text_color="black", fontsize=8):
    """Draw a rounded box with centered multi-line text."""
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.5, zorder=2)
    ax.add_patch(box)
    lines = text.split("\n")
    cx, cy = xy[0] + width / 2, xy[1] + height / 2
    for i, line in enumerate(lines):
        y_offset = (len(lines) - 1) * 0.35 - i * 0.7
        ax.text(cx, cy + y_offset, line, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=3)
    return box


def draw_arrow(ax, start_xy, end_xy, color="gray", lw=1.5):
    """Draw a simple arrow between two points."""
    ax.annotate("", xy=end_xy, xytext=start_xy,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, connectionstyle="arc3,rad=0"))


def draw_error_ellipse(ax, center, width, height, angle_deg, color, alpha=0.3, linewidth=1.5):
    """Draw a filled error ellipse."""
    from matplotlib.patches import Ellipse
    ell = Ellipse(center, width, height, angle=angle_deg,
                  facecolor=color, edgecolor=color, alpha=alpha,
                  linewidth=linewidth, zorder=4)
    ax.add_patch(ell)
    return ell


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_aspect("equal")

    # ── Layout parameters ──
    box_w, box_h = 1.85, 1.15
    y_top = 5.0
    y_bot = 1.85
    x_starts = [0.3, 2.45, 4.60, 6.75]
    box_labels_trad = [
        "GNSS\n$z_i$ (position only)",
        "Candidate Search\nFixed Radius",
        "Emission\n$\\mathcal{N}(0,\\sigma^2 I)$",
        "Trustworthiness\nNone  ✗",
    ]
    box_labels_cmm = [
        "GNSS\n$z_i, \\mathbf{\\Sigma}_i, \\mathrm{PL}_i$",
        "Candidate Search\nHPL-Adaptive",
        "Emission\n$\\mathcal{N}(0,\\mathbf{\\Sigma}_i)$",
        "Trustworthiness\n$\\mathrm{TW}_t = 0.94$  ✓",
    ]

    # ── Section labels ──
    ax.text(5.0, 6.65, "(a) Traditional HMM Map Matching", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLOR_TRAD_TEXT)
    ax.text(5.0, 3.55, "(b) Proposed CMM Framework", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLOR_CMM_BORDER)

    # Horizontal separator
    ax.axhline(y=4.25, xmin=0.02, xmax=0.98, color="gray", linewidth=1.0,
               linestyle="--", alpha=0.5)

    # ── Draw traditional row ──
    for i, (x0, label) in enumerate(zip(x_starts, box_labels_trad)):
        color = COLOR_RED if i == 3 else COLOR_TRAD_BG
        txt_color = "white" if i == 3 else COLOR_TRAD_TEXT
        draw_rounded_box(ax, (x0, y_top), box_w, box_h, color, COLOR_TRAD_BORDER,
                         label, txt_color, fontsize=7.5)
        if i < 3:
            draw_arrow(ax, (x0 + box_w + 0.05, y_top + box_h / 2),
                       (x_starts[i + 1] - 0.05, y_top + box_h / 2),
                       COLOR_TRAD_BORDER)

    # ── Draw CMM row ──
    for i, (x0, label) in enumerate(zip(x_starts, box_labels_cmm)):
        color = COLOR_GREEN if i == 3 else COLOR_CMM_BG
        txt_color = "white" if i == 3 else COLOR_CMM_BORDER
        draw_rounded_box(ax, (x0, y_bot), box_w, box_h, color, COLOR_CMM_BORDER,
                         label, txt_color, fontsize=7.5)
        if i < 3:
            draw_arrow(ax, (x0 + box_w + 0.05, y_bot + box_h / 2),
                       (x_starts[i + 1] - 0.05, y_bot + box_h / 2),
                       COLOR_CMM_BORDER)

    # ── Visual comparison insets ──
    # Traditional: circle + point
    trad_center = (x_starts[0] + box_w / 2 + 0.3, y_top - 0.5)
    circle = plt.Circle(trad_center, 0.22, fill=False, edgecolor=COLOR_TRAD_BORDER,
                        linewidth=1.2, linestyle="--")
    ax.add_patch(circle)
    ax.plot(trad_center[0], trad_center[1], "o", color=COLOR_RED, markersize=4)
    ax.text(trad_center[0] + 0.35, trad_center[1], "Isotropic\n$\\sigma$ fixed",
            fontsize=6, color=COLOR_TRAD_TEXT, va="center")

    # CMM: ellipse + point
    cmm_center = (x_starts[0] + box_w / 2 + 0.3, y_bot - 0.5)
    draw_error_ellipse(ax, cmm_center, 0.55, 0.22, 30, COLOR_CMM_BORDER, alpha=0.2, linewidth=1.2)
    ax.plot(cmm_center[0], cmm_center[1], "o", color=COLOR_RED, markersize=4)
    ax.text(cmm_center[0] + 0.35, cmm_center[1], "Anisotropic\n$\\mathbf{\\Sigma}_i$ dynamic",
            fontsize=6, color=COLOR_CMM_BORDER, va="center")

    # ── Mini reliability diagram inset in CMM TW box ──
    inset_ax = ax.inset_axes([6.90, 2.10, 0.70, 0.55])
    inset_ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    bins_x = np.linspace(0.1, 0.9, 8)
    bins_y = bins_x + np.random.default_rng(42).normal(0, 0.04, 8)
    inset_ax.scatter(bins_x, bins_y, s=8, c=COLOR_CMM_BORDER, zorder=3)
    inset_ax.set_xlim(0, 1); inset_ax.set_ylim(0, 1)
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    inset_ax.set_aspect("equal")
    inset_ax.text(0.5, -0.5, "Calibrated ✓", ha="center", fontsize=5.5,
                  color=COLOR_GREEN, transform=inset_ax.transAxes)

    # ── Title ──
    fig.suptitle("From Isotropic Heuristics to GNSS-Consistent Probabilistic Map Matching",
                 fontsize=12, fontweight="bold", y=0.98)

    out_svg = FIGS_DIR / "abstract_figure.svg"
    out_png = FIGS_DIR / "abstract_figure.png"
    fig.savefig(out_svg, dpi=DPI, format="svg")
    fig.savefig(out_png, dpi=DPI, format="png")
    plt.close(fig)
    print(f"Saved {out_svg}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
