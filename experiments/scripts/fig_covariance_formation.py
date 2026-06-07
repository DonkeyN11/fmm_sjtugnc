#!/usr/bin/env python3
"""Fig.4: GNSS Covariance Formation — skyplot + noise model + error ellipse.
Compact 3-panel layout optimized for IEEE single-column width.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 300; C = {"blue": "#2471A3", "red": "#C0392B", "gray": "#7F8C8D",
                "light": "#D5D8DC", "bg": "#F8F9F9"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5,"axes.titlesize":9,
    "axes.labelsize":7.5,"legend.fontsize":6.5,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

# ── Panel (a): Polar skyplot ──
def skyplot(ax):
    rng = np.random.default_rng(42)
    az = rng.uniform(0, 360, 10); el = rng.uniform(15, 88, 10)
    ax.set_aspect("equal")
    for r_el in [30, 60, 90]:
        t = np.linspace(0, 2*np.pi, 200)
        ax.plot(r_el*np.cos(t), r_el*np.sin(t), color=C["gray"], lw=0.4, alpha=0.5)
    for a in [0, 90, 180, 270]:
        rad = np.radians(a)
        ax.plot([0,100*np.cos(rad)],[0,100*np.sin(rad)], color=C["gray"], lw=0.3, alpha=0.4)
    for label, x, y, va, ha in [("N",0,102,"bottom","center"),("E",102,0,"center","left"),
                                  ("S",0,-102,"top","center"),("W",-102,0,"center","right")]:
        ax.text(x, y, label, ha=ha, va=va, fontsize=7, fontweight="bold")
    az_r = np.radians(az); xs = el*np.sin(az_r); ys = el*np.cos(az_r)
    ax.scatter(xs, ys, s=45, c=C["blue"], edgecolors="white", lw=0.8, zorder=5)
    for i,(x,y) in enumerate(zip(xs,ys)):
        ax.annotate(f"$s_{{{i+1}}}$", (x,y), textcoords="offset points", xytext=(4,4), fontsize=6, color=C["blue"])
    ax.set_xlim(-108,108); ax.set_ylim(-108,108); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) Satellite Geometry $\\mathbf{H}$\n$m=10$ tracked satellites", pad=6)

# ── Panel (b): Pseudorange weight visualization ──
def noise_panel(ax):
    # Show noise values as a clean horizontal bar chart
    sigma = [0.6, 1.5, 0.3, 2.0, 0.8, 1.2, 1.1, 0.4, 1.8, 0.7]
    y = np.arange(len(sigma))[::-1]
    colors = [C["blue"] if s < 1.0 else C["red"] for s in sigma]
    ax.barh(y, sigma, height=0.6, color=colors, alpha=0.75, edgecolor="white", lw=0.5)
    for i, s in enumerate(sigma):
        ax.text(s+0.05, y[i], f"{s:.1f}m", va="center", fontsize=6, color=C["gray"])
    ax.set_xlim(0, 2.5); ax.set_yticks([])
    ax.set_xlabel("$\\sigma_{\\rho}$ (m)", fontsize=7)
    ax.set_title("(b) Pseudorange Noise $\\mathbf{W}$\n$\\mathbf{W}=\\mathrm{diag}(\\sigma_{\\rho,i}^{-2})$", pad=6)
    ax.axvline(1.0, color=C["gray"], lw=0.5, ls="--", alpha=0.5)
    ax.text(1.0, -0.8, "1 m", ha="center", fontsize=6, color=C["gray"])

# ── Panel (c): Error ellipse ──
def ellipse_panel(ax):
    ax.set_aspect("equal"); ax.set_xlim(-4,4); ax.set_ylim(-4,4)
    ax.plot([-3.5,3.5], [0.8,-0.6], color=C["gray"], lw=2, alpha=0.6, zorder=1, label="Road segment")
    ax.plot(0, 0, "o", color=C["red"], ms=7, zorder=4, markeredgecolor="white", mew=1)
    ell = Ellipse((0,0), width=4.2, height=1.5, angle=22, fill=True,
                  facecolor=C["blue"], alpha=0.12, edgecolor=C["blue"], lw=2, zorder=2)
    ax.add_patch(ell)
    # Axes
    ang = np.radians(22)
    ax.annotate("", xy=(2.1*np.cos(ang),2.1*np.sin(ang)), xytext=(0,0),
                arrowprops=dict(arrowstyle="<->",color=C["blue"],lw=1.5))
    ax.text(1.1,1.2,"$\\sigma_{\\mathrm{major}}$",fontsize=7,color=C["blue"],fontweight="bold")
    ax.annotate("", xy=(-0.75*np.sin(ang),0.75*np.cos(ang)), xytext=(0,0),
                arrowprops=dict(arrowstyle="<->",color=C["blue"],lw=1,ls="dashed"))
    ax.text(-1.8,0.1,"$\\sigma_{\\mathrm{minor}}$",fontsize=7,color=C["blue"])
    # Cov matrix box
    ax.text(0.5, 0.05, "$\\mathbf{\\Sigma}_i = [$ $\\sigma_E^2$, $\\sigma_{EN}$ ; $\\sigma_{EN}$, $\\sigma_N^2$ $]$",
            transform=ax.transAxes, fontsize=8, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C["gray"],alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(c) Error Ellipse $\\mathbf{\\Sigma}_i$", pad=6)
    ax.legend(loc="upper right",fontsize=6.5,framealpha=0.8)

# ── Main ──
def main():
    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.75, 0.95], wspace=0.32,
                          left=0.03, right=0.97, top=0.86, bottom=0.12)
    skyplot(fig.add_subplot(gs[0,0]))
    noise_panel(fig.add_subplot(gs[0,1]))
    ellipse_panel(fig.add_subplot(gs[0,2]))
    # Arrows between panels
    for xp in [0.36, 0.635]:
        fig.text(xp, 0.48, "$\\longrightarrow$", ha="center", va="center",
                 fontsize=14, color=C["gray"], fontweight="bold")
    # Bottom equation
    fig.text(0.5, 0.02, "$\\mathbf{\\Sigma}_x\\!=\\!(\\mathbf{H}^{\\mathsf{T}}\\mathbf{W}\\mathbf{H})^{-1}$  $\\rightarrow$  horizontal block $\\mathbf{\\Sigma}_i$  $\\rightarrow$  anisotropic emission model",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",fc=C["bg"],ec=C["gray"],alpha=0.8))
    fig.suptitle("GNSS Positioning Covariance: From Satellite Geometry to Error Ellipse", fontsize=10, fontweight="bold", y=0.97)
    for fmt in ["svg","png"]:
        fig.savefig(FIGS/f"covariance.{fmt}", dpi=DPI, format=fmt)
    plt.close(fig); print("Saved covariance.svg/.png")

if __name__ == "__main__": main()
