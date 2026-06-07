#!/usr/bin/env python3
"""Fig.5: HPL-Based Adaptive Candidate Search — good vs degraded geometry."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 300; C = {"blue":"#2471A3","red":"#C0392B","green":"#27AE60","gray":"#95A5A6","road":"#BDC3C7","sel":"#2471A3"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9.5,
    "axes.labelsize":8,"legend.fontsize":7,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

def make_roads():
    return [[(-3.5,2.5),(3.5,2.5)],[(-4,-0.5),(4,0.5)],[(-3.5,-2.5),(3.5,-2.5)],
            [(-2.5,4),(-2.5,-4)],[(0,4.5),(0,-4.5)],[(2.5,4),(2.5,-4)],
            [(-3,-2),(0,1)],[(1,-1),(3.5,1.5)]]

def hpl_plot(ax, center, hpl, roads, title, cov=None):
    ax.set_aspect("equal"); ax.set_xlim(-4.8,4.8); ax.set_ylim(-4.8,4.8)
    # Road network
    for r in roads: ax.plot(*zip(*r), color=C["road"], lw=1.2, zorder=1)
    # HPL disk
    ax.add_patch(plt.Circle(center, hpl, fc=C["blue"], alpha=0.07, ec=C["blue"], lw=2, ls="--", zorder=2))
    # Find intersecting
    cands = []
    for r in roads:
        (x1,y1),(x2,y2) = r; dx,dy = x2-x1, y2-y1
        t = max(0,min(1,((center[0]-x1)*dx+(center[1]-y1)*dy)/(dx*dx+dy*dy+1e-9)))
        px,py = x1+t*dx, y1+t*dy
        if (px-center[0])**2+(py-center[1])**2 <= hpl**2:
            ax.plot(*zip(*r), color=C["sel"], lw=2.5, zorder=3); cands.append((px,py))
    # Cov ellipse
    if cov is not None:
        from matplotlib.patches import Ellipse
        eigvals, eigvecs = np.linalg.eigh(cov)
        w, h = 2*np.sqrt(max(eigvals[0],1e-9)), 2*np.sqrt(max(eigvals[1],1e-9))
        a = float(np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0])))
        ax.add_patch(Ellipse(center,w,h,angle=a,fc=C["red"],alpha=0.12,ec=C["red"],lw=1.2))
    # GNSS
    ax.plot(*center,"o",color=C["red"],ms=8,mec="white",mew=1,zorder=5)
    # Candidates
    for px,py in cands: ax.plot(px,py,"s",color=C["green"],ms=7,mec="white",mew=0.8,zorder=4)
    # Info box
    n_cands = len(cands)
    ax.text(0.97,0.97,f"HPL = {hpl:.0f} m\n{n_cands} candidates", transform=ax.transAxes,
            ha="right",va="top",fontsize=8.5,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=C["gray"],alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, pad=8)

def main():
    roads = make_roads(); center = (0.5, 0.2)
    cov_good = np.array([[0.4,0.15],[0.15,0.3]]); cov_bad = np.array([[2.8,2.0],[2.0,2.5]])
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7.2,4.0))
    hpl_plot(ax1, center, 1.5, roads, "(a) Good Geometry — HPL = 1.5 m, Tight Search", cov_good)
    hpl_plot(ax2, center, 3.5, roads, "(b) Degraded Geometry — HPL = 3.5 m, Wide Search", cov_bad)
    # Shared legend
    handles = [mpatches.Patch(color=C["blue"],alpha=0.3), plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=C["red"],ms=7),
               plt.Line2D([0],[0],marker="s",color="w",markerfacecolor=C["green"],ms=6)]
    fig.legend(handles, ["HPL search disk", "GNSS $z_i$", "Candidates $x_{i,j}$"],
               loc="lower center", ncol=3, fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5,0.01))
    fig.suptitle("HPL-Based Adaptive Candidate Search: $\\mathrm{HPL}=K\\!\\cdot\\!\\sigma_{\\mathrm{major}}$",
                 fontsize=10.5, fontweight="bold", y=0.97)
    for fmt in ["svg","png"]: fig.savefig(FIGS/f"PLsearch.{fmt}",dpi=DPI,format=fmt)
    plt.close(fig); print("Saved PLsearch.svg/.png")

if __name__ == "__main__": main()
