#!/usr/bin/env python3
"""Fig.6: Mahalanobis vs Euclidean Projection — original + whitened space."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 300; C = {"blue":"#2471A3","red":"#C0392B","green":"#27AE60","yellow":"#F39C12","road":"#566573","gray":"#95A5A6"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9.5,
    "axes.labelsize":8,"legend.fontsize":7,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

def main():
    road_s = np.array([-2.8, 1.8]); road_e = np.array([2.8, -0.8])
    road_v = road_e - road_s; road_l = np.linalg.norm(road_v); road_d = road_v/road_l
    perp = np.array([-road_d[1], road_d[0]])
    z_i = np.array([0.0, 0.0])
    cov = 0.15**2*np.outer(road_d,road_d) + 2.0**2*np.outer(perp,perp)
    L = np.linalg.cholesky(cov); L_inv = np.linalg.inv(L)

    # Euclidean projection
    t_e = max(0, min(1, np.dot(z_i-road_s, road_d)/road_l))
    x_euc = road_s + t_e*road_d

    # Mahalanobis projection (Euclidean in whitened)
    z_w = L_inv @ z_i; r_ws = L_inv @ road_s; r_we = L_inv @ road_e
    r_wv = r_we - r_ws; r_wl = np.linalg.norm(r_wv)
    t_m = max(0, min(1, np.dot(z_w-r_ws, r_wv)/(r_wl**2+1e-9)))
    x_w = r_ws + t_m*r_wv; x_maha = L @ x_w

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.05, right=0.97, top=0.88, bottom=0.08)

    # ═══ PANEL (a): Original Space ═══
    ax = fig.add_subplot(gs[0,0]); ax.set_aspect("equal")
    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.2, 3.2)
    # Road
    ax.plot([road_s[0],road_e[0]],[road_s[1],road_e[1]],color=C["road"],lw=4,zorder=2,label="Road segment")
    # Cov ellipses 1/2/3-sigma
    for s, a in [(3,0.04),(2,0.08),(1,0.14)]:
        eigvals, eigvecs = np.linalg.eigh(cov)
        w = 2*np.sqrt(max(eigvals[0],1e-9))*s; h = 2*np.sqrt(max(eigvals[1],1e-9))*s
        ang = float(np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0])))
        ax.add_patch(Ellipse(z_i,w,h,angle=ang,fc=C["blue"],alpha=a,ec="none"))
    # GNSS
    ax.plot(*z_i,"o",color=C["red"],ms=9,mec="white",mew=1.2,zorder=5)
    ax.annotate("$z_i$", z_i, textcoords="offset points", xytext=(8,-12), fontsize=9, color=C["red"], fontweight="bold")
    # Euclidean projection
    ax.plot(*x_euc,"s",color=C["yellow"],ms=9,mec="#B7950B",mew=1,zorder=5)
    ax.plot([z_i[0],x_euc[0]],[z_i[1],x_euc[1]],"--",color=C["yellow"],lw=2,zorder=4)
    ax.annotate("$x^{\\mathrm{Euc}}$", x_euc, textcoords="offset points", xytext=(6,10), fontsize=8, color="#B7950B")
    # Mahalanobis projection
    ax.plot(*x_maha,"D",color=C["green"],ms=8,mec="#1E8449",mew=1,zorder=5)
    ax.plot([z_i[0],x_maha[0]],[z_i[1],x_maha[1]],"-",color=C["green"],lw=2,zorder=4)
    ax.annotate("$x^{\\mathrm{Maha}}$", x_maha, textcoords="offset points", xytext=(-10,8), fontsize=8, color="#1E8449")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) Original Coordinate Space\nanisotropic uncertainty ellipse", pad=8)

    # ═══ PANEL (b): Whitened Space ═══
    ax = fig.add_subplot(gs[0,1]); ax.set_aspect("equal")
    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.2, 3.2)
    # Road in whitened
    ax.plot([r_ws[0],r_we[0]],[r_ws[1],r_we[1]],color=C["road"],lw=4,zorder=2,label="Road (whitened)")
    # Unit circles
    t = np.linspace(0,2*np.pi,200)
    for r,a in [(3,0.04),(2,0.08),(1,0.14)]:
        ax.fill(z_w[0]+r*np.cos(t),z_w[1]+r*np.sin(t),fc=C["blue"],alpha=a,ec=C["blue"],lw=0.6)
    # Whitened GNSS
    ax.plot(*z_w,"o",color=C["red"],ms=9,mec="white",mew=1.2,zorder=5)
    ax.annotate("$\\tilde{z}_i$", z_w, textcoords="offset points", xytext=(8,-12), fontsize=9, color=C["red"],fontweight="bold")
    # Whitened projection (Euclidean = Mahalanobis equivalent)
    ax.plot(*x_w,"D",color=C["green"],ms=8,mec="#1E8449",mew=1,zorder=5)
    ax.plot([z_w[0],x_w[0]],[z_w[1],x_w[1]],"-",color=C["green"],lw=2,zorder=4)
    ax.annotate("$\\tilde{x}^{\\mathrm{Euc}}$", x_w, textcoords="offset points", xytext=(6,10), fontsize=8, color="#1E8449")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Whitened Space  $(\\mathbf{L}^{-1})$\nisotropic unit-circle uncertainty  $\\rightarrow$ Euclidean projection", pad=8)

    # Center arrow + Cholesky annotation
    fig.text(0.5, 0.52, "$\\mathbf{L}^{-1}$", ha="center", va="center", fontsize=16, fontweight="bold", color=C["blue"],
             bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C["gray"],alpha=0.9))
    fig.text(0.5, 0.43, "whitening\ntransform", ha="center", va="center", fontsize=7, color=C["gray"])

    # Bottom equation
    fig.text(0.5, 0.02, "$\\min_{x\\in s}(z_i-x)^{\\mathsf{T}}\\mathbf{\\Sigma}_i^{-1}(z_i-x) = \\min_{\\tilde{x}\\in\\tilde{s}}\\|\\tilde{z}_i-\\tilde{x}\\|^2$  —  Mahalanobis in original = Euclidean in whitened",
             ha="center",va="bottom",fontsize=8.5,fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",fc="#F8F9F9",ec=C["gray"],alpha=0.8))
    fig.suptitle("Mahalanobis vs. Euclidean Projection Under Anisotropic GNSS Uncertainty", fontsize=10.5, fontweight="bold", y=0.97)

    for fmt in ["svg","png"]: fig.savefig(FIGS/f"maha_vs_eu.{fmt}",dpi=DPI,format=fmt)
    plt.close(fig); print("Saved maha_vs_eu.svg/.png")

if __name__ == "__main__": main()
