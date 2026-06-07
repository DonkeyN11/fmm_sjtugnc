#!/usr/bin/env python3
"""Fig.7: Emission Probability — anisotropic original vs whitened isotropic frame."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 300; C = {"blue":"#2471A3","red":"#C0392B","green":"#27AE60","road":"#566573","gray":"#95A5A6"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9.5,
    "axes.labelsize":8,"legend.fontsize":7,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

def main():
    road_s = np.array([-3.0, 1.5]); road_e = np.array([3.0, -0.5])
    z_i = np.array([0.0, 0.0])
    perp = np.array([0.316, 0.949])
    cov = 0.1**2*np.outer(np.array([0.949,-0.316]),np.array([0.949,-0.316])) + 1.8**2*np.outer(perp,perp)
    L = np.linalg.cholesky(cov); L_inv = np.linalg.inv(L)

    z_w = L_inv @ z_i; r_ws = L_inv @ road_s; r_we = L_inv @ road_e
    r_wv = r_we - r_ws; r_wl = np.linalg.norm(r_wv)
    t = max(0,min(1,np.dot(z_w-r_ws,r_wv)/(r_wl**2+1e-9)))
    x_w = r_ws + t*r_wv; x = L @ x_w
    ts = np.linspace(0,1,200)
    road_pts = np.array([road_s + ti*(road_e-road_s) for ti in ts])
    w_pts = np.array([r_ws + ti*(r_we-r_ws) for ti in ts])

    # Compute EP along road in both spaces
    eps, eps_w = [], []
    for pt, wp in zip(road_pts, w_pts):
        eps.append(np.exp(-0.5*(pt-z_i).T @ np.linalg.inv(cov) @ (pt-z_i)))
        eps_w.append(np.exp(-0.5*np.sum((wp-z_w)**2)))
    eps, eps_w = np.array(eps), np.array(eps_w)

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.05, right=0.97, top=0.88, bottom=0.10)
    norm = plt.Normalize(0, 1)

    # ═══ (a) Original Frame ═══
    ax = fig.add_subplot(gs[0,0]); ax.set_aspect("equal")
    ax.set_xlim(-4,4); ax.set_ylim(-3.5,3.5)
    ax.plot([road_s[0],road_e[0]],[road_s[1],road_e[1]],color=C["road"],lw=4,zorder=2)
    ax.plot(*z_i,"o",color=C["red"],ms=9,mec="white",mew=1.2,zorder=5)
    ax.annotate("$z_i$", z_i, textcoords="offset points", xytext=(7,-12), fontsize=9, color=C["red"], fontweight="bold")
    ax.plot(*x,"D",color=C["green"],ms=8,mec="#1E8449",mew=1,zorder=5)
    ax.annotate("$x_{i,j}$", x, textcoords="offset points", xytext=(5,10), fontsize=8, color="#1E8449")
    ax.plot([z_i[0],x[0]],[z_i[1],x[1]],"-",color=C["green"],lw=2,zorder=4)
    for i in range(len(ts)-1):
        ax.plot(road_pts[i:i+2,0], road_pts[i:i+2,1], color=plt.cm.viridis(norm(eps[i])), lw=3.5, alpha=0.85, zorder=1)
    for s,ls in [(2,"--"),(1,"-")]:
        eigvals, eigvecs = np.linalg.eigh(cov)
        w = 2*np.sqrt(max(eigvals[0],1e-9))*s; h = 2*np.sqrt(max(eigvals[1],1e-9))*s
        ang = float(np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0])))
        ax.add_patch(Ellipse(z_i,w,h,angle=ang,fc="none",ec=C["blue"],lw=1.2,ls=ls,alpha=0.35))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=15, pad=0.02)
    cbar.set_label("Emission density", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) Original Frame\nAnisotropic Gaussian $\\propto\\exp(-\\frac{1}{2}\\Delta\\mathbf{x}^{\\mathsf{T}}\\mathbf{\\Sigma}_i^{-1}\\Delta\\mathbf{x})$", pad=8)

    # ═══ (b) Whitened Frame ═══
    ax = fig.add_subplot(gs[0,1]); ax.set_aspect("equal")
    ax.set_xlim(-4,4); ax.set_ylim(-3.5,3.5)
    ax.plot([r_ws[0],r_we[0]],[r_ws[1],r_we[1]],color=C["road"],lw=4,zorder=2)
    ax.plot(*z_w,"o",color=C["red"],ms=9,mec="white",mew=1.2,zorder=5)
    ax.annotate("$\\tilde{z}_i$", z_w, textcoords="offset points", xytext=(7,-12), fontsize=9, color=C["red"], fontweight="bold")
    ax.plot(*x_w,"D",color=C["green"],ms=8,mec="#1E8449",mew=1,zorder=5)
    ax.annotate("$\\tilde{x}_{i,j}$", x_w, textcoords="offset points", xytext=(5,10), fontsize=8, color="#1E8449")
    ax.plot([z_w[0],x_w[0]],[z_w[1],x_w[1]],"-",color=C["green"],lw=2,zorder=4)
    for i in range(len(ts)-1):
        ax.plot(w_pts[i:i+2,0], w_pts[i:i+2,1], color=plt.cm.viridis(norm(eps_w[i])), lw=3.5, alpha=0.85, zorder=1)
    t_c = np.linspace(0,2*np.pi,200)
    for r,ls in [(2,"--"),(1,"-")]:
        ax.plot(z_w[0]+r*np.cos(t_c),z_w[1]+r*np.sin(t_c),color=C["blue"],lw=1.2,ls=ls,alpha=0.35)
    sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax, shrink=0.5, aspect=15, pad=0.02)
    cbar2.set_label("Emission density", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Whitened Frame\nIsotropic Gaussian $\\propto\\exp(-\\frac{1}{2}\\|\\tilde{z}_i-\\tilde{x}\\|^2)$", pad=8)

    fig.text(0.5, 0.52, "$\\mathbf{L}^{-1}$", ha="center", va="center", fontsize=16, fontweight="bold", color=C["blue"],
             bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C["gray"],alpha=0.9))
    fig.text(0.5, 0.43, "Cholesky\nwhitening", ha="center", va="center", fontsize=7, color=C["gray"])
    fig.suptitle("Emission Probability: Anisotropic Gaussian in Original → Isotropic in Whitened Space",
                 fontsize=10.5, fontweight="bold", y=0.97)
    for fmt in ["svg","png"]: fig.savefig(FIGS/f"measprob.{fmt}",dpi=DPI,format=fmt)
    plt.close(fig); print("Saved measprob.svg/.png")

if __name__ == "__main__": main()
