#!/usr/bin/env python3
"""Fig: the residual decomposition and the total emission covariance, Eq. (sigma_total).

Panel (a) is the schematic the equation describes: the residual d = z - x is not the
receiver error alone. The fix z sits at the true position p plus the GNSS error epsilon,
and the candidate x sits on the DIGITISED centreline, which is displaced from the true
one by the map's georeferencing error eta. The vehicle is itself off the true centreline
by its lateral offset lambda. Both eta and lambda are uncorrelated with epsilon and have
no preferred direction, which is why Eq. (sigma_total) adds an isotropic term.

Panel (b) draws the two covariances to scale, and the point of it is how small the
reported one is: 0.81 m against 10.6 m is a dot at the centre of the circle, so it is
shown magnified in the inset rather than left invisible.

Panel (c) is the consequence. The likelihood the model computed before the correction
decayed by a factor of 1e-48 at the 12 m separation of a dual carriageway, so two rival
carriageways could not both be entertained and the posterior was pinned at 1 by
arithmetic rather than by evidence. With the map term the same ratio is 0.53.

Panel (a) is qualitative in SCALE -- 0.81 m beside 10.6 m cannot be drawn to scale in one
panel -- but the offsets are perpendicular by construction and share one scale with each
other, and panels (b) and (c) carry the magnitudes. The construction is built in a
road-aligned frame and rotated at draw time, so no label sits near a line it does not
belong to.

Palette validated with the dataviz validator (light mode): all six checks pass, worst
adjacent CVD pair dE 15.5 (deutan) / 10.9 (tritan), all four marks above 3:1 contrast.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

PROJECT = Path(__file__).resolve().parents[2]
FIGS = PROJECT / "docs/calib/figs"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 300
C = {"gnss": "#C0392B",    # the fix and its error
     "map": "#8E44AD",     # map georeferencing error
     "lane": "#16A085",    # lateral offset within the carriageway
     "cov": "#2471A3",     # the covariance ellipses
     "true": "#95A5A6",    # true centreline
     "drawn": "#566573"}   # digitised centreline

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 6.6, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "figure.dpi": DPI,
                     "savefig.dpi": DPI, "savefig.bbox": "tight",
                     "text.usetex": False})

SIG_REPORTED = 0.81    # median reported semi-major axis, Haikou
SIG_MAP = 10.6         # sqrt(10^2 + 3.5^2), the manuscript's delta_road decomposition
SIG_TOTAL = float(np.hypot(SIG_REPORTED, SIG_MAP))
SEP_PARALLEL = 12.0    # separation of the two carriageways in the Traj 22 geometry

# Panel (a) geometry, in the road-aligned frame. ETA and LAM are drawn large relative to
# the segment length because they ARE large: 10.6 m and 3.5 m against a segment tens of
# metres long, which is the whole reason the term matters.
TH_DEG = -9.0
ETA, LAM = 2.30, 0.95
XP = 1.10
X_FIX = -0.55


def panel_a(ax):
    th = np.deg2rad(TH_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def W(pt):
        return R @ np.asarray(pt, float)

    r = (XP, 0.0)                          # true centreline point
    p = (XP, LAM)                          # vehicle, off the centreline by lambda
    z = (X_FIX, LAM + 0.45)                # the fix: p + epsilon
    x = (X_FIX, ETA)                       # candidate: projection onto the drawn line

    for y0, col, ls, lw in ((0.0, C["true"], "--", 1.7), (ETA, C["drawn"], "-", 2.7)):
        ln = np.array([W((t, y0)) for t in (-2.4, 2.4)])
        ax.plot(ln[:, 0], ln[:, 1], ls, color=col, lw=lw, zorder=1)

    def arrow(tail, head, col, style="<->", lw=1.5, ms=10):
        ax.annotate("", xy=W(head), xytext=W(tail),
                    arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                    shrinkA=0, shrinkB=0, mutation_scale=ms))

    # map error: between the two centrelines, label pushed clear to the left
    arrow((-1.85, 0.0), (-1.85, ETA), C["map"], lw=1.6, ms=11)
    ax.text(*W((-2.02, ETA * 0.50)), "$\\eta(s_i)$", color=C["map"], fontsize=9.5,
            fontweight="bold", ha="right", va="center")

    # lane offset: vehicle against the true centreline
    arrow((XP, 0.0), (XP, LAM), C["lane"], ms=9)
    ax.text(*W((XP - 0.18, LAM * 0.5)), "$\\lambda_i$", color=C["lane"], fontsize=9.5,
            fontweight="bold", ha="right", va="center")

    # the reported covariance around the fix
    for s_, al in ((2, 0.10), (1, 0.22)):
        ax.add_patch(Ellipse(W(z), 0.80 * s_, 0.72 * s_, angle=TH_DEG, fc=C["cov"],
                             alpha=al, ec="none", zorder=2))

    # the residual, its label to the right of its own arrow
    arrow(x, z, C["gnss"], style="-|>", lw=1.8, ms=12)
    ax.text(*W((X_FIX + 0.24, (ETA + z[1]) / 2)), "$\\mathbf{d}_{i,j}$", color=C["gnss"],
            fontsize=9, fontweight="bold", ha="left", va="center")

    # points, each labelled on a side chosen to be empty
    ax.plot(*W(r), "o", color=C["true"], ms=5, zorder=4)
    ax.text(*W((XP + 0.16, -0.24)), "$r_i$", color="#6E7B7F", fontsize=8.5,
            ha="left", va="top")
    ax.plot(*W(p), "o", color=C["lane"], ms=5.5, zorder=4)
    ax.text(*W((XP + 0.24, LAM + 0.22)), "$p_i$", color=C["lane"], fontsize=8.5,
            ha="left", va="bottom")
    ax.plot(*W(z), "o", color=C["gnss"], ms=6.5, mec="white", mew=1.1, zorder=5)
    ax.text(*W((X_FIX - 0.62, z[1] + 0.26)), "$z_i$", color=C["gnss"], fontsize=9.5,
            fontweight="bold", ha="center", va="center")
    ax.text(*W((X_FIX - 0.70, z[1] - 0.18)), "$\\Sigma_i$", color=C["cov"], fontsize=8,
            ha="center", va="center")
    ax.plot(*W(x), "D", color=C["cov"], ms=5.6, mec="white", mew=1.0, zorder=5)
    ax.text(*W((X_FIX + 0.26, ETA + 0.12)), "$x_{i,j}$", color=C["cov"], fontsize=9,
            fontweight="bold", ha="left", va="bottom")

    ax.set_xlim(-2.95, 2.55)
    ax.set_ylim(-0.68, 2.95)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) What the residual contains", pad=7)


def panel_b(ax):
    """The two covariances to scale; the reported one is a dot, so inset it."""
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(2 * SIG_MAP * np.cos(th), 2 * SIG_MAP * np.sin(th),
            fc=C["cov"], alpha=0.13, ec="none", zorder=1)
    ax.plot(2 * SIG_MAP * np.cos(th), 2 * SIG_MAP * np.sin(th),
            color=C["cov"], lw=1.2, zorder=2)
    ax.plot(2 * SIG_TOTAL * np.cos(th), 2 * SIG_TOTAL * np.sin(th),
            "--", color=C["cov"], lw=0.9, alpha=0.75, zorder=2)

    ax.plot(0, 0, "o", color=C["gnss"], ms=3.2, zorder=4)
    ax.annotate("reported $\\Sigma_i$\n(shown $13\\times$ in inset)",
                xy=(0, 0), xytext=(-2 * SIG_MAP * 1.18, -2 * SIG_MAP * 1.22),
                fontsize=7, color=C["gnss"], ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C["gnss"], lw=0.9,
                                shrinkA=1, shrinkB=3, mutation_scale=8))
    ax.text(0, 2 * SIG_MAP * 1.12, "$\\bar{\\Sigma}_i=\\Sigma_i"
            "+\\sigma_{\\mathrm{map}}^2\\mathbf{I}_2$",
            color=C["cov"], fontsize=8.5, ha="center", va="bottom")
    ax.text(0, -2 * SIG_MAP * 0.70, "$\\sigma_{\\mathrm{map}}=10.6$ m",
            color=C["cov"], fontsize=8, ha="center", va="center")

    axins = ax.inset_axes([0.60, 0.43, 0.37, 0.30])
    axins.add_patch(Ellipse((0, 0), 2 * SIG_REPORTED, 2 * SIG_REPORTED,
                            fc=C["cov"], alpha=0.22, ec=C["cov"], lw=0.9))
    axins.plot(0, 0, "o", color=C["gnss"], ms=3.0, zorder=4)
    axins.set_xlim(-1.7 * SIG_REPORTED, 1.7 * SIG_REPORTED)
    axins.set_ylim(-1.7 * SIG_REPORTED, 1.7 * SIG_REPORTED)
    axins.set_aspect("equal")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("$\\Sigma_i$, $\\sigma\\!\\approx\\!0.81$ m", fontsize=6.5, pad=2)
    for sp in axins.spines.values():
        sp.set_edgecolor("#B0BEC5")
        sp.set_linewidth(0.7)

    ax.set_xlim(-2 * SIG_MAP * 1.24, 2 * SIG_MAP * 1.24)
    ax.set_ylim(-2 * SIG_MAP * 1.52, 2 * SIG_MAP * 1.42)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) The two covariances, to scale", pad=7)


def panel_c(ax):
    """Why it matters: the likelihood at the carriageway separation."""
    d = np.linspace(0, 30, 800)

    def prof(sig):
        return np.exp(-0.5 * d ** 2 / sig ** 2)

    ax.semilogy(d, prof(SIG_REPORTED), color=C["gnss"], lw=1.8,
                label="without the map term, $\\sigma=0.81$ m")
    ax.semilogy(d, prof(SIG_TOTAL), color=C["cov"], lw=1.8,
                label="as modelled, $\\sigma=10.6$ m")

    ax.axvline(SEP_PARALLEL, color=C["map"], lw=1.1, ls=":", zorder=1)
    ax.text(SEP_PARALLEL + 0.8, 2e-4, "rival\ncarriageway\n(12 m)", color=C["map"],
            fontsize=6.8, va="top", ha="left", linespacing=1.25)

    r_old = prof(SIG_REPORTED)[np.argmin(abs(d - SEP_PARALLEL))]
    r_new = prof(SIG_TOTAL)[np.argmin(abs(d - SEP_PARALLEL))]
    ax.annotate("likelihood ratio across the gap\n"
                f"before $\\approx 10^{{{np.log10(r_old):.0f}}}$, after ${r_new:.2f}$",
                xy=(13.2, 3e-20), fontsize=7, color="#37474F", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.34", fc="#F8F9F9", ec="#CFD8DC", lw=0.7))

    ax.set_xlim(0, 30)
    ax.set_ylim(1e-50, 3.0)
    ax.set_xlabel("perpendicular distance from the candidate (m)", labelpad=2)
    ax.set_ylabel("relative likelihood")
    ax.legend(loc="lower left", framealpha=0.94, borderpad=0.42)
    ax.grid(alpha=0.18, lw=0.5, which="major")
    ax.set_title("(c) What the omitted term did", pad=7)


def main():
    fig = plt.figure(figsize=(7.2, 3.55))
    gs = fig.add_gridspec(1, 3, wspace=0.25, left=0.015, right=0.985,
                          top=0.825, bottom=0.235, width_ratios=[1.22, 1.0, 0.95])
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[0, 2]))

    fig.text(0.5, 0.022,
             "$\\bar{\\Sigma}_i=\\Sigma_i+\\sigma_{\\mathrm{map}}^2\\mathbf{I}_2$, "
             "$\\ \\sigma_{\\mathrm{map}}=\\sqrt{\\sigma_\\eta^2+\\sigma_\\lambda^2}"
             "=\\sqrt{10^2+3.5^2}=10.6$ m   ---   combined in variance, because a "
             "radius allowance is a bound and a covariance is not",
             ha="center", va="bottom", fontsize=7.5, color="#263238",
             bbox=dict(boxstyle="round,pad=0.34", fc="#F8F9F9", ec="#B0BEC5",
                       lw=0.7, alpha=0.95))
    fig.suptitle("The residual is not the receiver error alone: separating its sources "
                 "gives the emission covariance",
                 fontsize=9.5, fontweight="bold", y=0.972)

    for fmt in ("png", "svg"):
        fig.savefig(FIGS / f"sigma_total.{fmt}", dpi=DPI, format=fmt)
    plt.close(fig)
    print(f"Saved {FIGS/'sigma_total.png'} and .svg")


if __name__ == "__main__":
    main()
