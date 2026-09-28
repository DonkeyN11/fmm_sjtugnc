#!/usr/bin/env python3
"""Diagnostic: the junction topology around the two disputed windows.

The earlier check compared only the GT-labelled edge against the matched one, which
cannot distinguish "the label names the wrong road" from "the label names the right
road but the edge does not extend where the vehicle was". Both windows turn out to be
the second: the track projects onto the last 3 % of edge 65439 in one case and onto the
start vertex of edge 11157 in the other, and 11157 is only 129 m long.

This draws EVERY edge incident to the junction, with the RTK track as well as the SPP
track, so the question can be settled by looking rather than by inference. The two
receivers agree to ~3 m, so the RTK track is what the vehicle actually did.

Caveat that limits what the picture can settle: the vector geometry is OSM, the imagery
is a third-party basemap, and neither is a survey. The RTK track is the only independent
thing here, and it is independent of the map but not of the edge LABELS -- which is
exactly what is in question.

DIAGNOSTIC. Not referenced by any .tex; not in the Overleaf zip manifest.
"""
import csv
import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from osgeo import ogr
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "experiments/scripts"))
import verify_paper_numbers as v  # noqa: E402

OUT = PROJECT / "docs/calib/figs/gt_topology.png"
DPI = 300
LEG = FontProperties(family="Noto Sans CJK SC", size=6.4)
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK SC"],
                     "axes.unicode_minus": False,
                     "font.size": 8, "axes.titlesize": 9,
                     "figure.dpi": DPI, "savefig.dpi": DPI,
                     "savefig.bbox": "tight"})

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
CACHE = {}
M_PER_DEG = 111320.0

# Colours: the two disputants, then everything else at the junction.
COL = {"65439": "#8E44AD", "11157": "#8E44AD",          # GT labels
       "65440": "#2471A3", "101366": "#2471A3",          # matched, window 1
       "11156": "#2471A3",                               # matched, window 2
       "11158": "#16A085", "65441": "#F39C12"}

WINDOWS = [("seq 1653-1696", (1653, 1696),
            ["65439", "65441", "65440", "101366", "11158", "11157"]),
           ("seq 2502-2511", (2502, 2511),
            ["11157", "11156", "11158", "65440", "101366"])]


def deg2num(lon, lat, z):
    n = 2 ** z
    return (int((lon + 180.0) / 360.0 * n),
            int((1.0 - math.log(math.tan(math.radians(lat)) +
                                1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n))


def num2deg(xt, yt, z):
    n = 2.0 ** z
    return (xt / n * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yt / n)))))


def satellite(w, e, s, n, z):
    x0, y0 = deg2num(w, n, z)
    x1, y1 = deg2num(e, s, z)
    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    nc, nr = x1 - x0 + 1, y1 - y0 + 1
    if nc * nr > 64 and z > 15:
        return satellite(w, e, s, n, z - 1)
    mos = Image.new("RGB", (nc * 256, nr * 256), (40, 44, 48))
    for dy in range(nr):
        for dx in range(nc):
            key = (z, x0 + dx, y0 + dy)
            if key not in CACHE:
                try:
                    r = requests.get(TILE_URL.format(z=z, y=key[2], x=key[1]),
                                     timeout=15, headers={"User-Agent": "CMM-Diag/1.0"})
                    CACHE[key] = (Image.open(BytesIO(r.content)).convert("RGB")
                                  if r.status_code == 200 else None)
                except Exception:
                    CACHE[key] = None
            if CACHE[key] is not None:
                mos.paste(CACHE[key], (dx * 256, dy * 256))
    l, t = num2deg(x0, y0, z)
    rr, b = num2deg(x1 + 1, y1 + 1, z)
    return mos, (l, rr, b, t)


def load():
    ds = ogr.Open(str(PROJECT / "input/map/hainan/edges.shp"))  # keep ds alive
    lyr = ds.GetLayer(0)
    recs = {}
    lyr.ResetReading()
    for f in lyr:
        k = str(f.GetField("key"))
        g = f.GetGeometryRef()
        hw = f.GetField("highway")
        if isinstance(hw, (list, tuple)):
            hw = "/".join(hw)
        recs[k] = dict(P=np.array([g.GetPoint(i)[:2] for i in range(g.GetPointCount())]),
                       hw=str(hw).strip("[]'\" "), nm=f.GetField("name"),
                       u=f.GetField("u"), v=f.GetField("v"))
    return recs


def main():
    recs = load()
    aligned = {}
    for r in csv.DictReader(open(PROJECT / "data/real_vehicle/hainan_06/processed/"
                                 "aligned.csv", newline=""), delimiter=";"):
        if r["id"].strip() in v.EXCLUDE_TRAJ:
            continue
        aligned[(r["id"].strip(), v.norm_ts(r["timestamp"]))] = r

    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.01, right=0.99,
                          top=0.86, bottom=0.10)

    for col, (name, (lo, hi), edges) in enumerate(WINDOWS):
        pts = []
        for k, a in aligned.items():
            if a["id"].strip() != "11":
                continue
            if not (lo <= int(a["uni_seq"]) <= hi):
                continue
            try:
                pts.append((int(a["uni_seq"]), float(a["gt_x"]), float(a["gt_y"]),
                            float(a["obs_x"]), float(a["obs_y"])))
            except ValueError:
                pass
        if not pts:
            continue
        pts.sort()
        rx = np.array([p[1] for p in pts])
        ry = np.array([p[2] for p in pts])
        sx = np.array([p[3] for p in pts])
        sy = np.array([p[4] for p in pts])
        cx, cy = float(rx.mean()), float(ry.mean())
        span = max(np.ptp(np.concatenate([rx, sx])), np.ptp(np.concatenate([ry, sy])))
        pad = max(0.0016, span * 1.5)
        zoom = int(np.clip(round(math.log2(984.4 / max(span, 1e-6))) - 1, 15, 19))

        ax = fig.add_subplot(gs[0, col])
        img, (L, R, B, T) = satellite(cx - pad, cx + pad, cy - pad, cy + pad, zoom)
        ax.imshow(img, extent=(L, R, B, T), origin="upper", zorder=0,
                  interpolation="bilinear")

        # every edge in view, faint, so the reader can see what else is there
        for k, r in recs.items():
            P = r["P"]
            if (cx - pad < P[:, 0].mean() < cx + pad) and (cy - pad < P[:, 1].mean() < cy + pad):
                ax.plot(P[:, 0], P[:, 1], color="white", lw=1.2, alpha=0.30, zorder=1)

        for k in edges:
            if k not in recs:
                continue
            P = recs[k]["P"]
            cc = COL.get(k, "#607D8B")
            lw = 3.0 if k in ("65439", "11157") else 2.4
            ax.plot(P[:, 0], P[:, 1], color=cc, lw=lw, zorder=3,
                    path_effects=HALO, solid_capstyle="round")
            nm = recs[k]["nm"]
            # Label only where the edge is actually in view. An edge can extend far
            # outside the window, and labelling its midpoint there would put the text
            # off-axes -- which savefig(bbox="tight") then expands the canvas to
            # include, producing a huge mostly-blank figure.
            inside = P[(P[:, 0] > cx - pad) & (P[:, 0] < cx + pad) &
                       (P[:, 1] > cy - pad) & (P[:, 1] < cy + pad)]
            if len(inside):
                mid = inside[len(inside) // 2]
                disputant = k in ("65439", "11157")
                lbl = (f" {k}  {recs[k]['hw'][:12]}{',' + nm if nm else ''}"
                       if disputant else f" {k}")
                ax.text(mid[0], mid[1], lbl,
                        color=cc, fontsize=5.4, ha="left", va="center", zorder=7,
                        fontproperties=LEG, clip_on=True,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

        ax.plot(rx, ry, "-", color="#1B5E20", lw=1.7, zorder=5, path_effects=HALO,
                label="RTK ground-truth positions")
        ax.plot(sx, sy, "-", color="#C0392B", lw=1.4, zorder=5, alpha=0.95,
                path_effects=HALO, label="SPP fix")
        ax.plot(rx, ry, "o", color="#1B5E20", ms=2.4, zorder=6, mec="white", mew=0.4)
        ax.plot(sx, sy, "s", color="#C0392B", ms=2.0, zorder=6, mec="white", mew=0.4)

        # junction nodes, so the chain is readable
        for node, lbl in ((575340, "575340"), (100069, "100069")):
            hit = [r["P"][0] for r in recs.values() if node in (r["u"], r["v"])]
            if hit:
                p = hit[0]
                if not (cx - pad < p[0] < cx + pad and cy - pad < p[1] < cy + pad):
                    continue
                ax.plot(p[0], p[1], "*", color="#F1C40F", ms=9, zorder=8,
                        mec="black", mew=0.6)
                ax.text(p[0], p[1], f" node {lbl}", fontsize=5.4, zorder=9,
                        va="bottom", ha="left", color="black",
                        fontproperties=LEG,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + pad)
        ax.set_aspect(1.0 / math.cos(math.radians(cy)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({'ab'[col]}) Traj 11, {name}", pad=5, fontsize=8.5)
        leg = ax.legend(loc="lower left", framealpha=0.90, borderpad=0.3,
                        labelspacing=0.22, handlelength=1.3, prop=LEG)
        leg.set_zorder(10)
        m_per_deg_x = M_PER_DEG * math.cos(math.radians(cy))
        bar = 50.0 / m_per_deg_x
        x0, y0 = cx - pad * 0.93, cy - pad * 0.90
        ax.add_patch(Rectangle((x0, y0), bar, pad * 0.022, fc="white", ec="black",
                               lw=0.6, zorder=8))
        ax.text(x0 + bar / 2, y0 + pad * 0.045, "50 m", fontsize=5.8, ha="center",
                va="bottom", color="white", zorder=9,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        ax.text(0.005, 0.995, "Imagery: Esri, Maxar, Earthstar Geographics",
                transform=ax.transAxes, fontsize=4.6, color="white", ha="left",
                va="top", zorder=9,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])

    fig.suptitle("Is the ground-truth edge the wrong road, or the right road with the "
                 "wrong extent?", fontsize=9.2, fontweight="bold", y=0.965)
    fig.text(0.5, 0.017,
             "Purple = the edge the RTK labels name. Green = RTK positions, red = SPP "
             "fix; the two agree to ~3 m, so green is what the vehicle did. "
             "In both windows the track lies past the END of the purple edge.",
             ha="center", va="bottom", fontsize=6.8, color="#37474F",
             bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9F9", ec="#B0BEC5", lw=0.7))
    fig.savefig(OUT, dpi=DPI)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
