#!/usr/bin/env python3
"""Diagnostic: is the ground-truth defect real, or is the matcher wrong?

For every epoch the new model scored "wrong", compute two distances from the SPP fix:
to the edge the RTK ground truth names, and to the edge the matcher chose. Where the
second is much the smaller, the matcher is geometrically right and the LABEL is wrong.

The map panels carry a satellite layer so the ENVIRONMENT can be judged directly -- a
dual carriageway, a slip road beside a mainline, a service road -- rather than inferred
from the vector geometry alone. The distance profiles below them are what make this a
check rather than an assertion: they show the fix leaving the labelled edge while
staying on the matched one.

Two geometry notes, both of which change what the picture means if ignored.
1. A degree of longitude is shorter than a degree of latitude by cos(latitude), so the
   map panels use set_aspect(1/cos(lat)). Without it the imagery is stretched by 6 % at
   Haikou and the apparent distances are wrong.
2. The satellite imagery is a third-party view, not an independent survey. It is a
   sanity check on which road looks built up and which looks like a service road; the
   quantitative claim rests on the distances, which come from the receiver's own fixes.

Tiles: Esri World Imagery, attributed in-figure. This is a DIAGNOSTIC, not a manuscript
figure -- it is not referenced by any .tex and is not in the Overleaf zip manifest.
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

OUT = PROJECT / "docs/calib/figs/gt_defect_check.png"
DPI = 300

C = {"obs": "#C0392B", "gt": "#8E44AD", "matched": "#2471A3"}
HALO = [pe.withStroke(linewidth=3.4, foreground="white")]
# 白驹大道 needs a CJK face, and matplotlib 3.11 here does not fall back through
# font.sans-serif for a missing glyph -- an explicit FontProperties is required.
# Noto Sans CJK also carries Latin, so one property serves the whole legend.
LEG = FontProperties(family="Noto Sans CJK SC", size=6.8)

# DejaVu Sans carries no CJK glyphs, and two of the edges in this figure are named
# in Chinese (白驹大道). Listing a CJK face as a fallback lets matplotlib resolve
# those per glyph instead of drawing a tofu box; the Latin text is unaffected
# because DejaVu is still first in the list.
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK SC",
                                        "Droid Sans Fallback"],
                     "axes.unicode_minus": False, "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "figure.dpi": DPI,
                     "savefig.dpi": DPI, "savefig.bbox": "tight"})

M_PER_DEG = 111320.0
RUNS = [("seq 1653-1696", (1653, 1696)), ("seq 2502-2511", (2502, 2511))]

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TILE_CACHE = {}


def deg2num(lon, lat, zoom):
    n = 2 ** zoom
    return (int((lon + 180.0) / 360.0 * n),
            int((1.0 - math.log(math.tan(math.radians(lat)) +
                                1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n))


def num2deg(xt, yt, zoom):
    n = 2.0 ** zoom
    return (xt / n * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yt / n)))))


def get_satellite(west, east, south, north, zoom):
    """Mosaic of Esri World Imagery covering the box; returns (image, extent)."""
    x0, y0 = deg2num(west, north, zoom)
    x1, y1 = deg2num(east, south, zoom)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    nc, nr = x1 - x0 + 1, y1 - y0 + 1
    if nc * nr > 64 and zoom > 15:
        return get_satellite(west, east, south, north, zoom - 1)
    mosaic = Image.new("RGB", (nc * 256, nr * 256), (40, 44, 48))
    got = 0
    for dy in range(nr):
        for dx in range(nc):
            key = (zoom, x0 + dx, y0 + dy)
            if key not in TILE_CACHE:
                try:
                    r = requests.get(TILE_URL.format(z=zoom, y=key[2], x=key[1]),
                                     timeout=15, headers={"User-Agent": "CMM-Diag/1.0"})
                    TILE_CACHE[key] = (Image.open(BytesIO(r.content)).convert("RGB")
                                       if r.status_code == 200 else None)
                except Exception:
                    TILE_CACHE[key] = None
            t = TILE_CACHE[key]
            if t is not None:
                mosaic.paste(t, (dx * 256, dy * 256))
                got += 1
    left, top = num2deg(x0, y0, zoom)
    right, bottom = num2deg(x1 + 1, y1 + 1, zoom)
    print(f"    zoom {zoom}: {nc}x{nr}={nc*nr} tiles, {got} retrieved")
    return mosaic, (left, right, bottom, top)


def load_network():
    # Keep the DataSource alive: `ogr.Open(...).GetLayer(0)` drops the only reference
    # to the parent, which is then collected while the layer is still being read. In
    # this process that is a segfault, not an exception, so the reference is held
    # explicitly and released only after the geometry has been copied out.
    ds = ogr.Open(str(PROJECT / "input/map/hainan/edges.shp"))
    lyr = ds.GetLayer(0)
    poly, meta = {}, {}
    lyr.ResetReading()
    for f in lyr:
        k = str(f.GetField("key"))
        g = f.GetGeometryRef()
        poly[k] = np.array([g.GetPoint(i)[:2] for i in range(g.GetPointCount())])
        meta[k] = (f.GetField("highway"), f.GetField("name"))
    return poly, meta


def dist_to(px, py, P):
    best = np.inf
    for i in range(len(P) - 1):
        ax, ay = P[i]
        bx, by = P[i + 1]
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby
        if L2 == 0:
            d = np.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L2))
            d = np.hypot(px - (ax + t * abx), py - (ay + t * aby))
        best = min(best, d)
    return best * M_PER_DEG


def collect(poly):
    aligned = {}
    for r in csv.DictReader(open(PROJECT / "data/real_vehicle/hainan_06/"
                                 "processed/aligned.csv", newline=""), delimiter=";"):
        if r["id"].strip() in v.EXCLUDE_TRAJ:
            continue
        aligned[(r["id"].strip(), v.norm_ts(r["timestamp"]))] = r
    run = {}
    for r in csv.DictReader(open(PROJECT / "data/archive_20260927/final/"
                                 "row5_map10p6.csv", newline=""), delimiter=";"):
        run[(r["id"].strip(), v.norm_ts(r["timestamp"]))] = r
    out = []
    for k, r in run.items():
        a = aligned.get(k)
        if a is None or a["id"].strip() != "11":
            continue
        g = a["gt_edge"].strip()
        if g in ("0", "-1", ""):
            continue
        try:
            x, y = float(a["obs_x"]), float(a["obs_y"])
        except ValueError:
            continue
        m = r["cpath"].strip()
        out.append(dict(seq=int(a["uni_seq"]), x=x, y=y, gt=g, m=m,
                        dg=dist_to(x, y, poly[g]) if g in poly else np.nan,
                        dm=dist_to(x, y, poly[m]) if m in poly else np.nan))
    out.sort(key=lambda d: d["seq"])
    return out


def main():
    poly, meta = load_network()
    obs = collect(poly)

    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.16, left=0.06,
                          right=0.985, top=0.885, bottom=0.075,
                          height_ratios=[1.35, 1.0])

    for col, (name, (lo, hi)) in enumerate(RUNS):
        win = [d for d in obs if lo <= d["seq"] <= hi]
        if not win:
            continue
        gts, ms = {d["gt"] for d in win}, {d["m"] for d in win}
        px = np.array([d["x"] for d in win])
        py = np.array([d["y"] for d in win])
        cx, cy = float(px.mean()), float(py.mean())
        span_deg = max(px.ptp(), py.ptp())
        pad = max(0.00075, span_deg * 0.80)
        zoom = int(np.clip(round(math.log2(984.4 / max(span_deg, 1e-6))), 16, 19))

        # ---- map over satellite ----
        ax = fig.add_subplot(gs[0, col])
        img, (L, R, B, T) = get_satellite(cx - pad, cx + pad, cy - pad, cy + pad, zoom)
        ax.imshow(img, extent=(L, R, B, T), origin="upper", zorder=0,
                  interpolation="bilinear")
        # faint vector network, so the map's own geometry stays visible on the imagery
        for k, P in poly.items():
            if cx - pad < P[:, 0].mean() < cx + pad and cy - pad < P[:, 1].mean() < cy + pad:
                ax.plot(P[:, 0], P[:, 1], color="white", lw=1.5, alpha=0.35, zorder=1)
        for k, cc in [(k, C["gt"]) for k in gts] + [(k, C["matched"]) for k in ms]:
            ax.plot(poly[k][:, 0], poly[k][:, 1], color=cc, lw=2.6, zorder=3,
                    path_effects=HALO, solid_capstyle="round")
        ax.plot(px, py, "-", color=C["obs"], lw=1.6, zorder=4, path_effects=HALO)
        ax.plot(px, py, "o", color=C["obs"], ms=2.8, zorder=5,
                mec="white", mew=0.5)
        ax.plot(px[0], py[0], "s", color="#2E7D32", ms=5, zorder=6,
                mec="white", mew=0.7)

        hx = (R - L) * 0.5
        hy = (T - B) * 0.5
        ax.set_xlim(cx - hx, cx + hx)          # cover exactly what the mosaic covers
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_aspect(1.0 / math.cos(math.radians(cy)))   # metres, not degrees
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({'ab'[col]}) Traj 11, {name}", pad=5, fontsize=8.5)
        def tag(k):
            hw, nm = meta.get(k, ("?", None))
            if isinstance(hw, (list, tuple)):
                hw = "/".join(hw)
            hw = str(hw).strip("[]\'\" ").replace("\'", "")
            return f"{hw}, {nm}" if nm else f"{hw}, unnamed"
        gtag = " / ".join(tag(k) for k in sorted(gts))
        mtag = " / ".join(tag(k) for k in sorted(ms))
        ax.plot([], [], color=C["gt"], lw=2.6,
                label=f"RTK ground truth: edge {', '.join(sorted(gts))} ({gtag})")
        ax.plot([], [], color=C["matched"], lw=2.6,
                label=f"matcher chose: {', '.join(sorted(ms))} ({mtag})")
        ax.plot([], [], color=C["obs"], lw=1.6, label="SPP fix (the observation)")
        leg = ax.legend(loc="lower left", framealpha=0.90, borderpad=0.32,
                        labelspacing=0.24, handlelength=1.4, prop=LEG)
        leg.set_zorder(10)
        # scale bar
        m_per_deg_x = M_PER_DEG * math.cos(math.radians(cy))
        bar = 100.0 / m_per_deg_x
        x0 = cx - hx * 0.92
        y0 = cy - hy * 0.90
        ax.add_patch(Rectangle((x0, y0), bar, hy * 0.022, fc="white", ec="black",
                               lw=0.6, zorder=7))
        ax.text(x0 + bar / 2, y0 + hy * 0.045, "100 m", color="white", fontsize=6.2,
                ha="center", va="bottom", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
        ax.text(0.006, 0.994, "Imagery: Esri, Maxar, Earthstar Geographics",
                transform=ax.transAxes, fontsize=4.8, color="white", ha="left",
                va="top", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.3, foreground="black")])

        # ---- distance profile ----
        ax2 = fig.add_subplot(gs[1, col])
        sq = [d["seq"] for d in win]
        ax2.plot(sq, [d["dg"] for d in win], "-o", color=C["gt"], lw=1.6, ms=3.0,
                 label="fix to the edge the GT names")
        ax2.plot(sq, [d["dm"] for d in win], "-s", color=C["matched"], lw=1.6, ms=3.0,
                 label="fix to the edge the matcher chose")
        ax2.set_xlabel("epoch (uni_seq)", labelpad=1)
        ax2.set_ylabel("distance from the fix (m)", labelpad=1)
        ax2.grid(alpha=0.20, lw=0.5)
        ax2.legend(loc="upper left", framealpha=0.92, borderpad=0.32,
                   labelspacing=0.24, prop=LEG)
        ax2.set_title(f"({'cd'[col]}) distance from the fix, same epochs",
                      pad=5, fontsize=8.5)

    fig.suptitle("Ground-truth defect or matcher error?  Traj 11, the two runs "
                 "with a decisive margin", fontsize=9.5, fontweight="bold", y=0.972)
    fig.text(0.5, 0.010,
             "If the ground truth were right, the purple curve would sit BELOW the blue "
             "one. It sits above it throughout, by up to 120 m.",
             ha="center", va="bottom", fontsize=7.5, color="#37474F",
             bbox=dict(boxstyle="round,pad=0.32", fc="#F8F9F9", ec="#B0BEC5", lw=0.7))
    fig.savefig(OUT, dpi=DPI)
    plt.close(fig)
    print(f"Saved {OUT}")

    for name, (lo, hi) in RUNS:
        w = [d for d in obs if lo <= d["seq"] <= hi]
        marg = np.array([d["dg"] - d["dm"] for d in w])
        print(f"  {name}: n={len(w)}  margin min {marg.min():.1f} / median "
              f"{np.median(marg):.1f} / max {marg.max():.1f} m")


if __name__ == "__main__":
    main()
