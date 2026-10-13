#!/usr/bin/env python3
"""Every wrong epoch of the six Haikou trajectories on satellite imagery, for manual
ground-truth inspection (request: Donkey.Ning, 2026-10-13).

WHY THIS EXISTS. The audit's correctness labels come from RTK-vs-network edge equality;
whether the *label* is right -- the RTK point really sitting on the edge the label
names -- has so far only been spot-checked. This script puts each of the 247 wrong
epochs on Esri World Imagery with four objects drawn: the SPP fixes, the RTK truth
points, the LABELLED edge, and the edge the matcher chose. Where the RTK point sits on
the blue (matched) edge rather than the purple (labelled) one, the label is the suspect,
not the matcher; where it sits on neither, the environment needs the imagery to judge
(dual carriageway, slip road, service road).

OUTPUTS. One PNG per trajectory under data/archive_20261013/error_maps/ -- a satellite
overview with the wrong epochs marked, then zoomed clusters with the geometry above.
Plus data/archive_20261013/wrong_epoch_geometry.csv, one row per wrong epoch, with the
four distances (RTK/SPP x labelled/matched, in metres); a wrong epoch whose RTK point is
far from its labelled edge is a candidate labelling defect, and the CSV sorts for it.

GEOMETRY NOTES (both change what the picture means if ignored).
1. A degree of longitude is shorter than a degree of latitude by cos(latitude); the map
   panels use set_aspect(1/cos(lat)), and the distance columns use the project's
   111320 m/deg convention (E-W compressed by the same cos factor).
2. The satellite layer is a third-party view, not an independent survey: it is a sanity
   check on which road looks built up, the quantitative claim rests on the distances.

Imagery: Esri World Imagery, attributed in-figure. Diagnostic figures, not manuscript
figures.

Usage: python3 experiments/scripts/fig_error_audit_maps.py [traj ...]
"""

import csv
import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from osgeo import ogr
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'experiments/scripts'))
import verify_paper_numbers as v      # noqa: E402
import real_final_tables as R         # noqa: E402

OUTDIR = PROJECT / 'data/archive_20261013/error_maps'  # outside docs/ (2026-10-13)
CSV_OUT = PROJECT / 'data/archive_20261013/wrong_epoch_geometry.csv'
DPI = 300
M_PER_DEG = 111320.0
MAXCLUST = 12                          # inset panels per trajectory (clusters merged)
C = {'obs': '#C0392B', 'gt': '#8E44AD', 'matched': '#2471A3', 'rtk': '#00A651',
     'wrong': '#E67E22'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
                     'legend.fontsize': 6.8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.unicode_minus': False, 'savefig.dpi': DPI,
                     'savefig.bbox': 'tight'})

TILE_URL = ('https://server.arcgisonline.com/ArcGIS/rest/services/'
            'World_Imagery/MapServer/tile/{z}/{y}/{x}')
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
    x0, y0 = deg2num(west, north, zoom)
    x1, y1 = deg2num(east, south, zoom)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    nc, nr = x1 - x0 + 1, y1 - y0 + 1
    if nc * nr > 64 and zoom > 14:
        return get_satellite(west, east, south, north, zoom - 1)
    mosaic = Image.new('RGB', (nc * 256, nr * 256), (40, 44, 48))
    for dy in range(nr):
        for dx in range(nc):
            key = (zoom, x0 + dx, y0 + dy)
            if key not in TILE_CACHE:
                try:
                    r = requests.get(TILE_URL.format(z=key[0], y=key[2], x=key[1]),
                                     timeout=15, headers={'User-Agent': 'CMM-Diag/1.0'})
                    TILE_CACHE[key] = (Image.open(BytesIO(r.content)).convert('RGB')
                                       if r.status_code == 200 else None)
                except Exception:
                    TILE_CACHE[key] = None
            t = TILE_CACHE[key]
            if t is not None:
                mosaic.paste(t, (dx * 256, dy * 256))
    left, top = num2deg(x0, y0, zoom)
    right, bottom = num2deg(x1 + 1, y1 + 1, zoom)
    return mosaic, (left, right, bottom, top)


def zoom_for(span_deg, target_tiles=16):
    for z in (19, 18, 17, 16, 15):
        n = 2 ** z
        nc = int(span_deg / 360.0 * n) + 2
        if nc * nc <= target_tiles:
            return z
    return 15


def load_edges_by_fid():
    ds = ogr.Open(str(PROJECT / 'input/map/hainan/edges.shp'))
    lyr = ds.GetLayer(0)
    poly = {}
    for f in lyr:
        g = f.GetGeometryRef()
        poly[f.GetFID()] = np.array([g.GetPoint(i)[:2] for i in range(g.GetPointCount())])
    return poly


def dist_m(px, py, P):
    best = np.inf
    for i in range(len(P) - 1):
        ax, ay = P[i]; bx, by = P[i + 1]
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby
        if L2 == 0:
            d = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L2))
            d = math.hypot(px - (ax + t * abx), py - (ay + t * aby))
        best = min(best, d)
    return best * M_PER_DEG


def draw_edge(ax, P, color, lw=2.2, label=None, zorder=3):
    ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, zorder=zorder, label=label,
            solid_capstyle='round')


def main():
    only = set(sys.argv[1:])
    poly = load_edges_by_fid()

    aligned = {}
    with open(R.ALIGNED, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            if r['id'].strip() in v.EXCLUDE_TRAJ:
                continue
            aligned[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r
    run = R.load_run(R.OUTDIR / 'cmm.csv')

    wrong = []
    for k, r in run.items():
        a = aligned.get(k)
        if a is None:
            continue
        g = a['gt_edge'].strip()
        if g in ('0', '-1', ''):
            continue
        if v.is_edge_match(r['cpath'], g):
            continue
        try:
            ox, oy = float(a['obs_x']), float(a['obs_y'])
        except ValueError:
            print(f'  {k}: no SPP fix -- skipped')
            continue
        try:
            tx, ty = float(a['gt_x']), float(a['gt_y'])
        except ValueError:
            tx = ty = None                # labelled but no RTK point: drawn without one
        m = r['cpath'].strip()
        m = v.REV_MAP.get(m, m) if m else m   # draw the twin the label uses
        try:
            g_i, m_i = int(g), int(m)
        except ValueError:
            print(f'  {k}: non-numeric edge id (gt {g!r}, m {m!r}) -- skipped')
            continue
        if g_i not in poly or m_i not in poly:
            print(f'  {k}: geometry missing (gt {g_i in poly}, m {m_i in poly}) -- skipped')
            continue
        rec = dict(tid=k[0], ts=k[1], seq=a['uni_seq'], obs=(ox, oy), rtk=(tx, ty),
                   gt=g_i, m=m_i,
                   d_rtk_gt=dist_m(tx, ty, poly[g_i]) if tx is not None else float('nan'),
                   d_rtk_m=dist_m(tx, ty, poly[m_i]) if tx is not None else float('nan'),
                   d_obs_gt=dist_m(ox, oy, poly[g_i]),
                   d_obs_m=dist_m(ox, oy, poly[m_i]))
        wrong.append(rec)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['traj', 'uni_seq', 'timestamp', 'gt_edge', 'matched_edge',
                    'd_rtk_to_label_m', 'd_rtk_to_matched_m',
                    'd_spp_to_label_m', 'd_spp_to_matched_m', 'label_suspect'])
        for r in sorted(wrong, key=lambda r: (r['tid'], r['ts'])):
            suspect = bool(r['d_rtk_m'] + 5 < r['d_rtk_gt'])   # nan-safe: False
            w.writerow([r['tid'], r['seq'], int(r['ts']), r['gt'], r['m'],
                        '' if math.isnan(r['d_rtk_gt']) else f"{r['d_rtk_gt']:.1f}",
                        '' if math.isnan(r['d_rtk_m']) else f"{r['d_rtk_m']:.1f}",
                        f"{r['d_obs_gt']:.1f}", f"{r['d_obs_m']:.1f}", int(suspect)])
    print(f'wrote {CSV_OUT} ({len(wrong)} wrong epochs)')

    by_traj = {}
    for r in wrong:
        by_traj.setdefault(r['tid'], []).append(r)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tid in sorted(by_traj):
        if only and tid not in only:
            continue
        recs = sorted(by_traj[tid], key=lambda r: r['ts'])
        # cluster along time/gap, then merge nearest pairs until <= MAXCLUST
        groups = [[recs[0]]]
        for r in recs[1:]:
            p = groups[-1][-1]
            gap_s = r['ts'] - p['ts']
            jump_m = math.hypot((r['obs'][0] - p['obs'][0]) * M_PER_DEG *
                                math.cos(math.radians(r['obs'][1])),
                                (r['obs'][1] - p['obs'][1]) * M_PER_DEG)
            (groups[-1].append(r) if gap_s <= 60 and jump_m <= 400 else groups.append([r]))
        while len(groups) > MAXCLUST:
            gaps = [groups[i + 1][0]['ts'] - groups[i][-1]['ts']
                    for i in range(len(groups) - 1)]
            i = int(np.argmin(gaps))
            groups[i:i + 2] = [groups[i] + groups[i + 1]]

        n_ins = len(groups)
        rows = (n_ins + 1) // 2
        fig = plt.figure(figsize=(7.4, 2.6 + rows * 2.15))
        gs = fig.add_gridspec(rows + 1, 2, hspace=0.32, wspace=0.06, left=0.045,
                              right=0.985, top=0.92, bottom=0.04,
                              height_ratios=[1.35] + [1.0] * rows)

        # --- overview ---
        ax = fig.add_subplot(gs[0, :])
        xs = [r['obs'][0] for r in recs]; ys = [r['obs'][1] for r in recs]
        pad = 0.006
        img, ext = get_satellite(min(xs) - pad, max(xs) + pad,
                                 min(ys) - pad, max(ys) + pad,
                                 zoom_for(max(max(xs) - min(xs),
                                              (max(ys) - min(ys)) / 0.9) + 2 * pad, 36))
        ax.imshow(img, extent=ext, zorder=1)
        ax.plot(xs, ys, color='white', lw=0.6, alpha=0.8, zorder=2)
        ax.scatter(xs, ys, s=7, color=C['wrong'], edgecolor='k', linewidths=0.15,
                   zorder=3)
        ax.set_aspect(1.0 / math.cos(math.radians(np.mean(ys))))
        ax.set_title(f'Trajectory {tid}: {len(recs)} wrong epochs of '
                     f'{len(aligned)} aligned rows (orange points; satellite: Esri)',
                     fontsize=8.5)
        ax.set_xlabel('longitude'); ax.set_ylabel('latitude')

        # --- insets ---
        for gi, grp in enumerate(groups):
            ax = fig.add_subplot(gs[1 + gi // 2, gi % 2])
            gx = [r['obs'][0] for r in grp]; gy = [r['obs'][1] for r in grp]
            tx = [r['rtk'][0] for r in grp if r['rtk'][0] is not None]
            ty = [r['rtk'][1] for r in grp if r['rtk'][0] is not None]
            allx = gx + tx + [c for r in grp for c in poly[r['gt']][:, 0]] + \
                   [c for r in grp for c in poly[r['m']][:, 0]]
            ally = gy + ty + [c for r in grp for c in poly[r['gt']][:, 1]] + \
                   [c for r in grp for c in poly[r['m']][:, 1]]
            mx = (max(allx) - min(allx)); my = max(ally) - min(ally)
            padx = max(0.0007, mx * 0.12); pady = max(0.0007, my * 0.12)
            img, ext = get_satellite(min(allx) - padx, max(allx) + padx,
                                     min(ally) - pady, max(ally) + pady,
                                     zoom_for(max(mx + 2 * padx, my + 2 * pady), 16))
            ax.imshow(img, extent=ext, zorder=1)
            for r in grp:
                draw_edge(ax, poly[r['gt']], C['gt'], lw=2.4, zorder=3,
                          label='labelled edge' if gi == 0 and r is grp[0] else None)
            for r in grp:
                if r['m'] != r['gt']:
                    draw_edge(ax, poly[r['m']], C['matched'], lw=2.0, zorder=3,
                              label='matched edge' if gi == 0 and r is grp[0] else None)
            ax.scatter(gx, gy, s=9, color=C['obs'], zorder=4,
                       label='SPP fix' if gi == 0 else None)
            if tx:
                ax.scatter(tx, ty, s=26, marker='*', color=C['rtk'], edgecolor='k',
                           linewidths=0.25, zorder=5,
                           label='RTK truth' if gi == 0 else None)
            ax.set_aspect(1.0 / math.cos(math.radians(np.mean(ally))))
            ax.set_xticks([]); ax.set_yticks([])
            t0, t1 = int(grp[0]['ts']), int(grp[-1]['ts'])
            dh, r0 = divmod(t0 % 86400, 3600); m0 = r0 // 60
            dh1, r1 = divmod(t1 % 86400, 3600); m1 = r1 // 60
            dmax = (np.nanmax([r['d_rtk_gt'] for r in grp])
                    if any(not math.isnan(r['d_rtk_gt']) for r in grp) else float('nan'))
            susp = [r for r in grp if r['d_rtk_m'] + 5 < r['d_rtk_gt']]
            dmax_s = 'n/a' if math.isnan(dmax) else f'{dmax:.0f}'
            ttl = f'{t0} (day {t0 // 86400}) {dh:02d}:{m0:02d} - {dh1:02d}:{m1:02d} | ' \
                  f'{len(grp)} ep | RTK->label max {dmax_s} m'
            if susp:
                ttl += f' | LABEL? {len(susp)}'
            ax.set_title(ttl, fontsize=6.5,
                         color='#B00020' if susp else 'black')
        handles = [Line2D([], [], color=C['gt'], lw=2.4, label='labelled edge'),
                   Line2D([], [], color=C['matched'], lw=2.0, label='matched edge'),
                   Line2D([], [], marker='o', color=C['obs'], lw=0, ms=4, label='SPP fix'),
                   Line2D([], [], marker='*', color=C['rtk'], lw=0, ms=8,
                          markeredgecolor='k', markeredgewidth=0.25, label='RTK truth')]
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.005))
        out = OUTDIR / f'traj{tid}.png'
        fig.savefig(out, facecolor='white')
        plt.close(fig)
        print(f'{out}: {len(recs)} wrong epochs in {n_ins} clusters, '
              f'{sum(1 for r in recs if r["d_rtk_m"] + 5 < r["d_rtk_gt"])} label-suspect')


if __name__ == '__main__':
    main()
