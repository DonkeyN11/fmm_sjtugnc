#!/usr/bin/env python3
"""Interactive per-epoch audit pages for every wrong HMM match (request: Donkey.Ning,
2026-10-13 -- replace the static PNG panels, which were too cramped to zoom).

OUTPUT LAYOUT. data/archive_20261013/error_maps/traj<tid>/ for each of the five trajectories
that has wrong epochs (traj 14 has none):
  * err_ts<timestamp>.html  -- ONE page per wrong epoch: a folium/Leaflet slippy map on
    Esri World Imagery (live tiles, so pan/zoom are real), carrying
      - the trajectory context (the alignment fixes around the epoch, white),
      - the wrong epoch's SPP fix (red marker; popup: ids, edge names, all distances),
      - the LABELLED edge (purple, click for id/name/highway/oneway/length),
      - the MATCHED edge (blue, same popup, plus this epoch's alpha and alpha-beta tw),
      - every CANDIDATE at the epoch (orange: the projection point at the fix, popup
        with the edge id and its distance; computed with the matcher's own predicate,
        degree-space distance <= 0.0026949335, candidate_membership.py's machinery),
      - the truth point(s) as a circle whose colour is the NMEA fix quality (green
        fixed / orange float / red not fixed; absent for the one labelled epoch that
        carries no RTK coordinates),
      - a fixed info panel with the same ids/distances as text and prev/next links.
  * index.html  -- trajectory overview: every wrong epoch as a marker whose popup links
    to its page, plus a table of all of them sorted by time.
  * the old traj<tid>.png is moved into the folder as the static companion.
A root docs/calib/figs/error_maps/index.html links the five folders.

WHY THESE FIELDS. Ordering the pages by timestamp (not uni_seq, which is unstable
across alignments -- see mark_missing_road_gt.py) and putting every identity number in
a click popup is what re-labelling a gt_edge needs: which edge is labelled, which the
matcher chose, which were even offered, what the RTK point says, and what the edges are
called.

Usage: python3 experiments/scripts/error_maps_html.py
"""

import ast
import csv
import html
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import folium
import numpy as np
from branca.element import Element
from osgeo import ogr

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'experiments/scripts'))
import verify_paper_numbers as v            # noqa: E402
import real_final_tables as R               # noqa: E402
import candidate_membership as CM           # noqa: E402

ROOT = PROJECT / 'data/archive_20261013/error_maps'   # outside docs/: the docs tree
                                                       # carries the manuscript and its
                                                       # figures only (Donkey.Ning,
                                                       # 2026-10-13)
CSV_IN = PROJECT / 'data/archive_20261013/wrong_epoch_geometry.csv'
M_PER_DEG = 111320.0
C = {'gt': '#8E44AD', 'matched': '#2471A3', 'obs': '#C0392B', 'rtk': '#00A651',
     'cand': '#E67E22'}
ESRI = ('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/'
        'MapServer/tile/{z}/{y}/{x}')
ESRI_ATTR = 'Esri World Imagery'


def _flatten(v):
    """The shapefile carries several attributes as stringified Python lists
    ("['primary', 'motorway']"); render them as plain comma lists."""
    if v is None:
        return ''
    s = str(v).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            return ', '.join(str(x) for x in ast.literal_eval(s))
        except (ValueError, SyntaxError):
            pass
    return s


def edge_attr(fid, meta):
    name, hw, ow, ln = meta.get(fid, ('', '', '', ''))
    bits = []
    for v in (name, hw):
        s = _flatten(v)
        if s:
            bits.append(html.escape(s))
    # oneway is stored 'T'/'F' in this shapefile (never '1'/'true') -- the earlier
    # ('1','true','True') test matched nothing and the badge silently never rendered
    # (found by the 2026-10-13 verification pass; 16,905 of 152,547 edges are 'T').
    if str(ow).strip().upper() in ('T', 'TRUE', '1'):
        bits.append('one-way')
    if ln:
        try:
            bits.append(f'{float(ln):.0f} m')
        except ValueError:
            pass
    return ', '.join(bits)


TRAJ_MAP = {11: '1.1', 12: '1.2', 13: '1.3', 14: '1.4',
            21: '2.1', 22: '2.2', 23: '2.3'}

# reverse of v.REV_MAP, built once: REV_MAP is not an involution (3.29 % of its
# 135,564 entries do not map back), so twin tests must look in BOTH directions or
# they miss a twin (the yellow nearest-edge line used to draw the labelled edge's
# own twin on 18 of 73 pages -- 2026-10-13 verification note).
REV_INV = {}
for _k, _v in v.REV_MAP.items():
    REV_INV.setdefault(_v, _k)


def twin_closure(*ids):
    c = set()
    for x in ids:
        if x is None:
            continue
        c.add(int(x))
        t1 = v.REV_MAP.get(str(x))
        if t1:
            c.add(int(t1))
        t2 = REV_INV.get(str(x))
        if t2:
            c.add(int(t2))
    return c


def bearing_at(P, lat, lon):
    """Compass bearing (deg, 0 = north, clockwise) of the polyline's digitisation
    direction at its closest point to (lat, lon), E-W scaled by cos(lat).

    Answers 'which way does this id point' for CANDIDATE edges, whose one-way badge
    alone cannot -- drawing 80+ extra arrowheads per page would be unreadable, so
    the cue goes into their popups instead (2026-10-13 verification note)."""
    pts = np.asarray(P, dtype=float)
    best = (1e18, 0)
    for i in range(len(pts) - 1):
        ax, ay = pts[i][1], pts[i][0]
        bx, by = pts[i + 1][1], pts[i + 1][0]
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((lon - ax) * abx + (lat - ay) * aby) / L2))
        cx, cy = ax + t * abx, ay + t * aby
        d2 = (lon - cx) ** 2 + (lat - cy) ** 2
        if d2 < best[0]:
            best = (d2, i)
    i = best[1]
    dx = (pts[i + 1][1] - pts[i][1]) * math.cos(math.radians(lat))
    dy = pts[i + 1][0] - pts[i][0]
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def load_rtk_quality(tid):
    """epoch ts -> NMEA GGA fix quality (4 = RTK fixed, 5 = float, 1/2 = not fixed).

    Parsed with build_real_data_gt.py's two-pass GGA/RMC scheme so the same epoch
    timestamps come out. Necessary because 11 of the 246 wrong epochs with a truth
    point are NOT fixed solutions -- 8 autonomous (q=1) and 3 float (q=5) -- yet the
    pages had called every star 'RTK ground truth' (found by the 2026-10-13
    verification pass; the GGA quality field is what build_real_data_gt.py never
    reads)."""
    # The raw solution files live under hainan_06/ (build_real_data_gt.py's own
    # RTK_BASE = data/real_vehicle is stale -- the data was reorganised into the
    # per-city subdirectory after that script was written).
    f = (PROJECT / 'data/real_vehicle/hainan_06' / TRAJ_MAP[int(tid)] / '实时定位结果'
         / 'rtk_solution_clean.txt')
    if not f.exists():
        return {}
    time_to_date, current_date = {}, None
    for line in open(f, encoding='ascii', errors='ignore'):
        i = line.find('$')
        if i < 0:
            continue
        fields = line[i:].split('*')[0].split(',')
        mt = fields[0][3:]
        if mt == 'RMC' and len(fields) >= 10:
            try:
                current_date = date(int(fields[9][4:6]) + 2000,
                                    int(fields[9][2:4]), int(fields[9][0:2]))
            except ValueError:
                pass
            if current_date:
                time_to_date[fields[1]] = current_date
        elif mt == 'GGA' and len(fields) >= 10:
            if fields[1] and current_date:
                time_to_date[fields[1]] = current_date
    out = {}
    for line in open(f, encoding='ascii', errors='ignore'):
        i = line.find('$')
        if i < 0:
            continue
        fields = line[i:].split('*')[0].split(',')
        if fields[0][3:] != 'GGA' or len(fields) < 10:
            continue
        t = fields[1]
        d = time_to_date.get(t)
        if not d:
            continue
        try:
            h, m = int(t[0:2]), int(t[2:4])
            s = float(t[4:])
            ts = round(datetime(d.year, d.month, d.day, h, m, int(s),
                                int((s - int(s)) * 1e6), tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
        out[ts] = fields[6]
    return out


def rtk_quality(qmap, ts):
    """build_real_data_gt.py's matching rule: exact epoch, then +-1 s."""
    for k in (int(ts), int(ts) + 1, int(ts) - 1):
        if k in qmap:
            return qmap[k]
    return None


def load_edges():
    ds = ogr.Open(str(PROJECT / 'input/map/hainan/edges.shp'))
    lyr = ds.GetLayer(0)
    poly, meta = {}, {}
    for f in lyr:
        fid = f.GetFID()
        g = f.GetGeometryRef()
        poly[fid] = [(g.GetY(i), g.GetX(i)) for i in range(g.GetPointCount())]
        meta[fid] = (f.GetField('name'), f.GetField('highway'), f.GetField('oneway'),
                     f.GetField('length'))
    return poly, meta


def project_point(px, py, P):
    """Closest point on polyline P to the fix (px = lon, py = lat); returns
    (closest_lon, closest_lat, distance_m).

    P is in folium order, [(lat, lon), ...] -- load_edges stores it that way because
    that is what folium.PolyLine eats. The first version of this helper read P[i] as
    (lon, lat) and returned transposed points, which put every candidate POINT marker
    off the map on all 247 pages (the candidate EDGE polylines were unaffected); it
    was caught by the 2026-10-13 adversarial verification pass, whose headless
    chromium bounding-box test found 5792/5792 candidate markers with a longitude in
    Leaflet's latitude slot, with zero console errors -- a silent loss."""
    best = (1e18, px, py)
    for i in range(len(P) - 1):
        ax, ay = P[i][1], P[i][0]
        bx, by = P[i + 1][1], P[i + 1][0]
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby
        if L2 == 0:
            cx, cy = ax, ay
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L2))
            cx, cy = ax + t * abx, ay + t * aby
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best[0]:
            best = (d2, cx, cy)
    return best[1], best[2], math.sqrt(best[0]) * M_PER_DEG


def direction_arrow(P, color, weight=2.0):
    """A 3-point open arrowhead at ~2/3 along the polyline, pointing in the
    digitisation direction of the edge id (the direction the vertex order encodes).

    The one-way badge says an edge is one-way but not which way its id points; for
    the parallel-carriageway / reversed-twin relabelling decisions that direction is
    the deciding fact, so the gt and matched edges get an explicit arrowhead.
    Returns folium-ordered [(lat, lon), ...] or None."""
    pts = np.asarray(P, dtype=float)
    if len(pts) < 2:
        return None
    seg = np.hypot(np.diff(pts[:, 1]), np.diff(pts[:, 0]))       # (dlon, dlat) approx
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return None
    i = int(np.searchsorted(cum, 0.66 * cum[-1]))
    i = max(1, min(i, len(pts) - 1))
    vx, vy = pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0]  # (dlon, dlat)
    n = math.hypot(vx, vy)
    if n == 0:
        return None
    ux, uy = vx / n, vy / n
    L = 12.0 / M_PER_DEG                                           # 12 m arrowhead
    tip = (float(pts[i][0]), float(pts[i][1]))
    wings = []
    # wings sit BEHIND the tip, splayed +-30 deg off the reverse direction:
    #   wing = tip - L*R(+-30 deg)*u
    # The first version used +-150 deg, whose wings land AHEAD of the tip (R(150) =
    # -R(-30)), drawing every arrowhead backwards -- caught by the 2026-10-13
    # verification pass on 494/494 rendered arrows; 200/247 gt edges are one-way,
    # so that inversion flipped the page's only direction cue.
    for ang in (30.0, -30.0):
        a = math.radians(ang)
        wx = math.cos(a) * ux - math.sin(a) * uy
        wy = math.sin(a) * ux + math.cos(a) * uy
        wings.append((tip[0] - L * wy, tip[1] - L * wx))
    # sanity: both wings must lie behind the tip along u
    assert all((w[1] - tip[1]) * ux + (w[0] - tip[0]) * uy < 0 for w in wings), \
        'arrowhead wing placed ahead of the tip'
    return [wings[0], tip, wings[1]]


def panel_html(rec, prev_href, next_href):
    g, m = rec['gt'], rec['m']
    m_val = f"<b>{m}</b>"
    if rec.get('m_raw') not in (None, m):
        m_val += (f" <span style='color:#777'>(run cpath {rec['m_raw']} &mdash; reverse "
                  f"twin, drawn geometry identical)</span>")
    # the matched edge's attributes matter as much as the gt's (name/oneway/length);
    # the twin-note rewrite of 2026-10-13 dropped them from this row and the
    # verification pass caught the asymmetry against the gt row
    m_val += f"<br><span style='color:#555'>{rec['m_attr'] or '&mdash;'}</span>"
    rows = [
        ('traj / uni_seq', f"{rec['tid']} / {rec['seq']}"),
        ('timestamp', f"{int(rec['ts'])}"),
        (f"<span style='color:{C['gt']}'>&#9632;</span> gt_edge <b>{g}</b>",
         rec['gt_attr'] or '&mdash;'),
        (f"<span style='color:{C['matched']}'>&#9632;</span> matched", m_val),
        ('RTK&rarr;gt / &rarr;matched' if rec.get('q') in ('4', None)
         else 'truth&rarr;gt / &rarr;matched (not fixed)',
         f"{rec['d_rtk_gt']:.1f} / {rec['d_rtk_m']:.1f} m" if rec['d_rtk_gt'] is not None
         else 'no RTK coords'),
    ]
    if rec.get('q') is not None:
        qcolors = {'4': '#006400', '5': '#B8860B', '1': '#B00020', '2': '#B00020'}
        qt = f"<b style='color:{qcolors.get(str(rec['q']), '#555')}'>{rec['q']}" + \
             {'4': ' (RTK fixed)', '5': ' (float &mdash; decimetre, not fixed)',
              '1': ' (autonomous &mdash; NOT RTK)',
              '2': ' (DGPS &mdash; NOT RTK)'}.get(str(rec['q']), '') + '</b>'
        rows.append(('RTK fix quality', qt))
    rows += [
        ('SPP&rarr;gt / &rarr;matched', f"{rec['d_obs_gt']:.1f} / {rec['d_obs_m']:.1f} m"),
        ('tw alpha / alpha-beta', f"{rec['tw_a']:.3f} / {rec['tw_ab']:.3f}"),
        ('candidates in disk', str(rec['n_cand'])),
    ]
    trs = ''.join(f"<tr><td style='padding-right:8px;color:#555'>{k}</td>"
                  f"<td>{val}</td></tr>" for k, val in rows)
    nav = []
    if prev_href:
        nav.append(f"<a href='{prev_href}'>prev</a>")
    if next_href:
        nav.append(f"<a href='{next_href}'>next</a>")
    nav.append("<a href='index.html'>index</a>")
    banner = ''
    if rec.get('suspect'):
        banner = ("<div style='margin-bottom:5px;color:#B00020;font-weight:bold'>"
                  "&#9888; LABEL SUSPECT &mdash; the RTK point is &gt;5 m closer to the "
                  "matched edge than to the labelled one</div>")
    return (f"<div style='position:fixed;top:10px;right:10px;z-index:9999;"
            f"background:rgba(255,255,255,0.95);padding:9px 11px;border:1px solid #888;"
            f"border-radius:4px;font-size:12px;max-width:360px;font-family:sans-serif'>"
            f"{banner}<table style='border-collapse:collapse'>{trs}</table>"
            f"<div style='margin-top:6px'>{' &middot; '.join(nav)}</div>"
            f"<div style='margin-top:5px;color:#666;font-size:11px'>"
            f"purple = labelled edge &middot; blue = matched &middot; orange = candidates "
            f"&middot; red = SPP &middot; yellow = edge under the truth point<br>"
            f"large circle = truth point, colour = fix quality (green fixed &middot; "
            f"orange float &middot; red not fixed &middot; grey unknown); small arrows = "
            f"the direction the edge id points<br>"
            f"candidates within ~10 m overlap at this zoom &mdash; zoom in for per-dot "
            f"clicks; coincident ids share one dot<br>"
            f"distances are the project's degree-space convention (degrees &times; 111320; "
            f"E&ndash;W separations are overstated by up to ~6% at this latitude)"
            f"</div></div>")


def popup(title, lines):
    body = ''.join(f"<div>{l}</div>" for l in lines)
    return folium.Popup(f"<div style='font-family:sans-serif;font-size:12px'>"
                        f"<b>{title}</b>{body}</div>", max_width=340)


def main():
    poly, meta = load_edges()
    a, grid, _ = CM.load_segments()          # segment arrays + grid, matcher's predicate
    # CM arrays are lon/lat; keep aligned indexing
    aligned = {}
    with open(R.ALIGNED, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            if r['id'].strip() in v.EXCLUDE_TRAJ:
                continue
            aligned[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r
    run = R.load_run(R.OUTDIR / 'cmm.csv')

    recs = {}
    with open(CSV_IN, newline='') as f:
        for r in csv.DictReader(f):
            recs.setdefault(r['traj'], []).append(
                dict(tid=r['traj'], seq=int(r['uni_seq']), ts=float(r['timestamp']),
                     gt=int(r['gt_edge']), m=int(r['matched_edge']),
                     suspect=int(r['label_suspect']),
                     d_rtk_gt=float(r['d_rtk_to_label_m']) if r['d_rtk_to_label_m'] else None,
                     d_rtk_m=float(r['d_rtk_to_matched_m']) if r['d_rtk_to_matched_m'] else None,
                     d_obs_gt=float(r['d_spp_to_label_m']), d_obs_m=float(r['d_spp_to_matched_m'])))

    # move the old PNG companions into their folders
    for tid in recs:
        d = ROOT / f'traj{tid}'
        d.mkdir(parents=True, exist_ok=True)
        png = ROOT / f'traj{tid}.png'
        if png.exists():
            png.rename(d / png.name)

    ctx_by_traj = {}
    for k2, r2 in aligned.items():
        try:
            ctx_by_traj.setdefault(k2[0], []).append(
                (v.norm_ts(r2['timestamp']), float(r2['obs_x']), float(r2['obs_y'])))
        except ValueError:
            continue
    for lst in ctx_by_traj.values():
        lst.sort()

    root_links = []
    for tid in sorted(recs):
        folder = ROOT / f'traj{tid}'
        by_ts = sorted(recs[tid], key=lambda r: r['ts'])
        pages = []
        qmap = load_rtk_quality(tid)
        if not qmap:
            print(f'WARNING traj{tid}: RTK quality map is empty (file missing or '
                  f'unreadable) -- stars fall back to grey/unknown')
        for i, rec in enumerate(by_ts):
            key = (tid, rec['ts'])
            al = aligned[key]
            row = run[key]
            g, m = rec['gt'], rec['m']
            m_raw_s = row['cpath'].strip()
            rec['m_raw'] = int(m_raw_s) if m_raw_s else None
            rec['gt_attr'] = edge_attr(g, meta)
            rec['m_attr'] = edge_attr(m, meta)
            rec['tw_a'] = float(row['trustworthiness'])
            rec['tw_ab'] = float(row['trustworthiness_smoothed'])
            ox, oy = float(al['obs_x']), float(al['obs_y'])
            try:
                tx, ty = float(al['gt_x']), float(al['gt_y'])
            except ValueError:
                tx = ty = None
            rec['q'] = rtk_quality(qmap, rec['ts']) if tx is not None else None
            if rec['q'] == '':
                rec['q'] = None

            # candidates at this epoch, matcher's predicate
            fids, dists = CM.min_dist_per_fid(a, grid, ox, oy)
            cand = [(int(fd), float(dd) * M_PER_DEG)
                    for fd, dd in zip(fids, dists) if dd <= CM.RADIUS]
            cand.sort(key=lambda t: t[1])
            rec['n_cand'] = len(cand)

            m_f = folium.Map(location=[oy, ox], zoom_start=18, tiles=None,
                             control_scale=True)
            folium.TileLayer(tiles=ESRI, attr=ESRI_ATTR, name='satellite',
                             max_zoom=19).add_to(m_f)

            # context trajectory: +-40 alignment rows around the epoch
            ctx = ctx_by_traj[tid]
            idx = [j for j, (t2, _, _) in enumerate(ctx) if t2 == rec['ts']][0]
            win = ctx[max(0, idx - 40): idx + 41]
            folium.PolyLine([(y2, x2) for _, x2, y2 in win], color='white', weight=2,
                            opacity=0.85, tooltip='trajectory context (SPP fixes)').add_to(m_f)

            # candidate edges (orange); the POINT markers are drawn after the gt and
            # matched polylines, at the end of this block, so that a candidate point
            # lying on the gt/matched edge stays clickable (weight-5 lines drawn later
            # would otherwise intercept the click -- found by the 2026-10-13 E2E test:
            # a real click at a candidate point sitting on the labelled edge opened the
            # labelled-edge popup instead)
            gt_twin_s = v.REV_MAP.get(str(g))
            gt_twin = int(gt_twin_s) if gt_twin_s else None
            cand_pts = []
            for rank, (fid, dm) in enumerate(cand, 1):
                tag = ''
                if fid == g:
                    tag += ' &mdash; GT'
                elif gt_twin is not None and fid == gt_twin:
                    tag += f' &mdash; GT (reverse twin of {g})'
                if fid == m:
                    tag += ' &mdash; MATCHED'
                    if rec['m_raw'] not in (None, m):
                        tag += f' (reverse twin of run cpath {rec["m_raw"]})'
                elif rec['m_raw'] is not None and fid == rec['m_raw']:
                    tag += f' &mdash; run cpath {rec["m_raw"]} (page draws its twin {m})'
                kept = 'kept (rank &le; 16)' if rank <= 16 else 'dropped by k=16 cap'
                brg = bearing_at(poly[fid], oy, ox)
                folium.PolyLine([p for p in poly[fid]], color=C['cand'], weight=1.6,
                                opacity=0.85,
                                tooltip=f'candidate edge {fid}',
                                popup=popup(f'candidate edge {fid}',
                                            [f'name/highway: {edge_attr(fid, meta) or "&mdash;"}',
                                             f'fix&rarr;edge {dm:.1f} m{tag}',
                                             f'id direction at this point: bearing '
                                             f'{brg:.0f}&deg; (0=N, clockwise)'])).add_to(m_f)
                cx, cy, _ = project_point(ox, oy, poly[fid])
                cand_pts.append((rank, fid, dm, tag, kept, cy, cx, brg))

            # labelled gt edge, then matched edge
            folium.PolyLine([p for p in poly[g]], color=C['gt'], weight=5, opacity=0.9,
                            popup=popup(f'LABELLED (ground-truth) edge {g}',
                                        [f'name/highway: {edge_attr(g, meta) or "&mdash;"}',
                                         ('RTK&rarr;edge' if rec.get('q') in ('4', None)
                                          else 'truth point&rarr;edge')
                                         + f' {rec["d_rtk_gt"]:.1f} m'
                                         if rec['d_rtk_gt'] is not None
                                         else 'no RTK coords']),
                            tooltip=f'labelled edge {g}').add_to(m_f)
            ar = direction_arrow(poly[g], C['gt'])
            if ar:
                folium.PolyLine(ar, color=C['gt'], weight=2.4,
                                tooltip='direction of the labelled edge id').add_to(m_f)
            if m != g:
                m_lines = [f'name/highway: {edge_attr(m, meta) or "&mdash;"}',
                           f'tw alpha {rec["tw_a"]:.3f}, '
                           f'alpha-beta {rec["tw_ab"]:.3f}',
                           f'SPP&rarr;edge {rec["d_obs_m"]:.1f} m']
                if rec['m_raw'] not in (None, m):
                    m_lines.append(f'run cpath was <b>{rec["m_raw"]}</b>; this page draws '
                                   f'its reverse twin {m} (identical geometry)')
                folium.PolyLine([p for p in poly[m]], color=C['matched'], weight=5,
                                opacity=0.9,
                                popup=popup(f'MATCHED edge {m}', m_lines),
                                tooltip=f'matched edge {m}').add_to(m_f)
                arm = direction_arrow(poly[m], C['matched'])
                if arm:
                    folium.PolyLine(arm, color=C['matched'], weight=2.4,
                                    tooltip='direction of the matched edge id').add_to(m_f)

            # SPP fix of the epoch. Drawn BELOW the candidate points: a candidate
            # often projects within a metre of the fix, and with the fix on top its
            # 6-px circle shadowed the candidate at exactly the pixel the user clicks
            # (2026-10-13 verification note); the fix keeps its own full popup at the
            # rim and after zoom.
            dpre = 'RTK' if rec.get('q') in ('4', None) else 'truth point'
            fix_lines = [f'fix {ox:.6f}, {oy:.6f}',
                         f'matched edge <b>{m}</b> &mdash; tw alpha '
                         f'{rec["tw_a"]:.2f}, alpha-beta {rec["tw_ab"]:.2f}',
                         f'gt edge <b>{g}</b>',
                         f'candidates in disk: {len(cand)}',
                         f'{dpre}&rarr;gt {rec["d_rtk_gt"]:.1f} m, {dpre}&rarr;matched '
                         f'{rec["d_rtk_m"]:.1f} m'
                         if rec['d_rtk_gt'] is not None else 'no RTK coords for this epoch']
            folium.CircleMarker([oy, ox], radius=6, color='black', weight=1, fill=True,
                                fill_color=C['obs'], fill_opacity=1.0,
                                popup=popup(f'SPP fix, epoch {int(rec["ts"])}', fix_lines),
                                tooltip=f'SPP fix ({int(rec["ts"])})').add_to(m_f)

            # the truth circle and the yellow relabel line are polyline/furniture
            # layers: drawn BEFORE the candidate point dots, which must stay the
            # topmost SVG layer or their centres become unclickable (round-5
            # E2E: the yellow line's stroke and the r=8 truth circle intercepted
            # every click in a dense candidate cluster)
            if tx is not None:
                # truth point as a COMPACT circle, not the 35x45-px awesome-marker
                # star: at the full-disk default zoom (fitBounds, ~1.1 m/px) that
                # star was a ~39x50 m div shadowing every candidate dot near the fix
                # (2026-10-13 round-5 verification: per-dot click precision broke).
                # Colour carries the GGA fix quality.
                qhex = {'4': '#00A651', '5': '#FF8C00', '1': '#C0392B',
                        '2': '#C0392B'}.get(str(rec['q']), '#808080')
                qline = (f'NMEA fix quality {rec["q"]}'
                         + {'4': ' (RTK fixed)', '5': ' (float)', '1': ' (autonomous)',
                            '2': ' (DGPS)'}.get(str(rec['q']), '')) \
                    if rec['q'] is not None else 'fix quality unknown (grey)'
                qlines = [f'{tx:.6f}, {ty:.6f}', qline]
                if rec['q'] not in ('4', None):
                    qlines.append('<b style="color:#B00020">NOT a fixed solution '
                                  '&mdash; treat this point as a cautionary '
                                  'reference, not as cm-level truth</b>')
                folium.CircleMarker([ty, tx], radius=8, color='black', weight=1.5,
                                    fill=True, fill_color=qhex, fill_opacity=1.0,
                                    popup=popup('Ground-truth point (receiver solution)',
                                                qlines),
                                    tooltip=f'ground truth (q={rec["q"]})').add_to(m_f)
                # relabel-target affordance: the edge under the truth point may be
                # neither the label nor the match (the both-wrong case)
                tf, td = CM.min_dist_per_fid(a, grid, tx, ty)
                if len(tf):
                    j = int(np.argmin(td))
                    nf, nd = int(tf[j]), float(td[j]) * M_PER_DEG
                    if nd <= 60.0 and nf not in twin_closure(g, m, rec['m_raw']):
                        folium.PolyLine([p for p in poly[nf]], color='#FFD700',
                                        weight=3, dash_array='6,4',
                                        popup=popup(
                                            f'nearest edge to the ground-truth point '
                                            f'(edge {nf})',
                                            [f'name/highway: '
                                             f'{edge_attr(nf, meta) or "&mdash;"}',
                                             f'truth point&rarr;edge {nd:.1f} m',
                                             'the relabel candidate when the current '
                                             'gt_edge is wrong']),
                                        tooltip=f'nearest to truth point ({nf})'
                                        ).add_to(m_f)
                        arn = direction_arrow(poly[nf], '#FFD700', weight=2.0)
                        if arn:
                            folium.PolyLine(arn, color='#FFD700', weight=2.0,
                                            tooltip='direction of the relabel-target '
                                                    'id').add_to(m_f)

            # candidate POINT markers, on top of every polyline (and the fix) so each
            # is clickable at its exact pixel. Candidates whose projections coincide
            # to 1e-5 deg (reverse twins project to the SAME point -- identical
            # geometry reversed) merge into ONE dot whose popup lists every id at
            # that point; two stacked dots would leave only the top one clickable.
            groups = {}
            for cp in cand_pts:
                groups.setdefault((round(cp[5], 5), round(cp[6], 5)), []).append(cp)
            for (ylat, xlon), grp in groups.items():
                if len(grp) == 1:
                    rank, fid, dm, tag, kept, _, _, brg = grp[0]
                    popup_obj = popup(f'candidate point {fid}',
                                      [f'edge id <b>{fid}</b>',
                                       f'name/highway: {edge_attr(fid, meta) or "&mdash;"}',
                                       f'fix&rarr;edge {dm:.1f} m',
                                       f'id direction at this point: bearing '
                                       f'{brg:.0f}&deg; (0=N, clockwise)',
                                       f'distance rank {rank}/{len(cand)} &mdash; '
                                       f'{kept} (equal distances ordered by id){tag}'])
                    tip = f'cand {fid} ({dm:.1f} m)'
                else:
                    fids_s = ' / '.join(str(cp[1]) for cp in grp)
                    lines = [f'<b>{len(grp)} ids project to this same point</b>']
                    for rank, fid, dm, tag, kept, _, _, brg in grp:
                        lines.append(f'edge id <b>{fid}</b> &mdash; fix&rarr;edge '
                                     f'{dm:.1f} m; {edge_attr(fid, meta) or "&mdash;"}; '
                                     f'id direction at this point: bearing '
                                     f'{brg:.0f}&deg; (0=N, clockwise); '
                                     f'rank {rank}/{len(cand)} ({kept}){tag}')
                    popup_obj = popup(f'candidate point(s) {fids_s}', lines)
                    tip = f'cands {fids_s}'
                folium.CircleMarker([ylat, xlon], radius=5, color='black', weight=1,
                                    fill=True, fill_color=C['cand'], fill_opacity=0.95,
                                    popup=popup_obj, tooltip=tip).add_to(m_f)
            # open on the whole search disk: on 63 pages the zoom-18 default clipped
            # far candidates (up to 300 m) out of the initial viewport (2026-10-13
            # verification note); the disk IS the page's core evidence
            m_f.fit_bounds([[oy - 1.06 * CM.RADIUS, ox - 1.06 * CM.RADIUS],
                            [oy + 1.06 * CM.RADIUS, ox + 1.06 * CM.RADIUS]])

            href = f'err_ts{int(rec["ts"])}.html'
            prev_href = pages[i - 1][1] if i > 0 else None
            next_href = f'err_ts{int(by_ts[i + 1]["ts"])}.html' if i < len(by_ts) - 1 else None
            m_f.get_root().html.add_child(Element(panel_html(rec, prev_href, next_href)))
            (folder / href).write_text(m_f.get_root().render(), encoding='utf-8')
            pages.append((rec, href))
        print(f'traj{tid}: {len(pages)} pages')

        # ---- trajectory index ----
        oy0 = sum(float(aligned[(tid, r['ts'])]['obs_y']) for r, _ in pages) / len(pages)
        ox0 = sum(float(aligned[(tid, r['ts'])]['obs_x']) for r, _ in pages) / len(pages)
        mi = folium.Map(location=[oy0, ox0], zoom_start=13, tiles=None, control_scale=True)
        folium.TileLayer(tiles=ESRI, attr=ESRI_ATTR, name='satellite',
                         max_native_zoom=19, max_zoom=20).add_to(mi)
        tab = []
        for rec, href in pages:
            ox, oy = float(aligned[(tid, rec['ts'])]['obs_x']), \
                float(aligned[(tid, rec['ts'])]['obs_y'])
            folium.CircleMarker([oy, ox], radius=6, color='black', weight=1, fill=True,
                                fill_color=C['obs'], fill_opacity=0.9,
                                popup=folium.Popup(
                                    f"<b>{int(rec['ts'])}</b> (seq {rec['seq']})<br>"
                                    f"gt {rec['gt']} vs matched {rec['m']}<br>"
                                    f"<a href='{href}'>open page</a>", max_width=260),
                                tooltip=str(int(rec['ts']))).add_to(mi)
            tab.append(f"<tr><td>{int(rec['ts'])}</td><td>{rec['seq']}</td>"
                       f"<td>{rec['gt']}</td><td>{rec['m']}</td>"
                       f"<td><a href='{href}'>open</a></td></tr>")
        table = ("<div style='position:fixed;top:10px;right:10px;z-index:9999;"
                 "background:rgba(255,255,255,0.95);padding:8px 10px;border:1px solid "
                 "#888;border-radius:4px;font-size:12px;max-height:80vh;overflow:auto;"
                 "font-family:sans-serif'><b>Trajectory " + tid + " &mdash; wrong epochs"
                 "</b><table style='border-collapse:collapse'><tr style='color:#555'>"
                 "<td>ts</td><td>seq</td><td>gt</td><td>matched</td><td></td></tr>"
                 + ''.join(tab) + "</table></div>")
        mi.get_root().html.add_child(Element(table))
        (folder / 'index.html').write_text(mi.get_root().render(), encoding='utf-8')
        root_links.append(f"<li><a href='traj{tid}/index.html'>traj{tid}</a> "
                          f"&mdash; {len(pages)} wrong epochs</li>")

    (ROOT / 'index.html').write_text(
        "<html><body style='font-family:sans-serif'><h3>Wrong-match audit "
        "(one page per wrong epoch, clickable ids)</h3><ul>"
        + ''.join(root_links) + "</ul></body></html>", encoding='utf-8')
    print(f'wrote {ROOT}/index.html and per-trajectory folders')


if __name__ == '__main__':
    main()
