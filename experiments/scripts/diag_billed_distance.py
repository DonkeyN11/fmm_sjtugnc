#!/usr/bin/env python3
"""D11-R (manuscript R09): billing weight as the along-path increment, not the edge length.

WHAT WAS WRONG IN THE FIRST VERSION. B_t was the length of the matched edge. At 1 Hz an
epoch advances 10-30 m, while a matched edge is 143-10,138 m long, so the same edge was
charged once per epoch it hosted: sum B came to 46,200 km against ~200 km actually driven,
and the resulting "mis-billed distance" exceeded the total distance. B_t must be the
progress along the matched path between consecutive epochs.

DEFINITION USED HERE.
  s_t       arc-length offset of the matched point pgeom_t on its matched edge cpath_t
            (nearest polyline segment, cumulative metres from the edge start)
  same edge, |ds| plausible   B_t = |s_t - s_{t-1}|
  different edge, plausible   B_t = (L_{t-1} - s_{t-1}) + s_t   (rest of the old edge plus
                              what the new edge has already carried)
  degenerate                  B_t = ||z_t - z_{t-1}|| (the GNSS displacement): a time gap
                              over 5 s, a missing edge, or a projection jump whose implied
                              length is implausible against the GNSS displacement
Plausibility bound for both computed rules: d_gnss/2 <= B <= 4*d_gnss + 50 m. Every rule's
count and its share of sum B are reported, so the fallback cannot hide.

HARD CHECK (aborts if it fails). Sum of B over all 16,155 epochs must be of the order of the
driven distance: 16,154 s at 1 Hz, six trajectories, 269.3 min. A value in the 10^4 km range
would mean the definition is still wrong and the script stops.

BIAS. With c_t the correctness of the matched edge and tw_t the score,
  M = sum B_t (1 - c_t),  Mhat = sum B_t (1 - tw_t),  bias = Mhat - M = sum B_t (c_t - tw_t).
A cluster bootstrap over the six trajectories (whole trajectories redrawn with replacement,
2,000 replicates, fixed seed) gives the interval: one realisation's total is not a result.

Usage: python3 experiments/scripts/diag_billed_distance.py
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from osgeo import ogr

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R      # noqa: E402
import verify_paper_numbers as v   # noqa: E402

DIAG2 = BASE / 'data/archive_20261019/diag2'
RUN = BASE / 'data/archive_20260929/scalar'
SHP = BASE / 'input/map/hainan/edges.shp'
GAP_S = 5.0
SEED = 20261019
REPS = 2000
MAX_TOTAL_M = 1.0e6          # abort threshold for the sanity check


def parse_xy(s):
    i, j = s.find('('), s.find(')')
    if i < 0 or j < 0:
        return None
    p = s[i + 1:j].split()
    if len(p) < 2:
        return None
    try:
        return float(p[0].rstrip(',')), float(p[1].rstrip(','))
    except ValueError:
        return None


def load_geometry(fids):
    ds = ogr.Open(str(SHP))
    ly = ds.GetLayer(0)
    out = {}
    for f in ly:
        fid = f.GetFID()
        if fid not in fids:
            continue
        g = f.GetGeometryRef()
        parts = ([g] if g.GetGeometryName() == 'LINESTRING'
                 else [g.GetGeometryRef(k) for k in range(g.GetGeometryCount())])
        pts = []
        for p in parts:
            pts.extend((p.GetX(i), p.GetY(i)) for i in range(p.GetPointCount()))
        if len(pts) >= 2:
            out[fid] = pts
    return out


def offset_of(pts, px, py):
    """(arc-length offset of the projection, total polyline length) in metres."""
    k = math.cos(math.radians(py))
    cum = 0.0
    best_d2, best_s = float('inf'), 0.0
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg = v.haversine_m(ax, ay, bx, by)
        axk, bxk, pxk = ax * k, bx * k, px * k
        vx, vy = bxk - axk, by - ay
        wx, wy = pxk - axk, py - ay
        l2 = vx * vx + vy * vy
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / l2)) if l2 > 0 else 0.0
        d2 = (wx - t * vx) ** 2 + (wy - t * vy) ** 2
        if d2 < best_d2:
            best_d2, best_s = d2, cum + t * seg
        cum += seg
    return best_s, cum


def main():
    gt, _ts = R.load_gt()
    rows = list(csv.DictReader(open(RUN / 'cmm.csv', newline=''), delimiter=';'))
    vit = {(r['id'].strip(), v.norm_ts(r['timestamp'])): r
           for r in csv.DictReader(open(RUN / 'cmm_viterbi.csv', newline=''), delimiter=';')}
    fids = {int(r['cpath']) for r in rows if r['cpath'].strip().isdigit()} - {0}
    print(f'epochs={len(rows)}  distinct matched edges={len(fids)}', flush=True)
    geo = load_geometry(fids)
    print(f'polylines loaded={len(geo)}', flush=True)

    ep = []
    for r in rows:
        tid = r['id'].strip()
        ts = v.norm_ts(r['timestamp'])
        fid = int(r['cpath']) if r['cpath'].strip().isdigit() else 0
        p = parse_xy(r['pgeom'])
        o = parse_xy(r['ogeom'])
        s_off = L = None
        if fid in geo and p:
            s_off, L = offset_of(geo[fid], p[0], p[1])
        ep.append(dict(tid=tid, ts=ts, fid=fid, s=s_off, L=L,
                       ox=None if o is None else o[0], oy=None if o is None else o[1],
                       row=r))
    by_traj = defaultdict(list)
    for e in ep:
        by_traj[e['tid']].append(e)

    rule_ct = Counter()
    rule_len = defaultdict(float)
    recs = []
    for tid, es in by_traj.items():
        es.sort(key=lambda z: z['ts'])
        for i, e in enumerate(es):
            if i == 0:
                rule_ct['first_epoch'] += 1
                continue
            p = es[i - 1]
            d_gnss = float('nan')
            if None not in (e['ox'], e['oy'], p['ox'], p['oy']):
                d_gnss = v.haversine_m(p['ox'], p['oy'], e['ox'], e['oy'])
            B, rule = float('nan'), 'none'
            if e['s'] is None or p['s'] is None or not (d_gnss == d_gnss):
                B, rule = d_gnss, 'no_geometry'
            elif e['ts'] - p['ts'] > GAP_S:
                B, rule = d_gnss, 'gap_dgnss'
            elif e['fid'] == p['fid']:
                cand = abs(e['s'] - p['s'])
                if d_gnss / 2 <= cand <= 4 * d_gnss + 50:
                    B, rule = cand, 'same_edge'
                else:
                    B, rule = d_gnss, 'same_edge_implausible'
            else:
                cand = (p['L'] - p['s']) + e['s']
                if d_gnss / 2 <= cand <= 4 * d_gnss + 50:
                    B, rule = cand, 'cross_edge'
                else:
                    B, rule = d_gnss, 'cross_edge_implausible'
            if not (B == B) or B < 0:
                rule_ct['missing'] += 1
                continue
            rule_ct[rule] += 1
            rule_len[rule] += B
            key = (tid, e['ts'])
            ge = gt.get(key)
            correct = None if ge is None else v.is_edge_match(str(e['fid']), ge[0])
            try:
                tw_ab = float(e['row']['trustworthiness_smoothed'])
            except (KeyError, ValueError):
                tw_ab = float('nan')
            vr = vit.get(key)
            try:
                tw_d = float(vr['trustworthiness']) if vr else float('nan')
            except (KeyError, ValueError):
                tw_d = float('nan')
            recs.append(dict(traj=tid, ts=e['ts'], fid=e['fid'], rule=rule,
                             B_m=B, d_gnss_m=d_gnss,
                             edge_len_m=e['L'] if e['L'] is not None else float('nan'),
                             s_m=e['s'] if e['s'] is not None else float('nan'),
                             correct='' if correct is None else int(correct),
                             tw_alphabeta=tw_ab, tw_delta=tw_d))
    total_B = sum(r['B_m'] for r in recs)
    dur = sum(max(e['ts'] for e in es) - min(e['ts'] for e in es) for es in by_traj.values())
    print(f'SUM_B over all epochs = {total_B:.1f} m = {total_B/1000:.1f} km'
          f'   span={dur/60:.1f} min   mean speed={total_B/dur:.2f} m/s')
    print('rules: ' + ', '.join(f'{k}={rule_ct[k]} ({rule_len[k]/1000:.1f} km)'
                                for k in sorted(rule_ct)))
    if total_B > MAX_TOTAL_M:
        print('ABORT: sum B is in the 10^4 km range -- the definition is still wrong.')
        return 2

    DIAG2.mkdir(parents=True, exist_ok=True)
    with open(DIAG2 / 'billed_epochs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    lab = [r for r in recs if r['correct'] != '']
    tids = sorted({r['traj'] for r in lab})

    def arm(col):
        use = [r for r in lab if r[col] == r[col]]
        B = np.array([r['B_m'] for r in use])
        c = np.array([r['correct'] for r in use], float)
        s = np.array([r[col] for r in use])
        per = {}
        for t in tids:
            m = np.array([r['traj'] for r in use]) == t
            if m.any():
                per[t] = dict(n=int(m.sum()), B_m=float(B[m].sum()),
                              M_m=float((B[m] * (1 - c[m])).sum()),
                              Mhat_m=float((B[m] * (1 - s[m])).sum()))
                per[t]['bias_m'] = per[t]['Mhat_m'] - per[t]['M_m']
        M = float((B * (1 - c)).sum())
        Mh = float((B * (1 - s)).sum())
        # cluster bootstrap over whole trajectories
        rng = np.random.default_rng(SEED)
        tb = np.array([per[t]['B_m'] if t in per else 0.0 for t in tids])
        tm = np.array([per[t]['M_m'] if t in per else 0.0 for t in tids])
        th = np.array([per[t]['Mhat_m'] if t in per else 0.0 for t in tids])
        reps = np.empty(REPS)
        rel = np.empty(REPS)
        for k in range(REPS):
            idx = rng.integers(0, len(tids), len(tids))
            reps[k] = th[idx].sum() - tm[idx].sum()
            rel[k] = reps[k] / tb[idx].sum()
        ci = np.percentile(reps, [2.5, 97.5])
        cirel = np.percentile(rel, [2.5, 97.5])
        return dict(score_col=col, n=int(len(use)), sum_B_m=float(B.sum()),
                    M_m=M, Mhat_m=Mh, bias_m=Mh - M,
                    bias_pct_of_M=100.0 * (Mh - M) / M if M else float('nan'),
                    boot_mean_m=float(reps.mean()), ci95_m=[float(ci[0]), float(ci[1])],
                    ci95_pct=[float(cirel[0] * 100), float(cirel[1] * 100)],
                    n_boot=REPS, seed=SEED, per_traj=per,
                    frac_ci_excludes_zero=bool(ci[0] > 0 or ci[1] < 0))

    out = dict(sum_B_all_epochs_m=total_B, span_min=dur / 60.0,
               mean_speed_mps=total_B / dur, rules={k: [rule_ct[k], rule_len[k]]
                                                    for k in rule_ct},
               n_labeled=len(lab), arms={a: arm(f'tw_{a}') for a in ('alphabeta', 'delta')})
    (DIAG2 / 'billed_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # --- strata: matched-edge length (relabelled) and the increment itself ---
    def strata(key_fn, labels, fname, unit):
        lab2 = [r for r in lab if r['tw_alphabeta'] == r['tw_alphabeta']]
        vals = np.array([key_fn(r) for r in lab2])
        qs = np.quantile(vals, np.linspace(0.1, 0.9, 9))
        idx = np.digitize(vals, qs)
        rowsout = []
        tot_bias = sum(r['B_m'] * (r['correct'] - r['tw_alphabeta']) for r in lab2)
        for g in range(10):
            m = idx == g
            if not m.any():
                continue
            sub = [r for r, mm in zip(lab2, m) if mm]
            B = np.array([r['B_m'] for r in sub])
            c = np.array([r['correct'] for r in sub], float)
            s = np.array([r['tw_alphabeta'] for r in sub])
            rowsout.append(dict(decile=g, n=int(m.sum()), share=float(m.mean()),
                                key_mean=float(vals[m].mean()),
                                key_min=float(vals[m].min()), key_max=float(vals[m].max()),
                                B_mean_m=float(B.mean()), mean_tw=float(s.mean()),
                                acc=float(c.mean()), gap=float(s.mean() - c.mean()),
                                M_m=float((B * (1 - c)).sum()),
                                Mhat_m=float((B * (1 - s)).sum()),
                                bias_m=float((B * (c - s)).sum()),
                                share_of_bias=float((B * (c - s)).sum() / tot_bias)
                                if tot_bias else float('nan')))
        with open(DIAG2 / fname, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rowsout[0]))
            w.writeheader()
            w.writerows(rowsout)
        print(f'-- strata by {labels} ({unit}) -> {fname}')
        print(f'{"dec":>3}{"n":>7}{"key_mean":>10}{"B_mean":>9}{"mean_tw":>9}'
              f'{"acc":>9}{"gap":>10}{"bias_m":>11}{"bias%":>8}')
        for r in rowsout:
            print(f'{r["decile"]:>3}{r["n"]:>7}{r["key_mean"]:>10.1f}{r["B_mean_m"]:>9.2f}'
                  f'{r["mean_tw"]:>9.5f}{r["acc"]:>9.5f}{r["gap"]:>10.5f}'
                  f'{r["bias_m"]:>11.1f}{100*r["share_of_bias"]:>8.2f}')
        return rowsout

    strata(lambda r: r['edge_len_m'], 'matched-edge length', 'strata_matched_edge.csv', 'm')
    strata(lambda r: r['B_m'], 'along-path increment', 'strata_increment.csv', 'm')

    for a, d in out['arms'].items():
        print(f'== {a}: n={d["n"]}  sum_B={d["sum_B_m"]:.1f} m')
        print(f'   M={d["M_m"]:.1f} m   Mhat={d["Mhat_m"]:.1f} m   bias={d["bias_m"]:+.1f} m'
              f' ({d["bias_pct_of_M"]:+.3f} % of M)')
        print(f'   cluster 95% CI: [{d["ci95_m"][0]:+.1f}, {d["ci95_m"][1]:+.1f}] m'
              f'  = [{d["ci95_pct"][0]:+.3f}, {d["ci95_pct"][1]:+.3f}] %'
              f'   excludes_zero={d["frac_ci_excludes_zero"]}')
        for t, pd in d['per_traj'].items():
            print(f'     traj {t}: n={pd["n"]:>5}  B={pd["B_m"]:>9.0f} m'
                  f'  M={pd["M_m"]:>8.1f}  Mhat={pd["Mhat_m"]:>8.1f}  bias={pd["bias_m"]:>+8.1f}')
    print(f'wrote {DIAG2}/billed_summary.json, billed_epochs.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
