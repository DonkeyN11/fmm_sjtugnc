#!/usr/bin/env python3
"""Is the ground truth inside the candidate set? Membership and impact statistics.

WHY THIS EXISTS (question from Donkey.Ning, 2026-10-13). ECE/BSS are computed from one
(score, correctness) pair per audited epoch -- the matched candidate's tw. This script
answers the complementary question the sample-space claim needs: for how many audited
epochs was the ground-truth edge not among the candidates the matcher was given at all?
Such an epoch is wrong for a reason no re-ranking of its candidates can repair (a
sample-space miss, usually a GNSS deviation larger than the disk); an epoch whose truth
was offered and mis-ranked is an ordinary matcher error (recoverable by re-ranking). The
two have different readings and different remedies, and this script counts them and
prices the first on the audit's own metrics.

REPLICATES THE MATCHER'S PREDICATE EXACTLY (src/network/network.cpp:742-760): a candidate
exists iff the point-to-polyline distance from the fix to the edge, computed as RAW
coordinate Euclidean distance (degrees, EPSG:4326), is <= the operating point's radius.
paper1_cmm_scalar.xml sets k_search=0 and delta_road=0.0026949335, so the disk is the
constant 0.0026949335 deg, the same one paper1_fmm_baseline.xml searches. In true metres
at Haikou's latitude that disk is an ellipse -- 300 m north-south, 281 m east-west -- and
the replication deliberately keeps the degree-space rule rather than "correcting" it,
because the question is what the matcher actually saw. The retained set is the KNN cut
(network.cpp:775-782, k=16): min(16, all-within-radius) nearest by the same distance;
min_candidates=1 is inert. The paper's Section V-E note that "the cap binds nowhere in
the range" is a statement about the average (8.82 candidates/epoch); the rank of the gt
edge is reported here so a cap effect would show if it existed.

GT AND JOIN RULES: identical to the evaluators. Evaluable = gt_edge not in {'0','-1',''}
(mark_missing_road_gt.py / verify_paper_numbers.py); traj 12 excluded (EXCLUDE_TRAJ);
labels via verify_paper_numbers.is_edge_match on the joined cpath; CaMM joined on
(id, norm_ts(timestamp)) through real_final_tables.load_run, FMM through seq->timestamp.
The 401 no-candidate epochs carry no label by construction and are outside this
population.

Usage: python3 experiments/scripts/candidate_membership.py
Output: printed summary + data/archive_20261013/candidate_membership.json
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from osgeo import ogr

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import verify_paper_numbers as v        # noqa: E402
import real_final_tables as R           # noqa: E402
import three_estimator_tables as T      # noqa: E402

RADIUS = 0.0026949335                   # degrees; paper1_cmm_scalar.xml delta_road
K_CAP = 16
CELL = 0.005                            # degrees per grid cell (~550 m)
OUT = BASE / 'data/archive_20261013/candidate_membership.json'


def load_segments():
    """All edges.shp segments as arrays, plus a cell -> segment-index grid.

    FID is the edge identity: edges.shp has no `id` attribute, and cpath/gt_edge were
    compared successfully all along, so the FID is the id space both share."""
    ds = ogr.Open(str(BASE / 'input/map/hainan/edges.shp'))
    ly = ds.GetLayer(0)
    n_feat = ly.GetFeatureCount()
    lon0, lat0, lon1, lat1, fid = [], [], [], [], []
    for f in ly:
        g = f.GetGeometryRef()
        parts = ([g] if g.GetGeometryName() == 'LINESTRING'
                 else [g.GetGeometryRef(k) for k in range(g.GetGeometryCount())])
        for p in parts:
            n = p.GetPointCount()
            xs = [p.GetX(i) for i in range(n)]
            ys = [p.GetY(i) for i in range(n)]
            for i in range(n - 1):
                lon0.append(xs[i]); lat0.append(ys[i])
                lon1.append(xs[i + 1]); lat1.append(ys[i + 1])
                fid.append(f.GetFID())
    a = dict(lon0=np.array(lon0), lat0=np.array(lat0),
             lon1=np.array(lon1), lat1=np.array(lat1), fid=np.array(fid))
    # grid: each segment into every cell its bounding box touches
    grid = defaultdict(list)
    i0 = np.floor(np.minimum(a['lon0'], a['lon1']) / CELL).astype(np.int64)
    i1 = np.floor(np.maximum(a['lon0'], a['lon1']) / CELL).astype(np.int64)
    j0 = np.floor(np.minimum(a['lat0'], a['lat1']) / CELL).astype(np.int64)
    j1 = np.floor(np.maximum(a['lat0'], a['lat1']) / CELL).astype(np.int64)
    for k in range(len(a['fid'])):
        for ci in range(i0[k], i1[k] + 1):
            for cj in range(j0[k], j1[k] + 1):
                grid[(ci, cj)].append(k)
    for key in grid:
        grid[key] = np.array(grid[key], dtype=np.int64)
    return a, grid, n_feat


def min_dist_per_fid(a, grid, px, py):
    """(fid array, min point-to-polyline distance array, in degrees)."""
    ci0 = int(np.floor((px - RADIUS) / CELL)); ci1 = int(np.floor((px + RADIUS) / CELL))
    cj0 = int(np.floor((py - RADIUS) / CELL)); cj1 = int(np.floor((py + RADIUS) / CELL))
    idx = [grid[(ci, cj)] for ci in range(ci0, ci1 + 1)
           for cj in range(cj0, cj1 + 1) if (ci, cj) in grid]
    if not idx:
        return np.array([], dtype=np.int64), np.array([])
    idx = np.concatenate(idx)
    x0, y0 = a['lon0'][idx], a['lat0'][idx]
    x1, y1 = a['lon1'][idx], a['lat1'][idx]
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - x0) * dx + (py - y0) * dy) / np.where(L2 > 0, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = x0 + t * dx, y0 + t * dy
    d2 = (px - cx) ** 2 + (py - cy) ** 2
    f = a['fid'][idx]
    order = np.lexsort((d2, f))
    f, d2 = f[order], d2[order]
    first = np.ones(len(f), dtype=bool)
    first[1:] = f[1:] != f[:-1]
    return f[first], np.sqrt(d2[first])


def main():
    # --- fixes and GT from the alignment (the matcher read the table this mirrors) ---
    obs = {}
    n_rows = n_valid = n_invalid = 0
    with open(R.ALIGNED, newline='') as fh:
        for r in csv.DictReader(fh, delimiter=';'):
            tid = r['id'].strip()
            if tid in v.EXCLUDE_TRAJ:
                continue
            n_rows += 1
            key = (tid, v.norm_ts(r['timestamp']))
            try:
                obs[key] = (float(r['obs_x']), float(r['obs_y']))
            except ValueError:
                obs[key] = None             # alignment row without an SPP fix
            if r['gt_edge'].strip() in ('0', '-1', ''):
                n_invalid += 1
            else:
                n_valid += 1
    gt, ts_by_seq = R.load_gt()
    assert n_valid == len(gt), (n_valid, len(gt))

    # --- runs, the manuscript's join (arm definitions from three_estimator_tables) ---
    arms = {}
    for label, fname, col, use_seq in T.REAL_PROPOSED:
        name = label.split(' (')[0]
        arms[name] = (R.load_run(R.OUTDIR / fname, ts_by_seq if use_seq else None), col)

    a, grid, n_feat = load_segments()
    all_fids = set(a['fid'].tolist())

    # --- per-epoch membership ---
    recs = []
    n_no_fix = 0
    for key, (gt_edge, _, _) in gt.items():
        ge = int(gt_edge)
        if obs.get(key) is None:
            n_no_fix += 1
            recs.append(dict(key=key, gt=ge, in_disk=None, rank=None,
                             n_within=0, n_pool=0))
            continue
        px, py = obs[key]
        fids, dists = min_dist_per_fid(a, grid, px, py)
        within = fids[dists <= RADIUS]
        in_disk = bool((within == ge).any())
        rank = None
        d_gt_deg = None
        if in_disk:
            d_gt_deg = float(dists[fids == ge][0])
            rank = int((dists[fids != ge] < d_gt_deg).sum()) + 1
        rec = dict(key=key, gt=ge, in_disk=in_disk, rank=rank,
                   d_gt_deg=d_gt_deg, n_within=int(len(within)), n_pool=int(len(fids)))
        for name, (run, col) in arms.items():
            r = run.get(key)
            s = None
            if r is not None:
                try:
                    s = float(r[col])
                except (ValueError, KeyError):
                    s = None
                if s is not None and not (0.0 <= s <= 1.0):
                    s = None                    # -999 "no value" sentinel
            rec[name] = s
            if r is not None:
                rec[name + ':ok'] = bool(v.is_edge_match(r['cpath'].strip(), gt_edge))
        recs.append(rec)

    ev = [r for r in recs if all(r[k] is not None for k in arms)]
    out = dict(n_rows=n_rows, n_valid=n_valid, n_invalid=n_invalid, n_eval=len(ev),
               n_gt_without_fix=n_no_fix)
    print(f'rows(excl traj12) {n_rows} | valid {n_valid} | invalid {n_invalid} | '
          f'gt-without-fix {n_no_fix} | joined {len(ev)}')

    per_traj = defaultdict(lambda: [0, 0, 0])
    for r in ev:
        t = per_traj[r['key'][0]]
        t[0] += 1; t[1] += int(r['in_disk']); t[2] += int(r['alpha*beta:ok'])
    out['per_traj'] = {}
    print(f"\n{'traj':<6}{'eval':>6}{'truth-in-disk':>15}{'missing':>9}{'wrong':>7}{'acc%':>8}")
    for tid in sorted(per_traj):
        n, nd, ok = per_traj[tid]
        out['per_traj'][tid] = dict(n=n, in_disk=nd, missing=n - nd, wrong=n - ok)
        print(f"{tid:<6}{n:>6}{nd:>15}{n - nd:>9}{n - ok:>7}{100 * ok / n:>8.2f}")

    n_in = sum(1 for r in ev if r['in_disk'])
    out['n_gt_in_disk'] = n_in
    out['n_gt_not_in_disk'] = len(ev) - n_in
    print(f"\ntruth in disk: {n_in}/{len(ev)} = {100 * n_in / len(ev):.3f} %")
    print(f"truth NOT in disk (unrecoverable by re-ranking): {len(ev) - n_in}")

    # cap effect: truth inside the disk but outside the retained k=16
    capped = [r for r in ev if r['in_disk'] and r['rank'] > K_CAP]
    out['n_gt_in_disk_dropped_by_cap'] = len(capped)
    print(f"in disk but outside the k={K_CAP} cut: {len(capped)}")

    # --- how close is the truth? the radius ladder ---
    d_gt_m = np.array([r['d_gt_deg'] * 111320.0 for r in ev])   # degree distance, the
    # matcher's own convention (0.0026949335 deg <-> "300 m")
    pct = np.percentile(d_gt_m, [50, 90, 95, 99, 99.9, 100])
    out['fix_to_gt_edge_m'] = dict(percentiles=[float(x) for x in pct],
                                   ladder={})
    print('\ndistance from fix to the ground-truth edge, in the matcher\'s degree-space '
          'convention (m):')
    print('  median %.1f | p90 %.1f | p95 %.1f | p99 %.1f | p99.9 %.1f | max %.1f'
          % tuple(pct))
    print("radius ladder (truth in disk if r >= d):")
    for r_m in (300.0, 200.0, 150.0, 100.0, 75.0, 50.0, 30.0, 20.0, 15.0, 10.0):
        k = int((d_gt_m <= r_m).sum())
        out['fix_to_gt_edge_m']['ladder'][str(r_m)] = k
        print(f"  r = {r_m:>5.0f} m : {k:>6}/{len(ev)} = {100 * k / len(ev):.2f} %")

    # --- impact of the unrecoverable epochs on the audit (none at the paper's disk) ---
    miss = [r for r in ev if not r['in_disk']]
    wrong = [r for r in ev if not r['alpha*beta:ok']]
    w_miss = [r for r in wrong if not r['in_disk']]
    print(f"\nwrong epochs total {len(wrong)}; of them truth-missing {len(w_miss)} "
          f"({100 * len(w_miss) / max(len(wrong), 1):.1f} %), recoverable {len(wrong) - len(w_miss)}")
    dw = np.array([r['d_gt_deg'] * 111320.0 for r in wrong])
    out['wrong_epochs'] = dict(n=len(wrong), n_missing=len(w_miss),
                               d_gt_p50=float(np.median(dw)),
                               d_gt_p90=float(np.percentile(dw, 90)),
                               n_d_gt_over_100m=int((dw > 100).sum()))
    print(f"wrong epochs' fix-to-truth distance (m): median {np.median(dw):.1f}, "
          f"p90 {np.percentile(dw, 90):.1f}, over 100 m: {(dw > 100).sum()}")
    if miss:
        for name in ('delta', 'alpha', 'alpha*beta'):
            s = np.array([r[name] for r in ev])
            y = np.array([1.0 if r[name + ':ok'] else 0.0 for r in ev])
            sm = np.array([r[name] for r in miss])
            total_l1 = float(np.abs(s - y).sum()); total_bs = float(((s - y) ** 2).sum())
            out[f'impact_{name}'] = dict(
                n_missing=len(sm), mean_score_missing=float(sm.mean()),
                n_missing_ge_0p9=int((sm >= 0.9).sum()), n_missing_eq_1=int((sm == 1.0).sum()),
                share_L1=float(sm.sum() / total_l1), share_BS=float((sm ** 2).sum() / total_bs))
            o = out[f'impact_{name}']
            print(f"{name:<10} missing: mean tw {o['mean_score_missing']:.3f}, "
                  f">=0.9: {o['n_missing_ge_0p9']}, =1.0: {o['n_missing_eq_1']}; "
                  f"carry {100 * o['share_L1']:.1f} % of L1, {100 * o['share_BS']:.1f} % of BS")
    else:
        print('no truth-missing epoch exists in the audited population at this disk, '
              'so there is no misplaced mass from it to price')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
