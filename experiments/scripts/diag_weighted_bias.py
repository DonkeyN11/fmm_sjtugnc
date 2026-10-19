#!/usr/bin/env python3
"""D11 (manuscript R09): calibration stratified by the billing weight.

Proposition 1 conditions on the score alone, so a low binned ECE establishes calibration on
average, not conditional on how much is charged. The billing step is the matched edge's
length B_t; a wrong match on a long edge costs more than one on a short edge. This reports,
per decile of B_t, the mean score, the empirical correctness and their difference, plus the
quantity the proposition is actually about -- the bias of the estimated mis-billed distance,

    M   = sum_t B_t (1 - c_t)          true mis-billed distance
    Mhat= sum_t B_t (1 - tw_t)         score-based estimate
    bias= Mhat - M = sum_t B_t (c_t - tw_t)

which is zero in expectation only if the score is conditionally calibrated given the weight
(or if B is deterministic, the sufficient condition the manuscript omits).

Usage: python3 experiments/scripts/diag_weighted_bias.py [score_col] [out_name]
"""
import csv
import sys
from pathlib import Path

import numpy as np
from osgeo import ogr

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R      # noqa: E402
import verify_paper_numbers as v   # noqa: E402

DIAG = BASE / 'data/archive_20261019/diag'
CMM = BASE / 'data/archive_20260929/scalar/cmm.csv'
SHP = BASE / 'input/map/hainan/edges.shp'


def edge_lengths():
    """FID -> polyline length in metres, the id space cpath and gt_edge share."""
    ds = ogr.Open(str(SHP))
    ly = ds.GetLayer(0)
    out = {}
    for f in ly:
        g = f.GetGeometryRef()
        parts = ([g] if g.GetGeometryName() == 'LINESTRING'
                 else [g.GetGeometryRef(k) for k in range(g.GetGeometryCount())])
        L = 0.0
        for p in parts:
            n = p.GetPointCount()
            for i in range(n - 1):
                L += v.haversine_m(p.GetX(i), p.GetY(i), p.GetX(i + 1), p.GetY(i + 1))
        out[f.GetFID()] = L
    return out


def main():
    col = sys.argv[1] if len(sys.argv) > 1 else 'trustworthiness_smoothed'
    name = sys.argv[2] if len(sys.argv) > 2 else 'weighted_bias.csv'
    gt, _ = R.load_gt()
    run = R.load_run(CMM)
    lens = edge_lengths()
    y, s, b, miss, nodigit = [], [], [], 0, 0
    for key, (ge, _gx, _gy) in gt.items():
        r = run.get(key)
        if r is None:
            continue
        try:
            sc = float(r[col])
        except (KeyError, ValueError):
            continue
        if sc <= -900.0:
            continue
        cp = r['cpath'].strip()
        if not cp.isdigit():
            nodigit += 1
            continue
        B = lens.get(int(cp))
        if B is None or B <= 0:
            miss += 1
            continue
        y.append(1.0 if v.is_edge_match(cp, ge) else 0.0)
        s.append(sc)
        b.append(B)
    y, s, b = np.array(y), np.array(s), np.array(b)
    qs = np.quantile(b, np.linspace(0.1, 0.9, 9))
    dec = np.digitize(b, qs)
    rows = []
    for g in range(10):
        m = dec == g
        if not m.any():
            continue
        rows.append(dict(decile=g, n=int(m.sum()), share=float(m.mean()),
                         B_mean_m=float(b[m].mean()), B_min_m=float(b[m].min()),
                         B_max_m=float(b[m].max()), mean_tw=float(s[m].mean()),
                         acc=float(y[m].mean()),
                         gap=float(s[m].mean() - y[m].mean()),
                         weighted_gap=float(m.mean() * (s[m].mean() - y[m].mean()))))
    sb, st, sy = b.sum(), (b * s).sum(), (b * y).sum()
    M_true, M_hat = sy * 0 + (b * (1 - y)).sum(), (b * (1 - s)).sum()
    summary = dict(
        score_col=col, n=int(len(y)), n_cpath_not_digit=int(nodigit), n_no_length=int(miss),
        mean_B=float(b.mean()), sum_B_m=float(sb),
        mean_tw=float(s.mean()), acc=float(y.mean()),
        ECE=float(v.ece(s, y, n=10)),
        signed_gap=float(s.mean() - y.mean()),
        M_true_m=float(M_true), M_hat_m=float(M_hat),
        bias_m=float(M_hat - M_true),
        bias_pct_of_true=float(100.0 * (M_hat - M_true) / M_true) if M_true else float('nan'),
        weighted_score_mean=float(st / sb), weighted_acc=float(sy / sb),
        weighted_gap=float((st - sy) / sb),
    )
    DIAG.mkdir(parents=True, exist_ok=True)
    with open(DIAG / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'score={col} n={summary["n"]}  ECE10={summary["ECE"]:.6f}'
          f'  mean_tw={summary["mean_tw"]:.6f}  acc={summary["acc"]:.6f}')
    print(f'{"dec":>4}{"n":>7}{"pop%":>7}{"B_mean":>9}{"mean_tw":>9}{"acc":>9}'
          f'{"gap":>10}{"wtd_gap":>10}')
    for r in rows:
        print(f'{r["decile"]:>4}{r["n"]:>7}{100*r["share"]:>7.2f}{r["B_mean_m"]:>9.1f}'
              f'{r["mean_tw"]:>9.5f}{r["acc"]:>9.5f}{r["gap"]:>10.5f}{r["weighted_gap"]:>10.6f}')
    print(f'M_true={summary["M_true_m"]:.1f} m  M_hat={summary["M_hat_m"]:.1f} m'
          f'  bias={summary["bias_m"]:+.2f} m ({summary["bias_pct_of_true"]:+.3f} %)')
    print(f'B-weighted: mean_tw={summary["weighted_score_mean"]:.6f}'
          f' acc={summary["weighted_acc"]:.6f} gap={summary["weighted_gap"]:+.6f}')
    print(f'wrote {DIAG / name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
