#!/usr/bin/env python3
"""Analyses requested by the 2026-10-10 review round (review_2026-10-10/EDITORIAL_DECISION.md).

A. Cross-matcher ECE under a common label reference, and the Brier decomposition
   (reliability - resolution + uncertainty) per arm.  [P0-2 / DA M1]
B. Threshold table for the proposed smoothed and filtering posteriors: coverage,
   error rate among certified epochs, trajectory-clustered interval.  [P1-8 / R3 W1]
C. Trajectory 11's non-evaluable epochs counted as wrong for BOTH matchers, pooled
   and per-trajectory, mirroring the 54-epoch sensitivity.  [P0-3 / R1 W1, R2 W3, DA M3]
D. Fix-quality share over the evaluable epochs, if the ground-truth file carries a
   quality flag.  [R2 D1 warn]

Reads only committed artifacts (data/archive_20260929/scalar/*.csv and the aligned
table); writes nothing.  Every number printed here is meant to be pasted into the
revision with its construction stated.

Usage: python3 experiments/scripts/review_20261010_analyses.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

SEED = 20261010
NREP = 2000
ALIGNED = BASE / 'data/real_vehicle/hainan_06/processed/aligned.csv'
GT_RAW = BASE / 'data/real_vehicle/hainan_06/processed/ground_truth.csv'


def paired_arrays():
    """(proposed smoothed, proposed alpha, baseline native, baseline smoothed) x
    (label, score), aligned epoch-by-epoch on the SAME keys.  Both matchers'
    correctness labels are kept separately so the common-label subset is explicit."""
    gt, ts_by_seq = R.load_gt()
    prop = R.load_run(R.OUTDIR / 'cmm.csv')               # CaMM: timestamp join
    base = R.load_run(R.OUTDIR / 'fmm.csv', ts_by_seq)    # baseline native score
    bpost = R.load_run(R.OUTDIR / 'fmm_post.csv', ts_by_seq)
    rows = []
    for key, (gt_edge, _, _) in gt.items():
        rp, rb, rq = prop.get(key), base.get(key), bpost.get(key)
        if rp is None or rb is None:
            continue
        cp = v.is_edge_match(rp['cpath'].strip(), gt_edge)
        cb = v.is_edge_match(rb['cpath'].strip(), gt_edge)

        def sc(r, col):
            if r is None:
                return np.nan
            try:
                x = float(r[col])
            except (ValueError, KeyError):
                return np.nan
            return x if x > -900.0 else np.nan

        rows.append((key[0], 1.0 if cp else 0.0, 1.0 if cb else 0.0,
                     sc(rp, 'trustworthiness_smoothed'), sc(rp, 'trustworthiness'),
                     sc(rb, 'trustworthiness'),
                     sc(rq, 'trustworthiness_smoothed') if rq is not None else np.nan))
    return rows


def ece_of(s, y, nbin=10):
    return v.ece(s, y, n=nbin)


def brier_decomp(s, y, nbin=10):
    """BS = REL - RES + UNC with equal-width bins, last closed on the right."""
    edges = np.linspace(0.0, 1.0, nbin + 1)
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, nbin - 1)
    p = y.mean()
    rel = res = 0.0
    for b in range(nbin):
        m = idx == b
        nb = m.sum()
        if nb == 0:
            continue
        conf, acc = s[m].mean(), y[m].mean()
        rel += nb * (conf - acc) ** 2
        res += nb * (acc - p) ** 2
    n = len(y)
    return dict(rel=rel / n, res=res / n, unc=p * (1 - p),
                bs=float(((s - y) ** 2).mean()), p=p, n=n)


def clustered_error_ci(err, traj, tau_mask):
    """Error rate among the selected epochs, resampled by trajectory."""
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for t, e, m in zip(traj, err, tau_mask):
        if m:
            by[t].append(e)
    keys = sorted(by)
    if not keys:
        return np.nan, np.nan, np.nan
    point = float(np.mean([e for k in keys for e in by[k]]))
    reps = []
    for _ in range(NREP):
        draw = rng.choice(keys, size=len(keys), replace=True)
        vals = [e for k in draw for e in by[k]]
        reps.append(np.mean(vals))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    rows = paired_arrays()
    traj = np.array([r[0] for r in rows])
    cp = np.array([r[1] for r in rows])
    cb = np.array([r[2] for r in rows])
    s_sm = np.array([r[3] for r in rows])
    s_al = np.array([r[4] for r in rows])
    s_base = np.array([r[5] for r in rows])
    s_bpost = np.array([r[6] for r in rows])
    finite = ~np.isnan(s_sm) & ~np.isnan(s_base)
    print(f'paired epochs (both matchers scored): {finite.sum()}')
    print(f'base rates: proposed path {cp[finite].mean():.4f}  '
          f'baseline path {cb[finite].mean():.4f}')

    # ---------------- A. common-label ECE + Brier decomposition ----------------
    print('\n== A. cross-matcher ECE under a common label reference ==')
    same = finite & (cp == cb)
    diff = finite & (cp != cb)
    print(f'common-label epochs: {same.sum()} ({100*same.sum()/finite.sum():.1f}%)   '
          f'disagreeing: {diff.sum()} ({100*diff.sum()/finite.sum():.1f}%)')
    print(f'ECE on common-label subset: proposed smoothed {ece_of(s_sm[same], cp[same]):.4f}'
          f'  |  baseline native {ece_of(s_base[same], cb[same]):.4f}')
    print(f'ECE on disagreeing subset:  proposed smoothed {ece_of(s_sm[diff], cp[diff]):.4f}'
          f'  |  baseline native {ece_of(s_base[diff], cb[diff]):.4f}')
    print('among the disagreeing epochs, whose label?  proposed correct / baseline wrong:'
          f' {int(((cp == 1) & (cb == 0) & diff).sum())}  '
          f'baseline correct / proposed wrong: {int(((cp == 0) & (cb == 1) & diff).sum())}')
    for lab, s, y in (('proposed smoothed', s_sm, cp), ('baseline native', s_base, cb)):
        d = brier_decomp(s[finite], y[finite])
        print(f'Brier decomposition | {lab:<18} BS={d["bs"]:.4f} = '
              f'REL {d["rel"]:.4f} - RES {d["res"]:.4f} + UNC {d["unc"]:.4f} '
              f'(p={d["p"]:.4f}, check {d["rel"]-d["res"]+d["unc"]:.4f})')

    # ---------------- B. threshold table ----------------
    print('\n== B. threshold table (proposed arms; clustered 95% CI on the error rate) ==')
    for lab, s, y in (('smoothed ab', s_sm, cp), ('filtering a', s_al, cp),
                      ('baseline native', s_base, cb)):
        m = ~np.isnan(s)
        print(f'-- {lab}')
        for tau in (0.5, 0.9, 0.99, 1.0):
            sel = m & (s >= tau)
            share = 100.0 * sel.sum() / m.sum()
            if sel.sum() == 0:
                print(f'   tau={tau:<4}  share {share:5.1f}%  (no epochs)')
                continue
            err = 1.0 - y
            pt, lo, hi = clustered_error_ci(err, traj, sel)
            print(f'   tau={tau:<4}  share {share:5.1f}%  n {sel.sum():5d}  '
                  f'err among certified {100*pt:5.2f}% [{100*lo:.2f}, {100*hi:.2f}]  '
                  f'wrong epochs {int((sel & (y == 0)).sum())}')

    # ---------------- C. traj 11 non-evaluable counted wrong ----------------
    print('\n== C. non-evaluable epochs counted as wrong for both matchers ==')
    ex = defaultdict(int)
    with open(ALIGNED, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            tid = r['id'].strip()
            if tid in v.EXCLUDE_TRAJ:
                continue
            if r['gt_edge'].strip() == '0':
                ex[tid] += 1
    gt, ts_by_seq = R.load_gt()
    prop = R.load_run(R.OUTDIR / 'cmm.csv')
    base = R.load_run(R.OUTDIR / 'fmm.csv', ts_by_seq)

    def acc(run, keys=None):
        n = ok = 0
        for key, (gt_edge, _, _) in gt.items():
            if keys is not None and key not in keys:
                continue
            r = run.get(key)
            if r is None:
                continue
            n += 1
            ok += int(v.is_edge_match(r['cpath'].strip(), gt_edge))
        return n, 100.0 * ok / n if n else float('nan')

    extra11 = ex.get('11', 0)
    extra_all = sum(ex.values())
    print(f'traj-11 gt_edge=0 epochs (aligned): {extra11}   all trajectories: {extra_all}')
    for nm, run in (('proposed', prop), ('baseline', base)):
        n0, a0 = acc(run)
        _, a11 = acc(run, keys={k for k in gt if k[0] == '11'})
        _, a11w = acc(run, keys={k for k in gt if k[0] == '11'})
        # pooled with traj-11 zeros counted wrong
        n11e = sum(1 for (t, _) in gt if t == '11')
        ok11 = a11 / 100.0 * n11e
        print(f'  {nm:<9} pooled {a0:.2f}%  ->  with {extra11} traj-11 epochs wrong: '
              f'{100.0 * (a0 / 100.0 * n0) / (n0 + extra11):.2f}%'
              f'   | traj 11 {a11:.2f}% -> {100.0 * ok11 / (n11e + extra11):.2f}%'
              f'   | with ALL {extra_all} non-eval wrong: '
              f'{100.0 * (a0 / 100.0 * n0) / (n0 + extra_all):.2f}%')

    # ---------------- D. fix-quality share over evaluable epochs ----------------
    print('\n== D. ground-truth fix quality over evaluable epochs ==')
    with open(GT_RAW, newline='') as f:
        rd = csv.DictReader(f, delimiter=';')
        cols = rd.fieldnames
        print(f'ground_truth.csv columns: {cols}')
        cand = [c for c in cols if any(t in c.lower() for t in
                                       ('qual', 'fix', 'status', 'sol', 'flag'))]
        if not cand:
            print('no quality/fix column present -- fix share not computable here')
        else:
            c0 = cand[0]
            import collections
            dist = collections.Counter()
            for r in rd:
                if r['id'].strip() in v.EXCLUDE_TRAJ:
                    continue
                if r['gt_edge'].strip() in ('0', '-1', ''):
                    continue
                dist[r[c0].strip()] += 1
            tot = sum(dist.values())
            print(f'{c0} over evaluable rows: {dict(dist.most_common(6))}  (n={tot})')


if __name__ == '__main__':
    main()
