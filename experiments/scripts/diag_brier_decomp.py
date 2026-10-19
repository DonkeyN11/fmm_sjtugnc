#!/usr/bin/env python3
"""D8 (manuscript R23): the sample range of every printed Brier term.

The manuscript prints REL/RES/UNC = 0.0003/0.0054/0.0185 (smoothed posterior) against
0.0968/0.0066/0.0437 (baseline native score) "on the 12,398 epochs where both paths share a
correctness label". Identical labels force identical base rates and therefore identical UNC,
so 0.0185 against 0.0437 cannot both be that subset. This recomputes Murphy's decomposition
(Brier = UNC - RES + REL, exp_sim_sigma_sweep._decompose) for all six arms, on the full
population and on the shared-label subset, at three bin counts.

The shared subset is defined ONCE, from the two matched paths' cpath labels, and applied to
every arm and every bin count: defining it per arm would make each arm trivially "shared"
with itself, which is the defect this check exists to catch. Every arm is evaluated on the
same epoch set, so the six arms differ only in their score column.

Usage: python3 experiments/scripts/diag_brier_decomp.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R        # noqa: E402
import verify_paper_numbers as v     # noqa: E402
import exp_sim_sigma_sweep as S      # noqa: E402

DIAG = BASE / 'data/archive_20261019/diag'
RUN = BASE / 'data/archive_20260929/scalar'
PATH_PROPOSED = RUN / 'cmm.csv'
PATH_BASELINE = RUN / 'fmm.csv'
ARMS = (
    ('proposed_delta', RUN / 'cmm_viterbi.csv', 'trustworthiness', False),
    ('proposed_alpha', RUN / 'cmm.csv', 'trustworthiness', False),
    ('proposed_alphabeta', RUN / 'cmm.csv', 'trustworthiness_smoothed', False),
    ('baseline_delta', RUN / 'fmm.csv', 'trustworthiness', True),
    ('baseline_alpha', RUN / 'fmm_post.csv', 'trustworthiness', True),
    ('baseline_alphabeta', RUN / 'fmm.csv', 'trustworthiness_smoothed', True),
)


def build():
    """One record per epoch on which all six arms have a score: same flag + per-arm value."""
    gt, ts_by_seq = R.load_gt()
    prop = R.load_run(PATH_PROPOSED)
    base = R.load_run(PATH_BASELINE, ts_by_seq)
    runs = {name: R.load_run(p, ts_by_seq if use_seq else None)
            for name, p, _c, use_seq in ARMS}
    rec, dropped = [], 0
    for key, (ge, _gx, _gy) in gt.items():
        a, b = prop.get(key), base.get(key)
        if a is None or b is None:
            dropped += 1
            continue
        same = (v.is_edge_match(a['cpath'].strip(), ge)
                == v.is_edge_match(b['cpath'].strip(), ge))
        vals, ok = {}, True
        for name, _p, col, _u in ARMS:
            r = runs[name].get(key)
            if r is None:
                ok = False
                break
            try:
                s = float(r[col])
            except (KeyError, ValueError):
                ok = False
                break
            if s <= -900.0:
                ok = False
                break
            vals[name] = (1.0 if v.is_edge_match(r['cpath'].strip(), ge) else 0.0, s)
        if not ok:
            dropped += 1
            continue
        rec.append((same, vals))
    return rec, dropped


def report(y, s, nbins):
    y, s = np.asarray(y, float), np.asarray(s, float)
    unc, res, rel = S._decompose(y, s, nbins=nbins)
    brier = float(((s - y) ** 2).mean())
    return dict(n=int(len(y)), base_rate=float(y.mean()), unc=unc, res=res, rel=rel,
                brier=brier, identity_err=float(brier - (unc - res + rel)),
                ece=float(v.ece(s, y, n=nbins)))


def main():
    rec, dropped = build()
    same = np.array([r[0] for r in rec], bool)
    print(f'epochs with all six arms={len(rec)}  dropped={dropped}'
          f'  shared_label={int(same.sum())} ({100*same.mean():.2f}%)')
    rows = []
    for name in [a[0] for a in ARMS]:
        y = np.array([r[1][name][0] for r in rec])
        s = np.array([r[1][name][1] for r in rec])
        for tag, m in (('full', np.ones(len(rec), bool)), ('shared_label', same)):
            for nb in (10, 15, 20):
                rows.append(dict(arm=name, subset=tag, bins=nb, **report(y[m], s[m], nb)))
    DIAG.mkdir(parents=True, exist_ok=True)
    with open(DIAG / 'brier_decomp.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{"arm":<22}{"subset":<13}{"b":>3}{"n":>8}{"base":>9}{"UNC":>10}'
          f'{"RES":>10}{"REL":>10}{"ECE":>10}{"id_err":>9}')
    for r in rows:
        if r['bins'] != 10:
            continue
        print(f'{r["arm"]:<22}{r["subset"]:<13}{r["bins"]:>3}{r["n"]:>8}'
              f'{r["base_rate"]:>9.5f}{r["unc"]:>10.6f}{r["res"]:>10.6f}'
              f'{r["rel"]:>10.6f}{r["ece"]:>10.6f}{r["identity_err"]:>9.1e}')
    print(f'wrote {DIAG / "brier_decomp.csv"} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
