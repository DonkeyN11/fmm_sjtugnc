#!/usr/bin/env python3
"""Does the sampling rate move the calibration of the three estimators?

WHY THIS FAMILY EXISTS. Every cell of the shipped corpus is 1 Hz at 12 m/s, so the
step is ~12 m while the fabricated position sigma is ~1.5 sigma_rho. That single
ratio fixes how much a transition can say: the heading term's own signal-to-noise
|v| / sqrt(n^T Sigma n) is 1.5-7.6 in the corpus against 27.9 on Haikou (see
exp_sim_sigma_sweep's docstring), and the same ratio governs how much a candidate at
t+1 constrains the candidate at t. Whether the estimator ordering -- and the
calibration level itself -- is a property of the estimator or of the 1 Hz sampling
grid is therefore untested by the shipped corpus.

WHAT IS HELD FIXED. Path length (points x speed / rate is ~12 km in every cell, the
same as the shipped ones), pseudorange noise (sigma_rho = 10 and 30 m, the same two
levels as the shipped cells this family is compared against), the covariance model,
the number of trajectories (10), and the seed per sigma family (137 / 397, the same
ones the shipped sigma_10 and sigma_30 cells record in their metadata.json). What
moves is only the sampling interval: 1.0, 0.5, 0.2 Hz, i.e. 12 m, 24 m and 60 m per
step.

THE 1 Hz CELLS ARE REGENERATED HERE RATHER THAN READ FROM THE SHIPPED CORPUS. The
generator is not frozen -- `point_edge_ids` in particular has been touched since the
shipped corpus was written -- so a low-rate cell scored against a shipped 1 Hz cell
would carry any generator drift into the rate contrast. Regenerating all six cells in
one run costs a few minutes and makes the rate axis the only difference; the shipped
1 Hz numbers are still printed beside the regenerated ones as the drift check.

ARMS PER CELL. The same four as the manuscript's corpus table, plus one:
    delta     cmm scalar-viterbi, `trustworthiness`
    alpha     cmm scalar,         `trustworthiness`
    alpha*beta cmm scalar,        `trustworthiness_smoothed`
    baseline  fmm,                `trustworthiness`   (the corpus's own FMM config)
and the same three estimators again at a model with the heading term disabled:
    no-heading delta/alpha/alpha*beta   cmm scalar-direction-off[-viterbi]
The no-heading arms exist because the heading term is gated by
`direction_max_interval` (default 2 s): at 0.5 Hz a step is exactly 2 s and at
0.2 Hz it is 5 s, so the term switches ITSELF off somewhere along this axis, and the
scalar arms at 1 Hz and 0.2 Hz would then be different models. The rate contrast is
read from the no-heading rows, which come from a model that is identical at all
three rates; the `scalar` rows show separately what the guard's self-disablement
does to the delivered configuration.

Usage:
    python3 experiments/scripts/exp_sim_lowrate.py                 # generate, run, tabulate
    python3 experiments/scripts/exp_sim_lowrate.py --skip-generate # reuse existing cells
    CMM_SIM_OUTDIR=... python3 ...                                 # move the run archive

Written outputs go to OUTDIR (an archive directory, gitignored); the corpus cells are
written under data/simulation_lowrate/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import exp_sim_sigma_sweep as S  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

ROOT = BASE / 'data/simulation_lowrate'
OUTDIR = Path(os.environ.get('CMM_SIM_OUTDIR') or (BASE / 'data/archive_20260930/lowrate'))
GEN = BASE / 'experiments/scripts/generate_data_cmm.py'
SHAPEFILE = BASE / 'input/map/hainan/edges.shp'

# (tag, sigma_rho, sample_rate_hz, points, seed). Points are chosen so that
# points * speed / rate is ~12 km in every row -- the shipped corpus's path length.
CELLS = [
    ('sigma_10_hz10', 10.0, 1.0, 1000, 137),
    ('sigma_10_hz05', 10.0, 0.5, 500, 137),
    ('sigma_10_hz02', 10.0, 0.2, 200, 137),
    ('sigma_30_hz10', 30.0, 1.0, 1000, 397),
    ('sigma_30_hz05', 30.0, 0.5, 500, 397),
    ('sigma_30_hz02', 30.0, 0.2, 200, 397),
]

# The shipped 1 Hz cells, for the generator-drift check. Read from the archived runs,
# never re-run here.
SHIPPED = [('sigma_10 (shipped)', 'sigma_10', BASE / 'data/simulation/sigma_10/no_occlusion/no_fault'),
           ('sigma_30 (shipped)', 'sigma_30', BASE / 'data/simulation/sigma_30/no_occlusion/no_fault')]
SHIPPED_RUNS = Path(os.environ.get('CMM_SIM_SHIPPED') or (BASE / 'data/archive_20260928/sim'))


def generate(tag, sigma, rate, points, seed, jobs):
    d = ROOT / tag
    if (d / 'metadata.json').exists():
        print(f'  {tag}: already generated')
        return d
    cmd = [sys.executable, str(GEN), '--output-dir', str(d),
           '--shapefile', str(SHAPEFILE), '--count', '10', '--points', str(points),
           '--speed', '12.0', '--sample-rate', str(rate), '--num-sats', '8',
           '--calib-ratio', '0.2', '--min-sigma-pr', str(sigma), '--max-sigma-pr', str(sigma),
           '--seed', str(seed), '--jobs', str(jobs)]
    print('  ' + ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f'{tag}: generation failed')
    print('  ' + '\n  '.join(r.stdout.strip().split('\n')[-3:]), flush=True)
    return d


def evalcol(d, csv_path, col):
    """The Section V metric set plus the tail counts, on one (run, column)."""
    y, s, n_eval = S.pairs(d, csv_path, col)
    st = S.stats(y, s, n_eval)
    # S.stats' empty branch omits `sat`; a run whose score column is absent from an
    # old archive (e.g. a pre-smoothing file read for trustworthiness_smoothed)
    # lands there, and the table must still print rather than crash.
    for k, v0 in (('sat', float('nan')), ('n_wrong', 0), ('wrong_ge_09', 0),
                  ('fa_lt_05', float('nan')), ('log', float('nan'))):
        st.setdefault(k, v0)
    if len(y):
        st['log'] = v.log_score(y, s)
        wrong = y == 0.0
        st['n_wrong'] = int(wrong.sum())
        st['wrong_ge_09'] = int((wrong & (s >= 0.9)).sum())
        right = ~wrong
        st['fa_lt_05'] = 100.0 * float((right & (s < 0.5)).sum()) / max(int(right.sum()), 1)
    else:
        st.update(log=float('nan'), n_wrong=0, wrong_ge_09=0, fa_lt_05=float('nan'))
    return st


def arms_of(tag, d):
    """(label, csv path, score column) for one cell, in table order."""
    return [
        ('delta', OUTDIR / f'{tag}_cmm_scalar-viterbi.csv', 'trustworthiness'),
        ('alpha', OUTDIR / f'{tag}_cmm_scalar.csv', 'trustworthiness'),
        ('alpha*beta', OUTDIR / f'{tag}_cmm_scalar.csv', 'trustworthiness_smoothed'),
        ('baseline delta', OUTDIR / f'{tag}_fmm.csv', 'trustworthiness'),
        ('no-heading delta', OUTDIR / f'{tag}_cmm_scalar-direction-off-viterbi.csv', 'trustworthiness'),
        ('no-heading alpha', OUTDIR / f'{tag}_cmm_scalar-direction-off.csv', 'trustworthiness'),
        ('no-heading alpha*beta', OUTDIR / f'{tag}_cmm_scalar-direction-off.csv',
         'trustworthiness_smoothed'),
    ]


FMT = {'n_eval': '{:d}', 'acc': '{:.2f}', 'ece': '{:.4f}', 'auc': '{:.4f}',
       'bss': '{:+.3f}', 'sat': '{:.1f}', 'wrong_ge_09': '{:d}', 'n_wrong': '{:d}',
       'fa_lt_05': '{:.2f}'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--cmm-bin', type=Path, default=BASE / 'build' / 'cmm')
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    S.OUTDIR = OUTDIR  # the runner functions read the module global

    cells = []
    print('=== generation ===', flush=True)
    for tag, sigma, rate, points, seed in CELLS:
        d = ROOT / tag if args.skip_generate else generate(tag, sigma, rate, points, seed, args.jobs)
        cells.append((tag, sigma, rate, d))

    print('\n=== matcher runs ===', flush=True)
    for tag, sigma, rate, d in cells:
        for arm in ('scalar', 'scalar-viterbi', 'scalar-direction-off',
                    'scalar-direction-off-viterbi'):
            out = OUTDIR / f'{tag}_cmm_{arm}.csv'
            if out.exists():
                print(f'  {out.name}: exists', flush=True)
                continue
            p = S.run_cmm(d, tag, False, arm, args.cmm_bin)
            print(f'  cmm[{arm}] {tag}: {"ok" if p else "FAILED"} -> {p}', flush=True)
        out = OUTDIR / f'{tag}_fmm.csv'
        if out.exists():
            print(f'  {out.name}: exists', flush=True)
        else:
            p, synth = S.run_fmm(d, tag, False)
            print(f'  fmm {tag}: {"ok" if p else "FAILED"} (traj synthesised: {synth})', flush=True)

    print('\n=== metrics ===', flush=True)
    out = {'cells': []}
    for tag, sigma, rate, d in cells:
        arm_rows, arms = [], arms_of(tag, d)
        for label, path, col in arms:
            if not path.exists():
                print(f'  {tag} {label}: missing run {path}', flush=True)
                continue
            st = evalcol(d, path, col)
            arm_rows.append((label, st))
        cell = dict(tag=tag, sigma_rho=sigma, rate_hz=rate, step_m=12.0 / rate,
                    dir=str(d.relative_to(BASE)),
                    arms={label: st for label, st in arm_rows})
        out['cells'].append(cell)
        for label, st in arm_rows:
            print(f'{tag:<18}{rate:>4} Hz  {label:<22}' +
                  ''.join(f'{FMT[c].format(st[c]):>11}' for c in
                          ('n_eval', 'acc', 'ece', 'auc', 'bss', 'sat',
                           'n_wrong', 'wrong_ge_09', 'fa_lt_05')), flush=True)
        print(flush=True)

    # The shipped 1 Hz cells, read from the archived runs: the drift check.
    print('=== shipped 1 Hz cells, archived runs (drift check) ===', flush=True)
    for label, tag, d in SHIPPED:
        rows = [('delta', SHIPPED_RUNS / f'{tag}_cmm_scalar-viterbi.csv', 'trustworthiness'),
                ('alpha', SHIPPED_RUNS / f'{tag}_cmm_scalar.csv', 'trustworthiness'),
                ('alpha*beta', SHIPPED_RUNS / f'{tag}_cmm_scalar.csv', 'trustworthiness_smoothed'),
                ('baseline delta', SHIPPED_RUNS / f'{tag}_fmm.csv', 'trustworthiness')]
        for arm, path, col in rows:
            if not path.exists():
                continue
            st = evalcol(d, path, col)
            print(f'{label:<18}{1.0:>4} Hz  {arm:<22}' +
                  ''.join(f'{FMT[c].format(st[c]):>11}' for c in
                          ('n_eval', 'acc', 'ece', 'auc', 'bss', 'sat',
                           'n_wrong', 'wrong_ge_09', 'fa_lt_05')), flush=True)
        print(flush=True)

    jf = OUTDIR / 'lowrate.json'
    json.dump(out, open(jf, 'w'), indent=1)
    print(f'wrote {jf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())