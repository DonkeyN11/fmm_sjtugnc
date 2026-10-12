#!/usr/bin/env python3
"""The paper's one experiment: three estimators of the same posterior, on two datasets.

WHY THIS FILE EXISTS. The manuscript's claim is a single sentence -- the matched-segment
posterior may be used as a probability -- and the experiment that bears on it is a single
comparison. One Viterbi path is computed; three numbers are read off it:

    delta_t / sum_j delta_t(j)          the Viterbi score        (what the matcher optimises)
    alpha_t / sum_j alpha_t(j)          the filtering posterior  (evidence up to t)
    alpha_t beta_t / sum_j alpha_t beta_t   the smoothed posterior   (evidence, whole trajectory)

The three differ ONLY in the evidence set they condition on. The candidate set, the
emission, the transition weights and the decoded path are identical, because all three are
columns of one run: `tw_use_viterbi` selects which recursion is written to
`trustworthiness`, and `trustworthiness_smoothed` is the backward pass off the same forward
recursion. So this is a genuine ablation -- nothing else moves.

They are compared on ECE and ROC AUC, plus the Brier skill score, on the real-vehicle set
and on the simulation corpus. Nothing else is a free parameter of the study, and nothing
else is reported: there is no post-hoc recalibration arm (a temperature or an isotonic fit
would be a fourth object, fitted on the labels it is then scored against), and no component
ablation of the observation model (the model this paper uses has no covariance shape to
ablate -- see input/config/paper1_cmm_scalar.xml).

THE CLASSICAL BASELINE appears as one comparator, scored with its own native Viterbi score,
because that is what an HMM map matcher actually reports. Its filtering and smoothed arms
are computed here too and printed, so the reader can see that the ordering of the three
estimators is a property of the estimator and not of the observation model -- but they are
not part of the claim.

Provenance. Before this file, the numbers were produced by an ad-hoc script under /tmp,
which is how a debug configuration came to be typeset as the delivered one (see
`four-tables-had-no-producer` in the project memory). Every cell below is reproducible by
running this file.

Usage:
    python3 experiments/scripts/three_estimator_tables.py            # tables to stdout
    python3 experiments/scripts/three_estimator_tables.py --json F   # also write JSON

Environment: CMM_OUTDIR / CMM_CONFIG move the real-data run (defaults are the delivered
operating point); CMM_SIM_OUTDIR moves the corpus run. Both are read through
real_final_tables and exp_sim_sigma_sweep, so there is exactly one definition of each path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import exp_sim_sigma_sweep as S  # noqa: E402
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

# (name, file, score column, join through seq->timestamp).
#
# The proposed matcher's delta arm is a SEPARATE FILE because `tw_use_viterbi` is a config
# key, not an output column: the run that writes delta overwrites `trustworthiness` with it.
# `cmm.csv` and `cmm_viterbi.csv` are the same configuration on the same input, so the path
# is the same and only the score differs -- that is what makes the three comparable.
#
# The baseline needs no second file for its smoothed arm: `trustworthiness_smoothed` always
# uses the forward recursion alpha, whatever `tw_use_viterbi` says, so `fmm.csv` carries
# delta and alpha*beta side by side and `fmm_post.csv` carries alpha and alpha*beta.
REAL_PROPOSED = [
    ('delta (Viterbi score)', 'cmm_viterbi.csv', 'trustworthiness', False),
    ('alpha (filtering)', 'cmm.csv', 'trustworthiness', False),
    ('alpha*beta (smoothed)', 'cmm.csv', 'trustworthiness_smoothed', False),
]
REAL_BASELINE = [
    ('delta (Viterbi score)', 'fmm.csv', 'trustworthiness', True),
    ('alpha (filtering)', 'fmm_post.csv', 'trustworthiness', True),
    ('alpha*beta (smoothed)', 'fmm.csv', 'trustworthiness_smoothed', True),
]

# The corpus arms. `scalar` is the operating point this paper reports: the 2x2 covariance
# is used nowhere, the per-epoch sigma is sqrt((sde^2+sdn^2)/2), and candidate projection
# is Euclidean. `scalar-viterbi` is the same configuration with tw_use_viterbi=true.
SIM_DELTA = 'scalar-viterbi'
SIM_MAIN = 'scalar'

ESTIMATORS = ('delta', 'alpha', 'alpha*beta')


def _row(label, res):
    """One printed row: the four reported quantities plus the two they are defined from."""
    return dict(label=label, n=res['n'], n_eval=res.get('n_eval', res['n']),
                acc=res['acc'], ece=res['ece'], auc=res['auc'], bss=res['bss'],
                brier=res['brier'], log=res.get('log', float('nan')),
                sep=res.get('sep', float('nan')), sat=res.get('sat', float('nan')))


def real_table():
    """The real-vehicle table, on the current population and the current operating point."""
    gt, ts_by_seq = R.load_gt()
    out = {'n_evaluable': len(gt), 'proposed': [], 'baseline': []}
    for block, arms in (('proposed', REAL_PROPOSED), ('baseline', REAL_BASELINE)):
        for label, fname, col, use_seq in arms:
            path = R.OUTDIR / fname
            if not path.exists():
                raise SystemExit(f'missing run {path}; produce it with real_final_tables.py '
                                 f'(CMM_OUTDIR={R.OUTDIR})')
            res = R.evaluate(path, gt, col, ts_by_seq if use_seq else None)
            y, s = res['labels'], res['scores']
            res['log'] = v.log_score(y, s)
            out[block].append(_row(label, res))
    return out


def sim_table():
    """The corpus table: every cell x three estimators."""
    out = []
    for tag, sigma_rho, data_dir in S.cells():
        # The cell label carries slashes for the degraded and mismatch families; the
        # filenames carry underscores. `exp_sim_sigma_sweep.main` does the same
        # substitution, so the two scripts address the same files.
        safe = tag.replace('/', '_')
        d_delta = S.OUTDIR / f'{safe}_cmm_{SIM_DELTA}.csv'
        d_main = S.OUTDIR / f'{safe}_cmm_{SIM_MAIN}.csv'
        d_fmm = S.OUTDIR / f'{safe}_fmm.csv'
        if not (d_delta.exists() and d_main.exists()):
            raise SystemExit(f'missing corpus runs for {tag} in {S.OUTDIR}; produce them '
                             f'with exp_sim_sigma_sweep.py --cmm-arm {SIM_DELTA} and '
                             f'--cmm-arm {SIM_MAIN}')
        cell = {'cell': tag, 'sigma_rho': sigma_rho, 'arms': {}}
        specs = [
            ('delta', d_delta, 'trustworthiness'),
            ('alpha', d_main, 'trustworthiness'),
            ('alpha*beta', d_main, 'trustworthiness_smoothed'),
        ]
        for name, path, col in specs:
            y, s, n_eval = S.pairs(data_dir, path, col)
            st = S.stats(y, s, n_eval)
            st['log'] = v.log_score(y, s) if len(y) else float('nan')
            cell['arms'][name] = _row(name, st)
        if d_fmm.exists():
            y, s, n_eval = S.pairs(data_dir, d_fmm, 'trustworthiness')
            st = S.stats(y, s, n_eval)
            st['log'] = v.log_score(y, s) if len(y) else float('nan')
            cell['arms']['baseline_delta'] = _row('delta (Viterbi score)', st)
        out.append(cell)
    return out


def _fmt(r, cols):
    return ''.join(f'{fmt.format(r[k]):>10}' for k, fmt in cols)


REAL_COLS = (('n', '{:d}'), ('acc', '{:.2f}'), ('ece', '{:.4f}'),
             ('auc', '{:.4f}'), ('bss', '{:+.3f}'), ('brier', '{:.4f}'), ('log', '{:+.3f}'))
SIM_COLS = (('n_eval', '{:d}'), ('acc', '{:.2f}'), ('ece', '{:.4f}'),
            ('auc', '{:.4f}'), ('bss', '{:+.3f}'))


def _hdr(cols, first=24):
    names = {'n': 'n', 'n_eval': 'n eval', 'acc': 'acc %', 'ece': 'ECE', 'auc': 'AUC',
             'bss': 'BSS', 'brier': 'Brier', 'log': 'log'}
    return f'{"":<{first}}' + ''.join(f'{names[k]:>10}' for k, _ in cols)


def print_real(t):
    print('=' * 100)
    print(f'REAL VEHICLE -- {t["n_evaluable"]:,} evaluable epochs, '
          f'operating point {R.CMM_CONFIG}, runs in {R.OUTDIR}')
    print('=' * 100)
    print(_hdr(REAL_COLS))
    for block, title in (('proposed', 'the proposed matcher'),
                         ('baseline', 'the classical baseline')):
        print(f'  {title}')
        for r in t[block]:
            print(f'    {r["label"]:<22}' + _fmt(r, REAL_COLS))
    print('\n  BSS is 1 - Brier / (p(1-p)) at each arm\'s own base rate p, so it is the')
    print('  arm\'s squared error against the do-nothing constant that always answers p.')


def print_sim(t):
    print()
    print('=' * 100)
    print(f'SIMULATION CORPUS -- {len(t)} cells x 3 estimators, runs in {S.OUTDIR}')
    print('=' * 100)
    print(f'{"cell":<28}{"estimator":<14}' + _hdr(SIM_COLS[1:], first=0))
    for cell in t:
        for i, name in enumerate(ESTIMATORS):
            r = cell['arms'][name]
            head = f'{cell["cell"]:<28}' if i == 0 else f'{"":<28}'
            print(f'{head}{name:<14}' + _fmt(r, SIM_COLS[1:]))
        if 'baseline_delta' in cell['arms']:
            print(f'{"":<24}{"baseline d":<14}' + _fmt(cell['arms']['baseline_delta'],
                                                        SIM_COLS[1:]))
        print()

    # The summary the manuscript states: does the estimator ordering hold in every cell?
    print('-' * 100)
    print('ordering of the three estimators, per cell')
    n_ece = n_auc = n_bss = 0
    worst_ece = (None, 0.0)
    for cell in t:
        a = cell['arms']
        d_ece = a['delta']['ece'] - a['alpha*beta']['ece']
        d_auc = a['alpha*beta']['auc'] - a['delta']['auc']
        d_bss = a['alpha*beta']['bss'] - a['delta']['bss']
        n_ece += d_ece > 0
        n_auc += d_auc > 0
        n_bss += d_bss > 0
        if d_ece < worst_ece[1]:
            worst_ece = (cell['cell'], d_ece)
        print(f'  {cell["cell"]:<30}ECE  delta {a["delta"]["ece"]:.4f} -> ab '
              f'{a["alpha*beta"]["ece"]:.4f}  ({d_ece:+.4f})   '
              f'AUC {d_auc:+.4f}   BSS {a["delta"]["bss"]:+.2f} -> '
              f'{a["alpha*beta"]["bss"]:+.2f}')
    n = len(t)
    print(f'\n  smoothing lowers ECE in {n_ece}/{n} cells, raises AUC in {n_auc}/{n}, '
          f'raises BSS in {n_bss}/{n}')
    if worst_ece[0]:
        print(f'  the one adverse ECE cell is {worst_ece[0]} ({worst_ece[1]:+.4f})')

    # delta and alpha are the same recursion with max in place of log-sum-exp, so on a
    # corpus where one candidate dominates they agree; the real data is where they part.
    d_da = [c['arms']['delta']['ece'] - c['arms']['alpha']['ece'] for c in t]
    print(f'  delta vs alpha: ECE differs by {min(d_da):+.4f} to {max(d_da):+.4f} '
          f'(median {float(np.median(d_da)):+.4f}) -- on this corpus the two are the same '
          f'score, which is why the real-vehicle gap between them is the informative one')

    # BSS divides by p(1-p) at the arm's own base rate. Written as -excess/p(1-p), where
    # the excess is the arm's squared error over the constant's, it is the EXCESS a small
    # denominator magnifies; an excess is a real quantity, so "the ratio is large because
    # the denominator is small" is only half the sentence. What the small denominator
    # takes away is the power to resolve the excess. Both halves matter and the manuscript
    # says both (Section IV): where the base rate is thick enough the deficit is a property
    # of the score, and in the thinnest cell here -- sigma_01, about a dozen wrong epochs
    # in 10^4 and a denominator carrying a ~29 % relative standard error -- neither the
    # level of the column nor the ordering within it can be tested.
    thin = [f'{c["cell"]} ({c["arms"]["alpha*beta"]["acc"]:.2f}%)' for c in t
            if c['arms']['alpha*beta']['acc'] > 99.0]
    if thin:
        print('  caution: in ' + ', '.join(thin) + ' the base rate is so close to 1 that')
        print('           p(1-p) is ~1e-3: BSS is a ratio whose small denominator magnifies')
        print('           the arm\'s excess of squared error over the constant\'s, and ECE')
        print('           and AUC are the readable columns. In the thinnest of them the')
        print('           excess is itself unresolvable, which is not the same as absent:')
        print('           where the base is thicker the deficit is the score\'s own.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=Path, default=None,
                    help='also write both tables as JSON to this path')
    args = ap.parse_args()

    real = real_table()
    sim = sim_table()
    print_real(real)
    print_sim(sim)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({'real': real, 'sim': sim}, args.json.open('w'), indent=1, default=float)
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
