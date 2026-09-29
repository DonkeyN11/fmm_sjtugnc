#!/usr/bin/env python3
"""The 2x2 attribution grid: emission model x transition factor, on the real data.

WHY. The manuscript's comparison is the proposed matcher against the classical FMM
baseline, and the two differ in TWO ways at once:

    emission     constant isotropic sigma = 100 m      vs  per-epoch sigma_i + map term
    transitions  raw w = min(1, d_gnss/d_road)         vs  row-normalised t = w / sum_b w

(the baseline's forward recursion multiplies the raw factor -- see the note in
fmm_algorithm.cpp's update_layer -- and the proposed matcher row-normalises). Calling
the pair an "observation model" comparison, as the tables currently do, bundles the
two. This script runs the two missing cells so each difference can be priced:

    (constant sigma, raw)          the baseline as shipped          -- reuse archive
    (constant sigma, row-normalised)  fmm      with tp_normalized=true   -- new run
    (per-epoch sigma, row-normalised) the proposed matcher as shipped -- reuse archive
    (per-epoch sigma, raw)            cmm      with tp_normalized=false  -- new run

Read against the delivered runs, so only the four new runs are produced (their inputs,
configs and evaluators are the delivered ones -- R._config and R.evaluate, no copy).

THE EXPECTATION THIS TESTS, stated before running it, because it is the shape the
paper needs: if the row-normalisation is what makes the posterior arms calibrated,
the (constant sigma, row-normalised) cell should recover most of the ECE gap from
0.197 to 0.007 on its own. The earlier one-line measurement recorded in
fmm_algorithm.cpp points the other way -- normalising the baseline's own factors
costs ECE 0.197 -> 0.247, Brier 0.162 -> 0.190, AUC 0.788 -> 0.719, separation
0.352 -> 0.214 on the smoothed arm -- so the honest answer may be that the emission
model carries the calibration and the transition factor carries the ranking.

Usage: python3 experiments/scripts/exp_tp_attribution.py            # prints the grid
       CMM_OUTDIR=... EXP_TP_OUTDIR=... python3 ...                 # move the runs
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import exp_sim_sigma_sweep as S  # noqa: E402
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

DELIVERED = R.OUTDIR                              # the shipped operating-point runs
OUTDIR = Path(os.environ.get('EXP_TP_OUTDIR') or (BASE / 'data/archive_20260930/tpattr'))

CMM_CFG = 'paper1_cmm_scalar.xml'
FMM_CFG = 'paper1_fmm_baseline.xml'


def _run(kind, cfg_text, out):
    """Write the substituted config to a temp file, run the binary, return success."""
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w') as fh:
        fh.write(cfg_text)
        cfg = fh.name
    log = OUTDIR / f'{out.stem}.log'
    with open(log, 'w') as lf:
        r = subprocess.run([str(BASE / 'build' / kind), cfg], cwd=BASE,
                           stdout=lf, stderr=subprocess.STDOUT)
    Path(cfg).unlink()
    return r.returncode == 0


def make_runs():
    """The four runs that are missing from the grid; existing ones are left alone."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # tw_use_viterbi is set through S._set_key for the CMM run because
    # paper1_cmm_scalar.xml does not carry the element -- R._config's substitution
    # would raise rather than insert it. The FMM template does carry it, so there
    # _config's own substitution is the one that keeps the two runs comparable.
    jobs = [
        # (name, binary, config, tp_normalized, tw_use_viterbi or None)
        ('cm_raw', 'cmm', CMM_CFG, 'false', 'false'),
        ('cm_raw_viterbi', 'cmm', CMM_CFG, 'false', 'true'),
        ('fm_norm', 'fmm', FMM_CFG, 'true', None),
        ('fm_norm_post', 'fmm', FMM_CFG, 'true', 'false'),
    ]
    for name, kind, cfg, tp, vit in jobs:
        out = OUTDIR / f'{name}.csv'
        if out.exists():
            print(f'{name}: exists')
            continue
        if kind == 'cmm':
            text = R._config(cfg, out)
            text = S._set_key(text, 'tw_use_viterbi', vit)
        else:
            text = R._config(cfg, out, **({'viterbi': vit} if vit else {}))
        text = S._set_key(text, 'tp_normalized', tp)
        ok = _run(kind, text, out)
        n = sum(1 for _ in open(out)) - 1 if out.exists() else 0
        print(f'{name}: {"ok" if ok else "RUN FAILED"}, {n} rows -> {out}')
        if not ok:
            return 1
    return 0


# (cell label, emission, transition, {estimator: (file, column, use_seq)})
def cells():
    fmm = dict(delta=(DELIVERED / 'fmm.csv', 'trustworthiness', True),
               alpha=(DELIVERED / 'fmm_post.csv', 'trustworthiness', True),
               smoothed=(DELIVERED / 'fmm.csv', 'trustworthiness_smoothed', True))
    fmm_norm = dict(delta=(OUTDIR / 'fm_norm.csv', 'trustworthiness', True),
                    alpha=(OUTDIR / 'fm_norm_post.csv', 'trustworthiness', True),
                    smoothed=(OUTDIR / 'fm_norm.csv', 'trustworthiness_smoothed', True))
    cmm = dict(delta=(DELIVERED / 'cmm_viterbi.csv', 'trustworthiness', False),
               alpha=(DELIVERED / 'cmm.csv', 'trustworthiness', False),
               smoothed=(DELIVERED / 'cmm.csv', 'trustworthiness_smoothed', False))
    cmm_raw = dict(delta=(OUTDIR / 'cm_raw_viterbi.csv', 'trustworthiness', False),
                   alpha=(OUTDIR / 'cm_raw.csv', 'trustworthiness', False),
                   smoothed=(OUTDIR / 'cm_raw.csv', 'trustworthiness_smoothed', False))
    return [('constant sigma, raw', '100 m', 'raw', fmm),
            ('constant sigma, row-norm', '100 m', 'row-norm', fmm_norm),
            ('per-epoch sigma, raw', 'per-epoch', 'raw', cmm_raw),
            ('per-epoch sigma, row-norm', 'per-epoch', 'row-norm', cmm)]


def metrics(path, col, gt, ts_by_seq, use_seq):
    res = R.evaluate(path, gt, col, ts_by_seq if use_seq else None)
    res['n_eval'] = res.pop('n')  # evaluate()'s key; the table header says n eval
    y, s = res['labels'], res['scores']
    res['log'] = v.log_score(y, s)
    wrong = y == 0.0
    res['n_wrong'] = int(wrong.sum())
    res['wrong_ge_09'] = int((wrong & (s >= 0.9)).sum())
    return res


COLS = (('n_eval', 'n eval', '{:d}'), ('acc', 'acc %', '{:.2f}'), ('ece', 'ECE', '{:.4f}'),
        ('auc', 'AUC', '{:.4f}'), ('bss', 'BSS', '{:+.3f}'), ('sep', 'sep', '{:.3f}'),
        ('sat', 'sat %', '{:.1f}'), ('n_wrong', 'wrong', '{:d}'),
        ('wrong_ge_09', 'wrong>=.9', '{:d}'))


def main():
    if make_runs():
        return 1
    gt, ts_by_seq = R.load_gt()
    out = []
    hdr = f'{"cell":<28}{"estimator":<10}' + ''.join(f'{c[1]:>10}' for c in COLS)
    print('\n' + '=' * len(hdr))
    print(f'REAL VEHICLE, {len(gt):,} evaluable epochs -- emission x transition factor')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for label, emis, trans, arms in cells():
        cell = dict(cell=label, emission=emis, transition=trans, arms={})
        for i, key in enumerate(('delta', 'alpha', 'smoothed')):
            path, col, use_seq = arms[key]
            if not path.exists():
                print(f'{label:<28}{key:<10}  missing {path}')
                continue
            m = metrics(path, col, gt, ts_by_seq, use_seq)
            # The aligned arrays and the per-trajectory map are for callers that need
            # another metric; they are ndarrays and do not belong in the JSON artefact.
            cell['arms'][key] = {k: val for k, val in m.items()
                                 if k not in ('labels', 'scores', 'per_traj')}
            head = f'{label:<28}' if i == 0 else f'{"":<28}'
            print(f'{head}{key:<10}' + ''.join(f'{f.format(m[k]):>10}' for k, _, f in COLS))
        print()
        out.append(cell)
    json.dump(out, open(OUTDIR / 'tp_attribution.json', 'w'), indent=1)
    print(f'wrote {OUTDIR / "tp_attribution.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())