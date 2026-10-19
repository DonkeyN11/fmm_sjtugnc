#!/usr/bin/env python3
"""D10b (manuscript R06): the paper-level held-out metric, one matcher run per fold.

The constant-level check (diag_loto.py) shows the heading constants move a lot between
folds: c 0.202-0.428, b 0.0016-0.0114 rad^2. The manuscript's sentence, however, is about
a held-out *metric*, so the only way to test it is to run the matcher with each fold's
constants and read the metrics on that fold's held-out trajectory.

Six folds x two runs (the delivered config writes the filtered/smoothed arms; a second run
carries tw_use_viterbi for the delta arm), all from input/config/paper1_cmm_scalar.xml with
exactly direction_kappa_scale and direction_kappa_floor substituted. Outputs go to
diag2/loto_runs/, never over the delivered archive.

Usage: python3 experiments/scripts/diag_loto_rerun.py
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R        # noqa: E402
import verify_paper_numbers as v     # noqa: E402

DIAG2 = BASE / 'data/archive_20261019/diag2'
RUNS = DIAG2 / 'loto_runs'
SRC = BASE / 'input/config/paper1_cmm_scalar.xml'
OLD_OUT = '<file>data/archive_20260929/scalar/cmm.csv</file>'
VIT_ANCHOR = '<filtered>true</filtered>'
VIT_LINE = '<filtered>true</filtered>\n    <tw_use_viterbi>true</tw_use_viterbi>'


def build(name, c, b, viterbi):
    text = SRC.read_text(encoding='utf-8')
    for key, val in (('direction_kappa_scale', c), ('direction_kappa_floor', b)):
        old = f'<{key}>'
        i = text.index(old)
        j = text.index(f'</{key}>', i)
        assert text.count(old) == 1, key
        text = text[:i] + f'<{key}>{val!r}' + text[j:]
    assert text.count(OLD_OUT) == 1
    out = RUNS / f'{name}.csv'
    text = text.replace(OLD_OUT, f'<file>{out.relative_to(BASE)}</file>')
    if viterbi:
        assert text.count(VIT_ANCHOR) == 1
        text = text.replace(VIT_ANCHOR, VIT_LINE)
    cfg = RUNS / f'{name}.xml'
    cfg.write_text(text, encoding='utf-8')
    return cfg, out


def run(name, c, b, viterbi):
    cfg, out = build(name, c, b, viterbi)
    if out.exists() and out.stat().st_size > 0:
        return out
    log = RUNS / f'{name}.log'
    with open(log, 'w') as lf:
        rc = subprocess.run([str(BASE / 'build/cmm'), str(cfg)], cwd=BASE,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f'{name}: RUN FAILED rc={rc}')
        return None
    print(f'{name}: ok rows={sum(1 for _ in open(out)) - 1}', flush=True)
    return out


def traj_metrics(path, col, gt, tid):
    """ECE / accuracy on one trajectory, through the paper's own join and metric."""
    run = R.load_run(path)
    y, s = [], []
    ok = n = 0
    for key, (ge, _x, _y) in gt.items():
        if key[0] != tid:
            continue
        r = run.get(key)
        if r is None:
            continue
        n += 1
        corr = v.is_edge_match(r['cpath'].strip(), ge)
        ok += int(corr)
        try:
            sc = float(r[col])
        except (KeyError, ValueError):
            continue
        if sc <= -900.0:
            continue
        y.append(1.0 if corr else 0.0)
        s.append(sc)
    y, s = np.array(y), np.array(s)
    return dict(n=n, acc=100.0 * ok / n if n else float('nan'),
                n_scored=len(s), ece=v.ece(s, y) if len(s) > 10 else float('nan'))


def main():
    DIAG2.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [r for r in csv.DictReader(open(DIAG2.parent / 'diag' / 'loto.csv'))
             if r['fold'] not in ('ALL(insample)',)]
    gt, _ = R.load_gt()
    delivered = R.OUTDIR
    rows = []
    for f in folds:
        held, c, b = f['fold'], float(f['c']), float(f['b_rad2'])
        main_csv = run(f'fold{held}', c, b, False)
        vit_csv = run(f'fold{held}_viterbi', c, b, True)
        if main_csv is None:
            continue
        rec = dict(fold=held, c=c, b_rad2=b, n_loto=int(f['n_test']))
        for arm, (path, col) in {'alpha': (main_csv, 'trustworthiness'),
                                 'alphabeta': (main_csv, 'trustworthiness_smoothed'),
                                 'delta': (vit_csv, 'trustworthiness')}.items():
            if path is None:
                continue
            m = traj_metrics(path, col, gt, held)
            d = traj_metrics(delivered / ('cmm_viterbi.csv' if arm == 'delta' else 'cmm.csv'),
                             col, gt, held)
            rec[f'{arm}_acc_loto'] = m['acc']
            rec[f'{arm}_acc_delivered'] = d['acc']
            rec[f'{arm}_ece_loto'] = m['ece']
            rec[f'{arm}_ece_delivered'] = d['ece']
            rec[f'{arm}_d_ece'] = (m['ece'] - d['ece']
                                   if m['ece'] == m['ece'] and d['ece'] == d['ece'] else float('nan'))
            rec[f'{arm}_n'] = m['n_scored']
        rows.append(rec)
        print(f'fold {held}: c={c:.3f} b={b:.5f} '
              + ' '.join(f'{a}_dECE={rec.get(f"{a}_d_ece", float("nan")):+.5f}'
                         for a in ('delta', 'alpha', 'alphabeta')), flush=True)
    with open(DIAG2 / 'd10_rerun.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {DIAG2 / "d10_rerun.csv"} ({len(rows)} folds)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
