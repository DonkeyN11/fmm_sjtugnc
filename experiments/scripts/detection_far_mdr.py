#!/usr/bin/env python3
"""The mismatch-detection reading of the score, in the field's own units: MDR at stated
false-alarm rates, six arms, real data.

WHY THIS EXISTS. The paper reports detection through AUC and separation, which rank the
arms but name no operating point. The monitoring consumer this paper's motivation
invokes -- flag a match, fall back, or trust it -- asks the operational question:
at an alarm budget of 0.5/1/2/5 per cent false alarms, how often does the score fail
to catch a wrong match? That number does not appear anywhere in the paper, and without
it the "mismatch detection" application is asserted rather than reported.

THE DETECTOR IS THE SCORE ITSELF, ALARMED LOW: a mismatch alarm fires when
tw < tau. False alarm = alarm on a correct epoch; missed detection = no alarm on a
wrong epoch. For each target FAR the operating point is the largest FAR at or below
the target (the most sensitive point inside the budget), and tau is printed so the
threshold is reproducible. AUC is reported beside it as the rank summary the tables
already quote, so this table cannot disagree with tab:audit.

ARMS. The same six as tab:audit, imported from three_estimator_tables' arm
definitions rather than restated, so the join rules (CaMM on timestamp, FMM through
seq->timestamp) are the ones the manuscript's tables use.

Usage:
    python3 experiments/scripts/detection_far_mdr.py
Output: printed table + data/archive_20261012/detection_far_mdr.json
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R              # noqa: E402
import verify_paper_numbers as v           # noqa: E402
import three_estimator_tables as T         # noqa: E402

TARGETS = (0.5, 1.0, 2.0, 5.0)             # per cent false alarms
OUT = Path('/home/ncz/fmm_sjtugnc/data/archive_20261012/detection_far_mdr.json')


def load_arm(gt, ts_by_seq, fname, col, use_seq):
    path = R.OUTDIR / fname
    if not path.exists():
        raise SystemExit(f'missing run {path}; produce it with real_final_tables.py')
    _, _, labels, scores, _, _, _ = R._pairs(
        path, gt, col, ts_by_seq if use_seq else None)
    return labels, np.asarray(scores)


def operating_points(labels, scores):
    """(FAR %, MDR %, tau) at each target, plus the AUC and the counts."""
    correct = labels == 1.0
    wrong = ~correct
    n_c, n_w = int(correct.sum()), int(wrong.sum())
    taus = np.unique(scores)
    out = []
    for target in TARGETS:
        best = None
        for tau in taus:
            alarm = scores < tau
            far = 100.0 * float((alarm & correct).sum()) / max(n_c, 1)
            if far <= target:
                mdr = 100.0 * float((~alarm & wrong).sum()) / max(n_w, 1)
                if best is None or far > best[0]:
                    best = (far, mdr, float(tau))
        out.append(best if best else (0.0, 100.0, float(taus.max())))
    return dict(rows=[dict(target=t, far=r[0], mdr=r[1], tau=r[2])
                      for t, r in zip(TARGETS, out)],
                auc=v.auc(labels, scores), n_correct=n_c, n_wrong=n_w)


def main():
    gt, ts_by_seq = R.load_gt()
    result = {'n_evaluable': len(gt), 'arms': {}}
    hdr = f"{'arm':<26}{'AUC':>7}" + ''.join(f'{"MDR@FAR=" + f"{t:g}%":>13}'
                                             for t in TARGETS)
    print(hdr)
    print('-' * len(hdr))
    for block, arms in (('proposed', T.REAL_PROPOSED), ('baseline', T.REAL_BASELINE)):
        for label, fname, col, use_seq in arms:
            labels, scores = load_arm(gt, ts_by_seq, fname, col, use_seq)
            res = operating_points(labels, scores)
            result['arms'][f'{block}/{label}'] = res
            print(f"{block + ' ' + label:<26}{res['auc']:>7.3f}" +
                  ''.join(f"{r['mdr']:>13.2f}" for r in res['rows']))
    # the thresholds, so the operating points are reproducible
    print('\ntau at each operating point:')
    for name, res in result['arms'].items():
        print(f"  {name:<40}" + ' '.join(f"{r['target']:g}%:{r['tau']:.6g}"
                                         for r in res['rows']))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
