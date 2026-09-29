#!/usr/bin/env python3
"""Rebuild tab:accounting -- the epoch flow from collection to evaluation.

WHY THIS FILE EXISTS. The table's numbers (15,421 collected, 2,361 non-evaluable,
13,069 evaluable) had no producer anywhere in the repository: `git grep -E '15421|15,421'`
and `'2361|2,361'` both return nothing over experiments/, tests/ and input/. They were
produced by a script that was never kept, which is the same defect that
ablate_real_components.py was written to repair for tab:ablation -- a table whose numbers
reproduce but whose generator is not tracked, so a re-run cannot be told from a stale copy.

WHAT EACH COLUMN IS.

  Collected      rows in the input table (data/real_vehicle/hainan_06/cmm_input_points_k1.csv)
                 for that trajectory. This is what was handed to the matcher.
  Non-evaluable  epochs whose ground-truth edge is 0, -1 or absent, broken down by which:
                   0  = outside the mapped network, or on a road the map does not contain
                   -1 = no RTK label at all (the receiver was not fixed)
                   -- = the epoch is not in the union the alignment built
                 The distinction matters because the three have different causes and only
                 the first is a statement about the map.
  Evaluable      Collected - Non-evaluable. These are the epochs every metric in Section V
                 is computed over.
  CaMM acc.      segment accuracy on the evaluable epochs of that trajectory, at whatever
                 operating point real_final_tables.CMM_CONFIG names.

Trajectory 12 is excluded from the evaluation (a policy, not a data defect -- see
aligned.csv's own note), so it is reported on its own line and not in the subtotal.

Usage: python3 experiments/scripts/epoch_accounting.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402


def main():
    # --- collected: what the matcher was given -------------------------------
    collected = defaultdict(int)
    with open(BASE / R.GPS, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            collected[r['id'].strip()] += 1

    # --- non-evaluable, by cause --------------------------------------------
    non_eval = defaultdict(lambda: defaultdict(int))
    with open(R.ALIGNED, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            tid = r['id'].strip()
            e = r['gt_edge'].strip()
            if e not in ('0', '-1', ''):
                continue
            non_eval[tid]['0' if e == '0' else '-1' if e == '-1' else '--'] += 1

    # --- accuracy per trajectory, from the operating point's own run --------
    gt, ts_by_seq = R.load_gt()
    run = R.load_run(R.OUTDIR / 'cmm.csv', None)
    per = defaultdict(lambda: [0, 0])
    for key, (gt_edge, _, _) in gt.items():
        r = run.get(key)
        if r is None:
            continue
        per[key[0]][0] += 1
        per[key[0]][1] += int(v.is_edge_match(r['cpath'].strip(), gt_edge))

    # --- the alignment's own row count, per trajectory -----------------------
    aligned = defaultdict(int)
    with open(R.ALIGNED, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            aligned[r['id'].strip()] += 1

    tids = sorted(t for t in collected if t not in v.EXCLUDE_TRAJ)
    hdr = (f'{"Traj.":<8}{"Collected":>11}{"Non-eval.":>11}{"(0/-1/--)":>12}'
           f'{"Evaluable":>11}{"CaMM acc. %":>13}')
    print(hdr)
    print('-' * len(hdr))
    sc = sn = se = sk = 0
    for t in tids:
        n0 = non_eval[t]['0']
        n1 = non_eval[t]['-1']
        nd = non_eval[t]['--']
        ne = n0 + n1 + nd
        # Evaluable comes from the ALIGNMENT, not from Collected - Non-eval. The two
        # differ because aligned.csv enumerates the union of four sources and so holds
        # rows the input table does not: 15,430 against 15,421 over the six evaluated
        # trajectories. Subtracting would therefore print a column that does not match
        # the population every metric in Section V is computed over.
        ev = aligned[t] - ne
        m, ok = per.get(t, (0, 0))
        acc = 100.0 * ok / m if m else float('nan')
        sc += collected[t]; sn += ne; se += ev; sk += ok
        print(f'{t:<8}{collected[t]:>11,}{ne:>7,} ({n0}/{n1}/{nd}){ev:>6,}{acc:>12.2f}')
    print('-' * len(hdr))
    print(f'{"Subtotal":<8}{sc:>11,}{sn:>7,}{"":>12}{se:>6,}{100.0 * sk / se:>12.2f}')

    ex = sorted(t for t in collected if t in v.EXCLUDE_TRAJ)
    for t in ex:
        n0 = non_eval[t]['0']; n1 = non_eval[t]['-1']; nd = non_eval[t]['--']
        ne = n0 + n1 + nd
        print(f'{t + " (excl.)":<8}{collected[t]:>11,}{ne:>7,} ({n0}/{n1}/{nd})'
              f'{aligned[t] - ne:>6,}{"--":>12}')

    al_all = sum(aligned[t] for t in tids)
    print(f'\nnote: the three columns do not sum, and are not meant to. Collected counts '
          f'the INPUT\n      TABLE ({sc:,}); Non-eval and Evaluable count the ALIGNMENT '
          f'({al_all:,}), whose union\n      holds {al_all - sc:,} rows the input table does '
          f'not. Any statement that the columns\n      close is wrong by that margin.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
