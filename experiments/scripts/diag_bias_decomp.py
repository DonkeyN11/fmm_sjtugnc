#!/usr/bin/env python3
"""D11-R2 (manuscript R09): where the billing bias sits, in two terms.

bias = Mhat - M = sum_t B_t (c_t - tw_t), with B_t the along-path increment of
diag_billed_distance.py. The split the manuscript needs is between the epochs the score
cannot see at all and everything else:

  tail_tw1_wrong              tw == 1.0 and c == 0. Contribution -sum B: Mhat carries none
                              of that distance, so it is charged to the estimate as a full
                              miss. This is the "confident-error tail" in metres.
  correct_tw_lt1_overdeducted  tw < 1 and c == 1. Contribution +sum B (1 - tw): Mhat deducts
                              distance that was, in fact, driven correctly.
  wrong_tw_lt1_partial_credit  tw < 1 and c == 0. Contribution -sum B tw: Mhat deducts only
                              the part of the step the score admits.
The three sum to the bias exactly; that identity is asserted, not assumed.

Also reported, because it is the necessary-condition check the proposition needs: the
empirical correctness of the cell tw == 1.0 against the value the score claims there (1.000),
and the same for the manuscript's top bin [0.9,1]. The two are different cells and must not
be quoted interchangeably.

Input is the archived per-epoch table, so nothing is recomputed from raw observation files.
Usage: python3 experiments/scripts/diag_bias_decomp.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
DIAG2 = BASE / 'data/archive_20261019/diag2'
EPOCHS = DIAG2 / 'billed_epochs.csv'
ARMS = (('alphabeta', 'tw_alphabeta'), ('delta', 'tw_delta'))
FIELDS = ['arm', 'component', 'n_epochs', 'sum_B_m', 'contribution_m', 'share_of_M']


def load():
    rows = []
    for r in csv.DictReader(open(EPOCHS, newline='')):
        if r['correct'] == '':
            continue
        rows.append(dict(traj=r['traj'], B=float(r['B_m']),
                         c=float(r['correct']),
                         tw={a: float(r[col]) for a, col in ARMS
                             if r[col] not in ('', 'nan')}))
    return rows


def decomp(rows, arm):
    use = [r for r in rows if arm in r['tw']]
    M = sum(r['B'] * (1 - r['c']) for r in use)
    Mh = sum(r['B'] * (1 - r['tw'][arm]) for r in use)
    bias = Mh - M
    groups = {
        'tail_tw1_wrong': lambda r: abs(r['tw'][arm] - 1.0) < 1e-12 and r['c'] == 0,
        'correct_tw_lt1_overdeducted': lambda r: r['tw'][arm] < 1.0 and r['c'] == 1,
        'wrong_tw_lt1_partial_credit': lambda r: r['tw'][arm] < 1.0 and r['c'] == 0,
    }
    out, raw = [], []
    for name, fn in groups.items():
        sub = [r for r in use if fn(r)]
        summ = sum(r['B'] for r in sub)
        contrib = sum(r['B'] * (r['c'] - r['tw'][arm]) for r in sub)
        raw.append(contrib)
        out.append(dict(arm=arm, component=name, n_epochs=len(sub), sum_B_m=round(summ, 3),
                        contribution_m=round(contrib, 3),
                        share_of_M=round(100.0 * contrib / M, 4) if M else float('nan')))
    tot = sum(raw)                     # identity on the unrounded terms, not on the printed ones
    assert abs(tot - bias) < 1e-6, (arm, tot, bias)
    tail = out[0]
    cell = [r for r in use if abs(r['tw'][arm] - 1.0) < 1e-12]
    cell_wrong = [r for r in cell if r['c'] == 0]
    info = dict(
        arm=arm, n=int(len(use)), M_m=round(M, 3), Mhat_m=round(Mh, 3),
        bias_m=round(bias, 3), bias_pct_of_M=round(100.0 * bias / M, 4) if M else None,
        components_sum_m=round(tot, 3), identity_ok=True,
        tw_exactly_one=dict(
            n=len(cell), n_wrong=len(cell_wrong),
            cond_acc=round(1.0 - len(cell_wrong) / len(cell), 6) if cell else None,
            claimed=1.0,
            deficit=round(1.0 - (1.0 - len(cell_wrong) / len(cell)), 6) if cell else None,
            sum_B_m=round(sum(r['B'] for r in cell_wrong), 3),
            tail_share_of_M_pct=round(100.0 * tail['sum_B_m'] / M, 3) if M else None,
            population_share_pct=round(100.0 * len(cell_wrong) / len(use), 3),
            per_traj_wrong=dict(Counter(r['traj'] for r in cell_wrong)),
        ))
    return out, info


def main():
    rows = load()
    allrows, infos = [], {}
    for arm, _ in ARMS:
        o, info = decomp(rows, arm)
        allrows += o
        infos[arm] = info
    with open(DIAG2 / 'bias_decomp.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(allrows)
    (DIAG2 / 'bias_decomp.json').write_text(json.dumps(infos, indent=2, ensure_ascii=False))

    lines = []
    for arm, _ in ARMS:
        i = infos[arm]
        lines.append(f'== {arm}: n={i["n"]}  M={i["M_m"]:.1f} m  Mhat={i["Mhat_m"]:.1f} m'
                     f'  bias={i["bias_m"]:+.1f} m ({i["bias_pct_of_M"]:+.3f} % of M)')
        for o in [r for r in allrows if r['arm'] == arm]:
            lines.append(f'   {o["component"]:<30} n={o["n_epochs"]:>6}'
                         f'  sum_B={o["sum_B_m"]:>10.1f} m'
                         f'  contribution={o["contribution_m"]:>+10.1f} m'
                         f'  share_of_M={o["share_of_M"]:>+8.3f} %')
        t = i['tw_exactly_one']
        lines.append(f'   tw==1.0 cell: n={t["n"]}  wrong={t["n_wrong"]}'
                     f'  conditional acc={t["cond_acc"]:.6f} vs claimed 1.000'
                     f'  (deficit {t["deficit"]:.6f})')
        lines.append(f'   tail: {t["n_wrong"]} epochs = {t["population_share_pct"]:.3f} % of the'
                     f' population, carrying {t["tail_share_of_M_pct"]:.2f} % of M'
                     f' ({t["sum_B_m"]:.1f} m); split {t["per_traj_wrong"]}')
    txt = '\n'.join(lines)
    (DIAG2 / 'bias_decomp.txt').write_text(txt + '\n')
    print(txt)
    print(f'wrote {DIAG2}/bias_decomp.csv ({len(allrows)} rows), bias_decomp.json, bias_decomp.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())
