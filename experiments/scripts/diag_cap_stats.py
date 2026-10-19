#!/usr/bin/env python3
"""D6 (manuscript R20): the candidate cap is not proven unbound by a mean.

Per-epoch candidate counts come from the run's own `candidates` column through
fig_candidate_counts.count_candidates, which counts "(lon,lat,w)" triples rather than
separators (an empty "()" is zero candidates, not one). Two counters are reported side by
side because the manuscript's "11.2 candidates per epoch" is the evaluator's own count
(real_final_tables._n_candidates) on the 13,069 evaluable epochs, while the cap share has
to be read on the same rows to be comparable.

With --cap24 the run at k = 24 (diag_cap24_run.py) is evaluated too, so the cost of raising
the cap is measured for all three arms rather than assumed.

Usage: python3 experiments/scripts/diag_cap_stats.py [--cap24 <csv>] [--cap24v <csv>]
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R          # noqa: E402
import verify_paper_numbers as v       # noqa: E402
import fig_candidate_counts as F       # noqa: E402

CAP = 16
DIAG = BASE / 'data/archive_20261019/diag'
RUN = BASE / 'data/archive_20260929/scalar'


def describe(c, cap=CAP):
    c = np.asarray(c, dtype=int)
    return dict(n_epochs=int(len(c)), mean=round(float(c.mean()), 6),
                median=float(np.median(c)), max=int(c.max()), min=int(c.min()),
                p90=float(np.percentile(c, 90)), p95=float(np.percentile(c, 95)),
                p99=float(np.percentile(c, 99)),
                n_at_cap=int((c >= cap).sum()),
                share_at_cap=round(float((c >= cap).mean()), 6),
                n_above_cap=int((c > cap).sum()),
                hist={str(k): int(n) for k, n in sorted(Counter(c.tolist()).items())})


def arms(paths, gt):
    """ECE / accuracy for the arms a run provides. Accuracy is path-level, not arm-level."""
    out = {}
    for name, (path, col) in paths.items():
        if not path.exists():
            out[name] = None
            continue
        d = R.evaluate(path, gt, col)
        out[name] = dict(n=d['n'], acc=round(d['acc'], 6), ece=round(d['ece'], 8),
                         bss=round(d['bss'], 8), sat=round(d['sat'], 4),
                         cand_mean=round(d['cand'], 6))
    return out


def main():
    gt, _ = R.load_gt()
    res = {'cap': CAP, 'runs': {}}
    for tag, f in (('r100', 'cmm_r100.csv'), ('r200', 'cmm_r200.csv'),
                   ('r300_delivered', 'cmm.csv'), ('r500', 'cmm_r500.csv')):
        p = RUN / f
        if not p.exists():
            continue
        raw = F.count_candidates(p)
        n, ok, lab, sc, errs, cands, per = R._pairs(p, gt, 'trustworthiness')
        res['runs'][tag] = dict(
            file=str(p),
            all_rows_counter=describe(raw),
            evaluable_paper_counter=describe(np.asarray(cands, dtype=int)),
            evaluable_n=int(len(cands)),
        )
    # the delivered run, three arms
    res['arms_cap16'] = arms({
        'delta': (RUN / 'cmm_viterbi.csv', 'trustworthiness'),
        'alpha': (RUN / 'cmm.csv', 'trustworthiness'),
        'alphabeta': (RUN / 'cmm.csv', 'trustworthiness_smoothed'),
    }, gt)
    cap24 = None
    cap24v = None
    for i, a in enumerate(sys.argv):
        if a == '--cap24':
            cap24 = Path(sys.argv[i + 1])
        if a == '--cap24v':
            cap24v = Path(sys.argv[i + 1])
    if cap24 and cap24.exists():
        res['runs']['cap24'] = dict(
            file=str(cap24),
            all_rows_counter=describe(F.count_candidates(cap24)),
            evaluable_paper_counter=describe(np.asarray(
                R._pairs(cap24, gt, 'trustworthiness')[5], dtype=int)),
            evaluable_n=int(len(R._pairs(cap24, gt, 'trustworthiness')[5])))
        pv = cap24v if (cap24v and cap24v.exists()) else None
        res['arms_cap24'] = arms({
            'delta': (pv, 'trustworthiness') if pv else (Path('/nonexistent'), 'x'),
            'alpha': (cap24, 'trustworthiness'),
            'alphabeta': (cap24, 'trustworthiness_smoothed'),
        }, gt)
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / 'cap_stats.json').write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for tag, d in res['runs'].items():
        a = d['all_rows_counter']
        e = d['evaluable_paper_counter']
        print(f'{tag:<15} rows={a["n_epochs"]}  max={a["max"]:>2}'
              f'  mean(all)={a["mean"]:.3f}  at_cap(all)={a["n_at_cap"]}'
              f' ({100*a["share_at_cap"]:.2f}%)  |  mean(evaluable)={e["mean"]:.3f}'
              f'  max_ev={e["max"]}  at_cap(ev)={e["n_at_cap"]} ({100*e["share_at_cap"]:.2f}%)')
    for k in ('arms_cap16', 'arms_cap24'):
        if k not in res:
            continue
        print(f'-- {k}')
        for arm, d in res[k].items():
            if d is None:
                print(f'   {arm:<10} (not run)')
                continue
            print(f'   {arm:<10} n={d["n"]} acc={d["acc"]:.4f} ece={d["ece"]:.8f}'
                  f' bss={d["bss"]:+.6f} sat={d["sat"]:.2f} cand={d["cand_mean"]:.4f}')
    print(f'wrote {DIAG / "cap_stats.json"}')


if __name__ == '__main__':
    main()
