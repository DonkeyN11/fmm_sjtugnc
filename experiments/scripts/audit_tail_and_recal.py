#!/usr/bin/env python3
"""Rebuild tab:tail and tab:recal -- where the score is overconfident, and what a
post-hoc fix on the baseline buys.

WHY THIS FILE EXISTS. Both tables entered the manuscript during the 2026-09-28 review
round with no producer in the repository: `git grep -E '4185|4,185'`, `'3294|3,294'` and
`'isotonic'` over experiments/, tests/ and input/ all return nothing. That is the same
defect ablate_real_components.py and epoch_accounting.py were written to repair -- a
table whose numbers reproduce but whose generator is not tracked, so nobody can tell a
re-run from a stale copy. Neither table can be regenerated at a new operating point
without one, which is what made this file necessary rather than merely tidy.

TAB:TAIL answers the two questions the binned calibration error cannot.

  * Where is the score not saturated? A reliability diagram over mostly-1.000 scores
    measures almost nothing, so the calibration is recomputed on the epochs scored below
    one, and below 0.999. If the score is calibrated only because it is accurate, the
    sub-saturated ECE is the column that shows it.
  * How bad is the tail? `wrong & > 0.9` counts epochs that are wrong AND reported above
    0.9, as a share of that arm's own errors. A score can pass a binned audit and still
    put a third of its mistakes above 0.9, and the log score is the only column that
    prices that.

TAB:RECAL gives the baseline the strongest post-hoc repair available and asks whether it
reaches the proposed arm. Two monotone fits, temperature scaling (one parameter,
p -> sigmoid(logit(p)/T)) and isotonic regression (free-form, monotone), both trained by
trajectory leave-one-out: the fit for a held-out trajectory sees only the other five.
Leave-one-out rather than a random split because the six trajectories are the independent
unit -- a random epoch split would let epochs of one trajectory train a fit applied to
their own neighbours, which is the same leakage the clustered intervals exist to avoid.

**THE TEMPERATURE ROW IN THE MANUSCRIPT DOES NOT REPRODUCE, and this script is what
found it (2026-09-29).** It read ECE 0.034 / Brier 0.048 / BSS -0.10 / log -0.180 /
AUC 0.758. Neither of those two groups of numbers is reachable from the baseline's native
score:

  * NO temperature gives ECE 0.034. Sweeping T over [0.1, 10] the expected calibration
    error bottoms out at 0.188 (T = 0.7) and is 0.193 at T = 1, the raw score. A
    one-parameter monotone transform of this score cannot reach a quarter of the
    claimed figure, in sample or out of it.
  * NO temperature changes the AUC AT ALL, in population. A global monotone map preserves
    the ranking exactly, and a per-fold fit preserves each fold's ranking, so the pooled
    AUC can move only through the pooling of differently-scaled folds -- and the fitted
    temperatures here span just 1.33 to 1.78, far too narrow to produce 0.758.

What the row should read is the measurement below: temperature scaling makes the binned
calibration WORSE, and the reason is a genuine property of the score rather than a bug.
The baseline's native score has mean 0.761 against a base rate of 0.947, i.e. it is
UNDER-confident on average, but its 5.3 % of confident errors dominate the likelihood, so
an NLL-fitted temperature flattens it further and moves mass away from where the
correct answers are. Fitting to ECE instead does not rescue it either (0.188). Only the
free-form monotone fit reaches a calibrated score, and it stops at the filtering arm's
level rather than the smoothed arm's.

The isotonic row DOES reproduce exactly (0.040 / 0.050 / -0.14 / -0.214 / 0.728), which is
what establishes that the input, the join and the metric definitions here are the same
ones the manuscript used. The defect is confined to one row.

The AUC column carries a dagger in the manuscript because both fits are monotone WITHIN a
fold and therefore preserve that fold's ranking exactly; the pooled AUC moves only because
the folds are pooled on different scales. It is not evidence that recalibration hurts
discrimination.

Usage: python3 experiments/scripts/audit_tail_and_recal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

# (label, file, score column, join through the seq->timestamp map)
ARMS = [
    ('CaMM filtering', 'cmm.csv', 'trustworthiness', False),
    ('CaMM smoothed', 'cmm.csv', 'trustworthiness_smoothed', False),
    ('FMM native', 'fmm.csv', 'trustworthiness', True),
    ('FMM smoothed', 'fmm.csv', 'trustworthiness_smoothed', True),
]


def series(path, col, ts_by_seq, gt):
    """(labels, scores, trajectory ids) over the epochs this run answers."""
    run = R.load_run(path, ts_by_seq)
    y, s, t = [], [], []
    for key, (gt_edge, _, _) in gt.items():
        r = run.get(key)
        if r is None:
            continue
        try:
            p = float(r[col])
        except (ValueError, KeyError):
            continue
        if p <= -900.0:
            continue
        y.append(1.0 if v.is_edge_match(r['cpath'].strip(), gt_edge) else 0.0)
        s.append(p)
        t.append(key[0])
    return np.array(y), np.array(s), np.array(t)


def _ece(y, s):
    return v.ece(s, y) if len(y) and 0 < y.mean() < 1 else float('nan')


def tail(y, s):
    out = {}
    for name, thr in (('<1', 1.0), ('<0.999', 0.999)):
        m = s < thr
        out[name] = (int(m.sum()), _ece(y[m], s[m]))
    m = s < 1.0
    out['auc_sub'] = v.auc(y[m], s[m]) if m.sum() and 0 < y[m].mean() < 1 else float('nan')
    wrong = y == 0
    hi = s > 0.9
    out['wrong_hi'] = int((wrong & hi).sum())
    out['wrong_hi_one'] = int((wrong & (s >= 1.0)).sum())
    out['share'] = 100.0 * (wrong & hi).sum() / wrong.sum() if wrong.sum() else float('nan')
    return out


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _fit_T(p, y, lo=-3.0, hi=3.0):
    """Temperature by 1-D grid then bisection on the NLL of sigmoid(logit(p)/T)."""
    z = _logit(p)

    def nll(logT):
        T = np.exp(logT)
        q = 1.0 / (1.0 + np.exp(-z / T))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    grid = np.linspace(lo, hi, 241)
    best = min(grid, key=nll)
    a, b = best - (hi - lo) / 240, best + (hi - lo) / 240
    for _ in range(60):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        if nll(m1) < nll(m2):
            b = m2
        else:
            a = m1
    return float(np.exp((a + b) / 2))


def loo_recal(y, s, t, kind):
    """Pooled out-of-fold recalibration, folds = trajectories."""
    from sklearn.isotonic import IsotonicRegression
    out = np.empty_like(s, dtype=float)
    for held in np.unique(t):
        tr, te = t != held, t == held
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            out[te] = s[te]
            continue
        if kind == 'temperature':
            T = _fit_T(s[tr], y[tr])
            out[te] = 1.0 / (1.0 + np.exp(-_logit(s[te]) / T))
        else:
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso.fit(s[tr], y[tr])
            out[te] = iso.predict(s[te])
        out[te] = np.clip(out[te], 0.0, 1.0)
    return out


def row(label, y, s):
    p = float(y.mean())
    brier = float(((s - y) ** 2).mean())
    return (f'{label:<24}{v.ece(s, y):>8.3f}{brier:>9.3f}'
            f'{1 - brier / (p * (1 - p)):>+8.2f}{v.log_score(y, s):>+9.3f}'
            f'{v.auc(y, s):>8.3f}')


def main():
    gt, ts_by_seq = R.load_gt()
    print(f'{len(gt)} evaluable epochs, operating point {R.CMM_CONFIG}\n')
    data = {}
    for lab, f, col, use_seq in ARMS:
        data[lab] = series(R.OUTDIR / f, col, ts_by_seq if use_seq else None, gt)

    print('=== tab:tail ===')
    hdr = (f'{"arm":<16}{"n <1":>8}{"ECE <1":>9}{"n <.999":>10}{"ECE <.999":>10}'
           f'{"AUC sub":>9}{"wrong>0.9":>11}{"=1.0":>7}{"share err":>11}')
    print(hdr)
    print('-' * len(hdr))
    for lab, _, _, _ in ARMS:
        y, s, _ = data[lab]
        d = tail(y, s)
        print(f'{lab:<16}{d["<1"][0]:>8,}{d["<1"][1]:>9.3f}{d["<0.999"][0]:>10,}'
              f'{d["<0.999"][1]:>10.3f}{d["auc_sub"]:>9.3f}{d["wrong_hi"]:>11,}'
              f'{d["wrong_hi_one"]:>7,}{d["share"]:>10.1f}%')

    print('\n=== tab:recal ===')
    hdr = f'{"score":<24}{"ECE":>8}{"Brier":>9}{"BSS":>8}{"log":>9}{"AUC":>8}'
    print(hdr)
    print('-' * len(hdr))
    for kind in ('raw', 'temperature', 'isotonic'):
        y, s, t = data['FMM native']
        fit = s if kind == 'raw' else loo_recal(y, s, t, kind)
        label = 'FMM native, raw' if kind == 'raw' else f'+ {kind} (LOO)'
        print(row(label, y, fit))
    for lab in ('CaMM smoothed', 'CaMM filtering'):
        y, s, _ = data[lab]
        print(row(f'{lab}, raw', y, s))
    print('\n(note: the recalibration fits are monotone within every fold, so a '
          'held-out AUC drop is\na pooling artefact of mixing folds fitted on different '
          'scales, not a loss of ranking.)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
