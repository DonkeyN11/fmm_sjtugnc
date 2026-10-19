#!/usr/bin/env python3
"""Section 4's operating-point readings: four sweeps over the six self-consistent arms.

THE ARMS AND THEIR LABEL RULE ARE IMPORTED, NOT RESTATED. online_audit_selfconsistent is
the file that implements the ruling: an online score describes the online decision
i_on = argmax_j delta_t(j), so its outcome is whether THAT decision is correct, while the
batch alpha-beta arm keeps the batch path's correctness. `load_model` / `arm_metrics`
give all six arms on the same 13,069 epochs in the same order, so every table below is
paired by construction.

THE FOUR TABLES
  bin_sweep.csv        ECE at B = 5, 10, 15, 20 equal-width bins, last bin right-closed
                       (verify_paper_numbers.ece: [b/B,(b+1)/B) with [.,1] closed last)
  log_sweep.csv        mean log score at clip = 1e-12 .. 1e-4, beside each arm's own
                       base-rate constant and the skill (log - const_log)
  certification.csv    share certified at tau (score >= tau) and the error rate among the
                       certified, with a trajectory-clustered 95 % bootstrap interval
  detection.csv        MDR at FAR <= 0.5/1/2/5 %, threshold printed, via the repo's own
                       detection_far_mdr.operating_points (imported, not re-implemented)

GATE BEFORE ANY NUMBER IS WRITTEN. The loader must reproduce WP1a's archived
audit_table_selfconsistent_full.csv: ECE at B = 10 and the log score at the default clip
are compared against that file for all six arms, and the script exits on a mismatch. If
this gate fails, nothing below it means anything.

Usage: python3 experiments/scripts/diag_operating_points.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import verify_paper_numbers as v            # noqa: E402
import real_final_tables as rft             # noqa: E402
import online_audit as oa                   # noqa: E402
import online_audit_selfconsistent as SC    # noqa: E402
import detection_far_mdr as DFM             # noqa: E402

OUT = BASE / 'data/archive_20261019/diag3'
AR = BASE / 'data/archive_20261019/online'
SCALAR = BASE / 'data/archive_20260929/scalar'
BINS = (5, 10, 15, 20)
CLIPS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
TAUS = (0.5, 0.9, 0.99, 1.0)
SEED = oa.SEED          # 20261019, the same seed as intervals_selfconsistent.csv
NBOOT = oa.NBOOT


def archive_values():
    """(model, arm) -> {statistic: value} from WP1a's archived table."""
    out = defaultdict(dict)
    path = AR / 'audit_table_selfconsistent_full.csv'
    for r in csv.DictReader(open(path, newline='')):
        out[(r['model'], r['arm'])][r['statistic']] = float(r['value'])
    return out


def const_log(labels):
    p = float(np.mean(labels))
    return float(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def cert_error(lab, sc, tau):
    m = sc >= tau
    if not m.any():
        return float('nan')
    return 100.0 * float(1.0 - lab[m].mean())


def main():
    oa.gt, oa.ts_by_seq = rft.load_gt()
    models = {
        'cmm': SC.load_model('cmm', AR / 'raw_cmm.csv', SCALAR / 'cmm.csv', 'timestamp'),
        'fmm': SC.load_model('fmm', AR / 'raw_fmm.csv', SCALAR / 'fmm.csv', 'seq'),
    }
    ref = archive_values()

    # ------------------------------------------------------------------ gate
    bad = []
    for model, d in models.items():
        for arm in ('online_delta', 'online_alpha', 'batch_alphabeta'):
            lab, sc = SC.arm_metrics(d, arm)
            e10 = v.ece(sc, lab, n=10)
            lg = v.log_score(lab, sc)
            r = ref[(model, arm)]
            if abs(e10 - r['ece']) > 1e-6:
                bad.append((model, arm, 'ece', e10, r['ece']))
            if abs(lg - r['log']) > 1e-6:
                bad.append((model, arm, 'log', lg, r['log']))
    if bad:
        for b in bad:
            print('  GATE MISMATCH', b)
        sys.exit('FATAL: loader does not reproduce audit_table_selfconsistent_full.csv')
    print('gate: 6 arms reproduce WP1a ece(B=10) and log to 1e-6')

    OUT.mkdir(parents=True, exist_ok=True)
    arms = [(m, a) for m in ('cmm', 'fmm')
            for a in ('online_delta', 'online_alpha', 'batch_alphabeta')]
    data = {}
    for model, arm in arms:
        d = models[model]
        lab, sc = SC.arm_metrics(d, arm)
        data[(model, arm)] = (d['traj'], lab, sc)
        assert len(lab) == 13069, (model, arm, len(lab))

    # -------------------------------------------------------------- 1. bin sweep
    rows = []
    for model, arm in arms:
        _t, lab, sc = data[(model, arm)]
        for B in BINS:
            rows.append(dict(model=model, arm=arm, labels=SC.LABEL_ONLINE
                             if arm.startswith('online') else SC.LABEL_BATCH,
                             bins=B, n=len(lab), ece=round(float(v.ece(sc, lab, n=B)), 8)))
    with open(OUT / 'bin_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n-- bin sweep (ECE)')
    print(f'{"arm":<26}' + ''.join(f'{"B=" + str(b):>11}' for b in BINS))
    for model, arm in arms:
        sub = [r for r in rows if r['model'] == model and r['arm'] == arm]
        print(f'{model + " " + arm:<26}' + ''.join(f'{r["ece"]:>11.6f}' for r in sub))
    ab = [r['ece'] for r in rows if r['model'] == 'cmm' and r['arm'] == 'batch_alphabeta']
    print(f'   cmm batch_alphabeta range over B: {min(ab):.6f} .. {max(ab):.6f}')

    # -------------------------------------------------------------- 2. log sweep
    rows = []
    for model, arm in arms:
        _t, lab, sc = data[(model, arm)]
        cl = const_log(lab)
        for clip in CLIPS:
            lg = float(v.log_score(lab, sc, eps=clip))
            rows.append(dict(model=model, arm=arm,
                             labels=SC.LABEL_ONLINE if arm.startswith('online') else SC.LABEL_BATCH,
                             clip=f'{clip:.0e}', log=round(lg, 8),
                             const_log=round(cl, 8), skill=round(lg - cl, 8)))
    with open(OUT / 'log_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n-- log sweep (cmm; const_log in brackets)')
    print(f'{"arm":<26}' + ''.join(f'{"clip " + c:>12}' for c in
                                   ('1e-12', '1e-10', '1e-8', '1e-6', '1e-4')))
    for model, arm in arms:
        if model != 'cmm':
            continue
        sub = [r for r in rows if r['model'] == model and r['arm'] == arm]
        print(f'{arm:<26}' + ''.join(f'{r["log"]:>12.6f}' for r in sub)
              + f'   const_log={sub[0]["const_log"]:+.6f}')

    # ---------------------------------------------------------- 3. certification
    rng = np.random.default_rng(SEED)
    rows = []
    for model, arm in arms:
        traj, lab, sc = data[(model, arm)]
        for tau in TAUS:
            m = sc >= tau
            n_cert = int(m.sum())
            rate = 100.0 * n_cert / len(sc)
            err = cert_error(lab, sc, tau)
            # Nan-safe: with few certified epochs a replicate can certify none, and a plain
            # percentile then returns NaN for the whole interval. The fraction of such
            # replicates is recorded rather than hidden.
            rng_t = np.random.default_rng(SEED)
            uniq = sorted(set(traj))
            groups = {t: np.where(traj == t)[0] for t in uniq}
            reps = np.empty(NBOOT)
            for b in range(NBOOT):
                pick = rng_t.choice(len(uniq), len(uniq), replace=True)
                idx = np.concatenate([groups[uniq[i]] for i in pick])
                reps[b] = cert_error(lab[idx], sc[idx], tau)
            nan_frac = float(np.isnan(reps).mean())
            if n_cert and nan_frac < 0.5:
                lo = float(np.nanpercentile(reps, 2.5))
                hi = float(np.nanpercentile(reps, 97.5))
            else:
                lo = hi = float('nan')
            rows.append(dict(model=model, arm=arm,
                             labels=SC.LABEL_ONLINE if arm.startswith('online') else SC.LABEL_BATCH,
                             tau=tau, n=len(sc), n_certified=n_cert,
                             cert_rate_pct=round(rate, 4),
                             err_rate_pct=round(err, 4) if err == err else '',
                             err_ci_lo=round(lo, 4) if lo == lo else '',
                             err_ci_hi=round(hi, 4) if hi == hi else '',
                             boot_nan_frac=nan_frac,
                             ci='trajectory-clustered bootstrap 95%, seed '
                                + str(SEED) + ', ' + str(NBOOT) + ' reps'))
    with open(OUT / 'certification.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n-- certification (rate % / error % [95% CI])')
    for tau in (0.9, 1.0):
        print(f'   tau={tau}')
        for model, arm in arms:
            r = [x for x in rows if x['model'] == model and x['arm'] == arm
                 and x['tau'] == tau][0]
            ci = (f'[{r["err_ci_lo"]:.2f},{r["err_ci_hi"]:.2f}]'
                  if r['err_ci_lo'] != '' else 'n/a')
            print(f'     {model + " " + arm:<26} cert={r["cert_rate_pct"]:>7.2f} %'
                  f'  err={r["err_rate_pct"] if r["err_rate_pct"] != "" else "n/a":>7}'
                  f'  {ci}')

    # ------------------------------------------------------------- 4. detection
    rows = []
    for model, arm in arms:
        _t, lab, sc = data[(model, arm)]
        res = DFM.operating_points(lab, sc)
        for r in res['rows']:
            rows.append(dict(model=model, arm=arm,
                             labels=SC.LABEL_ONLINE if arm.startswith('online') else SC.LABEL_BATCH,
                             far_budget=r['target'], far_achieved=round(r['far'], 4),
                             mdr=round(r['mdr'], 4), thr=r['tau'],
                             auc=round(res['auc'], 6)))
    with open(OUT / 'detection.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n-- detection (MDR % at FAR <=)')
    print(f'{"arm":<26}' + ''.join(f'{"FAR=" + f"{t:g}%":>12}' for t in DFM.TARGETS))
    for model, arm in arms:
        sub = [r for r in rows if r['model'] == model and r['arm'] == arm]
        print(f'{model + " " + arm:<26}' + ''.join(f'{r["mdr"]:>12.2f}' for r in sub))

    print(f'\nwrote {OUT}/bin_sweep.csv, log_sweep.csv, certification.csv, detection.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
