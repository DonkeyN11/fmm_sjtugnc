#!/usr/bin/env python3
"""D10 (manuscript R06): leave-one-trajectory-out fit of the heading constants.

The manuscript's stability evidence is that "leave-one-trajectory-out changes no held-out
metric beyond its third decimal". Two questions follow: what each fold's (c, b) is, and
whether the correctness filter and the iterative step of a fold saw only the training
trajectories.

The sample builder is fit_direction_kappa.samples() -- the same function the delivered fit
uses -- called once per fold on the five training trajectories and once on the held-out one.
The regression is fit_direction_kappa.fit(), so c and b are the manuscript's own estimators
rather than a re-derivation.

The held-out metric has to be one whose units survive: the model predicts
sigma_theta^2 = c * (n^T Sigma n / |v|^2) + b, so the transportable diagnostic is the
standardised residual  z2 = theta^2 / predicted, whose RMS is ~1 when the fold's constants
predict the held-out trajectory's heading scatter. theta RMS and predicted RMS are reported
in degrees beside it.

What the answer to the second question is, stated here so the output can be read: the sample
definition requires the matcher's cpath to equal the ground-truth edge, and the matcher was
run once, with c = 0.227 and b = 0.0084 rad^2 already in its configuration
(paper1_cmm_scalar.xml, printed at the end). No fold re-runs the matcher with that fold's
constants, so which samples survive on the held-out trajectory is decided by a run whose
heading term was fitted on all six. Only the (c, b) regression is refit.

Usage: python3 experiments/scripts/diag_loto.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R        # noqa: E402
import verify_paper_numbers as v     # noqa: E402
import fit_direction_kappa as K      # noqa: E402

DIAG = BASE / 'data/archive_20261019/diag'
RUN = BASE / 'data/archive_20260929/scalar/cmm.csv'
CFG = BASE / 'input/config/paper1_cmm_scalar.xml'


def config_constants():
    t = CFG.read_text(encoding='utf-8')
    return {k: float(re.search(rf'<{k}>([-\d.eE]+)</{k}>', t).group(1))
            for k in ('direction_kappa_scale', 'direction_kappa_floor')}


def metrics(th, x, a, b):
    pred = a * x + b
    z2 = th ** 2 / pred
    return dict(
        theta_rms_deg=float(np.degrees(np.sqrt((th ** 2).mean()))),
        sigma_pred_deg=float(np.degrees(np.sqrt(pred.mean()))),
        z2_rms=float(np.sqrt((z2 ** 2).mean())),
        z2_mean=float(z2.mean()))


def main():
    geo, _ds = K.read_network()
    gt, _ts = R.load_gt()
    gt_edges = {(k[0], k[1]): e for k, (e, _x, _y) in gt.items()}
    obs_rows = list(csv.DictReader(open(K.REAL_OBS, newline=''), delimiter=';'))
    run_rows = list(csv.DictReader(open(RUN, newline=''), delimiter=';'))

    def key_of(r):
        return (r['id'].strip(), v.norm_ts(r['timestamp']))

    all_tids = sorted({r['id'].strip() for r in obs_rows})
    per_tid = {}
    for t in all_tids:
        o = [r for r in obs_rows if r['id'].strip() == t]
        u = [r for r in run_rows if r['id'].strip() == t]
        th, L, spec = K.samples(o, u, gt_edges, geo, key_of)
        if len(th):
            per_tid[t] = (th, L, spec)
        print(f'  samples {t}: n={len(th)}', flush=True)
    tids = sorted(per_tid)
    excluded = [t for t in all_tids if t not in per_tid]

    def cat(ts):
        return (np.concatenate([per_tid[t][0] for t in ts]),
                np.concatenate([per_tid[t][1] for t in ts]),
                np.concatenate([per_tid[t][2] for t in ts]))

    rows = []
    th, L, spec = cat(tids)
    a, b, rms, se_a, se_b = K.fit(th, L, spec)
    m = metrics(th, spec / (L * L), a, b)
    rows.append(dict(fold='ALL(insample)', n_train=len(th), n_test=0, c=a, b_rad2=b,
                     sig_b_deg=float(np.degrees(np.sqrt(max(b, 0)))), se_c=se_a, se_b=se_b,
                     z_b=float(b / se_b) if se_b else float('nan'), rms_train=rms,
                     **m))
    for held in tids:
        train = [t for t in tids if t != held]
        th_t, L_t, sp_t = cat(train)
        a, b, rms, se_a, se_b = K.fit(th_t, L_t, sp_t)
        th_o, L_o, sp_o = per_tid[held]
        mo = metrics(th_o, sp_o / (L_o * L_o), a, b)
        rows.append(dict(fold=held, n_train=len(th_t), n_test=len(th_o), c=a, b_rad2=b,
                         sig_b_deg=float(np.degrees(np.sqrt(max(b, 0)))), se_c=se_a,
                         se_b=se_b, z_b=float(b / se_b) if se_b else float('nan'),
                         rms_train=rms, **mo))
    DIAG.mkdir(parents=True, exist_ok=True)
    with open(DIAG / 'loto.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = config_constants()
    (DIAG / 'loto_config.json').write_text(json.dumps(
        dict(run=str(RUN), config=str(CFG), config_constants=cfg,
             trajectories_used=tids, trajectories_without_samples=excluded,
             refit_per_fold='(c,b) regression only; matcher run and correctness filter are '
                            'the delivered run for every fold'), indent=2))
    print(f'{"fold":<14}{"n_tr":>7}{"n_te":>6}{"c":>7}{"b rad2":>9}{"sig_b":>7}'
          f'{"z_b":>6}{"rms_tr":>8}{"th_rms_deg":>11}{"sig_pr_deg":>11}{"z2_mean":>9}{"z2_rms":>8}')
    for r in rows:
        print(f'{r["fold"]:<14}{r["n_train"]:>7}{r["n_test"]:>6}{r["c"]:>7.3f}'
              f'{r["b_rad2"]:>9.5f}{r["sig_b_deg"]:>7.2f}{r["z_b"]:>6.1f}{r["rms_train"]:>8.4f}'
              f'{r["theta_rms_deg"]:>11.3f}{r["sigma_pred_deg"]:>11.3f}'
              f'{r["z2_mean"]:>9.3f}{r["z2_rms"]:>8.2f}')
    print(f'config constants: kappa_scale={cfg["direction_kappa_scale"]}'
          f'  kappa_floor={cfg["direction_kappa_floor"]}')
    print(f'trajectories used: {tids}')
    print(f'trajectories without samples: {excluded}')
    print(f'wrote {DIAG / "loto.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
