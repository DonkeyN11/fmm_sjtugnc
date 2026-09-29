#!/usr/bin/env python3
"""
Verification script: compare CMM/FMM output against paper claims.
Matches by TIMESTAMP, uses reverse edge map, excludes gt_edge ∈ {0, -1, ''}.

Usage:
    python experiments/scripts/verify_paper_numbers.py

Outputs a comprehensive comparison of all metrics against paper claims.
"""

import argparse, csv, json, math, numpy as np
from pathlib import Path
from collections import defaultdict
import re

BASE = Path(__file__).resolve().parents[2]
REV_MAP = json.load(open(BASE / 'experiments/config/reverse_edge_map.json'))

# Trajectories excluded from every reported real-data figure.
#
# Trajectory 12 is not a trajectory traversal. Its 734 epochs span a 73.5 m x
# 60.9 m box, against 8742 m x 5612 m for traj 11; it sits on one single road
# edge for its whole length, so "segment accuracy" there is one edge counted 133
# times; and its filtering posterior is exactly 1.0 at every one of the 734
# epochs -- one distinct value, because there is never a competing candidate to
# be uncertain between.
#
# Including it does not add a measurement, it damages the two metrics that are
# supposed to discriminate: accuracy is handed 100 % on 133 free epochs, and AUC
# falls toward 0.5 because a rank statistic has no score variation to rank. The
# manuscript reports the 6-trajectory set; this constant is where that policy is
# stated.
#
# It belongs here, in the evaluation layer, and NOT as an absence of rows in
# aligned.csv. aligned.csv is the complete factual record (7 trajectories /
# 16155 epochs). An exclusion baked into the data file is invisible from the
# evaluator and silently changes the denominator of every metric derived from
# it, which is exactly how the policy went unrecorded until now.
EXCLUDE_TRAJ = {"12"}

# Paper claims — the revised manuscript's Tables V and VI (revision of 2026-09-28),
# over the 13,069 evaluable epochs (6 trajectories; traj 12 excluded).
#
# These were the 2026-09-02 numbers until 2026-09-22, when §V was regenerated at one
# configuration. The old set was not merely stale, it pointed the WRONG WAY on the
# paper's own headline result: it asserted CaMM AUC 0.721 > FMM 0.583, whereas the
# submitted tables report 0.649 < 0.859. A claim-checker that reverses the direction
# of the paper's central negative finding is worse than none, so update this dict
# whenever §V moves, and never let it drift a revision behind the .tex again.
#
# Values are the manuscript's ROUNDED figures, so check()'s tolerances do the work.
# The manuscript's reported values, as the regression target. Re-based 2026-09-28 on
# the current operating point and the current runs; the CMM column is the FILTERING
# arm, which is the one this script's estimator most closely matches (it reads the
# `trustworthiness` column). The smoothed arm is not scored here -- add a column if
# the paper's headline moves to it.
#
# 'cmm' entries below were previously 96.9 / 0.044 / 0.649 / 5.5 / 13.5, which is
# ablation-ladder row 7 (HPL-adaptive disk, phmi = 1e-5). The paper reports row 5.
#
# Re-anchored 2026-09-30 on the scalar run (see the note at CMM_CSV). Four values moved
# and all four were the anisotropic run's: traj 11 96.69 -> 96.65, traj 21 98.74 -> 98.69,
# traj 22 97.74 -> 97.79, All 98.13 -> 98.11, plus the alpha AUC 0.836 -> 0.838 and the
# separation block below. Each is now the value docs/calib/calib.tex prints -- the
# per-trajectory ones in tab:per_traj and the AUC in tab:intervals -- so the dict can be
# read as the manuscript's figures rather than as one run's output. The tw_sep block is
# the exception: the manuscript prints no such row, so those entries are the delivered
# run's own numbers (CMM column = the FILTERING arm, FMM column = the delta arm) and are
# anchored only for internal consistency.
PAPER = {
    'accuracy': {
        '11': (2385, 96.65, 84.57), '13': (1908, 97.90, 96.80),
        '14': (352, 100, 67.33), '21': (3581, 98.69, 99.80), '22': (2123, 97.79, 99.01),
        '23': (2720, 98.79, 99.04), 'ALL': (13069, 98.11, 95.42),
    },
    'tw_sep': {'cmm_corr': 0.958, 'cmm_wrong': 0.528, 'cmm_sep': 0.429,
               'fmm_corr': 0.781, 'fmm_wrong': 0.350, 'fmm_sep': 0.442, 'ratio': 0.97},
    'ece': {'cmm': 0.038, 'fmm': 0.193},
    'auc': {'cmm': 0.838, 'fmm': 0.865},
    'pos_err': {'cmm_mean': 5.66, 'cmm_p95': 13.7, 'fmm_mean': 6.0, 'fmm_p95': 15.0},
    # The submitted manuscript reports no false-match rate. The script still measures
    # one; None means "print it, do not score it against a claim that does not exist".
    'fmr': {'cmm': None, 'fmm': None},
}

def is_edge_match(m, g):
    m=m.strip(); g=g.strip()
    # An empty match is never a match. The two REV_MAP tests below fall back to
    # the empty string for any edge the map does not list, so with m == '' the
    # second of them reads REV_MAP.get(g,'') == '' and returns True for every
    # ground-truth edge outside the map -- i.e. every failed or absent epoch was
    # credited as a hit. That is the epoch a comparison most needs counted
    # against the matcher, and it silently inflated accuracy by exactly the
    # failure rate: on the r=0.003 fair FMM run, 5247 of 13123 epochs carry an
    # empty cpath but the score read 91.1 % instead of the true 36.5 %.
    # compare_real_metrics_enu.py has always guarded this at its call site
    # (`if cpath else False`); the guard belongs here, in the one definition.
    if not m or not g: return False
    if m==g: return True
    if REV_MAP.get(m,'')==g: return True
    if REV_MAP.get(g,'')==m: return True
    return False

def parse_point(wkt):
    m=re.match(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)',str(wkt))
    return (float(m.group(1)),float(m.group(2))) if m else (None,None)

def haversine_m(lon1,lat1,lon2,lat2):
    mlat=math.radians((lat1+lat2)/2)
    return math.hypot((lon1-lon2)*111320*math.cos(mlat),(lat1-lat2)*111320)

def ece(confs,corrects,n=10):
    e=0.0;N=len(confs)
    for i in range(n):
        lo,hi=i/n,(i+1)/n
        mask=(confs>=lo)&(confs<hi) if i<n-1 else (confs>=lo)&(confs<=hi)
        nb=mask.sum()
        if nb==0:continue
        e+=nb/N*abs(confs[mask].mean()-corrects[mask].mean())
    return e

def log_score(labels, scores, eps=1e-12):
    """Mean LOG LIKELIHOOD of the binary outcome, in NATS. Higher is better.

    The sign convention matters and is the one the manuscript uses: this returns
    log p(y | score) averaged over epochs, so every arm's value is negative, a perfect
    forecaster scores 0, a forecaster who always says one half scores log(0.5) = -0.693,
    and -0.410 (the baseline's native score on Haikou) is worse than -0.173 (the
    smoothed posterior). Reporting the negated quantity is equally common in the
    literature and would print every column with the opposite sign, which is a silent
    way for a table and its prose to disagree.

    The one metric in the audit that is unbounded, which is why it is reported beside the
    Brier score rather than instead of it: a single confident error costs more than any
    number of humble ones, so it is the column that exposes the tail the binned
    calibration error cannot see.

    Scores at exactly 1 or 0 are clipped by `eps`. That is a real truncation, not a
    formality: a saturated arm reports 0.9+ on most epochs, and without the clip the log
    score of an arm with a single wrong-and-certain epoch would be infinite, which would
    make the column incomparable across arms rather than informative. The same clip is
    applied to every arm, so the comparison is fair, and the saturation rate is reported
    separately so the reader can see how much clipping a column carries.
    """
    labels = np.asarray(labels, dtype=float)
    s = np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)
    return float((labels * np.log(s) + (1.0 - labels) * np.log(1.0 - s)).mean())


def auc(labels,scores):
    """AUC as the tie-corrected Mann-Whitney rank statistic.

    Do NOT compute this by integrating the ROC step curve with trapz: when
    scores tie, trapz interpolates linearly across a step that is actually
    vertical, which biases the result in an unpredictable direction. FMM's
    trustworthiness is near-degenerate (mean 0.015 with many exact ties), so
    the bias there is severe -- trapz gives 0.613 for FMM and 0.696 for CMM,
    reversing the CMM > FMM ordering the paper reports. The rank statistic
    gives 0.568 / 0.719, matching the manuscript's 0.583 / 0.721.
    """
    labels=np.asarray(labels, dtype=float)
    scores=np.asarray(scores, dtype=float)
    order=np.argsort(scores, kind='mergesort'); s=scores[order]
    n=len(s)
    ranks=np.empty(n)
    i=0
    while i<n:                       # average ranks within each tie group
        j=i
        while j+1<n and s[j+1]==s[i]: j+=1
        ranks[i:j+1]=(i+j)/2.0+1.0
        i=j+1
    n_pos=int(labels.sum()); n_neg=n-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    is_pos=(labels[order]==1)
    return float((ranks[is_pos].sum()-n_pos*(n_pos+1)/2.0)/(n_pos*n_neg))

def roc_curve_and_auc(labels,scores):
    """ROC step curve for plotting, plus the tie-corrected rank AUC of auc().

    Returns (fpr, tpr, auc_value).

    The curve and the scalar are computed by different means on purpose, and the
    split is the point of this function. Plotting needs the empirical step curve;
    the *reported number* must not be its trapezoidal area, because trapz
    interpolates linearly across vertical steps and the simulation corpora carry
    large blocks of exact ties (an epoch with no ground-truth edge scores 0.5; a
    saturated posterior ties at 1.0). So return the curve for the figure and take
    the scalar from auc(), which is the convention commit f728584 adopted and
    which eval_sim_corpus.py, eval_result_csv.py, inventory_real_metrics.py and
    fig_roc_real.py already use.

    Callers that previously integrated their own step curve with np.trapz got a
    different number from this on the same labels -- on the sigma-sweep corpus at
    sigma=1 m, 0.4813 (trapz) against 0.6955 (rank), which is the difference
    between reporting chance and reporting real discrimination. Migrate by
    delegating here rather than by copying the integration.
    """
    labels=np.asarray(labels, dtype=float)
    scores=np.asarray(scores, dtype=float)
    order=np.argsort(scores, kind='mergesort')[::-1]
    labels_sorted=labels[order]
    n_pos=int(np.sum(labels_sorted)); n_neg=len(labels_sorted)-n_pos
    if n_pos==0 or n_neg==0:
        return np.array([0.0,1.0]), np.array([0.0,1.0]), 0.5
    tpr=np.cumsum(labels_sorted)/n_pos
    fpr=np.cumsum(1-labels_sorted)/n_neg
    tpr=np.concatenate([[0.0], tpr]); fpr=np.concatenate([[0.0], fpr])
    return fpr, tpr, auc(labels, scores)

def norm_ts(s):
    """Key a timestamp column numerically.

    The tables disagree on formatting ('1750306260' vs '1750306260.0'), so the
    join must be numeric. Do NOT normalise by stripping trailing zeros: that
    maps 1750306260 and 175030626 to the same key, which would silently merge
    two distinct epochs. float() is exact for every value in these tables.
    """
    return float(s)

PROC = BASE / 'data/real_vehicle/hainan_06/processed'

# Which CMM run to verify. The default is the current binary's output on the
# live real-data config, input/config/cmm_real_0729.xml. A verifier that reads
# a file the current code cannot produce is verifying nothing, so anything else
# has to be asked for explicitly with --cmm.
#
# Measured 2026-09-16, all four evaluated over the same 13123 evaluable epochs
# (6 trajectories; traj 12 excluded, see EXCLUDE_TRAJ):
#
#   run                       covariance    PL      acc%      ECE     AUC  failed
#   cmm_result_0729.csv       calibrated    legacy  97.2262  0.0391  0.7592     0
#   cmm_result_fixed_en.csv   --            ENU     96.7690  0.0412  0.7191     0
#   (raw GST + ENU PL)        raw GST       ENU     95.4507  0.0496  0.6232   112
#   (calibrated GST + ENU PL) calibrated    ENU     94.6582  0.0567  0.6657     0
#
# The SUBMITTED manuscript's runs, and therefore the defaults. They were produced
# on 2026-09-21/22 and live under data/archive_20260922/ (gitignored), not in
# processed/: `git grep 95.0316` returns zero tracked hits, so nothing under version
# control reproduces Table V. See data/archive_20260922/PROVENANCE.md for the mapping
# from each table to its file and config.
#
#   real_final/cmm_final.csv   CaMM  96.8986 / 0.0437 / 0.6493   (Tables V, VI, VIII row 7)
#   real_final/fmm_final.csv   FMM   95.0316 / 0.1903 / 0.8589   (Tables V, VI, VIII row 1)
#       configs: cmm_final <- newdelta/real_d13p5.xml ; fmm_final <- real_final/fmm2.xml
#
# Neither matched a `processed/` file, which is why this script previously verified
# a superseded revision. The residents are older configurations and give different
# answers: cmm_result_0729.csv is 97.2262 / 0.0391 / 0.7592 (calibrated GST + legacy
# PL) and fmm_result.csv is 88.9888 / 0.8766 / 0.5682 (r = 0.03 deg, a 3.3 km radius,
# where 300 m was intended). Point --cmm/--fmm at those to reproduce the 2026-09-02
# revision, not this one.
#
# The 2026-09-15 ECEF->ENU protection-level correction is a separate open question:
# the corrected HPL is tighter (median 10.9 m vs 22.8 m) and every metric is worse,
# because a search radius of exactly HPL admits fewer candidates.
# Reproduce both with `experiments/scripts/real_final_tables.py`, which reruns the
# matchers at the paper's operating point and writes these files.
#
# WHICH ARCHIVE THIS DEFAULTS TO IS PART OF THE CHECK, not an incidental path.
# These pointed at `archive_20260928/final` until 2026-09-30, and that archive is the
# ANISOTROPIC run (use_mahalanobis=true, isotropic_emission_scale=0 -- see its cmm.log).
# The manuscript reports the SCALAR operating point, `input/config/paper1_cmm_scalar.xml`,
# whose runs are in `archive_20260929/scalar`. Every §V figure in docs/calib/calib.tex
# matches the scalar run to its printed precision: per-trajectory 96.65 / 97.90 / 100.00 /
# 98.69 / 97.79 / 98.79 and All 98.11, against the anisotropic run's 96.69 / 97.90 /
# 100.00 / 98.74 / 97.74 / 98.79 and 98.13. The old anchor still passed -- the two runs
# differ by less than check()'s tolerance -- so the mismatch was invisible in the output
# and would have stayed invisible until one archive was regenerated. That is the failure
# this dict's own comment warns about, one level down: not a stale claim, a wrong run
# under a correct claim.
CMM_CSV = BASE / 'data/archive_20260929/scalar/cmm.csv'
FMM_CSV = BASE / 'data/archive_20260929/scalar/fmm.csv'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cmm', type=Path, default=CMM_CSV)
    ap.add_argument('--fmm', type=Path, default=FMM_CSV)
    args = ap.parse_args()
    # Load GT by (id, timestamp). An epoch is evaluable when it has a road edge
    # -- edge correctness compares cpath against gt_edge and needs no
    # coordinates. gt_x/gt_y are only required for the position-error metric,
    # so a row with unparseable coordinates stays in the accuracy denominator
    # and contributes (None, None) to the error stats instead of being dropped.
    # Dropping those rows instead loses 204 of the 13123 evaluable epochs and
    # silently changes the reported accuracy, which is what exp6_real_accuracy
    # and the paper both count as 13123.
    gt_by_ts = {}
    # (id, seq) -> timestamp, needed to resolve FMM's seq column; see load_mr.
    #
    # BUILT FROM THE INPUT TABLE, NOT FROM aligned.csv, since 2026-09-28. FMM's `seq`
    # is a 0-based index into the trajectory it was given, so the file that defines
    # it is the one it read -- data/real_vehicle/hainan_06/cmm_input_points_k1.csv.
    # This map used aligned.csv's `uni_seq` until then, and that happened to agree
    # because aligned.csv's union of four sources enumerated exactly the same epochs
    # in the same order. Once it did not (11 FMM-only keys entered the union, whose
    # timestamps no observation carries), uni_seq and seq diverged and every later
    # epoch of a trajectory was joined to the wrong label: FMM scored 93.23 % with a
    # 27.6 m mean position error instead of 95.42 % with 6.0 m, and printed a
    # perfectly plausible table while doing it. The input table's own (id, timestamp)
    # pairs, sorted per trajectory, are what `seq` counts.
    #
    # This is deliberately independent of EXCLUDE_TRAJ: it is a mechanical lookup, and
    # the matcher result files still contain the excluded trajectory's rows
    # (fmm_final.csv carries all 734 of traj 12's). Excluding here instead would leave
    # those rows unresolvable and silently demote the whole FMM file to a timestamp
    # join -- which is the failure this map exists to prevent.
    ts_by_seq = {}
    _per_traj = defaultdict(list)
    with open(BASE/'data/real_vehicle/hainan_06/cmm_input_points_k1.csv',newline='') as f:
        for r in csv.DictReader(f,delimiter=';'):
            _per_traj[r['id'].strip()].append(norm_ts(r['timestamp']))
    for _tid, _stamps in _per_traj.items():
        for _i, _ts in enumerate(sorted(_stamps)):
            ts_by_seq[(_tid, str(_i))] = _ts

    n_excluded = 0
    with open(BASE/'data/real_vehicle/hainan_06/processed/aligned.csv',newline='') as f:
        for r in csv.DictReader(f,delimiter=';'):
            tid = r['id'].strip()
            if tid in EXCLUDE_TRAJ:
                n_excluded += 1
                continue
            gt_e = r['gt_edge'].strip()
            if gt_e in ('0','-1',''): continue
            try: gx=float(r['gt_x']); gy=float(r['gt_y'])
            except: gx=gy=None
            gt_by_ts[(tid, norm_ts(r['timestamp']))] = (gt_e, gx, gy)

    print(f"Population: {len(gt_by_ts)} evaluable epochs from "
          f"{sorted({k[0] for k in gt_by_ts})}; "
          f"EXCLUDE_TRAJ={sorted(EXCLUDE_TRAJ)} dropped {n_excluded} raw rows "
          f"(traj {', '.join(sorted(EXCLUDE_TRAJ))})")

    # Load a matcher result into the (id, timestamp) key space.
    #
    # The join key is NOT the same for both matchers, and getting this wrong fails
    # silently rather than loudly. FMM writes a `timestamp` column that is NOT a
    # per-epoch timestamp: in real_final/fmm_final.csv trajectory 11 has 2,696 rows
    # but only 25 distinct timestamp values, so a (id, timestamp) join collapses
    # ~108 epochs onto one key and matches roughly 150 of the 13,123 evaluable
    # epochs while still printing a plausible-looking table. FMM's `seq` is instead
    # a unique 0-based row index into aligned.csv's `uni_seq` (verified: ranges and
    # counts correspond exactly, 0..2695 vs 0..2695 for traj 11), so resolve
    # seq -> timestamp through ts_by_seq.
    #
    # CaMM is the opposite: its timestamp is unique per epoch but its `seq` is not,
    # because the matcher splits trajectories and seq restarts at 0 while the
    # timestamps keep running. So: use seq when it is unique per id AND resolves
    # through aligned.csv, and the literal timestamp otherwise.
    #
    # Failed epochs (FAILED_NO_CANDIDATE, FAILED_DISCONNECTED) carry the sentinel
    # trustworthiness -999.0 to mark "no value". Averaging that into the TW means
    # or the AUC ranks produces nonsense -- pointing an early version of this
    # script at the ENU run yielded TW_wrong = -368. Drop the sentinel instead:
    # such an epoch is still an accuracy miss (its cpath is empty, so
    # is_edge_match returns False), it just contributes no trustworthiness.
    TW_SENTINEL = -900.0

    def load_mr(path):
        rows = list(csv.DictReader(open(path,newline=''), delimiter=';'))
        has_seq = bool(rows) and 'seq' in rows[0]
        use_seq = False
        if has_seq:
            per_id = defaultdict(set)
            for r in rows:
                per_id[r['id'].strip()].add(r['seq'].strip())
            # unique seq per id, and every one resolves through aligned.csv
            use_seq = all(len(v) == sum(1 for r in rows if r['id'].strip() == k)
                          for k, v in per_id.items())
            if use_seq:
                use_seq = all((r['id'].strip(), r['seq'].strip()) in ts_by_seq
                              for r in rows)
        if has_seq and not use_seq:
            print(f"  note: {path.name} has a non-unique `seq`; joining on timestamp")
        elif use_seq:
            print(f"  note: {path.name} joined on (id, seq) -> aligned.csv uni_seq")

        d = {}
        for r in rows:
            tid = r['id'].strip()
            if use_seq:
                key = (tid, ts_by_seq[(tid, r['seq'].strip())])
            else:
                key = (tid, norm_ts(r['timestamp']))
            try: tw = float(r['trustworthiness'])
            except ValueError: tw = TW_SENTINEL
            if tw <= TW_SENTINEL: tw = None
            d[key] = (r['cpath'].strip(), tw, r.get('ogeom',''), r.get('pgeom',''))
        return d

    cmm = load_mr(args.cmm)
    fmm = load_mr(args.fmm)
    # The FULL path, not the basename: which archive the claims are checked against is
    # the one thing a reader cannot infer from the metric names, and printing only
    # `cmm.csv` is how a wrong anchor stays invisible. See the note at CMM_CSV.
    print(f"CMM run: {args.cmm}   FMM run: {args.fmm}")

    # Evaluate
    cmm_pt = defaultdict(lambda:{'c':0,'t':0,'perr':[]})
    fmm_pt = defaultdict(lambda:{'c':0,'t':0,'perr':[]})
    cmm_tw_c=[]; cmm_tw_w=[]
    fmm_tw_c=[]; fmm_tw_w=[]

    for key,(gt_e,gx,gy) in gt_by_ts.items():
        tid = key[0]
        if key in cmm:
            cpath,tw,ogeom,pgeom = cmm[key]
            cmm_pt[tid]['t'] += 1
            if is_edge_match(cpath,gt_e):
                cmm_pt[tid]['c']+=1
                if tw is not None: cmm_tw_c.append(tw)
            else:
                if tw is not None: cmm_tw_w.append(tw)
            px,py = parse_point(pgeom) if pgeom else (None,None)
            if px is not None and gx is not None: cmm_pt[tid]['perr'].append(haversine_m(px,py,gx,gy))

        if key in fmm:
            fpath,ftw,_,fpgeom = fmm[key]
            fmm_pt[tid]['t'] += 1
            if is_edge_match(fpath,gt_e):
                fmm_pt[tid]['c']+=1
                if ftw is not None: fmm_tw_c.append(ftw)
            else:
                if ftw is not None: fmm_tw_w.append(ftw)
            px,py = parse_point(fpgeom) if fpgeom else (None,None)
            if px is not None and gx is not None: fmm_pt[tid]['perr'].append(haversine_m(px,py,gx,gy))

    cmm_tw_c=np.array(cmm_tw_c); cmm_tw_w=np.array(cmm_tw_w)
    fmm_tw_c=np.array(fmm_tw_c); fmm_tw_w=np.array(fmm_tw_w)

    # Print comparison
    print("="*80)
    print("  PAPER NUMBER VERIFICATION")
    print("="*80)
    print(f"\n{'Metric':<40} {'Paper':>12} {'Computed':>12} {'Δ':>10} {'Status':>10}")
    print("-"*88)

    tc={'cc':0,'ct':0,'fc':0,'ft':0}
    for tid in sorted(cmm_pt.keys()):
        cd=cmm_pt[tid]; fd=fmm_pt.get(tid,{'c':0,'t':0})
        tc['cc']+=cd['c']; tc['ct']+=cd['t']; tc['fc']+=fd['c']; tc['ft']+=fd['t']

    for tid in sorted(cmm_pt.keys()):
        cd=cmm_pt[tid]; pn,pc,pf = PAPER['accuracy'].get(tid,(0,0,0))
        ca=cd['c']/cd['t']*100 if cd['t']>0 else 0
        check(f"Traj{tid} CMM Acc (%)", pc, ca)
        fa=fmm_pt[tid]['c']/fmm_pt[tid]['t']*100 if fmm_pt[tid]['t']>0 else 0
        check(f"Traj{tid} FMM Acc (%)", pf, fa)

    ca_all=tc['cc']/tc['ct']*100; fa_all=tc['fc']/tc['ft']*100
    check("CMM Accuracy (%)", PAPER['accuracy']['ALL'][1], ca_all)
    check("FMM Accuracy (%)", PAPER['accuracy']['ALL'][2], fa_all)
    check("Eval epochs", PAPER['accuracy']['ALL'][0], tc['ct'])

    sep_c=cmm_tw_c.mean()-cmm_tw_w.mean(); sep_f=fmm_tw_c.mean()-fmm_tw_w.mean()
    check("CMM TW corr mean", PAPER['tw_sep']['cmm_corr'], cmm_tw_c.mean())
    check("CMM TW wrong mean", PAPER['tw_sep']['cmm_wrong'], cmm_tw_w.mean())
    check("CMM TW separation", PAPER['tw_sep']['cmm_sep'], sep_c)
    check("FMM TW separation", PAPER['tw_sep']['fmm_sep'], sep_f)
    check("TW Ratio", PAPER['tw_sep']['ratio'], sep_c/sep_f if sep_f>0 else 0)

    c_corr=np.concatenate([np.ones(len(cmm_tw_c)),np.zeros(len(cmm_tw_w))])
    c_all=np.concatenate([cmm_tw_c,cmm_tw_w])
    f_corr=np.concatenate([np.ones(len(fmm_tw_c)),np.zeros(len(fmm_tw_w))])
    f_all=np.concatenate([fmm_tw_c,fmm_tw_w])
    check("CMM ECE", PAPER['ece']['cmm'], ece(c_all,c_corr))
    check("FMM ECE", PAPER['ece']['fmm'], ece(f_all,f_corr))
    check("CMM AUC", PAPER['auc']['cmm'], auc(c_corr,c_all))
    check("FMM AUC", PAPER['auc']['fmm'], auc(f_corr,f_all))

    all_cperr=np.concatenate([np.array(d['perr']) for d in cmm_pt.values() if d['perr']])
    all_fperr=np.concatenate([np.array(d['perr']) for d in fmm_pt.values() if d['perr']])
    check("CMM pos err mean", PAPER['pos_err']['cmm_mean'], all_cperr.mean())
    check("CMM pos err P95", PAPER['pos_err']['cmm_p95'], np.percentile(all_cperr,95))
    check("FMM pos err mean", PAPER['pos_err']['fmm_mean'], all_fperr.mean())
    check("FMM pos err P95", PAPER['pos_err']['fmm_p95'], np.percentile(all_fperr,95))

    fmr_c=len(cmm_tw_w)/(len(cmm_tw_c)+len(cmm_tw_w))*100
    check("CMM FMR (%)", PAPER['fmr']['cmm'], fmr_c)

def check(name, paper_v, comp_v):
    """Compare a computed metric against the manuscript's claim.

    Tolerances are RELATIVE, not absolute. The manuscript rounds to three
    significant figures -- 96.8986 is printed as 96.9, 95.0316 as 95.0, 5.4739 as
    5.5 -- so a fixed 0.02 band flags correct percentage values as ⚠️ REVIEW and a
    checker that cries wolf is a checker nobody reads. Scale the bands to the
    magnitude of the claim, with a floor for the near-zero metrics (ECE, AUC
    margins) where a relative band would collapse to nothing.
    """
    # paper_v is None when the manuscript reports no such claim. Print the computed
    # value and mark it unscored rather than inventing a target to compare against.
    if paper_v is None:
        print(f"{name:<40} {'(not reported)':>12} {comp_v:>12.4f} {'--':>10} {'n/a':>10}")
        return
    delta = comp_v - paper_v
    tol_exact = max(0.005, 0.001 * abs(paper_v))
    tol_ok = max(0.02, 0.005 * abs(paper_v))
    if abs(delta) <= tol_exact: status = "✅ EXACT"
    elif abs(delta) <= tol_ok: status = "✅ OK"
    elif abs(delta) < 0.5: status = "⚠️ REVIEW"
    else: status = "❌ UPDATE"
    print(f"{name:<40} {paper_v:>12.4f} {comp_v:>12.4f} {delta:>+10.4f} {status:>10}")

if __name__ == '__main__':
    main()
