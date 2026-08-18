#!/usr/bin/env python3
"""Compute per-sigma and pooled metrics for Fig.7 (simulation sigma sweep).

Reads, for each sigma level in data/simulation/<dir>/no_occlusion/no_fault/:
  - ground_truth.csv        : per-trajectory point_edge_ids JSON array (per-epoch gt edge)
  - fmm_result.csv          : FMM       -> matched edge = cpath, score = trustworthiness
  - cmm_result.csv          : CaMM-cov  -> matched edge = opath, score = trustworthiness
  - cmm_result_iso.csv      : CaMM-iso  -> matched edge = opath, score = trustworthiness

Metrics per config per sigma:
  - acc      : segment accuracy (matched edge == gt edge) over evaluable epochs
  - ece      : 10-bin reliability ECE of trustworthiness vs segment-level correctness
  - tw_mean  : mean trustworthiness
  - auc      : ROC AUC, label = 1 if mismatch (matched_edge != gt_edge), score = trustworthiness
  - auc_mm   : mismatch-detection AUC with the paper's orientation
               (higher score => more likely mismatch), i.e. 1 - auc (labels swapped);
               this matches the '-trustworthiness' convention used in
               evaluate_match_metrics.py / the paper table.

Alignment note: result rows are aligned to ground-truth epochs by
(trajectory id, rounded timestamp), NOT by the seq column. The seq column is
segment-relative in point-mode output and collides for trajectories split into
multiple segments (observed: sigma_10 traj 2, sigma_15 traj 9 in the iso runs).

Output: experiments/output/fig7_metrics.json
"""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
SIM = BASE / 'data/simulation'
OUT_DIR = BASE / 'experiments/output'
OUT = OUT_DIR / 'fig7_metrics.json'

SIGMA_DIRS = ['sigma_01', 'sigma_05', 'sigma_10', 'sigma_15',
              'sigma_20', 'sigma_25', 'sigma_30']
SIGMA_VAL = dict(zip(SIGMA_DIRS, [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]))

CONFIGS = ['FMM', 'CaMM-iso', 'CaMM-cov']
N_BINS = 10

# ── helpers ──────────────────────────────────────────────────────────────────

def sniff_delimiter(path):
    with open(path, newline='', encoding='utf-8') as f:
        first = f.readline()
    return ';' if first.count(';') > first.count(',') else ','


def load_gt_edges(gt_csv):
    """Return {traj_id: [gt_edge_id per point]} from ground_truth.csv."""
    gt = {}
    with open(gt_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            try:
                edges = json.loads(row['point_edge_ids'])
            except (json.JSONDecodeError, TypeError):
                continue
            gt[row['id']] = [str(e) for e in edges]
    return gt


def load_obs_ts_index(obs_csv):
    """Return {traj_id: {rounded_ts: point_index}} from the observations file."""
    delim = sniff_delimiter(obs_csv)
    idx = defaultdict(dict)
    n_per_id = defaultdict(int)
    with open(obs_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=delim):
            tid = row.get('id', '')
            try:
                t = round(float(row['timestamp']))
            except (ValueError, TypeError):
                continue
            idx[tid].setdefault(t, n_per_id[tid])
            n_per_id[tid] += 1
    return idx


def load_result(result_csv, obs_ts_idx, gt_edges):
    """Align result rows to gt epochs by (id, rounded timestamp).

    Returns dict {traj_id: {point_idx: {'edge': matched_edge, 'tw': score}}}
    and the number of rows that could not be aligned.
    """
    delim = sniff_delimiter(result_csv)
    out = defaultdict(dict)
    n_unmatched = 0
    with open(result_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=delim):
            tid = row.get('id', '').strip()
            try:
                t = round(float(row['timestamp']))
            except (ValueError, TypeError):
                n_unmatched += 1
                continue
            pt_idx = obs_ts_idx.get(tid, {}).get(t)
            if pt_idx is None or pt_idx >= len(gt_edges.get(tid, [])):
                n_unmatched += 1
                continue
            matched = row.get('cpath', '').strip() if 'cpath' in row and row.get('opath', '').strip() == '' else None
            # FMM files have no 'opath' column; CaMM files use 'opath'.
            if 'opath' in row:
                matched = row.get('opath', '').strip()
            else:
                matched = row.get('cpath', '').strip()
            try:
                tw = float(row.get('trustworthiness', 'nan'))
            except ValueError:
                tw = float('nan')
            if matched == '':
                matched = None
            out[tid][pt_idx] = {'edge': matched, 'tw': tw}
    return out, n_unmatched


def compute_ece(confidences, correct, n_bins=N_BINS):
    """10-bin reliability ECE; correct = 1 if segment-level match is right."""
    n = len(confidences)
    if n == 0:
        return float('nan')
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            idxs = [j for j in range(n) if lo <= confidences[j] <= hi]
        else:
            idxs = [j for j in range(n) if lo <= confidences[j] < hi]
        nb = len(idxs)
        if nb == 0:
            continue
        mc = sum(confidences[j] for j in idxs) / nb
        ac = sum(correct[j] for j in idxs) / nb
        ece += nb / n * abs(mc - ac)
    return ece


def compute_auc_manual(scores, pos):
    """Mann-Whitney U AUC. pos = 1 for positive class. Higher score => more likely positive."""
    n_pos = sum(pos)
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, pos), key=lambda x: x[0])
    rank = 1
    i = 0
    sum_rank_pos = 0.0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                sum_rank_pos += avg_rank
        i = j
    auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


try:
    from sklearn.metrics import roc_auc_score as _sk_auc

    def _auc(scores, pos):
        if sum(pos) == 0 or sum(pos) == len(pos):
            return None
        return _sk_auc(pos, scores)
except ImportError:
    def _auc(scores, pos):
        return compute_auc_manual(scores, pos)


def summarize_per_epoch(epochs):
    """epochs: list of {'edge': matched_edge_or_None, 'tw': float}"""
    valid = [e for e in epochs if e['edge'] is not None and not math.isnan(e['tw'])]
    if not valid:
        return None
    tws = [e['tw'] for e in valid]
    correct = [1 if e['edge'] == e['gt'] else 0 for e in valid]
    mism = [1 - c for c in correct]
    n = len(valid)
    n_mism = sum(mism)
    acc = (n - n_mism) / n
    ece = compute_ece(tws, correct)
    tw_mean = sum(tws) / n
    auc = _auc(tws, mism)                      # task convention: label 1 = mismatch
    auc_mm = None if auc is None else 1.0 - auc  # paper convention: higher score => more mismatch
    return {'acc': acc, 'ece': ece, 'tw_mean': tw_mean,
            'auc': auc, 'auc_mm': auc_mm, 'n_epochs': n, 'n_mismatch': n_mism}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    result = {'per_sigma': {}, 'pooled': {}}
    pooled = {c: {'tws': [], 'correct': [], 'mism': []} for c in CONFIGS}
    total_unmatched = 0

    for sd in SIGMA_DIRS:
        sub = SIM / sd / 'no_occlusion' / 'no_fault'
        gt_edges = load_gt_edges(sub / 'ground_truth.csv')
        obs_ts = load_obs_ts_index(sub / 'observations.csv')          # cov pipeline obs
        obs_ts_iso = load_obs_ts_index(sub / 'observations_iso_cmm.csv')  # iso pipeline obs

        files = {
            'FMM':      (sub / 'fmm_result.csv', obs_ts),
            'CaMM-iso': (sub / 'cmm_result_iso.csv', obs_ts_iso),
            'CaMM-cov': (sub / 'cmm_result.csv', obs_ts),
        }

        entry = {'sigma': SIGMA_VAL[sd]}
        for cfg in CONFIGS:
            fpath, oidx = files[cfg]
            aligned, n_un = load_result(fpath, oidx, gt_edges)
            total_unmatched += n_un
            epochs = []
            for tid, gt_list in gt_edges.items():
                for pt_idx, gt_e in enumerate(gt_list):
                    r = aligned.get(tid, {}).get(pt_idx)
                    if r is None:
                        continue
                    epochs.append({'edge': r['edge'], 'tw': r['tw'], 'gt': gt_e})
            summ = summarize_per_epoch(epochs)
            entry[cfg] = {
                'acc': round(summ['acc'], 6) if summ else None,
                'ece': round(summ['ece'], 6) if summ else None,
                'auc': round(summ['auc'], 6) if summ and summ['auc'] is not None else None,
                'tw_mean': round(summ['tw_mean'], 6) if summ else None,
                'auc_mm': round(summ['auc_mm'], 6) if summ and summ['auc_mm'] is not None else None,
                'n_epochs': summ['n_epochs'] if summ else 0,
                'n_mismatch': summ['n_mismatch'] if summ else 0,
            }
            if summ:
                pooled[cfg]['tws'].extend(tws := [e['tw'] for e in epochs])
                pooled[cfg]['correct'].extend(c for c, e in zip([1 if e['edge'] == e['gt'] else 0 for e in epochs], epochs))
                pooled[cfg]['mism'].extend(1 - c for c, e in zip([1 if e['edge'] == e['gt'] else 0 for e in epochs], epochs))
        result['per_sigma'][sd] = entry

    # pooled
    for cfg in CONFIGS:
        tws = pooled[cfg]['tws']
        mism = pooled[cfg]['mism']
        auc = _auc(tws, mism)
        auc_mm = None if auc is None else 1.0 - auc
        result['pooled'][cfg] = {
            'auc': round(auc, 6) if auc is not None else None,
            'auc_mm': round(auc_mm, 6) if auc_mm is not None else None,
            'n_epochs': len(tws),
            'n_mismatch': sum(mism),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")

    # ── summary table ──
    hdr = f"{'sigma':>8s} {'cfg':>9s} {'acc':>7s} {'ece':>7s} {'tw_mean':>8s} {'auc':>7s} {'auc_mm':>7s} {'n':>6s} {'mism':>5s}"
    print('\n' + '=' * 78)
    print(hdr)
    print('-' * 78)
    for sd in SIGMA_DIRS:
        for cfg in CONFIGS:
            e = result['per_sigma'][sd][cfg]
            print(f"{SIGMA_VAL[sd]:8.1f} {cfg:>9s} "
                  f"{e['acc'] if e['acc'] is not None else float('nan'):7.4f} "
                  f"{e['ece'] if e['ece'] is not None else float('nan'):7.4f} "
                  f"{e['tw_mean'] if e['tw_mean'] is not None else float('nan'):8.4f} "
                  f"{e['auc'] if e['auc'] is not None else float('nan'):7.4f} "
                  f"{e['auc_mm'] if e['auc_mm'] is not None else float('nan'):7.4f} "
                  f"{e['n_epochs']:6d} {e['n_mismatch']:5d}")
    print('-' * 78)
    for cfg in CONFIGS:
        e = result['pooled'][cfg]
        print(f"{'POOLED':>8s} {cfg:>9s} "
              f"{'':>7s} {'':>7s} {'':>8s} "
              f"{e['auc'] if e['auc'] is not None else float('nan'):7.4f} "
              f"{e['auc_mm'] if e['auc_mm'] is not None else float('nan'):7.4f} "
              f"{e['n_epochs']:6d} {e['n_mismatch']:5d}")
    print('=' * 78)
    print(f"\nNote: 'auc' uses the task convention (label 1 = mismatch, score = trustworthiness, "
          f"higher = more confident).")
    print(f"'auc_mm' = 1 - auc is the mismatch-detection AUC in the paper's orientation "
          f"(higher score => more likely mismatch, cf. '-trustworthiness' in evaluate_match_metrics.py).")
    if total_unmatched:
        print(f"\nWARNING: {total_unmatched} result rows could not be aligned to ground truth and were skipped.")
    else:
        print(f"\nAll result rows aligned to ground-truth epochs by (id, rounded timestamp).")


if __name__ == '__main__':
    main()
