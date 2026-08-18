#!/usr/bin/env python3
"""Plot Fig.7c (ECE vs sigma) and Fig.7d (pooled ROC) from fig7_metrics.json.

Fig.7c reads ECE values directly from experiments/output/fig7_metrics.json.
Fig.7d recomputes the pooled ROC curves from the raw simulation result CSVs
(reusing the (id, rounded timestamp) alignment in fig7_metrics.py) because the
JSON only stores scalar AUC values, not FPR/TPR arrays. The paper's ROC
orientation is used: score = -trustworthiness, positive = mismatch
(cf. evaluate_match_metrics.py '-trustworthiness'), i.e. AUC = auc_mm.

Output: experiments/output/fig7_new_ece_roc.png (200 DPI)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, str(Path(__file__).parent))
import fig7_metrics as m7  # reuse alignment helpers + dir/fig constants

BASE = Path('/home/ncz/fmm_sjtugnc')
SIM = BASE / 'data/simulation'
OUT = BASE / 'experiments/output/fig7_new_ece_roc.png'

# ── Fig.7c: ECE vs sigma ─────────────────────────────────────────────────────
with open(BASE / 'experiments/output/fig7_metrics.json', encoding='utf-8') as f:
    metrics = json.load(f)

sigmas = [m7.SIGMA_VAL[sd] for sd in m7.SIGMA_DIRS]
ece_iso = [metrics['per_sigma'][sd]['CaMM-iso']['ece'] for sd in m7.SIGMA_DIRS]
ece_cov = [metrics['per_sigma'][sd]['CaMM-cov']['ece'] for sd in m7.SIGMA_DIRS]

# ── Fig.7d: pooled ROC from raw CSVs ─────────────────────────────────────────
pooled = {c: {'scores': [], 'mism': []} for c in m7.CONFIGS}
for sd in m7.SIGMA_DIRS:
    sub = SIM / sd / 'no_occlusion' / 'no_fault'
    gt_edges = m7.load_gt_edges(sub / 'ground_truth.csv')
    obs_ts = m7.load_obs_ts_index(sub / 'observations.csv')
    obs_ts_iso = m7.load_obs_ts_index(sub / 'observations_iso_cmm.csv')
    files = {
        'FMM':      (sub / 'fmm_result.csv', obs_ts),
        'CaMM-iso': (sub / 'cmm_result_iso.csv', obs_ts_iso),
        'CaMM-cov': (sub / 'cmm_result.csv', obs_ts),
    }
    for cfg in m7.CONFIGS:
        fpath, oidx = files[cfg]
        aligned, _ = m7.load_result(fpath, oidx, gt_edges)
        for tid, gt_list in gt_edges.items():
            for pt_idx, gt_e in enumerate(gt_list):
                r = aligned.get(tid, {}).get(pt_idx)
                if r is None or r['edge'] is None or r['tw'] is None:
                    continue
                import math
                if math.isnan(r['tw']):
                    continue
                pooled[cfg]['scores'].append(-r['tw'])          # paper orientation
                pooled[cfg]['mism'].append(1 if r['edge'] != gt_e else 0)

roc = {}
for cfg in m7.CONFIGS:
    fpr, tpr, _ = roc_curve(pooled[cfg]['mism'], pooled[cfg]['scores'])
    roc[cfg] = {'fpr': fpr, 'tpr': tpr, 'auc': sk_auc(fpr, tpr)}
    # sanity check vs JSON pooled auc_mm
    j = metrics['pooled'][cfg]['auc_mm']
    print(f"pooled {cfg:>9s}: recomputed AUC={roc[cfg]['auc']:.6f}  json auc_mm={j:.6f}")

# ── figure ───────────────────────────────────────────────────────────────────
C_COV, C_ISO, C_FMM = '#d62728', '#1f77b4', '#7f7f7f'

fig, (axc, axr) = plt.subplots(1, 2, figsize=(14, 6))

# Fig.7c
axc.plot(sigmas, ece_iso, color=C_ISO, ls='--', marker='o', lw=2,
         label='CaMM-iso')
axc.plot(sigmas, ece_cov, color=C_COV, ls='-', marker='s', lw=2,
         label='CaMM-cov')
axc.set_xlabel('Pseudorange noise $\\sigma$ (m)', fontsize=12)
axc.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
axc.set_title('ECE Calibration: Isotropic vs Covariance Emission', fontsize=14)
axc.grid(True, alpha=0.3)
axc.legend(fontsize=12)
axc.set_xticks(sigmas)

# Fig.7d
axr.plot([0, 1], [0, 1], color='k', ls=':', lw=1, label='Chance (AUC = 0.5)')
axr.plot(roc['FMM']['fpr'], roc['FMM']['tpr'], color=C_FMM, ls='--', lw=2,
         label=f"FMM Viterbi (AUC = {roc['FMM']['auc']:.3f})")
axr.plot(roc['CaMM-iso']['fpr'], roc['CaMM-iso']['tpr'], color=C_ISO,
         ls='-.', lw=2,
         label=f"CaMM-iso TW (AUC = {roc['CaMM-iso']['auc']:.3f})")
axr.plot(roc['CaMM-cov']['fpr'], roc['CaMM-cov']['tpr'], color=C_COV,
         ls='-', lw=2,
         label=f"CaMM-cov TW (AUC = {roc['CaMM-cov']['auc']:.3f})")
axr.set_xlabel('False Positive Rate', fontsize=12)
axr.set_ylabel('True Positive Rate', fontsize=12)
axr.set_title('Pooled ROC: Mismatch Detection (7 $\\sigma$ levels $\\times$ 10K epochs)',
              fontsize=14)
axr.grid(True, alpha=0.3)
axr.legend(fontsize=12, loc='lower right')

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
