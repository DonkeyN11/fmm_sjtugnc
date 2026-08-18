#!/usr/bin/env python3
"""Regenerate CaMM-iso results with FMM-matching fixed sigma (0.001° ≈ 111m)."""
import csv, json, math, os, re, subprocess, sys, tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
CMM_BIN = BASE / 'build/cmm'
SIGMA_DIRS = ['sigma_01', 'sigma_05', 'sigma_10', 'sigma_15', 'sigma_20', 'sigma_25', 'sigma_30']
SIGMA_VALS = [1, 5, 10, 15, 20, 25, 30]
FMM_SIGMA_DEG = 0.001  # Fixed isotropic sigma matching FMM's gps_error

PP_RE = re.compile(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', re.I)
def pp(wkt):
    m = PP_RE.search(wkt or '')
    return (float(m.group(1)), float(m.group(2))) if m else None

def compute_ece(confs, labels, n_bins=10):
    N = len(confs); ece = 0.0
    for i in range(n_bins):
        lo, hi = i/n_bins, (i+1)/n_bins
        idxs = [j for j in range(N) if lo<=confs[j]<=hi] if i==n_bins-1 else [j for j in range(N) if lo<=confs[j]<hi]
        nb = len(idxs)
        if nb == 0: continue
        mc = sum(confs[j] for j in idxs) / nb
        ac = sum(labels[j] for j in idxs) / nb
        ece += nb/N * abs(mc - ac)
    return ece

def auc_manual(scores, labels):
    """Manual AUC computation."""
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    # Sum ranks of positives (Wilcoxon-Mann-Whitney)
    rank_sum = 0.0; tied_group = []
    for i, (score, label) in enumerate(pairs):
        if tied_group and score != pairs[tied_group[-1]][0]:
            tied_group = []
        tied_group.append(i)
        if label == 1:
            rank_sum += i + 1  # 1-indexed rank
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def main():
    print("=" * 70)
    print("  Regenerate CaMM-iso with FMM-matched sigma (0.001° ≈ 111m)")
    print("=" * 70)

    # ── Step 1: Generate iso observations + run CMM ──
    for sigma_dir, sigma_val in zip(SIGMA_DIRS, SIGMA_VALS):
        data_dir = BASE / 'data/simulation' / sigma_dir / 'no_occlusion' / 'no_fault'
        obs_file = data_dir / 'observations.csv'
        iso_out = data_dir / 'observations_iso_fixed.csv'

        if not obs_file.exists():
            print(f"  {sigma_dir}: observations.csv not found, skipping")
            continue

        # Create iso observations with FMM-matched sigma
        print(f"\n  {sigma_dir} (σ_pr={sigma_val}m): generating iso data...")
        n_rows = 0
        with open(obs_file, newline='', encoding='utf-8') as fin, \
             open(iso_out, 'w', newline='', encoding='utf-8') as fout:
            reader = csv.DictReader(fin, delimiter=';')
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for row in reader:
                row['sde'] = f'{FMM_SIGMA_DEG:.12f}'
                row['sdn'] = f'{FMM_SIGMA_DEG:.12f}'
                row['sdne'] = '0.0'
                writer.writerow(row)
                n_rows += 1
        print(f"    Wrote {n_rows} rows")

        # Run CMM
        cmm_out = data_dir / 'cmm_result_iso_fixed.csv'
        if cmm_out.exists():
            print(f"    CMM output exists, skipping")
            continue

        tree = ET.parse(BASE / 'input/config/cmm_config_omp.xml')
        root = tree.getroot()
        root.find('output').find('file').text = str(cmm_out)
        root.find('input').find('gps').find('file').text = str(iso_out)
        root.find('input').find('network').find('file').text = str(BASE / 'input/map/hainan/edges.shp')
        root.find('input').find('ubodt').find('file').text = str(BASE / 'input/map/hainan_ubodt_indexed.bin')
        params = root.find('parameters')
        overrides = {
            'k': '16', 'min_candidates': '1', 'protection_level_multiplier': '10',
            'use_mahalanobis': 'false', 'direction_penalty': 'false',
            'temperature_adapt': 'false', 'background_prob': '0.1',
            'phmi': '0.00001', 'lag_steps': '0', 'normalized': 'false',
            'filtered': 'false', 'cumulative_reverse_pct': '0.0',
        }
        for tag, val in overrides.items():
            el = params.find(tag)
            if el is None: el = ET.SubElement(params, tag)
            el.text = val

        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
        tree.write(tmp, encoding='utf-8', xml_declaration=True)
        tmp.close()

        print(f"    Running CMM...")
        r = subprocess.run([str(CMM_BIN), tmp.name], capture_output=True, text=True, cwd=str(BASE), timeout=600)
        os.unlink(tmp.name)
        if r.returncode != 0:
            print(f"    FAILED: {r.stderr[-200:]}")
            continue
        lines = sum(1 for _ in open(cmm_out))
        print(f"    OK: {lines} lines")

    # ── Step 2: Compute metrics ──
    print(f"\n{'='*70}")
    print("  Computing ECE + ROC metrics")
    print(f"{'='*70}")

    all_results = {'per_sigma': {}, 'pooled': {}}
    pooled_data = {'FMM': {'scores': [], 'labels': []},
                   'CaMM-iso': {'scores': [], 'labels': []},
                   'CaMM-cov': {'scores': [], 'labels': []}}

    for sigma_dir, sigma_val in zip(SIGMA_DIRS, SIGMA_VALS):
        data_dir = BASE / 'data/simulation' / sigma_dir / 'no_occlusion' / 'no_fault'
        gt_file = data_dir / 'ground_truth.csv'
        if not gt_file.exists():
            print(f"  {sigma_dir}: no ground_truth.csv, skipping")
            continue

        # Load ground truth
        gt_edges = {}
        with open(gt_file, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter=';'):
                traj_id = row['id']
                pt_edges_raw = row.get('point_edge_ids', '[]')
                try: pt_edges = json.loads(pt_edges_raw)
                except: continue
                for seq, edge_id in enumerate(pt_edges):
                    gt_edges[(traj_id, str(seq))] = str(edge_id)

        sigma_results = {}
        configs = {
            'FMM': (data_dir / 'fmm_result.csv', 'cpath'),
            'CaMM-iso': (data_dir / 'cmm_result_iso_fixed.csv', 'opath'),
            'CaMM-cov': (data_dir / 'cmm_result.csv', 'opath'),
        }

        for label, (fpath, edge_field) in configs.items():
            if not fpath.exists():
                print(f"  {sigma_dir}/{label}: file not found")
                continue
            trusts, labels = [], []
            with open(fpath, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f, delimiter=';'):
                    key = (row.get('id','').strip(), row.get('seq','').strip())
                    gt = gt_edges.get(key)
                    if gt is None: continue
                    edge = row.get(edge_field, '').strip()
                    status = row.get('status', '')
                    try: tw = float(row.get('trustworthiness','') or '0')
                    except: tw = 0
                    is_correct = (not ('FAILED' in status or not edge)) and (str(edge) == gt)
                    labels.append(1 if is_correct else 0)
                    trusts.append(tw)

            n = len(trusts)
            if n == 0: continue
            ece = compute_ece(trusts, labels)
            acc = sum(labels) / n
            tw_mean = sum(trusts) / n

            # ROC: mismatch detection (label=1 means mismatch)
            mismatch_labels = [1 - l for l in labels]
            # For FMM: higher TW = more confident = lower mismatch probability
            # ROC expects higher score = more likely positive (mismatch), so use -TW
            roc_scores = [-t for t in trusts]
            auc = auc_manual(roc_scores, mismatch_labels)

            sigma_results[label] = {
                'ece': ece, 'acc': acc, 'auc': auc, 'tw_mean': tw_mean, 'n': n,
            }

            # Pool
            pooled_data[label]['scores'].extend(roc_scores)
            pooled_data[label]['labels'].extend(mismatch_labels)

        all_results['per_sigma'][sigma_dir] = {'sigma': sigma_val, **sigma_results}
        print(f"  σ={sigma_val}m: " + " | ".join(
            f"{l}: ECE={sigma_results[l]['ece']:.4f} AUC={sigma_results[l]['auc']:.4f} Acc={sigma_results[l]['acc']:.4f}"
            for l in ['FMM', 'CaMM-iso', 'CaMM-cov'] if l in sigma_results))

    # Pooled AUC
    for label in ['FMM', 'CaMM-iso', 'CaMM-cov']:
        if pooled_data[label]['scores']:
            auc = auc_manual(pooled_data[label]['scores'], pooled_data[label]['labels'])
            all_results['pooled'][label] = {'auc': auc, 'n': len(pooled_data[label]['scores'])}
            print(f"\n  Pooled {label}: AUC={auc:.4f}, n={len(pooled_data[label]['scores'])}")

    # Save
    out_json = BASE / 'experiments/output/fig7_metrics_fixed.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_json}")

    # ── Step 3: ECE summary table ──
    print(f"\n{'='*70}")
    print("  ECE Summary")
    print(f"{'='*70}")
    print(f"  {'σ':>6s}  {'FMM':>10s}  {'CaMM-iso':>10s}  {'CaMM-cov':>10s}")
    for sd, sv in zip(SIGMA_DIRS, SIGMA_VALS):
        r = all_results['per_sigma'].get(sd, {})
        fmm_e = r.get('FMM', {}).get('ece', float('nan'))
        iso_e = r.get('CaMM-iso', {}).get('ece', float('nan'))
        cov_e = r.get('CaMM-cov', {}).get('ece', float('nan'))
        print(f"  {sv:6d}  {fmm_e:10.4f}  {iso_e:10.4f}  {cov_e:10.4f}")

    # ── Step 4: Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ECE panel
        sigmas = [all_results['per_sigma'][sd]['sigma'] for sd in SIGMA_DIRS if sd in all_results['per_sigma']]
        iso_ece = [all_results['per_sigma'][sd].get('CaMM-iso',{}).get('ece',np.nan) for sd in SIGMA_DIRS if sd in all_results['per_sigma']]
        cov_ece = [all_results['per_sigma'][sd].get('CaMM-cov',{}).get('ece',np.nan) for sd in SIGMA_DIRS if sd in all_results['per_sigma']]

        ax1.plot(sigmas, iso_ece, 's--', color='#1f77b4', linewidth=2, markersize=8, label='CaMM-iso (σ=0.001°)')
        ax1.plot(sigmas, cov_ece, 'o-', color='#d62728', linewidth=2, markersize=8, label='CaMM-cov (WLS covariance)')
        ax1.set_xlabel('Pseudorange noise σ (m)', fontsize=12)
        ax1.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
        ax1.set_title('ECE: Isotropic vs Covariance Emission', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, max(max(iso_ece), max(cov_ece)) * 1.15)

        # ROC panel
        for label, color, style in [('FMM', '#7f7f7f', '--'), ('CaMM-iso', '#1f77b4', '-.'), ('CaMM-cov', '#d62728', '-')]:
            scores = np.array(pooled_data[label]['scores'])
            labs = np.array(pooled_data[label]['labels'])
            # Sort by score descending
            idx = np.argsort(scores)[::-1]
            labs_sorted = labs[idx]
            scores_sorted = scores[idx]
            n_pos = labs.sum()
            n_neg = len(labs) - n_pos
            tpr = np.cumsum(labs_sorted) / n_pos
            fpr = np.cumsum(1 - labs_sorted) / n_neg
            auc_val = all_results['pooled'][label]['auc']
            ax2.plot(fpr, tpr, style, color=color, linewidth=2, label=f'{label} (AUC={auc_val:.3f})')

        ax2.plot([0, 1], [0, 1], ':', color='black', linewidth=1, label='Chance (AUC=0.5)')
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('Pooled ROC: Mismatch Detection (7 σ levels)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        plot_path = BASE / 'experiments/output/fig7_new_ece_roc.png'
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
    except ImportError as e:
        print(f"Plot skipped: {e}")


if __name__ == '__main__':
    main()
