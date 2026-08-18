#!/usr/bin/env python3
"""
Re-run HMM (FMM) baseline on real vehicle data with FAIR configuration.

The paper's original FMM config used r=0.03° (~3.3km) and gps_error=0.001° (~111m),
which are 10-100× too large for EPSG:4326 coordinates. This inflates the CaMM-HMM gap.

Corrected fair config:
  - k=16 (same as CaMM)
  - r=0.003° (~300m at equator, comparable to CaMM's effective HPL search radius)
  - gps_error=0.0005° (~50m at equator, FMM's default in metric units)

Also runs FMM with several alternative configs for sensitivity analysis:
  - r=0.001° (~100m), gps_error=0.0003° (~30m) — tighter
  - r=0.006° (~600m), gps_error=0.001° (~100m) — original paper values

Usage:
  python3 python/experiments/rerun_fmm_baseline.py
"""

import csv, math, re, subprocess, sys, os, tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
FMM_BIN = BASE / 'build/fmm'
REAL_DATA = BASE / 'data/real_vehicle/hainan_06/cmm_input_points.csv'
CMM_RESULT = BASE / 'data/real_vehicle/hainan_06/processed/cmm_result.csv'
OUTPUT_DIR = BASE / 'data/real_vehicle/mr/fmm_fair_rerun'
THRESHOLD_M = 5.0
N_BINS = 10

PP_RE = re.compile(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', re.I)

def parse_point(wkt):
    m = PP_RE.search(wkt or '')
    return (float(m.group(1)), float(m.group(2))) if m else None

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def compute_ece(confidences, labels, n_bins=N_BINS):
    N = len(confidences)
    ece = 0.0
    per_bin = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            idxs = [j for j in range(N) if lo <= confidences[j] <= hi]
        else:
            idxs = [j for j in range(N) if lo <= confidences[j] < hi]
        n_b = len(idxs)
        if n_b == 0: continue
        mc = sum(confidences[j] for j in idxs) / n_b
        acc = sum(labels[j] for j in idxs) / n_b
        ece += n_b / N * abs(mc - acc)
        per_bin.append({'lo': lo, 'hi': hi, 'mc': mc, 'acc': acc, 'n': n_b})
    return ece, per_bin

def compute_brier(confidences, labels):
    return sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)


# ═══════════════════════════════════════════════════════════════════════════
# FMM Config Variants
# ═══════════════════════════════════════════════════════════════════════════

FMM_CONFIGS = {
    'fair_default': {
        'k': 16, 'r': 0.003, 'gps_error': 0.0005, 'reverse_tolerance': 0.0,
        'label': 'Fair (r=0.003°, σ=0.0005°)'
    },
    'fair_tight': {
        'k': 16, 'r': 0.001, 'gps_error': 0.0003, 'reverse_tolerance': 0.0,
        'label': 'Tight (r=0.001°, σ=0.0003°)'
    },
    'original_paper': {
        'k': 16, 'r': 0.03, 'gps_error': 0.001, 'reverse_tolerance': 0.0,
        'label': 'Original (r=0.03°, σ=0.001°)'
    },
}


def make_fmm_xml(gps_file, out_file, k, radius, gps_error):
    """Create FMM config XML for real data."""
    tree = ET.parse(BASE / 'input/config/fmm_config_omp.xml')
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    root.find('input').find('gps').find('file').text = str(gps_file)
    params = root.find('parameters')
    for tag, val in [('k', str(k)), ('r', str(radius)), ('gps_error', str(gps_error))]:
        el = params.find(tag)
        if el is None: el = ET.SubElement(params, tag)
        el.text = val
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


def run_fmm(gps_file, out_file, k, radius, gps_error):
    """Run FMM binary."""
    tmp_xml = make_fmm_xml(gps_file, out_file, k, radius, gps_error)
    try:
        subprocess.run([str(FMM_BIN), str(tmp_xml)], check=True,
                       capture_output=True, cwd=str(BASE), timeout=600)
    except subprocess.CalledProcessError as e:
        print(f"  FMM FAILED: {e.stderr[:300] if e.stderr else str(e)}")
        os.unlink(tmp_xml)
        return None
    os.unlink(tmp_xml)
    return out_file


def analyze_fmm_real(out_path, gt_points):
    """Analyze FMM output matched to CMM ogeom by sequential order."""
    trusts, errors = [], []
    gt_idx = 0
    n_failed = 0
    with open(out_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            status = row.get('status', '')
            if 'FAILED' in status:
                n_failed += 1
                gt_idx += 1
                continue
            pg = parse_point(row.get('pgeom', ''))
            if pg is None:
                gt_idx += 1
                continue
            # Sequential matching (FMM doesn't preserve good timestamps)
            if gt_idx < len(gt_points):
                gt = gt_points[gt_idx]
            else:
                break
            gt_idx += 1
            errors.append(haversine_m(*gt, *pg))
            try:
                trusts.append(float(row.get('trustworthiness', '0')))
            except (ValueError, TypeError):
                continue
    return trusts, errors, n_failed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ground truth (ogeom) from CMM result
    print("Loading ground truth from CMM result...")
    gt_points = []
    with open(CMM_RESULT, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            og = parse_point(row.get('ogeom', ''))
            if og: gt_points.append(og)
    print(f"  {len(gt_points)} ground truth points loaded")

    # Load CMM trustworthiness for comparison
    cmm_tw = []
    with open(CMM_RESULT, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            try:
                cmm_tw.append(float(row.get('trustworthiness', '0')))
            except (ValueError, TypeError):
                pass
    print(f"  {len(cmm_tw)} CMM trustworthiness values loaded")

    print(f"\n{'='*80}")
    print(f"  HMM Baseline Re-run — Real Vehicle Data (7 trajectories, Haikou)")
    print(f"{'='*80}")

    results = {}
    for config_name, cfg in FMM_CONFIGS.items():
        print(f"\n  [{config_name}] {cfg['label']}")
        out_file = str(OUTPUT_DIR / f'fmm_{config_name}.csv')

        if Path(out_file).exists():
            print(f"    Using existing result file")
        else:
            run_fmm(REAL_DATA, out_file, cfg['k'], cfg['r'], cfg['gps_error'])

        if not Path(out_file).exists():
            print(f"    Output file not found, skipping")
            continue

        trusts, errors, n_failed = analyze_fmm_real(out_file, gt_points)

        if not trusts:
            print(f"    No valid matches found!")
            continue

        labels = [1 if e <= THRESHOLD_M else 0 for e in errors]
        ece, bins = compute_ece(trusts, labels)
        brier = compute_brier(trusts, labels)
        acc = sum(labels) / len(labels)
        err_mean = sum(errors) / len(errors)
        err_median = sorted(errors)[len(errors) // 2]

        results[config_name] = {
            'ece': ece, 'brier': brier, 'accuracy': acc,
            'n': len(trusts), 'n_failed': n_failed,
            'error_mean': err_mean, 'error_median': err_median,
        }

        print(f"    n={len(trusts)}  failed={n_failed}  Acc(≤{THRESHOLD_M}m)={acc:.4f}")
        print(f"    ECE={ece:.4f}  Brier={brier:.4f}")
        print(f"    Error: mean={err_mean:.1f}m  median={err_median:.1f}m")

    # ── Comparison Summary ──
    print(f"\n{'='*80}")
    print(f"  COMPARISON: CMM vs HMM (fair config)")
    print(f"{'='*80}")

    # Compute CMM metrics
    cmm_labels = [1 if e <= THRESHOLD_M else 0 for e in
                  [haversine_m(*gt_points[i], *parse_point(...)) for i in range(len(gt_points))]]
    # Actually we need CMM errors — load from CMM result
    cmm_errors = []
    with open(CMM_RESULT, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            og = parse_point(row.get('ogeom', ''))
            pg = parse_point(row.get('pgeom', ''))
            if og and pg:
                cmm_errors.append(haversine_m(*og, *pg))
    cmm_labels = [1 if e <= THRESHOLD_M else 0 for e in cmm_errors]
    cmm_ece, _ = compute_ece(cmm_tw[:len(cmm_labels)], cmm_labels)
    cmm_brier = compute_brier(cmm_tw[:len(cmm_labels)], cmm_labels)
    cmm_acc = sum(cmm_labels) / len(cmm_labels) if cmm_labels else 0
    cmm_err_mean = sum(cmm_errors) / len(cmm_errors) if cmm_errors else 0

    print(f"\n  {'Config':<25s}  {'ECE':>8s}  {'Brier':>8s}  {'Acc':>8s}  {'Err_mean':>10s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    print(f"  {'CMM (original)':<25s}  {cmm_ece:8.4f}  {cmm_brier:8.4f}  {cmm_acc:8.4f}  {cmm_err_mean:8.1f}m")

    for config_name, cfg in FMM_CONFIGS.items():
        r = results.get(config_name)
        if r:
            print(f"  {cfg['label']:<25s}  {r['ece']:8.4f}  {r['brier']:8.4f}  {r['accuracy']:8.4f}  {r['error_mean']:8.1f}m")

    # ── Save results ──
    import json
    out_json = OUTPUT_DIR / 'fmm_rerun_results.json'
    with open(out_json, 'w') as f:
        json.dump({
            'cmm': {'ece': cmm_ece, 'brier': cmm_brier, 'accuracy': cmm_acc,
                    'error_mean': cmm_err_mean, 'n': len(cmm_errors)},
            'fmm_configs': {k: v for k, v in results.items()},
        }, f, indent=2)
    print(f"\n  Results saved to {out_json}")


if __name__ == '__main__':
    main()
