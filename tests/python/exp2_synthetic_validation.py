#!/usr/bin/env python3
"""
Experiment 2: Synthetic-data validation of CMM posterior calibration.

Pipeline:
  1. Generate synthetic trajectories on the Hainan road network using the
     existing virtual-GNSS-constellation generator (generate_data_cmm.py).
  2. Post-process observations.csv to include full 6-element covariance
     (sdu/sdeu/sdun columns, required by CMM config).
  3. Run CMM+lag-sweep and FMM on the synthetic data.
  4. Compute calibration metrics (ECE, Brier, LogLoss) and compare against
     the real-data results from exp1_multitraj_lag_sweep.

Key insight: since synthetic vehicles ARE on the same road network used for
matching, the match quality should be near-perfect. This isolates the
emission-model calibration from road-network / off-road issues.

Usage:
  python3 tests/python/exp2_synthetic_validation.py [--trajs 10] [--points 500]
"""

import csv, json, math, re, subprocess, tempfile, sys, os
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[2]
GEN_SCRIPT = BASE / 'python/generate_data_cmm.py'
SHAPEFILE = BASE / 'input/map/haikou/edges.shp'
CMM_NETWORK = BASE / 'input/map/haikou/edges.shp'
CMM_UBODT = BASE / 'input/map/haikou_ubodt_indexed.bin'
XML_TEMPLATE = BASE / 'input/config/cmm_config_omp.xml'
FMM_CONFIG = BASE / 'input/config/fmm_config_omp.xml'
CMM_BIN = BASE / 'build/cmm'
FMM_BIN = BASE / 'build/fmm'
SYNTH_DIR = BASE / 'dataset-hainan-06/mr/synthetic'
THRESHOLD_M = 5.0
LAGS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry & Metrics
# ═══════════════════════════════════════════════════════════════════════════════

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

def compute_ece(confidences, labels, n_bins=10):
    N = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            idxs = [j for j in range(N) if lo <= confidences[j] <= hi]
        else:
            idxs = [j for j in range(N) if lo <= confidences[j] < hi]
        n_b = len(idxs)
        if n_b == 0:
            continue
        mc = sum(confidences[j] for j in idxs) / n_b
        acc = sum(labels[j] for j in idxs) / n_b
        ece += n_b / N * abs(mc - acc)
    return ece


def compute_metrics(confidences, labels):
    ece = compute_ece(confidences, labels)
    brier = sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)
    eps = 1e-15
    logloss = -sum(l * math.log(max(c, eps)) + (1 - l) * math.log(max(1 - c, eps))
                   for c, l in zip(confidences, labels)) / len(confidences)
    return {'ece': ece, 'brier': brier, 'logloss': logloss}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate synthetic data
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_trajs, n_points_per_traj, speed, sample_rate):
    """Run generate_data_cmm.py with Hainan shapefile. Returns path to obs CSV."""
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(GEN_SCRIPT),
        '--count', str(n_trajs),
        '--points', str(n_points_per_traj),
        '--speed', str(speed),
        '--sample-rate', str(sample_rate),
        '--num-sats', '8',
        '--shapefile', str(SHAPEFILE),
        '--output-dir', str(SYNTH_DIR),
        '--seed', '42',
        '--start-id', '1',
        '--jobs', '1',
    ]
    print(f"  Generating {n_trajs} trajectories ({n_points_per_traj} pts each)...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  Generation FAILED:\n{result.stderr[:500]}")
        sys.exit(1)
    print(f"  Done. Output in {SYNTH_DIR}")
    return SYNTH_DIR / 'observations.csv'


def reproject_to_wgs84(src_path, dst_path):
    """Reproject UTM coordinates AND covariance to WGS84 (lon, lat) degrees.

    Covariance Jacobian at (lat):
      dx_deg / dx_m = 1 / (111320 * cos(lat_rad))
      dy_deg / dy_m = 1 / 111320

    For covariance Sigma_m in meters, Sigma_deg = J * Sigma_m * J^T
    """
    import pyproj
    import numpy as np
    utm_epsg = 32649  # UTM zone 49N for Hainan
    transformer = pyproj.Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    METERS_PER_DEG_LAT = 111320.0

    with open(src_path, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin, delimiter=';')
        new_header = list(reader.fieldnames)
        with open(dst_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=new_header, delimiter=';')
            writer.writeheader()
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                lon, lat = transformer.transform(x, y)
                row['x'] = f'{lon:.8f}'
                row['y'] = f'{lat:.8f}'

                # Jacobian: meters → degrees at this latitude
                cos_lat = math.cos(math.radians(lat))
                jx = 1.0 / (METERS_PER_DEG_LAT * cos_lat)  # d_lon / d_x_m
                jy = 1.0 / METERS_PER_DEG_LAT               # d_lat / d_y_m

                # Covariance in meters → degrees via Jacobian rotation
                # Sigma_deg = J * Sigma_m * J^T
                # J = [[jx, 0], [0, jy]]
                sde_m = float(row['sde'])
                sdn_m = float(row['sdn'])
                sdne_m = float(row['sdne'])
                # Sigma_m = [[sde_m^2, sdne_m], [sdne_m, sdn_m^2]]
                # Sigma_deg[0,0] = jx^2 * sde_m^2
                # Sigma_deg[1,1] = jy^2 * sdn_m^2
                # Sigma_deg[0,1] = jx * jy * sdne_m
                sde_deg = abs(jx) * sde_m
                sdn_deg = abs(jy) * sdn_m
                sdne_deg = jx * jy * sdne_m

                row['sde'] = f'{sde_deg:.12f}'
                row['sdn'] = f'{sdn_deg:.12f}'
                row['sdne'] = f'{sdne_deg:.16f}'
                # Vertical components (ignored in 2D matching, keep 0)
                row['sdu'] = '0.0'
                row['sdeu'] = '0.0'
                row['sdun'] = '0.0'
                # protection_level also needs scaling: scale * max(jx, jy)
                pl_m = float(row['protection_level'])
                pl_deg = pl_m * max(abs(jx), abs(jy))
                row['protection_level'] = f'{pl_deg:.12f}'

                writer.writerow(row)
    print(f"  Reprojected (coords + covariance) to WGS84 → {dst_path}")
    return dst_path


def fix_observations_csv(src_path, dst_path):
    """Add missing sdu/sdeu/sdun columns to match CMM config format."""
    with open(src_path, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin, delimiter=';')
        new_header = ['id', 'timestamp', 'x', 'y',
                      'sde', 'sdn', 'sdu', 'sdne', 'sdeu', 'sdun',
                      'protection_level']
        with open(dst_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=new_header, delimiter=';')
            writer.writeheader()
            for row in reader:
                writer.writerow({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'x': row['x'],
                    'y': row['y'],
                    'sde': row['sde'],
                    'sdn': row['sdn'],
                    'sdu': '0.0',
                    'sdne': row['sdne'],
                    'sdeu': '0.0',
                    'sdun': '0.0',
                    'protection_level': row['protection_level'],
                })
    print(f"  Fixed observations → {dst_path}")
    return dst_path


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: XML config builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_cmm_xml(xml_template, gps_csv, out_file, lag):
    """Create temp XML pointing at synthetic GPS CSV, with given lag_steps."""
    tree = ET.parse(xml_template)
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    params = root.find('parameters')
    lag_el = params.find('lag_steps')
    if lag_el is None:
        lag_el = ET.SubElement(params, 'lag_steps')
    lag_el.text = str(lag)
    root.find('input').find('gps').find('file').text = str(gps_csv)
    # Switch network and UBODT to match synthetic data (Haikou)
    root.find('input').find('network').find('file').text = str(CMM_NETWORK)
    root.find('input').find('ubodt').find('file').text = str(CMM_UBODT)
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


def build_fmm_xml(fmm_xml, gps_csv, out_file):
    """Create temp FMM config pointing at synthetic GPS CSV."""
    tree = ET.parse(fmm_xml)
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    root.find('input').find('gps').find('file').text = str(gps_csv)
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Run matching & compute metrics
# ═══════════════════════════════════════════════════════════════════════════════

def run_cmm_lag_sweep(gps_csv, lags):
    """Run CMM for each lag value, return {lag: metrics_dict}."""
    results = {}
    for lag in lags:
        out_file = str(SYNTH_DIR / f'cmm_synth_lag{lag:03d}.csv')
        tmp_xml = build_cmm_xml(XML_TEMPLATE, gps_csv, out_file, lag)
        try:
            subprocess.run([str(CMM_BIN), str(tmp_xml)], check=True,
                          capture_output=True, cwd=str(BASE), timeout=300)
        except subprocess.CalledProcessError as e:
            print(f"    CMM lag={lag} FAILED")
            os.unlink(tmp_xml)
            continue
        os.unlink(tmp_xml)
        m = analyze_cmm_output(out_file)
        m['lag'] = lag
        results[lag] = m
    return results


def run_fmm(gps_csv):
    """Run FMM once, return metrics dict."""
    out_file = str(SYNTH_DIR / 'fmm_synth.csv')
    tmp_xml = build_fmm_xml(FMM_CONFIG, gps_csv, out_file)
    try:
        subprocess.run([str(FMM_BIN), str(tmp_xml)], check=True,
                      capture_output=True, cwd=str(BASE), timeout=120)
    except subprocess.CalledProcessError as e:
        print(f"    FMM FAILED")
        os.unlink(tmp_xml)
        return None
    os.unlink(tmp_xml)
    return analyze_fmm_output(out_file)


def analyze_cmm_output(out_path):
    """Compute calibration metrics from CMM output CSV."""
    trusts, eps, errors = [], [], []
    with open(out_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            og = parse_point(row.get('ogeom', ''))
            pg = parse_point(row.get('pgeom', ''))
            if og is None or pg is None:
                continue
            err = haversine_m(*og, *pg)
            errors.append(err)
            try:
                trusts.append(float(row.get('trustworthiness', '0')))
                eps.append(float(row.get('ep', '0')))
            except (ValueError, TypeError):
                continue

    if not errors:
        return {'n': 0}

    labels = [1 if e <= THRESHOLD_M else 0 for e in errors]
    n = len(errors)
    m = compute_metrics(trusts, labels)
    m['n'] = n
    m['n_correct'] = sum(labels)
    m['error_mean'] = sum(errors) / n
    m['error_median'] = sorted(errors)[n // 2]
    m['error_max'] = max(errors)

    # EP calibration
    ep_labels = labels
    ep_m = compute_metrics(eps, ep_labels)
    m['ece_ep'] = ep_m['ece']
    m['brier_ep'] = ep_m['brier']
    return m


def analyze_fmm_output(out_path):
    """Compute calibration metrics from FMM output CSV using ogeom from CMM."""
    # FMM doesn't output ogeom. Need to match to CMM output by timestamp.
    # But the synthetic FMM output also won't have ogeom.
    # Compute error = distance from original GPS (in input) to pgeom (matched).
    # This requires reading the ground truth points.
    # For synthetic data, use the observation x,y as ground truth.
    trusts, eps, errors = [], [], []
    # We need to match FMM output (has timestamp, pgeom) with synthetic input (has x,y as ground truth)
    # Load ground truth from reprojected observations
    obs_file = SYNTH_DIR / 'observations_cmm.csv'
    gt_by_ts = {}
    with open(obs_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            ts = row.get('timestamp', '').strip()
            gt_by_ts[ts] = (float(row['x']), float(row['y']))

    with open(out_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            pg = parse_point(row.get('pgeom', ''))
            if pg is None:
                continue
            ts = row.get('timestamp', '').strip()
            gt = gt_by_ts.get(ts)
            if gt is None:
                continue
            err = haversine_m(*gt, *pg)
            errors.append(err)
            try:
                trusts.append(float(row.get('trustworthiness', '0')))
                eps.append(float(row.get('ep', '0')))
            except (ValueError, TypeError):
                continue

    if not errors:
        return {'n': 0}

    labels = [1 if e <= THRESHOLD_M else 0 for e in errors]
    n = len(errors)
    m = compute_metrics(trusts, labels)
    m['n'] = n
    m['n_correct'] = sum(labels)
    m['error_mean'] = sum(errors) / n
    m['error_median'] = sorted(errors)[n // 2]
    m['error_max'] = max(errors)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--trajs', type=int, default=10, help='Number of trajs')
    ap.add_argument('--points', type=int, default=500, help='Points per traj')
    ap.add_argument('--speed', type=float, default=12.0)
    ap.add_argument('--rate', type=float, default=1.0)
    ap.add_argument('--skip-gen', action='store_true')
    args = ap.parse_args()

    print(f"{'=' * 78}")
    print(f"  Experiment 2: Synthetic Data Validation")
    print(f"  Shapefile: {SHAPEFILE}")
    print(f"{'=' * 78}")

    # ── Step 1: Generate ──
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    final_csv = SYNTH_DIR / 'observations_cmm.csv'

    if not args.skip_gen or not final_csv.exists():
        print(f"\n  [Step 1] Generating synthetic trajectories...")
        obs_csv = generate_synthetic_data(args.trajs, args.points, args.speed, args.rate)
        fixed_csv = SYNTH_DIR / 'observations_fixed.csv'
        fix_observations_csv(obs_csv, fixed_csv)
        reproject_to_wgs84(fixed_csv, final_csv)
    else:
        print(f"\n  [Step 1] Using existing {final_csv}")

    print(f"\n  [Step 2] Running CMM lag sweep ({len(LAGS)} lags)...")
    cmm_results = run_cmm_lag_sweep(final_csv, LAGS)

    print(f"\n  [Step 3] Running FMM baseline...")
    fmm_result = run_fmm(final_csv)

    # ── Summary ──
    print(f"\n{'=' * 78}")
    print(f"  RESULTS — Synthetic Data (vehicle ON road network)")
    print(f"{'=' * 78}")

    # Error stats
    total = sum(cmm_results[l]['n'] for l in LAGS for _ in [0] if l in cmm_results)
    if 0 in cmm_results:
        r0 = cmm_results[0]
        print(f"\n  Synthetic CMM: {r0['n']} points (lag=0)")
        print(f"    Error: mean={r0['error_mean']:.1f}m  median={r0['error_median']:.1f}m  max={r0['error_max']:.1f}m")
        print(f"    Correct ≤{THRESHOLD_M:.0f}m: {r0['n_correct']/r0['n']*100:.1f}%")

    if fmm_result and fmm_result.get('n', 0) > 0:
        print(f"\n  Synthetic FMM: {fmm_result['n']} points")
        print(f"    Error: mean={fmm_result['error_mean']:.1f}m  median={fmm_result['error_median']:.1f}m  max={fmm_result['error_max']:.1f}m")
        print(f"    Correct ≤{THRESHOLD_M:.0f}m: {fmm_result['n_correct']/fmm_result['n']*100:.1f}%")

    # Calibration table
    print(f"\n  Calibration (CMM trustworthiness):")
    print(f"  {'lag':>5s}  {'ECE(tw)':>10s}  {'ECE(ep)':>10s}  {'Brier':>10s}  {'LogLoss':>10s}  {'%Correct':>10s}  {'ΔECE':>10s}")
    print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    prev_ece = None
    for lag in LAGS:
        if lag not in cmm_results:
            continue
        r = cmm_results[lag]
        d = f"{prev_ece - r['ece']:+10.6f}" if prev_ece is not None else ""
        pct = f"  {r['n_correct'] / r['n'] * 100:9.1f}%  "
        print(f"  {lag:5d}  {r['ece']:10.6f}  {r['ece_ep']:10.6f}  {r['brier']:10.6f}  {r['logloss']:10.6f}{pct}{d}")
        prev_ece = r['ece']

    # FMM calibration
    if fmm_result and fmm_result.get('n', 0) > 0:
        print(f"\n  FMM calibration (trustworthiness):")
        print(f"    ECE = {fmm_result['ece']:.4f}  Brier = {fmm_result['brier']:.4f}  LogLoss = {fmm_result['logloss']:.4f}")

    # Comparison table
    print(f"\n{'=' * 78}")
    print(f"  COMPARISON: Synthetic vs Real Data")
    print(f"{'=' * 78}")
    print(f"  {'':20s}  {'Synthetic':>15s}  {'Real (mean)':>15s}  {'Improvement':>15s}")
    print(f"  {'-' * 20}  {'-' * 15}  {'-' * 15}  {'-' * 15}")

    best_synth_lag = min(cmm_results, key=lambda l: cmm_results[l]['ece']) if cmm_results else 0
    best_synth = cmm_results.get(best_synth_lag, {'ece': 1, 'brier': 1})
    best_synth0 = cmm_results.get(0, best_synth)

    print(f"  {'ECE (lag=0)':20s}  {best_synth0['ece']:15.4f}  {0.477:15.4f}  {best_synth0['ece'] - 0.477:+15.4f}")
    print(f"  {'ECE (best lag)':20s}  {best_synth['ece']:15.4f}  {0.477:15.4f}  {best_synth['ece'] - 0.477:+15.4f}")
    print(f"  {'%Correct (lag=0)':20s}  {best_synth0['n_correct']/best_synth0['n']*100 if best_synth0.get('n') else 0:14.1f}%  {'--':>15s}  {'--':>15s}")
    print(f"  {'Best lag':20s}  {best_synth_lag:>15d}")
    print(f"  {'Error median (m)':20s}  {best_synth0['error_median']:15.1f}  {3.0:15.1f}  --")


if __name__ == '__main__':
    main()
