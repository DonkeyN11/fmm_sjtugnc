#!/usr/bin/env python3
"""
Experiment 5: Component-wise ablation study for CaMM calibration.

Tests four progressive configurations across 3×3 experimental conditions:
  Sigma (pseudorange noise): 1m, 10m, 30m
  Sample interval:           1s, 10s, 30s

Configurations (cumulative components):
  A. HMM    — FMM baseline: isotropic EP, fixed-radius search, normalized Viterbi TW
  B. CaMM-EP — + HPL-adaptive search + covariance-based Mahalanobis EP (no direction, no τ)
  C. CaMM-Dir — + direction-consistency von Mises penalty
  D. CaMM-Full — + entropy-aware temperature scaling τ + background state

Output:
  - Per-condition ECE values (console table)
  - 3×3 grouped bar chart: rows=sigma, cols=sample_interval, bars=configs
  - JSON results file for paper integration

Usage:
  python3 python/experiments/exp5_ablation_components.py [--trajs 10] [--points 500]
"""

import csv, json, math, re, subprocess, sys, os, tempfile, time
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict
import argparse

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parents[2]
GEN_SCRIPT = BASE / 'experiments/scripts/generate_data_cmm.py'
SHAPEFILE = BASE / 'input/map/hainan/edges.shp'      # Full Hainan network (152,547 edges)
CMM_NETWORK = SHAPEFILE
CMM_UBODT = BASE / 'input/map/hainan_ubodt_indexed.bin'
CMM_BIN = BASE / 'build/cmm'
FMM_BIN = BASE / 'build/fmm'
OUTPUT_DIR = BASE / 'experiments/output/8_ablation'
THRESHOLD_M = 5.0
N_BINS = 10

# ═══════════════════════════════════════════════════════════════════════════════
# Experimental grid
# ═══════════════════════════════════════════════════════════════════════════════
SIGMA_VALUES = [1.0, 10.0, 30.0]        # Pseudorange noise σ (m)
SAMPLE_RATES = [1.0, 0.1, 1.0/30.0]     # Hz → intervals: 1s, 10s, 30s
INTERVAL_LABELS = ['1s', '10s', '30s']

PP_RE = re.compile(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', re.I)

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

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

def compute_brier(confidences, labels):
    return sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate synthetic data
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(sigma_pr, sample_rate, n_trajs, n_points, speed, seed):
    """Generate synthetic data for one (sigma, sample_rate) combination."""
    dataset_dir = OUTPUT_DIR / f'sigma{sigma_pr:.0f}_int{1.0/sample_rate:.0f}s'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    obs_csv = dataset_dir / 'observations.csv'
    if obs_csv.exists():
        print(f"    Using existing {obs_csv}")
        return dataset_dir, obs_csv

    cmd = [
        sys.executable, str(GEN_SCRIPT),
        '--count', str(n_trajs),
        '--points', str(n_points),
        '--speed', str(speed),
        '--sample-rate', str(sample_rate),
        '--num-sats', '8',
        '--min-sigma-pr', str(sigma_pr),
        '--max-sigma-pr', str(sigma_pr),
        '--shapefile', str(SHAPEFILE),
        '--output-dir', str(dataset_dir),
        '--seed', str(seed),
        '--start-id', '1',
        '--jobs', '1',
    ]
    print(f"    Generating sigma={sigma_pr}m, rate={sample_rate}Hz ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    FAILED:\n{result.stderr[:500]}")
        return None, None
    print(f"    Generated {dataset_dir}")
    return dataset_dir, obs_csv


def reproject_to_wgs84(src_path, dst_path):
    """Reproject UTM coordinates + covariance to WGS84 degrees.

    Detects if input is already in degree range and skips if so.
    """
    import pyproj
    import numpy as np

    # Check if data is already in WGS84 degree range
    with open(src_path, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin, delimiter=';')
        first_row = next(reader)
        x0, y0 = float(first_row['x']), float(first_row['y'])

    # If x in [-180, 180] and y in [-90, 90], assume already WGS84
    if -180 <= x0 <= 180 and -90 <= y0 <= 90:
        print(f"    Data already in WGS84 degree range, copying as-is")
        import shutil
        shutil.copy(src_path, dst_path)
        return dst_path

    utm_epsg = 32649
    transformer = pyproj.Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    METERS_PER_DEG_LAT = 111320.0

    with open(src_path, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin, delimiter=';')
        fieldnames = reader.fieldnames
        with open(dst_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for row in reader:
                x = float(row['x']); y = float(row['y'])
                lon, lat = transformer.transform(x, y)
                cos_lat = math.cos(math.radians(lat))
                jx = 1.0 / (METERS_PER_DEG_LAT * cos_lat)
                jy = 1.0 / METERS_PER_DEG_LAT

                row['x'] = f'{lon:.8f}'
                row['y'] = f'{lat:.8f}'
                sde_m = float(row['sde']); sdn_m = float(row['sdn']); sdne_m = float(row['sdne'])
                row['sde'] = f'{abs(jx) * sde_m:.12f}'
                row['sdn'] = f'{abs(jy) * sdn_m:.12f}'
                row['sdne'] = f'{jx * jy * sdne_m:.16f}'
                if 'sdu' in fieldnames:
                    row['sdu'] = '0.0'
                    row['sdeu'] = '0.0'
                    row['sdun'] = '0.0'
                pl_m = float(row['protection_level'])
                row['protection_level'] = f'{pl_m * max(abs(jx), abs(jy)):.12f}'
                writer.writerow(row)
    return dst_path


def fix_for_cmm(src_path, dst_path):
    """Add sdu/sdeu/sdun columns and reorder to match CMM config format."""
    with open(src_path, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin, delimiter=';')
        with open(dst_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=[
                'id', 'timestamp', 'x', 'y',
                'sde', 'sdn', 'sdu', 'sdne', 'sdeu', 'sdun',
                'protection_level'
            ], delimiter=';')
            writer.writeheader()
            for row in reader:
                writer.writerow({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'x': row['x'], 'y': row['y'],
                    'sde': row['sde'], 'sdn': row['sdn'],
                    'sdu': row.get('sdu', '0.0'),
                    'sdne': row['sdne'],
                    'sdeu': row.get('sdeu', '0.0'),
                    'sdun': row.get('sdun', '0.0'),
                    'protection_level': row['protection_level'],
                })
    return dst_path


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Run matching
# ═══════════════════════════════════════════════════════════════════════════════

def make_fmm_xml(gps_csv, out_file, k=16, radius=0.008, gps_error=0.0005):
    """Create FMM config XML with fair parameters for degree-based coordinates.

    radius=0.008° ≈ 800m at equator. This is generous enough to find candidates
    but still much more constrained than the original paper's 0.03° (3.3km).
    gps_error=0.0005° ≈ 50m, matching FMM's default in metric units.
    """
    tree = ET.parse(BASE / 'input/config/fmm_config_omp.xml')
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    root.find('input').find('gps').find('file').text = str(gps_csv)
    # Ensure FMM uses the SAME network as the ground truth generation
    root.find('input').find('network').find('file').text = str(SHAPEFILE)
    root.find('input').find('ubodt').find('file').text = str(CMM_UBODT)
    # Remove use_omp to avoid buffered output issues
    other = root.find('other')
    for tag in ['use_omp']:
        el = other.find(tag)
        if el is not None: other.remove(el)
    params = root.find('parameters')
    for tag, val in [('k', str(k)), ('r', str(radius)), ('gps_error', str(gps_error))]:
        el = params.find(tag)
        if el is None: el = ET.SubElement(params, tag)
        el.text = val
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


def make_cmm_xml(gps_csv, out_file, config_overrides):
    """Create CMM config XML with specified overrides."""
    tree = ET.parse(BASE / 'input/config/cmm_config_omp.xml')
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    root.find('input').find('gps').find('file').text = str(gps_csv)
    root.find('input').find('network').find('file').text = str(CMM_NETWORK)
    root.find('input').find('ubodt').find('file').text = str(CMM_UBODT)
    params = root.find('parameters')
    for tag, val in config_overrides.items():
        el = params.find(tag)
        if el is None: el = ET.SubElement(params, tag)
        el.text = str(val)
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


def run_fmm(gps_csv, out_file):
    """Run FMM and return output path."""
    tmp_xml = make_fmm_xml(gps_csv, out_file)
    try:
        subprocess.run([str(FMM_BIN), str(tmp_xml)], check=True,
                       capture_output=True, cwd=str(BASE), timeout=300)
    except subprocess.CalledProcessError as e:
        print(f"      FMM FAILED: {e.stderr[:200] if e.stderr else str(e)}")
        os.unlink(tmp_xml)
        return None
    os.unlink(tmp_xml)
    return out_file


def run_cmm(gps_csv, out_file, overrides):
    """Run CMM with given config overrides."""
    tmp_xml = make_cmm_xml(gps_csv, out_file, overrides)
    try:
        subprocess.run([str(CMM_BIN), str(tmp_xml)], check=True,
                       capture_output=True, cwd=str(BASE), timeout=300)
    except subprocess.CalledProcessError as e:
        print(f"      CMM FAILED: {e.stderr[:200] if e.stderr else str(e)}")
        os.unlink(tmp_xml)
        return None
    os.unlink(tmp_xml)
    return out_file


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Analyze results — SEGMENT-LEVEL correctness
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth_edges(gt_csv_path):
    """Load per-epoch ground truth edge IDs from ground_truth.csv.

    Returns: dict {(traj_id, seq): edge_id_string}
    """
    gt_edges = {}
    with open(gt_csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            traj_id = row['id']
            pt_edges_raw = row.get('point_edge_ids', '[]')
            try:
                pt_edges = json.loads(pt_edges_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            for seq, edge_id in enumerate(pt_edges):
                gt_edges[(traj_id, str(seq))] = str(edge_id)
    return gt_edges


def analyze_fmm_output_segments(out_path, gt_edges):
    """Load FMM output, compare matched segment (cpath) to ground truth edge.

    FMM point-mode output has 'cpath' field = matched edge ID per epoch.
    """
    trusts, labels = [], []
    n_failed = 0
    with open(out_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            traj_id = row.get('id', '').strip()
            seq = row.get('seq', '').strip()
            status = row.get('status', '')
            cpath = row.get('cpath', '').strip()

            gt_edge = gt_edges.get((traj_id, seq))
            if gt_edge is None:
                continue  # no ground truth for this epoch

            # Label = 1 if matched edge matches ground truth edge
            if 'FAILED' in status or not cpath:
                labels.append(0)
                n_failed += 1
            else:
                labels.append(1 if str(cpath) == gt_edge else 0)

            try:
                trusts.append(float(row.get('trustworthiness', '0')))
            except (ValueError, TypeError):
                trusts.append(0.0)

    return trusts, labels, n_failed


def analyze_cmm_output_segments(out_path, gt_edges):
    """Load CMM output, compare matched segment (opath) to ground truth edge.

    CMM output has 'opath' field = matched edge ID per epoch.
    """
    trusts, labels = [], []
    n_failed = 0
    with open(out_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            traj_id = row.get('id', '').strip()
            seq = row.get('seq', '').strip()
            status = row.get('status', '')
            opath = row.get('opath', '').strip()

            gt_edge = gt_edges.get((traj_id, seq))
            if gt_edge is None:
                continue

            if 'FAILED' in status or not opath:
                labels.append(0)
                n_failed += 1
            else:
                labels.append(1 if str(opath) == gt_edge else 0)

            try:
                trusts.append(float(row.get('trustworthiness', '0')))
            except (ValueError, TypeError):
                trusts.append(0.0)

    return trusts, labels, n_failed


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Configuration definitions
# ═══════════════════════════════════════════════════════════════════════════════

def get_cmm_configs():
    """Return the three CMM ablation configurations.

    All share: k=16, min_candidates=1, protection_level_multiplier=10,
    phmi=0, lag_steps=0, map_error_std=5e-6, normalized=false

    Key design choice: background_prob=0.1 is included in ALL CaMM variants
    because it is essential for preventing over-confidence when GNSS quality
    is poor (all road candidates have low EP). Without it, softmax artificially
    inflates the best candidate's posterior → ECE explodes at high sigma.
    """
    common = {
        'k': '16',
        'min_candidates': '1',
        'protection_level_multiplier': '10',
        'reverse_tolerance': '0.0',
        'normalized': 'false',
        'use_mahalanobis': 'true',
        'filtered': 'false',
        'phmi': '0.0',              # Disable PHMI normalization for clean ablation
        'phmi_pl_multiplier': '1',
        'background_prob': '0.1',    # Essential for calibration; included in all variants
        'lag_steps': '0',
        'map_error_std': '5.0e-6',
        'cumulative_reverse_pct': '0.0',
    }

    configs = {}

    # B: CaMM-EP (HPL search + covariance EP, no direction, no temperature)
    configs['B_CaMM-EP'] = {
        **common,
        'direction_penalty': 'false',
        'temperature_adapt': 'false',
    }

    # C: CaMM-Dir (+ direction consistency)
    configs['C_CaMM-Dir'] = {
        **common,
        'direction_penalty': 'true',
        'temperature_adapt': 'false',
    }

    # D: CaMM-Full (+ temperature scaling)
    configs['D_CaMM-Full'] = {
        **common,
        'direction_penalty': 'true',
        'temperature_adapt': 'true',
    }

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trajs', type=int, default=10)
    ap.add_argument('--points', type=int, default=500)
    ap.add_argument('--speed', type=float, default=12.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-gen', action='store_true')
    ap.add_argument('--skip-fmm', action='store_true')
    ap.add_argument('--skip-cmm', action='store_true')
    ap.add_argument('--condition', type=str, default=None,
                    help='Run single condition, e.g. "sigma1_int1s"')
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Build experimental grid
    # ═══════════════════════════════════════════════════════════════════════
    conditions = []
    for sigma in SIGMA_VALUES:
        for rate, label in zip(SAMPLE_RATES, INTERVAL_LABELS):
            conditions.append({
                'sigma': sigma, 'rate': rate, 'label': label,
                'name': f'sigma{sigma:.0f}_int{label}'
            })

    if args.condition:
        conditions = [c for c in conditions if c['name'] == args.condition]
        if not conditions:
            print(f"Unknown condition: {args.condition}")
            print(f"Available: {[c['name'] for c in conditions]}")
            return 1

    all_results = {}
    grand_table = []

    for cond in conditions:
        sigma, rate, label, name = cond['sigma'], cond['rate'], cond['label'], cond['name']
        print(f"\n{'='*70}")
        print(f"  Condition: σ_pr={sigma:.0f}m, Δt={label}")
        print(f"{'='*70}")

        # ── Generate data ──
        dataset_dir, obs_csv = generate_dataset(
            sigma, rate, args.trajs, args.points, args.speed, args.seed)
        if obs_csv is None:
            continue

        # Reproject + fix
        obs_wgs84 = dataset_dir / 'observations_wgs84.csv'
        obs_cmm = dataset_dir / 'observations_cmm.csv'

        if not args.skip_gen or not obs_cmm.exists():
            reproject_to_wgs84(obs_csv, obs_wgs84)
            fix_for_cmm(obs_wgs84, obs_cmm)

        # Build ground truth: per-epoch segment ID from ground_truth.csv
        gt_csv = dataset_dir / 'ground_truth.csv'
        gt_edges = load_ground_truth_edges(gt_csv)
        print(f"    Loaded {len(gt_edges)} ground truth edge labels")

        cond_results = {}

        # ── A: HMM baseline (FMM) ──
        fmm_out = str(dataset_dir / 'result_fmm.csv')
        if not args.skip_fmm:
            print(f"    [A] HMM baseline (FMM)...")
            fmm_out = run_fmm(obs_cmm, fmm_out)
        if fmm_out and Path(fmm_out).exists():
            trusts, labels, n_failed = analyze_fmm_output_segments(fmm_out, gt_edges)
        else:
            trusts, labels, n_failed = [], [], 0
        if trusts:
            ece = compute_ece(trusts, labels)
            brier = compute_brier(trusts, labels)
            acc = sum(labels) / len(labels)
            cond_results['A_HMM'] = {
                'ece': ece, 'brier': brier, 'accuracy': acc,
                'n': len(trusts), 'n_failed': n_failed,
            }
            print(f"      ECE={ece:.4f}  Brier={brier:.4f}  Acc(seg)={acc:.4f}  n={len(trusts)}  failed={n_failed}")

        # ── B/C/D: CaMM variants ──
        cmm_configs = get_cmm_configs()
        for config_name, overrides in cmm_configs.items():
            cmm_out = str(dataset_dir / f'result_{config_name}.csv')
            if not args.skip_cmm:
                print(f"    [{config_name.split('_')[0]}] {config_name} ...")
                cmm_out = run_cmm(obs_cmm, cmm_out, overrides)
            if cmm_out and Path(cmm_out).exists():
                trusts, labels, n_failed = analyze_cmm_output_segments(cmm_out, gt_edges)
            else:
                trusts, labels, n_failed = [], [], 0
            if trusts:
                ece = compute_ece(trusts, labels)
                brier = compute_brier(trusts, labels)
                acc = sum(labels) / len(labels)
                cond_results[config_name] = {
                    'ece': ece, 'brier': brier, 'accuracy': acc,
                    'n': len(trusts), 'n_failed': n_failed,
                }
                print(f"      ECE={ece:.4f}  Brier={brier:.4f}  Acc(seg)={acc:.4f}  n={len(trusts)}  failed={n_failed}")

        all_results[name] = cond_results

        # Build table row
        for cfg_key, cfg_label in [
            ('A_HMM', 'HMM'), ('B_CaMM-EP', 'CaMM-EP'),
            ('C_CaMM-Dir', 'CaMM-Dir'), ('D_CaMM-Full', 'CaMM-Full')
        ]:
            r = cond_results.get(cfg_key, {})
            grand_table.append({
                'condition': name, 'sigma': sigma, 'interval': label,
                'config': cfg_label,
                'ece': r.get('ece', float('nan')),
                'brier': r.get('brier', float('nan')),
                'accuracy': r.get('accuracy', float('nan')),
                'n': r.get('n', 0), 'n_failed': r.get('n_failed', 0),
            })

    # ═══════════════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  ABLATION RESULTS — ECE per configuration")
    print(f"{'='*90}")
    header = f"  {'Condition':<22s}  {'HMM':>8s}  {'CaMM-EP':>8s}  {'CaMM-Dir':>8s}  {'CaMM-Full':>8s}"
    print(header)
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for cond in conditions:
        name = cond['name']
        r = all_results.get(name, {})
        hmm_ece = r.get('A_HMM', {}).get('ece', float('nan'))
        ep_ece = r.get('B_CaMM-EP', {}).get('ece', float('nan'))
        dir_ece = r.get('C_CaMM-Dir', {}).get('ece', float('nan'))
        full_ece = r.get('D_CaMM-Full', {}).get('ece', float('nan'))
        print(f"  {name:<22s}  {hmm_ece:8.4f}  {ep_ece:8.4f}  {dir_ece:8.4f}  {full_ece:8.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # ECE delta analysis (contributions per component)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  COMPONENT CONTRIBUTIONS (ΔECE)")
    print(f"{'='*90}")
    print(f"  {'Condition':<22s}  {'HMM→EP':>10s}  {'EP→Dir':>10s}  {'Dir→Full':>10s}  {'HMM→Full':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for cond in conditions:
        name = cond['name']
        r = all_results.get(name, {})
        hmm_e = r.get('A_HMM', {}).get('ece', float('nan'))
        ep_e = r.get('B_CaMM-EP', {}).get('ece', float('nan'))
        dir_e = r.get('C_CaMM-Dir', {}).get('ece', float('nan'))
        full_e = r.get('D_CaMM-Full', {}).get('ece', float('nan'))
        d1 = ep_e - hmm_e if not math.isnan(ep_e) and not math.isnan(hmm_e) else float('nan')
        d2 = dir_e - ep_e if not math.isnan(dir_e) and not math.isnan(ep_e) else float('nan')
        d3 = full_e - dir_e if not math.isnan(full_e) and not math.isnan(dir_e) else float('nan')
        dt = full_e - hmm_e if not math.isnan(full_e) and not math.isnan(hmm_e) else float('nan')
        print(f"  {name:<22s}  {d1:+10.4f}  {d2:+10.4f}  {d3:+10.4f}  {dt:+10.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════════
    results_json = OUTPUT_DIR / 'ablation_results.json'
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {'sigma_values': SIGMA_VALUES, 'intervals': INTERVAL_LABELS,
                       'n_trajs': args.trajs, 'n_points': args.points,
                       'threshold_m': THRESHOLD_M, 'n_bins': N_BINS},
            'results': {name: {k: v for k, v in cr.items()}
                       for name, cr in all_results.items()},
            'table': grand_table,
        }, f, indent=2)
    print(f"\nResults saved to {results_json}")

    # ═══════════════════════════════════════════════════════════════════════
    # Plot: 3×3 grid with matplotlib
    # ═══════════════════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharey=False)
        config_labels = ['HMM', 'CaMM-EP', 'CaMM-Dir', 'CaMM-Full']
        config_keys = ['A_HMM', 'B_CaMM-EP', 'C_CaMM-Dir', 'D_CaMM-Full']
        colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#d62728']
        x = np.arange(len(config_labels))
        bar_width = 0.6

        for row_idx, sigma in enumerate(SIGMA_VALUES):
            for col_idx, (rate, int_label) in enumerate(zip(SAMPLE_RATES, INTERVAL_LABELS)):
                ax = axes[row_idx, col_idx]
                name = f'sigma{sigma:.0f}_int{int_label}'
                r = all_results.get(name, {})
                ece_vals = []
                for ck in config_keys:
                    v = r.get(ck, {}).get('ece', float('nan'))
                    ece_vals.append(v if not math.isnan(v) else 0.0)

                bars = ax.bar(x, ece_vals, bar_width, color=colors, edgecolor='white', linewidth=0.5)

                # Annotate bar values
                for bar, val in zip(bars, ece_vals):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

                ax.set_xticks(x)
                ax.set_xticklabels(config_labels, fontsize=8, rotation=15)
                ax.set_title(f'σ={sigma:.0f}m, Δt={int_label}', fontsize=11, fontweight='bold')
                ax.set_ylabel('ECE', fontsize=10)
                ax.set_ylim(0, max(max(ece_vals) * 1.25, 0.1))
                ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Component-wise Ablation: ECE Calibration Error\n'
                     'HMM → +Covariance EP → +Direction → +Temperature',
                     fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout()

        plot_path = OUTPUT_DIR / 'ablation_ece_3x3.png'
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available; skipping plot generation.")


if __name__ == '__main__':
    main()
