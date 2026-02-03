#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import re
import os

def parse_wkt_point(wkt):
    if pd.isna(wkt) or not isinstance(wkt, str): return None
    match = re.search(r"POINT\s*\(([-\d.]+)\s+([-\d.]+)\)", wkt, re.IGNORECASE)
    if match: return float(match.group(1)), float(match.group(2))
    return None

def calc_dist_m(lon1, lat1, lon2, lat2):
    avg_lat = np.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * np.cos(avg_lat)
    dy = (lat1 - lat2) * 111320.0
    return np.sqrt(dx**2 + dy**2)

def main():
    print("Loading datasets...")
    df_cmm = pd.read_csv("dataset-hainan-06/mr/cmm_results_filtered.csv", sep=';')
    df_fmm = pd.read_csv("dataset-hainan-06/mr/fmm_results_filtered.csv", sep=';')
    df_input = pd.read_csv("dataset-hainan-06/cmm_input_points.csv", sep=';')

    # Merge PL for CMM
    input_meta = df_input[['timestamp', 'protection_level']].drop_duplicates('timestamp')
    df_cmm = pd.merge(df_cmm, input_meta, on='timestamp', how='left')
    df_cmm['pl_m'] = df_cmm['protection_level'] * 111320.0

    print("Processing CMM errors...")
    cmm_data = []
    for _, row in df_cmm.iterrows():
        truth = parse_wkt_point(row['ogeom'])
        result = parse_wkt_point(row['pgeom'])
        if truth and result:
            err = calc_dist_m(truth[0], truth[1], result[0], result[1])
            cmm_data.append({
                'timestamp': row['timestamp'],
                'error': err,
                'pl_m': row['pl_m'],
                'ep': row['ep'],
                'trustworthiness': row['trustworthiness']
            })
    df_cmm_p = pd.DataFrame(cmm_data)

    print("Processing FMM errors...")
    input_map = {row['timestamp']: (row['x'], row['y']) for _, row in df_input.iterrows()}
    fmm_data = []
    for _, row in df_fmm.iterrows():
        ts = row['timestamp']
        if ts in input_map:
            truth = input_map[ts]
            result = parse_wkt_point(row['pgeom'])
            if result:
                err = calc_dist_m(truth[0], truth[1], result[0], result[1])
                fmm_data.append({'error': err})
    df_fmm_p = pd.DataFrame(fmm_data)

    # REASONABLE FILTER: Physical Constraint (Error < 2.0 * PL)
    print("Applying Reasonable Filter (Physical Integrity)...")
    df_cmm_filtered = df_cmm_p[df_cmm_p['error'] <= (df_cmm_p['pl_m'] * 2.0)].copy()

    # Plot Distribution
    plt.figure(figsize=(12, 6))
    bins = np.linspace(0, 30, 60)
    plt.hist(df_fmm_p['error'], bins=bins, alpha=0.4, label='FMM Error', density=True, color='green')
    plt.hist(df_cmm_p['error'], bins=bins, alpha=0.2, label='CMM Raw Error', density=True, color='blue')
    plt.hist(df_cmm_filtered['error'], bins=bins, alpha=0.7, label='CMM Filtered (Integrity)', density=True, color='red')
    plt.title('Error Distribution Comparison (Zoomed 0-30m)')
    plt.xlabel('Error (meters)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs("output", exist_ok=True)
    plt.savefig('output/final_error_dist.png')

    # ROC Curve
    print("Generating ROC curve...")
    df_cmm_p['is_correct'] = (df_cmm_p['error'] < 10).astype(int)
    # Combining trustworthiness and ep for a single ranking score
    # Primary: trustworthiness, Secondary: ep
    df_cmm_p = df_cmm_p.sort_values(by=['trustworthiness', 'ep'], ascending=[True, True])
    df_cmm_p['score_rank'] = np.arange(len(df_cmm_p))
    
    fpr, tpr, _ = roc_curve(df_cmm_p['is_correct'], df_cmm_p['score_rank'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CMM ROC Curve (Correctness: Error < 10m)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('output/final_cmm_roc.png')

    print("\n--- Final Statistics ---")
    print(f"FMM Mean Error: {df_fmm_p['error'].mean():.2f}m")
    print(f"CMM Raw Mean Error: {df_cmm_p['error'].mean():.2f}m")
    print(f"CMM Filtered Mean Error: {df_cmm_filtered['error'].mean():.2f}m")
    print(f"CMM Filtered Count: {len(df_cmm_filtered)} / {len(df_cmm_p)}")

if __name__ == "__main__":
    main()
