#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import re
import os

def parse_wkt_point(wkt):
    if pd.isna(wkt) or not isinstance(wkt, str):
        return None
    match = re.search(r"POINT\s*\(([-\d.]+)\s+([-\d.]+)\)", wkt, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

def calc_dist_m(lon1, lat1, lon2, lat2):
    """
    Calculate distance in meters between two lon/lat points using local approximation.
    1 degree lat approx 111320m
    1 degree lon approx 111320m * cos(lat)
    """
    avg_lat = np.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * np.cos(avg_lat)
    dy = (lat1 - lat2) * 111320.0
    return np.sqrt(dx**2 + dy**2)

def main():
    # File paths
    cmm_res_path = "dataset-hainan-06/mr/cmm_results_filtered.csv"
    fmm_res_path = "dataset-hainan-06/mr/fmm_results_filtered.csv"
    input_pts_path = "dataset-hainan-06/cmm_input_points.csv"

    print("Loading datasets...")
    # Read datasets
    # CMM results use semicolon delimiter
    df_cmm = pd.read_csv(cmm_res_path, sep=';')
    # FMM results also use semicolon delimiter
    df_fmm = pd.read_csv(fmm_res_path, sep=';')
    # Input points use semicolon delimiter
    df_input = pd.read_csv(input_pts_path, sep=';')

    # 1. Process CMM Error
    # Ground truth: ogeom, Result: pgeom
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
                'ep': row['ep'],
                'trustworthiness': row['trustworthiness']
            })
    df_cmm_processed = pd.DataFrame(cmm_data)

    # 2. Process FMM Error
    # Match FMM pgeom with Input Points (x, y) by timestamp
    print("Processing FMM errors...")
    # Create a mapping for input points
    input_map = {row['timestamp']: (row['x'], row['y']) for _, row in df_input.iterrows()}
    
    fmm_data = []
    for _, row in df_fmm.iterrows():
        ts = row['timestamp']
        if ts in input_map:
            truth = input_map[ts]
            result = parse_wkt_point(row['pgeom'])
            if result:
                err = calc_dist_m(truth[0], truth[1], result[0], result[1])
                fmm_data.append({'timestamp': ts, 'error': err})
    df_fmm_processed = pd.DataFrame(fmm_data)

    # 3. Filtering CMM
    print("Filtering CMM results (ep >= 0.5, trustworthiness >= 0.999)...")
    df_cmm_filtered = df_cmm_processed[
        (df_cmm_processed['ep'] >= 0.5) & 
        (df_cmm_processed['trustworthiness'] >= 0.999)
    ].copy()

    # 4. Plotting Error Distributions
    print("Plotting error distributions...")
    plt.figure(figsize=(12, 6))
    
    # Histogram parameters
    bins = np.linspace(0, 50, 50)
    
    plt.hist(df_fmm_processed['error'], bins=bins, alpha=0.5, label='FMM Error', density=True, color='green')
    plt.hist(df_cmm_processed['error'], bins=bins, alpha=0.3, label='CMM Raw Error', density=True, color='blue')
    plt.hist(df_cmm_filtered['error'], bins=bins, alpha=0.7, label='CMM Filtered Error', density=True, color='orange')
    
    plt.title('Error Distribution Comparison (0-50m)')
    plt.xlabel('Error (meters)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs("output", exist_ok=True)
    plt.savefig('output/error_distribution.png')
    print("Saved distribution plot to output/error_distribution.png")

    # 5. ROC Curve for CMM
    # Correct if error < 10m
    # Sort by trustworthiness (primary) and ep (secondary)
    print("Generating ROC curve for CMM...")
    df_cmm_processed['is_correct'] = (df_cmm_processed['error'] < 10).astype(int)
    
    # Use lexicographical sort keys for ROC
    # Higher is better for scores
    # We combine them into a single score for the roc_curve function
    # Note: trustworthiness is primary, ep is secondary.
    # Since we need a score, we can rank them.
    df_cmm_processed = df_cmm_processed.sort_values(by=['trustworthiness', 'ep'], ascending=[True, True])
    df_cmm_processed['score_rank'] = np.arange(len(df_cmm_processed))
    
    fpr, tpr, thresholds = roc_curve(df_cmm_processed['is_correct'], df_cmm_processed['score_rank'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Incorrect Matched Points > Threshold)')
    plt.ylabel('True Positive Rate (Correct Matched Points > Threshold)')
    plt.title('CMM ROC Curve (Correctness: Error < 10m)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('output/cmm_roc_curve.png')
    print("Saved ROC curve to output/cmm_roc_curve.png")

    # Summary Stats
    print("\n--- Summary Statistics ---")
    print(f"FMM Mean Error: {df_fmm_processed['error'].mean():.2f}m")
    print(f"CMM Raw Mean Error: {df_cmm_processed['error'].mean():.2f}m")
    print(f"CMM Filtered Mean Error: {df_cmm_filtered['error'].mean():.2f}m")
    print(f"CMM Sample Count: {len(df_cmm_processed)}")
    print(f"CMM Filtered Sample Count: {len(df_cmm_filtered)} ({len(df_cmm_filtered)/len(df_cmm_processed)*100:.1f}%)")

if __name__ == "__main__":
    main()
