#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    print("Loading data...")
    df_cmm = pd.read_csv("dataset-hainan-06/mr/cmm_results_filtered.csv", sep=';')
    df_input = pd.read_csv("dataset-hainan-06/cmm_input_points.csv", sep=';')
    
    # Merge PL from input points
    input_pl = df_input[['timestamp', 'protection_level']].drop_duplicates('timestamp')
    df = pd.merge(df_cmm, input_pl, on='timestamp', how='left')
    # Convert PL from degrees to meters
    df['protection_level_m'] = df['protection_level'] * 111320.0

    print("Calculating base errors...")
    errors = []
    for _, row in df.iterrows():
        truth = parse_wkt_point(row['ogeom'])
        result = parse_wkt_point(row['pgeom'])
        err = calc_dist_m(truth[0], truth[1], result[0], result[1]) if truth and result else np.nan
        errors.append(err)
    df['error_m'] = errors
    df = df.dropna(subset=['error_m']).copy()

    # Calculate Smoothness using matched points (pgeom)
    # distance(pgeom[i], pgeom[i-1]) / distance(ogeom[i], ogeom[i-1])
    print("Calculating path smoothness...")
    pgeom_coords = [parse_wkt_point(g) for g in df['pgeom']]
    ogeom_coords = [parse_wkt_point(g) for g in df['ogeom']]
    
    smooth_ratios = [1.0] # First point default
    for i in range(1, len(df)):
        p1, p2 = pgeom_coords[i-1], pgeom_coords[i]
        o1, o2 = ogeom_coords[i-1], ogeom_coords[i]
        
        d_matched = calc_dist_m(p1[0], p1[1], p2[0], p2[1])
        d_obs = calc_dist_m(o1[0], o1[1], o2[0], o2[1])
        
        ratio = d_matched / (d_obs + 0.1) if d_obs > 0.1 else 1.0
        smooth_ratios.append(ratio)
    
    df['smooth_ratio'] = smooth_ratios

    # Define filtering strategies
    
    # 1. Original simple filter
    df_orig = df[(df['ep'] >= 0.5) & (df['trustworthiness'] >= 0.999)]
    
    # 2. Strategy A: Physical Consistency (Error < PL * factor)
    # PL is approx 4.4m (0.00004 deg). We use a conservative factor.
    df_strat_a = df[df['error_m'] <= (df['protection_level_m'] * 2.0)]
    
    # 3. Strategy B: Path Smoothness (Matched dist shouldn't be much larger than Obs dist)
    df_strat_b = df[df['smooth_ratio'] < 1.5]
    
    # 4. Strategy C: Refined Composite (Physical + Smooth + Trust)
    df_strat_c = df[
        (df['error_m'] <= (df['protection_level_m'] * 3.0)) & 
        (df['smooth_ratio'] < 2.0) & 
        (df['trustworthiness'] > 0.95)
    ]

    # Evaluation
    results = [
        ("Raw CMM", df),
        ("Simple Filter (Old)", df_orig),
        ("Physical (Error < 1.5*PL)", df_strat_a),
        ("Smoothness (SP/EU < 1.5)", df_strat_b),
        ("Composite (Trust + Smooth)", df_strat_c)
    ]

    print("\n" + "="*60)
    print(f"{ 'Strategy':<30} | {'Count':<10} | {'Mean Error':<10}")
    print("-" * 60)
    for name, d in results:
        print(f"{name:<30} | {len(d):<10} | {d['error_m'].mean():.2f}m")
    print("="*60)

    # Plot Comparison
    plt.figure(figsize=(12, 7))
    bins = np.linspace(0, 100, 100)
    for name, d in results:
        plt.hist(d['error_m'], bins=bins, alpha=0.5, label=name, density=True, histtype='step', lw=2)
    
    plt.title('CMM Filtering Strategies Comparison (Error Distribution)')
    plt.xlabel('Error (meters)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/cmm_filter_comparison.png')
    print("\nComparison plot saved to output/cmm_filter_comparison.png")

if __name__ == "__main__":
    main()
