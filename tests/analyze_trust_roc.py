#!/usr/bin/env python3
"""
Analyze map matching results and plot ROC curve based on trustworthiness.

Compares CMM and FMM results against ground truth GPS coordinates.
A match is considered correct if the distance to ground truth is < 10m.

NOTE:
- CMM outputs 'pgeom' which is the MATCHED point on the road network
- FMM outputs 'pgeom' which is the ORIGINAL GPS observation point (not matched)
  Therefore, FMM accuracy analysis shows GPS error, not matching error
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os

# Path configuration
BASE_DIR = "/home/dell/fmm_sjtugnc/dataset-hainan-06"
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "cmm_input_points.csv")
CMM_RESULTS_FILE = os.path.join(BASE_DIR, "mr", "cmm_results_trust.csv")
FMM_RESULTS_FILE = os.path.join(BASE_DIR, "mr", "fmm_results_filtered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "mr")

# Distance threshold for correct match (meters)
DISTANCE_THRESHOLD = 10.0


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
        return np.nan

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in meters
    r = 6371000
    return c * r


def parse_point(point_str):
    """Parse WKT POINT string to (lon, lat) tuple."""
    if pd.isna(point_str) or point_str == '':
        return None, None
    # Format: POINT(lon lat)
    point_str = point_str.strip()
    if point_str.startswith('POINT('):
        coords = point_str[6:-1].split()
        return float(coords[0]), float(coords[1])
    return None, None


def calculate_distance(lon1, lat1, lon2, lat2):
    """Calculate geodesic distance in meters between two points."""
    return haversine_distance(lon1, lat1, lon2, lat2)


def load_ground_truth(filepath):
    """Load ground truth GPS coordinates."""
    df = pd.read_csv(filepath, sep=';')
    # Create sequential index for each trajectory
    df['seq'] = df.groupby('id').cumcount()
    # Create a composite key for matching
    df['key'] = df['id'].astype(str) + '_' + df['seq'].astype(str)
    return df


def load_cmm_results(filepath):
    """Load CMM results and extract matched points with trustworthiness."""
    df = pd.read_csv(filepath, sep=';')

    # Parse original and matched points
    df[['ogem_lon', 'ogem_lat']] = df['ogeom'].apply(
        lambda x: pd.Series(parse_point(x))
    )
    df[['pgeom_lon', 'pgeom_lat']] = df['pgeom'].apply(
        lambda x: pd.Series(parse_point(x))
    )

    # Create key for matching with ground truth using (id, seq)
    df['key'] = df['id'].astype(str) + '_' + df['seq'].astype(str)

    return df


def load_fmm_results(filepath):
    """
    Load FMM results.
    WARNING: FMM output 'pgeom' is the ORIGINAL GPS observation point, NOT the matched point.
    FMM does not output matched point geometry (mgeom) like CMM does.
    """
    df = pd.read_csv(filepath, sep=';')

    # Parse point geometry (this is the original GPS observation point)
    df[['pgeom_lon', 'pgeom_lat']] = df['pgeom'].apply(
        lambda x: pd.Series(parse_point(x))
    )

    # Create key for matching with ground truth using (id, seq)
    df['key'] = df['id'].astype(str) + '_' + df['seq'].astype(str)

    return df


def analyze_cmm_accuracy(df_results, df_gt):
    """Analyze CMM matching accuracy against ground truth."""
    # Merge with ground truth
    merged = df_results.merge(df_gt[['key', 'x', 'y']], on='key', how='inner')

    # Calculate distance from MATCHED point to ground truth
    merged['distance'] = merged.apply(
        lambda row: calculate_distance(
            row['pgeom_lon'], row['pgeom_lat'],
            row['x'], row['y']
        ), axis=1
    )

    # Remove NaN distances
    merged = merged.dropna(subset=['distance'])

    # Calculate accuracy metrics
    is_correct = merged['distance'] < DISTANCE_THRESHOLD
    accuracy = is_correct.mean() * 100

    # Statistics
    mean_error = merged['distance'].mean()
    median_error = merged['distance'].median()
    std_error = merged['distance'].std()

    print(f"\n{'='*60}")
    print(f"CMM Matching Accuracy Analysis")
    print(f"(Comparing MATCHED points to ground truth GPS)")
    print(f"{'='*60}")
    print(f"Total points: {len(merged)}")
    print(f"Accuracy (< {DISTANCE_THRESHOLD}m): {accuracy:.2f}%")
    print(f"Mean error: {mean_error:.2f}m")
    print(f"Median error: {median_error:.2f}m")
    print(f"Std error: {std_error:.2f}m")
    print(f"Min error: {merged['distance'].min():.2f}m")
    print(f"Max error: {merged['distance'].max():.2f}m")

    # Error distribution
    print(f"\nError distribution:")
    print(f"  < 5m:  {(merged['distance'] < 5).sum()} ({(merged['distance'] < 5).mean()*100:.1f}%)")
    print(f"  5-10m: {((merged['distance'] >= 5) & (merged['distance'] < 10)).sum()} ({((merged['distance'] >= 5) & (merged['distance'] < 10)).mean()*100:.1f}%)")
    print(f"  10-20m: {((merged['distance'] >= 10) & (merged['distance'] < 20)).sum()} ({((merged['distance'] >= 10) & (merged['distance'] < 20)).mean()*100:.1f}%)")
    print(f"  20-50m: {((merged['distance'] >= 20) & (merged['distance'] < 50)).sum()} ({((merged['distance'] >= 20) & (merged['distance'] < 50)).mean()*100:.1f}%)")
    print(f"  > 50m:  {(merged['distance'] >= 50).sum()} ({(merged['distance'] >= 50).mean()*100:.1f}%)")

    # Trustworthiness statistics
    if 'trustworthiness' in merged.columns:
        print(f"\nTrustworthiness statistics:")
        print(f"  Mean: {merged['trustworthiness'].mean():.4f}")
        print(f"  Median: {merged['trustworthiness'].median():.4f}")
        print(f"  Min: {merged['trustworthiness'].min():.4f}")
        print(f"  Max: {merged['trustworthiness'].max():.4f}")

        # Trustworthiness vs accuracy
        for thresh in [0.5, 0.7, 0.9, 0.95, 0.99]:
            subset = merged[merged['trustworthiness'] >= thresh]
            if len(subset) > 0:
                acc = (subset['distance'] < DISTANCE_THRESHOLD).mean() * 100
                print(f"  Accuracy at trust >= {thresh}: {acc:.2f}% (n={len(subset)})")

    return merged, is_correct


def analyze_fmm_data(df_results, df_gt):
    """
    Analyze FMM results.
    NOTE: FMM does not output matched point geometry, only original GPS points.
    This shows GPS observation error, NOT matching error.
    """
    # Merge with ground truth
    merged = df_results.merge(df_gt[['key', 'x', 'y']], on='key', how='inner')

    # Calculate distance from ORIGINAL point to ground truth
    merged['distance'] = merged.apply(
        lambda row: calculate_distance(
            row['pgeom_lon'], row['pgeom_lat'],
            row['x'], row['y']
        ), axis=1
    )

    # Remove NaN distances
    merged = merged.dropna(subset=['distance'])

    print(f"\n{'='*60}")
    print(f"FMM Data Analysis")
    print(f"WARNING: FMM outputs original GPS points, NOT matched points!")
    print(f"This shows GPS error, NOT map-matching accuracy.")
    print(f"{'='*60}")
    print(f"Total points: {len(merged)}")

    if len(merged) > 0:
        mean_error = merged['distance'].mean()
        median_error = merged['distance'].median()
        std_error = merged['distance'].std()

        print(f"Mean GPS error: {mean_error:.2f}m")
        print(f"Median GPS error: {median_error:.2f}m")
        print(f"Std GPS error: {std_error:.2f}m")
        print(f"Min GPS error: {merged['distance'].min():.2f}m")
        print(f"Max GPS error: {merged['distance'].max():.2f}m")

        # Error distribution
        print(f"\nGPS Error distribution:")
        print(f"  < 5m:  {(merged['distance'] < 5).sum()} ({(merged['distance'] < 5).mean()*100:.1f}%)")
        print(f"  5-10m: {((merged['distance'] >= 5) & (merged['distance'] < 10)).sum()} ({((merged['distance'] >= 5) & (merged['distance'] < 10)).mean()*100:.1f}%)")
        print(f"  10-20m: {((merged['distance'] >= 10) & (merged['distance'] < 20)).sum()} ({((merged['distance'] >= 10) & (merged['distance'] < 20)).mean()*100:.1f}%)")
        print(f"  > 20m:  {(merged['distance'] >= 20).sum()} ({(merged['distance'] >= 20).mean()*100:.1f}%)")

        # Trustworthiness statistics
        if 'trustworthiness' in merged.columns:
            print(f"\nTrustworthiness statistics:")
            print(f"  Mean: {merged['trustworthiness'].mean():.4f}")
            print(f"  Median: {merged['trustworthiness'].median():.4f}")
            print(f"  Min: {merged['trustworthiness'].min():.4f}")
            print(f"  Max: {merged['trustworthiness'].max():.4f}")

    return merged


def plot_roc_curve(merged_cmm, is_correct_cmm, merged_fmm):
    """
    Plot ROC curve using trustworthiness as classifier.
    Only CMM can be properly evaluated since FMM lacks matched points.
    """
    plt.figure(figsize=(10, 8))

    # CMM ROC curve
    if 'trustworthiness' in merged_cmm.columns and len(merged_cmm) > 0:
        fpr_cmm, tpr_cmm, thresholds_cmm = roc_curve(is_correct_cmm, merged_cmm['trustworthiness'])
        roc_auc_cmm = auc(fpr_cmm, tpr_cmm)

        plt.plot(fpr_cmm, tpr_cmm, 'b-', linewidth=2,
                 label=f'CMM (AUC = {roc_auc_cmm:.3f})')

        # Find optimal threshold (Youden's J statistic)
        youden_j = tpr_cmm - fpr_cmm
        optimal_idx = np.argmax(youden_j)
        optimal_threshold_cmm = thresholds_cmm[optimal_idx]
        print(f"\nCMM Optimal trustworthiness threshold: {optimal_threshold_cmm:.4f}")
        print(f"  at TPR={tpr_cmm[optimal_idx]:.3f}, FPR={fpr_cmm[optimal_idx]:.3f}")

        # Print accuracy at various thresholds
        print(f"\nCMM Accuracy at different trustworthiness thresholds:")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
            subset = merged_cmm[merged_cmm['trustworthiness'] >= thresh]
            if len(subset) > 0:
                acc = (subset['distance'] < DISTANCE_THRESHOLD).mean() * 100
                print(f"  trust >= {thresh:.2f}: {acc:.2f}% (n={len(subset)})")

    else:
        print("Warning: No CMM data for ROC curve")

    # FMM note - cannot plot proper ROC without ground truth matching
    if merged_fmm is not None and 'trustworthiness' in merged_fmm.columns:
        print(f"\nFMM: Cannot plot ROC curve - FMM output does not contain matched points")
        print(f"      (only original GPS points, so ground truth matching cannot be evaluated)")

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve: Trustworthiness as Match Quality Classifier\n(Ground Truth: Distance < {DISTANCE_THRESHOLD}m)',
              fontsize=13)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'trust_roc_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nROC curve saved to: {output_path}")
    plt.close()


def plot_error_distribution(merged_cmm):
    """Plot CMM error distribution histogram."""
    if 'distance' not in merged_cmm.columns or len(merged_cmm) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error distribution histogram
    axes[0].hist(merged_cmm['distance'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(DISTANCE_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold ({DISTANCE_THRESHOLD}m)')
    axes[0].axvline(merged_cmm['distance'].median(), color='green', linestyle='-', linewidth=2,
                    label=f'Median ({merged_cmm["distance"].median():.1f}m)')
    axes[0].set_xlabel('Error (m)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('CMM Error Distribution', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CDF of error
    sorted_errors = np.sort(merged_cmm['distance'])
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cdf, 'b-', linewidth=2)
    axes[1].axvline(DISTANCE_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold ({DISTANCE_THRESHOLD}m)')
    axes[1].axhline(0.9, color='gray', linestyle=':', linewidth=1,
                    label='90th percentile')
    axes[1].set_xlabel('Error (m)', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title('CMM Error CDF', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'cmm_error_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Error distribution plot saved to: {output_path}")
    plt.close()


def plot_trust_vs_error(merged_cmm):
    """Plot trustworthiness vs error scatter plot."""
    if 'distance' not in merged_cmm.columns or 'trustworthiness' not in merged_cmm.columns:
        return

    plt.figure(figsize=(10, 6))

    plt.scatter(merged_cmm['trustworthiness'], merged_cmm['distance'],
                alpha=0.3, s=10, c=merged_cmm['distance'],
                cmap='RdYlGn_r', vmin=0, vmax=30)

    plt.axhline(DISTANCE_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Accuracy Threshold ({DISTANCE_THRESHOLD}m)')
    plt.xlabel('Trustworthiness', fontsize=12)
    plt.ylabel('Error (m)', fontsize=12)
    plt.title('CMM: Trustworthiness vs Matching Error', fontsize=13)
    plt.colorbar(label='Error (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, min(50, merged_cmm['distance'].max() * 1.1)])

    output_path = os.path.join(OUTPUT_DIR, 'trust_vs_error.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Trust vs Error plot saved to: {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    print("="*60)
    print("Map Matching Error Analysis and ROC Curve")
    print("="*60)
    print(f"\nDistance threshold for correct match: {DISTANCE_THRESHOLD}m")

    # Load data
    print("\nLoading data...")
    df_gt = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"  Ground truth: {len(df_gt)} points")

    df_cmm = load_cmm_results(CMM_RESULTS_FILE)
    print(f"  CMM results: {len(df_cmm)} points")

    df_fmm = load_fmm_results(FMM_RESULTS_FILE)
    print(f"  FMM results: {len(df_fmm)} points")

    # Analyze CMM
    merged_cmm, is_correct_cmm = analyze_cmm_accuracy(df_cmm, df_gt)

    # Analyze FMM (limited due to no matched points)
    merged_fmm = analyze_fmm_data(df_fmm, df_gt)

    # Plot ROC curve
    print("\n" + "="*60)
    print("Generating ROC Curve...")
    print("="*60)
    plot_roc_curve(merged_cmm, is_correct_cmm, merged_fmm)

    # Plot error distributions
    print("\n" + "="*60)
    print("Generating CMM Error Distribution Plots...")
    print("="*60)
    plot_error_distribution(merged_cmm)

    # Plot trust vs error
    print("\n" + "="*60)
    print("Generating Trust vs Error Plot...")
    print("="*60)
    plot_trust_vs_error(merged_cmm)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
