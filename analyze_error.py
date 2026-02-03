#!/usr/bin/env python3
"""
Error analysis for FMM and CMM results.

Analyzes:
1. FMM errors: pgeom vs ground truth (from cmm_input_points.csv)
2. CMM errors: pgeom vs ogeom
3. Filtered CMM errors (ep >= 0.5, trustworthiness >= 0.999)
4. Error distribution plots
5. ROC curve for CMM (using trustworthiness and ep)
"""

import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Point:
    """Represent a point with coordinates."""
    x: float  # longitude
    y: float  # latitude

    def to_tuple(self):
        return (self.x, self.y)


def parse_point(geom_str: str) -> Point:
    """Parse POINT(x y) from WKT."""
    if not geom_str or not geom_str.strip():
        raise ValueError(f"Empty geometry string")
    # Remove "POINT(" and ")" and split
    match = re.search(r'POINT\(([^ ]+) ([^)]+)\)', geom_str)
    if match:
        return Point(float(match.group(1)), float(match.group(2)))
    raise ValueError(f"Cannot parse POINT from: {geom_str}")


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def point_distance(p1: Point, p2: Point) -> float:
    """Calculate distance between two points in meters."""
    return haversine_distance(p1.x, p1.y, p2.x, p2.y)


def load_cmm_results(filepath: Path) -> List[Dict]:
    """Load CMM results from CSV."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                ogeom = parse_point(row['ogeom'])
                pgeom = parse_point(row['pgeom'])
                results.append({
                    'id': row['id'],
                    'seq': int(row['seq']),
                    'timestamp': row['timestamp'],
                    'ogeom': ogeom,
                    'pgeom': pgeom,
                    'ep': float(row['ep']),
                    'trustworthiness': float(row['trustworthiness']),
                })
            except (ValueError, KeyError) as e:
                # Skip rows with empty or invalid geometries
                continue
    return results


def load_fmm_results(filepath: Path) -> List[Dict]:
    """Load FMM results from CSV."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                pgeom = parse_point(row['pgeom'])
                results.append({
                    'id': row['id'],
                    'seq': int(row['seq']),
                    'timestamp': row['timestamp'],
                    'pgeom': pgeom,
                })
            except (ValueError, KeyError) as e:
                # Skip rows with empty or invalid geometries
                continue
    return results


def load_ground_truth_points(filepath: Path) -> Dict[str, Dict[int, Point]]:
    """Load ground truth points from CMM input CSV.

    Returns: {traj_id: {timestamp: Point}}
    """
    gt_points = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            traj_id = row['id']
            timestamp = row['timestamp']
            # Parse timestamp - may be float or int
            try:
                ts = int(float(timestamp))
            except ValueError:
                ts = timestamp

            point = Point(float(row['x']), float(row['y']))

            if traj_id not in gt_points:
                gt_points[traj_id] = {}
            gt_points[traj_id][ts] = point

    return gt_points


def match_fmm_to_ground_truth(fmm_results: List[Dict], gt_points: Dict) -> List[Tuple[Point, Point]]:
    """Match FMM results to ground truth points by timestamp.

    Returns: List of (ground_truth, result) tuples
    """
    matched_pairs = []

    for result in fmm_results:
        traj_id = result['id']
        timestamp = int(result['timestamp'])

        if traj_id in gt_points and timestamp in gt_points[traj_id]:
            gt_point = gt_points[traj_id][timestamp]
            result_point = result['pgeom']
            matched_pairs.append((gt_point, result_point))

    return matched_pairs


def calculate_errors(pairs: List[Tuple[Point, Point]]) -> np.ndarray:
    """Calculate errors (distances) for matched pairs."""
    errors = []
    for gt_point, result_point in pairs:
        error = point_distance(gt_point, result_point)
        errors.append(error)
    return np.array(errors)


def print_statistics(name: str, errors: np.ndarray):
    """Print error statistics."""
    print(f"\n{'='*60}")
    print(f"{name} Error Statistics")
    print(f"{'='*60}")
    print(f"Total points: {len(errors)}")
    print(f"Mean error: {np.mean(errors):.2f} m")
    print(f"Median error: {np.median(errors):.2f} m")
    print(f"Std deviation: {np.std(errors):.2f} m")
    print(f"Min error: {np.min(errors):.2f} m")
    print(f"Max error: {np.max(errors):.2f} m")

    # Percentiles
    for p in [50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(errors, p):.2f} m")

    # Percentage below threshold
    for threshold in [5, 10, 20, 50]:
        count = np.sum(errors < threshold)
        pct = 100 * count / len(errors)
        print(f"Error < {threshold}m: {count}/{len(errors)} ({pct:.1f}%)")


def plot_error_distributions(fmm_errors: np.ndarray, cmm_errors: np.ndarray,
                             cmm_filtered_errors: np.ndarray, output_path: Path):
    """Plot error distributions for FMM and CMM."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram
    ax = axes[0, 0]
    bins = np.logspace(np.log10(0.1), np.log10(1000), 50)
    ax.hist(fmm_errors, bins=bins, alpha=0.6, label='FMM', color='blue', density=True)
    ax.hist(cmm_errors, bins=bins, alpha=0.6, label='CMM (all)', color='green', density=True)
    ax.hist(cmm_filtered_errors, bins=bins, alpha=0.6, label='CMM (filtered)', color='red', density=True)
    ax.set_xscale('log')
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative distribution
    ax = axes[0, 1]
    sorted_fmm = np.sort(fmm_errors)
    sorted_cmm = np.sort(cmm_errors)
    sorted_cmm_filt = np.sort(cmm_filtered_errors)

    ax.plot(sorted_fmm, np.arange(1, len(sorted_fmm) + 1) / len(sorted_fmm),
            label='FMM', color='blue', linewidth=2)
    ax.plot(sorted_cmm, np.arange(1, len(sorted_cmm) + 1) / len(sorted_cmm),
            label='CMM (all)', color='green', linewidth=2)
    ax.plot(sorted_cmm_filt, np.arange(1, len(sorted_cmm_filt) + 1) / len(sorted_cmm_filt),
            label='CMM (filtered)', color='red', linewidth=2)
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 3. Box plot
    ax = axes[1, 0]
    data_to_plot = [
        np.log10(fmm_errors + 0.001),  # Log scale for box plot
        np.log10(cmm_errors + 0.001),
        np.log10(cmm_filtered_errors + 0.001)
    ]
    bp = ax.boxplot(data_to_plot, tick_labels=['FMM', 'CMM\n(all)', 'CMM\n(filtered)'], patch_artist=True)
    colors = ['blue', 'green', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Error (log10 m)')
    ax.set_title('Error Distribution (Box Plot)')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    stats_data = [
        ['Method', 'Mean', 'Median', '90th %ile', '95th %ile', '<10m'],
        ['FMM', f'{np.mean(fmm_errors):.1f}', f'{np.median(fmm_errors):.1f}',
         f'{np.percentile(fmm_errors, 90):.1f}', f'{np.percentile(fmm_errors, 95):.1f}',
         f'{100*np.sum(fmm_errors<10)/len(fmm_errors):.1f}%'],
        ['CMM (all)', f'{np.mean(cmm_errors):.1f}', f'{np.median(cmm_errors):.1f}',
         f'{np.percentile(cmm_errors, 90):.1f}', f'{np.percentile(cmm_errors, 95):.1f}',
         f'{100*np.sum(cmm_errors<10)/len(cmm_errors):.1f}%'],
        ['CMM (filtered)', f'{np.mean(cmm_filtered_errors):.1f}', f'{np.median(cmm_filtered_errors):.1f}',
         f'{np.percentile(cmm_filtered_errors, 90):.1f}', f'{np.percentile(cmm_filtered_errors, 95):.1f}',
         f'{100*np.sum(cmm_filtered_errors<10)/len(cmm_filtered_errors):.1f}%'],
    ]
    table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nError distribution plot saved to: {output_path}")


def plot_roc_curve(cmm_results: List[Dict], output_path: Path, threshold_m: float = 10.0):
    """Plot ROC curve for CMM using trustworthiness and ep as ranking criteria.

    Correct is defined as error < threshold_m.
    """
    # Calculate errors for all CMM results
    errors_with_metrics = []
    for result in cmm_results:
        gt_point = result['ogeom']
        result_point = result['pgeom']
        error = point_distance(gt_point, result_point)
        is_correct = error < threshold_m

        errors_with_metrics.append({
            'error': error,
            'is_correct': is_correct,
            'trustworthiness': result['trustworthiness'],
            'ep': result['ep'],
        })

    # Sort by trustworthiness (desc, primary), then ep (desc, secondary)
    # High trustworthiness = more confident
    # High ep = higher emission probability
    sorted_results = sorted(errors_with_metrics,
                           key=lambda x: (-x['trustworthiness'], -x['ep']))

    # Calculate ROC metrics
    n_total = len(sorted_results)
    n_correct = sum(1 for r in sorted_results if r['is_correct'])

    if n_correct == 0:
        print("\nWarning: No correct samples found for ROC analysis")
        return

    # Calculate TPR and FPR at different thresholds
    tpr_list = []  # True Positive Rate (Recall)
    fpr_list = []  # False Positive Rate
    thresholds = []

    for i in range(n_total + 1):
        # Consider top-i samples as positive predictions
        if i == 0:
            tp = 0
            fp = 0
        else:
            top_i = sorted_results[:i]
            tp = sum(1 for r in top_i if r['is_correct'])
            fp = sum(1 for r in top_i if not r['is_correct'])

        tpr = tp / n_correct if n_correct > 0 else 0
        # False positives: predicted as correct but actually incorrect
        # Total incorrect samples
        n_incorrect = n_total - n_correct
        fpr = fp / n_incorrect if n_incorrect > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        thresholds.append(i)

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(fpr_list, tpr_list, 'b-', linewidth=2, label='CMM ROC Curve')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')

    # Fill area under curve
    ax.fill_between(fpr_list, tpr_list, alpha=0.3)

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
    ax.set_title(f'CMM ROC Curve (Correct = Error < {threshold_m}m)\nSorted by Trustworthiness → Emission Probability',
                fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Calculate AUC
    auc = np.trapezoid(tpr_list, fpr_list)
    ax.text(0.6, 0.2, f'AUC = {auc:.4f}', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved to: {output_path}")
    print(f"AUC: {auc:.4f}")

    # Print some statistics
    print(f"\nROC Analysis (threshold = {threshold_m}m):")
    print(f"Total samples: {n_total}")
    print(f"Correct samples: {n_correct} ({100*n_correct/n_total:.1f}%)")
    print(f"Incorrect samples: {n_total - n_correct} ({100*(n_total-n_correct)/n_total:.1f}%)")


def main():
    # File paths
    cmm_results_path = Path('dataset-hainan-06/mr/cmm_results_filtered.csv')
    fmm_results_path = Path('dataset-hainan-06/mr/fmm_results_filtered.csv')
    gt_points_path = Path('dataset-hainan-06/cmm_input_points.csv')

    # Output paths
    plot_output = Path('dataset-hainan-06/mr/error_analysis.png')
    roc_output = Path('dataset-hainan-06/mr/roc_curve.png')

    print("Loading data...")
    cmm_results = load_cmm_results(cmm_results_path)
    fmm_results = load_fmm_results(fmm_results_path)
    gt_points = load_ground_truth_points(gt_points_path)

    print(f"CMM results: {len(cmm_results)} points")
    print(f"FMM results: {len(fmm_results)} points")
    print(f"Ground truth points: {sum(len(v) for v in gt_points.values())} points")

    # Match FMM to ground truth
    print("\nMatching FMM results to ground truth...")
    fmm_pairs = match_fmm_to_ground_truth(fmm_results, gt_points)
    print(f"Matched FMM pairs: {len(fmm_pairs)}")

    # Calculate FMM errors
    fmm_errors = calculate_errors(fmm_pairs)
    print_statistics("FMM", fmm_errors)

    # Calculate CMM errors (all)
    print("\nCalculating CMM errors...")
    cmm_pairs = [(r['ogeom'], r['pgeom']) for r in cmm_results]
    cmm_errors = calculate_errors(cmm_pairs)
    print_statistics("CMM (all)", cmm_errors)

    # Filter CMM results
    print("\nFiltering CMM results (ep >= 0.5, trustworthiness >= 0.999)...")
    cmm_filtered = [r for r in cmm_results if r['ep'] >= 0.5 and r['trustworthiness'] >= 0.999]
    print(f"Filtered CMM results: {len(cmm_filtered)} / {len(cmm_results)} ({100*len(cmm_filtered)/len(cmm_results):.1f}%)")

    cmm_filtered_pairs = [(r['ogeom'], r['pgeom']) for r in cmm_filtered]
    cmm_filtered_errors = calculate_errors(cmm_filtered_pairs)
    print_statistics("CMM (filtered)", cmm_filtered_errors)

    # Plot error distributions
    print("\nGenerating error distribution plots...")
    plot_error_distributions(fmm_errors, cmm_errors, cmm_filtered_errors, plot_output)

    # Plot ROC curve
    print("\nGenerating ROC curve...")
    plot_roc_curve(cmm_results, roc_output, threshold_m=10.0)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
