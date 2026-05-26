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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

DPI = 300
COLOR_CMM = "#2166ac"
COLOR_FMM = "#b2182b"
COLOR_FILTERED = "#4393c3"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})


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
    """Plot error CDF comparison for FMM, CMM, and filtered CMM."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # CDF curves
    for data, color, label in [
        (fmm_errors, COLOR_FMM, "FMM (isotropic)"),
        (cmm_errors, COLOR_CMM, "CMM (all)"),
        (cmm_filtered_errors, COLOR_FILTERED, "CMM (filtered)"),
    ]:
        sorted_vals = np.sort(data)
        probs = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.step(sorted_vals, probs, where="post", color=color, linewidth=1.2, label=label)

    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Error Cumulative Distribution Function")
    ax.set_xscale("log")
    ax.set_xlim(0.1, None)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    print(f"Error CDF saved to: {output_path}")
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    ax.plot(fpr_list, tpr_list, color=COLOR_CMM, linewidth=1.5, label="CMM (trustworthiness)")

    # Calculate AUC
    auc = np.trapezoid(tpr_list, fpr_list)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label=f"Random (AUC=0.500)")

    ax.fill_between(fpr_list, tpr_list, alpha=0.12, color=COLOR_CMM)

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve (error threshold = {threshold_m:.0f} m)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")

    ax.text(0.57, 0.18, f"AUC = {auc:.4f}", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.7"))

    fig.tight_layout()
    plt.savefig(output_path, dpi=DPI)
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

    # Output paths — save to paper's figs/ directory
    figs_dir = Path("docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    error_cdf_output = figs_dir / 'error_cdf_hist.png'
    roc_output = figs_dir / 'ROC.png'

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

    # Plot error CDF
    print("\nGenerating error CDF...")
    plot_error_distributions(fmm_errors, cmm_errors, cmm_filtered_errors, error_cdf_output)

    # Plot ROC curve
    print("\nGenerating ROC curve...")
    plot_roc_curve(cmm_results, roc_output, threshold_m=10.0)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
