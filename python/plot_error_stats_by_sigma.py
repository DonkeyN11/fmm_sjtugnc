#!/usr/bin/env python3
"""
Advanced version: Plot positioning error statistics under different pseudorange noise (Sigma).
Features:
1. Upper part: Boxplot and mean line showing error norm statistical distribution.
2. Lower part: 2D error covariance ellipses under each Sigma column showing spatial distribution characteristics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

# Keep original data reading logic references
from plot_error_vectors import (
    _compute_errors_for_trajs,
    _load_sigma_map,
    _read_csv_auto,
)


def _calc_covariance_ellipse(points_2d, n_std=2.0):
    """
    Calculate covariance ellipse parameters for 2D point set.

    Args:
        points_2d: (N, 2) numpy array.
        n_std: Standard deviation multiplier (2.0 ≈ 95% confidence interval).

    Returns:
        width, height, angle_degrees (for matplotlib.patches.Ellipse)
    """
    if len(points_2d) < 2:
        return 0, 0, 0

    cov = np.cov(points_2d, rowvar=False)

    # Compute eigenvalues and eigenvectors
    # Using eigh for symmetric matrices (covariance is always symmetric), more stable
    vals, vecs = np.linalg.eigh(cov)

    # Sort eigenvalues, ensuring width corresponds to largest eigenvalue
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Calculate rotation angle (angle between max eigenvector and x-axis)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Calculate ellipse axis lengths (std * n_std * 2 for full length)
    # Ensure eigenvalues are non-negative (numerical errors may cause tiny negative values)
    vals = np.maximum(vals, 0)
    width, height = 2 * n_std * np.sqrt(vals)

    return width, height, theta


def plot_advanced_stats(groups_2d: dict, output_path: Path, title: str, limit_percentile: float = 99.0):
    """
    Plot combined chart: Upper part boxplot, lower part covariance ellipses.

    Args:
        groups_2d: Dictionary {sigma_key: (N, 2) numpy array of 2D errors}
    """
    # 1. Data preparation
    # Ensure sigma keys are numeric and sorted
    sorted_sigmas_keys = sorted(groups_2d.keys(), key=lambda x: float(x))

    # Prepare DataFrame for Seaborn boxplot (store norms)
    plot_data_list = []
    # Prepare list for drawing ellipses (store 2D vectors)
    vectors_2d_list = []

    means = []

    for sigma_key in sorted_sigmas_keys:
        errors_2d = groups_2d[sigma_key]

        # Ensure it's (N, 2) array
        if errors_2d.ndim != 2 or errors_2d.shape[1] < 2:
            print(f"Warning: Data for sigma={sigma_key} is not 2D. Skipping ellipses.")
            vectors_2d_list.append(np.zeros((0, 2)))
            # If not 2D, assume it's 1D norm
            errors_norm = errors_2d.flatten() if errors_2d.ndim > 1 else errors_2d
        else:
            vectors_2d_list.append(errors_2d[:, :2])  # Take first two columns (e.g., North, East)
            errors_norm = np.linalg.norm(errors_2d[:, :2], axis=1)

        means.append(np.mean(errors_norm))

        # Add to DataFrame for boxplot
        for err_val in errors_norm:
            plot_data_list.append({'Sigma': str(sigma_key), 'Error Norm (m)': err_val})

    df = pd.DataFrame(plot_data_list)

    # 2. Setup plotting environment
    sns.set_theme(style="white", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(14, 9))

    # ---------------------------------------------------------
    # Upper part: Draw boxplot and trend line (Y > 0)
    # ---------------------------------------------------------
    # Use Seaborn to draw boxplot
    # order parameter ensures boxplot follows our sorted sigma order
    sns.boxplot(x="Sigma", y="Error Norm (m)", data=df, order=[str(s) for s in sorted_sigmas_keys],
                ax=ax, width=0.5, linewidth=1.2, fliersize=2, palette="viridis", saturation=0.75)

    # Draw mean trend line
    # Seaborn's categorical X-axis is actually 0, 1, 2... coordinates
    x_coords = np.arange(len(sorted_sigmas_keys))
    ax.plot(x_coords, means, color='red', marker='o', linestyle='--',
            linewidth=2, markersize=7, label='Mean Error Norm', zorder=10)

    # Calculate Y-axis upper limit (for setting display range)
    y_limit_upper = np.percentile(df['Error Norm (m)'], limit_percentile) * 1.2

    # ---------------------------------------------------------
    # Lower part: Draw covariance ellipses (Y < 0)
    # ---------------------------------------------------------
    # Define ellipse region parameters
    ellipse_region_height = y_limit_upper * 0.6  # Lower region height is 60% of main plot
    y_offset_baseline = -ellipse_region_height / 2.0  # Y-coordinate baseline for ellipse centers

    # Draw separator line (Y=0)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax.text(x_coords[-1] + 0.6, y_offset_baseline, "Spatial Covariance\n(2σ Ellipses)",
            va='center', ha='left', fontsize=10, color='#333333', rotation=-90)

    # Get color palette used by boxplot for consistency
    palette = sns.color_palette("viridis", len(sorted_sigmas_keys))

    # Scaling factor: To prevent large noise ellipses from overlapping, set max size limit
    # We want the largest ellipse width to not exceed the distance between categories (i.e., 1.0)
    max_possible_width = 0.0
    temp_ellipse_params = []
    for vec_2d in vectors_2d_list:
        w, h, angle = _calc_covariance_ellipse(vec_2d, n_std=2.0)
        temp_ellipse_params.append((w, h, angle))
        max_possible_width = max(max_possible_width, w, h)

    # Calculate scaling factor so largest ellipse width is about 0.8 units (leave some gap)
    scaling_factor = 0.8 / max_possible_width if max_possible_width > 0 else 1.0

    # Loop to draw ellipse for each Sigma
    for i, (w, h, angle) in enumerate(temp_ellipse_params):
        if w == 0 or h == 0:
            continue

        # Key: Place ellipse at (x_coords[i], y_offset_baseline)
        # and apply scaling factor
        ellipse = Ellipse(xy=(x_coords[i], y_offset_baseline),
                          width=w * scaling_factor,
                          height=h * scaling_factor,
                          angle=angle,
                          edgecolor=palette[i],
                          facecolor=palette[i],
                          alpha=0.4, linewidth=1.5, zorder=5)
        ax.add_patch(ellipse)

        # Optional: Draw a small center point
        ax.scatter(x_coords[i], y_offset_baseline, color=palette[i], s=10, zorder=6)

    # ---------------------------------------------------------
    # Global beautification and label settings
    # ---------------------------------------------------------
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(r'Pseudorange Noise $\sigma_{pr}$ (m)', fontsize=13, labelpad=10)
    # Remove default Y-axis label, we'll customize it
    ax.set_ylabel('')

    # Set final Y-axis range including negative region
    ax.set_ylim(-ellipse_region_height * 1.1, y_limit_upper)

    # Customize Y-axis tick labels
    # Get current ticks
    yticks = ax.get_yticks()
    # Filter out ticks <= 0, keep only positive ticks for showing norm in upper region
    yticks_positive = [yt for yt in yticks if yt > 0]
    ax.set_yticks(yticks_positive)
    # Add main Y-axis label text
    ax.text(-0.8, y_limit_upper / 2, 'Position Error Norm (m)\n(Statistical Distribution)',
            rotation=90, va='center', ha='center', fontsize=12)

    # Add grid lines (only effective for upper region)
    ax.grid(True, axis='y', which='major', linestyle='--', alpha=0.6)
    # Can also manually add light grid for lower region
    for yt in np.linspace(0, -ellipse_region_height, 5):
        if yt != 0:
            ax.axhline(y=yt, color='gray', linestyle=':', alpha=0.3)

    ax.legend(loc='upper left', frameon=True)

    # Adjust layout to accommodate new elements
    plt.tight_layout()
    # Extra space adjustment to prevent rightmost text from being cut off
    plt.subplots_adjust(right=0.92)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated advanced stats plot: {output_path}")
    plt.close()


def _plot_dataset_advanced(dataset_root: Path, output_dir: Path | None, limit_percentile: float) -> Path:
    points_path = dataset_root / "raw_data" / "observations.csv"
    truth_path = dataset_root / "raw_data" / "ground_truth_points.csv"

    if not points_path.exists() or not truth_path.exists():
        print(f"Warning: Raw data files not found in {dataset_root}, skipping...")
        return Path()

    points_all = _read_csv_auto(points_path)
    truth_all = _read_csv_auto(truth_path)

    try:
        sigma_map = _load_sigma_map(dataset_root / "raw_data" / "metadata.json")
    except Exception:
        print(f"Warning: Could not load sigma map for {dataset_root}")
        return Path()

    # Build sigma groups: mapping from sigma value to list of trajectory IDs
    sigma_groups: dict[float, list[int]] = {}
    for traj_id, sigma in sigma_map.items():
        # Convert traj_id to int if it's stored as string in sigma_map
        traj_id_int = int(traj_id) if isinstance(traj_id, str) else traj_id
        sigma_groups.setdefault(float(sigma), []).append(traj_id_int)

    # Compute 2D error vectors for each sigma group
    groups_2d = {}
    for sigma, traj_ids in sigma_groups.items():
        errors = _compute_errors_for_trajs(points_all, truth_all, traj_ids)
        if errors.size > 0:
            # errors is (N, 2) array with 2D error vectors
            groups_2d[sigma] = errors

    if not groups_2d:
        print(f"No groups found for {dataset_root}")
        return Path()

    title = f"Positioning Error Analysis: Statistical & Spatial ({dataset_root.name})"
    out_dir = output_dir if output_dir else (dataset_root / "statistic")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{dataset_root.name}_error_stats_advanced.png"

    # Call new advanced plotting function
    plot_advanced_stats(groups_2d, output_path, title, limit_percentile)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot advanced sigma error statistics with covariance ellipses")
    parser.add_argument(
        "--dataset_roots",
        nargs="+",
        required=True,
        help="Dataset roots (each contains raw_data/observations.csv and metadata.json)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for sigma-group plots (default: <dataset_root>/statistic)",
    )
    parser.add_argument(
        "--limit_percentile",
        type=float,
        default=98.0,
        help="Y-axis limit percentile for boxplots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    for dataset_root in [Path(p) for p in args.dataset_roots]:
        _plot_dataset_advanced(dataset_root, output_dir, args.limit_percentile)


if __name__ == "__main__":
    main()
