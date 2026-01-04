#!/usr/bin/env python3
"""
Plot error statistics grouped by sigma_pr for full datasets.

Usage example:
  python python/plot_error_stats_by_sigma.py \
    --dataset_roots simulation/huge_sigma simulation/small_sigma \
    --output_dir simulation/statistic \
    --global_limits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_error_vectors import (
    _compute_errors_for_trajs,
    _limits_from_errors,
    _load_sigma_map,
    _mean_cov_from_points,
    _pick_id_seq_columns,
    _plot_sigma_group_grid,
    _read_csv_auto,
)


def _plot_dataset(dataset_root: Path, output_dir: Path | None, global_limits: bool, limit_percentile: float) -> Path:
    points_path = dataset_root / "raw_data" / "observations.csv"
    truth_path = dataset_root / "raw_data" / "ground_truth_points.csv"
    metadata_path = dataset_root / "raw_data" / "metadata.json"
    if not points_path.exists() or not truth_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing raw_data files under {dataset_root}: observations.csv, ground_truth_points.csv, metadata.json"
        )

    points_all = _read_csv_auto(points_path)
    truth_all = _read_csv_auto(truth_path)
    sigma_map = _load_sigma_map(metadata_path)
    if not sigma_map:
        raise ValueError(f"No sigma mapping found in {metadata_path}")

    id_col_p, _ = _pick_id_seq_columns(points_all)
    sigma_groups: dict[float, list[int]] = {}
    for traj_id, sigma in sigma_map.items():
        sigma_groups.setdefault(float(sigma), []).append(traj_id)

    groups: list[tuple[float, np.ndarray, np.ndarray]] = []
    for sigma in sorted(sigma_groups.keys()):
        traj_ids = sigma_groups[sigma]
        errors = _compute_errors_for_trajs(points_all, truth_all, traj_ids)
        points_subset = points_all[points_all[id_col_p].isin(traj_ids)]
        mean_cov = _mean_cov_from_points(points_subset, errors)
        groups.append((sigma, errors, mean_cov))

    lims = None
    if global_limits:
        errors_all = _compute_errors_for_trajs(points_all, truth_all, list(sigma_map.keys()))
        if errors_all.size:
            errors_all_centered = errors_all - errors_all.mean(axis=0)
            lims = _limits_from_errors(errors_all_centered, limit_percentile)

    title = f"{dataset_root.name} error statistics by sigma_pr"
    out_dir = output_dir if output_dir else (dataset_root / "statistic")
    output_path = out_dir / f"{dataset_root.name}_error_stats_by_sigma.png"
    _plot_sigma_group_grid(groups, output_path, title, limit_percentile, lims)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sigma-grouped error statistics")
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
        "--global_limits",
        action="store_true",
        help="Use percentile limits computed across all trajectories for consistent scales",
    )
    parser.add_argument(
        "--limit_percentile",
        type=float,
        default=99.0,
        help="Percentile for axis limits (used with and without --global_limits)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    for dataset_root in [Path(p) for p in args.dataset_roots]:
        output_path = _plot_dataset(dataset_root, output_dir, args.global_limits, args.limit_percentile)
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
