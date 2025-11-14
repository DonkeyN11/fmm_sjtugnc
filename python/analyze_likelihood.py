#!/usr/bin/env python3
"""
Plot frequency histograms for ep, tp, and trustworthi probability columns.

Usage example:
    python analyze_likelihood.py \
        --ground-truth monte_carlo_cps/input/ground_truth.csv \
        output/cmm_result_rearranged.csv output/fmm_positive_rearranged.csv

This script expects semicolon separated CSV files with at least ep, tp, and
cumu (or cumu_prob) numeric columns. For each metric, the script opens a figure
window containing one subplot per input file.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Map canonical metric names to possible column names in the CSV files.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "ep": ("ep",),
    "tp": ("tp",),
    "trustworthiness": ("trust", "trust_prob", "trustworthiness"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ep/tp/trustworthiness histograms from one or more rearranged CSV files "
            "and optionally compare trajectory errors against ground truth."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to semicolon separated CSV files (e.g. cmm_result_rearranged.csv).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save each metric figure as PNG.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip showing the figures (useful for headless runs).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional ground truth CSV (containing LINESTRING geometries) to compute error histograms.",
    )
    return parser.parse_args(argv)


def set_max_csv_field_size() -> None:
    """Raise the csv module field size limit to handle very wide columns."""
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size //= 2
            if max_size <= 0:
                raise RuntimeError("Unable to set CSV field size limit.") from None


def parse_numeric_values(value: object) -> List[float]:
    """Extract numeric values from a scalar that may represent a list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]

    if isinstance(value, str):
        cleaned = value.replace("[", "").replace("]", "")
        pieces = [piece.strip() for piece in cleaned.split(",")]
        numbers: List[float] = []
        for piece in pieces:
            if not piece:
                continue
            try:
                numbers.append(float(piece))
            except ValueError:
                continue
        return numbers

    return []


def parse_linestring(text: Optional[str]) -> List[Tuple[float, float]]:
    """Parse a simple LINESTRING WKT into a list of points."""
    if not text:
        return []
    text = text.strip()
    if not text or not text.upper().startswith("LINESTRING"):
        return []
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        return []
    body = text[open_idx + 1 : close_idx]
    coords: List[Tuple[float, float]] = []
    for token in body.split(","):
        parts = token.strip().split()
        if len(parts) != 2:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return coords


def read_numeric_series(df: pd.DataFrame, metric: str) -> Tuple[pd.Series, str]:
    """Return a numeric pandas Series and the column name used for a given metric."""
    for column in COLUMN_ALIASES[metric]:
        if column in df.columns:
            values: List[float] = []
            for item in df[column]:
                values.extend(parse_numeric_values(item))

            numeric = pd.Series(values)
            numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

            return numeric, column
    raise KeyError(f"None of the aliases {COLUMN_ALIASES[metric]} found for metric '{metric}'.")


def subplot_layout(n_series: int) -> Tuple[int, int]:
    """Compute a roughly square subplot grid for n_series subplots."""
    cols = max(1, math.ceil(math.sqrt(n_series)))
    rows = math.ceil(n_series / cols)
    return rows, cols


def prepare_metric_data(
    input_paths: Iterable[Path],
) -> Dict[str, List[Tuple[str, pd.Series]]]:
    """Load requested columns for each metric while keeping per-file labels."""
    data: Dict[str, List[Tuple[str, pd.Series]]] = {metric: [] for metric in COLUMN_ALIASES}

    for path in input_paths:
        try:
            df = pd.read_csv(path, sep=";", engine="python")
        except Exception as exc:  # pragma: no cover - surface error context
            print(f"[ERROR] Failed to read '{path}': {exc}", file=sys.stderr)
            continue

        label = path.stem
        for metric in COLUMN_ALIASES:
            try:
                series, column_name = read_numeric_series(df, metric)
            except KeyError:
                print(
                    f"[WARN] Column for metric '{metric}' not found in '{path}'. Skipping this subplot.",
                    file=sys.stderr,
                )
                continue

            if series.empty:
                print(
                    f"[WARN] Metric '{metric}' in '{path}' has no numeric data after cleaning.",
                    file=sys.stderr,
                )
                continue

            data[metric].append((label, series))

    return data


def load_ground_truth_geoms(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Load ground truth LINESTRING per trajectory ID."""
    mapping: Dict[str, List[Tuple[float, float]]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id")
            if not traj_id:
                continue
            coords = parse_linestring(row.get("geom"))
            if not coords:
                continue
            mapping[str(traj_id).strip()] = coords
    return mapping


def compute_errors_for_dataset(
    result_path: Path,
    ground_truth: Dict[str, List[Tuple[float, float]]],
) -> pd.Series:
    """Compute Euclidean distances between result points and ground truth."""
    errors: List[float] = []
    if not result_path.exists():
        return pd.Series(dtype=float)
    with result_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id")
            if not traj_id:
                continue
            traj_id = traj_id.strip()
            truth_coords = ground_truth.get(traj_id)
            if not truth_coords:
                continue
            estimate_coords = parse_linestring(row.get("pgeom"))
            if not estimate_coords:
                continue
            count = min(len(truth_coords), len(estimate_coords))
            if count == 0:
                continue
            for idx in range(count):
                est_x, est_y = estimate_coords[idx]
                truth_x, truth_y = truth_coords[idx]
                errors.append(math.hypot(est_x - truth_x, est_y - truth_y))
    return pd.Series(errors, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def prepare_error_datasets(
    input_paths: Iterable[Path],
    ground_truth_path: Path,
) -> List[Tuple[str, pd.Series]]:
    """Compute error series for each dataset with respect to ground truth."""
    ground_truth = load_ground_truth_geoms(ground_truth_path)
    datasets: List[Tuple[str, pd.Series]] = []
    for path in input_paths:
        series = compute_errors_for_dataset(path, ground_truth)
        if series.empty:
            print(
                f"[WARN] No error data computed for '{path}'. "
                "Ensure pgeom coordinates align with ground truth.",
                file=sys.stderr,
            )
            continue
        datasets.append((path.stem, series))
    return datasets


def plot_error_histogram(
    datasets: Sequence[Tuple[str, pd.Series]],
    bins: int,
) -> plt.Figure | None:
    """Plot error distributions for multiple datasets in a single figure."""
    if not datasets:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    color_cycle = plt.rcParams.get("axes.prop_cycler")
    colors = color_cycle.by_key().get("color", []) if color_cycle else []
    for idx, (label, series) in enumerate(datasets):
        color = colors[idx % len(colors)] if colors else None
        ax.hist(
            series,
            bins=bins,
            alpha=0.5,
            edgecolor="black",
            label=label,
            color=color,
            density=True,
        )
        if not series.empty:
            perc = np.percentile(series, 99.99)
            ax.axvline(
                perc,
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"{label} 99.99% = {perc:.2f} m",
            )
    ax.set_xlabel("Position Error (metres)")
    ax.set_ylabel("Density")
    ax.set_title("Trajectory Error vs Ground Truth")
    ax.legend()
    fig.tight_layout()
    try:
        fig.canvas.manager.set_window_title("Trajectory error histogram")
    except Exception:
        pass
    return fig


def plot_metric_histograms(
    metric: str,
    datasets: Sequence[Tuple[str, pd.Series]],
    bins: int,
) -> plt.Figure | None:
    """Create the figure containing one histogram per dataset for a metric."""
    if not datasets:
        print(f"[INFO] No data for metric '{metric}'. Figure will not be created.", file=sys.stderr)
        return None

    rows, cols = subplot_layout(len(datasets))
    figsize = (cols * 5, rows * 4)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for ax, (label, series) in zip(axes_flat, datasets):
        ax.hist(
            series,
            bins=bins,
            color="#4C72B0",
            alpha=0.75,
            edgecolor="black",
            density=True,
        )
        ax.set_title(label)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Density")

    # Remove unused axes if any.
    for ax in axes_flat[len(datasets) :]:
        fig.delaxes(ax)

    fig.suptitle(f"{metric.upper()} Histogram")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    try:
        fig.canvas.manager.set_window_title(f"{metric.upper()} histograms")
    except Exception:
        pass  # Backend might not support window title updates.

    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_max_csv_field_size()

    if args.bins <= 0:
        print("[ERROR] --bins must be a positive integer.", file=sys.stderr)
        return 1

    input_paths = [Path(path) for path in args.inputs]
    data_by_metric = prepare_metric_data(input_paths)

    figures: List[Tuple[str, plt.Figure]] = []
    for metric, datasets in data_by_metric.items():
        fig = plot_metric_histograms(metric, datasets, args.bins)
        if fig is not None:
            figures.append((metric, fig))

    if args.ground_truth is not None:
        try:
            error_datasets = prepare_error_datasets(input_paths, args.ground_truth)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            error_datasets = []
        if error_datasets:
            fig_error = plot_error_histogram(error_datasets, args.bins)
            if fig_error is not None:
                figures.append(("error", fig_error))

    if args.save_dir is not None and figures:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for metric, fig in figures:
            output_path = args.save_dir / f"{metric}_hist.png"
            fig.savefig(output_path, dpi=150)
            print(f"[INFO] Saved {metric.upper()} histogram to {output_path}")

    if not args.no_show and figures:
        plt.show()

    return 0 if figures else 2


if __name__ == "__main__":
    raise SystemExit(main())
