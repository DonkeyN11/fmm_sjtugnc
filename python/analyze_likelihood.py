#!/usr/bin/env python3
"""
Plot frequency histograms for ep, tp, and trustworthi probability columns.

Usage example:
    python analyze_eptpcumu.py output/cmm_result_rearranged.csv output/fmm_positive_rearranged.csv

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
from typing import Dict, Iterable, List, Sequence, Tuple

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
        description="Plot ep/tp/trustworthiness histograms from one or more rearranged CSV files.",
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
        ax.hist(series, bins=bins, color="#4C72B0", alpha=0.75, edgecolor="black")
        ax.set_title(label)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Frequency")

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
