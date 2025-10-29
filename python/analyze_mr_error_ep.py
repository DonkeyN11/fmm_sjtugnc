#!/usr/bin/env python3
"""
Generate distribution histograms for EP, computed speed, and error metrics
from the mr_error_ep dataset. The script splits the dataset into N partitions,
builds histograms for each partition, and highlights the 99.99% quantile on
every subplot while trimming extreme outliers for clearer x-axis ranges.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

# Use a non-interactive backend to support headless execution.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CANONICAL_COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "ep": ("ep",),
    "spdist": ("spdist", "sp_dist", "spatial_distance", "distance"),
    "duration": ("duration", "time", "time_s", "travel_time"),
    "error": ("error", "err"),
}
METRIC_QUANTILE_PROBS: Dict[str, float] = {"error": 0.9999, "speed": 0.9999, "ep": 0.0001}


def split_dataframe(df: pd.DataFrame, partitions: int) -> Tuple[pd.DataFrame, ...]:
    if partitions <= 0:
        raise ValueError("Number of partitions must be a positive integer.")

    indices = np.array_split(np.arange(len(df)), partitions)
    splits = []
    for idx in indices:
        if idx.size:
            splits.append(df.iloc[idx])
        else:
            splits.append(df.iloc[0:0])
    return tuple(splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split mr_error_ep dataset into partitions and plot histograms for "
            "EP, computed speed (spdist/duration), and error metrics."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("output/mr_error_ep_.txt"),
        help="Path to the input dataset (default: output/mr_error_ep_.txt).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where figures will be saved (default: output).",
    )
    parser.add_argument(
        "-n",
        "--partitions",
        type=int,
        default=10,
        help="Number of partitions to split the dataset into (default: 10).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help=(
            "Minimum number of data points required to compute statistics for a "
            "partition (default: 5). If a partition has fewer points, it will be "
            "noted on the subplot."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("output/mr_error_ep_summary.txt"),
        help=(
            "Path to write the quantile summary report (default: "
            "output/mr_error_ep_summary.txt). Use '-' to skip writing a file."
        ),
    )
    return parser.parse_args()


def resolve_column(
    columns: Iterable[str], candidates: Tuple[str, ...], metric: str
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(
        f"Could not find a column for '{metric}'. "
        f"Available columns: {', '.join(columns)}"
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(int(1e9))

    size_attempts = [int(1e9), int(5e9), sys.maxsize]
    for limit in size_attempts:
        try:
            csv.field_size_limit(limit)
        except OverflowError:
            continue

    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except EmptyDataError as exc:
        raise ValueError(f"The input file '{path}' is empty.") from exc
    except ParserError as exc:
        if "field larger than field limit" not in str(exc).lower():
            raise
        escalated_limits = [int(1e10), int(5e10), sys.maxsize]
        for limit in escalated_limits:
            try:
                csv.field_size_limit(limit)
            except OverflowError:
                continue

            try:
                df = pd.read_csv(path, sep=None, engine="python")
                break
            except ParserError as inner_exc:
                if "field larger than field limit" not in str(inner_exc).lower():
                    raise
        else:
            raise ParserError(
                "Unable to read CSV because a field exceeds the configured CSV parser limit."
            ) from exc

    df.columns = [col.strip().lower() for col in df.columns]
    if df.empty:
        raise ValueError(f"The input file '{path}' does not contain any rows.")
    return df


def extract_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    resolved = {}
    for metric, candidates in CANONICAL_COLUMN_CANDIDATES.items():
        resolved_name = resolve_column(df.columns, candidates, metric)
        resolved[metric] = resolved_name

    trimmed_df = df[list(resolved.values())].copy()
    trimmed_df.rename(columns={source: target for target, source in resolved.items()}, inplace=True)

    # Ensure numeric dtype for calculations.
    for column in ("ep", "spdist", "duration", "error"):
        trimmed_df[column] = pd.to_numeric(trimmed_df[column], errors="coerce")
    return trimmed_df


def compute_speed(df: pd.DataFrame) -> pd.Series:
    duration = df["duration"].replace({0: np.nan})
    return df["spdist"] / duration


def clip_outliers(series: pd.Series) -> pd.Series:
    clean_series = series.dropna()
    if clean_series.empty:
        return clean_series

    lower = clean_series.quantile(0.001, interpolation="nearest")
    upper = clean_series.quantile(0.999, interpolation="nearest")

    trimmed = clean_series[(clean_series >= lower) & (clean_series <= upper)]
    # Fall back to the original series if clipping removed everything.
    return trimmed if not trimmed.empty else clean_series


def partition_quantiles(
    df: pd.DataFrame,
    partitions: int,
    metric: str,
    quantile: float,
) -> List[float]:
    quantiles: List[float] = []
    for partition in split_dataframe(df, partitions):
        series = partition[metric].dropna()
        if series.empty:
            quantiles.append(np.nan)
            continue
        quantiles.append(series.quantile(quantile, interpolation="nearest"))
    return quantiles


def build_quantile_report(df: pd.DataFrame, partitions: int) -> str:
    lines = [f"Quantile summary across {partitions} partitions:"]
    for metric in ("error", "speed", "ep"):
        quantile_prob = METRIC_QUANTILE_PROBS[metric]
        quantiles = partition_quantiles(df, partitions, metric, quantile_prob)
        formatted_values = []
        for idx, value in enumerate(quantiles, start=1):
            if not np.isfinite(value):
                formatted_values.append(f"P{idx}=NaN")
            else:
                formatted_values.append(f"P{idx}={value:.6f}")
        direction = "left" if METRIC_QUANTILE_PROBS[metric] <= 0.5 else "right"
        lines.append(f"{metric} ({direction} quantile): {', '.join(formatted_values)}")
    return "\n".join(lines)


def pick_bins(series: pd.Series) -> int:
    size = series.size
    if size <= 1:
        return 1
    # Use the square-root choice as a balanced default for histogram bins.
    return max(10, min(60, int(np.sqrt(size))))


def format_partition_title(metric: str, index: int, total: int, count: int) -> str:
    return f"{metric.upper()} Part {index + 1}/{total} (n={count})"


def plot_histograms(
    partitions: Iterable[pd.DataFrame],
    metric: str,
    output_path: Path,
    min_points: int,
) -> None:
    partitions = list(partitions)
    total_parts = len(partitions)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.ravel()

    for idx, (partition, ax) in enumerate(zip(partitions, axes)):
        series = partition[metric].dropna()
        ax.set_title(format_partition_title(metric, idx, total_parts, series.size))

        if series.size < min_points:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")
            continue

        trimmed = clip_outliers(series)
        bins = pick_bins(trimmed)
        ax.hist(trimmed, bins=bins, color="#0072B2", edgecolor="black", alpha=0.75)

        quantile_prob = METRIC_QUANTILE_PROBS[metric]
        quantile_value = trimmed.quantile(quantile_prob, interpolation="nearest")
        if np.isfinite(quantile_value):
            ax.axvline(quantile_value, color="#D55E00", linestyle="--", linewidth=1.5)
            ax.text(
                quantile_value,
                ax.get_ylim()[1] * 0.95,
                "0.01% quantile" if metric == "ep" else "99.99% quantile",
                rotation=90,
                va="top",
                ha="right",
                color="#D55E00",
                fontsize=9,
            )

        x_min, x_max = trimmed.min(), trimmed.max()
        if np.isfinite(x_min) and np.isfinite(x_max):
            if x_min == x_max:
                delta = max(abs(x_min) * 0.1, 1.0)
                x_min -= delta
                x_max += delta
            else:
                span = x_max - x_min
                padding = max(span * 0.05, 1e-9)
                x_min -= padding
                x_max += padding
            ax.set_xlim(x_min, x_max)

        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")

    # Hide any unused subplots if partitions < axes size.
    for ax in axes[len(partitions) :]:
        ax.set_visible(False)

    fig.suptitle(f"{metric.upper()} Distribution by Partition", fontsize=16)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    df = extract_required_columns(df)
    df["speed"] = compute_speed(df)

    summary_text = build_quantile_report(df, args.partitions)
    print(summary_text)

    if str(args.summary_path) != "-":
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(summary_text + "\n")

    partitions = split_dataframe(df, args.partitions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_histograms(partitions, "ep", args.output_dir / "mr_error_ep_ep_hist.png", args.min_points)
    plot_histograms(
        partitions,
        "speed",
        args.output_dir / "mr_error_ep_speed_hist.png",
        args.min_points,
    )
    plot_histograms(
        partitions,
        "error",
        args.output_dir / "mr_error_ep_error_hist.png",
        args.min_points,
    )


if __name__ == "__main__":
    main()
