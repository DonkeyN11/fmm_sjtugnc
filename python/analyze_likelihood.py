#!/usr/bin/env python3
"""
Plot frequency histograms for ep, tp, trustworthiness probability columns, and
optional trajectory error CDF/stats against ground truth, including a
trustworthiness-vs-error scatter plot. Ground-truth comparisons operate per
trajectory and require matched point counts.

Usage example:
    python analyze_likelihood.py \
        --ground-truth monte_carlo_cps/input/ground_truth.csv \
        output/cmm_result_rearranged.csv output/fmm_positive_rearranged.csv

This script expects semicolon separated CSV files with at least ep, tp, and
cumu (or cumu_prob) numeric columns. For each metric, the script opens a figure
window containing one subplot per input file. When ground truth trajectories
are supplied, the script additionally plots trajectory error histograms and
per-dataset E/N direction error mean ± standard deviation bars.
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

# Maximum distance (metres) to consider an estimated point as lying on the ground truth path.
ON_PATH_TOLERANCE_M = 1.0


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


def parse_edge_ids(value: object) -> List[str]:
    """Parse a list-like edge id string into a list of string ids."""
    raw_numbers = parse_numeric_values(value)
    if raw_numbers:
        return [str(int(num)) for num in raw_numbers]

    if isinstance(value, str):
        parts = [token.strip() for token in value.replace("[", "").replace("]", "").split(",")]
        cleaned = []
        for token in parts:
            if not token:
                continue
            try:
                cleaned.append(str(int(token)))
            except ValueError:
                cleaned.append(token)
        return cleaned

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


def point_on_polyline(point: Tuple[float, float], polyline: Sequence[Tuple[float, float]], tolerance: float) -> bool:
    """Return True if point is within tolerance of any segment in the polyline."""
    if not polyline:
        return False
    px, py = point
    min_dist_sq = float("inf")
    for (x1, y1), (x2, y2) in zip(polyline, polyline[1:]):
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0.0:
            dist_sq = (px - x1) ** 2 + (py - y1) ** 2
        else:
            t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            if min_dist_sq <= tolerance * tolerance:
                return True
    return min_dist_sq <= tolerance * tolerance


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


def load_ground_truth_data(path: Path) -> Dict[str, Dict[str, object]]:
    """Load ground truth coordinates and edge ids per trajectory id."""
    mapping: Dict[str, Dict[str, object]] = {}
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
            mapping[str(traj_id).strip()] = {
                "coords": coords,
                "edge_ids": parse_edge_ids(row.get("edge_ids")),
            }
    return mapping


def load_ground_truth_geoms(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Load ground truth LINESTRING per trajectory ID."""
    detailed = load_ground_truth_data(path)
    return {traj_id: info["coords"] for traj_id, info in detailed.items()}


def compute_error_components_for_dataset(
    result_path: Path,
    ground_truth: Dict[str, List[Tuple[float, float]]],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Euclidean errors and their E/N direction components using all available
    paired points per trajectory (clipping to the shorter of pgeom/ground truth).

    Returns a tuple of (magnitude_series, e_component_series, n_component_series).
    """
    errors: List[float] = []
    delta_e_list: List[float] = []
    delta_n_list: List[float] = []
    if not result_path.exists():
        empty = pd.Series(dtype=float)
        return empty, empty, empty
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

            for est, truth in zip(estimate_coords[:count], truth_coords[:count]):
                est_x, est_y = est
                truth_x, truth_y = truth
                delta_e = est_x - truth_x
                delta_n = est_y - truth_y
                errors.append(math.hypot(delta_e, delta_n))
                delta_e_list.append(delta_e)
                delta_n_list.append(delta_n)

    def clean_series(values: List[float]) -> pd.Series:
        return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

    return clean_series(errors), clean_series(delta_e_list), clean_series(delta_n_list)


def prepare_error_datasets(
    input_paths: Iterable[Path],
    ground_truth_path: Path,
) -> Tuple[List[Tuple[str, pd.Series]], List[Tuple[str, pd.Series, pd.Series]]]:
    """Compute error and axis-specific series for each dataset with matched point counts."""

    def group_label(path: Path) -> str:
        name = path.stem.lower()
        if "cmm" in name:
            return "CMM"
        if "fmm" in name:
            return "FMM"
        return path.stem

    ground_truth = load_ground_truth_geoms(ground_truth_path)
    grouped: Dict[str, List[Tuple[pd.Series, pd.Series, pd.Series]]] = {}

    for path in input_paths:
        magnitude, delta_e, delta_n = compute_error_components_for_dataset(path, ground_truth)
        if magnitude.empty:
            print(
                f"[WARN] No error data computed for '{path}'. "
                "Ensure pgeom coordinates align with ground truth and point counts match.",
                file=sys.stderr,
            )
            continue
        label = group_label(path)
        grouped.setdefault(label, []).append((magnitude, delta_e, delta_n))

    datasets: List[Tuple[str, pd.Series]] = []
    axis_datasets: List[Tuple[str, pd.Series, pd.Series]] = []
    for label, series_list in grouped.items():
        magnitudes = pd.concat([m for m, _, _ in series_list], ignore_index=True)
        e_series = pd.concat([e for _, e, _ in series_list], ignore_index=True)
        n_series = pd.concat([n for _, _, n in series_list], ignore_index=True)
        datasets.append((label, magnitudes))
        axis_datasets.append((label, e_series, n_series))

    return datasets, axis_datasets


def _extract_trust_values(row: Dict[str, object], target_len: int) -> List[float]:
    """Return trustworthiness values per point, broadcasting when needed."""
    if target_len <= 0:
        return []
    for column in COLUMN_ALIASES["trustworthiness"]:
        value = row.get(column)
        if value is None:
            continue
        numbers = parse_numeric_values(value)
        if not numbers:
            continue
        if len(numbers) == target_len:
            return [float(num) for num in numbers]
        if len(numbers) == 1:
            return [float(numbers[0])] * target_len
        mean_val = float(np.mean(numbers))
        return [mean_val] * target_len
    return []


def compute_trust_error_pairs(
    result_path: Path, ground_truth: Dict[str, List[Tuple[float, float]]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of (trustworthiness, per-point trajectory error) for a dataset."""
    trust_values: List[float] = []
    errors: List[float] = []
    if not result_path.exists():
        return np.array([]), np.array([])

    with result_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id")
            if not traj_id:
                continue
            truth_coords = ground_truth.get(str(traj_id).strip())
            if not truth_coords:
                continue
            estimate_coords = parse_linestring(row.get("pgeom"))
            if not estimate_coords:
                continue

            count = min(len(truth_coords), len(estimate_coords))
            if count == 0:
                continue

            deltas = [
                math.hypot(est_x - truth_x, est_y - truth_y)
                for (est_x, est_y), (truth_x, truth_y) in zip(estimate_coords[:count], truth_coords[:count])
            ]
            trust_values_row = _extract_trust_values(row, len(deltas))

            if deltas and trust_values_row:
                trust_values.extend(trust_values_row)
                errors.extend(deltas)

    return np.array(trust_values, dtype=float), np.array(errors, dtype=float)


def prepare_trust_error_datasets(
    input_paths: Iterable[Path], ground_truth_path: Path
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Compute trustworthiness vs error pairs, grouped for CMM/FMM separation."""

    def group_label(path: Path) -> str:
        name = path.stem.lower()
        if "cmm" in name:
            return "CMM"
        if "fmm" in name:
            return "FMM"
        return path.stem

    ground_truth = load_ground_truth_geoms(ground_truth_path)
    grouped: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for path in input_paths:
        trust, errs = compute_trust_error_pairs(path, ground_truth)
        if trust.size == 0 or errs.size == 0:
            print(
                f"[WARN] No trust/error pairs computed for '{path}'. "
                "Ensure trustworthiness columns and pgeom align with ground truth.",
                file=sys.stderr,
            )
            continue

        label = group_label(path)
        grouped.setdefault(label, []).append((trust, errs))

    datasets: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for label, arrs in grouped.items():
        trust_all = np.concatenate([trust for trust, _ in arrs]) if arrs else np.array([])
        err_all = np.concatenate([errs for _, errs in arrs]) if arrs else np.array([])
        if trust_all.size == 0 or err_all.size == 0:
            continue
        datasets.append((label, trust_all, err_all))

    return datasets


def compute_correctness_for_dataset(
    result_path: Path, ground_truth: Dict[str, Dict[str, object]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (trust_values, correct_flags) per point for one dataset."""
    trust_values: List[float] = []
    correct_flags: List[int] = []
    if not result_path.exists():
        return np.array([]), np.array([])

    with result_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id_raw = row.get("id")
            if not traj_id_raw:
                continue
            traj_id = str(traj_id_raw).strip()
            truth_entry = ground_truth.get(traj_id)
            if not truth_entry:
                continue
            truth_coords = truth_entry.get("coords") or []
            estimate_coords = parse_linestring(row.get("pgeom"))
            if not truth_coords or not estimate_coords:
                continue

            count = min(len(truth_coords), len(estimate_coords))
            if count == 0:
                continue

            trust_values_row = _extract_trust_values(row, count)
            if not trust_values_row:
                continue

            correct_row = [
                int(point_on_polyline(est, truth_coords, ON_PATH_TOLERANCE_M))
                for est in estimate_coords[:count]
            ]
            trust_values.extend(trust_values_row)
            correct_flags.extend(correct_row)

    return np.array(trust_values, dtype=float), np.array(correct_flags, dtype=float)


def prepare_correctness_datasets(
    input_paths: Iterable[Path], ground_truth_path: Path
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Compute trustworthiness and correctness flags per point, grouped CMM/FMM."""

    def group_label(path: Path) -> str:
        name = path.stem.lower()
        if "cmm" in name:
            return "CMM"
        if "fmm" in name:
            return "FMM"
        return path.stem

    ground_truth = load_ground_truth_data(ground_truth_path)
    grouped: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for path in input_paths:
        trust, correct = compute_correctness_for_dataset(path, ground_truth)
        if trust.size == 0 or correct.size == 0:
            print(
                f"[WARN] No correctness data for '{path}'. "
                "Ensure ids exist and pgeom/ground truth geom are present.",
                file=sys.stderr,
            )
            continue

        label = group_label(path)
        grouped.setdefault(label, []).append((trust, correct))

    datasets: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for label, arrs in grouped.items():
        trust_all = np.concatenate([t for t, _ in arrs]) if arrs else np.array([])
        correct_all = np.concatenate([c for _, c in arrs]) if arrs else np.array([])
        if trust_all.size == 0 or correct_all.size == 0:
            continue
        datasets.append((label, trust_all, correct_all))

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


def plot_error_cdf(datasets: Sequence[Tuple[str, pd.Series]]) -> plt.Figure | None:
    """Plot cumulative distribution functions for trajectory errors."""
    if not datasets:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    color_cycle = plt.rcParams.get("axes.prop_cycler")
    colors = color_cycle.by_key().get("color", []) if color_cycle else []
    plotted = False

    for idx, (label, series) in enumerate(datasets):
        if series.empty:
            continue
        sorted_vals = np.sort(series.to_numpy())
        probs = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        color = colors[idx % len(colors)] if colors else None
        ax.step(sorted_vals, probs, where="post", label=label, color=color)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Position Error (metres)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_title("Trajectory error cumulative distribution")
    ax.legend()
    fig.tight_layout()

    try:
        fig.canvas.manager.set_window_title("Trajectory error CDF")
    except Exception:
        pass
    return fig


def plot_trust_vs_error(
    datasets: Sequence[Tuple[str, np.ndarray, np.ndarray]]
) -> plt.Figure | None:
    """Plot trustworthiness (x) vs mean trajectory error (y) scatter."""
    if not datasets:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    color_cycle = plt.rcParams.get("axes.prop_cycler")
    colors = color_cycle.by_key().get("color", []) if color_cycle else []

    for idx, (label, trust, errs) in enumerate(datasets):
        if trust.size == 0 or errs.size == 0:
            continue
        color = colors[idx % len(colors)] if colors else None
        ax.scatter(
            trust,
            errs,
            label=f"{label} (n={len(trust)})",
            alpha=0.6,
            edgecolors="black",
            linewidths=0.4,
            color=color,
        )

    ax.set_xlabel("Trustworthiness")
    ax.set_ylabel("Mean trajectory error (metres)")
    ax.set_title("Trustworthiness vs trajectory error")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    try:
        fig.canvas.manager.set_window_title("Trustworthiness vs error")
    except Exception:
        pass
    return fig


def compute_accuracy_curve(
    trust: np.ndarray, correct: np.ndarray, thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute accuracy and coverage for a set of thresholds."""
    total = len(trust)
    accuracy: List[float] = []
    coverage: List[float] = []
    if total == 0:
        return np.array([]), np.array([])
    for tau in thresholds:
        mask = trust >= tau
        count = int(np.count_nonzero(mask))
        coverage.append(count / total if total else 0.0)
        if count == 0:
            accuracy.append(np.nan)
        else:
            accuracy.append(float(np.mean(correct[mask])))
    return np.array(accuracy, dtype=float), np.array(coverage, dtype=float)


def plot_accuracy_thresholds(
    datasets: Sequence[Tuple[str, np.ndarray, np.ndarray]], thresholds: np.ndarray
) -> plt.Figure | None:
    """Plot accuracy vs threshold with coverage on a secondary axis."""
    if not datasets:
        return None

    fig, ax_acc = plt.subplots(figsize=(8, 5))
    ax_cov = ax_acc.twinx()

    color_cycle = plt.rcParams.get("axes.prop_cycler")
    colors = color_cycle.by_key().get("color", []) if color_cycle else []

    for idx, (label, trust, correct) in enumerate(datasets):
        if trust.size == 0 or correct.size == 0:
            continue
        acc, cov = compute_accuracy_curve(trust, correct, thresholds)
        color = colors[idx % len(colors)] if colors else None
        ax_acc.plot(thresholds, acc, label=f"{label} accuracy (n={len(trust)})", color=color, linewidth=2.0)
        ax_cov.plot(thresholds, cov, label=f"{label} coverage", color=color, linestyle="--", alpha=0.7)

    ax_acc.set_xlabel("Trustworthiness threshold")
    ax_acc.set_ylabel("Accuracy")
    ax_cov.set_ylabel("Coverage (fraction of points)")
    ax_acc.set_ylim(0.0, 1.05)
    ax_cov.set_ylim(0.0, 1.05)
    ax_acc.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    fig.tight_layout()

    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    handles_cov, labels_cov = ax_cov.get_legend_handles_labels()
    ax_acc.legend(handles_acc + handles_cov, labels_acc + labels_cov, loc="lower left", fontsize="small")

    try:
        fig.canvas.manager.set_window_title("Accuracy vs threshold")
    except Exception:
        pass
    return fig


def plot_accuracy_vs_coverage(
    datasets: Sequence[Tuple[str, np.ndarray, np.ndarray]], thresholds: np.ndarray
) -> plt.Figure | None:
    """Plot accuracy as a function of achieved coverage."""
    if not datasets:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    color_cycle = plt.rcParams.get("axes.prop_cycler")
    colors = color_cycle.by_key().get("color", []) if color_cycle else []

    for idx, (label, trust, correct) in enumerate(datasets):
        if trust.size == 0 or correct.size == 0:
            continue
        acc, cov = compute_accuracy_curve(trust, correct, thresholds)
        valid = ~np.isnan(acc)
        if not np.any(valid):
            continue
        color = colors[idx % len(colors)] if colors else None
        ax.plot(cov[valid], acc[valid], label=f"{label} (n={len(trust)})", color=color, linewidth=2.0)

    ax.set_xlabel("Coverage (fraction of points retained)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    try:
        fig.canvas.manager.set_window_title("Accuracy vs coverage")
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


def plot_axis_error_stats(
    axis_datasets: Sequence[Tuple[str, pd.Series, pd.Series]]
) -> plt.Figure | None:
    """Plot mean and standard deviation for E and N direction errors."""
    if not axis_datasets:
        return None

    labels = [label for label, _, _ in axis_datasets]
    e_means = [series.mean() if not series.empty else 0.0 for _, series, _ in axis_datasets]
    e_stds = [series.std(ddof=1) if len(series) > 1 else 0.0 for _, series, _ in axis_datasets]
    n_means = [series.mean() if not series.empty else 0.0 for _, _, series in axis_datasets]
    n_stds = [series.std(ddof=1) if len(series) > 1 else 0.0 for _, _, series in axis_datasets]

    fig_width = max(8, len(labels) * 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4))

    def plot_component(ax, means, stds, title, color):
        positions = np.arange(len(labels))
        ax.bar(positions, means, yerr=stds, capsize=6, color=color, alpha=0.8, edgecolor="black")
        ax.axhline(0.0, color="gray", linewidth=1, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Error (meters)")
        ax.set_title(title)

    plot_component(axes[0], e_means, e_stds, "E-direction error mean ± std", "#1f77b4")
    plot_component(axes[1], n_means, n_stds, "N-direction error mean ± std", "#ff7f0e")
    fig.tight_layout()

    try:
        fig.canvas.manager.set_window_title("E/N direction error statistics")
    except Exception:
        pass
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
            error_datasets, axis_error_datasets = prepare_error_datasets(input_paths, args.ground_truth)
            trust_error_datasets = prepare_trust_error_datasets(input_paths, args.ground_truth)
            correctness_datasets = prepare_correctness_datasets(input_paths, args.ground_truth)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            error_datasets = []
            axis_error_datasets = []
            trust_error_datasets = []
            correctness_datasets = []
        if error_datasets:
            fig_error = plot_error_histogram(error_datasets, args.bins)
            if fig_error is not None:
                figures.append(("error", fig_error))
            fig_cdf = plot_error_cdf(error_datasets)
            if fig_cdf is not None:
                figures.append(("error_cdf", fig_cdf))
        if axis_error_datasets:
            fig_axis = plot_axis_error_stats(axis_error_datasets)
            if fig_axis is not None:
                figures.append(("axis_error", fig_axis))
        if trust_error_datasets:
            fig_scatter = plot_trust_vs_error(trust_error_datasets)
            if fig_scatter is not None:
                figures.append(("trust_error", fig_scatter))
        if correctness_datasets:
            thresholds = np.linspace(0.0, 1.0, 101)
            fig_acc = plot_accuracy_thresholds(correctness_datasets, thresholds)
            if fig_acc is not None:
                figures.append(("accuracy_threshold", fig_acc))
            fig_pc = plot_accuracy_vs_coverage(correctness_datasets, thresholds)
            if fig_pc is not None:
                figures.append(("accuracy_coverage", fig_pc))

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
